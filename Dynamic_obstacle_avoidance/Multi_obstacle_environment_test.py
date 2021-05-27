#!/usr/bin/python
# -*- coding: utf-8 -*-
# Time: 2021-4-2
# Author: ZYunfei
# Name: The final test environment which includes many dynamic obstacles
# File func: Final test file
import numpy as np
import torch
from Dynamic_obstacle_avoidance.config import Config
from Dynamic_obstacle_avoidance.dynamic_obstacle_environment import obs_list
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Environment:
    def __init__(self,index):
        """基本参数："""
        self.V0 = 1
        self.threshold = 0.2
        self.step_size = 0.1
        self.lam = 8  # 越大考虑障碍物速度越明显

        self.start = np.array([0,2,5],dtype=float)
        self.goal = np.array([10, 10, 5.5], dtype=float)

        self.time_log = 0
        self.time_step = 0.1

        self.xmax = 10 / 180 * np.pi  # 偏航角速度最大值  每个步长允许变化的角度
        self.gammax = 10 / 180 * np.pi  # 爬升角速度最大值  每个步长允许变化的角度
        self.maximumClimbingAngle = 100 / 180 * np.pi  # 最大爬升角
        self.maximumSubductionAngle = - 75 / 180 * np.pi  # 最大俯冲角

        self.obs_num = 4           # 障碍物数量
        self.obs_r = [1, 1, 1, 1]  # 障碍物半径表

        self.path = {}                 # 记录所有障碍物的trace字典
        for i in range(self.obs_num):
            self.path[i] = np.array([[]]).reshape(-1,3)


        if index == 1:                            # 从obs_list中抽取obs_num个障碍trace  换测试环境在这句换
            self.obs = obs_list[0:4]
        elif index == 2:
            self.obs = obs_list[4:8]
        elif index == 3:
            self.obs = [obs_list[i] for i in [0,1,4,5]]
        elif index == 4:
            self.obs = [obs_list[i] for i in [0,1,6,7]]
        elif index == 5:
            self.obs = [obs_list[i] for i in [2,3,4,5]]
        elif index == 6:
            self.obs = [obs_list[i] for i in [2,3,6,7]]
        else: print("初始化env的index需在1-6之间!")


    def reset(self):
        self.time_log = 0                          # 时间归0
        for i in range(self.obs_num):              # 所有障碍物的路径记录数组置空
            self.path[i] = np.array([[]]).reshape(-1, 3)

    def update_obs_pos(self,uav_pos):
        """更新所有障碍位置，并判断谁与uav最近，返回它的dic"""
        distance = np.inf
        temp_dic = None
        for i in range(self.obs_num):
            self.time_log, dic = self.obs[i](self.time_log, self.time_step)
            self.path[i] = np.vstack((self.path[i], dic['obsCenter']))
            self.time_log -= self.time_step  # obs函数内自带了time_log的自增，这里需要减掉

            if self.distanceCost(dic['obsCenter'],uav_pos)-self.obs_r[i] < distance:
                distance = self.distanceCost(dic['obsCenter'],uav_pos)
                temp_dic = dic
                temp_dic['obs_r'] = self.obs_r[i]
                temp_dic['d'] = distance - self.obs_r[i]
        self.time_log += self.time_step

        return temp_dic

    def calDynamicState(self, uav_pos, obs_center, obs_R, v_obs):
        """强化学习模型获得的state。"""
        s1 = (obs_center - uav_pos)*(self.distanceCost(obs_center,uav_pos)-obs_R)/self.distanceCost(obs_center,uav_pos)
        s2 = self.goal - uav_pos
        s3 = v_obs
        return np.append(s1,[s2,s3])

    def calRepulsiveMatrix(self, uavPos, obsCenter, obsR, row0):
        n = self.partialDerivativeSphere(obsCenter, uavPos, obsR)
        tempD = self.distanceCost(uavPos, obsCenter) - obsR
        row = row0 * np.exp(1-1/(self.distanceCost(uavPos,self.goal)*tempD))
        T = self.calculateT(obsCenter, uavPos, obsR)
        repulsiveMatrix = np.dot(-n,n.T) / T**(1/row) / np.dot(n.T,n)[0][0]
        return repulsiveMatrix

    def calTangentialMatrix(self, uavPos, obsCenter, obsR, theta, sigma0):
        n = self.partialDerivativeSphere(obsCenter, uavPos, obsR)
        T = self.calculateT(obsCenter, uavPos, obsR)
        partialX = (uavPos[0] - obsCenter[0]) * 2 / obsR ** 2
        partialY = (uavPos[1] - obsCenter[1]) * 2 / obsR ** 2
        partialZ = (uavPos[2] - obsCenter[2]) * 2 / obsR ** 2
        tk1 = np.array([partialY, -partialX, 0],dtype=float).reshape(-1,1)
        tk2 = np.array([partialX*partialZ, partialY*partialZ, -partialX**2-partialY**2],dtype=float).reshape(-1,1)
        originalPoint = np.array([np.cos(theta), np.sin(theta), 0]).reshape(1,-1)
        tk = self.trans(originalPoint, tk1.squeeze(), tk2.squeeze(), n.squeeze())
        tempD = self.distanceCost(uavPos, obsCenter) - obsR
        sigma = sigma0 * np.exp(1-1/(self.distanceCost(uavPos,self.goal)*tempD))
        tangentialMatrix = tk.dot(n.T) / T**(1/sigma) / self.calVecLen(tk.squeeze()) / self.calVecLen(n.squeeze())
        return tangentialMatrix
    def getqNext(self, uavPos, obsCenter, vObs, obsR, row0, sigma0, theta, qBefore):
        u = self.initField(uavPos, self.V0, self.goal)
        repulsiveMatrix = self.calRepulsiveMatrix(uavPos, obsCenter, obsR, row0)
        tangentialMatrix = self.calTangentialMatrix(uavPos, obsCenter, obsR, theta, sigma0)
        T = self.calculateT(obsCenter, uavPos, obsR)
        vp = np.exp(-T / self.lam) * vObs
        M = np.eye(3) + repulsiveMatrix + tangentialMatrix
        ubar = (M.dot(u - vp.reshape(-1, 1)).T + vp.reshape(1, -1)).squeeze()
        # 限制ubar的模长，避免进入障碍内部后轨迹突变
        if self.calVecLen(ubar) > 5:
            ubar = ubar/self.calVecLen(ubar)*5
        if qBefore[0] is None:
            uavNextPos = uavPos + ubar * self.step_size
        else:
            uavNextPos = uavPos + ubar * self.step_size
            _, _, _, _, qNext = self.kinematicConstrant(uavPos, qBefore, uavNextPos)
        return uavNextPos

    def kinematicConstrant(self, q, qBefore, qNext):
        """
        运动学约束函数 返回(上一时刻航迹角，上一时刻爬升角，约束后航迹角，约束后爬升角，约束后下一位置qNext)
        """
        # 计算qBefore到q航迹角x1,gam1
        qBefore2q = q - qBefore
        if qBefore2q[0] != 0 or qBefore2q[1] != 0:
            x1 = np.arcsin(np.abs(qBefore2q[1] / np.sqrt(qBefore2q[0] ** 2 + qBefore2q[1] ** 2)))  # 这里计算的角限定在了第一象限的角 0-pi/2
            gam1 = np.arcsin(qBefore2q[2] / np.sqrt(np.sum(qBefore2q ** 2)))
        else:
            return None, None, None, None, qNext
        # 计算q到qNext航迹角x2,gam2
        q2qNext = qNext - q
        x2 = np.arcsin(np.abs(q2qNext[1] / np.sqrt(q2qNext[0] ** 2 + q2qNext[1] ** 2)))  # 这里同理计算第一象限的角度
        gam2 = np.arcsin(q2qNext[2] / np.sqrt(np.sum(q2qNext ** 2)))

        # 根据不同象限计算矢量相对于x正半轴的角度 0-2 * pi
        if qBefore2q[0] > 0 and qBefore2q[1] > 0:
            x1 = x1
        if qBefore2q[0] < 0 and qBefore2q[1] > 0:
            x1 = np.pi - x1
        if qBefore2q[0] < 0 and qBefore2q[1] < 0:
            x1 = np.pi + x1
        if qBefore2q[0] > 0 and qBefore2q[1] < 0:
            x1 = 2 * np.pi - x1
        if qBefore2q[0] > 0 and qBefore2q[1] == 0:
            x1 = 0
        if qBefore2q[0] == 0 and qBefore2q[1] > 0:
            x1 = np.pi / 2
        if qBefore2q[0] < 0 and qBefore2q[1] == 0:
            x1 = np.pi
        if qBefore2q[0] == 0 and qBefore2q[1] < 0:
            x1 = np.pi * 3 / 2


        # 根据不同象限计算与x正半轴的角度
        if q2qNext[0] > 0 and q2qNext[1] > 0:
            x2 = x2
        if q2qNext[0] < 0 and q2qNext[1] > 0:
            x2 = np.pi - x2
        if q2qNext[0] < 0 and q2qNext[1] < 0:
            x2 = np.pi + x2
        if q2qNext[0] > 0 and q2qNext[1] < 0:
            x2 = 2 * np.pi - x2
        if q2qNext[0] > 0 and q2qNext[1] == 0:
            x2 = 0
        if q2qNext[0] == 0 and q2qNext[1] > 0:
            x2 = np.pi / 2
        if q2qNext[0] < 0 and q2qNext[1] == 0:
            x2 = np.pi
        if q2qNext[0] == 0 and q2qNext[1] < 0:
            x2 = np.pi * 3 / 2

        # 约束航迹角x   xres为约束后的航迹角
        deltax1x2 = self.angleVec(q2qNext[0:2], qBefore2q[0:2])  # 利用点乘除以模长乘积求xoy平面投影的夹角
        if deltax1x2 < self.xmax:
            xres = x2
        elif x1 - x2 > 0 and x1 - x2 < np.pi:  # 注意这几个逻辑
            xres = x1 - self.xmax
        elif x1 - x2 > 0 and x1 - x2 > np.pi:
            xres = x1 + self.xmax
        elif x1 - x2 < 0 and x2 - x1 < np.pi:
            xres = x1 + self.xmax
        else:
            xres = x1 - self.xmax

        # 约束爬升角gam   注意：爬升角只用讨论在-pi/2到pi/2区间，这恰好与arcsin的值域相同。  gamres为约束后的爬升角
        if np.abs(gam1 - gam2) <= self.gammax:
            gamres = gam2
        elif gam2 > gam1:
            gamres = gam1 + self.gammax
        else:
            gamres = gam1 - self.gammax
        if gamres > self.maximumClimbingAngle:
            gamres = self.maximumClimbingAngle
        if gamres < self.maximumSubductionAngle:
            gamres = self.maximumSubductionAngle

        # 计算约束过后下一个点qNext的坐标
        Rq2qNext = self.distanceCost(q, qNext)
        deltax = Rq2qNext * np.cos(gamres) * np.cos(xres)
        deltay = Rq2qNext * np.cos(gamres) * np.sin(xres)
        deltaz = Rq2qNext * np.sin(gamres)

        qNext = q + np.array([deltax, deltay, deltaz])
        return x1, gam1, xres, gamres, qNext

    def initField(self, pos, V0, goal):
        """计算初始流场，返回列向量。"""
        temp1 = pos[0] - goal[0]
        temp2 = pos[1] - goal[1]
        temp3 = pos[2] - goal[2]
        temp4 = self.distanceCost(pos, goal)
        return -np.array([temp1, temp2, temp3], dtype=float).reshape(-1, 1) * V0 / temp4

    def calPathLen(self, path):
        """计算一个轨迹的长度。"""
        num = path.shape[0]
        len = 0
        for i in range(num - 1):
            len += self.distanceCost(path[i, :], path[i + 1, :])
        return len

    def trans(self, originalPoint, xNew, yNew, zNew):
        """
        坐标变换后地球坐标下坐标
        newX, newY, newZ是新坐标下三个轴上的方向向量
        返回列向量
        """
        lenx = self.calVecLen(xNew)
        cosa1 = xNew[0] / lenx
        cosb1 = xNew[1] / lenx
        cosc1 = xNew[2] / lenx

        leny = self.calVecLen(yNew)
        cosa2 = yNew[0] / leny
        cosb2 = yNew[1] / leny
        cosc2 = yNew[2] / leny

        lenz = self.calVecLen(zNew)
        cosa3 = zNew[0] / lenz
        cosb3 = zNew[1] / lenz
        cosc3 = zNew[2] / lenz

        B = np.array([[cosa1, cosb1, cosc1],
                      [cosa2, cosb2, cosc2],
                      [cosa3, cosb3, cosc3]],dtype=float)

        invB = np.linalg.inv(B)
        return np.dot(invB, originalPoint.T)

    def save_data(self):
        np.savetxt('./data_csv/start.csv', self.start, delimiter=',')
        np.savetxt('./data_csv/goal.csv', self.goal, delimiter=',')
        np.savetxt('./data_csv/obs_r_list.csv',self.obs_r,delimiter=',')
        for i in range(self.obs_num):
            np.savetxt('./data_csv/obs{}_trace.csv'.format(i),self.path[i],delimiter=',')


    @staticmethod
    def distanceCost(point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    @staticmethod
    def angleVec(vec1, vec2):  # 计算两个向量之间的夹角
        temp = np.dot(vec1, vec2) / np.sqrt(np.sum(vec1 ** 2)) / np.sqrt(np.sum(vec2 ** 2))
        temp = np.clip(temp, -1, 1)  # 可能存在精度误差导致上一步的temp略大于1，因此clip
        theta = np.arccos(temp)
        return theta


    @staticmethod
    def partialDerivativeSphere(obs, pos, r):
        """计算球障碍物方程偏导数，返回列向量。"""
        temp1 = pos[0] - obs[0]
        temp2 = pos[1] - obs[1]
        temp3 = pos[2] - obs[2]
        return np.array([temp1,temp2,temp3],dtype=float).reshape(-1,1)*2/r**2


    @staticmethod
    def calculateT(obs, pos, r):
        """计算T。"""
        temp1 = pos[0] - obs[0]
        temp2 = pos[1] - obs[1]
        temp3 = pos[2] - obs[2]
        return (temp1**2 + temp2**2 + temp3**2)/r**2

    @staticmethod
    def calVecLen(vec):
        """计算向量模长。"""
        return np.sqrt(np.sum(vec**2))

    @staticmethod
    def load_model(method):
        import sys
        sys.path.append('./IIFDS-{}-random_start'.format(method)) # 解决保存模型路径不是加载文件路径的情况
        dynamicController = torch.load('./IIFDS-{}-random_start/TrainedModel/dynamicActor.pkl'.format(method), map_location=device)
        return dynamicController


if __name__ == "__main__":
    from Dynamic_obstacle_avoidance.Method import transformAction  # 放这里，避免循环import
    from Dynamic_obstacle_avoidance.Method import get_reward_multiple
    env = Environment(1)              # 这里用不同的数字初始化不同的动态场景
    config = Config()
    METHOD = 'PPO'                       # 在这填写测试用的模型
    controller = env.load_model(METHOD)
    uav_pos = env.start
    uav_path = env.start.reshape(1,-1)   # 记录uav trace的array
    action_trace = np.array([]).reshape(-1,3)  # 记录动作轨迹的array （三个动作刚好n*3 shape）
    d_list = np.array([])                      # 存储距离威胁表面距离的表
    qBefore = [None,None,None]
    reward_sum = 0
    if_test_origin_ifds = False        # 是否使用未经训练的ifds算法
    threat_index = 0
    for step in range(500):
        data_dic = env.update_obs_pos(uav_pos)
        v_obs, obs_center, obs_R = data_dic['v'], data_dic['obsCenter'], data_dic['obs_r']
        if if_test_origin_ifds:
            a = [1,1.5,1.5]        # 默认ifds的三个动作取值
        else:
            state = env.calDynamicState(uav_pos,obs_center,obs_R,v_obs)
            state = torch.as_tensor(state, dtype=torch.float, device=device)
            action = controller(state).cpu().detach().numpy()
            a = transformAction(action, config.actionBound, config.act_dim)
            action_trace = np.append(action_trace,np.array(a).reshape(-1,3),axis=0)  # 保存动作的trace

        # 威胁系数计算
        if data_dic['d'] <= 0:
            threat_index = np.inf
        elif data_dic['d'] <= 0.4:
            threat_index += 1/data_dic['d']
        d_list = np.append(d_list,data_dic['d'])

        uav_next_pos = env.getqNext(uav_pos,obs_center,v_obs,obs_R,a[0],a[1],a[2],qBefore)
        if env.distanceCost(uav_next_pos,obs_center)<=obs_R:
            print("发生碰撞!")
        reward_sum += get_reward_multiple(env,uav_next_pos,data_dic)
        qBefore = uav_pos
        uav_pos = uav_next_pos
        if env.distanceCost(uav_pos,env.goal)<env.threshold:
            uav_path = np.vstack((uav_path, env.goal))
            _ = env.update_obs_pos(uav_pos)
            break
        uav_path = np.vstack((uav_path, uav_pos))
    print("路径的长度为{}，路径奖励值为{}，威胁指数为{}".format(env.calPathLen(uav_path),reward_sum,threat_index))

    # save trace
    np.savetxt('./data_csv/pathMatrix.csv',uav_path,delimiter=',')
    env.save_data()

    # save action trace
    np.savetxt('./data_csv/actionTrace{}.csv'.format(METHOD),action_trace,delimiter=',')

    # save distance between UAV and Obstacle
    np.savetxt('./data_csv/d_list_{}.csv'.format(if_test_origin_ifds),d_list,delimiter=',')









