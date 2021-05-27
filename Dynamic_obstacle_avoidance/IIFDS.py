#!/usr/bin/python
# -*- coding: utf-8 -*-
# Time: 2021-3-9
# Author: ZYunfei
# Name: IIFDS-DDPG
# File func: enviroment class
import numpy as np
from Dynamic_obstacle_avoidance.dynamic_obstacle_environment import obs_list
from Dynamic_obstacle_avoidance.Method import getReward

class IIFDS:
    """使用IIFDS类训练时每次必须reset"""
    def __init__(self):
        """基本参数："""
        self.V0 = 1
        self.threshold = 0.2
        self.stepSize = 0.1
        self.lam = 8           # 越大考虑障碍物速度越明显

        self.obsR  = 1.5
        self.start = np.array([0,2,5],dtype=float)
        self.goal = np.array([10,10,5.5],dtype=float)

        self.timelog = 0        # 时间，用来计算动态障碍的位置
        self.timeStep = 0.1

        self.xmax = 10 / 180 * np.pi  # 偏航角速度最大值  每个步长允许变化的角度
        self.gammax = 10 / 180 * np.pi  # 爬升角速度最大值  每个步长允许变化的角度
        self.maximumClimbingAngle = 100 / 180 * np.pi  # 最大爬升角
        self.maximumSubductionAngle = - 75 / 180 * np.pi  # 最大俯冲角

        self.vObs = None
        self.vObsNext = None

        self.path = np.array([[]]).reshape(-1,3)        # 保存动态球的运动轨迹

        self.env_num = len(obs_list)
        self.env = obs_list[0]

    def reset(self):
        self.timelog = 0                                         # 重置时间
        self.path = np.array([[]]).reshape(-1, 3)                # 清空障碍路径记录表
        self.env = obs_list[np.random.randint(0, self.env_num)]  # 随机一个训练环境

    def updateObs(self,if_test=False):
        """返回位置与速度。"""
        if if_test:
            """测试环境"""
            self.timelog, dic = obs_list[4](self.timelog, self.timeStep)
        else:
            """否则用reset时随机的一个环境"""
            self.timelog, dic = self.env(self.timelog, self.timeStep)
        self.vObs = dic['v']
        self.path = np.vstack((self.path, dic['obsCenter']))
        return dic

    def calDynamicState(self, uavPos, obsCenter):
        """强化学习模型获得的state。"""
        s1 = (obsCenter - uavPos)*(self.distanceCost(obsCenter,uavPos)-self.obsR)/self.distanceCost(obsCenter,uavPos)
        s2 = self.goal - uavPos
        s3 = self.vObs
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

    def getqNext(self, uavPos, obsCenter, vObs, row0, sigma0, theta, qBefore):
        u = self.initField(uavPos, self.V0, self.goal)
        repulsiveMatrix = self.calRepulsiveMatrix(uavPos, obsCenter, self.obsR, row0)
        tangentialMatrix = self.calTangentialMatrix(uavPos, obsCenter, self.obsR, theta, sigma0)
        T = self.calculateT(obsCenter, uavPos, self.obsR)
        vp = np.exp(-T / self.lam) * vObs
        M = np.eye(3) + repulsiveMatrix + tangentialMatrix
        ubar = (M.dot(u - vp.reshape(-1, 1)).T + vp.reshape(1, -1)).squeeze()
        # 限制ubar的模长，避免进入障碍内部后轨迹突变
        if self.calVecLen(ubar) > 5:
            ubar = ubar/self.calVecLen(ubar)*5
        if qBefore[0] is None:
            uavNextPos = uavPos + ubar * self.stepSize
        else:
            uavNextPos = uavPos + ubar * self.stepSize
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

    def loop(self):
        uavPos = self.start
        row0 = 0.5
        theta = 0.5
        sigma0 = 0.5
        path = self.start.reshape(1,-1)
        qBefore = [None, None, None]
        reward = 0
        for i in range(500):
            dic = self.updateObs(if_test=True)
            vObs, obsCenter = dic['v'], dic['obsCenter']
            uavNextPos = self.getqNext(uavPos, obsCenter, vObs, row0, sigma0, theta, qBefore)
            reward += getReward(obsCenter, uavNextPos, uavPos, qBefore, self)
            qBefore = uavPos
            uavPos = uavNextPos
            if self.distanceCost(uavPos,self.goal)<self.threshold:
                path = np.vstack((path, self.goal))
                _ = iifds.updateObs(if_test=True)
                break
            path = np.vstack((path, uavPos))
        print('路径的长度为:%f'%self.calPathLen(path))
        print('奖励为:%f' % reward)
        np.savetxt('./data_csv/pathMatrix.csv', path, delimiter=',')
        self.save_data()

    @staticmethod
    def distanceCost(point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def initField(self, pos, V0, goal):
        """计算初始流场，返回列向量。"""
        temp1 = pos[0] - goal[0]
        temp2 = pos[1] - goal[1]
        temp3 = pos[2] - goal[2]
        temp4 = self.distanceCost(pos,goal)
        return -np.array([temp1,temp2,temp3],dtype=float).reshape(-1,1)*V0/temp4

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

    def calPathLen(self, path):
        """计算一个轨迹的长度。"""
        num = path.shape[0]
        len = 0
        for i in range(num-1):
            len += self.distanceCost(path[i,:], path[i+1,:])
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



    @staticmethod
    def calVecLen(vec):
        """计算向量模长。"""
        return np.sqrt(np.sum(vec**2))

    @staticmethod
    def angleVec(vec1, vec2):  # 计算两个向量之间的夹角
        temp = np.dot(vec1, vec2) / np.sqrt(np.sum(vec1 ** 2)) / np.sqrt(np.sum(vec2 ** 2))
        temp = np.clip(temp, -1, 1)  # 可能存在精度误差导致上一步的temp略大于1，因此clip
        theta = np.arccos(temp)
        return theta


    def save_data(self):
        np.savetxt('./data_csv/start.csv', self.start, delimiter=',')
        np.savetxt('./data_csv/goal.csv', self.goal, delimiter=',')
        np.savetxt('./data_csv/obs_r.csv', np.array([self.obsR]), delimiter=',')
        np.savetxt('./data_csv/obs_trace.csv', self.path, delimiter=',')





if __name__ == "__main__":
    iifds = IIFDS()
    # uavPos = np.array([1,2,3])
    # obsCenter = np.array([2,7,3])
    # obsR = 1
    # row0 = 1
    #
    # print(iifds.calTangentialMatrix(uavPos, obsCenter, obsR, 0.2, 0.5))
    iifds.loop()
