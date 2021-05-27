#!/usr/bin/python
# -*- coding: utf-8 -*-
# Time: 2021-4-28
# Author: ZYunfei
# Name: APF Algorithm for Dynamic Environment
# File func: main func
import numpy as np

class APF:
    """单球形动态障碍APF避障"""
    def __init__(self):
        self.obs_trace = np.loadtxt(".\data_csv\obs_trace.csv",delimiter=",")
        self.obs_r = 1.5

        self.qgoal = np.array([10,10,5.5], dtype=float)
        self.x0 = np.array([0,2,5], dtype=float)
        self.step_size = 0.2
        self.dgoal = 5
        self.r0 = 5
        self.threshold = 0.2

        self.xmax = 10 / 180 * np.pi  # 偏航角速度最大值  每个步长允许变化的角度
        self.gammax = 10 / 180 * np.pi  # 爬升角速度最大值  每个步长允许变化的角度
        self.maximumClimbingAngle = 100 / 180 * np.pi  # 最大爬升角
        self.maximumSubductionAngle = - 75 / 180 * np.pi  # 最大俯冲角

        self.epsilon0 = 0.5
        self.eta0 = 0.05

    def attraction(self, q, epsilon):  # 计算引力的函数
        r = self.distanceCost(q, self.qgoal)
        if r <= self.dgoal:
            fx = epsilon * (self.qgoal[0] - q[0])
            fy = epsilon * (self.qgoal[1] - q[1])
            fz = epsilon * (self.qgoal[2] - q[2])
        else:
            fx = self.dgoal * epsilon * (self.qgoal[0] - q[0]) / r
            fy = self.dgoal * epsilon * (self.qgoal[1] - q[1]) / r
            fz = self.dgoal * epsilon * (self.qgoal[2] - q[2]) / r
        return np.array([fx, fy, fz])

    def repulsionForOneObstacle(self, q, eta, qobs): #这个版本的斥力计算函数计算的是一个障碍物的斥力 2020.12.24
        f0 = np.array([0, 0, 0])  # 初始化斥力的合力
        Rq2qgoal = self.distanceCost(q, self.qgoal)
        r = self.distanceCost(q, qobs)
        if r <= self.r0:
            tempfvec = eta * (1 / r - 1 / self.r0) * Rq2qgoal ** 2 / r ** 2 * self.differential(q, qobs) \
                       + eta * (1 / r - 1 / self.r0) ** 2 * Rq2qgoal * self.differential(q, self.qgoal)
            f0 = f0 + tempfvec
        else:
            tempfvec = np.array([0, 0, 0])
            f0 = f0 + tempfvec
        return f0

    def differential(self, q, other):   #向量微分
        output1 = (q[0] - other[0]) / self.distanceCost(q, other)
        output2 = (q[1] - other[1]) / self.distanceCost(q, other)
        output3 = (q[2] - other[2]) / self.distanceCost(q, other)
        return np.array([output1, output2, output3])

    def getqNext(self,q,qBefore,qObs):
        qBefore = np.array(qBefore)
        if qBefore[0] is None:
            unitCompositeForce = self.getUnitCompositeForce(q, qObs)
            qNext = q + self.step_size * unitCompositeForce  # 计算下一位置
        else:
            unitCompositeForce = self.getUnitCompositeForce(q, qObs)
            qNext = q + self.step_size * unitCompositeForce  # 计算下一位置
            _, _, _, _, qNext = self.kinematicConstrant(q, qBefore, qNext)
        return qNext

    def getUnitCompositeForce(self,q,qObs):
        Attraction = self.attraction(q, self.epsilon0)  # 计算引力
        Repulsion = np.array([0,0,0])
        tempD = self.distanceCost(q, qObs)
        repPoint = q + (qObs - q) * (tempD - self.obs_r) / tempD
        Repulsion = Repulsion + self.repulsionForOneObstacle(q, self.eta0, repPoint)
        compositeForce = Attraction + Repulsion  # 合力 = 引力 + 斥力
        unitCompositeForce = self.getUnitVec(compositeForce)  # 力单位化，apf中力只用来指示移动方向
        return unitCompositeForce

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

    @staticmethod
    def distanceCost(point1, point2):  # 求两点之间的距离函数
        return np.sqrt(np.sum((point1 - point2) ** 2))

    @staticmethod
    def getUnitVec(vec):  # 单位化向量方法
        unitVec = vec / np.sqrt(np.sum(vec ** 2))
        return unitVec

    @staticmethod
    def angleVec(vec1, vec2):  # 计算两个向量之间的夹角
        temp = np.dot(vec1, vec2) / np.sqrt(np.sum(vec1 ** 2)) / np.sqrt(np.sum(vec2 ** 2))
        temp = np.clip(temp, -1, 1)  # 可能存在精度误差导致上一步的temp略大于1，因此clip
        theta = np.arccos(temp)
        return theta

    def loop(self):
        q = self.x0.copy()
        qBefore = [None, None, None]
        path = self.x0.copy().reshape(-1,3)
        for i in range(len(self.obs_trace)):
            qNext = self.getqNext(q,qBefore,self.obs_trace[i,:])
            path = np.vstack((path,qNext))
            qBefore = q
            q = qNext
            if self.distanceCost(qNext, self.qgoal) < self.threshold:
                path = np.vstack((path, self.qgoal))
                break
        np.savetxt(".\data_csv\path.csv",path,delimiter=",")
        np.savetxt(".\data_csv\obs_r.csv",np.array([self.obs_r]),delimiter=",")

if __name__ == "__main__":
    apf = APF()
    apf.loop()