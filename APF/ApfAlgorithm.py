#!/usr/bin/python
# -*- coding: utf-8 -*-
# Time: 2021-3-8
# Author: ZYunfei
# Name: APF Algorithm
# File func: main func
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import operator

class APF:
    def __init__(self):
        #-------------------障碍物-------------------#
        """
        障碍物坐标在z>=0范围
        圆柱和圆锥从z=0开始
        """

        self.obstacle = np.array([[2,3,2],
                                  [4,7,1],
                                  [7,7,1]
                                  ], dtype=float)          # 球障碍物坐标
        self.Robstacle = np.array([2,1,1], dtype=float)  # 球半径
        self.cylinder = np.array([[5,5],
                                  [8,4]
                                  ], dtype=float)          # 圆柱体障碍物坐标（只需要给定圆形的x,y即可，无顶盖）
        self.cylinderR = np.array([1,1], dtype=float)      # 圆柱体障碍物半径
        self.cylinderH = np.array([4,3], dtype=float)        # 圆柱体高度
        self.cone = np.array([[5,2]
                             ],dtype=float)                # 圆锥底面中心坐标
        self.coneR = np.array([2],dtype=float)             # 圆锥底面圆半径
        self.coneH = np.array([3],dtype=float)             # 圆锥高度

        self.numberOfSphere = self.obstacle.shape[0]       # 球形障碍物的数量
        self.numberOfCylinder = self.cylinder.shape[0]     # 圆柱体障碍物的数量
        self.numberOfCone = self.cone.shape[0]             # 圆锥障碍物的数量

        #----------------------------------------------#
        self.qgoal = np.array([10,7, 2.5], dtype=float)   # 目标点
        self.x0 = np.array([-2, 1, 2], dtype=float)        # 轨迹起始点
        self.stepSize = 0.2                               # 物体移动的固定步长
        self.dgoal = 5                                    # 当q与qgoal距离超过它时将衰减一部分引力
        self.r0 = 5                                       # 斥力超过这个范围后将不复存在
        self.threshold = 0.2                              # q与qgoal距离小于它时终止训练或者仿真
        #------------运动学约束------------#
        self.xmax = 10/180 * np.pi                        # 偏航角速度最大值  每个步长允许变化的角度
        self.gammax = 10/180 * np.pi                      # 爬升角速度最大值  每个步长允许变化的角度
        self.maximumClimbingAngle = 100/180 * np.pi       # 最大爬升角
        self.maximumSubductionAngle = - 75 / 180 * np.pi  # 最大俯冲角

        #-------------路径（每次getqNext会自动往path添加路径）---------#
        self.path = self.x0.copy()
        self.path = self.path[np.newaxis, :]              # 增加一个维度

        #-------------一些参考参数可选择使用-------------#
        self.epsilon0 = 0.5
        self.eta0 = 0.1

    def calculateDynamicState(self, q):  # 计算空间一个坐标下指向每个障碍物中心的向量,返回字典，键为障碍物名称 fix：2021.2.9
        dic = {'sphere':[], 'cylinder':[], 'cone':[]}
        for i in range(self.numberOfSphere):
            dic['sphere'].append(self.obstacle[i,:] - q)
        for i in range(self.numberOfCylinder):
            dic['cylinder'].append(np.hstack((self.cylinder[i,:],q[2])) - q)
        for i in range(self.numberOfCone):
            dic['cone'].append(np.hstack((self.cone[i,:],self.coneH[i]/2)) - q)
        return dic

    def inRepulsionArea(self, q):  # 计算一个点位r0半径范围内的障碍物索引, 返回字典{'sphere':[1,2,..],'cylinder':[0,1,..]}  2021.1.6
        dic = {'sphere':[], 'cylinder':[], 'cone':[]}
        for i in range(self.numberOfSphere):
            if self.distanceCost(q, self.obstacle[i,:]) < self.r0:
                dic['sphere'].append(i)
        for i in range(self.numberOfCylinder):
            if self.distanceCost(q[0:2], self.cylinder[i,:]) < self.r0:
                dic['cylinder'].append(i)
        for i in range(self.numberOfCone):
            if self.distanceCost(q[0:2], np.hstack((self.cone[i,:],self.coneH[i]/2))) <self.r0:
                dic['cone'].append(i)
        return dic


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

    def repulsion(self, q, eta):  # 这个版本的斥力计算函数计算的是斥力因子都相同情况下的斥力 2020.12.24被替换
        f0 = np.array([0, 0, 0])  # 初始化斥力的合力
        Rq2qgoal = self.distanceCost(q, self.qgoal)
        for i in range(self.obstacle.shape[0]):       #球的斥力
            r = self.distanceCost(q, self.obstacle[i, :])
            if r <= self.r0:
                tempfvec = eta * (1 / r - 1 / self.r0) * Rq2qgoal ** 2 / r ** 2 * self.differential(q, self.obstacle[i, :]) \
                           + eta * (1 / r - 1 / self.r0) ** 2 * Rq2qgoal * self.differential(q, self.qgoal)
                f0 = f0 + tempfvec
            else:
                tempfvec = np.array([0, 0, 0])
                f0 = f0 + tempfvec

        for i in range(self.cylinder.shape[0]):       #圆柱体的斥力
            r = self.distanceCost(q[0:2], self.cylinder[i, :])
            if r <= self.r0:
                repulsionCenter = np.hstack((self.cylinder[i,:],q[2]))
                tempfvec = eta * (1 / r - 1 / self.r0) * Rq2qgoal ** 2 / r ** 2 * self.differential(q, repulsionCenter) \
                           + eta * (1 / r - 1 / self.r0) ** 2 * Rq2qgoal * self.differential(q, self.qgoal)
                f0 = f0 + tempfvec
            else:
                tempfvec = np.array([0, 0, 0])
                f0 = f0 + tempfvec
        return f0

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

    def getqNext(self, epsilon, eta1List, eta2List, eta3List, q, qBefore):   #eta和epsilon需要外部提供，eta1List为球的斥力列表，eta2List为圆柱体的斥力表 fix:2021.2.9
        """
        当qBefore为[None, None, None]时，意味着q是航迹的起始点，下一位置不需要做运动学约束，否则进行运动学约束
        """
        qBefore = np.array(qBefore)
        if qBefore[0] is None:
            unitCompositeForce = self.getUnitCompositeForce(q, eta1List, eta2List, eta3List, epsilon)
            qNext = q + self.stepSize * unitCompositeForce  # 计算下一位置
        else:
            unitCompositeForce = self.getUnitCompositeForce(q, eta1List, eta2List, eta3List, epsilon)
            qNext = q + self.stepSize * unitCompositeForce  # 计算下一位置
            _, _, _, _, qNext = self.kinematicConstrant(q, qBefore, qNext)
        self.path = np.vstack((self.path, qNext))  # 记录轨迹
        return qNext

    def getUnitCompositeForce(self,q,eta1List, eta2List, eta3List, epsilon):
        Attraction = self.attraction(q, epsilon)  # 计算引力
        Repulsion = np.array([0,0,0])
        for i in range(len(eta1List)): #对每个球形障碍物分别计算斥力并相加
            tempD = self.distanceCost(q, self.obstacle[i,:])
            repPoint = q + (self.obstacle[i,:] - q) * (tempD - self.Robstacle[i]) / tempD
            Repulsion = Repulsion + self.repulsionForOneObstacle(q, eta1List[i], repPoint)
        for i in range(len(eta2List)):
            if not q[2] <= 0 and q[2] < self.cylinderH[i]:
                centerPoint = np.hstack((self.cylinder[i,:],q[2]))
                tempD = self.distanceCost(q, centerPoint)
                repPoint = q + (centerPoint - q) * (tempD - self.cylinderR[i]) / tempD
            elif q[2] >= self.cylinderH[i]:
                repPoint = np.hstack((self.cylinder[i,:], self.cylinderH[i]))
            else:
                repPoint = np.hstack((self.cylinder[i,:], 0))
            Repulsion = Repulsion + self.repulsionForOneObstacle(q, eta2List[i], repPoint)
        for i in range(len(eta3List)):
            if not q[2] <= 0 and q[2] < self.coneH[i]:
                centerPoint = np.hstack((self.cone[i,:], q[2]))
                tempD = self.distanceCost(q, centerPoint)
                tempR = self.coneR[i] - q[2] * self.coneR[i] / self.coneH[i]
                repPoint = q + (centerPoint - q) * (tempD - tempR) / tempD
            elif q[2] >= self.coneH[i]:
                repPoint = np.hstack((self.cone[i,:], self.coneH[i]))
            else:
                repPoint = np.hstack((self.cone[i,:], 0))
            Repulsion = Repulsion + self.repulsionForOneObstacle(q, eta3List[i], repPoint)
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

    def checkCollision(self, q):
        """
        检查一个位置是否碰撞障碍物,如果碰撞返回[0,障碍物类型index, 碰撞障碍物index]，如果没有碰撞返回[1,-1, -1]
        """
        for i in range(self.numberOfSphere):#球的检测碰撞
            if self.distanceCost(q, self.obstacle[i, :]) <= self.Robstacle[i]:
                return np.array([0,0,i])
        for i in range(self.numberOfCylinder): #圆柱体的检测碰撞
            if 0 <= q[2] <= self.cylinderH[i] and self.distanceCost(q[0:2], self.cylinder[i, :]) <= self.cylinderR[i]:
                return np.array([0,1,i])
        for i in range(self.numberOfCone):
            if q[2] >= 0 and self.distanceCost(q[0:2], self.cone[i,:]) <= self.coneR[i] - q[2] * self.coneR[i] / self.coneH[i]:
                return np.array([0,2,i])
        return np.array([1,-1, -1])   #不撞的编码

    @staticmethod
    def distanceCost(point1, point2):  # 求两点之间的距离函数
        return np.sqrt(np.sum((point1 - point2) ** 2))

    @staticmethod
    def angleVec(vec1, vec2):  # 计算两个向量之间的夹角
        temp = np.dot(vec1, vec2) / np.sqrt(np.sum(vec1 ** 2)) / np.sqrt(np.sum(vec2 ** 2))
        temp = np.clip(temp,-1,1)  # 可能存在精度误差导致上一步的temp略大于1，因此clip
        theta = np.arccos(temp)
        return theta

    @staticmethod
    def getUnitVec(vec):   #单位化向量方法
        unitVec = vec / np.sqrt(np.sum(vec ** 2))
        return unitVec

    def calculateLength(self):
        """
        对类中自带的path进行距离计算，path会在getqNext函数中自动添加位置
        """
        sum = 0  # 轨迹距离初始化
        for i in range(self.path.shape[0] - 1):
            sum += apf.distanceCost(self.path[i, :], self.path[i + 1, :])
        return sum

    def saveCSV(self):   #保存数据方便matlab绘图(受不了matplotlib三维了)
        np.savetxt('.\data_csv\pathMatrix.csv', self.path, delimiter=',')
        np.savetxt('.\data_csv\obstacleMatrix.csv', self.obstacle, delimiter=',')
        np.savetxt('.\data_csv\RobstacleMatrix.csv', self.Robstacle, delimiter=',')
        np.savetxt('.\data_csv\cylinderMatrix.csv', self.cylinder, delimiter=',')
        np.savetxt('.\data_csv\cylinderRMatrix.csv', self.cylinderR, delimiter=',')
        np.savetxt('.\data_csv\cylinderHMatrix.csv', self.cylinderH, delimiter=',')
        np.savetxt('.\data_csv\coneMatrix.csv', self.cone, delimiter=',')
        np.savetxt('.\data_csv\coneRMatrix.csv', self.coneR, delimiter=',')
        np.savetxt('.\data_csv\coneHMatrix.csv', self.coneH, delimiter=',')

        np.savetxt('.\data_csv\start.csv', self.x0, delimiter=',')
        np.savetxt('.\data_csv\goal.csv', self.qgoal, delimiter=',')

    #--------测试用方法---------#
    def loop(self):             #循环仿真
        q = self.x0.copy()
        qBefore = [None, None, None]
        eta1List = [0.1 for i in range(self.obstacle.shape[0])]
        eta2List = [0.1 for i in range(self.cylinder.shape[0])]
        eta3List = [0.1 for i in range(self.cone.shape[0])]
        for i in range(500):
            qNext = self.getqNext(self.epsilon0,eta1List,eta2List,eta3List, q,qBefore)  # qBefore是上一个点
            qBefore = q

            # self.ax.plot3D([q[0], qNext[0]], [q[1], qNext[1]], [q[2], qNext[2]], color="k",linewidth=2)  # 绘制上一位置和这一位置
            #动态障碍物
            # self.calculateDynamicSphereXYZ()
            # self.dynamicSpherePath = np.vstack((self.dynamicSpherePath,self.dynamicSphereXYZ))
            # if i > 0:
            #     dynamicSphereImg.remove()
            # dynamicSphereImg = self.drawSphere(self.dynamicSphereXYZ, self.dynamicSphereR0)

            q = qNext

            if self.distanceCost(qNext,self.qgoal) < self.threshold:   #当与goal之间距离小于threshold时结束仿真，并将goal的坐标放入path
                self.path = np.vstack((self.path,self.qgoal))
                # self.dynamicSpherePath = np.vstack((self.dynamicSpherePath, self.dynamicSphereXYZ)) #保持和path的维度一致.方便matlab绘图
                # self.ax.plot3D([qNext[0], self.qgoal[0]], [qNext[1], self.qgoal[1]], [qNext[2], self.qgoal[2]], color="k",linewidth=2)  # 绘制上一位置和这一位置
                break
            # plt.pause(0.001)



if __name__ == "__main__":
    apf = APF()
    apf.loop()
    apf.saveCSV()
    print('轨迹距离为：',apf.calculateLength())







