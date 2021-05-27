import numpy as np
import matplotlib.pyplot as plt
from pylab import *                     #解决matplotlib无法显示中文问题
mpl.rcParams['font.sans-serif'] = ['SimHei']

class APF:
    def __init__(self):
        self.obstacle = np.array([[6, 4],
                                  [6,7],
                                  ])
        self.Robstacle = np.array([1,1])  #仅做测试
        self.qgoal = np.array([10,10])
        self.x0 = np.array([4,2])
        self.stepSize = 0.1
        self.iter = 1000
        self.epsilon = 0.8      #引力因子
        self.eta = 0.2           #斥力因子
        self.dgoal = 5
        self.r0 = 4
        self.path = self.x0.copy()
        self.path = self.path[np.newaxis,:]
        self.threshold = 0.5

    def distanceCost(self,point1,point2):
        return np.sqrt(np.sum((point1 - point2)**2))

    def attraction(self,q,qgoal,dgoal,epsilon):
        r = self.distanceCost(q,qgoal)
        if r <= dgoal:
            fx = epsilon * (qgoal[0] - q[0])
            fy = epsilon * (qgoal[1] - q[1])
        else:
            fx = dgoal * epsilon * (qgoal[0] - q[0]) / r
            fy = dgoal * epsilon * (qgoal[1] - q[1]) / r
        return np.array([fx,fy])

    def differential(self,q,other):
        output1 = (q[0] - other[0]) / self.distanceCost(q,other)
        output2 = (q[1] - other[1]) / self.distanceCost(q,other)
        return np.array([output1,output2])

    def repulsion(self,q,obstacle,r0,eta,qgoal):
        f0 = np.array([0,0])
        Rq2qgoal = self.distanceCost(q,qgoal)
        for i in range(obstacle.shape[0]):
            r = self.distanceCost(q,obstacle[i,:])
            if r <= r0:
                tempfvec = eta * (1 / r - 1 / r0) * Rq2qgoal ** 2 / r ** 2 * self.differential(q, obstacle[i,:]) \
                           + eta * (1/r - 1/r0) ** 2 * Rq2qgoal * self.differential(q,qgoal)
                f0 = f0 + tempfvec
            else:
                tempfvec = np.array([0,0])
                f0 = f0 + tempfvec
        return f0

    def loop(self):
        q = self.x0.copy()           #初始化位置
        for i in range(self.iter):
            Attraction = self.attraction(q,self.qgoal,self.dgoal,self.epsilon)
            Repulsion = self.repulsion(q,self.obstacle,self.r0,self.eta,self.qgoal)
            compositeForce = Attraction + Repulsion
            unitCompositeForce = compositeForce / np.sqrt(np.sum((compositeForce) ** 2))
            q = q + self.stepSize * unitCompositeForce
            self.path = np.vstack((self.path,q))
            if self.distanceCost(q,self.qgoal) < self.threshold:
                self.path = np.vstack((self.path,self.qgoal))
                break
    def draw(self):
        plt.scatter(self.obstacle[:,0], self.obstacle[:,1], marker='o', color='green', s=40, label='障碍物')
        plt.scatter(self.qgoal[0], self.qgoal[1], marker='o', color='red', s=100, label='目标点')
        plt.scatter(self.x0[0], self.x0[1], marker='o', color='blue', s=100, label='起始点')
        plt.plot(self.path[:,0],self.path[:,1],color="deeppink",linewidth=2,label = '无人机飞行轨迹')
        plt.legend(loc='best')  # 设置 图例所在的位置 使用推荐位置
        plt.grid()
        for i in range(self.Robstacle.shape[0]):
            self.drawCircle(self.obstacle[i,:],self.Robstacle[i])
        plt.axis('equal')
        plt.show()

    def drawCircle(self,pos,r):    #仅做测试
        theta = np.arange(0, 2 * np.pi, 0.01)
        a = pos[0]
        b = pos[1]
        x = a + r * np.cos(theta)
        y = b + r * np.sin(theta)
        plt.plot(x, y)

    def calculateTotalDistance(self):
        sum = 0
        for i in range(self.path.shape[0]-1):
            sum += self.distanceCost(self.path[i,:],self.path[i+1,:])
        return sum

if __name__ == "__main__":
    apf = APF()
    apf.loop()
    apf.draw()
    print("轨迹距离为:",apf.calculateTotalDistance())
