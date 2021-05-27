#!/usr/bin/python
# -*- coding: utf-8 -*-
# Time: 2021-1-15
# Author: ZYunfei
# Name: MADDPG-APF
# File func: test func
import torch
import numpy as np
import matplotlib.pyplot as plt
from Static_obstacle_avoidance.ApfAlgorithm import APF
from Static_obstacle_avoidance.Method import choose_action, checkPath, drawActionCurve, getReward
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    apf = APF()  # 实例化APF问题
    #加载模型
    actorDic = {'sphere':[], 'cylinder':[], 'cone':[]}
    for i in range(apf.numberOfSphere):
        actorDic['sphere'].append(torch.load('TrainedModel/Actor1.%d.pkl'%i,map_location=device))
    for i in range(apf.numberOfCylinder):
        actorDic['cylinder'].append(torch.load('TrainedModel/Actor2.%d.pkl'%i,map_location=device))
    for i in range(apf.numberOfCone):
        actorDic['cone'].append(torch.load('TrainedModel/Actor3.%d.pkl'%i,map_location=device))

    actionCurveDic = {'sphere':np.array([]), 'cylinder':np.array([]), 'cone':np.array([])}

    # apf.drawEnv()        #绘制环境
    q = apf.x0           #初始化位置
    qBefore = [None, None, None]
    rewardSum = 0
    for i in range(500):
        stateq = apf.calculateDynamicState(q)
        state1, state2, state3 = stateq['sphere'], stateq['cylinder'], stateq['cone']  #计算相对位置作为state
        eta1List = choose_action(actorDic['sphere'], state1)         #通过相对位置选择引力因子和斥力因子
        actionCurveDic['sphere'] = np.append(actionCurveDic['sphere'], eta1List)
        eta2List = choose_action(actorDic['cylinder'], state2)
        actionCurveDic['cylinder'] = np.append(actionCurveDic['cylinder'], eta2List)
        eta3List = choose_action(actorDic['cone'], state3)
        actionCurveDic['cone'] = np.append(actionCurveDic['cone'], eta3List)

        # interact with enviroment
        qNext = apf.getqNext(apf.epsilon0, eta1List, eta2List, eta3List, q, qBefore)
        qBefore = q

        flag = apf.checkCollision(qNext)
        rewardSum += getReward(flag, apf, None, None, qNext)

        # apf.ax.plot3D([q[0], qNext[0]],[q[1],qNext[1]],[q[2], qNext[2]],color="deeppink", linewidth=2)           #绘制上一位置和这一位置
        q = qNext
        if apf.distanceCost(q,apf.qgoal) < apf.threshold:      #如果与goal距离小于阈值则退出循环并将goal位置添加到path
            apf.path = np.vstack((apf.path,apf.qgoal))
            # apf.ax.plot3D([apf.path[-2,0],apf.path[-1,0]],[apf.path[-2,1],apf.path[-1,1]],[apf.path[-2,2],apf.path[-1,2]],color="deeppink", linewidth=2)
            break

    actionCurve1List = actionCurveDic['sphere'].reshape(-1, apf.numberOfSphere)
    actionCurve2List = actionCurveDic['cylinder'].reshape(-1, apf.numberOfCylinder)
    actionCurve3List = actionCurveDic['cone'].reshape(-1, apf.numberOfCone)
    drawActionCurve(actionCurve1List, 'sphere(s)')
    drawActionCurve(actionCurve2List, 'cylinder(s)')
    drawActionCurve(actionCurve3List, 'cone(s)')
    checkPath(apf,apf.path)
    apf.saveCSV()

    print('该路径的奖励总和为:%f' % rewardSum)

    plt.show()






