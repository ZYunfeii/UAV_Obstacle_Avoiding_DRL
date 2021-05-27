#!/usr/bin/python
# -*- coding: utf-8 -*-
# Time: 2021-2-18
# Author: ZYunfei
# Name: DDPG-APF
# File func: test func
import torch
import numpy as np
import matplotlib.pyplot as plt
from Static_obstacle_avoidance.ApfAlgorithm import APF
from Static_obstacle_avoidance.Method import choose_action, checkPath, drawActionCurve, getReward
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    apf = APF()  # 实例化APF问题
    centralizedActor = torch.load('TrainedModel/centralizedActor.pkl',map_location=device) #加载模型
    actionCurveDic = {'sphere':np.array([]), 'cylinder':np.array([]), 'cone':np.array([])}

    # apf.drawEnv()        #绘制环境
    q = apf.x0           #初始化位置
    qBefore = [None, None, None]
    rewardSum = 0
    for i in range(500):
        obsDicq = apf.calculateDynamicState(q)
        obs_sphere, obs_cylinder, obs_cone = obsDicq['sphere'], obsDicq['cylinder'], obsDicq['cone']
        obs_mix = obs_sphere + obs_cylinder + obs_cone
        obs = np.array([])  # 中心控制器接受所有状态集合
        for k in range(len(obs_mix)):
            obs = np.hstack((obs, obs_mix[k]))  # 拼接状态为一个1*n向量
        obs = torch.as_tensor(obs, dtype=torch.float, device=device)
        action = centralizedActor(obs).cpu().detach().numpy()
        # 分解动作向量
        action_sphere = action[0:apf.numberOfSphere]
        action_cylinder = action[apf.numberOfSphere:apf.numberOfSphere + apf.numberOfCylinder]
        action_cone = action[apf.numberOfSphere + apf.numberOfCylinder:apf.numberOfSphere + \
                             apf.numberOfCylinder + apf.numberOfCone]

        actionCurveDic['sphere'] = np.append(actionCurveDic['sphere'], action_sphere)
        actionCurveDic['cylinder'] = np.append(actionCurveDic['cylinder'], action_cylinder)
        actionCurveDic['cone'] = np.append(actionCurveDic['cone'], action_cone)

        qNext = apf.getqNext(apf.epsilon0, action_sphere, action_cylinder, action_cone, q, qBefore)
        qBefore = q

        flag = apf.checkCollision(qNext)
        rewardSum += getReward(flag, apf, qBefore, q, qNext)

        # apf.ax.plot3D([q[0], qNext[0]],[q[1],qNext[1]],[q[2], qNext[2]],color="deeppink", linewidth=2)           #绘制上一位置和这一位置
        q = qNext
        if apf.distanceCost(q,apf.qgoal) < apf.threshold:      #如果与goal距离小于阈值则退出循环并将goal位置添加到path
            apf.path = np.vstack((apf.path,apf.qgoal))
            # apf.ax.plot3D([apf.path[-2,0],apf.path[-1,0]],[apf.path[-2,1],apf.path[-1,1]],[apf.path[-2,2],apf.path[-1,2]],color="deeppink", linewidth=2)
            break

    # actionCurve1List = actionCurveDic['sphere'].reshape(-1, apf.numberOfSphere)
    # actionCurve2List = actionCurveDic['cylinder'].reshape(-1, apf.numberOfCylinder)
    # actionCurve3List = actionCurveDic['cone'].reshape(-1, apf.numberOfCone)
    # drawActionCurve(actionCurve1List, 'sphere(s)')
    # drawActionCurve(actionCurve2List, 'cylinder(s)')
    # drawActionCurve(actionCurve3List, 'cone(s)')
    checkPath(apf,apf.path)
    apf.saveCSV()

    # np.savetxt('F:\MasterDegree\毕业设计\FullyCentralizedDDPG\data_csv\ actionCurve1List.csv', actionCurve1List, delimiter=',')
    # np.savetxt('F:\MasterDegree\毕业设计\FullyCentralizedDDPG\data_csv\ actionCurve2List.csv', actionCurve2List, delimiter=',')
    # np.savetxt('F:\MasterDegree\毕业设计\FullyCentralizedDDPG\data_csv\ actionCurve3List.csv', actionCurve3List, delimiter=',')

    print('该路径的奖励总和为:%f' % rewardSum)

    plt.show()






