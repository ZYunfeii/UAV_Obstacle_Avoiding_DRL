#!/usr/bin/python
# -*- coding: utf-8 -*-
# Time: 2021-3-29
# Author: ZYunfei
# Name: Dynamic obstacle avoidance with reinforcement learning
# File func: test
"""这个文件是针对单动态障碍的测试环境，测试后打开matlab运行test.m即可得到可视化结果"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from Dynamic_obstacle_avoidance.IIFDS import IIFDS
from Dynamic_obstacle_avoidance.Method import getReward, transformAction, drawActionCurve
from Dynamic_obstacle_avoidance.config import Config
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    conf = Config()
    actionBound = conf.actionBound

    iifds = IIFDS()
    dynamicController = torch.load('TrainedModel/dynamicActor.pkl',map_location=device)
    actionCurve = np.array([])

    q = iifds.start
    qBefore = [None, None, None]
    path = iifds.start.reshape(1,-1)
    rewardSum = 0
    for i in range(500):
        dic = iifds.updateObs(if_test=True)
        vObs, obsCenter, obsCenterNext = dic['v'], dic['obsCenter'], dic['obsCenterNext']
        obs = iifds.calDynamicState(q, obsCenter)
        obs = torch.as_tensor(obs, dtype=torch.float, device=device)
        action = dynamicController(obs).cpu().detach().numpy()
        action = transformAction(action, actionBound, conf.act_dim)
        actionCurve = np.append(actionCurve, action)
        # 与环境交互
        qNext = iifds.getqNext(q, obsCenter, vObs, action[0], action[1], action[2], qBefore)
        rewardSum += getReward(obsCenterNext, qNext, q, qBefore, iifds)

        qBefore = q
        q = qNext

        if iifds.distanceCost(q, iifds.goal) < iifds.threshold:
            path = np.vstack((path, iifds.goal))
            _ = iifds.updateObs(if_test=True)
            break
        path = np.vstack((path, q))
    drawActionCurve(actionCurve.reshape(-1,3))
    np.savetxt('./data_csv/pathMatrix.csv', path, delimiter=',')
    iifds.save_data()
    routeLen = iifds.calPathLen(path)
    print('该路径的奖励总和为:%f，路径的长度为:%f' % (rewardSum,routeLen))
    plt.show()

