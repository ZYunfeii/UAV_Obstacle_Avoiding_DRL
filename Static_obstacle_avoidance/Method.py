#!/usr/bin/python
# -*- coding: utf-8 -*-
# Time: 2021-1-15
# Author: ZYunfei
# Name: MADDPG-APF
# File func: method func
import matplotlib.pyplot as plt
import torch
import numpy as np
import random

def getReward(flag,apf, qBefore, q,qNext):           #计算reward函数
    """
    :param flag: 碰撞检测标志位
    :param apf: 环境
    :param qBefore: 上一个位置
    :param q: 当前位置
    :param qNext: 下一个计算得到的位置
    :return: 奖励reward
    """
    reward = 0
    # Reward_col
    if flag[0] == 0:
        if flag[1] == 0:
            distance = apf.distanceCost(qNext, apf.obstacle[flag[2],:])
            reward += (distance - apf.Robstacle[flag[2]])/apf.Robstacle[flag[2]] -1
        if flag[1] == 1:
            distance = apf.distanceCost(qNext[0:2], apf.cylinder[flag[2],:])
            reward += (distance - apf.cylinderR[flag[2]])/apf.cylinderR[flag[2]] -1
        if flag[1] == 2:
            distance = apf.distanceCost(qNext[0:2], apf.cone[flag[2],:])
            r = apf.coneR[flag[2]] - qNext[2] * apf.coneR[flag[2]] / apf.coneH[flag[2]]
            reward += (distance - r)/r - 1
    else:
        # Reward_len
        distance1 = apf.distanceCost(qNext, apf.qgoal)
        distance2 = apf.distanceCost(apf.x0, apf.qgoal)
        if distance1 > apf.threshold:
            reward += -distance1/distance2
        else:
            reward += -distance1/distance2 + 3
    # if qNext[2]<=0:
    #     reward += qNext[2]*0.5

        # Reward_Ang
        # x1, gam1, xres, gamres, _ = apf.kinematicConstrant(q, qBefore, qNext)
        # if x1 != None:
        #     xDot = np.abs(x1 - xres)
        #     gamDot = np.abs(gam1 - gamres)
        #     reward += (- xDot / apf.xmax - gamDot / apf.gammax) * 0.1


    return reward


def choose_action(ActorList, s):
    """
    :param ActorList: actor网络列表
    :param s: 每个agent的state append形成的列表
    :return: 每个actor给每个对应的state进行动作输出的值append形成的列表
    """
    actionList = []
    for i in range(len(ActorList)):
        state = s[i]
        state = torch.as_tensor(state, dtype=torch.float, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        a = ActorList[i](state).cpu().detach().numpy()
        actionList.append(a[0])
    return actionList

def drawActionCurve(actionCurveList, obstacleName):
    """
    :param actionCurveList: 动作值列表
    :return: None 绘制图像
    """
    plt.figure()
    for i in range(actionCurveList.shape[1]):
        array = actionCurveList[:,i]
        plt.plot(np.arange(array.shape[0]), array, linewidth = 2, label = 'Rep%d curve'%i)
    plt.title('Variation diagram of repulsion factor of %s' %obstacleName)
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('value')
    # plt.legend(loc='best')

def checkCollision(apf, path):  # 检查轨迹是否与障碍物碰撞
    """
    :param apf: 环境
    :param path: 一个路径形成的列表
    :return: 1代表无碰撞 0代表碰撞
    """
    for i in range(path.shape[0]):
        if apf.checkCollision(path[i,:])[0] == 0:
            return 0
    return 1

def checkPath(apf, path):
    """
    :param apf: 环境
    :param path: 路径形成的列表
    :return: None 打印是否与障碍物有交点以及path的总距离
    """
    sum = 0  # 轨迹距离初始化
    for i in range(path.shape[0] - 1):
        sum += apf.distanceCost(path[i, :], path[i + 1, :])
    if checkCollision(apf, path) == 1:
        print('与障碍物无交点，轨迹距离为：', sum)
    else:
        print('与障碍物有交点，轨迹距离为：', sum)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def transformAction(actionBefore, actionBound, actionDim):
    actionAfter = []
    for i in range(actionDim):
        action_i = actionBefore[i]
        actionAfter.append((action_i+1)/2*(actionBound[1] - actionBound[0]) + actionBound[0])
    return actionAfter

class Arguments:
    def __init__(self, apf):
        self.obs_dim = 6 * (apf.numberOfSphere + apf.numberOfCylinder + apf.numberOfCone)
        self.act_dim = 1 * (apf.numberOfSphere + apf.numberOfCylinder + apf.numberOfCone)
        self.act_bound = [0.1, 3]













