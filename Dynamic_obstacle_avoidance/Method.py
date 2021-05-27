#!/usr/bin/python
# -*- coding: utf-8 -*-
# Time: 2021-3-10
# Author: ZYunfei
# Name: Dynamic obstacle avoidance with reinforcement learning
# File func: various methods

import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from Dynamic_obstacle_avoidance.Multi_obstacle_environment_test import Environment
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getReward(obsCenter, qNext, q, qBefore, iifds):
    """
    获取强化学习奖励值函数
    """
    distance = iifds.distanceCost(qNext, obsCenter)
    flag = True if distance <= iifds.obsR else False
    reward = 0
    if flag: # 与障碍物有交点
        reward += (distance - iifds.obsR)/iifds.obsR - 1
    else:
        if distance < iifds.obsR + 0.4:   # 威胁区
            tempR = iifds.obsR + 0.4
            reward += (distance - tempR)/tempR-0.3
        distance1 = iifds.distanceCost(qNext, iifds.goal)
        distance2 = iifds.distanceCost(iifds.start, iifds.goal)
        if distance1 > iifds.threshold:
            reward += -distance1/distance2
        else:
            reward += -distance1/distance2 + 3
        """奖励径直轨迹"""
        # q2qNext = qNext - q
        # q2goal = iifds.goal - q
        # theta = np.arccos(np.clip(q2qNext.dot(q2goal)/iifds.calVecLen(q2qNext)/iifds.calVecLen(q2goal),-1,1))
        # reward += -abs(theta) / (2*np.pi) * 0.2

        # if qBefore[0] is not None:
        #     x1, gam1, xres, gamres, _ = iifds.kinematicConstrant(q, qBefore, qNext)
        #     xDot = np.abs(x1 - xres)
        #     gamDot = np.abs(gam1 - gamres)
        #     reward += (- xDot / iifds.xmax - gamDot / iifds.gammax) * 0.5

    return reward

def get_reward_multiple(env,qNext,dic):
    """多动态障碍环境获取reward函数"""
    reward = 0
    distance = env.distanceCost(qNext,dic['obsCenter'])
    if distance<=dic['obs_r']:
        reward += (distance-dic['obs_r'])/dic['obs_r']-1
    else:
        if distance < dic['obs_r'] + 0.4:
            tempR = dic['obs_r'] + 0.4
            reward += (distance-tempR)/tempR-0.3
        distance1 = env.distanceCost(qNext, env.goal)
        distance2 = env.distanceCost(env.start, env.goal)
        if distance1 > env.threshold:
            reward += -distance1 / distance2
        else:
            reward += -distance1 / distance2 + 3
    return reward


def drawActionCurve(actionCurveList):
    """
    :param actionCurveList: 动作值列表
    :return: None 绘制图像
    """
    plt.figure()
    for i in range(actionCurveList.shape[1]):
        array = actionCurveList[:, i]
        if i == 0: label = 'row0'
        if i == 1: label = 'sigma0'
        if i == 2: label = 'theta'
        plt.plot(np.arange(array.shape[0]), array, linewidth=2, label=label)
    plt.title('Variation diagram')
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('value')
    plt.legend(loc='best')

def checkPath(apf):
    sum = 0
    for i in range(apf.path.shape[0] - 1):
        sum += apf.distanceCost(apf.path[i,:], apf.path[i+1,:])
    for i, j in zip(apf.path, apf.dynamicSphere_Path):
        if apf.distanceCost(i,j) <= apf.dynamicSphere_R:
            print('与障碍物有交点，轨迹距离为：', sum)
            return
    print('与障碍物无交点，轨迹距离为：', sum)

def transformAction(actionBefore, actionBound, actionDim):
    """将强化学习输出的动作映射到指定的动作范围"""
    actionAfter = []
    for i in range(actionDim):
        action_i = actionBefore[i]
        action_bound_i = actionBound[i]
        actionAfter.append((action_i+1)/2*(action_bound_i[1] - action_bound_i[0]) + action_bound_i[0])
    return actionAfter


def test(iifds, pi, conf):
    """动态单障碍环境测试训练效果"""
    iifds.reset()    # 重置环境
    q = iifds.start
    qBefore = [None, None, None]
    rewardSum = 0
    for i in range(500):
        dic = iifds.updateObs(if_test=True)
        vObs, obsCenter, obsCenterNext = dic['v'], dic['obsCenter'], dic['obsCenterNext']
        obs = iifds.calDynamicState(q, obsCenter)
        obs = torch.as_tensor(obs, dtype=torch.float, device=device)
        action = pi(obs).cpu().detach().numpy()
        action = transformAction(action, conf.actionBound, conf.act_dim)
        # 与环境交互
        qNext = iifds.getqNext(q, obsCenter, vObs, action[0], action[1], action[2], qBefore)
        rewardSum += getReward(obsCenterNext, qNext, q, qBefore, iifds)

        qBefore = q
        q = qNext

        if iifds.distanceCost(q, iifds.goal) < iifds.threshold:
            break
    return rewardSum

def test_multiple(pi, conf):
    """动态多障碍环境测试模型效果"""
    reward_list = []
    for index in range(1,7):      # 从1-6一共6个测试环境，遍历测试
        env = Environment(index)
        env.reset()
        q = env.start
        qBefore = [None, None, None]
        rewardSum = 0
        for i in range(500):
            data_dic = env.update_obs_pos(q)
            v_obs, obs_center, obs_R = data_dic['v'], data_dic['obsCenter'], data_dic['obs_r']
            state = env.calDynamicState(q, obs_center, obs_R, v_obs)
            state = torch.as_tensor(state, dtype=torch.float, device=device)
            action = pi(state).cpu().detach().numpy()
            a = transformAction(action, conf.actionBound, conf.act_dim)
            qNext = env.getqNext(q, obs_center, v_obs, obs_R, a[0], a[1], a[2], qBefore)
            rewardSum += get_reward_multiple(env,qNext,data_dic)
            qBefore = q
            q = qNext
            if env.distanceCost(q, env.goal) < env.threshold:
                break
        reward_list.append(rewardSum)
    return reward_list


def setup_seed(seed):
    """设置随机数种子函数"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

