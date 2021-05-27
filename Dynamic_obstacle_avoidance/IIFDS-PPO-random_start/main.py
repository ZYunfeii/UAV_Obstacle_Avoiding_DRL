#!/usr/bin/python
# -*- coding: utf-8 -*-
# Time: 2021-3-20
# Author: ZYunfei
# Name: Dynamic obstacle avoidance with PPO
# File func: main func

from PPOModel import *
from Dynamic_obstacle_avoidance.IIFDS import IIFDS
from Dynamic_obstacle_avoidance.Method import getReward, transformAction, setup_seed, test, test_multiple
from Dynamic_obstacle_avoidance.draw import Painter
import matplotlib.pyplot as plt
import random
from Dynamic_obstacle_avoidance.config import Config
import os
from time import time


if __name__ == "__main__":
    setup_seed(0)   # 设置随机数种子

    conf = Config()  # 初始化配置
    iifds = IIFDS()  # 初始化iifds算法环境
    obs_dim = conf.obs_dim  # state维度
    act_dim = conf.act_dim  # 动作维度

    dynamicController = AgentPPO(obs_dim, act_dim)
    if conf.if_load_weights and \
       os.path.exists('TrainedModel/act_weights.pkl') and \
       os.path.exists('TrainedModel/cri_weights.pkl'):
        dynamicController.ac.load_state_dict(torch.load('TrainedModel/act_weights.pkl'))
        dynamicController.ac_targ.load_state_dict(torch.load('TrainedModel/cri_weights.pkl'))

    actionBound = conf.actionBound  # 动作范围

    MAX_EPISODE = conf.MAX_EPISODE # 最大回合数
    MAX_STEP = conf.MAX_STEP  # 每回合最大步数
    batch_size = conf.batch_size
    rewardList = {1:[],2:[],3:[],4:[],5:[],6:[]}        # 记录各个测试环境的reward
    timeList = []
    dataList = []
    maxReward = -np.inf        # 这个变量用于在训练过程中保存目前最优reward模型的参数，先置负无穷

    begin_time = time()
    for episode in range(MAX_EPISODE):
        dynamicController.buffer.storage_list = list()  # ppo是online算法，每次都清空缓存再收集数据训练
        step_counter = 0
        reward_sum = 0
        while step_counter < dynamicController.buffer.max_memo:
            q = iifds.start + np.random.random(3)*3  # 随机起始位置，使训练场景更多样。
            qBefore = [None, None, None]             # qBefore主要用于算法内部实现UAV的运动学约束，第一个航路点没有上一个点，因此置为None
            iifds.reset()                            # iifds.reset后内部将随机出一个单障碍物动态环境用于训练
            for step_sum in range(MAX_STEP):
                dic = iifds.updateObs()      # 更新障碍物位置、速度信息
                vObs, obsCenter, obsCenterNext = dic['v'], dic['obsCenter'], dic['obsCenterNext']  # 获取障碍物信息
                obs = iifds.calDynamicState(q, obsCenter)     # 计算当前状态强化学习的state
                action, log_prob = dynamicController.select_action((obs,))  # 根据state选择动作值 统一输出(-1,1)

                actionAfter = transformAction(np.tanh(action), actionBound, act_dim)  # 将-1到1线性映射到对应动作值
                qNext = iifds.getqNext(q, obsCenter, vObs, actionAfter[0], actionAfter[1], actionAfter[2], qBefore) # 计算下一航路点
                obs_next = iifds.calDynamicState(qNext, obsCenterNext) # 获得下一航路点下强化学习的state
                reward = getReward(obsCenterNext, qNext, q, qBefore, iifds) # 获取单障碍物环境下的reward

                done = True if iifds.distanceCost(iifds.goal, qNext) < iifds.threshold else False # 如果接近终点就结束本回合训练
                mask = 0.0 if done else dynamicController.gamma  # 强化学习中的mask

                dynamicController.buffer.push(reward, mask, obs, action, log_prob) # 存储进缓存

                if done: break

                qBefore = q
                q = qNext
            step_counter += step_sum
        dynamicController.update_policy(batch_size, 8)  # 根据缓存数据训练模型，第二个参数为重复次数
        testReward = test_multiple(dynamicController.act,conf) # 在多障碍的环境下进行测试，判断agent是否真的学会了如何规划航路
        # print('Episode:', episode, 'Reward1:%2f' % testReward[0], 'Reward2:%2f' % testReward[1],
        #       'Reward3:%2f' % testReward[2], 'Reward4:%2f' % testReward[3],
        #       'Reward5:%2f' % testReward[4], 'Reward6:%2f' % testReward[5],
        #       'average reward:%2f' % np.mean(testReward))
        for index, data in enumerate(testReward):
            rewardList[index + 1].append(data)


        # if episode > MAX_EPISODE / 2:    # 一定回合数后可以开始保存模型了
        #     if np.mean(testReward) > maxReward:   # 如果当前回合的测试rewrad超过了历史最优水平，则保存
        #         maxReward = np.mean(testReward)
        #         print('当前episode累计平均reward历史最佳，已保存模型！')
        #         torch.save(dynamicController.act, 'TrainedModel/dynamicActor.pkl')
        #
        #         # 保存权重，便于各种场景迁移训练
        #         torch.save(dynamicController.act.state_dict(), 'TrainedModel/act_weights.pkl')
        #         # torch.save(dynamicController.cri.state_dict(), 'TrainedModel/cri_weights.pkl')  # cri可存储可不存储，在执行阶段不需要

        timeList.append(time()-begin_time)
        print(f'==程序运行时间:{timeList[-1]} episode:{episode} average_reward:{np.mean(testReward)}')
        dataList.append(np.mean(testReward))

    painter = Painter(load_csv=True, load_dir='./Multi-Processing-PPO/average_reward.csv')
    painter.addData(dataList, 'PPO',x=timeList)
    painter.saveData(save_dir='./Multi-Processing-PPO/average_reward.csv')
    painter.drawFigure()

    # 绘制
    # for index in range(1, 7):
    #     painter = Painter(load_csv=True, load_dir='F:/MasterDegree/毕业设计/实验数据/figure_data_d{}.csv'.format(index))
    #     painter.addData(rewardList[index], 'IIFDS-PPO')
    #     painter.saveData('F:/MasterDegree/毕业设计/实验数据/figure_data_d{}.csv'.format(index))
    #     painter.drawFigure()






