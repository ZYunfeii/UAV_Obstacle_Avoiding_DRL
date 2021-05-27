#!/usr/bin/python
# -*- coding: utf-8 -*-
# Time: 2021-5-14
# Author: ZYunfei
# Name: Multi-Processing PPO
# File func: main func

from PPOModel_MP import *
from Dynamic_obstacle_avoidance.IIFDS import IIFDS
from Dynamic_obstacle_avoidance.Method import getReward, transformAction, setup_seed, test, test_multiple
from Dynamic_obstacle_avoidance.draw import Painter
import random
from Dynamic_obstacle_avoidance.config import Config
from copy import deepcopy
import multiprocessing
import numpy as np
from time import time

def main():
    conf = Config()
    net = GlobalNet(conf.obs_dim,conf.act_dim) # 状态维度9 动作维度1
    ppo = AgentPPO(deepcopy(net))
    process_num = 12
    pipe_dict = dict((i, (pipe1, pipe2)) for i in range(process_num) for pipe1, pipe2 in (multiprocessing.Pipe(),))
    child_process_list = []
    rewardList = {1:[],2:[],3:[],4:[],5:[],6:[]}        # 记录各个测试环境的reward
    timeList = list()                                      # 记录时间的list
    dataList = list()
    maxReward = -np.inf
    for i in range(process_num):
        pro = multiprocessing.Process(target=child_process, args=(pipe_dict[i][1],i))
        child_process_list.append(pro)
    [pipe_dict[i][0].send(net) for i in range(process_num)]
    [p.start() for p in child_process_list]
    begin_time = time()
    for episode in range(conf.MAX_EPISODE):
        buffer_list = list()
        for i in range(process_num):
            receive = pipe_dict[i][0].recv()
            data = receive[0]
            buffer_list.append(data)
        ppo.update_policy_mp(conf.batch_size,8,buffer_list)
        net.act.load_state_dict(ppo.act.state_dict())
        net.cri.load_state_dict(ppo.cri.state_dict())
        [pipe_dict[i][0].send(net) for i in range(process_num)]
        testReward = test_multiple(ppo.act, conf)
        # print('Episode:', episode, 'Reward1:%2f' % testReward[0], 'Reward2:%2f' % testReward[1],
        #       'Reward3:%2f' % testReward[2], 'Reward4:%2f' % testReward[3],
        #       'Reward5:%2f' % testReward[4], 'Reward6:%2f' % testReward[5],
        #       'average reward:%2f' % np.mean(testReward))
        for index, data in enumerate(testReward):
            rewardList[index + 1].append(data)

        # if episode > conf.MAX_EPISODE *2/ 3:    # 一定回合数后可以开始保存模型了
        #     if np.mean(testReward) > maxReward:   # 如果当前回合的测试rewrad超过了历史最优水平，则保存
        #         maxReward = np.mean(testReward)
        #         print('当前episode累计平均reward历史最佳，已保存模型！')
        #         torch.save(ppo.act, 'dynamicActor.pkl')

        timeList.append(time()-begin_time)
        print(f'==程序运行时间:{timeList[-1]} episode:{episode} average reward:{np.mean(testReward)}')
        dataList.append(np.mean(testReward))

    [p.terminate() for p in child_process_list]

    painter = Painter(load_csv=True,load_dir='average_reward.csv')
    painter.addData(dataList,'MP12-PPO',x=timeList)
    painter.saveData(save_dir='average_reward.csv')
    painter.drawFigure()

    # for index in range(1, 7):
    #     painter = Painter(load_csv=True, load_dir='figure_data_d{}.csv'.format(index))
    #     painter.addData(rewardList[index], 'IIFDS-PPO-MP')
    #     painter.saveData('figure_data_d{}.csv'.format(index))
    #     painter.drawFigure()

def child_process(pipe,processID):
    setup_seed(10)
    conf = Config()
    iifds = IIFDS()

    while True:
        net = pipe.recv()
        # print(f'==进程{processID}收到net')
        dynamicController = AgentPPO(net,if_explore=True)
        dynamicController.buffer.storage_list = list()
        step_counter = 0
        while step_counter < dynamicController.buffer.max_memo:
            q = iifds.start + np.random.random(3)*3
            qBefore = [None, None, None]
            iifds.reset()
            for step_sum in range(conf.MAX_STEP):
                dic = iifds.updateObs()  # 更新障碍物位置、速度信息
                vObs, obsCenter, obsCenterNext = dic['v'], dic['obsCenter'], dic['obsCenterNext']  # 获取障碍物信息
                obs = iifds.calDynamicState(q, obsCenter)  # 计算当前状态强化学习的state
                action, log_prob = dynamicController.select_action((obs,),if_explore=True)  # 根据state选择动作值 统一输出(-1,1)

                actionAfter = transformAction(np.tanh(action), conf.actionBound, conf.act_dim)  # 将-1到1线性映射到对应动作值
                qNext = iifds.getqNext(q, obsCenter, vObs, actionAfter[0],
                                       actionAfter[1], actionAfter[2],qBefore)  # 计算下一航路点
                obs_next = iifds.calDynamicState(qNext, obsCenterNext)  # 获得下一航路点下强化学习的state
                reward = getReward(obsCenterNext, qNext, q, qBefore, iifds)  # 获取单障碍物环境下的reward

                done = True if iifds.distanceCost(iifds.goal, qNext) < iifds.threshold else False  # 如果接近终点就结束本回合训练
                mask = 0.0 if done else dynamicController.gamma  # 强化学习中的mask

                dynamicController.buffer.push(reward, mask, obs, action, log_prob)  # 存储进缓存

                if done: break

                qBefore = q
                q = qNext
            step_counter += step_sum

        transition = dynamicController.buffer.sample_all()
        r = transition.reward
        m = transition.mask
        a = transition.action
        s = transition.state
        log = transition.log_prob
        data = (r, m, s, a, log)
        pipe.send((data,))


if __name__ == '__main__':
    main()