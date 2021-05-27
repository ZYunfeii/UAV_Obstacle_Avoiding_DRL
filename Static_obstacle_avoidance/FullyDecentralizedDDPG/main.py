from DDPGModel import *
from Static_obstacle_avoidance.ApfAlgorithm import *
from Static_obstacle_avoidance.Method import getReward, setup_seed
import random
import matplotlib.pyplot as plt
from Static_obstacle_avoidance.draw import Painter
def constructDdpgNetList(apf, s_dim, a_dim, a_bound):
    ddpgNetForSphere = []
    for i in range(apf.numberOfSphere):
        ddpgNetForSphere.append(DDPG(s_dim, a_dim, a_bound))
    ddpgNetForCylinder = []
    for i in range(apf.numberOfCylinder):
        ddpgNetForCylinder.append(DDPG(s_dim, a_dim, a_bound))
    ddpgNetForCone = []
    for i in range(apf.numberOfCone):
        ddpgNetForCone.append(DDPG(s_dim, a_dim, a_bound))
    return ddpgNetForSphere, ddpgNetForCylinder, ddpgNetForCone

def ddpgReplayBufferStore(apf, qNext, ddpgNetForSphere, ddpgNetForCylinder,ddpgNetForCone,
                          action1List, action2List, action3List, state1, state1_, state2, state2_,
                          state3, state3_, reward, done):

    for i in range(len(ddpgNetForSphere)):
        ddpgNetForSphere[i].replay_buffer.store(state1[i], action1List[i], reward, state1_[i], done)
    for i in range(len(ddpgNetForCylinder)):
        ddpgNetForCylinder[i].replay_buffer.store(state2[i], action2List[i], reward, state2_[i], done)
    for i in range(len(ddpgNetForCone)):
        ddpgNetForCone[i].replay_buffer.store(state3[i], action3List[i], reward, state3_[i], done)

    # dic = apf.inRepulsionArea(qNext)
    # for i in range(len(dic['sphere'])):
    #     index = dic['sphere'][i]
    #     ddpgNetForSphere[index].replay_buffer.store(state1[index], action1List[index], reward, state1_[index], done)
    # for i in range(len(dic['cylinder'])):
    #     index = dic['cylinder'][i]
    #     ddpgNetForCylinder[index].replay_buffer.store(state2[index], action2List[index], reward, state2_[index], done)
if __name__ == '__main__':
    setup_seed(5)

    obs_dim = 6
    act_dim = 1
    act_bound = [0.1, 3]
    apf = APF()

    ddpgNetForSphere, ddpgNetForCylinder, ddpgNetForCone = constructDdpgNetList(apf, obs_dim, act_dim, act_bound)

    MAX_EPISODE = 1000
    MAX_STEP = 500
    update_every = 50
    batch_size = 100
    noise = 0.3
    rewardList = []
    maxReward = -np.inf
    for episode in range(MAX_EPISODE):
        q = apf.x0
        apf.reset()
        rewardSum = 0
        qBefore = [None, None, None]
        noise *= 0.995
        if noise <= 0.1: noise = 0.1
        for j in range(MAX_STEP):
            obsDicq = apf.calculateDynamicState(q)
            obs_sphere, obs_cylinder, obs_cone = obsDicq['sphere'], obsDicq['cylinder'], obsDicq['cone']
            eta1List = []
            action1List = []
            eta2List = []
            action2List = []
            eta3List = []
            action3List = []
            if episode > 30:
                for k in range(apf.numberOfSphere):
                    action = ddpgNetForSphere[k].get_action(obs_sphere[k], noise_scale=noise)
                    action1List.append(action)
                    eta = action[0]
                    eta1List.append(eta)
                for k in range(apf.numberOfCylinder):
                    action = ddpgNetForCylinder[k].get_action(obs_cylinder[k], noise_scale=noise)
                    action2List.append(action)
                    eta = action[0]
                    eta2List.append(eta)
                for k in range(apf.numberOfCone):
                    action = ddpgNetForCone[k].get_action(obs_cone[k], noise_scale=noise)
                    action3List.append(action)
                    eta = action[0]
                    eta3List.append(eta)
            else:
                for k in range(apf.numberOfSphere):
                    action = [random.uniform(act_bound[0],act_bound[1])]
                    action1List.append(action)
                    eta = action[0]
                    eta1List.append(eta)
                for k in range(apf.numberOfCylinder):
                    action = [random.uniform(act_bound[0], act_bound[1])]
                    action2List.append(action)
                    eta = action[0]
                    eta2List.append(eta)
                for k in range(apf.numberOfCone):
                    action = [random.uniform(act_bound[0], act_bound[1])]
                    action3List.append(action)
                    eta = action[0]
                    eta3List.append(eta)
            # 与环境交互
            qNext = apf.getqNext(apf.epsilon0, eta1List, eta2List, eta3List, q, qBefore)

            obsDicqNext = apf.calculateDynamicState(qNext)
            obs_sphere_next, obs_cylinder_next, obs_cone_next = obsDicqNext['sphere'], obsDicqNext['cylinder'], \
                                                                obsDicqNext['cone']

            flag = apf.checkCollision(qNext)
            reward = getReward(flag, apf, qBefore, q, qNext)
            rewardSum += reward

            done = True if apf.distanceCost(apf.qgoal, qNext) < apf.threshold else False
            ddpgReplayBufferStore(apf, qNext, ddpgNetForSphere, ddpgNetForCylinder, ddpgNetForCone,
                                  action1List, action2List, action3List, obs_sphere, obs_sphere_next, obs_cylinder,
                                  obs_cylinder_next, obs_cone, obs_cone_next, reward, done)  # ddpg网络存储样本

            if episode >= 10 and j % update_every == 0:
                for _ in range(update_every):
                    for k in range(apf.numberOfSphere):
                        if ddpgNetForSphere[k].replay_buffer.size >= batch_size:
                            batch = ddpgNetForSphere[k].replay_buffer.sample_batch(batch_size)
                            ddpgNetForSphere[k].update(data=batch)
                    for k in range(apf.numberOfCylinder):
                        if ddpgNetForCylinder[k].replay_buffer.size >= batch_size:
                            batch = ddpgNetForCylinder[k].replay_buffer.sample_batch(batch_size)
                            ddpgNetForCylinder[k].update(data=batch)
                    for k in range(apf.numberOfCone):
                        if ddpgNetForCone[k].replay_buffer.size >= batch_size:
                            batch = ddpgNetForCone[k].replay_buffer.sample_batch(batch_size)
                            ddpgNetForCone[k].update(data=batch)
            if done:
                break
            qBefore = q
            q = qNext


        print('Episode:', episode, 'Reward:%f' % rewardSum, 'noise:%f' % noise)
        rewardList.append(round(rewardSum,2))
        if episode > MAX_EPISODE*2/3:
            if rewardSum > maxReward:
                maxReward = rewardSum
                for i in range(apf.numberOfSphere):
                    torch.save(ddpgNetForSphere[i].ac.pi, 'TrainedModel/Actor1.%d.pkl'%i)
                for i in range(apf.numberOfCylinder):
                    torch.save(ddpgNetForCylinder[i].ac.pi, 'TrainedModel/Actor2.%d.pkl' % i)
                for i in range(apf.numberOfCone):
                    torch.save(ddpgNetForCone[i].ac.pi, 'TrainedModel/Actor3.%d.pkl' % i)

    # 绘制
    painter = Painter(load_csv=True, load_dir='F:/MasterDegree/毕业设计/实验数据/figure_data_5.csv')
    painter.addData(rewardList, 'Fully Decentralized DDPG')
    painter.saveData('F:/MasterDegree/毕业设计/实验数据/figure_data_5.csv')
    painter.drawFigure()





