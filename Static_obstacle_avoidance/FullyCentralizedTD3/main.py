from TD3Model import AgentTD3, ReplayBuffer
from Static_obstacle_avoidance.ApfAlgorithm import *
from Static_obstacle_avoidance.Method import getReward, setup_seed, transformAction, Arguments
import random
from Static_obstacle_avoidance.draw import *
import torch

if __name__ == '__main__':
    setup_seed(1)   # 设置随机数种子

    apf = APF()
    args = Arguments(apf)
    obs_dim = args.obs_dim
    act_dim = args.act_dim
    act_bound = args.act_bound

    centralizedContriller = AgentTD3()
    centralizedContriller.init(256,obs_dim,act_dim)
    buffer = ReplayBuffer(int(1e6), obs_dim, act_dim, False, True)
    gamma = 0.99
    MAX_EPISODE = 100
    MAX_STEP = 500
    batch_size = 512
    update_cnt = 0
    rewardList = []
    maxReward = -np.inf
    update_every = 50
    for episode in range(MAX_EPISODE):
        q = apf.x0
        apf.reset()
        rewardSum = 0
        qBefore = [None, None, None]
        for j in range(MAX_STEP):
            obsDicq = apf.calculateDynamicState(q)
            obs_sphere, obs_cylinder, obs_cone = obsDicq['sphere'], obsDicq['cylinder'], obsDicq['cone']
            obs_mix = obs_sphere + obs_cylinder + obs_cone
            obs = np.array([]) # 中心控制器接受所有状态集合
            for k in range(len(obs_mix)):
                obs = np.hstack((obs, obs_mix[k])) # 拼接状态为一个1*n向量
            if episode > 50:
                action = centralizedContriller.select_action(obs)
                action = transformAction(action,act_bound,act_dim)
                # 分解动作向量
                action_sphere = action[0:apf.numberOfSphere]
                action_cylinder = action[apf.numberOfSphere:apf.numberOfSphere + apf.numberOfCylinder]
                action_cone = action[apf.numberOfSphere + apf.numberOfCylinder:apf.numberOfSphere +\
                                          apf.numberOfCylinder + apf.numberOfCone]
            else:
                action_sphere = [random.uniform(act_bound[0],act_bound[1]) for k in range(apf.numberOfSphere)]
                action_cylinder = [random.uniform(act_bound[0],act_bound[1]) for k in range(apf.numberOfCylinder)]
                action_cone = [random.uniform(act_bound[0],act_bound[1]) for k in range(apf.numberOfCone)]
                action = action_sphere + action_cylinder + action_cone

            # 与环境交互
            qNext = apf.getqNext(apf.epsilon0, action_sphere, action_cylinder, action_cone, q, qBefore)
            obsDicqNext = apf.calculateDynamicState(qNext)
            obs_sphere_next, obs_cylinder_next, obs_cone_next = obsDicqNext['sphere'], obsDicqNext['cylinder'], obsDicqNext['cone']
            obs_mix_next = obs_sphere_next + obs_cylinder_next + obs_cone_next
            obs_next = np.array([])
            for k in range(len(obs_mix_next)):
                obs_next = np.hstack((obs_next, obs_mix_next[k]))
            flag = apf.checkCollision(qNext)
            reward = getReward(flag, apf, qBefore, q, qNext)
            rewardSum += reward

            done = True if apf.distanceCost(apf.qgoal, qNext) < apf.threshold else False
            mask = 0.0 if done else gamma
            other = (reward,mask,*action)
            buffer.append_buffer(obs,other)

            if episode >= 50 and j % update_every == 0:
                centralizedContriller.update_net(buffer,update_every,batch_size,1)
                update_cnt += update_every
            if done: break
            qBefore = q
            q = qNext


        print('Episode:', episode, 'Reward:%f' % rewardSum, 'update_cnt:%d' %update_cnt)
        rewardList.append(round(rewardSum,2))
        if episode > MAX_EPISODE*2/3:
            if rewardSum > maxReward:
                print('reward大于历史最优，已保存模型！')
                maxReward = rewardSum
                torch.save(centralizedContriller.act, 'TrainedModel/centralizedActor.pkl')

    # 绘制
    painter = Painter(load_csv=False,load_dir='./data_csv/figure_data.csv')
    painter.addData(rewardList, 'TD3',smooth=True)
    painter.saveData('./data_csv/figure_data.csv')
    painter.drawFigure()





