#!/usr/bin/python
# -*- coding: utf-8 -*-
# Time: 2021-1-15
# Author: ZYunfei
# Name: MADDPG-APF
# File func: arg func

import argparse
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser("reinforcement learning experiments for multiagent environments in apf")

    parser.add_argument("--device", default=device, help="torch device")
    # parser.add_argument("--max_grad_norm", type=float, default=0.5, help="max gradient norm for clip")
    parser.add_argument("--num_units_actor", type=int, default=128, help="number of units in the mlp")
    parser.add_argument("--num_units_critic", type=int, default=128, help="number of units in the mlp")
    parser.add_argument("--tao", type=int, default=0.01, help="how depth we exchange the par of the nn")
    parser.add_argument("--lr_a", type=float, default=1e-3, help="learning rate for adam optimizer")
    parser.add_argument("--lr_c", type=float, default=1e-3, help="learning rate for adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--batch_size", type=int, default=1256, help="number of episodes to optimize at the same time")
    parser.add_argument("--memory_size", type=int, default=1e6, help="number of data stored in the memory")

    parser.add_argument("--learning_start_step", type=int, default=10000, help="learning start steps")
    parser.add_argument("--actor_begin_work", type=int, default=60, help="after this episode, the actor begin to work")
    parser.add_argument("--learning_fre", type=int, default=5, help="learning frequency")
    parser.add_argument("--max_episode", type=int, default=1000, help="maximum episode length")
    parser.add_argument("--per_episode_max_len", type=int, default=500, help="maximum episode length")
    parser.add_argument("--action_limit_min", type=float, default=0.1, help="the minimum action value")
    parser.add_argument("--action_limit_max", type=float, default=3.0, help="the maximum action value")

    return parser.parse_args()




