#!/usr/bin/python
# -*- coding: utf-8 -*-
# Time: 2021-1-15
# Author: ZYunfei
# Name: MADDPG-APF
# File func: model(actor and critic) func
"""
actor 和 critic网络架构来源于openAI开源代码spinningup强化学习算法库，高性能。
"""
import torch
import torch.nn as nn

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, act_limit, hidden_sizes=(256,256), activation=nn.ReLU):
        super().__init__()

        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return (self.pi(obs) + 1) / 2 * (self.act_limit[1] - self.act_limit[0]) + self.act_limit[0]


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes=(512,512), activation=nn.ReLU):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.