#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as rd
import torch
import torch.nn as nn
from collections import namedtuple

def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)  # Tensor正交初始化
    torch.nn.init.constant_(layer.bias, bias_const) # 偏置常数初始化
class BufferTupleOnline:
    def __init__(self, max_memo):
        self.max_memo = max_memo
        self.storage_list = list()
        self.transition = namedtuple(
            'Transition',
            # ('state', 'value', 'action', 'log_prob', 'mask', 'next_state', 'reward')
            ('reward', 'mask', 'state', 'action', 'log_prob')
        )

    def push(self, *args):
        self.storage_list.append(self.transition(*args))

    def extend_memo(self, storage_list):
        self.storage_list.extend(storage_list)

    def sample_all(self):
        return self.transition(*zip(*self.storage_list))

    def __len__(self):
        return len(self.storage_list)

    def update_pointer_before_sample(self):
        pass  # compatibility
class ActorPPO(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()

        def idx_dim(i):
            return int(8 * 1.5 ** i)

        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, action_dim),
            )

        self.a_std_log = nn.Parameter(torch.zeros(1, action_dim) - 0.5, requires_grad=True) # 这个变量不仅是带梯度的，而且属于模型parameters的一部分
        self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))

        # layer_norm(self.net__mean[0], std=1.0)
        # layer_norm(self.net__mean[2], std=1.0)
        layer_norm(self.net[-1], std=0.01)  # output layer for action

    def forward(self, s):
        a_mean = self.net(s)
        return a_mean.tanh()

    def get__a__log_prob(self, state):
        a_mean = self.net(state)
        a_std = self.a_std_log.exp()
        a_noise = torch.normal(a_mean, a_std)

        # a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        a_delta = ((a_noise - a_mean) / a_std).pow(2) / 2
        log_prob = -(a_delta + (self.a_std_log + self.constant_log_sqrt_2pi))
        log_prob = log_prob.sum(1)
        return a_noise, log_prob

    def compute__log_prob(self, state, a_noise):
        a_mean = self.net(state)
        a_std = self.a_std_log.exp()

        a_delta = ((a_noise - a_mean) / a_std).pow(2) / 2
        log_prob = -(a_delta + (self.a_std_log + self.constant_log_sqrt_2pi))
        return log_prob.sum(1)
class CriticAdv(nn.Module):  # 2020-05-05 fix bug
    def __init__(self, state_dim, mid_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, 1),
            )

        # layer_norm(self.net[0], std=1.0)
        # layer_norm(self.net[2], std=1.0)
        layer_norm(self.net[-1], std=1.0)  # output layer for action

    def forward(self, s):
        q = self.net(s)
        return q