# README

This is a project about deep reinforcement learning autonomous obstacle avoidance algorithm for **UAV**. The whole project includes obstacle avoidance in **static environment** and obstacle avoidance in **dynamic environment**. In the static environment, **Multi-Agent Reinforcement Learning** and **artificial potential field algorithm** are combined. In the dynamic environment, the project adopts the combination of **disturbed flow field algorithm** and **single agent reinforcement learning algorithm**.

## Static environment

There are four methods to solve:

1. **MADDPG**
2. **Fully Centralized DDPG**
3. **Fully Decentralized DDPG**
4. **Fully Centralized TD3**

The third and the fourth methods perform better than others.

## Dynamic environment

There are four methods to solve:

1. **PPO+GAE(with multi-processing )**
2. **TD3**
3. **DDPG**
4. **SAC**

The first three methods perform just the same. PPO convergence needs less episodes. TD3 and DDPG converge fast. Though Soft Actor-Critic is an outstanding algorithm in DRL, it has no obvious effect in my environment.

## Traditional methods for  UAV path planning

Three traditional methods are written with **MATLAB**:

1. **A * search algorithm***
2. **RRT algorithm**
3. **Ant colony algorithm**

**C++:**

1. **D star algorithm**

The experiments show that A* search algorithm is much better than others but it is less effective than reinforcement learning path planning.

## Artificial potential field algorithm

This project provides the MATLAB and Python realization of artificial potential field algorithm.

**Python realization**: ./APF/APFPy2.py      ./APF/APFPy3.py    ./APF/ApfAlgorithm.py  (two-dimensional and three-dimensional)

**Matlab realization**: ./APF/APF_matlab (two-dimensional)

## IFDS and IIFDS algorithm

This is an obstacle avoidance planning algorithm based on flow field. I realize it with matlab. The code is in folder **IIFDS_and_IFDS**.

## How to begin trainning

For example, you want to train the agent in dynamic environment with TD3, what you need to do is just running the **main.py**, then **test.py**, finally open matlab and run the **test.m** to draw.

If you want to test the model in the environment with 4 obstacles, you just need to run  **Multi_obstacle_environment_test.py**.

## Requirements

numpy

torch

matplotlib

seaborn==0.11.1

## Files to illustrate

**calGs.m**: calculate the index Gs which shows the performance of the route.

**calLs.m**: calculate the index Ls which shows the performance of the route.

**draw.py**: this file includes the Painter class which can draw the reward curve of various methods.

**config.py**: this file give the setting of the parameters in trainning process of the algorithm such as the MAX_EPISODE, batch_size and so on.

**Method.py**: this file concludes many important methods such as how to calculate the reward of the agents.

**static_obstacle_environment.py**: there are many static obstacle environments' parameters in this file.

**dynamic_obstacle_environment.py**: there are many dynamic obstacle environments' parameters in this file.

**Multi_obstacle_environment_test.py**: this file test the dynamic model in the environment in dynamic_obstacle_environment.py.

**data_csv**: this file save some data such as the trace of UAV and the reward in trainning.

**AntColonybenchmark.m**: ACO algorithm realized by MATLAB.

**Astarbenchmark.m**: A\* algorithm realized by MATLAB.

**RRTbenchmark.m**: RRT algorithm realized by MATLAB.

# A simple simulation example

- ![avatar](/Dynamic_obstacle_avoidance/GIF/compare_aifds.gif)



*all rights reserved.*

