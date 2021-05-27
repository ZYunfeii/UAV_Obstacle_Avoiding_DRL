#!/usr/bin/python
# -*- coding: utf-8 -*-
# Time: 2021-3-29
# Author: ZYunfei
# File func: various dynamic obstacle environment
import numpy as np
"""提供多个单障碍物动态环境用于训练UAV"""
def obstacle1(time_now, time_step):
    obs_ref = np.array([5,5,5],dtype=float)
    dic = {}
    obsCenter = np.array([obs_ref[0] + 2 * np.cos(time_now),
                          obs_ref[1] + 2 * np.sin(time_now),
                          obs_ref[2]], dtype=float)
    vObs = np.array([-2 * np.sin(time_now),
                          2 * np.cos(time_now), 0])

    time_now += time_step

    obsCenterNext = np.array([obs_ref[0] + 2 * np.cos(time_now),
                              obs_ref[1] + 2 * np.sin(time_now),
                              obs_ref[2]], dtype=float)
    vObsNext = np.array([-2 * np.sin(time_now), 2 * np.cos(time_now), 0])
    dic['v'] = vObs
    dic['obsCenter'] = obsCenter
    dic['vNext'] = vObsNext
    dic['obsCenterNext'] = obsCenterNext

    return time_now, dic

def obstacle2(time_now, time_step):
    obs_ref = np.array([9,9,5.5],dtype=float)
    dic = {}
    obsCenter = np.array([obs_ref[0] -  0.5*time_now,
                          obs_ref[1] -  0.5*time_now,
                          obs_ref[2] + np.sin(2*time_now)], dtype=float)
    vObs = np.array([-0.5,-0.5, 2*np.cos(2*time_now)])
    time_now += time_step
    obsCenterNext = np.array([obs_ref[0] -  0.5*time_now,
                          obs_ref[1] -  0.5*time_now,
                          obs_ref[2] + np.sin(2*time_now)], dtype=float)
    vObsNext = np.array([-0.5,-0.5, 2*np.cos(2*time_now)])

    dic['v'] = vObs
    dic['obsCenter'] = obsCenter
    dic['vNext'] = vObsNext
    dic['obsCenterNext'] = obsCenterNext

    return time_now, dic

def obstacle3(time_now, time_step):
    obs_ref = np.array([5,10,5.5],dtype=float)
    dic = {}
    obsCenter = np.array([obs_ref[0],
                          obs_ref[1] -  time_now,
                          obs_ref[2] + np.sin(2*time_now)], dtype=float)
    vObs = np.array([0,-1, 2*np.cos(2*time_now)])
    time_now += time_step
    obsCenterNext = np.array([obs_ref[0],
                          obs_ref[1] - time_now,
                          obs_ref[2] + np.sin(2 * time_now)], dtype=float)
    vObsNext = np.array([0, -1, 2 * np.cos(2 * time_now)])

    dic['v'] = vObs
    dic['obsCenter'] = obsCenter
    dic['vNext'] = vObsNext
    dic['obsCenterNext'] = obsCenterNext

    return time_now, dic

def obstacle4(time_now, time_step):
    obs_ref = np.array([6,6,5],dtype=float)
    dic = {}
    obsCenter = np.array([obs_ref[0]+3*np.cos(0.5*time_now),
                          obs_ref[1] +  3*np.sin(0.5*time_now),
                          obs_ref[2] + np.sin(2*time_now)], dtype=float)
    vObs = np.array([-1.5*np.sin(0.5*time_now),1.5*np.cos(0.5*time_now), 2*np.cos(2*time_now)])
    time_now += time_step
    obsCenterNext = np.array([obs_ref[0]+3*np.cos(0.5*time_now),
                          obs_ref[1] +  3*np.sin(0.5*time_now),
                          obs_ref[2] + np.sin(2*time_now)], dtype=float)
    vObsNext = np.array([-1.5*np.sin(0.5*time_now),1.5*np.cos(0.5*time_now), 2*np.cos(2*time_now)])

    dic['v'] = vObs
    dic['obsCenter'] = obsCenter
    dic['vNext'] = vObsNext
    dic['obsCenterNext'] = obsCenterNext

    return time_now, dic

def obstacle5(time_now, time_step):
    obs_ref = np.array([5,8,5],dtype=float)
    dic = {}
    obsCenter = np.array([obs_ref[0]+3*np.sin(0.5*time_now),
                          obs_ref[1] +  3*np.cos(0.5*time_now),
                          obs_ref[2] + np.sin(0.5*time_now)], dtype=float)
    vObs = np.array([1.5*np.cos(0.5*time_now),-1.5*np.sin(0.5*time_now), 0.5*np.cos(0.5*time_now)])
    time_now += time_step
    obsCenterNext = np.array([obs_ref[0]+3*np.sin(0.5*time_now),
                          obs_ref[1] +  3*np.cos(0.5*time_now),
                          obs_ref[2] + np.sin(0.5*time_now)], dtype=float)
    vObsNext = np.array([1.5*np.cos(0.5*time_now),-1.5*np.sin(0.5*time_now), 0.5*np.cos(0.5*time_now)])

    dic['v'] = vObs
    dic['obsCenter'] = obsCenter
    dic['vNext'] = vObsNext
    dic['obsCenterNext'] = obsCenterNext

    return time_now, dic

def obstacle6(time_now, time_step):
    obs_ref = np.array([5,6,5],dtype=float)
    dic = {}
    obsCenter = np.array([obs_ref[0]+3*np.sin(0.5*time_now),
                          obs_ref[1] +  np.cos(0.5*time_now),
                          obs_ref[2] + np.sin(0.5*time_now)], dtype=float)
    vObs = np.array([1.5*np.cos(0.5*time_now),-0.5*np.sin(0.5*time_now), 0.5*np.cos(0.5*time_now)])
    time_now += time_step
    obsCenterNext = np.array([obs_ref[0]+3*np.sin(0.5*time_now),
                          obs_ref[1] +  np.cos(0.5*time_now),
                          obs_ref[2] + np.sin(0.5*time_now)], dtype=float)
    vObsNext = np.array([1.5*np.cos(0.5*time_now),-0.5*np.sin(0.5*time_now), 0.5*np.cos(0.5*time_now)])

    dic['v'] = vObs
    dic['obsCenter'] = obsCenter
    dic['vNext'] = vObsNext
    dic['obsCenterNext'] = obsCenterNext

    return time_now, dic

def obstacle7(time_now, time_step):
    obs_ref = np.array([10,10,5],dtype=float)
    dic = {}
    obsCenter = np.array([obs_ref[0] - time_now,
                          obs_ref[1] - time_now,
                          obs_ref[2] ], dtype=float)
    vObs = np.array([-1, -1, 0])
    time_now += time_step
    obsCenterNext = np.array([obs_ref[0] - time_now,
                          obs_ref[1] - time_now,
                          obs_ref[2] ], dtype=float)
    vObsNext = np.array([-1, -1, 0])

    dic['v'] = vObs
    dic['obsCenter'] = obsCenter
    dic['vNext'] = vObsNext
    dic['obsCenterNext'] = obsCenterNext

    return time_now, dic

def obstacle8(time_now, time_step):
    obs_ref = np.array([3,10,5],dtype=float)
    dic = {}
    time_thre = 8
    if time_now < time_thre:
        obsCenter = np.array([obs_ref[0] + 5*np.sin(np.pi/2+0.3*time_now),
                              obs_ref[1] + 5*np.cos(np.pi/2+0.3*time_now),
                              obs_ref[2]], dtype=float)
        vObs = np.array([1.5*np.cos(np.pi/2+0.3*time_now), -1.5*np.sin(np.pi/2+0.3*time_now), 0])
        time_now += time_step
        obsCenterNext = np.array([obs_ref[0] + 5*np.sin(np.pi/2+0.3*time_now),
                              obs_ref[1] + 5*np.cos(np.pi/2+0.3*time_now),
                              obs_ref[2] ], dtype=float)
        vObsNext = np.array([1.5*np.cos(np.pi/2+0.3*time_now), -1.5*np.sin(np.pi/2+0.3*time_now), 0])
    else:
        delta_time = time_now - time_thre
        time_cal = time_thre - delta_time
        obsCenter = np.array([obs_ref[0] + 5*np.sin(np.pi/2+0.3*time_cal),
                              obs_ref[1] + 5*np.cos(np.pi/2+0.3*time_cal),
                              obs_ref[2]], dtype=float)
        vObs = np.array([1.5*np.cos(np.pi/2+0.3*time_cal), -1.5*np.sin(np.pi/2+0.3*time_cal), 0])
        time_now += time_step
        time_cal -= time_step
        obsCenterNext = np.array([obs_ref[0] + 5*np.sin(np.pi/2+0.3*time_cal),
                              obs_ref[1] + 5*np.cos(np.pi/2+0.3*time_cal),
                              obs_ref[2] ], dtype=float)
        vObsNext = np.array([1.5*np.cos(np.pi/2+0.3*time_cal), -1.5*np.sin(np.pi/2+0.3*time_cal), 0])


    dic['v'] = vObs
    dic['obsCenter'] = obsCenter
    dic['vNext'] = vObsNext
    dic['obsCenterNext'] = obsCenterNext

    return time_now, dic

"""生成一个函数列表"""
obs_list = [obstacle1, obstacle2, obstacle3, obstacle4,
            obstacle5, obstacle6, obstacle7, obstacle8]


