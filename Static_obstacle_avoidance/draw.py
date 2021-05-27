#!/usr/bin/python
# -*- coding: utf-8 -*-
# Time: 2021-3-29
# Author: ZYunfei
# File func: draw func

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import os

class Painter:
    def __init__(self, load_csv, load_dir=None):
        if not load_csv:
            self.data = pd.DataFrame(columns=['episode reward','episode', 'Method'])
        else:
            self.load_dir = load_dir
            if os.path.exists(self.load_dir):
                print("==正在读取{}。".format(self.load_dir))
                self.data = pd.read_csv(self.load_dir).iloc[:,1:] # csv文件第一列是index，不用取。
                print("==读取完毕。")
            else:
                print("==不存在{}下的文件，Painter已经自动创建该csv。".format(self.load_dir))
                self.data = pd.DataFrame(columns=['episode reward', 'episode', 'Method'])
        self.xlabel = None
        self.ylabel = None
        self.title = None
        self.hue_order = None

    def setXlabel(self,label): self.xlabel = label

    def setYlabel(self, label): self.ylabel = label

    def setTitle(self, label): self.title = label

    def setHueOrder(self,order):
        """设置成['name1','name2'...]形式"""
        self.hue_order = order

    def addData(self, dataSeries, method, x=None, smooth = True):
        if smooth:
            dataSeries = self.smooth(dataSeries)
        size = len(dataSeries)
        if x is not None:
            if len(x) != size:
                print("请输入相同维度的x!")
                return
        for i in range(size):
            if x is not None:
                dataToAppend = {'episode reward':dataSeries[i],'episode':x[i],'Method':method}
            else:
                dataToAppend = {'episode reward':dataSeries[i],'episode':i+1,'Method':method}
            self.data = self.data.append(dataToAppend,ignore_index = True)

    def drawFigure(self,style="darkgrid"):
        """
        style: darkgrid, whitegrid, dark, white, ticks
        """
        sns.set_theme(style=style)
        sns.set_style(rc={"linewidth": 1})
        print("==正在绘图...")
        sns.relplot(data = self.data, kind = "line", x = "episode", y = "episode reward",
                    hue= "Method", hue_order=None)
        plt.title(self.title,fontsize = 12)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        print("==绘图完毕！")
        plt.show()

    def saveData(self, save_dir):
        self.data.to_csv(save_dir)
        print("==已将数据保存到路径{}下!".format(save_dir))

    def addCsv(self, add_load_dir):
        """将另一个csv文件合并到load_dir的csv文件里。"""
        add_csv = pd.read_csv(add_load_dir).iloc[:,1:]
        self.data = pd.concat([self.data, add_csv],axis=0,ignore_index=True)

    def deleteData(self,delete_data_name):
        """删除某个method的数据，删除之后需要手动保存，不会自动保存。"""
        self.data = self.data[~self.data['Method'].isin([delete_data_name])]
        print("==已删除{}下对应数据!".format(delete_data_name))

    def smoothData(self, smooth_method_name,N):
        """对某个方法下的reward进行MA滤波，N为MA滤波阶数。"""
        begin_index = -1
        mode = -1  # mode为-1表示还没搜索到初始索引， mode为1表示正在搜索末尾索引。
        for i in range(len(self.data)):
            if self.data.iloc[i]['Method'] == smooth_method_name and mode == -1:
                begin_index = i
                mode = 1
                continue
            if mode == 1 and self.data.iloc[i]['episode'] == 1:
                self.data.iloc[begin_index:i,0] = self.smooth(
                    self.data.iloc[begin_index:i,0],N = N
                )
                begin_index = -1
                mode = -1
                if self.data.iloc[i]['Method'] == smooth_method_name:
                    begin_index = i
                    mode = 1
            if mode == 1 and i == len(self.data) - 1:
                self.data.iloc[begin_index:,0]= self.smooth(
                    self.data.iloc[begin_index:,0], N=N
                )
        print("==对{}数据{}次平滑完成!".format(smooth_method_name,N))

    @staticmethod
    def smooth(data,N=5):
        n = (N - 1) // 2
        res = np.zeros(len(data))
        for i in range(len(data)):
            if i <= n - 1:
                res[i] = sum(data[0:2 * i+1]) / (2 * i + 1)
            elif i < len(data) - n:
                res[i] = sum(data[i - n:i + n +1]) / (2 * n + 1)
            else:
                temp = len(data) - i
                res[i] = sum(data[-temp * 2 + 1:]) / (2 * temp - 1)
        return res



if __name__ == "__main__":
    painter = Painter(load_csv=True, load_dir='./figure1.csv')
    painter.drawFigure(style="whitegrid")
