import os
import math
import numpy as np
import time
import random
import shutil
import cv2
from PIL import Image

import sys

sys.setrecursionlimit(20000)  #


# 匈牙利匹配方法原则是先到先得，能让则让
# Hungarian method for bipartite graph
def hungarian(matrixTF):
    # matrix to adjacent matrix
    edges = np.argwhere(matrixTF)  # 取出满足distance threshold条件的index，第0维度(行)表示pred point的id，第1维度(列)表示gt point的id。即边的两端
    lnum, rnum = matrixTF.shape
    graph = [[] for _ in range(lnum)]
    for edge in edges:
        # graph的index表示的是pred point的id，value表示的是gt point的id。这里的对应关系条件是满足distance threshold，因此也可能出现一对多的情况
        graph[edge[0]].append(edge[1])

    # deep first search
    match = [-1 for _ in range(rnum)]  # 记录gt point所对应的pred point
    vis = [-1 for _ in range(rnum)]  # 记录gt point是否被访问过

    def dfs(u):
        # u代表pred point的id，v代表gt point的id
        for v in graph[u]:
            if vis[v]:
                continue
            vis[v] = True  # 记录状态为访问过
            # 如果gt point暂无匹配，或者原来与该gt point匹配的pred point可以找到新的匹配
            # 如果之前的pred point能分配得到新的匹配，则此dfs(match[v])内部完成新的匹配并返回ture，这个递归技巧是关键
            if match[v] == -1 or dfs(match[v]):
                match[v] = u
                return True  # 返回匹配成功
        return False  # 循环结束，仍未找到匹配，返回匹配失败

    # for loop
    ans = 0
    for a in range(lnum):
        for i in range(rnum):
            vis[i] = False  # 重置vis
        if dfs(a):
            ans += 1

    # assignment matrix
    assign = np.zeros((lnum, rnum), dtype=bool)
    for i, m in enumerate(match):
        if m >= 0:
            assign[m, i] = True  # m是pred point的id，i是gt point的id， True表示两者match

    return ans, assign


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cur_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val
        self.count += 1
        self.avg = self.sum / self.count


class DistMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cur_val = []
        self.dist_set = []
        self.avg = 0  # 均值
        self.std = 0  # 标准差

    def update1(self, cur_val):
        self.cur_val = cur_val
        self.dist_set.extend(cur_val)
        if len(self.dist_set) == 0: # 防止训练开始阶段由于没有tp, numpy出现invalid value encountered in divide警告
            self.avg = 0
            self.std = 0
        else:
            self.avg = np.mean(self.dist_set)
            self.std = np.std(self.dist_set)


class AverageCategoryMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, num_class):
        self.num_class = num_class
        self.reset()

    def reset(self):
        self.cur_val = np.zeros(self.num_class)
        self.sum = np.zeros(self.num_class)

    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val
