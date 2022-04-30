# -*- coding: utf-8 -*-
"""
Created on Sat April 30 09:50:00 2022

@author: Xiao Peng
"""
from torch import nn
import torch
import matplotlib.pyplot as plt  # plt 用于显示图片
import numpy as np
import tools.xiao_feature_enhance as xiao_feature_enhance
import tools.xiao_global_feature as xgf
import tools.model as model

def feature_sorted(max):
    max=np.array(max)
    sort=np.zeros(len(max))
    for i in range(len(max)):
        m=np.argmax(max)
        sort[m]=i
        max[m]=-100000
    print(sort)
    return sort

