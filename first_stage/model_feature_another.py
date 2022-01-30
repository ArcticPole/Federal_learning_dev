# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:41:00 2022

@author: Xiao Peng
"""

import numpy as np
import torch

net0 = torch.load('trained_model/net_xiao_insert_fault_no_base.pkl')

import torch.nn as nn
from matplotlib import pyplot as plt


#print(net0)
conv_layers = []
model_weights = []
model_children = list(net0.children())
counter = 0

for i in range(len(model_children)):
    #print(i)
    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            if type(model_children[i][j])==nn.Conv2d:
                counter += 1
                model_weights.append(model_children[i][j].weight)
                conv_layers.append(model_children[i][j])

#print(counter)
# 画出卷积核的灰度图
# for i in range(4):
#     # print(model_weights[i])
#     print(model_weights[i].shape)
#     plt.subplot(2, 2, i+1)
#     plt.axis('off')
#     plt.imshow(model_weights[0][i][0, :, :].detach(), cmap='gray')
#plt.show()

def weight():
    return model_weights
