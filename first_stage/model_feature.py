# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 09:49:00 2021

@author: Xiao Peng
"""

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
def weight():
    net0 = torch.load('../trained_model/net_xiao_insert_fault_incline.pkl')
    conv_layers = []
    model_weights = []
    model_children = list(net0.children())
    counter = 0

    for i in range(len(model_children)):
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
    return model_weights
