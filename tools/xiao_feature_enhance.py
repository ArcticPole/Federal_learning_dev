# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 09:25:00 2022

@author: Xiao Peng
"""

import numpy as np
import torch.nn as nn
import torch
def f_e(weight):

    for i in range(0, len(weight)):  # len=4
        for j in range(0, len(weight[i])):  # i=0,len=32
            for k in range(0, len(weight[i][j])):  # i=0,j=0,len=1
                weight_max = max(max((weight[i][j][k]).detach().numpy().tolist()))
                weight_min = min(min((weight[i][j][k]).detach().numpy().tolist()))
                ran = weight_max - weight_min
                weight_med = weight_min + ran * 0.5
                critical_max = weight_med + ran * 0.4
                critical_min = weight_med - ran * 0.4
                for l in range(0, len(weight[i][j][k])):  # i=0,j=0,l=0,len=5
                    for m in range(0, len(weight[i][j][k][l])):
                        # print(weight[i][j][k][l][m])
                        # print(weight[i][j][k][l][m].detach().numpy()<critical_max)
                        # print(weight[i][j][k][l][m].detach().numpy()>critical_min)
                        if critical_max > weight[i][j][k][l][m].detach().numpy() > critical_min:
                            weight[i][j][k][l][m].data *= 0
                            weight[i][j][k][l][m].data += 1
                            weight[i][j][k][l][m].data *= weight_med
    print("feature enhanced")
    return weight
