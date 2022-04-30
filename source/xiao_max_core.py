# -*- coding: utf-8 -*-
"""
Created on Sat April 30 11:50:00 2022

@author: Xiao Peng
"""
# from torch import nn
import torch
# import matplotlib.pyplot as plt  # plt 用于显示图片
import numpy as np
# import tools.xiao_feature_enhance as xiao_feature_enhance
# import tools.xiao_global_feature as xgf
# import tools.model as model
import xiao_feature_sorted as xfs

def max_feature(net_list):
    # net_list=['enhanced_models/net_enhanced_insert_fault_all_ready.pkl',
    #           'enhanced_models/net_enhanced_insert_fault_foreign_body.pkl',
    #           'enhanced_models/net_enhanced_insert_fault_incline.pkl',
    #           'enhanced_models/net_enhanced_insert_fault_no_base.pkl',
    #           'enhanced_models/net_enhanced_L_fault_all_ready.pkl',
    #           'enhanced_models/net_enhanced_L_fault_foreign_body.pkl',
    #           'enhanced_models/net_enhanced_L_fault_incline.pkl',
    #           'enhanced_models/net_enhanced_L_fault_no_base.pkl']

    nl = len(net_list)
    all_max=[]

    for i in range(nl):
        max = []
        # print(i)
        net=torch.load(net_list[i])
        # weight=[net.weight1,net.weight2,net.weight3,net.weight4]
        weight = net.weight1
        for k in range(len(weight)):
            max_temp = 0
            temp = 0
            for i in range(0, len(weight[k])):  # len=4
                for j in range(0, len(weight[k][i])):  # i=0,len=32
                    # print(weight[k][i][j][m])
                    am = np.argmax(weight[k][i][j].detach().numpy())
                    bm = np.argmin(weight[k][i][j].detach().numpy())
                    max_temp=weight[k][i][j][am].detach().numpy()
                    # print(max_temp)
                    if max_temp>temp:
                        temp=max_temp
            # print(temp)
            max.append(temp)
        max=xfs.feature_sorted(max)
        all_max.append(max)
    return all_max
