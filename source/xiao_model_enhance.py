# -*- coding: utf-8 -*-
"""
Created on Sat April 30 09:10:00 2022

@author: Xiao Peng
"""
from torch import nn
import torch
import matplotlib.pyplot as plt  # plt 用于显示图片
import numpy as np
import tools.xiao_feature_enhance as xiao_feature_enhance
import tools.xiao_global_feature as xgf
import tools.model as model

net_list=['../global_models/source_models/net_xiao_insert_fault_all_ready.pkl',
          '../global_models/source_models/net_xiao_insert_fault_foreign_body.pkl',
          '../global_models/source_models/net_xiao_insert_fault_incline.pkl',
          '../global_models/source_models/net_xiao_insert_fault_no_base.pkl',
          '../global_models/source_models/net_xiao_L_fault_all_ready.pkl',
          '../global_models/source_models/net_xiao_L_fault_foreign_body.pkl',
          '../global_models/source_models/net_xiao_L_fault_incline.pkl',
          '../global_models/source_models/net_xiao_L_fault_no_base.pkl']

nl = len(net_list)
all_weights = xgf.catch_feature(net_list, nl)
exp = ["insert_fault","L_fault"]
kind = ['all_ready', 'foreign_body', 'incline', 'no_base']
# feature enhanced

for i in range(nl):
    # print(all_weights[i][0][0][0])
    all_weights[i] = xiao_feature_enhance.f_e(all_weights[i])

for i in range(nl):
    net = model.cnn2d_enhanced_model(all_weights[i])
    # state = {'model': net.state_dict(), 'weight': weight}
    # print(i)
    # print(all_weights[i][0][0][0])
    if i < nl/2:
        m=0
    else:
        m=1
    PATH = 'enhanced_models/net_enhanced_%s_%s.pkl' % (exp[m], kind[i-4*m])
    torch.save(net, PATH)  # save model

exit(0)