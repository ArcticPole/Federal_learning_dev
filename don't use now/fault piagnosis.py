# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 13:58:56 2019

@author: wangc
"""

import numpy as np
import torch

import matplotlib.pyplot as plt # plt 用于显示图片

## load the dataset 
import dataset
import xiaodataset

cifar = xiaodataset.FlameSet('gear_fault', 2304, '2D', 'incline',0)
#cifar = dataset.FlameSet('gear_fault', 2304, '2D', 'incline')  # 实例化数据集，加载数据集

trainloader,validloader = dataset.loaddata(cifar)  # 加载训练数据和测试数据 #xiao：在dataset.py里面

#print (len(trainloader)*50)
#print (len(validloader)*50)

## Define model
import model
net_classifier = model.CNN2d_classifier()  #选择神经网络模型
net_fitting_ball = model.CNN2d_fitting()
net_fitting_inner = model.CNN2d_fitting()
net_fitting_outer = model.CNN2d_fitting()  # xiao：原来是这几个模型

## Train model
import trainmodel
# net_fitting_inner = trainmodel.train(net_fitting_inner, trainloader, 'fitting')
net_xiao = trainmodel.train(net_classifier, trainloader, 'fitting')
## save and load trained-model
# PATH = 'trained_model/net_fitting_inner_2D_4096_unnormal.pkl' #net1为1D卷积神经网络模型，net2为2D卷积神经网络模型
#torch.save(net_fitting_inner, PATH)   # 保存训练过的模型
# net_fitting_inner = torch.load(PATH)     # 加载训练过的模型 xiao: 所以现在一直都在使用已经加载好的模型。
# print(net_fitting_inner)
## Test model
# net_fitting_inner.eval()

predict = []
label = []
for x, y in trainloader:             # 训练误差
    #x = x.view(x.size(0), 3*32*32)
    out = net_xiao(x)
#    print(out,y)
    out = out.detach().numpy().squeeze(1)
    y = y.numpy()
#    print(out,y)
    # out:[b, 10]
    predict = np.hstack((predict, out)) #xiao：将数据水平平铺
    label = np.hstack((label, y))
#    print(predict, label) 
    
    #loss = loss_function(out, y)
    #print(loss.item())

# print(predict, label)
plt.figure(figsize=(20,5))
plt.plot(predict[:50])
plt.plot(label[:50])
plt.show()
