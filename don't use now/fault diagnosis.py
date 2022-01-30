# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 17:35:43 2019

@author: wangc

Modify on Mon Nov 8 18:28:00 2021

@author: Xiao Peng
"""

import numpy as np
import torch
import matplotlib.pyplot as plt  # plt 用于显示图片

## load the dataset

import xiao_dataset_random
import xiaodataset

#cifar = dataset.FlameSet('gear_fault', 2304, '2D', 'incline')

cifar = xiao_dataset_random.FlameSet('insert_fault', 2304, '2D', 'try')

traindata_id, testdata_id = cifar._shuffle()  # xiao：随机生成训练数据集与测试数据集

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# create training and validation sampler objects
tr_sampler = SubsetRandomSampler(traindata_id)  # xiao：生成子数据例
val_sampler = SubsetRandomSampler(testdata_id)

# create iterator objects for train and valid datasets
# xiao：Dataloader是个迭代器，也是Pytorch的数据接口
# xiao: 数据批不恰当时会严重影响精准度
train_batch_size=30
trainloader = DataLoader(cifar, batch_size=train_batch_size, sampler=tr_sampler,
                         shuffle=False)  # dataset就是Torch的Dataset格式的对象；batch_size即每批训练的样本数量，默认为；
valid_batch_size=1
validloader = DataLoader(cifar, batch_size=valid_batch_size, sampler=val_sampler,
                         shuffle=False)  # shuffle表示是否需要随机取样本；num_workers表示读取样本的线程数。

# Define model
from torch import nn

import model

net = model.CNN2d_classifier_xiao()  # 选择神经网络模型

import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)

loss_function = nn.NLLLoss()  # classify
# loss_function = nn.MSELoss()  # fitting

train_loss, valid_loss = [], []

for epoch in range(200):
    net.train()
    for batch_idx, (x, y) in enumerate(trainloader):

        out = net(x)
        # print(out, y)
        loss = loss_function(out, y)

        loss.backward()  # 计算倒数
        optimizer.step()  # w' = w - Ir*grad 模型参数更新
        optimizer.zero_grad()

        # if batch_idx % 10 == 0:  # 训练过程，输出并记录损失值
        #     print(epoch, batch_idx, loss.item())

        train_loss.append(loss.item())  # loss仍然有一个图形副本。在这种情况中，可用.item()来释放它.(提高训练速度技巧)
    if loss.item()<0.01:
        print("break at epoch ",epoch)
        break
    if epoch==199:
        print("it need more than 200 epoch to best fit this situation")
index = np.linspace(1, len(train_loss), len(train_loss))  # 训练结束，绘制损失值变化图
plt.figure()
plt.plot(index, train_loss)
plt.title("clip size=2304")
plt.show()

PATH = 'trained_model/net_xiao.pkl' # net1为1D卷积神经网络模型，net2为2D卷积神经网络模型
# #PATH = 'trained_model/net_xiao10.pkl' # net1为1D卷积神经网络模型，net2为2D卷积神经网络模型
torch.save(net, PATH)

# Model class must be defined somewhere
#net = torch.load(PATH)     # 加载训练过的模型

# net.eval()

total_correct = 0
for x, y in trainloader:  # 训练误差

    out = net(x)
    # out:[b, 10]
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

    # loss = loss_function(out, y)
    # print(loss.item())

# train_batch_size确保得到的精准度真实 （xiao）
total_num = len(trainloader) * train_batch_size
acc = total_correct / total_num
print(total_correct,total_num)
print('train_acc', acc)

total_correct = 0
for x, y in validloader:  # 测试误差
    out = net(x)
    # print(out)
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    # print(pred,y)
    total_correct += correct

total_num = len(validloader) * valid_batch_size
acc = total_correct / total_num
print(total_correct,total_num)
print('test_acc', acc)
exit(1)