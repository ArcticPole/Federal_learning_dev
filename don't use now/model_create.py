# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 09:49:00 2021

@author: Xiao Peng
"""

import torch
import model_feature
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt  # plt 用于显示图片
import numpy as np

weight = model_feature.weight()
# print(len(weight[2]))
#print(weight)
class cnn2d_xiao_created(nn.Module):
    def __init__(self):
        super().__init__()
        # self.weight1 = nn.parameter(weight[1])
        # self.weight2 = nn.parameter(weight[2])
        # self.weight3 = nn.parameter(weight[3])
        # self.weight4 = nn.parameter(weight[4])
        self.features = nn.Sequential(
            #nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(2304, 288),
            nn.ReLU(inplace=True),
            nn.Linear(288, 72),
            nn.ReLU(inplace=True),
            nn.Linear(72, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # print(x.shape)
        x = F.conv2d(x, weight[0], bias=None, stride=1, padding=2, dilation=1, groups=1)
        x = F.max_pool2d(F.relu(x),kernel_size=2, stride=2)
        # print(x.shape)
        x = F.conv2d(x, weight[1], bias=None, stride=1, padding=1, dilation=1, groups=1)
        x = F.max_pool2d(F.relu(x),kernel_size=2, stride=2)
        # print(x.shape)
        x = F.conv2d(x, weight[2], bias=None, stride=1, padding=1, dilation=1, groups=1)
        x = F.max_pool2d(F.relu(x),kernel_size=2, stride=2)
        # print(x.shape)
        x = F.conv2d(x, weight[3], bias=None, stride=1, padding=1, dilation=1, groups=1)
        x = F.max_pool2d(F.relu(x),kernel_size=2, stride=2)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

import xiao_dataset_random as xdr

cifar = xdr.FlameSet('L_fault', 2304, '2D', 'classify')

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



net = cnn2d_xiao_created()     # 加载训练过的模型

import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)

loss_function = nn.NLLLoss()  # classify
# loss_function = nn.MSELoss()  # fitting

train_loss, valid_loss = [], []

for epoch in range(20):
    net.train()
    for batch_idx, (x, y) in enumerate(trainloader):

        out = net(x)
        # print(out, y)
        loss = loss_function(out, y)

        loss.backward()  # 计算倒数
        optimizer.step()  # w' = w - Ir*grad 模型参数更新
        optimizer.zero_grad()

        if batch_idx % 10 == 0:  # 训练过程，输出并记录损失值
            print(epoch, batch_idx, loss.item())

        train_loss.append(loss.item())  # loss仍然有一个图形副本。在这种情况中，可用.item()来释放它.(提高训练速度技巧)

index = np.linspace(1, len(train_loss), len(train_loss))  # 训练结束，绘制损失值变化图
plt.figure()
plt.plot(index, train_loss)
plt.title("clip size=2304")
plt.show()

total_correct = 0
for x, y in trainloader:  # 训练误差
    # print(x.shape)
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