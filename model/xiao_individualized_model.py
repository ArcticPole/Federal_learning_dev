# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:48:00 2022

@author: Xiao Peng
"""

from torch import nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt  # plt 用于显示图片
import numpy as np
import xiao_feature_enhance
import xiao_global_feature

import William_model_test as wt

# presentage = wt.data_identification()
# presentage.pop(4)
# presentage.pop(8)
# print(presentage)
# net_list=['global_models/source_models/net_xiao_L_fault_all_ready.pkl',
#           'global_models/source_models/net_xiao_L_fault_foreign_body.pkl',
#           'global_models/source_models/net_xiao_L_fault_incline.pkl',
#           'global_models/source_models/net_xiao_L_fault_no_base.pkl',
#           'global_models/source_models/net_xiao_insert_fault_all_ready.pkl',
#           'global_models/source_models/net_xiao_insert_fault_foreign_body.pkl',
#           'global_models/source_models/net_xiao_insert_fault_incline.pkl',
#           'global_models/source_models/net_xiao_insert_fault_no_base.pkl',
#           ]

# net_list = ['global_models/source_models/net_xiao_L_fault_classify.pkl',
#            'global_models/source_models/net_xiao_insert_fault_classify.pkl']
# presentage=[presentage[0],
#             presentage[1]]
# nl = len(net_list)
# all_weights = xiao_global_feature.catch_feature(net_list, nl)
# for i in range(len(presentage)):
#     if presentage[i] > 0.1:
#         print(presentage[i])
#         # all_weights[i] = xiao_feature_enhance.f_e(all_weights[i])
#         for j in range(len(all_weights[i])):
#             all_weights[i][j].data*=presentage[i]
#     else:
#         for j in range(len(all_weights[i])):
#             all_weights[i][j].data *= 0
#
# weight=all_weights[0]
#
# for i in range(1,nl):
#     for j in range(len(weight)):
#         weight[j].data += all_weights[i][j]
# for i in range(len(weight)):
#     weight[i].data /= pow(nl, 2)

class cnn2d_xiao_individual(nn.Module):

    def __init__(self):
        super(cnn2d_xiao_individual, self).__init__()

        presentage,pre = wt.data_identification()
        net_list = ['global_models/source_models/net_xiao_L_fault_classify.pkl',
                    'global_models/source_models/net_xiao_insert_fault_classify.pkl']
        print(pre)
        nl = len(net_list)
        all_weights = xiao_global_feature.catch_feature(net_list, nl)
        for i in range(len(pre)):
            all_weights[i] = xiao_feature_enhance.f_e(all_weights[i])
            for j in range(len(all_weights[i])):
                all_weights[i][j].data *= pre[i]

        weight = all_weights[0]

        for i in range(1, nl):
            for j in range(len(weight)):
                weight[j].data += all_weights[i][j]
        for i in range(len(weight)):
            weight[i].data /= pow(nl, 2)

        self.weight1 = weight[0]
        self.weight2 = weight[1]
        self.weight3 = weight[2]
        self.weight4 = weight[3]

        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(2304, 288),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2),
            nn.Linear(288, 72),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2),
            nn. Linear(72, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # print(x.shape)
        x = F.conv2d(x, self.weight1, bias=None, stride=1, padding=2, dilation=1, groups=1)
        x = F.max_pool2d(F.relu(x),kernel_size=2, stride=2)
        # print(x.shape)
        x = F.conv2d(x, self.weight2, bias=None, stride=1, padding=1, dilation=1, groups=1)
        x = F.max_pool2d(F.relu(x),kernel_size=2, stride=2)
        # print(x.shape)
        x = F.conv2d(x, self.weight3, bias=None, stride=1, padding=1, dilation=1, groups=1)
        x = F.max_pool2d(F.relu(x),kernel_size=2, stride=2)
        # print(x.shape)
        x = F.conv2d(x, self.weight4, bias=None, stride=1, padding=1, dilation=1, groups=1)
        x = F.max_pool2d(F.relu(x),kernel_size=2, stride=2)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x



##########################################checking########################################################
# import xiao_dataset_random as xdr
# cifar = xdr.FlameSet('global', 2304, '2D', 'global')
import William_dataset_random as wdr
cifar = wdr.FlameSet_test('all', 2304, '2D', 'classify')
traindata_id, testdata_id = cifar._shuffle()  # xiao：Randomly generate training data set and test data set

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# create training and validation sampler objects
tr_sampler = SubsetRandomSampler(traindata_id)
val_sampler = SubsetRandomSampler(testdata_id)

# create iterator objects for train and valid datasets
# xiao：Dataloader is an iterator and the data interface of pytorch
# xiao: Improper data batch will seriously affect the accuracy
train_batch_size=30
trainloader = DataLoader(cifar, batch_size=train_batch_size, sampler=tr_sampler,
                         shuffle=False)
# Dataset is an object in the dataset format of torch; batch_ Size is the number of samples in each batch of training.
valid_batch_size=1
validloader = DataLoader(cifar, batch_size=valid_batch_size, sampler=val_sampler,
                         shuffle=False)
# Shuffle indicates whether it is necessary to take samples at random;
# num_ Workers indicates the number of threads reading samples.


net = cnn2d_xiao_individual()  # 加载训练过的模型

from torch import optim

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

        loss.backward()  # Count down
        optimizer.step()  # w' = w - Ir*grad--Model parameter update
        optimizer.zero_grad()

        if batch_idx % 10 == 0:  # In the training process, output and record the loss value
            print(epoch, batch_idx, loss.item())

        train_loss.append(loss.item())
    if loss.item()<0.0001:
        print("break at epoch ",epoch)
        break
    if epoch == 999:
        print("it need more than 1000 epoch to best fit this situation")

index = np.linspace(1, len(train_loss), len(train_loss))  # 训练结束，绘制损失值变化图
plt.figure()
plt.plot(index, train_loss)
plt.show()
PATH = 'global_models/net_xiao_global_indi.pkl'
torch.save(net, PATH)

# print(net.weight1)
total_correct = 0
for x, y in trainloader:  # train error
    out = net(x)
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

    # loss = loss_function(out, y)
    # print(loss.item())

# train_batch_size make sure accuracy is correct （xiao）
total_num = len(trainloader) * train_batch_size
acc = total_correct / total_num
print('train_acc', acc)

total_correct = 0
for x, y in validloader:  # test error
    out = net(x)
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

total_num = len(validloader) * valid_batch_size
acc = total_correct / total_num
print('test_acc', acc)

conv_layers = []
model_weights = []
model_children = list(net.children())
counter = 0


exit(1)

