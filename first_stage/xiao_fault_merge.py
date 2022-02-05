# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 09:25:00 2022

@author: Xiao Peng
@editor: Lin Ze
"""

from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import numpy as np
import tools.xiao_feature_enhance as xiao_feature_enhance
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import tools.xiao_dataset_random as xdr
import torch.optim as optim
import first_stage.model_feature as mf
import first_stage.model_feature_another as mfa

class cnn2d_xiao_merge(nn.Module):
    def __init__(self,weight):
        super().__init__()
        self.weight1 = weight[0]
        self.weight2 = weight[1]
        self.weight3 = weight[2]
        self.weight4 = weight[3]
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
def merge_model(PATH):
    weight = mf.weight()
    other_weight = mfa.weight(PATH)
    weight=xiao_feature_enhance.f_e(weight)
    other_weight=xiao_feature_enhance.f_e(other_weight)

    for i in range(0,len(weight)):  # merge two kernel
        weight[i].data+=other_weight[i]

    cifar = xdr.FlameSet('insert_fault', 2304, '2D', 'try')
    traindata_id, testdata_id = cifar._shuffle()  
    # create training and validation sampler objects
    tr_sampler = SubsetRandomSampler(traindata_id) 
    val_sampler = SubsetRandomSampler(testdata_id)

    # create iterator objects for train and valid datasets
    train_batch_size=30
    trainloader = DataLoader(cifar, batch_size=train_batch_size, sampler=tr_sampler,
                            shuffle=False) 
    valid_batch_size=1
    validloader = DataLoader(cifar, batch_size=valid_batch_size, sampler=val_sampler,
                            shuffle=False) 

    net = cnn2d_xiao_merge(weight)   
    optimizer = optim.SGD(net.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)

    loss_function = nn.NLLLoss()  # classify
    # loss_function = nn.MSELoss()  # fitting
    train_loss, valid_loss = [], []
    for epoch in range(200):
        net.train()
        for batch_idx, (x, y) in enumerate(trainloader):
            out = net(x)
            loss = loss_function(out, y)
            loss.backward()  
            optimizer.step()  
            optimizer.zero_grad()
            train_loss.append(loss.item())  
        if loss.item()<0.01:
            print("break at epoch ",epoch)
            break
        if epoch==199:
            print("it need more than 200 epoch to best fit this situation")

    index = np.linspace(1, len(train_loss), len(train_loss))  
    plt.figure()
    plt.plot(index, train_loss)
    plt.title("clip size=2304")
    plt.show()
    PATH = 'merge_models/net_xiao_merge.pkl'
    torch.save(net, PATH)

    total_correct = 0
    for x, y in trainloader:
        out = net(x)
        pred = out.argmax(dim=1)
        correct = pred.eq(y).sum().float().item()
        total_correct += correct

    total_num = len(trainloader) * train_batch_size
    acc = total_correct / total_num
    print('train_acc of new model:', acc)

    total_correct = 0
    for x, y in validloader:  
        out = net(x)
        pred = out.argmax(dim=1)
        correct = pred.eq(y).sum().float().item()
        total_correct += correct

    total_num = len(validloader) * valid_batch_size
    acc = total_correct / total_num
    print('test_acc of new model:', acc)
    print('The new model has saved!')
    exit(1)