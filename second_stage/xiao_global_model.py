# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 09:25:00 2022

@author: Xiao Peng

generate global model
"""

from torch import nn
import torch
import matplotlib.pyplot as plt  # plt 用于显示图片
import numpy as np
import tools.xiao_feature_enhance as xiao_feature_enhance
import function

# net_list=['global_models/source_models/net_xiao_insert_fault_all_ready.pkl',
#           'global_models/source_models/net_xiao_insert_fault_foreign_body.pkl',
#           'global_models/source_models/net_xiao_insert_fault_incline.pkl',
#           'global_models/source_models/net_xiao_insert_fault_no_base.pkl',
#           'global_models/source_models/net_xiao_L_fault_all_ready.pkl',
#           'global_models/source_models/net_xiao_L_fault_foreign_body.pkl',
#           'global_models/source_models/net_xiao_L_fault_incline.pkl',
#           'global_models/source_models/net_xiao_L_fault_no_base.pkl']

# ---------------------------------------------model-----------------------------------------------------
# catch weight feature from L fault and insert fault classify models
net_list = ['../global_models/source_models/net_xiao_L_fault_classify.pkl',
            '../global_models/source_models/net_xiao_insert_fault_classify.pkl']
nl = len(net_list)
all_weights = function.catch_feature(net_list, nl)

# feature enhanced
for i in range(nl):
    all_weights[i] = xiao_feature_enhance.f_e(all_weights[i])

weight = all_weights[0]
for i in range(1, nl):
    for j in range(len(weight)):
        weight[j].data += all_weights[i][j]
for i in range(len(weight)):
    weight[i].data /= pow(nl, 2)
    # weight[i].data*=0

# for i in range(0,len(weight)):  # merge two kernel
#     weight[i].data+=other_weight[i]
# print(weight[0][0][0])


# ---------------------------------------------training-----------------------------------------------------

import tools.xiao_dataset_random as xdr

# Generate train and test datasets randomly
cifar = xdr.FlameSet('global', 2304, '2D', 'global')
train_data_id, test_data_id = cifar._shuffle()

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# create training and validation sampler objects
tr_sampler = SubsetRandomSampler(train_data_id)
val_sampler = SubsetRandomSampler(test_data_id)

# create iterator objects for train and valid datasets
# xiao: Inaccurate data batch may seriously affect accuracy
train_batch_size = 30
train_loader = DataLoader(cifar, batch_size=train_batch_size, sampler=tr_sampler,
                          shuffle=False)
valid_batch_size = 1
valid_loader = DataLoader(cifar, batch_size=valid_batch_size, sampler=val_sampler,
                          shuffle=False)
# Data_loader is an iterator and the data interface of Pytorch

# load trained model
import model
net = model.cnn2d_xiao_global()

import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)

loss_function = nn.NLLLoss()  # classify
# loss_function = nn.MSELoss()  # fitting


# training process
train_loss = []
print('epoch, batch_idx, loss.item()')
for epoch in range(300):
    net.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        out = net(x)
        loss = loss_function(out, y)
        loss.backward()  # Calculate the reciprocal

        optimizer.step()  # w' = w - Ir*grad.  Update model parameters
        optimizer.zero_grad()

        if batch_idx % 10 == 0:  # Training process. Output, record the loss value
            print(epoch, batch_idx, loss.item())
        train_loss.append(loss.item())
    if epoch == 999:
        print("it need more than 1000 epoch to best fit this situation")

# plot loss value change diagram and save trained model
index = np.linspace(1, len(train_loss), len(train_loss))
plt.figure()
plt.plot(index, train_loss)
plt.show()
PATH = '../global_models/net_xiao_global5.pkl'
torch.save(net, PATH)  # save model
state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'weight': weight}
PATH2 = '../global_models/net_xiao_global_para_hy_2.pkl'
torch.save(state, PATH2)  # save model parameters

# ---------------------------------------------accuracy-----------------------------------------------------

# calculate training accuracy
total_correct = 0
for x, y in train_loader:
    out = net(x)
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct
total_num = len(train_loader) * train_batch_size
acc = total_correct / total_num
print('train_acc', acc)

# calculate test accuracy
total_correct = 0
for x, y in valid_loader:
    out = net(x)
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct
total_num = len(valid_loader) * valid_batch_size
acc = total_correct / total_num
print('test_acc', acc)

exit(1)
