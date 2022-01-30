# -*- coding: utf-8 -*-
"""
Modified on Mon Jan 22 20:04:21 2022

@author: William
本文件主要是用来模拟本地无模型，发送部分打乱数据到云端，由云端识别场景并得到百分比的过程
相当于工业上的模拟验证云端分类模型的可靠性
"""

import torch
import third_stage.William_dataset_random as wdr
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
# import sys
# sys.path.append("C:/Users/86184/PycharmProjects/Federal_learning 2.0")
from tools.model import CNN2d_classifier_xiao


def data_identification():
    cifar = wdr.FlameSet_test('all', 2304, '2D', 'gear_normal')
    traindata_id, testdata_id = cifar._shuffle()  # xiao：Randomly generate training data set and test data set

    val_sampler = SubsetRandomSampler(testdata_id)
    valid_batch_size = 1
    validloader = DataLoader(cifar, batch_size=valid_batch_size, sampler=val_sampler,
                          shuffle=False)

    PATH = "../global_models/source_models/net_xiao_normal_normal.pkl"
    net = torch.load(PATH)

    times = 1
    total_correct = 0
    net.eval()
    proposal = [0, 0]
    for x, y in validloader:  # test error
        out = net(x)
        pred = out.argmax(dim=1)
        out_numpy=out.detach().numpy()
        out_numpy = out_numpy[0]

        proposal[0] = proposal[0] +out_numpy[4]
        proposal[1] = proposal[1] +out_numpy[9]

        correct = pred.eq(y).sum().float().item()
        if times == 1:
            C = pred
            times = times + 1
        else:
            C=torch.cat((C,pred),0)
        total_correct += correct
 
    CC=C.numpy().tolist()
    dict = {}
    num = [0]*10
    num_error = 0
    pre = [0]*10  # The percentage here is not in normal

    for key in CC:
        dict[key] = dict.get(key, 0) + 1
        num[key] = num[key] + 1
        num_error = num_error + 1
    for key in range(len(pre)):
        pre[key] = num[key]/num_error

    total = 1/abs(proposal[0]) + 1/abs(proposal[1])
    proposal[0] =  (1/abs(proposal[0]))/total
    proposal[1] =  (1/abs(proposal[1]))/total


    print(proposal)
    class_dic={0:"incline",1:"foreign_body",2:"no_base",4:"all_ready",5:"normal"}


    total_num = len(validloader) * valid_batch_size
    acc = total_correct / total_num

    pre = [pre[4], pre[9]]
    return proposal,pre