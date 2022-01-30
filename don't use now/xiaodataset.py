# -*- coding: utf-8 -*-
"""
Create on Mon Nov 8 18:28:00 2021

@author: Xiao Peng
"""

import torch

import scipy.io as scio  # 导入库函数
import numpy as np
from torch.utils import data
import pandas as pd

import os


def process1(datasetdata, length):  # [3,1024]->[3,32,32] list->array->tensor
    if length == 1024:
        traindata = torch.tensor(np.array(datasetdata).reshape(32, 32), dtype=torch.float)
    elif length == 4096:
        traindata = torch.tensor(np.array(datasetdata).reshape(64, 64), dtype=torch.float)
    elif length == 5184:
        traindata = torch.tensor(np.array(datasetdata).reshape(72, 72), dtype=torch.float)
    elif length == 2304:
        traindata = torch.tensor(np.array(datasetdata).reshape(48, 48), dtype=torch.float)

    return traindata


def process2(datasetdata, length):  # [3,1024] list->tensor
    #    Max = max(datasetdata)
    #    Min = min(datasetdata)
    #    for j in range(len(datasetdata)):
    #        datasetdata[j] = (datasetdata[j]-Min)/(Max-Min)-0.5
    traindata = torch.tensor(datasetdata, dtype=torch.float)
    return traindata


# 定义自己的数据集合
class FlameSet(data.Dataset):
    def __init__(self, exp, length, dimension, kind, index):

        if exp not in ('gear_fault', 'insert_fault', 'L_fault'):
            print("wrong experiment name: '{}'".format(exp))
            exit(1)
        if kind not in ('incline', 'foreign_body', 'no_base', 'all_ready', 'classify'):
            print("wrong rpm value: '{}'".format(kind))
            exit(1)  #xiao: 这个貌似是看是什么训练模型
        self.length = length
        self.data_id = 0
        self.dataset = np.zeros((0, self.length))  # xiao: 创建了一个空的array
        self.label = []
        self.traindata_id = []
        self.testdata_id = []

        if exp == 'gear_fault':
            rdir = 'data/gear_fault'
        elif exp == 'insert_fault':
            rdir = 'data/insert_fault'
        elif exp == 'L_fault':
            rdir = 'data/L_fault'

        if kind == 'incline':
            mydatalist = ['1_incline.csv', '5_normal.csv']
            mylabellist = [0, 1]  # 现在这里应该是打标签吧？（xiao）
        elif kind == 'foreign_body':
            mydatalist = ['2_foreign_body.csv', '5_normal.csv']
            mylabellist = [0, 1]
        elif kind == 'no_base':
            mydatalist = ['3_no_base.csv', '5_normal.csv']
            mylabellist = [0, 1]
        elif kind == 'all_ready':
            mydatalist = ['4_incline.csv', '5_normal.csv']
            mylabellist = [0, 1]
        elif kind == 'classify':
            mydatalist = ['1_incline.csv', '2_foreign_body.csv', '3_no_base.csv', '4_all_ready.csv', '5_normal.csv']
            mylabellist = [0, 1, 2, 3, 4]
        else:
            print("wrong rpm value: '{}'".format(kind))
            exit(1)


        for idx in range(len(mydatalist)):  # 遍历故障形式
            csvdata_path = os.path.join(rdir, mydatalist[idx])  # csv 文件路径
            csv_value_all = pd.read_csv(csvdata_path).values  # 导入csv数据

            csv_value=csv_value_all[:, [index, index+6]]
            # print(csv_value.shape)
            idx_last = -(csv_value.shape[0]*2 % self.length)//2  # xiao: 根据定义的长度，将数据切割成段
            # print(csv_value[:idx_last].shape)
            clips = csv_value[:idx_last].reshape(-1, self.length)  # xiao：切片
            # print(clips.shape)  # xiao: 切片的shape 经改进后切入了尽可能多的数据
            n = clips.shape[0]
            # print(idx)  # xiao：故障类型的index
            n_split = 4 * n // 5

            self.dataset = np.vstack((self.dataset, clips))  # xiao: 把切片导入到数据 vstack是垂直组合两个array
            self.label += [mylabellist[idx]] * n  # xiao:在这才是打标签吧
            # xiao: 将数据分成traindata(4/5)和testdata(1/5)
            self.traindata_id += list(range(self.data_id, (self.data_id + n_split)))
            self.testdata_id += list(range((self.data_id + n_split), (self.data_id + n)))
            self.data_id += n

        if dimension == '2D':
            self.transforms = process1
        elif dimension == '1D':
            self.transforms = process2
        else:
            print('input a wrong dimension')

    def _shuffle(self):

        return self.traindata_id, self.testdata_id

    def __getitem__(self, index):
        pil_img = self.dataset[index]  # 根据索引，读取一个3X32X32的列表
        # print(np.array(pil_img).shape)
        data = self.transforms(pil_img, self.length)
        data = data.unsqueeze(0)  # 输入数据为1通道时，在第一维度进行升维，确保训练数据x具有3个维度
        # print(data.shape)
        label = self.label[index]

        return data, label

    def __len__(self):
        return len(self.dataset)


def loaddata(cifar):
    traindata_id, testdata_id = cifar._shuffle()

    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import SubsetRandomSampler

    ## create training and validation sampler objects
    tr_sampler = SubsetRandomSampler(traindata_id)
    val_sampler = SubsetRandomSampler(testdata_id)
    ## create iterator objects for train and valid datasets
    trainloader = DataLoader(cifar, batch_size=50, sampler=tr_sampler,
                             shuffle=False)  # dataset就是Torch的Dataset格式的对象；batch_size即每批训练的样本数量，默认为；
    validloader = DataLoader(cifar, batch_size=50, sampler=val_sampler,
                             shuffle=False)  # shuffle表示是否需要随机取样本；num_workers表示读取样本的线程数。
    return trainloader, validloader  # xiao：此刻返回的即是数据批，在故障检测之中会用到



import matplotlib.pyplot as plt
cifar = FlameSet('gear_fault', 2304, '2D', 'incline',0)
target = 0
data = []
label = []
plt.figure(figsize=(20, 5*10))
for i in range(len(cifar)):
    x, y = cifar[i]
    if y == target:
        data.append(x)
        label.append(y)
        target += 1
        ax = plt.subplot(10, 1, target)
        ax.set_title("condation '{}'".format(y))
        ax.plot(x[0].numpy())
    i += 102

# plt.show()


'''
import matplotlib.pyplot as plt
cifar = FlameSet('48DriveEndFault','1772',4096,'2D', 'net_classifier')
target = 0
data=[]
label=[]
for i in range(len(cifar)):
    x,y = cifar[i]
    if y==target:
        data.append(x)
        label.append(y)
        target += 1
        plt.figure(figsize=(5,5))
        ax = plt.subplot(1,1,1)
        ax.set_title("condation '{}'".format(y))
        ax.imshow(x[0])
    i +=102

plt.show()
'''

# print (x)
# x = np.transpose(x.numpy(), (1, 2, 0))
# print (x.shape)
# idx = 1600
# plt.figure()        # 二维数据的可视化

# idx = 6000
# for i in range(3):
#    x,y = cifar[idx+i]
#    ax = plt.subplot(1, 3, i+1)
#    ax.imshow(x[0])
#    print(cifar[idx+i])
# plt.show()
