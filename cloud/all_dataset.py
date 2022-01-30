# -*- coding: utf-8 -*-
"""
Created on Tue Dec 7 19:28:00 2021

@author: Xiao Peng
"""

import torch
import random
import numpy as np
from torch.utils import data
import pandas as pd
import os

def getRandomIndex(n, x, d):
    # 索引范围为[0, n), 随机选x个不重复
    index = random.sample(range(d,d+n), x)
    return index

def process1(datasetdata, length):  # [3,1024]->[3,32,32] list->array->tensor

    if length == 1024:
        traindata = torch.tensor(np.array(datasetdata).reshape(32, 32), dtype=torch.float)
    elif length == 4096:
        traindata = torch.tensor(np.array(datasetdata).reshape(64, 64), dtype=torch.float)
    elif length == 5184:
        traindata = torch.tensor(np.array(datasetdata).reshape(72, 72), dtype=torch.float)
    elif length == 2304:
        traindata = torch.tensor(np.array(datasetdata).reshape(48, 48), dtype=torch.float)
    elif length == 576:
        traindata = torch.tensor(np.array(datasetdata).reshape(24, 24), dtype=torch.float)

    # print(traindata.shape)
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
    def __init__(self, length, dimension):

        self.length = length
        self.data_id = 0
        self.dataset = np.zeros((0, self.length))  # xiao: 创建了一个空的array
        self.label = []
        self.traindata_id = []
        self.testdata_id = []

        rdir = ['./data/gear_fault','./data/insert_fault','./data/L_fault']
        rdirlabel = [0, 1, 2]

        mydatalist = ['1_incline.csv', '2_foreign_body.csv', '3_no_base.csv', '4_all_ready.csv', '5_normal.csv']
        mylabellist = [0, 1, 2, 3, 4]

        for ridx in range(len(rdirlabel)):  # 遍历文件路径
            for idx in range(len(mydatalist)):  # 遍历故障形式
                csvdata_path = os.path.join(rdir[ridx], mydatalist[idx])  # csv 文件路径
                csv_value = pd.read_csv(csvdata_path).values  # 导入csv数据
                # print(csv_value.shape)
                idx_last = -(csv_value.shape[0]*12 % self.length)//12  # xiao: 根据定义的长度，将数据切割成段
                # print(csv_value[:idx_last].shape)
                clips = csv_value[:idx_last].reshape(-1, self.length)  # xiao：切片
                # print(clips.shape)  # xiao: 切片的shape 经改进后切入了尽可能多的数据
                n = clips.shape[0]
                # print(idx)  # xiao：故障类型的index
                # n_split = 4 * n // 5
                self.dataset = np.vstack((self.dataset, clips))  # xiao: 把切片导入到数据 vstack是垂直组合两个array
                self.label += [mylabellist[idx]+len(mydatalist)*rdirlabel[ridx]] * n  # xiao:在这才是打标签吧

                train_index = getRandomIndex(n, n*4//5,self.data_id)
                # 再讲test_index从总的index中减去就得到了train_index
                test_index = list(set(list(range(self.data_id,n+self.data_id)))-set(train_index))
                #print(train_index)
                #print(test_index)
                self.traindata_id += train_index
                self.testdata_id += test_index
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



