# -*- coding: utf-8 -*-
"""
Modified on Mon Jan 22 20:04:21 2022

@author: William
本文件主要是用来产生两种数据
1.训练分类模型的随机数据：Flameset_train
2.模拟工业场景的随机分类数据: Flameset_test

"""

import torch
import random
import numpy as np
from torch.utils import data
import pandas as pd
import os

def getRandomIndex(n, x, d):
    # The index range is [0, n), and X are randomly selected without repetition
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


class FlameSet_test(data.Dataset):
    def __init__(self, exp, length, dimension, kind):

        if exp not in ('gear_fault', 'insert_fault', 'L_fault','all','all_normal'):
            print("wrong experiment name: '{}'".format(exp))
            exit(1)
        if kind not in ('incline', 'foreign_body', 'no_base', 'all_ready', 'classify', 'classify_advanced','gear_normal'):
            print("wrong rpm value: '{}'".format(kind))
            exit(1)
        self.length = length
        self.data_id = 0
        self.dataset = np.zeros((0, self.length))  # xiao: create a empty array
        self.label = []
        self.traindata_id = []
        self.testdata_id = []

        if exp == 'gear_fault':
            rdir = '../data/gear_fault'
        elif exp == 'insert_fault':
            rdir = '../data/insert_fault'
        elif exp == 'L_fault':
            rdir = '../data/L_fault'
        elif exp == 'all':
            rdir_list =['../data/L_fault', '../data/insert_fault']

        if exp != 'all':
            if kind == 'incline':
                mydatalist = ['1_incline.csv']
                mylabellist = [0]
            elif kind == 'foreign_body':
                mydatalist = ['2_foreign_body.csv']
                mylabellist = [1]
            elif kind == 'no_base':
                mydatalist = ['3_no_base.csv']
                mylabellist = [2]
            elif kind == 'all_ready':
                mydatalist = ['4_all_ready.csv']
                mylabellist = [3]
            elif kind == 'classify':
                mydatalist = ['1_incline.csv', '2_foreign_body.csv', '3_no_base.csv', '4_all_ready.csv', '5_normal.csv']
                mylabellist = [0, 1, 2, 3, 4]
            elif kind == 'classify_advanced':
                mydatalist = ['1_incline.csv', '3_no_base.csv', '5_normal.csv']
                mylabellist = [0, 2, 4]
            elif kind == 'gear_normal':
                mydatalist = ['5_normal.csv']
                mylabellist = [14]
            else:
                print("wrong rpm value: '{}'".format(kind))
                exit(1)   
            for idx in range(len(mydatalist)):  # 遍历故障形式
                rows = 2000
                if mydatalist[idx] == '5_normal.csv' :
                    rows = 20000
                csvdata_path = os.path.join(rdir, mydatalist[idx])  # csv file path
                csv_value = pd.read_csv(csvdata_path,nrows=rows).values  # load csv data
                idx_last = -(csv_value.shape[0]*12 % self.length)//12
                # xiao: Cut the data into segments according to the defined length
                clips = csv_value[:idx_last].reshape(-1, self.length)
                # xiao: The sliced shape has been improved to cut in as much data as possible
                n = clips.shape[0]
                self.dataset = np.vstack((self.dataset, clips))
                # xiao: Importing slices into data vstack is a vertical combination of two arrays
                self.label += [mylabellist[idx]] * n  # xiao: now is add label
                test_index = getRandomIndex(n, n//2,self.data_id)
                self.testdata_id += test_index
                self.data_id += n

        else:
            if kind == 'gear_normal':
                mydatalist = ['5_normal.csv']
                mylabellist = [4, 9]
            if kind == 'classify':
                mydatalist = ['1_incline.csv', '2_foreign_body.csv', '3_no_base.csv', '4_all_ready.csv', '5_normal.csv']
                mylabellist = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
            i = 0
            for idx1 in range(len(rdir_list)):    
                for idx in range(len(mydatalist)):  # iterate all fault
                    csvdata_path = os.path.join(rdir_list[idx1], mydatalist[idx])  # csv file path
                    if idx1 == 0:
                        rows_num = 10000
                    elif idx1 == 1:
                        rows_num = 15000
                    csv_value = pd.read_csv(csvdata_path,nrows=rows_num).values  # load csv data
                    idx_last = -(csv_value.shape[0]*12 % self.length)//12
                    # xiao: Cut the data into segments according to the defined length
                    clips = csv_value[:idx_last].reshape(-1, self.length)
                    # xiao: The sliced shape has been improved to cut in as much data as possible
                    n = clips.shape[0]
                    self.dataset = np.vstack((self.dataset, clips))
                    # xiao: Importing slices into data vstack is a vertical combination of two arrays
                    self.label += [mylabellist[i]] * n
                    i = i + 1
                    train_index = getRandomIndex(n, n//2,self.data_id)
                    test_index = list(set(list(range(self.data_id,n+self.data_id)))-set(train_index))
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
        pil_img = self.dataset[index]  # Read a list of 3x32x32 according to the index
        # print(np.array(pil_img).shape)
        data = self.transforms(pil_img, self.length)
        data = data.unsqueeze(0)
        # When the input data is 1 channel, upgrade the dimension in the first dimension to ensure that the training data X has 3 dimensions
        # print(data.shape)
        label = self.label[index]

        return data, label

    def __len__(self):
        return len(self.dataset)