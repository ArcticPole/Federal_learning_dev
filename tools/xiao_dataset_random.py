# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 08:01:21 2019

@author: wangc

Modify on Mon Nov 8 18:28:00 2021

@author: Xiao Peng
"""

import torch
import random
import numpy as np
from torch.utils import data
import pandas as pd
import os

def getRandomIndex(n, x, d):
    # The index range is [0, n], and X are randomly selected without repetition
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


class FlameSet(data.Dataset):
    def __init__(self, exp, length, dimension, kind):

        if exp not in ('gear_fault', 'insert_fault', 'L_fault', 'global', 'normal', 'gear_all'):
            print("wrong experiment name: '{}'".format(exp))
            exit(1)
        if kind not in ('incline', 'foreign_body', 'no_base', 'all_ready', 'classify', 'try', 'global', 'normal'):
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
            kind_differ = 0
        elif exp == 'L_fault':
            rdir = '../data/L_fault'
            kind_differ = 0
        elif exp == 'global':
            rdir = '../data'
        elif exp == 'normal':
            rdir = '../data'
        else:
            exit(1)

        if kind == 'incline':
            mydatalist = ['1_incline.csv', '5_normal.csv']
            mylabellist = [0+kind_differ, 4+kind_differ]
        elif kind == 'foreign_body':
            mydatalist = ['2_foreign_body.csv', '5_normal.csv']
            mylabellist = [1+kind_differ, 4+kind_differ]
        elif kind == 'no_base':
            mydatalist = ['3_no_base.csv', '5_normal.csv']
            mylabellist = [2+kind_differ, 4+kind_differ]
        elif kind == 'all_ready':
            mydatalist = ['4_all_ready.csv', '5_normal.csv']
            mylabellist = [3+kind_differ, 4+kind_differ]
        elif kind == 'classify':
            mydatalist = ['1_incline.csv', '2_foreign_body.csv', '3_no_base.csv', '4_all_ready.csv', '5_normal.csv']
            mylabellist = [0+kind_differ, 1+kind_differ, 2+kind_differ, 3+kind_differ, 4+kind_differ]
        elif kind == 'try':
            mydatalist = ['1_incline.csv', '3_no_base.csv', '5_normal.csv']
            mylabellist = [0+kind_differ, 2+kind_differ, 4+kind_differ]
        elif kind == 'global':
            mydatalist = ['L_fault/1_incline.csv', 'L_fault/2_foreign_body.csv', 'L_fault/3_no_base.csv',
                          'L_fault/4_all_ready.csv', 'L_fault/5_normal.csv',
                          'insert_fault/1_incline.csv', 'insert_fault/2_foreign_body.csv', 'insert_fault/3_no_base.csv',
                          'insert_fault/4_all_ready.csv', 'insert_fault/5_normal.csv']
            mylabellist = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
            # mylabellist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        elif kind == 'normal':
            mydatalist = ['L_fault/5_normal.csv', 'insert_fault/5_normal.csv']
            mylabellist = [4, 9]
        elif kind == 'gear_all':
            mydatalist = ['1_incline.csv', '2_foreign_body.csv', '3_no_base.csv', '5_normal.csv']
            mylabellist = [0, 1, 2, 4]
        else:
            print("wrong rpm value: '{}'".format(kind))
            exit(1)


        for idx in range(len(mydatalist)):  # iterate fault
            csvdata_path = os.path.join(rdir, mydatalist[idx])  # csv file path
            csv_value = pd.read_csv(csvdata_path).values  # load csv data
            idx_last = -(csv_value.shape[0]*12 % self.length)//12
            # xiao: Cut the data into segments according to the defined length
            clips = csv_value[:idx_last].reshape(-1, self.length)
            # xiao: The sliced shape has been improved to cut in as much data as possible
            n = clips.shape[0]
            self.dataset = np.vstack((self.dataset, clips))
            # xiao: Importing slices into data vstack is a vertical combination of two arrays
            self.label += [mylabellist[idx]] * n  # xiao: now is adding label

            train_index = getRandomIndex(n, n*4//5,self.data_id)
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
        # When the input data is 1 channel,
        # upgrade the dimension in the first dimension to ensure that the training data X has 3 dimensions
        label = self.label[index]

        return data, label

    def __len__(self):
        return len(self.dataset)


def loaddata(cifar):
    traindata_id, testdata_id = cifar._shuffle()

    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import SubsetRandomSampler

    # create training and validation sampler objects
    tr_sampler = SubsetRandomSampler(traindata_id)
    val_sampler = SubsetRandomSampler(testdata_id)
    # create iterator objects for train and valid datasets
    trainloader = DataLoader(cifar, batch_size=50, sampler=tr_sampler,
                             shuffle=False)
    # Dataset is an object in the dataset format of torch; batch_ Size is the number of samples in each batch of training
    validloader = DataLoader(cifar, batch_size=50, sampler=val_sampler,
                             shuffle=False)
    # Shuffle indicates whether it is necessary to take samples at random; num_ Workers indicates the number of threads reading samples。
    return trainloader, validloader
    # xiao：At this moment, the data batch is returned, which will be used in fault detection

