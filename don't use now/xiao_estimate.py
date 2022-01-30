# -*- coding: utf-8 -*-
"""
Created on Mon Nov 8 18:28:00 2021

@author: Xiao Peng
"""

import scipy.io as scio  # 导入库函数
from scipy.fftpack import fft
import numpy as np
import torch
from torch.autograd import Variable

from torch.utils import data
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
from torchvision import transforms

## load the dataset

import xiaodataset
def model_vote():
    net0=torch.load('trained_model/net_xiao0.pkl')
    net1=torch.load('trained_model/net_xiao1.pkl')
    net2=torch.load('trained_model/net_xiao2.pkl')
    net3=torch.load('trained_model/net_xiao3.pkl')
    net4=torch.load('trained_model/net_xiao4.pkl')
    net5=torch.load('trained_model/net_xiao5.pkl')
    net0.eval()
    net1.eval()
    net2.eval()
    net3.eval()
    net4.eval()
    net5.eval()

    train_batch_size = 30
    pred0 = np.empty(shape=(0,1))
    pred1 = np.empty(shape=(0,1))
    pred2 = np.empty(shape=(0,1))
    pred3 = np.empty(shape=(0,1))
    pred4 = np.empty(shape=(0,1))
    pred5 = np.empty(shape=(0,1))
    outy = np.empty(shape=(0,1))

    apred0 = np.empty(shape=(0,1))
    apred1 = np.empty(shape=(0,1))
    apred2 = np.empty(shape=(0,1))
    apred3 = np.empty(shape=(0,1))
    apred4 = np.empty(shape=(0,1))
    apred5 = np.empty(shape=(0,1))
    aouty = np.empty(shape=(0,1))

    for ind in range(0, 6):
        cifar = xiaodataset.FlameSet('gear_fault', 2304, '2D', 'incline', ind)

        traindata_id, testdata_id = cifar._shuffle()  # xiao：随机生成训练数据集与测试数据集

        from torch.utils.data import DataLoader
        from torch.utils.data.sampler import SubsetRandomSampler

        # create training and validation sampler objects
        tr_sampler = SubsetRandomSampler(traindata_id)  # xiao：生成子数据例
        val_sampler = SubsetRandomSampler(testdata_id)

        # create iterator objects for train and valid datasets
        # xiao：Dataloader是个迭代器，也是Pytorch的数据接口
        # xiao: 数据批不恰当时会严重影响精准度


        # 看起来trainload也需要迭代对应。

        trainloader = DataLoader(cifar, batch_size=train_batch_size, sampler=tr_sampler,
                                 shuffle=False)  # dataset就是Torch的Dataset格式的对象；batch_size即每批训练的样本数量，默认为；
        valid_batch_size = 1
        validloader = DataLoader(cifar, batch_size=valid_batch_size, sampler=val_sampler,
                                 shuffle=False)  # shuffle表示是否需要随机取样本；num_workers表示读取样本的线程数。

        train_loss, valid_loss = [], []

        # net.eval()
        if ind==0:
            for x, y in trainloader:  # 训练误差
                out0 = net0(x)
                out0 = out0.argmax(dim=1).detach().numpy()
                out0=out0[:, np.newaxis]
                # print(out0.shape)
                # print("a")
                y=y[:, np.newaxis]
                pred0 = np.vstack((pred0, out0))
                outy = np.vstack((outy,y))
            for x, y in validloader:  # 训练误差
                out0 = net0(x)
                out0 = out0.argmax(dim=1).detach().numpy()
                out0=out0[:, np.newaxis]
                apred0 = np.vstack((apred0, out0))
                y = y[:, np.newaxis]
                aouty = np.vstack((aouty,y))
        if ind==1:
            for x, y in trainloader:  # 训练误差
                out1 = net1(x)
                out1 = out1.argmax(dim=1).detach().numpy()
                out1=out1[:, np.newaxis]
                pred1 = np.vstack((pred1,out1))
                # outy = np.vstack((outy,y))
            for x, y in validloader:  # 训练误差
                out1 = net1(x)
                out1 = out1.argmax(dim=1).detach().numpy()
                out1=out1[:, np.newaxis]
                apred1 = np.vstack((apred1,out1))
                # aouty = np.vstack((aouty,y))
        if ind==2:
            for x, y in trainloader:  # 训练误差
                out2 = net2(x)
                out2 = out2.argmax(dim=1).detach().numpy()
                out2=out2[:, np.newaxis]
                pred2 = np.vstack((pred2,out2))
                # outy = np.vstack((outy,y))
            for x, y in validloader:  # 训练误差
                out2 = net2(x)
                out2 = out2.argmax(dim=1).detach().numpy()
                out2=out2[:, np.newaxis]
                apred2 = np.vstack((apred2,out2))
                # aouty = np.vstack((aouty,y))
        if ind==3:
            for x, y in trainloader:  # 训练误差
                out3 = net3(x)
                out3 = out3.argmax(dim=1).detach().numpy()
                out3=out3[:, np.newaxis]
                pred3 = np.vstack((pred3,out3))
                # outy = np.vstack((outy,y))
            for x, y in validloader:  # 训练误差
                out3 = net1(x)
                out3 = out3.argmax(dim=1).detach().numpy()
                out3=out3[:, np.newaxis]
                apred3 = np.vstack((apred3,out3))
                # aouty = np.vstack((aouty,y))
        if ind==4:
            for x, y in trainloader:  # 训练误差
                out4 = net4(x)
                out4 = out4.argmax(dim=1).detach().numpy()
                out4=out4[:, np.newaxis]
                pred4 = np.vstack((pred4,out4))
                # outy = np.vstack((outy,y))
            for x, y in validloader:  # 训练误差
                out4 = net4(x)
                out4 = out4.argmax(dim=1).detach().numpy()
                out4=out4[:, np.newaxis]
                apred4 = np.vstack((apred4,out4))
                # aouty = np.vstack((aouty,y))
        if ind==5:
            for x, y in trainloader:  # 训练误差
                out5 = net5(x)
                out5 = out5.argmax(dim=1).detach().numpy()
                out5=out5[:, np.newaxis]
                pred5 = np.vstack((pred5,out5))
                # outy = np.vstack((outy,y))
            for x, y in validloader:  # 训练误差
                out5 = net5(x)
                out5 = out5.argmax(dim=1).detach().numpy()
                out5=out5[:, np.newaxis]
                apred5 = np.vstack((apred5,out5))
                # aouty = np.vstack((aouty,y))
        print(ind)

    pred = np.hstack((pred0,pred1,pred2,pred3,pred4,pred5))
    apred = np.hstack((apred0,apred1,apred2,apred3,apred4,apred5))
    # print(pred)
    # kind="fit"
    kind="classify"
    total_correct = 0
    for i in range(len(pred)):
        p=pred[i]
        if (kind=="fit"):
            ans=[0,0]
        else:
            ans=[0,0,0,0,0]
        for j in range(len(p)):
            # print(int(p[j]))
            ans[int(p[j])]+=1
        ans=np.array(ans)
        # print(ans)
        p=ans.argmax()
        # print(p,outy[i])
        correct = p==outy[i]
        total_correct += correct
    acc = total_correct / len(pred)
    print(total_correct)
    print('train_acc', acc)

    total_correct = 0
    for i in range(len(apred)):
        p=apred[i]
        if (kind=="fit"):
            ans=[0,0]
        else:
            ans=[0,0,0,0,0]
        for j in range(len(p)):
            ans[int(p[j])]+=1
        ans=np.array(ans)
        p=ans.argmax()
        correct = p==aouty[i]
        total_correct += correct
    acc = total_correct / len(apred)
    print('test_acc', acc)

