# -*- coding: utf-8 -*-
"""
Modified on Mon Jan 22 20:04:21 2022

@author: William
本文件主要是用来模拟本地无模型，发送部分打乱数据到云端，由云端识别场景并得到百分比的过程
相当于工业上的模拟验证云端分类模型的可靠性
"""

import torch
import William_dataset_random as wdr
import numpy as np


from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from model import cnn2d_xiao_global

def data_identification():
  # create training and validation sampler objects
  #tr_sampler = SubsetRandomSampler(traindata_id)  # xiao：生成子数据例
  #cifar = wdr.FlameSet_test('L_fault', 2304, '2D', 'classify_advanced')
  cifar = wdr.FlameSet_test('all', 2304, '2D', 'gear_normal')
  traindata_id, testdata_id = cifar._shuffle()  # xiao：随机生成训练数据集与测试数据集

  val_sampler = SubsetRandomSampler(testdata_id)

  # create iterator objects for train and valid datasets
  # xiao：Dataloader是个迭代器，也是Pytorch的数据接口
  # xiao: 数据批不恰当时会严重影响精准度
  #train_batch_size=30
  #trainloader = DataLoader(cifar, batch_size=train_batch_size, sampler=tr_sampler,
    #                       shuffle=False)  # dataset就是Torch的Dataset格式的对象；batch_size即每批训练的样本数量，默认为；
  valid_batch_size=1
  validloader = DataLoader(cifar, batch_size=valid_batch_size, sampler=val_sampler,
                          shuffle=False)

  

  #PATH = 'global_models/source_models/net_xiao_global10.pkl'
  PATH = 'global_models/source_models/net_xiao_normal_normal.pkl'
  net = torch.load(PATH)

  times = 1
  total_correct = 0
  net.eval()
  proposal = [0,0]
  for x, y in validloader:  # 测试误差
      out = net(x)
      #print(out)
      #print(x)
      pred = out.argmax(dim=1)
      out_numpy=out.detach().numpy()
      #out_numpy = torch.numpy(out)
      out_numpy = out_numpy[0]
      #max_index = np.argsort(out_numpy)
      #max_1 = torch.max(out)
      '''
      total = 1/abs(out_numpy[max_index[9]]) + 1/abs(out_numpy[max_index[8]])
      proposal[0] = proposal[0] + (1/abs(out_numpy[max_index[9]]))/total
      proposal[1] = proposal[1] + (1/abs(out_numpy[max_index[8]]))/total
      '''
      #print(out_numpy[4])
      #print(out_numpy[9])
      '''
      total = 1/abs(out_numpy[4]) + 1/abs(out_numpy[9])
      proposal[0] = proposal[0] + (1/abs(out_numpy[4]))/total
      proposal[1] = proposal[1] + (1/abs(out_numpy[9]))/total
      '''
      proposal[0] = proposal[0] +out_numpy[4]
      proposal[1] = proposal[1] +out_numpy[9]

      correct = pred.eq(y).sum().float().item()
      # print(pred,y)
      # print(pred,y)
      if times == 1:
        C = pred
        times = times + 1
      else:
        C=torch.cat((C,pred),0)
      total_correct += correct
  #print(C)
 
  CC=C.numpy().tolist()
  dict = {}
  num = [0]*10
  num_error = 0
  pre = [0]*10 #这里的百分比不算normal的
  '''
  for key in CC:
      dict[key] = dict.get(key, 0) + 1
      num[key] = num[key] + 1
      if key != 4 and key != 9:
        num_error = num_error + 1
  for key in range(len(pre)):
      if key != 4 and key != 9:
        pre[key] = num[key]/num_error
  '''
  for key in CC:
    dict[key] = dict.get(key, 0) + 1
    num[key] = num[key] + 1
    num_error = num_error + 1
  for key in range(len(pre)):
    pre[key] = num[key]/num_error

  total = 1/abs(proposal[0]) + 1/abs(proposal[1])
  proposal[0] =  (1/abs(proposal[0]))/total
  proposal[1] =  (1/abs(proposal[1]))/total

  '''
  pro_sum = proposal[0]+proposal[1]
  proposal[0] = proposal[0]/pro_sum
  proposal[1] = proposal[1]/pro_sum
  '''
  print(proposal)
  class_dic={0:"incline",1:"foreign_body",2:"no_base",4:"all_ready",5:"normal"}
  #print(dict)
  #print(num)
  #print(num_error)
  #print(pre)

  total_num = len(validloader) * valid_batch_size
  acc = total_correct / total_num
  #print(total_correct,total_num)
  #print('hhh')
  #print('test_acc', acc)
  #exit(1)
  pre = [pre[4], pre[9]]
  return proposal,pre