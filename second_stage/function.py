# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 10:09:00 2022

@author: Xiao Peng

catch_feature: catch weight feature from several models

model_accuracy_calculate: calculate given model's accuracy

"""
import torch
import torch.nn as nn


# netlist is address of models, n is number of models
# eg: net_list = ['global_models/source_models/net_xiao_L_fault_classify.pkl',
#             'global_models/source_models/net_xiao_insert_fault_classify.pkl']
#      n: len(netlist)
def catch_feature(netlist, n):
    weight_list = []
    for t in range(n):
        net = torch.load(netlist[t])
        conv_layers = []
        model_weights = []
        model_children = list(net.children())
        counter = 0

        for i in range(len(model_children)):
            if type(model_children[i]) == nn.Conv2d:
                counter += 1
                model_weights.append(model_children[i].weight)
                conv_layers.append(model_children[i])
            elif type(model_children[i]) == nn.Sequential:
                for j in range(len(model_children[i])):
                    if type(model_children[i][j]) == nn.Conv2d:
                        counter += 1
                        model_weights.append(model_children[i][j].weight)
                        conv_layers.append(model_children[i][j])
            weight_list.append(model_weights)
    return weight_list


# PATH: load model's path
# eg: PATH = 'global_models/source_models/net_xiao_global10.pkl'
def model_accuracy_calculate(PATH):
    # get dataset
    import xiao_dataset_random as xdr
    cifar = xdr.FlameSet('global', 2304, '2D', 'global')
    train_data_id, test_data_id = cifar._shuffle()

    # create training and validation sampler objects
    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import SubsetRandomSampler
    tr_sampler = SubsetRandomSampler(train_data_id)
    val_sampler = SubsetRandomSampler(test_data_id)
    train_batch_size = 30
    trainloader = DataLoader(cifar, batch_size=train_batch_size, sampler=tr_sampler,
                             shuffle=False)
    valid_batch_size = 1
    validloader = DataLoader(cifar, batch_size=valid_batch_size, sampler=val_sampler,
                             shuffle=False)

    # load model
    import torch
    net = torch.load(PATH)

    # calculate test accuracy
    total_correct = 0
    net.eval()
    for x, y in validloader:
        out = net(x)
        pred = out.argmax(dim=1)
        correct = pred.eq(y).sum().float().item()
        total_correct += correct
    total_num = len(validloader) * valid_batch_size
    acc = total_correct / total_num
    print(total_correct, total_num)
    print('test_acc', acc)
