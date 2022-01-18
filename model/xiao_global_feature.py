# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 10:09:00 2022

@author: Xiao Peng
"""
import torch
import torch.nn as nn


def catch_feature(netlist,n):
    weight_list=[]
    for i in range(n):
        net = torch.load(netlist[i])
        conv_layers = []
        model_weights = []
        model_children = list(net.children())
        counter = 0

        for i in range(len(model_children)):
            #print(i)
            if type(model_children[i]) == nn.Conv2d:
                counter += 1
                model_weights.append(model_children[i].weight)
                conv_layers.append(model_children[i])
            elif type(model_children[i]) == nn.Sequential:
                for j in range(len(model_children[i])):
                    if type(model_children[i][j])==nn.Conv2d:
                        counter += 1
                        model_weights.append(model_children[i][j].weight)
                        conv_layers.append(model_children[i][j])
            weight_list.append(model_weights)
    return weight_list
