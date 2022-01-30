import torch
import torch.nn as nn

exp='insert_fault'
kind=['incline', 'foreign_body', 'no_base', 'all_ready', 'classify']


ind = 0
PATH = 'trained_model/net_xiao_%s_%s.pkl' % (exp, kind[ind])
net = torch.load(PATH)

model_children = list(net.children())
counter = 0

for i in range(len(model_children)):
    # print(i)
    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        # one_weights.append(model_children[i].weight)
        # print(model_children[i].weight[0].shape)
        # conv_layers.append(model_children[i])
        # print(model_children[i].weight.shape)
        if (i == 0):
            first_weights = model_children[i].weight
        if (i == 1):
            second_weights = model_children[i].weight
        if (i == 2):
            third_weights = model_children[i].weight
        if (i == 3):
            forth_weights = model_children[i].weight


for ind in range(1,4):
    PATH = 'trained_model/net_xiao_%s_%s.pkl' % (exp, kind[ind])
    net = torch.load(PATH)

    conv_layers = []
    model_children = list(net.children())
    counter = 0



    for i in range(len(model_children)):
        # print(i)
        if type(model_children[i]) == nn.Conv2d:
            counter += 1

            # 拼接tensor
            # if (i == 0):
            #     first_weights = torch.cat((first_weights, model_children[i].weight), 0)
            # if (i == 1):
            #     second_weights = torch.cat((second_weights, model_children[i].weight), 0)
            # if (i == 2):
            #     third_weights = torch.cat((third_weights, model_children[i].weight), 0)
            # if (i == 3):
            #     forth_weights = torch.cat((forth_weights, model_children[i].weight), 0)

            if (i == 0):
                first_weights = first_weights + model_children[i].weight
            if (i == 1):
                second_weights = second_weights + model_children[i].weight
            if (i == 2):
                third_weights = third_weights + model_children[i].weight
            if (i == 3):
                forth_weights = forth_weights + model_children[i].weight

print(first_weights.shape)


def weights1():
    return first_weights

def weights2():
    return second_weights

def weights3():
    return third_weights

def weights4():
    return forth_weights