import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


def model_feature_hy(pkl):
    net0 = torch.load(pkl)
    conv_layers = []
    model_weights = []
    model_children = list(net0.children())
    counter = 0

    for i in range(len(model_children)):
        # print(i)
        if type(model_children[i]) == nn.Conv2d:  # 检查模型的直接子层中是否有卷积层。
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:  # 检查序列块中的瓶颈层是否包含任何卷积层。
            for j in range(len(model_children[i])):
                if type(model_children[i][j]) == nn.Conv2d:
                    counter += 1
                    model_weights.append(model_children[i][j].weight)
                    conv_layers.append(model_children[i][j])

    # print(counter)

    '''for i in range(4):
        # print(model_weights[i])
        #plt.subplot(2, 2, i + 1)
        #plt.axis('off')
        # plt.imshow(model_weights[0][i][0, :, :].detach(), cmap='gray')
    '''
    return model_weights
    # plt.show()
# def compare10(matrix1,matrix2):
#     max1=[]
#     max2=[]
#     count=0
#     for i in range(10):
#         max1.append(matrix1(np.argmax(matrix1)))

def similarity(weight1, weight2):
    a = 0.0
    count=0
    for i in range(0, len(weight1)):  # len=4
        for j in range(0, len(weight1[i])):  # i=0,len=32
            # for k in range(0, len(weight1[i][j])):  # i=0,j=0,len=1
            am=np.argmax(weight1[i][j].detach().numpy(),axis=0)
            bm=np.argmax(weight1[i][j].detach().numpy(),axis=1)
            m=[]
            an = np.argmax(weight2[i][j].detach().numpy(), axis=0)
            bn = np.argmax(weight2[i][j].detach().numpy(), axis=1)
            n = []

            ami = np.argmin(weight1[i][j].detach().numpy(), axis=0)
            bmi = np.argmin(weight1[i][j].detach().numpy(), axis=1)
            mi = []
            ani = np.argmin(weight2[i][j].detach().numpy(), axis=0)
            bni = np.argmin(weight2[i][j].detach().numpy(), axis=1)
            ni = []

            for k in range(len(am)):
                m.append((am[k], bm[k]))
                n.append((an[k], bn[k]))
                mi.append((ami[k], bmi[k]))
                ni.append((ani[k], bni[k]))
            # print((0,0)==(0,0))
            for k in range(len(m)):
                for h in range(len(n)):
                    if m[k]==n[h]:
                        count+=1
                    if mi[k]==ni[h]:
                        count+=1
            # a += abs((weight1[i][j].detach().numpy() - weight2[i][j].detach().numpy()))
    return count

address1=['trained_model/1/net_xiao_L_fault_all_ready.pkl',
         'trained_model/1/net_xiao_L_fault_foreign_body.pkl',
         'trained_model/1/net_xiao_L_fault_incline.pkl',
         'trained_model/1/net_xiao_L_fault_no_base.pkl',
          'trained_model/1/net_xiao_L_fault_classify.pkl']
address2=['trained_model/2/net_xiao_L_fault_all_ready.pkl',
          'trained_model/2/net_xiao_L_fault_foreign_body.pkl',
          'trained_model/2/net_xiao_L_fault_incline.pkl',
          'trained_model/2/net_xiao_L_fault_no_base.pkl',
          'trained_model/1/net_xiao_L_fault_classify.pkl']
m=[[address1[0],address2[0]],[address1[0],address2[1]],[address1[0],address2[2]],
   [address1[0],address2[3]],[address1[0],address1[4]]]
for i in range(5):
    weight1 = model_feature_hy(m[i][0])
    weight2 = model_feature_hy(m[i][1])
    sqa_sum = 0
    for j in range(4):
        sqa_sum += similarity(weight1[j], weight2[j])
    print(m[i],sqa_sum)

'''if sqrt > 50:
    print('two model for different error')
else:
    print('two model for nearly same error')
'''
'''
input1=weight1
input2=weight2
similarity=[]
#print(input1.size)
for i in range(4):
    cos=torch.cosine_similarity(input1[i],input2[i],dim=1)
    score = cos.data.cpu().numpy()
    print(cos)
    #print(score.item())
'''
'''
for i in range(0, len(weight)):  # len=4
    for j in range(0, len(weight[i])):  # i=0,len=32
        for k in range(0, len(weight[i][j])):  # i=0,j=0,len=1
            weight_max = max(max((weight[i][j][k]).detach().numpy().tolist()))
            weight_min = min(min((weight[i][j][k]).detach().numpy().tolist()))
            ran = weight_max - weight_min
            weight_med = weight_min + ran * 0.5
            critical_max = weight_med + ran * 0.25
            critical_min = weight_med - ran * 0.25
            # print(weight_med)
            for l in range(0, len(weight[i][j][k])):  # i=0,j=0,l=0,len=5
                for m in range(0, len(weight[i][j][k][l])):
                    # print(weight[i][j][k][l][m].detach().numpy()<critical_max)
                    # print(weight[i][j][k][l][m].detach().numpy()>critical_min)
                    if (weight[i][j][k][l][m].detach().numpy() < critical_max and weight[i][j][k][l][m].detach().numpy() > critical_min):
                        weight[i][j][k][l][m].data *= 0
                        weight[i][j][k][l][m].data += 1
                        weight[i][j][k][l][m].data *= weight_med
'''
