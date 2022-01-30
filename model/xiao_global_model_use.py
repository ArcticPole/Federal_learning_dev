import torch

import xiao_dataset_random as xdr

cifar = xdr.FlameSet('global', 2304, '2D', 'global')

traindata_id, testdata_id = cifar._shuffle()  # xiao：随机生成训练数据集与测试数据集

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# create training and validation sampler objects
tr_sampler = SubsetRandomSampler(traindata_id)  # xiao：生成子数据例
val_sampler = SubsetRandomSampler(testdata_id)

# create iterator objects for train and valid datasets
# xiao：Dataloader是个迭代器，也是Pytorch的数据接口
# xiao: 数据批不恰当时会严重影响精准度
train_batch_size=30
trainloader = DataLoader(cifar, batch_size=train_batch_size, sampler=tr_sampler,
                         shuffle=False)  # dataset就是Torch的Dataset格式的对象；batch_size即每批训练的样本数量，默认为；
valid_batch_size=1
validloader = DataLoader(cifar, batch_size=valid_batch_size, sampler=val_sampler,
                         shuffle=False)

from model import cnn2d_xiao_global

PATH = 'global_models/source_models/net_xiao_global10.pkl'
net = torch.load(PATH)


total_correct = 0
net.eval()
for x, y in validloader:  # 测试误差
    out = net(x)
    # print(out)
    # print(x)
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    # print(pred,y)
    total_correct += correct

total_num = len(validloader) * valid_batch_size
acc = total_correct / total_num
print(total_correct,total_num)
print('hhh')
print('test_acc', acc)
exit(1)