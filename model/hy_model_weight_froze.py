import xiao_dataset_random as xdr

# 本地data
data = xdr.FlameSet('insert_fault', 2304, '2D', 'try')
train_data_id, test_data_id = data._shuffle()  # xiao：随机生成训练数据集与测试数据集

# 模型结构
import torch
from torch import nn
import torch.nn.functional as F


# 这个结构是不是应该从云端发下来
class cnn2d_xiao_merge(nn.Module):
    def __init__(self):
        super().__init__()
        # self.weight1 = nn.parameter(weight[1])
        # self.weight2 = nn.parameter(weight[2])
        # self.weight3 = nn.parameter(weight[3])
        # self.weight4 = nn.parameter(weight[4])
        log_para = 'global_models/source_models/net_xiao_global_para_hy_2.pkl'
        # model = cnn2d_xiao_merge()
        checkpoint = torch.load(log_para)
        weight = checkpoint['weight']
        self.features = nn.Sequential(
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(2304, 288),
            nn.ReLU(inplace=True),
            nn.Linear(288, 72),
            nn.ReLU(inplace=True),
            nn.Linear(72, 10),
            nn.LogSoftmax(dim=1)
        )
        self.weight1 = nn.Parameter(data=weight[0], requires_grad=False)
        self.weight2 = nn.Parameter(data=weight[1], requires_grad=False)
        self.weight3 = nn.Parameter(data=weight[2], requires_grad=False)
        self.weight4 = nn.Parameter(data=weight[3], requires_grad=False)


    def forward(self, x):
        # print(x.shape)
        x = F.conv2d(x, self.weight1, bias=None, stride=1, padding=2, dilation=1, groups=1)
        x = F.max_pool2d(F.relu(x), kernel_size=2, stride=2)
        # print(x.shape)
        x = F.conv2d(x, self.weight2, bias=None, stride=1, padding=1, dilation=1, groups=1)
        x = F.max_pool2d(F.relu(x), kernel_size=2, stride=2)
        # print(x.shape)
        x = F.conv2d(x, self.weight3, bias=None, stride=1, padding=1, dilation=1, groups=1)
        x = F.max_pool2d(F.relu(x), kernel_size=2, stride=2)
        # print(x.shape)
        x = F.conv2d(x, self.weight4, bias=None, stride=1, padding=1, dilation=1, groups=1)
        x = F.max_pool2d(F.relu(x), kernel_size=2, stride=2)
        # # print(x.shape)  hy
        # x = F.conv2d(x, weight[4], bias=None, stride=1, padding=1, dilation=1, groups=1)
        # x = F.max_pool2d(F.relu(x), kernel_size=2, stride=2)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

# weight = model_feature_extract(log_dir)  # 写了个函数 和model_feature内容差不多
# print(weight)

# 测试数据
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# create training and validation sampler objects
tr_sampler = SubsetRandomSampler(train_data_id)  # xiao：生成子数据例
val_sampler = SubsetRandomSampler(test_data_id)
train_batch_size = 50
test_batch_size = 10
train_loader = DataLoader(data, batch_size=train_batch_size, sampler=tr_sampler,
                          shuffle=False)  # dataset就是Torch的Dataset格式的对象；batch_size即每批训练的样本数量，默认为；
test_loader = DataLoader(data, batch_size=test_batch_size, sampler=val_sampler,
                         shuffle=False)  # shuffle表示是否需要随机取样本；num_workers表示读取样本的线程数。

# 云端模型下载
log_para = 'global_models/source_models/net_xiao_global_para_hy_2.pkl'
model = cnn2d_xiao_merge()
checkpoint = torch.load(log_para)
# model.load_state_dict(checkpoint['model'])
start_epoch = checkpoint['epoch']
# weight = checkpoint['weight']
import torch.optim as optim
weight_before=checkpoint['weight']
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)
# optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)
optimizer.load_state_dict(checkpoint['optimizer'])
# a=checkpoint['optimizer']
# print(checkpoint['optimizer'])

# 模型测试
def test(model, test_loader):
    test_loss = 0
    correct = 0
    model.eval()  # 没有用model.train()
    # loss_function = nn.NLLLoss()  # classify
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            # print(model(x))
            out = model(x)
            optimizer.zero_grad()
            # loss = nn.NLLLoss(out, y)  # loss function
            # test_loss += loss.item()  # loss仍然有一个图形副本。在这种情况中，可用.item()来释放它.(提高训练速度技巧)
            pred = out.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
    # test_loss /= (batch_idx + 1)
    # print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loss), 100. * correct / len(test_loss)))
    total_num = len(test_loader) * test_batch_size
    acc = correct / total_num
    print('model_acc', acc)
    # 这里暂时没画图

# 模型训练
def train(model, train_loader, epoch):
    model.train()
    correct = 0
    train_loss = 0
    loss_function = nn.NLLLoss()
    # plt.figure()
    # i = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        #   i += 1
        out = model(x)
        loss = loss_function(out, y)  # loss function
        loss.backward()  # 计算倒数
        optimizer.step()  # w' = w - Ir*grad 模型参数更新
        optimizer.zero_grad()
        # test_loss += loss.item()  # loss仍然有一个图形副本。在这种情况中，可用.item()来释放它.(提高训练速度技巧)
        predict = out.max(1, keepdim=True)[1]
        correct += predict.eq(y.view_as(predict)).sum().item()
        # train_loss.append(loss.item())
        train_loss += loss
    #        plt.plot(i, loss.item())

    total_num = len(train_loader) * train_batch_size
    acc = correct / total_num
    # print('model_acc', acc)
    loss_mean = train_loss / (batch_idx + 1)
    print('Train Epoch: {}\t Loss: {:.6f}\t acc:{:.2f}'.format(epoch, loss_mean.item(), acc))

# 本地化，先test一次，再train几次
def local():
    # model.load_state_dict(torch.load(log_dir))
    # a = torch.load(log_dir)
    # train(model, train_loader, 11)
    test(model, test_loader)
    print('加载 epoch {} 成功！'.format(start_epoch))

    for k, v in model.named_parameters():
        if k != 'classifier.2.weight' and k != 'classifier.2.bias' \
                and k != 'classifier.4.weight' and k != 'classifier.4.bias':
            print(k,'false')
            # v.requires_grad = False  # 固定参数
    epochs = start_epoch + 10
    for epoch in range(start_epoch + 1, epochs):
        train(model, train_loader, epoch)
        test(model, test_loader)
    weight_new=[model.weight1,model.weight2,model.weight3,model.weight4]
    print('about weight:')
    for i in range(4):
        print(torch.equal(weight_new[i],weight_before[i]))
    # 保存模型
    # state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'weight': weight}
    # path = 'global_models/source_models/net_xiao_global_para_local.pkl'  # net1为1D卷积神经网络模型，net2为2D卷积神经网络模型
    # torch.save(state, path)

local()  # 执行本地化
for name, param in model.named_parameters():
    if param.requires_grad:
        print("requires_grad: True ", name)
    else:
        print("requires_grad: False ", name)
# for module in model.children():
#         print('net包含的模块为：\n',module)
#         for p in module.parameters():
#             print('当前module需要学习的参数为：\n',p)
# for name, module in model.named_children():
#     print(name,'fff')