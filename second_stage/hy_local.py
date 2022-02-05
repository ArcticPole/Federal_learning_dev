"""
hy_local.py
load model parameters from cloud, and train model with local data.
two training ways:  'weight loaded only', 'model, epoch, weight, optimizer loaded'
some parameters froze to increase training speed and accuracy
"""
way = 1  # 0/1     # [0,1] = ['weight loaded only', 'model, epoch, weight, optimizer loaded']
w = 0  # 0/1      # whether weight parameters frozen, 1 means frozen

# ---------------------------------------------model-----------------------------------------------------

# model
import torch
from torch import nn

# dataset
import tools.xiao_dataset_random as xdr

data = xdr.FlameSet('insert_fault', 2304, '2D', 'try')
train_data_id, test_data_id = data._shuffle()
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# create training and validation sampler objects
tr_sampler = SubsetRandomSampler(train_data_id)  # xiao：生成子数据例
val_sampler = SubsetRandomSampler(test_data_id)
train_batch_size = 50
test_batch_size = 10
train_loader = DataLoader(data, batch_size=train_batch_size, sampler=tr_sampler,
                          shuffle=False)
test_loader = DataLoader(data, batch_size=test_batch_size, sampler=val_sampler,
                         shuffle=False)

# global model loaded from cloud
import model

model = model.cnn2d_hy_merge_para()
# different training way choosing (different training speed and performance
import second_stage.s2_local as s2l
HOST = "127.0.0.1"
PATH_para = s2l.recieve_para(HOST)
#log_para = '../global_models/net_xiao_global_para_hy_2.pkl'
checkpoint = torch.load(PATH_para)
ways = ['weight loaded only', 'model, epoch, weight, optimizer loaded']

if way == 0:
    weight = checkpoint['weight']
    import torch.optim as optim

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, weight_decay=1e-6,
                          momentum=0.9, nesterov=True)
if way == 1:
    model.load_state_dict(checkpoint['model'])
    start_epoch = checkpoint['epoch']
    weight = checkpoint['weight']
    import torch.optim as optim

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, weight_decay=1e-6,
                          momentum=0.9, nesterov=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
print('training way: ', ways[way])


# ---------------------------------------------training-----------------------------------------------------


# test accuracy calculate
def test(model, test_loader):
    correct = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            out = model(x)
            optimizer.zero_grad()
            pred = out.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
    total_num = len(test_loader) * test_batch_size
    acc = correct / total_num
    print('model_acc,test', acc)


# model training
def train(model, train_loader, epoch):
    model.train()
    train_loss = 0
    loss_function = nn.NLLLoss()
    for batch_idx, (x, y) in enumerate(train_loader):
        out = model(x)
        loss = loss_function(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss
    loss_mean = train_loss / (batch_idx + 1)
    print('Train Epoch: {}\t Loss: {:.6f}\t'.format(epoch, loss_mean.item()), end=' ')


# specific parameters frozen so that when training, these parameters not change
# To increase training speed
def parameters_frozen(model):
    for k, v in model.named_parameters():
        if k != 'classifier.2.weight' and k != 'classifier.2.bias' \
                and k != 'classifier.4.weight' and k != 'classifier.4.bias' \
                and k != 'weight1' and k != 'weight2' and k != 'weight3' and k != 'weight4':
            print(k, 'parameter frozen')
            v.requires_grad = False  # parameters frozen


# local training, first calculate test accuracy, then train several times.
def local(w):
    test(model, test_loader)
    start_epoch = 1
    print('加载 epoch {} 成功！'.format(start_epoch))

    # whether weight parameters frozen, 1 means frozen
    if w:
        parameters_frozen(model)

    epochs = start_epoch + 30
    for epoch in range(start_epoch + 1, epochs):
        train(model, train_loader, epoch)
        test(model, test_loader)
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'weight': weight}
    path = 'local_models/net_xiao_global_para_local.pkl'  # net1为1D卷积神经网络模型，net2为2D卷积神经网络模型
    torch.save(state, path)


# ---------------------------------------------check-----------------------------------------------------


def model_parameters_check(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("requires_grad: True ", name)
        else:
            print("requires_grad: False ", name)


def whether_weight_parameters_change(model, checkpoint):
    weight_new = [model.weight1, model.weight2, model.weight3, model.weight4]
    print('about weight:')
    weight_before = checkpoint['weight']
    for i in range(4):
        print('weight'[i],torch.equal(weight_new[i], weight_before[i]))


local(w)
# model_parameters_check  # check model parameters' state
whether_weight_parameters_change(model, checkpoint)  # check whether weight parameters change
