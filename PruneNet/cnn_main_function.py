import os
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from PruneNet.cnn_training_loop import trainingLoop
from PruneNet.prune_lowest import prunelowest

## load mnist dataset
# use_cuda = torch.cuda.is_available()

root = './data'
if not os.path.exists(root):
    os.mkdir(root)

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
# if not exist, download mnist dataset
train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)

batch_size = 200
epochs = 1
log_interval = 100
learning_rate_conv2 = 0.01
learning_rate_conv4 = 0.003

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False)

# print
# '==>>> total trainning batch number: {}'.format(len(train_loader))
# print
# '==>>> total testing batch number: {}'.format(len(test_loader))


class Conv_2(nn.Module):
    def __init__(self):
        super(Conv_2, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5)
        nn.init.normal_(self.conv1.weight, 0, 0.1)
        self.conv2 = nn.Conv2d(64, 64, 5)
        nn.init.normal_(self.conv2.weight, 0, 0.1)
        self.fc1 = nn.Linear(4 * 4 * 64, 384)
        nn.init.normal_(self.fc1.weight, 0, 0.1)
        self.fc2 = nn.Linear(384, 192)
        nn.init.normal_(self.fc2.weight, 0, 0.1)
        self.fc3 = nn.Linear(192, 10)
        nn.init.normal_(self.fc3.weight, 0, 0.1)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def name(self):
        return "conv2"



class Conv_4(nn.Module):
    def __init__(self):
        super(Conv_4, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3)
        nn.init.xavier_normal(self.conv1.weight)
        self.conv2 = nn.Conv2d(64, 64, 3)
        nn.init.xavier_normal(self.conv2.weight)
        self.conv3 = nn.Conv2d(64, 128, 3)
        nn.init.xavier_normal(self.conv3.weight)
        self.conv4 = nn.Conv2d(128, 128, 3)
        nn.init.xavier_normal(self.conv4.weight)
        self.fc1 = nn.Linear(4 * 4 * 128, 256)
        nn.init.xavier_normal(self.fc1.weight)
        self.fc2 = nn.Linear(256, 256)
        nn.init.xavier_normal(self.fc2.weight)
        self.fc3 = nn.Linear(256, 10)
        nn.init.xavier_normal(self.fc3.weight)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 128)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def name(self):
        return "conv4"


## training
net = Conv_2()
prune_way = 'oneShot'
# if use_cuda:
#     model = net.cuda()


times = 5
percents_conv = [0.8, 1.0, 0.8]
percents_fc = [0.8, 0.8, 1.0]


if net.name() == 'conv2':
    path = 'Conv_2_'
    learning_rate = learning_rate_conv2
else:
    path = 'Conv_4_'
    learning_rate = learning_rate_conv4

for time in range(1, times + 1):
    # torch.save(net.state_dict(), path + 'Net/net_' + net.name() +'-100_fc-100_original.pkl')
    # x, y = trainingLoop(net, train_loader, epochs, batch_size, log_interval, learning_rate, test_loader, [], False)
    # torch.save(net.state_dict(), path + 'Net/net_' + net.name() + '-100_fc-100_' + prune_way + '.pkl')
    # np.save(path + 'Datum/x_' + net.name() + '-100_fc-100_' + prune_way + '_' + str(time) + '.npy', x)
    # np.save(path + 'Datum/y_' + net.name() + '-100_fc-100_' + prune_way + '_' + str(time) + '.npy', y)

    for i in range(0, len(percents_conv)):
        net.load_state_dict(torch.load(path + 'Net/net_' + net.name() + '-100_fc-100_' + prune_way + '.pkl'))
        if percents_fc[i] == 1:
            fcn_list = []
        else:
            fcn_list = [net.fc1, net.fc2, net.fc3]

        if percents_conv[i] == 1:
            cnn_list = []
        else:
            if net.name() == 'conv2':
                cnn_list = [net.conv1, net.conv2]
            else:
                cnn_list = [net.conv1, net.conv2, net.conv3, net.conv4]

        netlist = [fcn_list, cnn_list]
        mask = prunelowest(netlist, 0.8, 0.8)
        net.load_state_dict(torch.load(path + 'Net/net_' + net.name() +'-100_fc-100_original.pkl'))

        x, y = trainingLoop(net, train_loader, epochs, batch_size, log_interval, learning_rate, test_loader, mask, True)
        torch.save(net.state_dict(), path + 'Net/net_' + net.name() + '-' + str(round(float(percents_conv[i] * 1000)) / 10)
                   + '_fc-' + str(round(float(percents_fc[i] * 1000)) / 10) + '_' + prune_way + '.pkl')
        np.save(path + 'Datum/x_' + net.name() + '-' + str(round(float(percents_conv[i] * 1000)) / 10)
                   + '_fc-' + str(round(float(percents_fc[i] * 1000)) / 10) + '_' + prune_way + '_' + str(time) + '.npy', x)
        np.save(path + 'Datum/y_' + net.name() + '-' + str(round(float(percents_conv[i] * 1000)) / 10)
                   + '_fc-' + str(round(float(percents_fc[i] * 1000)) / 10) + '_' + prune_way + '_' + str(time) + '.npy', y)



# torch.save(model.state_dict(), model.name())

