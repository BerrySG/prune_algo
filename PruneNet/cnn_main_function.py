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
use_cuda = torch.cuda.is_available()

root = './data'
if not os.path.exists(root):
    os.mkdir(root)

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
# if not exist, download mnist dataset
train_set = dset.CIFAR10(root=root, train=True, transform=trans, download=True)
test_set = dset.CIFAR10(root=root, train=True, transform=trans, download=True)


# train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
# test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)

batch_size = 50
epochs = 30
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
        self.conv1 = nn.Conv2d(3, 64, 5)
        nn.init.normal_(self.conv1.weight, 0, 0.1)
        self.conv2 = nn.Conv2d(64, 64, 5)
        nn.init.normal_(self.conv2.weight, 0, 0.1)
        self.fc1 = nn.Linear(5 * 5 * 64, 384)
        nn.init.normal_(self.fc1.weight, 0, 0.1)
        self.fc2 = nn.Linear(384, 192)
        nn.init.normal_(self.fc2.weight, 0, 0.1)
        self.fc3 = nn.Linear(192, 10)
        nn.init.normal_(self.fc3.weight, 0, 0.1)
        self.soft = nn.LogSoftmax(dim=1)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5 * 5 * 64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.soft(x)
        return x

    def name(self):
        return "conv2"



class Conv_4(nn.Module):
    def __init__(self):
        super(Conv_4, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        nn.init.xavier_normal(self.conv1.weight)
        self.conv2 = nn.Conv2d(64, 64, 3)
        nn.init.xavier_normal(self.conv2.weight)
        self.conv3 = nn.Conv2d(64, 128, 3)
        nn.init.xavier_normal(self.conv3.weight)
        self.conv4 = nn.Conv2d(128, 128, 3)
        nn.init.xavier_normal(self.conv4.weight)
        self.fc1 = nn.Linear(5 * 5 * 128, 256)
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
        x = x.view(-1, 5 * 5 * 128)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def name(self):
        return "conv4"


## training
# net = Conv_2()
prune_way = 'iterative'
# if use_cuda:
#     model = net.cuda()







def Pruning(net, path, cnn_percent, fcn_percent, time, iter_times):
    net_name = net.name()
    for iter_time in range(1, iter_times + 1):
        print('iter_time = ' + str(iter_time))
        # trainingLoop(net, train_loader, epochs, batch_size, log_interval, learning_rate, test_loader, [], False)
        # torch.save(net.state_dict(), path + 'Net/net_' + net.name() + '-100_fc-100_' + '.pkl')
        net.load_state_dict(torch.load(path + 'Net/net_' + net.name() + '-100_fc-100_iter-0.pkl'))
        for i in range(iter_time):
            if fcn_percent == 1:
                fcn_list = []
            else:
                fcn_list = [net.fc1, net.fc2, net.fc3]

            if cnn_percent == 1:
                cnn_list = []
            else:
                if net.name() == 'conv2':
                    cnn_list = [net.conv1, net.conv2]
                else:
                    cnn_list = [net.conv1, net.conv2, net.conv3, net.conv4]

            netlist = [fcn_list, cnn_list]
            mask = prunelowest(netlist, fcn_percent, cnn_percent)
            X, Y = trainingLoop(net, train_loader, epochs, batch_size, log_interval, learning_rate, test_loader, mask, True)
        cnn_rate = str(round(float(cnn_percent * 1000)) / 10)
        fcn_rate = str(round(float(fcn_percent * 1000)) / 10)
        torch.save(net.state_dict(), path + 'Net/net_' + net_name + '-' + cnn_rate + '_fc-' + fcn_rate + 'iter-' + str(iter_time) + '.pkl')
        np.save(path + 'Datum/x_' + net_name + '-' + cnn_rate + '_fc-' + fcn_rate + '_' + 'iter-' + str(iter_time) + '_' + str(time) + '.npy', X)
        np.save(path + 'Datum/Y_' + net_name + '-' + cnn_rate + '_fc-' + fcn_rate + '_' + 'iter-' + str(iter_time) + '_' + str(time) + '.npy', Y)


# times = 5
iterative_maxTimes = 25

log_interval = 100
learning_rate_conv2 = 0.01
learning_rate_conv4 = 0.003

percents_conv = [0.8, 1, 0.8]
percents_fc = [0.8, 0.8, 1]

# net = Conv_4()
#
# if net.name() == 'conv2':
#     path = 'Conv_2_'
#     batch_size = 50
#     epochs = 30
#     learning_rate = learning_rate_conv2
# else:
#     path = 'Conv_4_'
#     batch_size = 50
#     epochs = 20
#     learning_rate = learning_rate_conv4

if __name__ == "__main__":

    times = 5

    net = Conv_2()
    if use_cuda:
        net = net.cuda()
    path = 'Conv_2_'
    learning_rate = learning_rate_conv2
    for time in range(1, times + 1):
        X, Y = trainingLoop(net, train_loader, epochs, batch_size, log_interval, learning_rate, test_loader, [], False)
        # print(X)
        # print(Y)
        np.save(path + 'Datum/x_' + net.name() + '-100_fc-100_iter-0_' + str(time) + '.npy', X)
        np.save(path + 'Datum/y_' + net.name() + '-100_fc-100_iter-0_' + str(time) + '.npy', Y)
        # x = np.array([1,2,3])
        # np.save(path + 'Datum/x_' + net.name() + '-test_' + str(time) + '.npy', x)
        torch.save(net.state_dict(), path + 'Net/net_' + net.name() + '-100_fc-100_iter-0' + '.pkl')
        # net.load_state_dict(torch.load(path + 'Net/net_' + net.name() + '-100_fc-100_iter-0.pkl'))
        for i in range(0, len(percents_conv)):
            Pruning(net, path, percents_conv[i], percents_fc[i], time, iterative_maxTimes)

    net = Conv_4()
    path = 'Conv_4_'
    learning_rate = learning_rate_conv4
    for time in range(1, times + 1):
        X, Y = trainingLoop(net, train_loader, epochs, batch_size, log_interval, learning_rate, test_loader, [], False)
        np.save(path + 'Datum/x_' + net.name() + '-100_fc-100_iter-0_' + str(time) + '.npy', X)
        np.save(path + 'Datum/y_' + net.name() + '-100_fc-100_iter-0_' + str(time) + '.npy', Y)
        torch.save(net.state_dict(), path + 'Net/net_' + net.name() + '-100_fc-100_iter-0' + '.pkl')
        # net.load_state_dict(torch.load(path + 'Net/net_' + net.name() + '-100_fc-100_iter-0.pkl'))
        for i in range(0, len(percents_conv)):
            Pruning(net, path, percents_conv[i], percents_fc[i], time, iterative_maxTimes)
