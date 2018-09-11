import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PruneNet.prune_lowest import prunelowest_full
from PruneNet.training_loop import trainingLoop


def simple_gradient():
     # print the gradient of 2x^2 + 5x
     x = Variable(torch.ones(2, 2) * 2, requires_grad=True)
     z = 2 * (x * x) + 5 * x
     # run the backpropagation
     z.backward(torch.ones(2, 2))
     print(x.grad)


def create_nn(batch_size=200, learning_rate=0.05, epochs=60,
              log_interval=100,):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(28 * 28, 300)
            self.relu_fc1 = nn.ReLU(inplace=True)
            nn.init.normal_(self.fc1.weight, 0, 0.1)
            self.fc2 = nn.Linear(300, 100)
            self.relu_fc2 = nn.ReLU(inplace=True)
            nn.init.normal_(self.fc2.weight, 0, 0.1)
            self.fc3 = nn.Linear(100, 10)
            nn.init.normal_(self.fc3.weight, 0, 0.1)

        def forward(self, x):
            # x = x.view(x.size(0), 28 * 28)
            # x = torch.sigmoid(self.fc1(x))
            # x = torch.sigmoid(self.fc2(x))
            # x = F.relu(self.fc1(x))
            # x = F.relu(self.fc2(x))
            x = self.fc1(x)
            x = self.relu_fc1(x)
            x = self.fc2(x)
            x = self.relu_fc2(x)
            x = self.fc3(x)
            return F.log_softmax(x, dim=1)

    net = Net()


    # torch.save(net.state_dict(), 'test1_TrainedNet/net_100_original.pkl')
    # net2 = Net()
    # net2.load_state_dict(torch.load('net_Full.pkl'))

    # training full connected network and save data
    #
    # net.load_state_dict(torch.load('test_TrainedNet/net_100_oneShot.pkl'))
    x, y = trainingLoop(net, train_loader, epochs, batch_size, log_interval, learning_rate, test_loader, [], False)
    # torch.save(net.state_dict(), 'test1_TrainedNet/net_100_oneShot.pkl')
    # np.save('test1_Datum/x_100_oneShot_' + str(time)+'.npy', x)
    # np.save('test1_Datum/y_100_oneShot_' + str(time)+'.npy', y)


    # load trained network and data
    # net.load_state_dict(torch.load('test_TrainedNet/net_100_oneShot.pkl'))
    # x_0 = np.load('test_Datum/x_100_oneShot_1.npy')
    # y_0 = np.load('test_Datum/y_100_oneShot_1.npy')

    percents = [0.90, 0.75, 0.351]
    for i in range(0, len(percents)):
        net.load_state_dict(torch.load('test1_TrainedNet/net_100_oneShot.pkl'))
        print(percents[i])
        mask = prunelowest_full(net, percents[i])
        net.load_state_dict(torch.load('test1_TrainedNet/net_100_original.pkl'))
        x, y = trainingLoop(net, train_loader, epochs, batch_size, log_interval, learning_rate, test_loader, mask, True)
        torch.save(net.state_dict(), 'test1_TrainedNet/net_'+str(percents[i]*100) + '_oneShot.pkl')
        np.save('test1_Datum/x_' + str(percents[i]*100) + '_oneShot_' + str(time) + '.npy', x)
        np.save('test1_Datum/y_' + str(percents[i]*100) + '_oneShot_' + str(time) + '.npy', y)

    # # prune to 99%
    # net.load_state_dict(torch.load('test_TrainedNet/net_100_oneShot.pkl'))
    # mask = prunelowest_full(net, 0.90)
    # net.load_state_dict(torch.load('test_TrainedNet/net_100_original.pkl'))
    # x_1, y_1 = trainingLoop(net, train_loader, epochs, batch_size, log_interval, learning_rate, test_loader, mask, True)
    # torch.save(net.state_dict(), 'test_TrainedNet/net_90_oneShot.pkl')
    # np.save('test_Datum/x_90_oneShot_' + str(i)+'.npy', x_1)
    # np.save('test_Datum/y_90_oneShot_' + str(i)+'.npy', y_1)
    #
    # # prune to 98%
    # net.load_state_dict(torch.load('test_TrainedNet/net_100_oneShot.pkl'))
    # mask = prunelowest_full(net, 0.75)
    # net.load_state_dict(torch.load('test_TrainedNet/net_100_original.pkl'))
    #
    # x_2, y_2 = trainingLoop(net, train_loader, epochs, batch_size, log_interval, learning_rate, test_loader, mask, True)
    # torch.save(net.state_dict(), 'test_TrainedNet/net_75_oneShot.pkl')
    # np.save('test_Datum/x_75_oneShot_' + str(i)+'.npy', x_2)
    # np.save('test_Datum/y_75_oneShot_' + str(i)+'.npy', y_2)
    #
    # # prune to 97%
    # net.load_state_dict(torch.load('test_TrainedNet/net_100_oneShot.pkl'))
    # mask = prunelowest_full(net, 0.35)
    # net.load_state_dict(torch.load('test_TrainedNet/net_100_original.pkl'))
    #
    # x_3, y_3 = trainingLoop(net, train_loader, epochs, batch_size, log_interval, learning_rate, test_loader, mask, True)
    # torch.save(net.state_dict(), 'test_TrainedNet/net_35_oneShot.pkl')
    # np.save('test_Datum/x_35_oneShot_' + str(i)+'.npy', x_3)
    # np.save('test_Datum/y_35_oneShot_' + str(i)+'.npy', y_3)

    # plt.plot(x_0, y_0, x_1, y_1, x_2, y_2, x_3, y_3)
    # # plt.axis([0, 202, 0.05, 1])
    # plt.axis([0, 300 * epochs, 0.93, 0.99])
    # plt.show()




if __name__ == "__main__":
    run_opt = 2
    if run_opt == 1:
        simple_gradient()
    elif run_opt == 2:
        for time in range(1, 6):
            # print('data_' + str(i) +'.npy')
            create_nn()
