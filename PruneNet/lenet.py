import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PruneNet.prune_lowest import prunelowest

def simple_gradient():
     # print the gradient of 2x^2 + 5x
     x = Variable(torch.ones(2, 2) * 2, requires_grad=True)
     z = 2 * (x * x) + 5 * x
     # run the backpropagation
     z.backward(torch.ones(2, 2))
     print(x.grad)

def create_nn(batch_size=200, learning_rate=0.01, epochs=60,
              log_interval=100):

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
            self.fc2 = nn.Linear(300, 100)
            self.fc3 = nn.Linear(100, 10)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return F.log_softmax(x)

    net = Net().cuda()
    print(net)

    # create a stochastic gradient descent optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # create a loss function
    criterion = nn.NLLLoss().cuda()

    # run the main training loop
    xAxis = []
    yAxis = []
    timesPerEpoch = len(train_loader.dataset)/batch_size
    for epoch in range(epochs):
        timeInEpoch = 1
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data).cuda(), Variable(target).cuda()
            # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
            data = data.view(-1, 28*28)
            optimizer.zero_grad()
            net_out = net(data)
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
            iteration = epoch*timesPerEpoch + timeInEpoch
            timeInEpoch = timeInEpoch + 1
            if batch_idx % log_interval == 0:
                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tIterations: {}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.data[0], epoch*timesPerEpoch + timeInEpoch))

                # run a test loop
                test_loss = 0
                correct = 0
                test = 0
                for data, target in test_loader:
                    data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda()
                    data = data.view(-1, 28 * 28).cuda()
                    net_out = net(data)
                    # sum up batch loss
                    test_loss += criterion(net_out, target).data[0]
                    pred = net_out.data.max(1)[1]  # get the index of the max log-probability
                    correct += pred.eq(target.data).sum()
                    # test = test + 1
                    # print('test = %d', test)

                test_loss /= len(test_loader.dataset)
                accuracy = 1.*correct.item() / len(test_loader.dataset)

                yAxis.append(accuracy)
                xAxis.append(iteration)

                print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                     test_loss, correct, len(test_loader.dataset),
                     100. * correct / len(test_loader.dataset)))

    plt.plot(xAxis, yAxis)
    plt.axis([0, 18000, 0.93, 0.99])
    plt.show()
    prunelowest(net.fc1, 0.351)
    prunelowest(net.fc2, 0.351)
    prunelowest(net.fc3, 0.351)
    # # run a test loop
    # test_loss = 0
    # correct = 0
    # test = 0
    # for data, target in test_loader:
    #     data, target = Variable(data, volatile=True), Variable(target)
    #     data = data.view(-1, 28 * 28)
    #     net_out = net(data)
    #     # sum up batch loss
    #     test_loss += criterion(net_out, target).data[0]
    #     pred = net_out.data.max(1)[1]  # get the index of the max log-probability
    #     correct += pred.eq(target.data).sum()
    #     # test = test + 1
    #     # print('test = %d', test)



    # test_loss /= len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    run_opt = 2
    if run_opt == 1:
        simple_gradient()
    elif run_opt == 2:
        create_nn()