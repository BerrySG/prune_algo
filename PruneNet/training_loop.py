import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim


def trainingLoop(net, train_loader, epochs, batch_size, log_interval, learning_rate, test_loader, mask, if_prune):

    # create a stochastic gradient descent optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0)
    # create a loss function
    criterion = nn.NLLLoss()
    # one shot pruning
    if if_prune:
        net.fc1.weight.data.mul_(mask[0])
        net.fc2.weight.data.mul_(mask[1])
        net.fc3.weight.data.mul_(mask[2])
    # end of one shot pruning
    xAxis = []
    yAxis = []
    timesPerEpoch = len(train_loader.dataset)/batch_size
    for epoch in range(epochs):

        timeInEpoch = 1
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
            data = data.view(-1, 28*28)
            optimizer.zero_grad()
            net_out = net(data)
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
            # if if_prune:
            #     net.fc1.weight.data.mul_(mask[0].detach())
            #     net.fc2.weight.data.mul_(mask[1].detach())
            #     net.fc3.weight.data.mul_(mask[2].detach())
            iteration = epoch*timesPerEpoch + timeInEpoch
            timeInEpoch = timeInEpoch + 1
            if batch_idx % log_interval == 0:
                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tIterations: {}'
                #       .format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item(), iteration))

                # run a test loop
                test_loss = 0
                correct = 0
                test = 0
                for data, target in test_loader:
                    with torch.no_grad():
                        data, target = Variable(data), Variable(target)
                    data = data.view(-1, 28 * 28)
                    net_out = net(data)
                    # sum up batch loss
                    # test_loss += criterion(net_out, target).data[0]
                    test_loss += criterion(net_out, target).item()
                    pred = net_out.data.max(1)[1]  # get the index of the max log-probability
                    correct += pred.eq(target.data).sum()
                    # test = test + 1
                    # print('test = %d', test)

                test_loss /= len(test_loader.dataset)
                accuracy = 1.*correct.item() / len(test_loader.dataset)

                yAxis.append(accuracy)
                xAxis.append(iteration)

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tIterations: {:.0f}\tAverage loss: {:.4f}\tAccuracy: {}/{} ({:.4f}%)'
                      .format(epoch, batch_idx * len(data), len(train_loader.dataset),
                              100. * batch_idx / len(train_loader), iteration,test_loss, correct, len(test_loader.dataset), 100. * accuracy))
                # print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tIterations: {}\tAverage loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.data[0], epoch*timesPerEpoch + timeInEpoch,
                #      test_loss, correct, len(test_loader.dataset),
                #      100. * correct / len(test_loader.dataset)))
    return np.asarray(xAxis), np.asarray(yAxis)