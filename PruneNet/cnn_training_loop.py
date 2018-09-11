import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


## load mnist dataset
use_cuda = torch.cuda.is_available()


def trainingLoop(net, train_loader, epochs, batch_size, log_interval, learning_rate, test_loader, mask, if_prune):
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()
    xAxis = []
    yAxis = []
    timesPerEpoch = len(train_loader.dataset) / batch_size

    for epoch in range(epochs):
        # trainning
        ave_loss = 0
        timeInEpoch = 1
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            if use_cuda:
                x, target = x.cuda(), target.cuda()
            x, target = Variable(x), Variable(target)
            out = net(x)
            loss = criterion(out, target)
            ave_loss = ave_loss * 0.9 + loss.item() * 0.1
            loss.backward()
            optimizer.step()

            if if_prune:
                mask_fcn = mask[0]
                mask_cnn = mask[1]
                if mask_fcn:
                    net.fc1.weight.data.mul_(mask_fcn[0])
                    net.fc2.weight.data.mul_(mask_fcn[1])
                    net.fc3.weight.data.mul_(mask_fcn[2])
                if mask_cnn:
                    if net.name() == 'conv2':
                        net.conv1.weight.data.mul_(mask_cnn[0])
                        net.conv2.weight.data.mul_(mask_cnn[1])
                    else:
                        net.conv1.weight.data.mul_(mask_cnn[0])
                        net.conv2.weight.data.mul_(mask_cnn[1])
                        net.conv3.weight.data.mul_(mask_cnn[2])
                        net.conv4.weight.data.mul_(mask_cnn[3])

            iteration = epoch * timesPerEpoch + timeInEpoch
            timeInEpoch = timeInEpoch + 1
            if batch_idx % log_interval == 0:

                # run a test loop
                test_loss = 0
                correct = 0
                test = 0
                for data, target in test_loader:
                    if use_cuda:
                        data, target = data.cuda(), target.cuda()
                    with torch.no_grad():
                        data, target = Variable(data), Variable(target)
                    net_out = net(data)
                    # sum up batch loss
                    test_loss += criterion(net_out, target).item()
                    pred = net_out.data.max(1)[1]  # get the index of the max log-probability
                    correct += pred.eq(target.data).sum()

                test_loss /= len(test_loader.dataset)
                accuracy = 1. * correct.item() / len(test_loader.dataset)

                yAxis.append(accuracy)
                xAxis.append(iteration)

                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tIterations: {:.0f}\tAverage loss: {:.4f}\tAccuracy: {}/{} ({:.4f}%)'
                        .format(epoch, batch_idx * len(data), len(train_loader.dataset),
                                100. * batch_idx / len(train_loader), iteration, test_loss, correct,
                                len(test_loader.dataset),
                                100. * accuracy))

    return xAxis, yAxis
    plt.plot(xAxis, yAxis)
    plt.axis([0, 300 * epochs, 0.93, 1])
    plt.show()