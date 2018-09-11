import numpy as np
import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [0.95, 0.96, 0.97, 0.98]
plt.plot(x, y)
# plt.show()

a = np.array([1, 2, 3, 4])
b = np.array([0.95, 0.96, 0.97, 0.98])
plt.plot(a, b)
# plt.show()

# np.save('a.npy', a)
# c = np.load('a.npy')

# print(c)


# x = np.load('xFullConnect.npy')
# y = np.load('yFullConnect.npy')
# # print('test')
# plt.plot(x, y)
# plt.axis([0, 18000, 0.93, 0.99])
# plt.show()

a = [1,1,1]
b = [2,2,2]

x = [[]]
x.append(a)
x.append(b)
x[0].append(a)
x[1].append(b)

# print(x)

# d = np.load('x_35.1_oneShot_1.npy')

x = np.zeros(shape=(2, 3))
a = np.array([1,1,1])
b = np.array([2,2,2])


x[0] = a
x[1] = b
np.concatenate(x[0],a)
np.concatenate(x[1],b)


# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# criterion = nn.CrossEntropyLoss()
#
# xAxis = []
# yAxis = []
# timesPerEpoch = len(train_loader.dataset)/batch_size
#
# for epoch in range(epochs):
#     # trainning
#     ave_loss = 0
#     timeInEpoch = 1
#     for batch_idx, (x, target) in enumerate(train_loader):
#         optimizer.zero_grad()
#         if use_cuda:
#             x, target = x.cuda(), target.cuda()
#         x, target = Variable(x), Variable(target)
#         out = model(x)
#         loss = criterion(out, target)
#         ave_loss = ave_loss * 0.9 + loss.item() * 0.1
#         loss.backward()
#         optimizer.step()
#         iteration = epoch * timesPerEpoch + timeInEpoch
#         timeInEpoch = timeInEpoch + 1
#         if batch_idx % 100 == 0:
#
#             # run a test loop
#             test_loss = 0
#             correct = 0
#             test = 0
#             for data, target in test_loader:
#                 if use_cuda:
#                     data, target = data.cuda(), target.cuda()
#                 with torch.no_grad():
#                     data, target = Variable(data), Variable(target)
#                 net_out = model(data)
#                 # sum up batch loss
#                 test_loss += criterion(net_out, target).item()
#                 pred = net_out.data.max(1)[1]  # get the index of the max log-probability
#                 correct += pred.eq(target.data).sum()
#
#
#             test_loss /= len(test_loader.dataset)
#             accuracy = 1. * correct.item() / len(test_loader.dataset)
#
#             yAxis.append(accuracy)
#             xAxis.append(iteration)
#
#             print(
#                 'Train Epoch: {} [{}/{} ({:.0f}%)]\tIterations: {:.0f}\tAverage loss: {:.4f}\tAccuracy: {}/{} ({:.4f}%)'
#                 .format(epoch, batch_idx * len(data), len(train_loader.dataset),
#                         100. * batch_idx / len(train_loader), iteration, test_loss, correct, len(test_loader.dataset),
#                         100. * accuracy))
#
#
# plt.plot(xAxis, yAxis)
# plt.axis([0, 300 * epochs, 0.93, 1])
# plt.show()