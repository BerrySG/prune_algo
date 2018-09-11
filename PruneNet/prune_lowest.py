import torch
import numpy as np
import math


def prunelowest(netlist, percentage_fcn, percentage_cnn):

    fcnlist = netlist[0]
    cnnlist = netlist[1]
    vec_fcn = torch.Tensor()
    vec_cnn = torch.Tensor()
    fcn_sizes = []
    cnn_sizes = []
    for fcns in fcnlist:
        vec_fcn = torch.cat((vec_fcn, fcns.weight.view(-1)), 0)
        fcn_sizes.append([fcns.weight.size()])
    for cnns in cnnlist:
        vec_cnn = torch.cat((vec_cnn, cnns.weight.view(-1)), 0)
        cnn_sizes.append([cnns.weight.size()])
    mask_list_fcn = []
    mask_list_cnn = []
    if len(fcnlist) > 0:
        prunelist_fcn = torch.topk(vec_fcn.abs(), math.floor(vec_fcn.size()[0] * (1 - percentage_fcn)), largest=False)
        pruneindex_fcn = prunelist_fcn[1]
        mask_fcn = np.ones(vec_fcn.size()[0])
        for i in pruneindex_fcn:
            mask_fcn[i] = 0
        mask_fcn = torch.from_numpy(mask_fcn).float()
        for j in range(len(fcnlist)):
            size = fcn_sizes[j][0]
            length = size[0] * size[1]
            if j == len(fcnlist) - 1:
                mask_fcn = mask_fcn.view(size)
                mask_list_fcn.append(mask_fcn)
                break
            mask_cur = mask_fcn[:length]
            mask_left = mask_fcn[length:]
            mask_cur = mask_cur.view(size)
            mask_list_fcn.append(mask_cur)
            mask_fcn = mask_left
    if len(cnnlist) > 0:
        prunelist_cnn = torch.topk(vec_cnn.abs(), math.floor(vec_cnn.size()[0] * (1 - percentage_cnn)), largest=False)
        pruneindex_cnn = prunelist_cnn[1]
        mask_cnn = np.ones(vec_cnn.size()[0])
        for i in pruneindex_cnn:
            mask_cnn[i] = 0
        mask_cnn = torch.from_numpy(mask_cnn).float()
        for j in range(len(cnnlist)):
            size = cnn_sizes[j][0]
            length = size[0] * size[1] * size[2] * size[3]
            if j == len(cnnlist) - 1:
                mask_cnn = mask_cnn.view(size)
                mask_list_cnn.append(mask_cnn)
                break
            mask_cur = mask_cnn[:length]
            mask_left = mask_cnn[length:]
            mask_cur = mask_cur.view(size)
            mask_list_cnn.append(mask_cur)
            mask_cnn = mask_left
    return [mask_list_fcn, mask_list_cnn]


def prunelowest_full(net, percentage):

    weight_1 = net.fc1.weight
    weight_2 = net.fc2.weight
    weight_3 = net.fc3.weight
    weights = torch.cat((weight_1.view(-1), weight_2.view(-1)), 0)
    weights = torch.cat((weights, weight_3.view(-1)), 0)
    length_1 = weight_1.size()[0] * weight_1.size()[1]
    length_2 = weight_2.size()[0] * weight_2.size()[1]
    length_3 = weight_3.size()[0] * weight_3.size()[1]
    vec_all = weights.view(-1)
    prunelist = torch.topk(vec_all.abs(), math.floor(vec_all.size()[0] * percentage), largest=False)
    pruneindex = prunelist[1]
    mask = np.ones(vec_all.size()[0])
    for i in pruneindex:
        mask[i] = 0
    mask = torch.from_numpy(mask).float()
    [mask_1, mask_2, mask_3] = torch.split(mask, [length_1, length_2, length_3], 0)
    mask_1 = mask_1.view(weight_1.size()[0], weight_1.size()[1])
    mask_2 = mask_2.view(weight_2.size()[0], weight_2.size()[1])
    mask_3 = mask_3.view(weight_3.size()[0], weight_3.size()[1])
    res = [mask_1, mask_2, mask_3]
    return res


def prunelowest_conv(net, percentage):

    weight_list = net.weight
    weight_1 = net.fc1.weight
    weight_2 = net.fc2.weight
    weight_3 = net.fc3.weight
    weights = torch.cat((weight_1.view(-1), weight_2.view(-1)), 0)
    weights = torch.cat((weights, weight_3.view(-1)), 0)
    length_1 = weight_1.size()[0] * weight_1.size()[1]
    length_2 = weight_2.size()[0] * weight_2.size()[1]
    length_3 = weight_3.size()[0] * weight_3.size()[1]
    vec_all = weights.view(-1)
    prunelist = torch.topk(vec_all.abs(), math.floor(vec_all.size()[0] * percentage), largest=False)
    pruneindex = prunelist[1]
    mask = np.ones(vec_all.size()[0])
    for i in pruneindex:
        mask[i] = 0
    mask = torch.from_numpy(mask).float()
    [mask_1, mask_2, mask_3] = torch.split(mask, [length_1, length_2, length_3], 0)
    mask_1 = mask_1.view(weight_1.size()[0], weight_1.size()[1])
    mask_2 = mask_2.view(weight_2.size()[0], weight_2.size()[1])
    mask_3 = mask_3.view(weight_3.size()[0], weight_3.size()[1])
    res = [mask_1, mask_2, mask_3]
    return res


def prunelowest__(netlist, percentage_fcn, percentage_cnn):

    fcnlist = netlist[0]
    cnnlist = netlist[1]
    vec_fcn = torch.Tensor()
    vec_cnn = torch.Tensor()
    fcn_sizes = []
    cnn_sizes = []
    for fcns in fcnlist:
        vec_fcn = torch.cat((vec_fcn, fcns.weight.view(-1)), 0)
        fcn_sizes.append([fcns.weight.size()])
    for cnns in cnnlist:
        vec_cnn = torch.cat((vec_cnn, cnns.weight.view(-1)), 0)
        cnn_sizes.append([cnns.weight.size()])
    mask_list_fcn = []
    mask_list_cnn = []
    if len(fcnlist) > 0:
        prunelist_fcn = torch.topk(vec_fcn.abs(), math.floor(vec_fcn.size()[0] * (1 - percentage_fcn)), largest=False)
        pruneindex_fcn = prunelist_fcn[1]
        mask_fcn = np.ones(vec_fcn.size()[0])
        for i in pruneindex_fcn:
            mask_fcn[i] = 0
        mask_fcn = torch.from_numpy(mask_fcn).float()
        for j in range(len(fcnlist)):
            size = fcn_sizes[j][0]
            length = size[0] * size[1]
            if j == len(fcnlist) - 1:
                mask_fcn = mask_fcn.view(size)
                mask_list_fcn.append(mask_fcn)
                break
            mask_cur = mask_fcn[:length]
            mask_left = mask_fcn[length:]
            mask_cur = mask_cur.view(size)
            mask_list_fcn.append(mask_cur)
            mask_fcn = mask_left
    if len(cnnlist) > 0:
        prunelist_cnn = torch.topk(vec_cnn.abs(), math.floor(vec_cnn.size()[0] * (1 - percentage_cnn)), largest=False)
        pruneindex_cnn = prunelist_cnn[1]
        mask_cnn = np.ones(vec_cnn.size()[0])
        for i in pruneindex_cnn:
            mask_cnn[i] = 0
        mask_cnn = torch.from_numpy(mask_cnn).float()
        for j in range(len(cnnlist)):
            size = cnn_sizes[j][0]
            length = size[0] * size[1] * size[2] * size[3]
            if j == len(cnnlist) - 1:
                mask_cnn = mask_cnn.view(size)
                mask_list_cnn.append(mask_cnn)
                break
            mask_cur = mask_cnn[:length]
            mask_left = mask_cnn[length:]
            mask_cur = mask_cur.view(size)
            mask_list_cnn.append(mask_cur)
            mask_cnn = mask_left
    return [mask_list_fcn, mask_list_cnn]
