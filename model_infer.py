import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
import numpy as np
from thop import profile
from matplotlib import pyplot as plt
from thop import profile
from scipy.io import loadmat
from quantize import QConv2d, QLinear


class MixedOp(nn.Module):
    def __init__(self, C_in, C_out, op_idx, layer_id, stride=1, num_bits_list=[0, ]):
        super(MixedOp, self).__init__()
        self.layer_id = layer_id
        self._op = OPS[PRIMITIVES[op_idx]](C_in, C_out, layer_id, stride, num_bits_list)

    def forward(self, x, num_bits):
        return self._op(x, num_bits)


    def forward_flops(self, size):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        flops, size_out = self._op.forward_flops(size)

        return flops, size_out


class OpticalConv(nn.Module):
    def __init__(self, mask=6):
        super(OpticalConv, self).__init__()

        self.out_channel = int(mask/2)

        self.filter0 = nn.Conv2d(1, self.out_channel, 7, padding=3, bias=True)
        self.filter1 = nn.Conv2d(1, self.out_channel, 7, padding=3, bias=True)
        self.filter2 = nn.Conv2d(1, self.out_channel, 7, padding=3, bias=True)

    def forward(self, x):
        x0 = self.filter0(x[:, 0: 1, :, :])
        x0_res = torch.sum(x0, dim=1)
        x0_res = x0_res.view(x0_res.shape[0], 1, x0_res.shape[1], x0_res.shape[2])
        x1 = self.filter1(x[:, 1: 2, :, :])
        x1_res = torch.sum(x1, dim=1)
        x1_res = x1_res.view(x1_res.shape[0], 1, x1_res.shape[1], x1_res.shape[2])
        x2 = self.filter2(x[:, 2: 3, :, :])
        x2_res = torch.sum(x2, dim=1)
        x2_res = x2_res.view(x2_res.shape[0], 1, x2_res.shape[1], x2_res.shape[2])
        return torch.cat((x0_res, x1_res, x2_res), 1)

class OpticalConv_fix():
    def __init__(self, mask=6):
        super(OpticalConv_fix, self).__init__()

        self.out_channel = int(mask/2)

        self.filter0 = nn.Conv2d(1, self.out_channel, 7, padding=3, bias=True)
        self.filter1 = nn.Conv2d(1, self.out_channel, 7, padding=3, bias=True)
        self.filter2 = nn.Conv2d(1, self.out_channel, 7, padding=3, bias=True)
        self.filter0 = self.filter0.cuda()
        self.filter1 = self.filter1.cuda()
        self.filter2 = self.filter2.cuda()
        # print(self.filter0.device)

    def forward(self, x):

        x0 = self.filter0(x[:, 0: 1, :, :])
        x0_res = torch.sum(x0, dim=1)
        x0_res = x0_res.view(x0_res.shape[0], 1, x0_res.shape[1], x0_res.shape[2])
        x1 = self.filter1(x[:, 1: 2, :, :])
        x1_res = torch.sum(x1, dim=1)
        x1_res = x1_res.view(x1_res.shape[0], 1, x1_res.shape[1], x1_res.shape[2])
        x2 = self.filter2(x[:, 2: 3, :, :])
        x2_res = torch.sum(x2, dim=1)
        x2_res = x2_res.view(x2_res.shape[0], 1, x2_res.shape[1], x2_res.shape[2])
        # print(self.filter0.weight.data)
        return torch.cat((x0_res, x1_res, x2_res), 1)


class GaborConv(nn.Module):

    def __init__(self, config):
        super(GaborConv, self).__init__()
        self.filter = nn.Conv2d(1, config.mask, 7, padding=3, bias=True)
        gabor = loadmat('gabor.mat')['filts']
        gabor = np.resize(gabor, (7,7,1,10))
        gabor = np.rollaxis(np.rollaxis(gabor, 2), 3, 1)
        gabor = gabor.reshape(10, 1, 7, 7)
        for i in range(config.mask):
            self.filter.weight.data[i] = torch.tensor(gabor)[i]
        # self.filter.weight.data[0] = torch.tensor(gabor)[0]
        # self.filter.weight.data[1] = torch.tensor(gabor)[2]
        # self.filter.weight.data[2] = torch.tensor(gabor)[4]
        # self.filter.weight.data[3] = torch.tensor(gabor)[6]
        # self.filter = self.filter.cuda()
        for param in self.filter.parameters():
            param.requires_grad = False
        print(config.mask)

    def forward(self, x):
        x0 = self.filter(x[:, 0: 1, :, :])
        x1 = self.filter(x[:, 1: 2, :, :])
        x2 = self.filter(x[:, 2: 3, :, :])
        x = x0 + x1 + x2
        return x


class FBNet_Infer(nn.Module):
    def __init__(self, alpha, config):
        super(FBNet_Infer, self).__init__()

        self.config = config

        if config.search_fix:
            self.optical_cnn = OpticalConv_fix(mask=config.mask)
        else:
            self.optical_cnn = OpticalConv(mask=config.mask)

        if config.std == 0:
            checkpoint = torch.load(config.pretrain)

            self.optical_cnn.filter0.weight.data = checkpoint['module.optical_cnn.filter0.weight']
            self.optical_cnn.filter0.bias.data = checkpoint['module.optical_cnn.filter0.bias']

            self.optical_cnn.filter1.weight.data = checkpoint['module.optical_cnn.filter1.weight']
            self.optical_cnn.filter1.bias.data = checkpoint['module.optical_cnn.filter1.bias']

            self.optical_cnn.filter2.weight.data = checkpoint['module.optical_cnn.filter2.weight']
            self.optical_cnn.filter2.bias.data = checkpoint['module.optical_cnn.filter2.bias']
        elif config.std > 0:
            checkpoint = torch.load(config.pretrain)

            std = config.std
            max0 = torch.max(checkpoint['module.optical_cnn.filter0.weight']).cpu().item()
            me0 = torch.median(torch.abs(checkpoint['module.optical_cnn.filter0.weight'])).cpu().item()
            max1 = torch.max(checkpoint['module.optical_cnn.filter1.weight']).cpu().item()
            me1 = torch.median(torch.abs(checkpoint['module.optical_cnn.filter1.weight'])).cpu().item()
            max2 = torch.max(checkpoint['module.optical_cnn.filter2.weight']).cpu().item()
            me2 = torch.median(torch.abs(checkpoint['module.optical_cnn.filter2.weight'])).cpu().item()
            stds = [std * me0, std * me1, std * me2]
            if config.std_use == 'max':
                stds = [std * max0, std * max1, std * max2]
            elif config.std_use == 'between':
                stds = [std * (max0 + me0) / 2, std * (max1 + me1) / 2, std * (max2 + me2) / 2]

            self.optical_cnn.filter0.weight.data = checkpoint['module.optical_cnn.filter0.weight']
            print("std: ", std, max0, stds[0], me0)
            # print(self.optical_cnn.filter0.weight.data)
            noise1 = torch.normal(mean=torch.zeros(list(self.optical_cnn.filter0.weight.data.shape)), std=stds[0]).cuda()
            # print(noise1)
            self.optical_cnn.filter0.weight.data += noise1
            print(self.optical_cnn.filter0.weight.data)
            self.optical_cnn.filter0.bias.data = checkpoint['module.optical_cnn.filter0.bias']

            self.optical_cnn.filter1.weight.data = checkpoint['module.optical_cnn.filter1.weight']
            self.optical_cnn.filter1.weight.data += torch.normal(mean=torch.zeros(list(self.optical_cnn.filter1.weight.data.shape)), std=stds[1]).cuda()
            self.optical_cnn.filter1.bias.data = checkpoint['module.optical_cnn.filter1.bias']

            self.optical_cnn.filter2.weight.data = checkpoint['module.optical_cnn.filter2.weight']
            self.optical_cnn.filter2.weight.data += torch.normal(mean=torch.zeros(list(self.optical_cnn.filter2.weight.data.shape)), std=stds[2]).cuda()
            self.optical_cnn.filter2.bias.data = checkpoint['module.optical_cnn.filter2.bias']



        print('trained mask loaded; search fix:', config.search_fix)

        op_idx_list = F.softmax(alpha, dim=-1).argmax(-1)

        self.num_classes = config.num_classes

        self.num_bits_list = config.num_bits_list

        self.num_layer_list = config.num_layer_list
        self.num_channel_list = config.num_channel_list
        self.stride_list = config.stride_list

        self.stem_channel = config.stem_channel
        self.header_channel = config.header_channel

        self.stem = ConvNorm(3, self.stem_channel, kernel_size=3, stride=1, padding=1, bias=False, num_bits_list=[0, ])

        self.cells = nn.ModuleList()

        layer_id = 1

        for stage_id, num_layer in enumerate(self.num_layer_list):
            for i in range(num_layer):
                if i == 0:
                    if stage_id == 0:
                        op = MixedOp(self.stem_channel, self.num_channel_list[stage_id], op_idx_list[layer_id - 1],
                                     layer_id, stride=self.stride_list[stage_id], num_bits_list=self.num_bits_list)
                    else:
                        op = MixedOp(self.num_channel_list[stage_id - 1], self.num_channel_list[stage_id],
                                     op_idx_list[layer_id - 1], layer_id, stride=self.stride_list[stage_id],
                                     num_bits_list=self.num_bits_list)
                else:
                    op = MixedOp(self.num_channel_list[stage_id], self.num_channel_list[stage_id],
                                 op_idx_list[layer_id - 1], layer_id, stride=1, num_bits_list=self.num_bits_list)

                layer_id += 1
                self.cells.append(op)

        self.header = ConvNorm(self.num_channel_list[-1], self.header_channel, kernel_size=1, num_bits_list=[0, ])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = QLinear(self.header_channel, self.num_classes)

        self._criterion = nn.CrossEntropyLoss().cuda()

    def forward(self, input, num_bits=0):

        if self.config.search_fix:
            input = self.optical_cnn.forward(input)
        else:
            input = self.optical_cnn(input)
        # print(self.optical_cnn.filter2.weight.data)
        # input = self.optical_cnn(input)

        out = self.stem(input, num_bits=0)

        for i, cell in enumerate(self.cells):
            out = cell(out, num_bits)

        out = self.fc(self.avgpool(self.header(out, num_bits=0)).view(out.size(0), -1), num_bits=0)

        return out

    def forward_flops(self, size):

        flops_total = []

        flops, size = self.stem.forward_flops(size)
        flops_total.append(flops)

        for i, cell in enumerate(self.cells):
            flops, size = cell.forward_flops(size)
            flops_total.append(flops)

        flops, size = self.header.forward_flops(size)
        flops_total.append(flops)

        return sum(flops_total)

    def _loss_backward(self, input, target, num_bits_list=None):
        if num_bits_list is None:
            num_bits_list = self.num_bits_list

        loss_val = [-1 for _ in num_bits_list]

        return loss_val
