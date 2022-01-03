from pdb import set_trace as bp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from thop import profile
from thop.count_hooks import count_convNd
import sys
import os.path as osp
from easydict import EasyDict as edict
from quantize import QConv2d
from slimmable_ops import USBatchNorm2d

__all__ = ['ConvBlock', 'Skip','ConvNorm', 'OPS']

flops_lookup_table = {}
flops_file_name = "flops_lookup_table.npy"
if osp.isfile(flops_file_name):
    flops_lookup_table = np.load(flops_file_name, allow_pickle=True).item()

Conv2d = QConv2d
BatchNorm2d = USBatchNorm2d

DWS_CHWISE_QUANT = True

custom_ops = {QConv2d: count_convNd}


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        assert (
            C % g == 0
        ), "Incompatible group size {} for input channel {}".format(g, C)
        return (
            x.view(N, g, int(C / g), H, W)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(N, C, H, W)
        )


class ConvBlock(nn.Module):
    '''
    conv => norm => activation
    use native Conv2d, not slimmable
    '''
    def __init__(self, C_in, C_out,  layer_id, expansion=1, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False, num_bits_list=[0,]):
        super(ConvBlock, self).__init__()
        self.C_in = C_in
        self.C_out = C_out

        self.layer_id = layer_id
        self.num_bits_list = num_bits_list

        assert type(expansion) == int
        self.expansion = expansion
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        if padding is None:
            # assume h_out = h_in / s
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.))
        else:
            self.padding = padding
        self.dilation = dilation
        assert type(groups) == int
        self.groups = groups
        self.bias = bias

        if self.groups > 1:
            self.shuffle = ChannelShuffle(self.groups)

        self.conv1 = Conv2d(C_in, C_in*expansion, kernel_size=1, stride=1, padding=0, dilation=1, groups=self.groups, bias=bias)
        self.bn1 = BatchNorm2d(C_in*expansion, self.num_bits_list)

        self.conv2 = Conv2d(C_in*expansion, C_in*expansion, kernel_size=self.kernel_size, stride=self.stride, 
                            padding=self.padding, dilation=1, groups=C_in*expansion, bias=bias, dws=True and DWS_CHWISE_QUANT)
        self.bn2 = BatchNorm2d(C_in*expansion, self.num_bits_list)

        self.conv3 = Conv2d(C_in*expansion, C_out, kernel_size=1, stride=1, padding=0, dilation=1, groups=self.groups, bias=bias)
        self.bn3 = BatchNorm2d(C_out, self.num_bits_list)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x, num_bits=0):
        identity = x
        x = self.relu(self.bn1(self.conv1(x, num_bits), num_bits))

        if self.groups > 1:
            x = self.shuffle(x)

        x = self.relu(self.bn2(self.conv2(x, num_bits), num_bits))
        x = self.bn3(self.conv3(x, num_bits), num_bits)

        if self.C_in == self.C_out and self.stride == 1:
            x += identity

        # x = self.relu(x)

        return x

    @staticmethod
    def _flops(h, w, C_in, C_out, expansion=1, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer_id = 1
        layer = ConvBlock(C_in, C_out, layer_id, expansion, kernel_size, stride, padding, dilation, groups, bias)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), custom_ops=custom_ops)
        return flops
    

    def forward_flops(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "ConvBlock_H%d_W%d_Cin%d_Cout%d_exp_%dkernel%d_stride%d_group%d"%(h_in, w_in, c_in, c_out, self.expansion, self.kernel_size, self.stride, self.groups)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = ConvBlock._flops(h_in, w_in, c_in, c_out, self.expansion, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
            flops_lookup_table[name] = flops
            np.save(flops_file_name, flops_lookup_table)

        return flops, (c_out, h_out, w_out)


class Skip(nn.Module):
    def __init__(self, C_in, C_out, layer_id, stride=1, num_bits_list=[0,]):
        super(Skip, self).__init__()
        assert stride in [1, 2]
        assert C_out % 2 == 0, 'C_out=%d'%C_out
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride

        self.layer_id = layer_id
        self.num_bits_list = num_bits_list

        self.kernel_size = 1
        self.padding = 0

        if stride == 2 or C_in != C_out:
            self.conv = Conv2d(C_in, C_out, 1, stride=stride, padding=0, bias=False)
            self.bn = BatchNorm2d(C_out, self.num_bits_list)
            self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _flops(h, w, C_in, C_out, stride=1):
        layer = Skip(C_in, C_out, stride)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), custom_ops=custom_ops)
        return flops


    def forward_flops(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "Skip_H%d_W%d_Cin%d_Cout%d_stride%d"%(h_in, w_in, c_in, c_out, self.stride)

        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = Skip._flops(h_in, w_in, c_in, c_out, self.stride)
            flops_lookup_table[name] = flops
            np.save(flops_file_name, flops_lookup_table)

        return flops, (c_out, h_out, w_out)


    def forward(self, x, num_bits=0):
        if hasattr(self, 'conv'):
            out = self.conv(x, num_bits)
            out = self.bn(out, num_bits)
            out = self.relu(out)
        else:
            out = x

        return out


class ConvNorm(nn.Module):
    '''
    conv => norm => activation
    use native Conv2d, not slimmable
    '''
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False, num_bits_list=[0,]):
        super(ConvNorm, self).__init__()
        self.C_in = C_in
        self.C_out = C_out

        self.num_bits_list = num_bits_list

        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        if padding is None:
            # assume h_out = h_in / s
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.))
        else:
            self.padding = padding
        self.dilation = dilation
        assert type(groups) == int
        self.groups = groups
        self.bias = bias

        self.conv = Conv2d(C_in, C_out, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, 
                            dilation=self.dilation, groups=self.groups, bias=bias)
        self.bn = BatchNorm2d(C_out, self.num_bits_list)

        self.relu = nn.ReLU(inplace=True)



    def forward(self, x, num_bits=0):
        x = self.relu(self.bn(self.conv(x, num_bits), num_bits))

        return x

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer = ConvNorm(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), custom_ops=custom_ops)
        return flops


    def forward_flops(self, size):
        c_in, h_in, w_in = size

        # assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "ConvNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d_group%d"%(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.groups)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = ConvNorm._flops(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
            flops_lookup_table[name] = flops
            np.save(flops_file_name, flops_lookup_table)

        return flops, (c_out, h_out, w_out)


OPS = {
    'k3_e1' : lambda C_in, C_out, layer_id, stride, num_bits_list: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=stride, groups=1, num_bits_list=num_bits_list),
    'k3_e1_g2' : lambda C_in, C_out, layer_id, stride, num_bits_list: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=stride, groups=2, num_bits_list=num_bits_list),
    'k3_e3' : lambda C_in, C_out, layer_id, stride, num_bits_list: ConvBlock(C_in, C_out, layer_id, expansion=3, kernel_size=3, stride=stride, groups=1, num_bits_list=num_bits_list),
    'k3_e6' : lambda C_in, C_out, layer_id, stride, num_bits_list: ConvBlock(C_in, C_out, layer_id, expansion=6, kernel_size=3, stride=stride, groups=1, num_bits_list=num_bits_list),
    'k5_e1' : lambda C_in, C_out, layer_id, stride, num_bits_list: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=5, stride=stride, groups=1, num_bits_list=num_bits_list),
    'k5_e1_g2' : lambda C_in, C_out, layer_id, stride, num_bits_list: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=5, stride=stride, groups=2, num_bits_list=num_bits_list),
    'k5_e3' : lambda C_in, C_out, layer_id, stride, num_bits_list: ConvBlock(C_in, C_out, layer_id, expansion=3, kernel_size=5, stride=stride, groups=1, num_bits_list=num_bits_list),
    'k5_e6' : lambda C_in, C_out, layer_id, stride, num_bits_list: ConvBlock(C_in, C_out, layer_id, expansion=6, kernel_size=5, stride=stride, groups=1, num_bits_list=num_bits_list),
    'skip' : lambda C_in, C_out, layer_id, stride, num_bits_list: Skip(C_in, C_out, layer_id, stride, num_bits_list=num_bits_list)
}

