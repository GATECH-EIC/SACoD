import torch.nn as nn
import torch.nn.functional as F


class USBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, num_bits_list=[0, ]):
        super(USBatchNorm2d, self).__init__(num_features)

        self.num_features = num_features
        self.num_bits_list = num_bits_list
        # for tracking performance during training
        self.bn = nn.ModuleList(
            [nn.BatchNorm2d(num_features) for _ in self.num_bits_list]
        )
        self.ratio = 1.

    def forward(self, input, num_bits):
        weight = self.weight
        bias = self.bias

        if len(self.num_bits_list) > 1:
            assert num_bits in self.num_bits_list
            idx = self.num_bits_list.index(num_bits)
        else:
            idx = 0

        y = self.bn[idx](input)

        return y
