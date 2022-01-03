'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class OpticalConv(nn.Module):
    def __init__(self, masks):
        super(OpticalConv, self).__init__()

        self.out_channel = int(masks/2)
        print(self.out_channel, masks)
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

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):

    def __init__(self, num_classes=10, num_masks=6, multiplier=1):
        super(MobileNetV2, self).__init__()
        # (expansion, out_planes, num_blocks, stride)
        self.cfg = [
               (1, int(16 * multiplier), 1, 1),
               (6, int(24 * multiplier), 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
               (6, int(32 * multiplier), 3, 2),
               (6, int(64 * multiplier), 4, 2),
               (6, int(96 * multiplier), 3, 1),
               (6, int(160 * multiplier), 3, 2),
               (6, int(320 * multiplier), 1, 1)]
        print(self.cfg)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(int(320 * multiplier), 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)
        self.optical_cnn = OpticalConv(num_masks)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(self.optical_cnn.filter0.weight.data)
        x = self.optical_cnn(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = MobileNetV2()
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
