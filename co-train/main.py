'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import time
import torchvision
from torchvision import datasets, transforms
from thop import profile
from thop.count_hooks import count_convNd

import os
import argparse

from models import *
from utils import progress_bar
from torch.optim.lr_scheduler import StepLR, MultiStepLR


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--masks', '-m', type=float, default=6)
parser.add_argument('--data', '-d', type=str, default="cifar100")
parser.add_argument('--batchSize', '-b', type=int, default=128)
parser.add_argument('--multiplier', '-mul', type=float, default=1.0)
parser.add_argument('--evaluate', '-eval', type=bool, default=False)
parser.add_argument('--std', "-std", type=float, default=0.0)
parser.add_argument("--std_use", "-use", type=str, default="median")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.data == 'cifar10':
    num_classes = 10
    trainloader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                       transform=transform_train),
        batch_size=args.batchSize, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transform_test),
        batch_size=100, shuffle=False, num_workers=2)
elif args.data == 'cifar100':
    num_classes = 100
    trainloader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=True, download=True,
                       transform=transform_train),
        batch_size=args.batchSize, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=False, transform=transform_test),
        batch_size=100, shuffle=False, num_workers=2)


# Model
print('==> Building model..')
net = MobileNetV2(num_masks=args.masks, num_classes=num_classes, multiplier=args.multiplier)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    # assert os.path.isdir('~/pytorch-cifar/checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/100-6.pt')
    state_dict = checkpoint['net']
    for key in state_dict.copy():
        state_dict[key[7:]] = state_dict[key]
    missing, unexpected = net.load_state_dict(checkpoint['net'], strict=False)
    print(missing)
    print(unexpected)
    # best_acc = checkpoint['acc']
    print(checkpoint['acc'])
    # start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, milestones=[80, 120, 160],gamma=0.1)

to_save = './ckpt/{}/'.format(time.strftime("%Y%m%d-%H%M%S"))

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.4f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.4f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    print(acc, best_acc)
    if acc > best_acc:
        print('Saving.........................')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, to_save + str(epoch) + "_%.4f.pt"%acc)
        best_acc = acc

if not os.path.isdir(to_save):
    os.makedirs(to_save)
net = net.cuda()
intput = torch.randn(1, 3, 32, 32).cuda()
flops, params = profile(net, inputs=(intput,))
f = open(to_save + "experiment.txt", "w+")
f.write("arch: MobileNetV2\r\n")
f.write("masks: %d\r\n" % (args.masks))
f.write("data: %s\r\n" % (args.data))
f.write("flops: %s\r\n" % (flops / 1e9))
f.write("params: %s\r\n" % (params / 1e6))
f.write("multiplier: %s\r\n" % (args.multiplier))
f.write("std: %f\r\n" % (args.std))
f.close()
print((flops / 1e9))
if (args.evaluate):
    total = 0
    i = 0
    for j in range(10):
        time_list = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs = inputs.to(device)
                start = time.time()
                outputs = net(inputs)
                end = time.time()
                i+=1
                if batch_idx > 10:
                    timelaps = end-start
                    time_list.append(end - start)

            avg = (sum(time_list) / len(time_list)) / args.batchSize * 1.0
            f.write("average time %f\n" % (avg ))
            print("average time %f\n" % (avg ))
            total += avg
    f.write("total average time %f\n" % (total / 10.0))
    print("total average time %f\n" % (total / 10.0))
elif (args.std > 0):
    filters = [net.optical_cnn.filter0,net.optical_cnn.filter1,net.optical_cnn.filter2]
    for filter in filters:
        maxi = torch.max(filter.weight.data)
        median = torch.median(torch.abs(filter.weight.data))
        std = (args.std * median).cpu().item()
        if (args.std_use == 'max'):
            print("max!")
            std = (args.std * maxi).cpu().item()
        print(args.std_use, args.std, std)
        noise = torch.normal(mean=torch.zeros(list(filter.weight.data.shape)), std=std).cuda()
        filter.weight.data += noise
        for param in filter.parameters():
            param.requires_grad=False
        print("std: ", args.std, "std real: ", std, "std type: ", args.std_use)
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        test(epoch)
        scheduler.step()
else:
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        test(epoch)
        scheduler.step()
