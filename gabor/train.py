from __future__ import division
import os
import sys
import time
import glob
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as dset
from torch.autograd import Variable

from tensorboardX import SummaryWriter

import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image

from config_train import config

from model_search import FBNet as Network
from model_infer import FBNet_Infer

from thop import profile
from thop.count_hooks import count_convNd

import operations
from quantize import QConv2d

operations.DWS_CHWISE_QUANT = config.dws_chwise_quant

custom_ops = {QConv2d: count_convNd}

def main():
    config.save = 'ckpt/{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
    logger = SummaryWriter(config.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info("args = %s", str(config))
    # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    state = torch.load(os.path.join(config.load_path, 'arch.pt'))
    # Model #######################################
    model = FBNet_Infer(state['alpha'], config=config)

    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32),), custom_ops=custom_ops)
    logging.info("params = %fMB, FLOPs = %fGB", params / 1e6, flops / 1e9)

    if type(config.pretrain) == str:
        state_dict = torch.load(config.pretrain)

        for key in state_dict.copy():
            if 'bn.0' in key:
                new_key_list = []

                for i in range(1, len(config.num_bits_list)):
                    new_key = []
                    new_key.extend(key.split('.')[:-2])
                    new_key.append(str(i))
                    new_key.append(key.split('.')[-1])
                    new_key = '.'.join(new_key)

                    state_dict[new_key] = state_dict[key]
        for key in state_dict.copy():
            new_key = key[7:]
            state_dict[new_key] = state_dict[key]
        # print(state_dict["module.optical_layer.filter.weight"])
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(missing)
        # print(unexpected)


    if config.std != 0:
        checkpoint = torch.load(config.pretrain)
        max = torch.max(checkpoint["module.optical_layer.filter.weight"]).cpu().item()
        median = torch.median(torch.abs(checkpoint["module.optical_layer.filter.weight"])).cpu().item()
        std = config.std * median
        if config.std_use == "max":
            std = config.std * max

        model.optical_layer.filter.weight.data = checkpoint["module.optical_layer.filter.weight"]
        # print(checkpoint.keys())
        noise = torch.normal(mean=torch.zeros(list(model.optical_layer.filter.weight.data.shape)), std=std).cuda()
        # print(noise)
        model.optical_layer.filter.weight.data += noise
        print("std: ", std, "required: ", config.std)
        # print(model.optical_layer.filter.weight.data)

    model = torch.nn.DataParallel(model).cuda()
    # model = model.cuda()


    if config.opt == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=config.betas)
    elif config.opt == 'Sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay)
    else:
        logging.info("Wrong Optimizer Type.")
        sys.exit()

    # lr policy ##############################
    total_iteration = config.nepochs * config.niters_per_epoch
    
    if config.lr_schedule == 'linear':
        lr_policy = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(config.nepochs, 0, config.decay_epoch).step)
    elif config.lr_schedule == 'exponential':
        lr_policy = torch.optim.lr_scheduler.ExponentialLR(optimizer, config.lr_decay)
    elif config.lr_schedule == 'multistep':
        lr_policy = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)
    elif config.lr_schedule == 'cosine':
        lr_policy = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(config.nepochs), eta_min=config.learning_rate_min)
    else:
        logging.info("Wrong Learning Rate Schedule Type.")
        sys.exit()


    # data loader ############################

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    if config.dataset == 'cifar10':
        train_data = dset.CIFAR10(root=config.dataset_path, train=True, download=True, transform=transform_train)
        test_data = dset.CIFAR10(root=config.dataset_path, train=False, download=True, transform=transform_test)
    elif config.dataset == 'cifar100':
        train_data = dset.CIFAR100(root=config.dataset_path, train=True, download=True, transform=transform_train)
        test_data = dset.CIFAR100(root=config.dataset_path, train=False, download=True, transform=transform_test)
    else:
        print('Wrong dataset.')
        sys.exit()

    train_loader_model = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size,
        pin_memory=True, num_workers=4)

    test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=config.batch_size,
                                          shuffle=False,
                                          num_workers=4)


    if config.eval_only:
        logging.info('Eval - Acc under different bits: ' + str(infer(0, model, test_loader, logger, config.num_bits_list)))
        sys.exit(0)

    tbar = tqdm(range(config.nepochs), ncols=80)
    for epoch in tbar:
        logging.info(config.save)
        logging.info("lr: " + str(optimizer.param_groups[0]['lr']))

        # training
        tbar.set_description("[Epoch %d/%d][train...]" % (epoch + 1, config.nepochs))
        train(train_loader_model, model, optimizer, lr_policy, logger, epoch, num_bits_list=config.num_bits_list)
        torch.cuda.empty_cache()
        lr_policy.step()

        # validation
        if epoch and not (epoch+1) % config.eval_epoch:
            tbar.set_description("[Epoch %d/%d][validation...]" % (epoch + 1, config.nepochs))
            
            with torch.no_grad():

                acc_bits = infer(epoch, model, test_loader, logger, config.num_bits_list)

                for i, num_bits in enumerate(config.num_bits_list):
                    logger.add_scalar('acc/val_bits_%d' % num_bits, acc_bits[i], epoch)
                    
                logging.info("Epoch: " + str(epoch) +" Acc under different bits: " + str(acc_bits))
                
                logger.add_scalar('flops/val', flops, epoch)
                logging.info("Epoch %d: flops %.3f"%(epoch, flops))

            save(model, os.path.join(config.save, 'weights_%d.pt'%epoch))

    save(model, os.path.join(config.save, 'weights.pt'))



def train(train_loader_model, model, optimizer, lr_policy, logger, epoch, num_bits_list):
    model.train()

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format, ncols=80)
    dataloader_model = iter(train_loader_model)


    for step in pbar:
        lr = optimizer.param_groups[0]['lr']

        optimizer.zero_grad()

        input, target = dataloader_model.next()

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        loss_value = [-1 for _ in num_bits_list]

        for num_bits in sorted(num_bits_list, reverse=True):
            logit = model(input, num_bits)
            loss = model.module._criterion(logit, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            loss_value[num_bits_list.index(num_bits)] = loss.item()

        for i, num_bits in enumerate(num_bits_list):
            if loss_value[i] != -1:
                logger.add_scalar('loss/num_bits_%d' % num_bits, loss_value[i], epoch*len(pbar)+step)

        pbar.set_description("[Step %d/%d]" % (step + 1, len(train_loader_model)))

    torch.cuda.empty_cache()
    del loss


def infer(epoch, model, test_loader, logger, num_bits_list):
    model.eval()
    # print(model.optical_layer.filter.weight.data)
    acc_bits = []

    for num_bits in num_bits_list:
        prec1_list = []

        for i, (input, target) in enumerate(test_loader):
            input_var = Variable(input, volatile=True).cuda()
            target_var = Variable(target, volatile=True).cuda()

            output = model(input_var, num_bits)
            prec1, = accuracy(output.data, target_var, topk=(1,))
            prec1_list.append(prec1)

        acc = sum(prec1_list)/len(prec1_list)
        acc_bits.append(acc)

    return acc_bits


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save(model, model_path):
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    main() 
