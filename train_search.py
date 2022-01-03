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

import time

from tensorboardX import SummaryWriter

import numpy as np
import matplotlib

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image

from config_search import config

from architect import Architect
from model_search import FBNet as Network
from model_infer import FBNet_Infer

from lr import LambdaLR
from perturb import Random_alpha

import operations

operations.DWS_CHWISE_QUANT = config.dws_chwise_quant


def main(pretrain=True):
    config.save = 'ckpt/{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
    logger = SummaryWriter(config.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    assert type(pretrain) == bool or type(pretrain) == str
    update_arch = True
    if pretrain == True:
        update_arch = False
    logging.info("args = %s", str(config))
    # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Model #######################################
    model = Network(config=config)
    model = torch.nn.DataParallel(model).cuda()

    if type(pretrain) == str:
        partial = torch.load(pretrain + "/weights.pt")
        state = model.state_dict()
        pretrained_dict = {k: v for k, v in partial.items() if k in state and state[k].size() == partial[k].size()}

        for key in partial:
            if 'bn.0' in key:
                new_key_list = []

                for i in range(1, len(config.num_bits_list)):
                    new_key = []
                    new_key.extend(key.split('.')[:-2])
                    new_key.append(str(i))
                    new_key.append(key.split('.')[-1])
                    new_key = '.'.join(new_key)

                    pretrained_dict[new_key] = partial[key]

        state.update(pretrained_dict)
        model.load_state_dict(state, strict=False)

    architect = Architect(model, config)

    # Optimizer ###################################
    base_lr = config.lr
    # parameters = []
    # parameters += list(model.module.stem.parameters())
    # parameters += list(model.module.cells.parameters())
    # parameters += list(model.module.header.parameters())
    # parameters += list(model.module.fc.parameters())

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
        lr_policy = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=LambdaLR(config.nepochs, 0, config.decay_epoch).step)
    elif config.lr_schedule == 'exponential':
        lr_policy = torch.optim.lr_scheduler.ExponentialLR(optimizer, config.lr_decay)
    elif config.lr_schedule == 'multistep':
        lr_policy = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)
    elif config.lr_schedule == 'cosine':
        lr_policy = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(config.nepochs),
                                                               eta_min=config.learning_rate_min)
    else:
        logging.info("Wrong Learning Rate Schedule Type.")
        sys.exit()

    # data loader ###########################
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

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(config.train_portion * num_train))

    train_loader_model = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=False, num_workers=config.num_workers)

    train_loader_arch = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=False, num_workers=config.num_workers)

    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=config.batch_size,
                                              shuffle=False,
                                              num_workers=config.num_workers)

    tbar = tqdm(range(config.nepochs), ncols=80)

    for epoch in tbar:
        logging.info(pretrain)
        logging.info(config.save)
        logging.info("lr: " + str(optimizer.param_groups[0]['lr']))

        logging.info("update arch: " + str(update_arch))

        lr_policy.step()

        if config.perturb_alpha:
            epsilon_alpha = 0.03 + (config.epsilon_alpha - 0.03) * epoch / config.nepochs
            logging.info('Epoch %d epsilon_alpha %e', epoch, epsilon_alpha)
        else:
            epsilon_alpha = 0

        temp = config.temp_init * config.temp_decay ** epoch
        logging.info("Temperature: " + str(temp))

        # training
        tbar.set_description("[Epoch %d/%d][train...]" % (epoch + 1, config.nepochs))
        train(train_loader_model, train_loader_arch, model, architect, optimizer, lr_policy, logger, epoch,
              num_bits_list=config.num_bits_list, update_arch=update_arch,
              epsilon_alpha=epsilon_alpha,
              criteria=config.criteria, temp=temp)
        torch.cuda.empty_cache()

        # validation
        if epoch and not (epoch + 1) % config.eval_epoch:
            tbar.set_description("[Epoch %d/%d][validation...]" % (epoch + 1, config.nepochs))

            save(model, os.path.join(config.save, 'weights_%d.pt' % epoch))

            with torch.no_grad():
                if pretrain == True:
                    acc_bits = infer(epoch, model, test_loader, logger, config.num_bits_list, temp=temp)

                    for i, num_bits in enumerate(config.num_bits_list):
                        logger.add_scalar('acc/val_bits_%d' % num_bits, acc_bits[i], epoch)

                    logging.info("Epoch: " + str(epoch) + " Acc under different bits: " + str(acc_bits))

                else:
                    acc_bits, metric = infer(epoch, model, test_loader, logger, config.num_bits_list, finalize=True,
                                             temp=temp)

                    for i, num_bits in enumerate(config.num_bits_list):
                        logger.add_scalar('acc/val_bits_%d' % num_bits, acc_bits[i], epoch)

                    logging.info("Epoch: " + str(epoch) + " Acc under different bits: " + str(acc_bits))

                    state = {}

                    logger.add_scalar('flops/val', metric, epoch)
                    logging.info("Epoch: %d Flops: %.3f" % (epoch, metric))
                    state["flops"] = metric

                    state['alpha'] = getattr(model.module, 'alpha')
                    state["acc"] = acc_bits

                    torch.save(state, os.path.join(config.save, "arch_%d.pt" % (epoch)))

                    if config.flops_weight > 0:
                        if metric < config.flops_min:
                            architect.flops_weight /= 2
                        elif metric > config.flops_max:
                            architect.flops_weight *= 2
                        logger.add_scalar("arch/flops_weight", architect.flops_weight, epoch + 1)
                        logging.info("arch_flops_weight = " + str(architect.flops_weight))

        if config.early_stop_by_skip:
            groups = config.num_layer_list[1:-1]
            num_block = groups[0]

            current_arch = getattr(model.module, 'alpha').data[1:-1].argmax(-1)

            early_stop = False

            for group_id in range(len(groups)):
                num_skip = 0
                for block_id in range(num_block):
                    if current_arch[group_id * num_block + block_id] == 8:
                        num_skip += 1
                if num_skip >= 2:
                    early_stop = True

            if early_stop:
                print('Early Stop at epoch %d.' % epoch)
                break

    if update_arch:
        torch.save(state, os.path.join(config.save, "arch.pt"))


def train(train_loader_model, train_loader_arch, model, architect, optimizer, lr_policy, logger, epoch, num_bits_list,
          update_arch=True, epsilon_alpha=0, criteria=None, temp=1):
    model.train()

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format, ncols=80)
    dataloader_model = iter(train_loader_model)
    dataloader_arch = iter(train_loader_arch)

    for step in pbar:
        input, target = dataloader_model.next()

        # end = time.time()

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # time_data = time.time() - end
        # end = time.time()

        if update_arch:
            pbar.set_description("[Step %d/%d]" % (step + 1, len(train_loader_arch)))

            input_search, target_search = dataloader_arch.next()
            input_search = input_search.cuda(non_blocking=True)
            target_search = target_search.cuda(non_blocking=True)

            loss_arch = architect.step(input, target, input_search, target_search, num_bits_list, temp=temp)

            if (step + 1) % 10 == 0:
                for i, num_bits in enumerate(num_bits_list):
                    if loss_arch[i] != -1:
                        logger.add_scalar('loss_arch/num_bits_%d' % num_bits, loss_arch[i], epoch * len(pbar) + step)

                logger.add_scalar('arch/flops_supernet', architect.flops_supernet, epoch * len(pbar) + step)

        # print(model.module.alpha[1])
        # print(model.module.ratio[1])

        if epsilon_alpha:
            Random_alpha(model, epsilon_alpha)

        optimizer.zero_grad()

        loss_value = [-1 for _ in num_bits_list]

        if criteria is not None:
            if criteria == 'min':
                num_bits = min(num_bits_list)
            else:
                num_bits = max(num_bits_list)

            logit = model(input, num_bits, temp=temp)
            loss = model.module._criterion(logit, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            loss_value[num_bits_list.index(num_bits)] = loss.item()

        else:
            for num_bits in sorted(num_bits_list, reverse=True):
                logit = model(input, num_bits, temp=temp)
                loss = model.module._criterion(logit, target)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

                loss_value[num_bits_list.index(num_bits)] = loss.item()

        for i, num_bits in enumerate(num_bits_list):
            if loss_value[i] != -1:
                logger.add_scalar('loss/num_bits_%d' % num_bits, loss_value[i], epoch * len(pbar) + step)

        # time_bw = time.time() - end
        # end = time.time()

        # print("[Step %d/%d]" % (step + 1, len(train_loader_model)), 'Loss:', loss, 'Time Data:', time_data, 'Time Forward:', time_fw, 'Time Backward:', time_bw)

        pbar.set_description("[Step %d/%d]" % (step + 1, len(train_loader_model)))

    torch.cuda.empty_cache()
    del loss
    if update_arch: del loss_arch


def infer(epoch, model, test_loader, logger, num_bits_list, finalize=False, temp=1):
    model.eval()
    prec1_list = []

    acc_bits = []

    for num_bits in num_bits_list:
        for i, (input, target) in enumerate(test_loader):
            input_var = Variable(input, volatile=True).cuda()
            target_var = Variable(target, volatile=True).cuda()

            output = model(input_var, num_bits)
            prec1, = accuracy(output.data, target_var, topk=(1,))
            prec1_list.append(prec1)

        acc = sum(prec1_list) / len(prec1_list)
        acc_bits.append(acc)

    if finalize:
        model_infer = FBNet_Infer(getattr(model.module, 'alpha'), config=config)

        flops = model_infer.forward_flops((3, 32, 32))
        return acc_bits, flops

    else:
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
    main(pretrain=config.pretrain) 
