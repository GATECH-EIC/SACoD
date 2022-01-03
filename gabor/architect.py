import sys
import os
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchcontrib
import numpy as np
from pdb import set_trace as bp
from thop import profile
from operations import *
from genotypes import PRIMITIVES


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, args):
        # self.network_momentum = args.momentum
        # self.network_weight_decay = args.weight_decay
        self.model = model
        self._args = args

        self.optimizer = torch.optim.Adam(list(self.model.module._arch_params.values()), lr=args.arch_learning_rate, betas=(0.5, 0.999))#, weight_decay=args.arch_weight_decay)
        
        self.flops_weight = args.flops_weight

        print("architect initialized!")


    def step(self, input_train, target_train, input_valid, target_valid, num_bits_list, temp=1):
        self.optimizer.zero_grad()

        loss, loss_flops = self._backward_step_flops(input_valid, target_valid, num_bits_list, temp=temp)

        # loss.backward()
        # self.optimizer.step()

        return loss


    def _backward_step_flops(self, input_valid, target_valid, num_bits_list, temp=1):
        loss_value = [-1 for _ in num_bits_list]

        for num_bits in sorted(num_bits_list, reverse=True):
            logit = self.model(input_valid, num_bits, temp=temp)
            loss = self.model.module._criterion(logit, target_valid)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_value[num_bits_list.index(num_bits)] = loss.item()

        # flops = self.model.module.forward_flops((16, 32, 32))
        flops = self.model.module.forward_flops((3, 32, 32), temp=temp)
            
        self.flops_supernet = flops
        loss_flops = self.flops_weight * flops

        loss_flops.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # print(flops, loss_flops, loss)
        return loss_value, loss_flops


