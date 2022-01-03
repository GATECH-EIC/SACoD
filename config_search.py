# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

C = edict()
config = C
cfg = C

C.seed = 12345

"""please config ROOT_dir and user when u first using"""
C.repo_name = 'SACoD'


"""Data Dir and Weight Dir"""
C.dataset_path = "~/SACoD"

C.dataset = 'cifar100'

if C.dataset == 'cifar10':
    C.num_classes = 10
elif C.dataset == 'cifar100':
    C.num_classes = 100
else:
    print('Wrong dataset.')
    sys.exit()


"""Image Config"""

C.num_train_imgs = 50000
C.num_eval_imgs = 10000

""" Settings for network, this would be different for each kind of model"""
C.bn_eps = 1e-5
C.bn_momentum = 0.1

"""Train Config"""

C.opt = 'Sgd'

C.momentum = 0.9
C.weight_decay = 5e-4

C.betas=(0.5, 0.999)
C.num_workers = 8

""" Search Config """
C.grad_clip = 5

C.pretrain = False
# C.pretrain = 'ckpt/pretrain'

C.std = -1

C.dws_chwise_quant = True

# C.num_layer_list = [1, 4, 4, 4, 4, 4, 1]
C.num_layer_list = [1, 4, 4, 4, 4, 4, 1]
C.num_channel_list = [16, 24, 32, 64, 112, 184, 352]
C.stride_list = [1, 1, 2, 2, 1, 2, 1]

# C.num_layer_list = [1, 2, 1]
# C.num_channel_list = [16, 128, 256]
# C.stride_list = [1, 2, 2]

C.stem_channel = 16
C.header_channel = 1504

C.num_bits_list = [0]

C.mask = 6
C.search_fix = False

C.trained_mask = ''

C.early_stop_by_skip = False

C.perturb_alpha = False
C.epsilon_alpha = 0.3

C.criteria = None

C.sample_func = 'gumbel_softmax'
C.temp_init = 3
C.temp_decay = 0.92

########################################

if C.pretrain == True:
    C.batch_size = 32
    C.niters_per_epoch = C.num_train_imgs // 2 // C.batch_size
    C.image_height = 32
    C.image_width = 32
    C.save = "pretrain"

    C.num_bits_list = [0]

    C.nepochs = 20
    C.eval_epoch = 1

    C.lr_schedule = 'cosine'
    C.lr = 0.025

    # C.lr_schedule = 'cosine'
    # C.lr = 0.025
    
    # linear 
    C.decay_epoch = 20
    # exponential
    C.lr_decay = 0.97
    # multistep
    C.milestones = [50, 100, 200]
    C.gamma = 0.1
    # cosine
    C.learning_rate_min = 0.001


else:
    C.batch_size = 64
    C.niters_per_epoch = C.num_train_imgs // 2 // C.batch_size
    C.image_height = 32
    C.image_width = 32
    C.save = "search-cifar100"

    C.nepochs = 50
    C.eval_epoch = 1

    C.lr_schedule = 'cosine'
    C.lr = 0.025

    # C.lr_schedule = 'cosine'
    # C.lr = 0.025

    # linear 
    C.decay_epoch = 20
    # exponential
    C.lr_decay = 0.97
    # multistep
    C.milestones = [50, 100, 200]
    C.gamma = 0.1
    # cosine
    C.learning_rate_min = 0.001


########################################

C.train_portion = 0.5

C.unrolled = False

C.arch_learning_rate = 3e-4

C.efficiency_metric = 'flops'

C.flops_weight = 0
C.flops_max = 1.3e8
C.flops_min = 1.2e8
