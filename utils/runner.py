#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/5/15 12:24
# @File     : runner.py
# @Project  : lab

import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn


def net(model_name, models, device):
    print(f"==========> Model {model_name} selected. <==========")
    net = getattr(models, model_name)().to(device)
    if device == 'cuda':
        net = nn.DataParallel(net)
        cudnn.benchmark = True
    return net


def loss(loss_name='CrossEntropyLoss'):
    loss = {
        'CrossEntropyLoss': nn.CrossEntropyLoss(),
        'MSELoss': nn.MSELoss(),
        'L1Loss': nn.L1Loss(),
        'SmoothL1Loss': nn.SmoothL1Loss(),
        'BCELoss': nn.BCELoss(),
        'BCEWithLogitsLoss': nn.BCEWithLogitsLoss(),
        'HuberLoss': nn.HuberLoss(),
    }
    return loss[loss_name]


def optimizer(optimizer_name, net, lr, momentum, weight_decay):
    optimizer = {
        'SGD': optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay),
        'Adam': optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay),
        'RMSprop': optim.RMSprop(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay),
        'Adadelta': optim.Adadelta(net.parameters(), lr=lr, weight_decay=weight_decay),
        'Adagrad': optim.Adagrad(net.parameters(), lr=lr, weight_decay=weight_decay),
        'Adamax': optim.Adamax(net.parameters(), lr=lr, weight_decay=weight_decay),
        'ASGD': optim.ASGD(net.parameters(), lr=lr, weight_decay=weight_decay),
        'LBFGS': optim.LBFGS(net.parameters(), lr=lr, weight_decay=weight_decay),
        'Rprop': optim.Rprop(net.parameters(), lr=lr, weight_decay=weight_decay),
    }
    return optimizer[optimizer_name]


def scheduler(scheduler_name, optimizer, **kwargs):
    '''
    :param scheduler_name:
    :param optimizer:
    :param kwargs:  mode, factor, patience, cooldown, min_lr, step_size,
                    gamma, milestones, T_max, eta_min, base_lr, max_lr,
                    steps_per_epoch, epochs, T_0, T_mult
    :return:
    '''
    scheduler = {
        'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs),
        ''' step_size=1, gamma=0.5 '''
        'StepLR': optim.lr_scheduler.StepLR(optimizer, **kwargs),
        ''' milestones=[10, 20, 30], gamma=0.1 '''
        'MultiStepLR': optim.lr_scheduler.MultiStepLR(optimizer, **kwargs),
        ''' gamma=0.9 '''
        'ExponentialLR': optim.lr_scheduler.ExponentialLR(optimizer, **kwargs),
        ''' T_max=5, eta_min=0.0001 '''
        'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs),
        ''' base_lr=0.001, max_lr=0.1 '''
        'CyclicLR': optim.lr_scheduler.CyclicLR(optimizer, **kwargs),
        ''' max_lr=0.1, steps_per_epoch=10, epochs=10 '''
        'OneCycleLR': optim.lr_scheduler.OneCycleLR(optimizer, **kwargs),
        ''' T_0=10, T_mult=1 '''
        'CosineAnnealingWarmRestarts': optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **kwargs),
    }
    return scheduler[scheduler_name]
