#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/5/15 12:24
# @File     : runner.py
# @Project  : lab

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim


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
    print(f"==========> Loss {loss_name} selected. <==========")
    return loss[loss_name]


def optimizer(optimizer_name, net, lr, **kwargs):
    momentum = kwargs['momentum'] if 'momentum' in kwargs else 0.9
    weight_decay = kwargs['weight_decay'] if 'weight_decay' in kwargs else 5e-4
    optimizer = {
        'SGD': optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay),
        'Adam': optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay),
        'RMSprop': optim.RMSprop(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay),
        'Adadelta': optim.Adadelta(net.parameters(), lr=lr, weight_decay=weight_decay),
        'Adagrad': optim.Adagrad(net.parameters(), lr=lr, weight_decay=weight_decay),
        'Adamax': optim.Adamax(net.parameters(), lr=lr, weight_decay=weight_decay),
        'ASGD': optim.ASGD(net.parameters(), lr=lr, weight_decay=weight_decay),
        'LBFGS': optim.LBFGS(net.parameters(), lr=lr),
        'Rprop': optim.Rprop(net.parameters(), lr=lr),
    }
    print(f"==========> Optimizer {optimizer_name} selected. <==========")
    return optimizer[optimizer_name]


def scheduler(scheduler_name, optimizer, **kwargs):
    """
    :param scheduler_name:
    :param optimizer:
    :param kwargs:  mode, factor, patience, cooldown, min_lr, step_size,
                    gamma, milestones, T_max, eta_min, base_lr, max_lr,
                    steps_per_epoch, epochs, T_0, T_mult
    :return:
    """
    scheduler = {
        # mode='min', factor=0.1, patience=10, cooldown=0, min_lr=0
        'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=kwargs['mode'],
                                                                  factor=kwargs['factor'], patience=kwargs['patience'],
                                                                  cooldown=kwargs['cooldown'], min_lr=kwargs['min_lr']),
        # step_size=1, gamma=0.5
        'StepLR': optim.lr_scheduler.StepLR(optimizer, step_size=kwargs['step_size'], gamma=kwargs['gamma']['StepLR']),
        # milestones=[10, 20, 30], gamma=0.1
        'MultiStepLR': optim.lr_scheduler.MultiStepLR(optimizer, milestones=kwargs['milestones'],
                                                      gamma=kwargs['gamma']['MultiStepLR']),
        # gamma=0.9
        'ExponentialLR': optim.lr_scheduler.ExponentialLR(optimizer, gamma=kwargs['gamma']['ExponentialLR']),
        # T_max=5, eta_min=0.0001
        'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                  T_max=kwargs['T_max'], eta_min=kwargs['eta_min']),
        # base_lr=0.001, max_lr=0.1
        'CyclicLR': optim.lr_scheduler.CyclicLR(optimizer, max_lr=kwargs['max_lr'], base_lr=kwargs['base_lr']),
        # max_lr=0.1, steps_per_epoch=10, epochs=10
        'OneCycleLR': optim.lr_scheduler.OneCycleLR(optimizer, max_lr=kwargs['max_lr'],
                                                    steps_per_epoch=kwargs['steps_per_epoch'], epochs=kwargs['epochs']),
        # T_0=10, T_mult=1
        'CosineAnnealingWarmRestarts': optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                                      T_0=kwargs['T_0'],
                                                                                      T_mult=kwargs['T_mult']),
    }
    print(f"==========> Scheduler {scheduler_name} selected. <==========")
    return scheduler[scheduler_name]
