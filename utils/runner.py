#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/5/15 12:24
# @File     : runner.py
# @Project  : lab

import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs


def net(model_name, models, device):
    print(f"==========> Model {model_name} selected. <==========")
    net = getattr(models, model_name)().to(device)
    if device == 'cuda':
        net = nn.DataParallel(net)
        cudnn.benchmark = True
    return net


def baseline(models, device):
    print(f"==========> Baseline Model ResNet18 selected. <==========")
    net = getattr(models, 'resNet18')()
    if device == 'cuda':
        net = nn.DataParallel(net)
        cudnn.benchmark = True
        cudnn.deterministic = True
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    torch.save(net.state_dict(), './checkpoints/resnet18.pth')
    return net


def net_list(names, models, device, activation_list=None):
    if activation_list:
        nets = [models.resNet18(activation=activation(activation_name)) for activation_name in activation_list]
    else:
        nets = [models.resNet18() for _ in names]
    if device == 'cuda':
        nets = [nn.DataParallel(net) for net in nets]
    for net in nets:
        net.load_state_dict(torch.load('./checkpoints/resnet18.pth'))
    return nets


def loss(loss_name='CrossEntropyLoss'):
    loss = {
        'CrossEntropyLoss': nn.CrossEntropyLoss(),
        'MSELoss': nn.MSELoss(),
        'BCELoss': nn.BCELoss(),
        'HuberLoss': nn.HuberLoss(),
        'L1Loss': nn.L1Loss(),
        'SmoothL1Loss': nn.SmoothL1Loss(),
        'BCEWithLogitsLoss': nn.BCEWithLogitsLoss(),
        'KLDivLoss': nn.KLDivLoss(),
        'NLLLoss': nn.NLLLoss(),
        'PoissonNLLLoss': nn.PoissonNLLLoss(),
        'CosineEmbeddingLoss': nn.CosineEmbeddingLoss(),
        'CTCLoss': nn.CTCLoss(),
        'HingeEmbeddingLoss': nn.HingeEmbeddingLoss(),
        'MarginRankingLoss': nn.MarginRankingLoss(),
        'MultiLabelMarginLoss': nn.MultiLabelMarginLoss(),
        'MultiLabelSoftMarginLoss': nn.MultiLabelSoftMarginLoss(),
        'MultiMarginLoss': nn.MultiMarginLoss(),
        'TripletMarginLoss': nn.TripletMarginLoss(),
        'SoftMarginLoss': nn.SoftMarginLoss(),
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


def activation(activation_name):
    activation = {
        'ReLU': nn.ReLU(),
        'ReLU6': nn.ReLU6(),
        'ELU': nn.ELU(),
        'SELU': nn.SELU(),
        'LeakyReLU': nn.LeakyReLU(),
        'PReLU': nn.PReLU(),
        'Hardtanh': nn.Hardtanh(),
        'Sigmoid': nn.Sigmoid(),
        'Tanh': nn.Tanh(),
        'Softmax': nn.Softmax(),
        'LogSoftmax': nn.LogSoftmax(),
        'Softplus': nn.Softplus(),
    }
    print(f"==========> Activation {activation_name} selected. <==========")
    return activation[activation_name]


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
        'ReduceLROnPlateau': lrs.ReduceLROnPlateau(optimizer, mode=kwargs['mode'],
                                                   factor=kwargs['factor'], patience=kwargs['patience'],
                                                   cooldown=kwargs['cooldown'], min_lr=kwargs['min_lr']),
        # step_size=1, gamma=0.5
        'StepLR': lrs.StepLR(optimizer, step_size=kwargs['step_size'], gamma=kwargs['gamma']['StepLR']),
        # milestones=[10, 20, 30], gamma=0.1
        'MultiStepLR': lrs.MultiStepLR(optimizer, milestones=kwargs['milestones'],
                                       gamma=kwargs['gamma']['MultiStepLR']),
        # gamma=0.9
        'ExponentialLR': lrs.ExponentialLR(optimizer, gamma=kwargs['gamma']['ExponentialLR']),
        # T_max=5, eta_min=0.0001
        'CosineAnnealingLR': lrs.CosineAnnealingLR(optimizer, T_max=kwargs['T_max'], eta_min=kwargs['eta_min']),
        # base_lr=0.001, max_lr=0.1
        'CyclicLR': lrs.CyclicLR(optimizer, max_lr=kwargs['max_lr'], base_lr=kwargs['base_lr'], cycle_momentum=False),
        # max_lr=0.1, steps_per_epoch=10, epochs=10
        'OneCycleLR': lrs.OneCycleLR(optimizer, max_lr=kwargs['max_lr'], cycle_momentum=False,
                                     steps_per_epoch=kwargs['steps_per_epoch'], epochs=kwargs['epochs']),
        # T_0=10, T_mult=1
        'CosineAnnealingWarmRestarts': lrs.CosineAnnealingWarmRestarts(optimizer,
                                                                       T_0=kwargs['T_0'], T_mult=kwargs['T_mult']),
    }
    print(f"==========> Scheduler {scheduler_name} selected. <==========")
    return scheduler[scheduler_name]
