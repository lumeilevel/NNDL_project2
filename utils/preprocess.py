#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/5/12 13:26
# @File     : preprocess.py
# @Project  : lab

import os
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as T


def reproduce(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_data_loader(cutout, batch_size, config):
    if isinstance(config['mean'], str) and isinstance(config['std'], str):
        mean, std = eval(config['mean']), eval(config['std'])
    else:
        mean, std = config['mean'], config['std']
    if cutout:
        from augmentation import Cutout
        transform_train = T.Compose([
            T.RandomCrop(config['size'], padding=config['padding']),
            T.RandomHorizontalFlip(),
            Cutout(config['p'], config['half_size']),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        transform_train = T.Compose(
            [
                T.RandomCrop(config['size'], padding=config['padding']),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=config['num_workers']
    )
    transform_test = T.Compose(
        [T.ToTensor(), T.Normalize(mean, std)]
    )
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=config['test_batch_size'], shuffle=False, num_workers=config['num_workers']
    )
    return train_loader, test_loader
