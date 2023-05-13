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

transform_test = T.Compose(
    [T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]
)
test_batch_size = 256


def reproduce(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_data_loader(cutout, batch_size, num_workers=2):
    if cutout:
        # from .cutout import Cutout
        # transform_train = T.Compose([
        #     T.RandomCrop(32, padding=4),
        #     T.RandomHorizontalFlip(),
        #     T.ToTensor(),
        #     Cutout(n_holes=1, length=16),
        # ])
        pass
    else:
        transform_train = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader
