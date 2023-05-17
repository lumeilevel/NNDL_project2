#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/5/12 13:26
# @File     : preprocess.py
# @Project  : lab

import os
import random

import matplotlib as mpl
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset


class PartialDataset(Dataset):
    def __init__(self, dataset, n_items=10):
        self.dataset = dataset
        self.n_items = n_items

    def __getitem__(self):
        return self.dataset.__getitem__()

    def __len__(self):
        return min(self.n_items, len(self.dataset))


def reproduce(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if not os.path.isdir('./log'):
        os.mkdir('./log')
    if not os.path.isdir('./checkpoints'):
        os.mkdir('./checkpoints')
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
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=config['num_workers']
    )
    transform_test = T.Compose(
        [T.ToTensor(), T.Normalize(mean, std)]
    )
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    test_loader = DataLoader(
        test_set, batch_size=config['test_batch_size'], shuffle=False, num_workers=config['num_workers']
    )
    return train_loader, test_loader


def get_cifar_loader(root='./data', batch_size=128, train=True, shuffle=True, num_workers=4, n_items=-1):
    data_transforms = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    dataset = torchvision.datasets.CIFAR10(
        root=root, train=train, download=True, transform=data_transforms
    )
    if n_items > 0:
        dataset = PartialDataset(dataset, n_items)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    mpl.use('Agg')
    return loader
