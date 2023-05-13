#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/5/12 11:18
# @File     : cifar.py
# @Project  : lab

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as T

from matplotlib import pyplot as plt

import os
import argparse
import random

import model

parser = argparse.ArgumentParser(description="Train a network on CIFAR-10 by PyTorch")
parser.add_argument(
    '--resume', '-r', action='store_true', help="Resume from checkpoint"
)
parser.add_argument(
    '--model', '-m',
    default='ResNet18',
    type=str,
    help="resNet18, resNet34, resNet50, resNet101, resNet152, resNeXt50_32x4d, resNeXt101_32x8d, "
         "wide_resNet50_2, wide_resNet101_2, PreActResNet18, ResNeXt29_32x4d, ResNeXt29_2x64d, \
        WideResNet28x10, DenseNet121, DPN26, DLA",
)
parser.add_argument('--batch-size', '-b', type=int, default=128, help='Training batch size')
parser.add_argument('--lr', default=0.1, type=float, help="Learning rate")
parser.add_argument('--epoch', '-e', default=100, type=int, help="Max training epochs")
parser.add_argument('--use-cutout', action='store_true', help="Use cutout augmentation")
parser.add_argument('--seed', '-s', default=1, type=int, help="Set random seed")
parser.add_argument('--hidden', nargs='+', default=0, type=int, help="Hidden layer size")
parser.add_argument('--dropout', default=0, type=int, help="Dropout rate in hidden layer")
args = parser.parse_args()


def reproduce(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    device = reproduce(args.seed)
    best_acc = 0  # Best test accuracy
    start_epoch = 0  # Start from epoch 0 or last checkpoint epoch


if __name__ == '__main__':
    main()
