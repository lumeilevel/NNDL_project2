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
import yaml
import model
import utils


def train(epoch):
    pass


def test(epoch):
    pass


def main(config):
    device = utils.reproduce(args.seed)
    best_acc, start_epoch = 0, 0
    train_loader, test_loader = utils.get_data_loader(args.cutout, args.batch_size, config)
    net = utils.net(args.model, model, device)
    if args.resume:
        # Load checkpoints.
        print("===> Resuming from checkpoint...")
        assert os.path.isdir('checkpoints'), "Error: no checkpoints directory found!"
        checkpoint = torch.load(f'./checkpoints/checkpoint_{args.model}.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc, start_epoch = checkpoint['acc'], checkpoint['epoch']
    criterion = utils.loss()
    optimizer = utils.optimizer('SGD', net, args.lr, config['momentum'], config['weight_decay'])
    scheduler = utils.scheduler('ReduceLROnPlateau', optimizer, mode='min', factor=0.5, patience=5, cooldown=5,)


if __name__ == '__main__':
    with open('config/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    parser = argparse.ArgumentParser(description="Train a network on CIFAR-10 by PyTorch")
    parser.add_argument(
        '--model', '-m',
        default='ResNet18',
        type=str,
        help="resNet18, resNet34, resNet50, resNet101, resNet152, resNeXt50_32x4d, resNeXt101_32x8d, "
             "wide_resNet50_2, wide_resNet101_2, PreActResNet18, ResNeXt29_32x4d, ResNeXt29_2x64d, \
            WideResNet28x10, DenseNet121, DPN26, DLA",
    )
    parser.add_argument('--batch-size', '-b', type=int, default=config['batch_size'], help='Training batch size')
    parser.add_argument('--lr', default=config['lr'], type=float, help="Learning rate")
    parser.add_argument('--epoch', '-e', default=config['epoch'], type=int, help="Max training epochs")
    parser.add_argument('--cutout', action='store_true', help="Use cutout augmentation")
    parser.add_argument('--seed', '-s', default=config['seed'], type=int, help="Set random seed")
    parser.add_argument('--hidden', nargs='+', default=config['hidden'], type=int, help="Hidden layer size")
    parser.add_argument('--dropout', default=config['dropout'], type=int, help="Dropout rate in hidden layer")
    parser.add_argument('--resume', '-r', action='store_true', help="Resume from checkpoint")
    args = parser.parse_args()
    main(config)
