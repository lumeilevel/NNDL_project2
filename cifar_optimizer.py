#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/5/16 10:54
# @File     : cifar_optimizer.py
# @Project  : lab

import argparse

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

import model
import utils


def train(nets, train_loader, train_batch_size, device, epoch, criterion, optimizers):
    for net in nets:
        net.train()
    train_loss, correct, total, batch_idx = [0] * len(nets), [0] * len(nets), 0, 0

    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), desc=f'Training Epoch {epoch + 1}\t'):
        inputs, targets = inputs.to(device), targets.to(device)
        targets_onehot = F.one_hot(targets, num_classes=10).to(torch.float)
        total += targets.size(0)

        for i, (net, optimizer) in enumerate(zip(nets, optimizers)):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets_onehot)
            loss.backward()
            optimizer.step()

            train_loss[i] += loss.item()
            predicted = outputs.max(1)[1]
            correct[i] += predicted.eq(targets).sum().item()

    epoch_loss = [loss / ((batch_idx + 1) * train_batch_size) for loss in train_loss]
    epoch_acc = [100.0 * c / total for c in correct]
    return epoch_loss, epoch_acc


def test(nets, test_loader, test_batch_size, device, epoch, criteria):
    for net in nets:
        net.eval()
    test_loss, correct, total = [0] * len(nets), [0] * len(nets), 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader), desc=f'Test     Epoch {epoch + 1}\t'):
            inputs, targets = inputs.to(device), targets.to(device)
            targets_onehot = F.one_hot(targets, num_classes=10).to(torch.float)
            total += targets.size(0)

            for i, (net, criterion) in enumerate(zip(nets, criteria)):
                outputs = net(inputs)
                loss = criterion(outputs, targets_onehot)

                test_loss[i] += loss.item()
                predicted = outputs.max(1)[1]
                correct[i] += predicted.eq(targets).sum().item()

    epoch_loss = [loss / ((batch_idx + 1) * test_batch_size) for loss in test_loss]
    epoch_acc = [100.0 * c / total for c in correct]
    return epoch_loss, epoch_acc


def main(config):
    device = utils.reproduce(args.seed)
    train_loader, test_loader = utils.get_data_loader(False, args.batch_size, config)
    net = utils.baseline(model, device)
    nets = utils.net_list(config['optimizers'], model, device)
    criterion = utils.loss('CrossEntropyLoss')
    optimizers = [utils.optimizer('Adam', net, config['lr_adam'], **config['optimizer'])] + \
                 [utils.optimizer(optimizer, net, args.lr, **config['optimizer'])
                  for optimizer in config['optimizers'][1:]]
    schedulers = [utils.scheduler('ReduceLROnPlateau', optimizer, **config['scheduler']) for optimizer in optimizers]

    train_loss, train_acc, test_loss, test_acc = (__ := [[] for _ in config['optimizers']]), __, __, __
    for epoch in range(args.epoch):
        train_l, train_a = train(nets, train_loader, args.batch_size, device, epoch, criterion, optimizers)
        test_l, test_a = test(nets, test_loader, config['test_batch_size'], device, epoch, criterion)

        for i, (train_loss_i, train_acc_i, test_loss_i, test_acc_i) in enumerate(zip(train_l, train_a, test_l, test_a)):
            train_loss[i].append(train_loss_i)
            train_acc[i].append(train_acc_i)
            test_loss[i].append(test_loss_i)
            test_acc[i].append(test_acc_i)
            schedulers[i].step(test_loss_i)

        utils.log_info_quota(epoch, args.epoch, train_l, train_a, test_l, test_a, config['optimizers'])
    utils.compare_quota(train_loss, test_loss, train_acc, test_acc, optimizers=config['optimizers'])


if __name__ == '__main__':
    with open('config/config_optimizer.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    parser = argparse.ArgumentParser(description="Compare loss function on CIFAR-10")
    parser.add_argument('--batch-size', '-b', type=int, default=config['batch_size'], help='Training batch size')
    parser.add_argument('--lr', default=config['lr'], type=float, help="Learning rate")
    parser.add_argument('--epoch', '-e', default=config['epoch'], type=int, help="Max training epochs")
    parser.add_argument('--seed', '-s', default=config['seed'], type=int, help="Set random seed")
    args = parser.parse_args()
    main(config)
