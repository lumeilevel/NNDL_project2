#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/5/12 11:18
# @File     : cifar.py
# @Project  : lab

import argparse
import os

import torch
import yaml

import model
import utils


def train(net, train_loader, train_batch_size, device, criterion, optimizer):
    net.train()
    train_loss, correct, total, batch_idx = 0, 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = outputs.max(1)[1]
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = train_loss / ((batch_idx + 1) * train_batch_size)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def test(net, test_loader, test_batch_size, device, criterion, epoch):
    global best_acc
    net.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            predicted = outputs.max(1)[1]
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = test_loss / ((batch_idx + 1) * test_batch_size)
    epoch_acc = 100.0 * correct / total

    # Save checkpoint
    if epoch_acc > best_acc:
        state = {
            'net': net.state_dict(),
            'acc': epoch_acc,
            'epoch': epoch,
        }
        print('Saving...')
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        torch.save(state, f'./checkpoints/checkpoint_{args.model.lower()}.pth')
        print('Saved...')
        best_acc = epoch_acc

    return epoch_loss, epoch_acc


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
    scheduler = utils.scheduler('ReduceLROnPlateau', optimizer, mode='min', factor=0.5, patience=5, cooldown=5, )
    train_loss, train_acc, test_loss, test_acc, lr_schedule = [], [], [], [], []
    for epoch in range(start_epoch, max_epoch := start_epoch + args.epoch):
        train_l, train_a = train(net, train_loader, args.batch_size, device, criterion, optimizer)
        test_l, test_a = test(net, test_loader, config['test_batch_size'], device, criterion, epoch)
        scheduler.step(test_l)

        train_loss.append(train_l)
        train_acc.append(train_a)
        test_loss.append(test_l)
        test_acc.append(test_a)
        lr_schedule.append(lr := optimizer.param_groups[0]['lr'])

        utils.log_info(epoch, max_epoch, train_l, train_a, test_l, test_a, best_acc, lr)
        utils.plot_history(train_loss, train_acc, test_loss, test_acc, lr_schedule, args.model)


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
