#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/5/15 14:52
# @File     : visualization.py
# @Project  : lab

import os
import numpy as np
from matplotlib import pyplot as plt


def plot_history(train_loss, test_loss, train_accuracy, test_accuracy, lr_schedule, model_name, figsize=(16, 8)):
    plt.figure(figsize=figsize)

    plt.subplot2grid((2, 4), (0, 0), colspan=3)
    plt.plot(train_loss, label='train')
    plt.plot(test_loss, label='test')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot2grid((2, 4), (1, 0), colspan=3)
    plt.plot(train_accuracy, label='train')
    plt.plot(test_accuracy, label='test')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot2grid((2, 4), (0, 3), rowspan=2)
    plt.plot(lr_schedule, label='lr')
    plt.legend()
    plt.title('Learning Rate')
    plt.xlabel('Epoch')

    if not os.path.isdir('log'):
        os.mkdir('log')
    plt.savefig(f'./log/{model_name.lower()}.png')
    plt.show()
    plt.close()


def compare_quota(train_loss, test_loss, train_accuracy, test_accuracy, figsize=(24, 12), row=2, col=2, **kwargs):
    if 'losses' in kwargs:
        quota = kwargs['losses']
        name = 'losses'
    elif 'optimizers' in kwargs:
        quota = kwargs['optimizers']
        name = 'optimizers'
    elif 'activations' in kwargs:
        quota = kwargs['activations']
        name = 'activations'
    else:
        raise ValueError('No quota to compare!')
    plt.figure(figsize=figsize)
    axisY = [train_loss, test_loss, train_accuracy, test_accuracy]
    labelY = [ds + la for la in ('Loss', 'Accuracy') for ds in ('Train ', 'Test ')]
    for idx in range(row * col):
        plt.subplot(row, col, idx + 1)
        for q, y in zip(quota, axisY[idx]):
            plt.plot(y, label=q[:-4] if name == 'losses' else q)
            plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel(labelY[idx])
    if not os.path.isdir('log'):
        os.mkdir('log')
    plt.savefig(f'./log/{name}.png')
    plt.show()
    print(f'Done ----- {name} comparison result saved!')
    plt.close()


def log_info(epoch, max_epoch, train_loss, train_acc, test_loss, test_acc, best_acc, lr):
    print(line := "=" * 75)
    print(f"Epoch: {epoch + 1} / {max_epoch}")
    print(f"Learning Rate: {lr:.6f}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}% | Best Acc: {best_acc:.2f}%")
    print(line)


def log_info_quota(epoch, max_epoch, train_loss, train_acc, test_loss, test_acc, quota):
    print(line := "=" * 75)
    for q, train_l, train_a, test_l, test_a in zip(quota, train_loss, train_acc, test_loss, test_acc):
        print(f"Epoch: {epoch + 1} / {max_epoch} under {q}")
        print(newline := "-" * 75)
        print(f"Train Loss: {train_l:.4f} | Train Acc: {train_a:.2f}%")
        print(f"Test  Loss: {test_l:.4f} | Test  Acc: {test_a:.2f}%")
        print(newline)
    print(line)


def plot_landscape(
    vgg_max, vgg_min, vggbn_max, vggbn_min, xlabel, ylabel, title, interval=40
):
    steps = np.arange(0, len(vgg_max), interval)
    vgg_max = vgg_max[steps]
    vgg_min = vgg_min[steps]
    vggbn_max = vggbn_max[steps]
    vggbn_min = vggbn_min[steps]

    plt.style.use('ggplot')
    plt.figure(figsize=(12, 9))
    plt.plot(steps, vgg_max, color='mediumturquoise', alpha=0.5)
    plt.plot(steps, vgg_min, color='mediumturquoise', alpha=0.5)
    plt.fill_between(
        steps, vgg_min, vgg_max, color='mediumturquoise', alpha=0.5, label='VGG'
    )
    plt.plot(steps, vggbn_max, color='coral', alpha=0.5)
    plt.plot(steps, vggbn_min, color='coral', alpha=0.5)
    plt.fill_between(
        steps, vggbn_min, vggbn_max, color='coral', alpha=0.5, label='VGG + BN'
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.savefig(os.path.join('./log', f'{ylabel.lower()}-landscape.png'))


def plot_beta_landscape(vgg_betas, vggbn_betas):
    vgg_max = np.max(vgg_betas, axis=0)
    vggbn_max = np.max(vggbn_betas, axis=0)

    steps = np.arange(0, len(vgg_max), 40)
    vgg_max = vgg_max[steps]
    vggbn_max = vggbn_max[steps]
    plt.figure(figsize=(12, 9))
    plt.plot(steps, vgg_max, color='mediumturquoise', label='VGG')
    plt.plot(steps, vggbn_max, color='coral', label='VGG + BN')
    plt.xlabel('Step')
    plt.ylabel('Beta')
    plt.legend()
    plt.title('Beta Landscape')
    plt.savefig(os.path.join('./log', 'beta-smooth.png'))
