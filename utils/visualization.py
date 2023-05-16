#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/5/15 14:52
# @File     : visualization.py
# @Project  : lab

import os

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


def compare_losses(train_loss, test_loss, train_accuracy, test_accuracy, losses, figsize=(16, 8), row=2, col=2):
    plt.figure(figsize=figsize)
    axisY = [train_loss, test_loss, train_accuracy, test_accuracy]
    labelY = [ds + la for la in ('Loss', 'Accuracy') for ds in ('Train', 'Test')]
    for idx in range(row * col):
        plt.subplot(row, col, idx + 1)
        for loss, y in zip(losses, axisY[idx]):
            plt.plot(y, label=loss[:-4])
            plt.legend()
            plt.xlabel('Epoch')
            plt.ylabel(labelY[idx])
    plt.show()
    if not os.path.isdir('log'):
        os.mkdir('log')
    plt.savefig(f'./log/criteria.png')
    print('Losses comparison result saved!')


def log_info(epoch, max_epoch, train_loss, train_acc, test_loss, test_acc, best_acc, lr):
    print(line := "=" * 75)
    print(f"Epoch: {epoch + 1} / {max_epoch}")
    print(f"Learning Rate: {lr:.6f}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | Best Acc: {best_acc:.2f}%")
    print(line)


def log_info_losses(epoch, max_epoch, train_loss, train_acc, test_loss, test_acc, losses):
    print(line := "=" * 75)
    for loss, train_l, train_a, test_l, test_a in zip(losses, train_loss, train_acc, test_loss, test_acc):
        print(f"Epoch: {epoch + 1} / {max_epoch} under {loss}")
        print(newline := "-" * 75)
        print(f"Train Loss: {train_l:.4f} | Train Acc: {train_a:.2f}%")
        print(f"Test Loss: {test_l:.4f} | Test Acc: {test_a:.2f}%")
        print(newline)
    print(line)
