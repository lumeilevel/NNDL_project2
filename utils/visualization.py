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

    if not os.path.isdir('../log'):
        os.mkdir('../log')
    plt.savefig(f'../log/{model_name.lower()}.png')
    plt.show()


def log_info(epoch, max_epoch, train_loss, train_acc, test_loss, test_acc, best_acc, lr):
    print(line := "=" * 50)
    print(f"Epoch: {epoch + 1} / {max_epoch}")
    print(f"Learning Rate: {lr:.6f}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | Best Acc: {best_acc:.2f}%")
    print(line)
