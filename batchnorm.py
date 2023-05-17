#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/5/17 0:09
# @File     : batchnorm.py
# @Project  : lab

import argparse

import numpy as np
import torch
import yaml
from tqdm import trange, tqdm

import model
import utils


def validate(model, val_loader, device):
    # Validation
    model.eval()

    total, correct = 0, 0
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    accuracy = correct / total
    return accuracy


def train(model, device, optimizer, criterion, train_loader, val_loader, scheduler=None, num_epochs=100):
    # Logging initialization
    train_accuracies = []
    val_accuracies = []
    losses = []  # Training loss
    dists = []  # Distance between two gradients
    betas = []  # Beta smoothness
    grad_pre = None  # Previous gradient
    weight_pre = None  # Previous weight

    model.to(device)

    for epoch in trange(num_epochs):
        # Training
        model.train()

        total, correct = 0, 0
        for data, target in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} '):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Epoch logging
            losses.append(loss.item())
            grad = model.classifier[-1].weight.grad.detach().clone()
            weight = model.classifier[-1].weight.detach().clone()
            if grad_pre is not None:
                grad_dist = torch.dist(grad, grad_pre).item()
                dists.append(grad_dist)
            if weight_pre is not None:
                weight_dist = torch.dist(weight, weight_pre).item()
                beta = grad_dist / weight_dist
                betas.append(beta)
            grad_pre = grad
            weight_pre = weight

        # Training accuracy
        train_accuracies.append(correct / total)

        # Validation accuracy
        val_accuracy = validate(model, val_loader, device)
        val_accuracies.append(val_accuracy)

        # Learning rate scheduler
        if scheduler is not None:
            scheduler.step()

    return train_accuracies, val_accuracies, losses, dists, betas


def explore_lr(name, lrs, device, criterion, train_loader, valid_loader, max_epochs, training, validation, loss, dist, beta):
    for lr in lrs:
        if name == 'vgg':
            net = model.VGG_A()
            print("======================VGG-A======================")
        else:
            net = model.VGG_A_BatchNorm()
            print("=================VGG-A-BarchNorm=================")
        print(f"Learning rate: {lr}")
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        train_acc, valid_acc, losses, dists, betas = train(
            net, device, optimizer, criterion, train_loader, valid_loader, None, max_epochs
        )
        training.append(train_acc)
        validation.append(valid_acc)
        loss.append(losses)
        dist.append(dists)
        beta.append(betas)


def main(config):
    device = utils.reproduce(args.seed)
    if device != 'cpu':
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    train_loader = utils.get_cifar_loader(train=True)
    val_loader = utils.get_cifar_loader(train=False)

    vgg_train, vgg_val, vgg_losses, vgg_dists, vgg_betas = [], [], [], [], []
    vggbn_train, vggbn_val, vggbn_losses, vggbn_dists, vggbn_betas = [], [], [], [], []
    criterion = utils.loss()

    explore_lr('vgg', config['lrs'], device, criterion, train_loader, val_loader,
               config['max_epochs'], vgg_train, vgg_val, vgg_losses, vgg_dists, vgg_betas)
    explore_lr('vggbn', config['lrs'], device, criterion, train_loader, val_loader,
               config['max_epochs'], vggbn_train, vggbn_val, vggbn_losses, vggbn_dists, vggbn_betas)

    vgg_losses, vggbn_losses = np.array(vgg_losses), np.array(vggbn_losses)
    vgg_dists, vggbn_dists = np.array(vgg_dists), np.array(vggbn_dists)
    vgg_betas, vggbn_betas = np.array(vgg_betas), np.array(vggbn_betas)
    vgg_max = np.max(vgg_losses, axis=0)
    vgg_min = np.min(vgg_losses, axis=0)
    vggbn_max = np.max(vggbn_losses, axis=0)
    vggbn_min = np.min(vggbn_losses, axis=0)
    utils.plot_landscape(vgg_max, vgg_min, vggbn_max, vggbn_min, 'Step', 'Loss', 'Loss Landscape')
    vgg_max = np.max(vgg_dists, axis=0)
    vgg_min = np.min(vgg_dists, axis=0)
    vggbn_max = np.max(vggbn_dists, axis=0)
    vggbn_min = np.min(vggbn_dists, axis=0)
    utils.plot_landscape(vgg_max, vgg_min, vggbn_max, vggbn_min, 'Step', 'Distance', 'Gradient Distance')
    utils.plot_beta_landscape(vgg_betas, vggbn_betas)


if __name__ == '__main__':
    with open('config/config_batchnorm.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    parser = argparse.ArgumentParser(description="Train a network on CIFAR-10 by PyTorch")
    parser.add_argument('--batch-size', '-b', type=int, default=config['batch_size'], help='Training batch size')
    parser.add_argument('--seed', '-s', default=config['seed'], type=int, help="Set random seed")
    args = parser.parse_args()
    main(config)
