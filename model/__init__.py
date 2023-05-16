#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/5/11 11:45
# @File     : __init__.py.py
# @Project  : lab

from .resnet import (
    resNet18,
    resNet34,
    resNet50,
    resNet101,
    resNet152,
    resNeXt50_32x4d,
    resNeXt101_32x8d,
    wide_resNet50_2,
    wide_resNet101_2,
)

from .densenet import (
    denseNet121,
    denseNet169,
    denseNet201,
    denseNet161,
    denseNet_cifar,
)

from .dla import DLA
from .dpn import (
    DPN,
    dpn26,
    dpn92,
)
