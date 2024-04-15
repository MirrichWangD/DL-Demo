# -*- coding: utf-8 -*-
"""
+++++++++++++++++++++++++++++++++++
@ File        : utils.py
@ Time        : 2024/04/15 3:24:10
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (Conda)
+++++++++++++++++++++++++++++++++++
...
+++++++++++++++++++++++++++++++++++
"""

# 导入基础模块
from typing_extensions import List, Tuple

# 导入依赖模块
import opencc

# 导入可视化模块
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch


def token_heatmap(
    tensor: List[torch.Tensor],
    sentences: List[List[str]] = None,
    ncols: int = 3,
    figsize: Tuple[int, int] = (7, 7),
    cbar: bool = True,
    annot: bool = True,
    fmt: str = ".0f",
    cmap=None,
):
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    for i in range(ncols):
        sns.heatmap(tensor[i], cbar=cbar, annot=annot, fmt=fmt, cmap=cmap, ax=axes[i])
        if sentences is not None:
            ticks = np.linspace(0.5, len(sentences[i]) - 0.5, len(sentences[i]))
            axes[i].set_yticks(ticks, rotation=0, labels=sentences[i])

    return fig, axes


def heatmap(
    tensor: torch.Tensor,
    sentence: Tuple[List[str], List[str]] = None,
    figsize: tuple = (7, 7),
    cbar: bool = True,
    annot: bool = True,
    fmt: str = ".0f",
    cmap=None,
):

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.heatmap(tensor, cbar=cbar, annot=annot, fmt=fmt, cmap=cmap, ax=ax)
    if sentence:
        if len(sentence[0]) != 0:
            ticks = np.linspace(0.5, len(sentence[0]) - 0.5, len(sentence[0]))
            ax.set_xticks(ticks, rotation=90, labels=sentence[0])
        if len(sentence[1]) != 0:
            ticks = np.linspace(0.5, len(sentence[1]) - 0.5, len(sentence[1]))
            ax.set_yticks(ticks, rotation=0, labels=sentence[1])

    return fig, ax


def heatmaps(
    tensor: list,
    sentence: Tuple[List[str], List[str]] = None,
    nums: tuple = (1, 1),
    titles: List[str] = None,
    figsize: tuple = (7, 7),
    cbar: bool = True,
    annot: bool = True,
    fmt: str = ".0f",
    cmap=None,
):
    fig, axes = plt.subplots(nums[0], nums[1], figsize=figsize)
    for i in range(len(tensor)):
        if nums[0] == 1:
            ax = axes[i]
        else:
            ax = axes[i // nums[1], i % nums[1]]
        sns.heatmap(tensor[i], cbar=cbar, annot=annot, fmt=fmt, cmap=cmap, ax=ax)
        if titles:
            ax.set_title(titles[i])
        if sentence:
            if len(sentence[0]) != 0:
                ticks = np.linspace(0.5, len(sentence[0]) - 0.5, len(sentence[0]))
                ax.set_xticks(ticks, rotation=90, labels=sentence[0])
            if len(sentence[1]) != 0:
                ticks = np.linspace(0.5, len(sentence[1]) - 0.5, len(sentence[1]))
                ax.set_yticks(ticks, rotation=0, labels=sentence[1])

    return fig, axes


def t2s(s):
    cc = opencc.OpenCC("t2s")

    return cc.convert(s)
