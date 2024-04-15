# -*- coding: utf-8 -*-
"""
+++++++++++++++++++++++++++++++++++
@ File        : visualize.py
@ Time        : 2024/01/22 14:30:58
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (Conda)
+++++++++++++++++++++++++++++++++++
针对 Transformer 的一些相关结构可视化脚本：
- Source Mask
- Source Padding Mask
- Target Mask
- Target Padding Mask
- Source Token
- Target Token
- Positional Encoding
指定 eng-zh.txt 文档的 29366 行 - 29370 行数据：
"Tom doesn't like to use the term ""a person of color"" because he thinks it implies that white people have no color."
汤姆不喜欢使用”有色人种“这个术语，因为他认为，根据这种说法白种人没有颜色。

If you don't want to put on sunscreen, that's your problem. Just don't come complaining to me when you get a sunburn.
你不想涂防晒霜是你的问题，但是晒伤了不要来抱怨。

Even now, I occasionally think I'd like to see you. Not the you that you are today, but the you I remember from the past.
即使是现在，我偶尔还是想见到你。不是今天的你，而是我记忆中曾经的你。

It's very easy to sound natural in your own native language, and very easy to sound unnatural in your non-native language.
你很容易把母语说得通顺流畅，却很容易把非母语说得不自然。

I got fired from the company, but since I have a little money saved up, for the time being, I won't have trouble with living expenses.
虽然我被公司解雇了，但是我还有点存款，所以目前不用担心生计问题。
+++++++++++++++++++++++++++++++++++
"""

# 导入基础模块
from pathlib import Path
from collections import OrderedDict
import warnings
import re

# 导入依赖模块
import pandas as pd
import numpy as np

# 导入绘图相关模块
import seaborn as sns
import matplotlib.pyplot as plt

# 导入 torch 相关模块
import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.data import get_tokenizer
from torchtext.vocab import vocab

"""+++++++++++++++++++++++++++
@@@ Config
+++++++++++++++++++++++++++"""

warnings.filterwarnings("ignore")
# 定义特殊TOKEN
PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]
# 定义 spacy 语言模型库，用于分词
# 注意！运行时请确保输入的 src_lang 和 tgt_lang 能够在此查询到相对应的 Spacy 语言模块，否则会构造数据集时报错
SPACY = {"en": "en_core_web_sm", "zh": "zh_core_web_sm"}
LANGUAGE = {"en": 0, "zh": 1}


"""++++++++++++++++++++++
@@@ Batch采样处理
++++++++++++++++++++++"""


def collate_fn(batch):
    """批次数据处理函数

    Args:
        batch (iter): 批次数据 (src_token(S_src), tgt_token(S_tgt))

    Returns:
        Tensor(S_src, N), Tensor(S_tgt, N)
    """
    src_batch, tgt_batch = [], []
    for src_token, tgt_token in batch:
        src_batch.append(src_token)
        tgt_batch.append(tgt_token)
    # 根据批次数据的最大长度，进行自动填充
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


"""++++++++++++++++++++++
@@@ Mask处理
++++++++++++++++++++++"""


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """
    生成方形后续掩码
    Args:
        sz: int 生成方形掩码的尺寸
        device: torch.device 运算硬件

    Returns:
        torch.tensor
    """
    # 生成倒三角全为 0 的矩阵
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    # 将上三角全部使用 -inf 填充（不包括对角线）
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src: torch.Tensor, tgt: torch.Tensor) -> object:
    """
    根据 src、tgt 生成 src_mask、tgt_mask、src_padding_mask、tgt_padding_mask
    Args:
        src: torch.Tensor 输入词索引向量 (S_src, N)
        tgt: torch.Tensor 输出词索引向量 (S_tgt, N)

    Returns:
        Tensor(S_src, S_src), Tensor(S_tgt, S_tgt), Tensor(N, S_src), Tensor(N, S_tgt)
    """
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    # 生成方阵掩码
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(src.device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(torch.bool)

    # 生成词索引 padding 的掩码
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


if __name__ == "__main__":
    # ----------- #
    # Config
    # ----------- #

    # 设置绘图参数
    plt.rcParams["font.family"] = ["Microsoft YaHei"]  # 使用微软雅黑字体
    cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
    # 输出文件夹
    output_dir = Path("imgs")
    output_dir.mkdir(exist_ok=True)

    # 数据参数
    data_dir = "../Tab-Separator/data/eng-zh.txt"
    src_dict = "../Tab-Separator/ckpt/en-zh/e6d6h8dm64df256ml40/words/src_dict.txt"
    tgt_dict = "../Tab-Separator/ckpt/en-zh/e6d6h8dm64df256ml40/words/tgt_dict.txt"

    # ------------ #
    # Loading Data
    # ------------ #

    # 读取分词器
    src_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
    tgt_tokenizer = get_tokenizer("spacy", language="zh_core_web_sm")
    # 读取 source 词典
    with open(src_dict, encoding="utf-8") as f:
        src_words = list(map(lambda i: i.strip(), f.readlines()))
    src_vocab = vocab(OrderedDict(zip(src_words, [10] * len(src_words))))
    # 读取 target 词典
    with open(tgt_dict, encoding="utf-8") as f:
        tgt_words = list(map(lambda i: i.strip(), f.readlines()))
    tgt_vocab = vocab(OrderedDict(zip(tgt_words, [10] * len(tgt_words))))

    # 读取数据文件
    texts = pd.read_table(data_dir, header=None, encoding="utf-8")
    # 指定特定索引数据（ eng-zh.txt 文件的 29366 行 - 29370 行）
    data_idx = [29365, 29366, 29367, 29368, 29369]
    # 将句子进行分词，转换成 Tensor
    data = []
    for idx in data_idx:
        src_sent = src_tokenizer(re.sub(r"\s+", " ", texts.iloc[idx][0]))
        tgt_sent = tgt_tokenizer(re.sub(r"\s+", " ", texts.iloc[idx][1]))
        src_token = torch.tensor([BOS_IDX] + src_vocab.lookup_indices(src_sent) + [EOS_IDX])
        tgt_token = torch.tensor([BOS_IDX] + tgt_vocab.lookup_indices(tgt_sent) + [EOS_IDX])
        data.append((src_token, tgt_token))

    # batch 处理
    src, tgt = collate_fn(data)
    # 词向量转换成句子
    sentences = []
    for i in range(len(data_idx)):
        src_word = src_vocab.lookup_tokens(src[:, i].flatten().tolist())
        tgt_word = tgt_vocab.lookup_tokens(tgt[:, i].flatten().tolist())
        sentences.append((src_word, tgt_word))

    # 创建对应的mask
    tgt_input = tgt[:-1, :]
    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

    # 生成绘图对象
    fig = [plt.subplots(1, 1, figsize=(8, 8)) for _ in range(2)]
    figs = [plt.subplots(1, 5, figsize=(7, 7)) for _ in range(4)]

    sns.heatmap(src_mask, ax=fig[0][1], cbar=False, annot=True, fmt=".0f")
    sns.heatmap(tgt_mask, ax=fig[1][1], cbar=False, annot=True, fmt=".0f")

    for i in range(len(data_idx)):
        # 处理部分需要的数据
        src_token = src[:, i : i + 1]
        src_sentence = src_vocab.lookup_tokens(src_token.flatten().tolist())
        tgt_token = tgt[:, i : i + 1]
        tgt_sentence = tgt_vocab.lookup_tokens(tgt_token.flatten().tolist())

        # 词向量可视化绘图
        sns.heatmap(
            src_token,
            ax=figs[0][1][i],
            cbar=False,
            annot=True,
            fmt=".0f",
            xticklabels=[i],
            cmap=cmap,
        )
        sns.heatmap(
            tgt_token,
            ax=figs[1][1][i],
            cbar=False,
            annot=True,
            fmt=".0f",
            xticklabels=[i],
            cmap=cmap,
        )
        # 设置坐标轴为词
        figs[0][1][i].set_yticks(
            np.linspace(0.5, src.shape[0] - 0.5, src.shape[0]),
            rotation=0,
            labels=src_sentence,
        )
        figs[1][1][i].set_yticks(
            np.linspace(0.5, tgt.shape[0] - 0.5, tgt.shape[0]),
            rotation=0,
            labels=tgt_sentence,
        )

        # 绘制 mask 相关图片
        sns.heatmap(
            src_padding_mask[i].unsqueeze(1),
            ax=figs[2][1][i],
            cbar=False,
            annot=True,
            fmt=".0f",
            xticklabels=[i],
            cmap=cmap,
        )
        sns.heatmap(
            tgt_padding_mask[i].unsqueeze(1),
            ax=figs[3][1][i],
            cbar=False,
            annot=True,
            fmt=".0f",
            xticklabels=[i],
            cmap=cmap,
        )
        # 设置坐标轴为词
        figs[2][1][i].set_yticks(
            np.linspace(0.5, src.shape[0] - 0.5, src.shape[0]),
            rotation=0,
            labels=src_sentence,
        )
        figs[3][1][i].set_yticks(
            np.linspace(0.5, tgt[:-1, :].shape[0] - 0.5, tgt[:-1, :].shape[0]),
            rotation=0,
            labels=tgt_sentence[:-1],
        )

    # 保存 Padding Mask 可视化图片
    fig[0][0].suptitle("Source Mask (en)")
    fig[0][0].tight_layout()
    fig[0][0].savefig(output_dir / "mask_src_token.png")

    fig[1][0].suptitle("Target Mask (zh)")
    fig[1][0].tight_layout()
    fig[1][0].savefig(output_dir / "mask_tgt_token.png")

    # 保存 Sentence Token 可视化图片
    figs[0][0].suptitle("Source Token (en)")
    figs[0][0].tight_layout()
    figs[0][0].savefig(output_dir / "data_src_token.png")

    figs[1][0].suptitle("Target Token (zh)")
    figs[1][0].tight_layout()
    figs[1][0].savefig(output_dir / "data_tgt_token.png")

    # 保存 Sentence Mask 可视化图片
    figs[2][0].suptitle("Source Padding Mask (en)")
    figs[2][0].tight_layout()
    figs[2][0].savefig(output_dir / "mask_src_padding.png")

    figs[3][0].suptitle("Target Padding Mask (zh)")
    figs[3][0].tight_layout()
    figs[3][0].savefig(output_dir / "mask_tgt_padding.png")
