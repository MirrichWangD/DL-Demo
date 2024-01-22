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
+++++++++++++++++++++++++++++++++++
"""

# 导入基础模块
import math
import warnings

# 导入依赖模块
from tqdm import tqdm
import pandas as pd
import numpy as np

# 导入绘图相关模块
import seaborn as sns
import matplotlib.pyplot as plt

# 导入 torch 相关模块
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, BatchSampler
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


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
# 设置绘图参数
plt.rcParams["font.family"] = ["Microsoft YaHei"]  # 使用微软雅黑字体
cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)


"""++++++++++++++++++++
@@@ 位置编码层
++++++++++++++++++++"""


class PositionalEncoding(nn.Module):
    """位置编码层"""

    def __init__(self, emb_size: int, dropout: float, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        # self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[: token_embedding.size(0), :])


"""++++++++++++++++++++
@@@ 构建数据集
++++++++++++++++++++"""


class TranslationDataset(Dataset):
    """机器翻译数据集"""

    def __init__(self, file_path: str, src_lang: str = "en", tgt_lang: str = "zh"):
        """

        Args:
            file_path: str <src>\t<tgt>\n 一行的txt文件地址
            src_lang: str 原始数据列索引
            tgt_lang: str 翻译数据列索引
        """
        super().__init__()
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_sentences = []
        self.tgt_sentences = []

        # 读取原始数据，获取长度
        texts = pd.read_table(file_path, header=None)
        self.length = texts.shape[0]

        # 通过 spacy 获取语言模型
        self.src_tokenizer = get_tokenizer("spacy", language=SPACY[src_lang])
        self.tgt_tokenizer = get_tokenizer("spacy", language=SPACY[tgt_lang])

        # 迭代原始数据，进行分词处理
        data = texts[[LANGUAGE[src_lang], LANGUAGE[tgt_lang]]].values
        for src, tgt in tqdm(data, total=self.length, desc="Loading Data"):
            self.src_sentences.append(self.src_tokenizer(src))
            self.tgt_sentences.append(self.tgt_tokenizer(tgt))

        # 构建词汇表对象
        self.src_vocab = build_vocab_from_iterator(self.src_sentences, 1, specials=SPECIALS)
        self.tgt_vocab = build_vocab_from_iterator(self.tgt_sentences, 1, specials=SPECIALS)

    def __len__(self):
        """数据集整体长度"""
        return self.length

    def __repr__(self):
        """字符串可视化显示数据集信息"""
        return (
            " Dataset Info ".center(50, "=")
            + "\n"
            + "| %-21s | %-22s |\n" % ("size", self.length)
            + "| %-21s | %-22s |\n" % (f"src vocab: {self.src_lang}", len(self.src_vocab))
            + "| %-21s | %-22s |\n" % (f"tgt vocab: {self.tgt_lang}", len(self.tgt_vocab))
            + "=" * 50
        )

    def __getitem__(self, idx):
        """根据索引 idx 获取 src、tgt 的 tokens"""
        # 通过 vocab 获取 token，并且前后插入起始、终止符号
        src = [BOS_IDX] + self.src_vocab.lookup_indices(self.src_sentences[idx]) + [EOS_IDX]
        tgt = [BOS_IDX] + self.tgt_vocab.lookup_indices(self.tgt_sentences[idx]) + [EOS_IDX]
        return torch.tensor(src), torch.tensor(tgt)


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
    # 可视化 Positional Encoding
    PE = PositionalEncoding(emb_size=512, dropout=0.1)
    # 保存图片
    sns.heatmap(PE.pos_embedding[:1024, 0], cmap=cmap)
    plt.title("Positional Encoding ($d_{model}=512$)")
    plt.ylabel("Length")
    plt.xlabel("$d_{model}=512$")
    plt.tight_layout()
    plt.savefig("imgs/positional_encoding.png")

    # 构造数据集
    dataset = TranslationDataset(file_path="../Tab-Separator/data/eng-zh.txt", src_lang="en", tgt_lang="zh")
    print(dataset)

    # 划分训练、验证、测试集，分批
    indices = np.arange(len(dataset))[::-1]
    batch_sampler = BatchSampler(indices, 32, False)
    # 生成数据迭代器
    db = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0, collate_fn=collate_fn)

    for src, tgt in db:
        if src.shape[0] == 16:
            # 生成绘图对象
            fig = [plt.subplots(1, 1) for _ in range(2)]
            figs = [plt.subplots(1, 4, figsize=(6, 4)) for _ in range(4)]

            # 生成 mask
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt[:-1, :])

            sns.heatmap(src_mask, ax=fig[0][1], cbar=False, annot=True, fmt=".0f")
            sns.heatmap(tgt_mask, ax=fig[1][1], cbar=False, annot=True, fmt=".0f")

            for idx in range(0, 4):
                # 获取数据，生成句子列表
                src_token = src[:, idx * 8 : (idx * 8) + 1]
                src_sentence = dataset.src_vocab.lookup_tokens(src_token.flatten().tolist())
                tgt_token = tgt[:, idx * 8 : (idx * 8) + 1]
                tgt_sentence = dataset.tgt_vocab.lookup_tokens(tgt_token.flatten().tolist())

                # %% 词向量可视化绘图
                sns.heatmap(
                    src_token,
                    ax=figs[0][1][idx],
                    cbar=False,
                    annot=True,
                    fmt=".0f",
                    xticklabels=[idx],
                    cmap=cmap,
                )
                sns.heatmap(
                    tgt_token,
                    ax=figs[1][1][idx],
                    cbar=False,
                    annot=True,
                    fmt=".0f",
                    xticklabels=[idx],
                    cmap=cmap,
                )
                # 设置坐标轴为词
                figs[0][1][idx].set_yticks(
                    np.linspace(0.5, src.shape[0] - 0.5, src.shape[0]),
                    rotation=0,
                    labels=src_sentence,
                )
                figs[1][1][idx].set_yticks(
                    np.linspace(0.5, tgt.shape[0] - 0.5, tgt.shape[0]),
                    rotation=0,
                    labels=tgt_sentence,
                )

                # %% 绘制 mask 相关图片
                sns.heatmap(
                    src_padding_mask[idx].unsqueeze(1),
                    ax=figs[2][1][idx],
                    cbar=False,
                    annot=True,
                    fmt=".0f",
                    xticklabels=[idx],
                    cmap=cmap,
                )
                sns.heatmap(
                    tgt_padding_mask[idx].unsqueeze(1),
                    ax=figs[3][1][idx],
                    cbar=False,
                    annot=True,
                    fmt=".0f",
                    xticklabels=[idx],
                    cmap=cmap,
                )
                # 设置坐标轴为词
                figs[2][1][idx].set_yticks(
                    np.linspace(0.5, src.shape[0] - 0.5, src.shape[0]),
                    rotation=0,
                    labels=src_sentence,
                )
                figs[3][1][idx].set_yticks(
                    np.linspace(0.5, tgt[:-1, :].shape[0] - 0.5, tgt[:-1, :].shape[0]),
                    rotation=0,
                    labels=tgt_sentence[:-1],
                )

            # 保存 Padding Mask 可视化图片
            fig[0][0].suptitle("Source Mask (en)")
            fig[0][0].tight_layout()
            fig[0][0].savefig("imgs/src_mask.png")

            fig[1][0].suptitle("Target Mask (zh)")
            fig[1][0].tight_layout()
            fig[1][0].savefig("imgs/tgt_mask.png")

            # 保存 Sentence Token 可视化图片
            figs[0][0].suptitle("Source Token (en)")
            figs[0][0].tight_layout()
            figs[0][0].savefig("imgs/src_token.png")

            figs[1][0].suptitle("Target Token (zh)")
            figs[1][0].tight_layout()
            figs[1][0].savefig("imgs/tgt_token.png")

            # 保存 Sentence Mask 可视化图片
            figs[2][0].suptitle("Source Padding Mask (en)")
            figs[2][0].tight_layout()
            figs[2][0].savefig("imgs/src_padding_mask.png")

            figs[3][0].suptitle("Target Padding Mask (zh)")
            figs[3][0].tight_layout()
            figs[3][0].savefig("imgs/tgt_padding_mask.png")
            break
