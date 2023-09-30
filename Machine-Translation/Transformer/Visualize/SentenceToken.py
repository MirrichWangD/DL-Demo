# -*- coding: UTF-8 -*-
"""
+++++++++++++++++++++++++++++++++++
@ File        : SentenceToken.py
@ Time        : 2023/4/26 15:31
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (Conda)
+++++++++++++++++++++++++++++++++++
批次的句子 Token 可视化程序
依赖模块（需要自行安装，没有版本号默认最新）：
    torch 1.12.1+cu116
    torchtext 0.13.1
    spacy 3.5.0
    matplotlib
    seaborn
    pandas
    tqdm
+++++++++++++++++++++++++++++++++++
"""

from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, BatchSampler
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

"""+++++++++++++++++++++++++++
@@@ 全局变量
+++++++++++++++++++++++++++"""

# 定义特殊TOKEN
PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]
# 定义 spacy 语言模型库，用于分词
# 注意！运行时请确保输入的 src_lang 和 tgt_lang 能够在此查询到相对应的 Spacy 语言模块，否则会构造数据集时报错
SPACY = {"en": "en_core_web_sm", "zh": "zh_core_web_sm"}
LANGUAGE = {"en": 0, "zh": 1}

"""++++++++++++++++++++
@@@ Config
++++++++++++++++++++"""

plt.rcParams["font.family"] = ["Microsoft YaHei"]  # 使用微软雅黑字体
cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)

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
    """
    DataLoader 批获取数据时进行的处理函数
    Args:
        batch: iter 批数据 (src_token(S_src), tgt_token(S_tgt))

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


"""++++++++++++++++++++
@@@ mpl绘图
++++++++++++++++++++"""

# 构造数据集
dataset = TranslationDataset(file_path="./data/eng-zh.txt", src_lang="en", tgt_lang="zh")
print(dataset)

# 划分训练、验证、测试集，分批
indices = np.arange(len(dataset))[::-1]
batch_sampler = BatchSampler(indices, 32, False)
# 生成数据迭代器
db = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0, collate_fn=collate_fn)

for src, tgt in db:
    if src.shape[0] == 16:
        fig, ax = plt.subplots(1, 4, figsize=(6, 4))
        fig1, ax1 = plt.subplots(1, 4, figsize=(6, 4))
        for idx in range(0, 4):
            # 获取数据，生成句子列表
            src_token = src[:, idx * 8 : (idx * 8) + 1]
            src_sentence = dataset.src_vocab.lookup_tokens(src_token.flatten().tolist())
            tgt_token = tgt[:, idx * 8 : (idx * 8) + 1]
            tgt_sentence = dataset.tgt_vocab.lookup_tokens(tgt_token.flatten().tolist())

            # 绘制热力图
            sns.heatmap(
                src_token,
                ax=ax[idx],
                cbar=False,
                annot=True,
                fmt=".0f",
                xticklabels=[idx],
                cmap=cmap,
            )
            sns.heatmap(
                tgt_token,
                ax=ax1[idx],
                cbar=False,
                annot=True,
                fmt=".0f",
                xticklabels=[idx],
                cmap=cmap,
            )
            # 设置坐标轴为词
            ax[idx].set_yticks(
                np.linspace(0.5, src.shape[0] - 0.5, src.shape[0]),
                rotation=0,
                labels=src_sentence,
            )
            ax1[idx].set_yticks(
                np.linspace(0.5, tgt.shape[0] - 0.5, tgt.shape[0]),
                rotation=0,
                labels=tgt_sentence,
            )

        # 保存图片
        fig.suptitle("输入Token向量")
        fig.tight_layout()
        fig.savefig("imgs/Token-src.png")

        fig1.suptitle("输出Token向量")
        fig1.tight_layout()
        fig1.savefig("imgs/Token-tgt.png")
        break
