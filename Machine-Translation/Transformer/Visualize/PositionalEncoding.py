# -*- coding: UTF-8 -*-
"""
+++++++++++++++++++++++++++++++++++
@ File        : PositionalEncoding.py
@ Time        : 2023/4/26 15:14
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (Conda)
+++++++++++++++++++++++++++++++++++
位置编码层可视化
依赖模块（需要自行安装，没有版本号默认最新）：
    torch 1.12.1+cu116
    matplotlib
    seaborn
+++++++++++++++++++++++++++++++++++
"""

import math
import torch
import torch.nn as nn

import seaborn as sns
import matplotlib.pyplot as plt

"""++++++++++++++++++++
@@@ Config
++++++++++++++++++++"""

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


d_model = 512
pe = PositionalEncoding(d_model, 0.1)

sns.heatmap(pe.pos_embedding[:d_model, 0], cmap=cmap)
plt.savefig("imgs/PositionalEncoding.png")
