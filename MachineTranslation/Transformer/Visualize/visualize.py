# -*- coding: utf-8 -*-
"""
+++++++++++++++++++++++++++++++++++
@ File        : visualize.py
@ Time        : 2024/04/09 22:06:23
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (Conda)
+++++++++++++++++++++++++++++++++++
...
+++++++++++++++++++++++++++++++++++
"""

# 导入基础模块
from typing import Optional, Tuple, List
from collections import OrderedDict
from pathlib import Path
import warnings
import math
import re

# 导入依赖模块
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 导入torch相关模块
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import vocab
from torchtext.data import get_tokenizer

# 设置忽略warning信息
warnings.filterwarnings("ignore")
# 设置绘图参数
sns.set(font_scale=1.25)
plt.rcParams["font.family"] = ["Microsoft YaHei"]  # 使用微软雅黑字体

# 定义特殊TOKENdd
PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]
# 定义 spacy 语言模型库，用于分词，该部分可以自行增加
# 注意！运行时请确保输入的 src_lang 和 tgt_lang 能够在此查询到相对应的 Spacy 语言模块，否则会构造数据集时报错
SPACY = {
    "de": "de_core_news_sm",  # German 德语
    "en": "en_core_web_sm",  # English 英语
    "zh": "zh_core_web_sm",  # Chinese 简中
    "fr": "fr_core_news_sm",  # French 法语
}

"""++++++++++++++++++
@@@ Config
++++++++++++++++++"""

# 模型参数
n_enc_layers = 6
n_dec_layers = 6
n_heads = 8
d_model = 512
d_ff = 2048
dropout = 0.1
max_len = 5000
resume = "../Tab-Separator/ckpt/en-de/e6d6h8dm512df2048ml5000/best_model.pth"

# 数据参数
data_dir = "../Tab-Separator/data/eng-deu.txt"
src_dict = "../Tab-Separator/ckpt/en-de/e6d6h8dm512df2048ml5000/words/src_dict.txt"
tgt_dict = "../Tab-Separator/ckpt/en-de/e6d6h8dm512df2048ml5000/words/tgt_dict.txt"

# cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
cmap = "RdBu_r"

# 设置训练硬件
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 获取输出目录
output_dir = Path("imgs")
output_dir.mkdir(exist_ok=True)

"""++++++++++++++++++++++
@@@ Heatmap 自定义函数
++++++++++++++++++++++"""


def heatmap(
    data: Optional[List],
    titles: List[str] = None,
    x_ticks: List[List[str]] = "auto",
    y_ticks: List[List[str]] = "auto",
    square: bool = False,
    n_rows: int = 1,
    n_cols: int = 1,
    fig_size: Tuple[int, int] = (15, 10),
    cbar: bool = True,
    annot: bool = False,
    fmt: str = ".0f",
    cmap=None,
) -> object:
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    for i in range(len(data)):
        if n_rows * n_cols == 1:
            ax = axes
        elif n_rows == 1 or n_cols == 1:
            ax = axes[i]
        else:
            ax = axes[i // n_cols, i % n_cols]

        sns.heatmap(
            data[i],
            cbar=cbar,
            annot=annot,
            fmt=fmt,
            ax=ax,
            square=square,
            cmap=cmap,
            annot_kws={"size": 10},
            xticklabels=x_ticks[i],
            yticklabels=y_ticks[i],
        )

        if titles:
            ax.set_title(titles[i])

    return fig, axes


"""++++++++++++++++++++++
@@@ Mask处理
++++++++++++++++++++++"""


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """
    生成方形 Attention 掩码
    Args:
        sz: int 生成方形掩码的尺寸

    Returns:
        torch.tensor
    """
    # 生成倒三角全为 0 的矩阵
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    # 将上三角全部使用 -inf 填充（不包括对角线）
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(source: torch.Tensor, target: torch.Tensor) -> object:
    """
    根据 src、tgt 生成 src_mask、tgt_mask、src_padding_mask、tgt_padding_mask
    Args:
        source: torch.Tensor 输入词索引向量 (S_src, N)
        target: torch.Tensor 输出词索引向量 (S_tgt, N)

    Returns:
        Tensor(S_src, S_src), Tensor(S_tgt, S_tgt), Tensor(N, S_src), Tensor(N, S_tgt)
    """
    src_seq_len = source.shape[0]
    tgt_seq_len = target.shape[0]

    # 生成方阵掩码
    target_mask = generate_square_subsequent_mask(tgt_seq_len).to(source.device)
    source_mask = torch.zeros((src_seq_len, src_seq_len), device=source.device).type(torch.bool)

    # 生成词索引 padding 的掩码
    source_padding_mask = (source == PAD_IDX).transpose(0, 1)
    target_padding_mask = (target == PAD_IDX).transpose(0, 1)
    return source_mask, target_mask, source_padding_mask, target_padding_mask


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
    for source, target in batch:
        src_batch.append(source)
        tgt_batch.append(target)
    # 根据批次数据的最大长度，进行自动填充
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


"""+++++++++++++++++++
@@@ Model Structure
+++++++++++++++++++"""


class PositionalEncoding(nn.Module):
    """位置编码层"""

    def __init__(self, emb_size: int, dropout: float, max_length: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_length).reshape(max_length, 1)
        pos_embedding = torch.zeros((max_length, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[: token_embedding.size(0), :])


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    """词向量嵌入层"""

    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Seq2Seq Network
class Transformer(nn.Module):
    """Transformer Seq2Seq模型"""

    def __init__(
        self,
        num_enc_layers: int,
        num_dec_layers: int,
        d_model: int,
        n_head: int,
        max_length: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super(Transformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_head,
            num_encoder_layers=num_enc_layers,
            num_decoder_layers=num_dec_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        # 输出全连接层：由于使用的 nn.CrossEntropyLoss 因此不需要作 log_softmax 处理
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout, max_length=max_length)

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_padding_mask: torch.Tensor,
        tgt_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src_emb,  # src
            tgt_emb,  # tgt_input
            src_mask,  # src_mask
            tgt_mask,  # tgt_mask
            None,  # memory_mask=None (attn_mask)
            src_padding_mask,  # src_padding_mask
            tgt_padding_mask,  # tgt_padding_mask
            memory_key_padding_mask,  # src_padding_mask
        )
        return self.generator(outs)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)


"""++++++++++++++++++++++++
@@@ Load Data & init model
++++++++++++++++++++++++"""

# 读取分词器
src_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
tgt_tokenizer = get_tokenizer("spacy", language="de_core_news_sm")
# 读取 source 词典
with open(src_dict, encoding="utf-8") as f:
    src_words = list(map(lambda i: i.strip(), f.readlines()))
src_vocab = vocab(OrderedDict(zip(src_words, [10] * len(src_words))))
# 读取 target 词典
with open(tgt_dict, encoding="utf-8") as f:
    tgt_words = list(map(lambda i: i.rstrip(), f.readlines()))
tgt_vocab = vocab(OrderedDict(zip(tgt_words, [10] * len(tgt_words))))

# 读取数据文件
texts = pd.read_table(data_dir, header=None, encoding="utf-8")
# 指定特定索引数据（ eng-zh.txt 文件的 29366 行 - 29370 行）
data_idx = [29365, 29366, 29367, 29368, 29369]
# 将句子进行分词，转换成 Tensor
data = []
for idx in data_idx:
    src_word = src_tokenizer(texts.iloc[idx][0])
    tgt_word = tgt_tokenizer(texts.iloc[idx][1])
    src_token = torch.tensor([BOS_IDX] + src_vocab.lookup_indices(src_word) + [EOS_IDX])
    tgt_token = torch.tensor([BOS_IDX] + tgt_vocab.lookup_indices(tgt_word) + [EOS_IDX])
    data.append((src_token, tgt_token))

# batch 处理
src, tgt = collate_fn(data)
# 词向量转换成句子
sentences = []
for i in range(len(data_idx)):
    src_word = src_vocab.lookup_tokens(src[:, i].flatten().tolist())
    tgt_word = tgt_vocab.lookup_tokens(tgt[:, i].flatten().tolist())
    sentences.append((src_word, tgt_word))
# 转移至运算设备
src, tgt = src.to(device), tgt.to(device)
# 创建对应的mask
tgt_input = tgt[:-1, :]
src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

# 模型初始化和读取权重
model = Transformer(
    n_enc_layers,
    n_dec_layers,
    d_model,
    n_heads,
    max_len,
    len(src_vocab),
    len(tgt_vocab),
    d_ff,
    dropout,
).to(device)

if resume:
    resume = torch.load(resume)
    model.load_state_dict(resume["model"])


"""+++++++++++++++++++
@@@ Token 可视化
+++++++++++++++++++"""

fig_src, ax_src = heatmap(
    [src[:, i : i + 1].cpu() for i in range(len(data_idx))],
    x_ticks=[[str(i)] for i in range(5)],
    y_ticks=[sent[0] for sent in sentences],
    n_cols=5,
    fig_size=(10, 10),
    cbar=False,
    annot=True,
    fmt="d",
    cmap=cmap,
)
fig_src.suptitle("Source Token (en)")
fig_src.tight_layout()
fig_src.savefig(output_dir / "data_src_token.png")

fig_tgt, ax_tgt = heatmap(
    [tgt[:, i : i + 1].cpu() for i in range(len(data_idx))],
    x_ticks=[[str(i)] for i in range(5)],
    y_ticks=[sent[1] for sent in sentences],
    n_cols=5,
    fig_size=(8, 8),
    cbar=False,
    annot=True,
    fmt="d",
    cmap=cmap,
)
fig_tgt.suptitle("Target Token (zh)")
fig_tgt.tight_layout()
fig_tgt.savefig(output_dir / "data_tgt_token.png")


"""++++++++++++++++++++++++++++++
@@@ Positional Encoding 可视化
++++++++++++++++++++++++++++++"""

fig_pe, ax_pe = heatmap(
    [model.positional_encoding.pos_embedding.cpu()[:40, 0, :64]],
    fig_size=(32, 10),
    x_ticks=[True],
    y_ticks=[True],
    cbar=True,
    annot=True,
    fmt=".2f",
    cmap=cmap,
)
fig_pe.suptitle("Positional Encoding")
fig_pe.tight_layout()
fig_pe.savefig(output_dir / "emb_positional_encoding.png")

"""++++++++++++++++++++++++++++++
@@@ Token Embeddings 可视化
++++++++++++++++++++++++++++++"""

# 词向量 Embeddings
src_emb = model.src_tok_emb(src)
tgt_emb = model.tgt_tok_emb(tgt_input)

fig_src_emb, ax_src_emb = heatmap(
    [src_emb.cpu().detach()[:, 0, :64]],
    x_ticks=[True],
    y_ticks=[sentences[0][0]],
    fig_size=(36, 6),
    cbar=True,
    annot=True,
    fmt=".2f",
    cmap=cmap,
)
fig_src_emb.suptitle("Source Embeddings")
fig_src_emb.tight_layout()
fig_src_emb.savefig(output_dir / "emb_src.png")

"""++++++++++++++++++++++++++++++
@@@ Positional Embeddings 可视化
++++++++++++++++++++++++++++++"""

# 词向量嵌入和位置编码累加
src_pos_emb = model.positional_encoding(src_emb)
tgt_pos_emb = model.positional_encoding(tgt_emb)

fig_src_pos_emb, ax_src_pos_emb = heatmap(
    [src_pos_emb[:, 0, :64].cpu().detach()],
    y_ticks=[sentences[0][0]],
    fig_size=(36, 10),
    cbar=True,
    annot=True,
    fmt=".2f",
    cmap=cmap,
)
fig_src_pos_emb.suptitle("Source Positional Embeddings")
fig_src_pos_emb.tight_layout()
fig_src_pos_emb.savefig(output_dir / "emb_src_pos.png")

"""+++++++++++++++++++++++++++++++++++++++++++++++
@@@ encoder 推理
+++++++++++++++++++++++++++++++++++++++++++++++"""

attn_mask = src_mask
key_padding_mask = src_padding_mask
last_attn = model.transformer.encoder.layers[-1].self_attn

out = src_pos_emb
for mod in model.transformer.encoder.layers[:-1]:
    out = mod(out, src_mask=src_mask, src_key_padding_mask=src_padding_mask)

query, key, value = out, out, out

# set sup shape vars
tgt_len, bsz, embed_dim = query.shape
src_len, _, _ = key.shape

head_dim = embed_dim // n_heads

#
# compute in-projection
#
q, k, v = F._in_projection_packed(query, key, value, last_attn.in_proj_weight, last_attn.in_proj_bias)

# prep attention mask
attn_mask = attn_mask.unsqueeze(0)

#
# reshape q, k, v for multihead attention and make em batch first
#
q = q.contiguous().view(tgt_len, bsz * n_heads, head_dim).transpose(0, 1)
k = k.contiguous().view(k.shape[0], bsz * n_heads, head_dim).transpose(0, 1)
v = v.contiguous().view(v.shape[0], bsz * n_heads, head_dim).transpose(0, 1)

# update source sequence length after adjustments
src_len = k.size(1)

# merge key padding and attention masks
key_padding_mask = (
    key_padding_mask.view(bsz, 1, 1, src_len).expand(-1, n_heads, -1, -1).reshape(bsz * n_heads, 1, src_len)
)
attn_mask = attn_mask.logical_or(key_padding_mask)

# convert mask to float
new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
new_attn_mask.masked_fill_(attn_mask, float("-inf"))
attn_mask = new_attn_mask

#
# (deep breath) calculate attention and out projection
#
attn_output, attn_output_weights = F._scaled_dot_product_attention(q, k, v, attn_mask, 0.0)
attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
attn_output = F.linear(attn_output, last_attn.out_proj.weight, last_attn.out_proj.bias)
attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

attn_output_weights = attn_output_weights.view(bsz, n_heads, tgt_len, src_len)


# -------------------------------- #
# Multihead Attention Head Weight
# -------------------------------- #

for i in range(n_heads):
    fig_smha_head_weight, ax_smha_head_weight = heatmap(
        [attn_output_weights[0, i].cpu().detach()],
        x_ticks=[sentences[0][0]],
        y_ticks=[sentences[0][0]],
        fig_size=(20, 10),
        cmap=cmap,
        annot=True,
        fmt=".2f",
    )
    fig_smha_head_weight.suptitle("Head-%d" % i)
    plt.show()

# ------------------------------------ #
# Multihead Attention Weight (Average)
# ------------------------------------ #

attn_output_weights = attn_output_weights.sum(dim=1) / n_heads
for i in range(len(data_idx)):
    fig_smha_head_weight, ax_smha_head_weight = heatmap(
        [attn_output_weights[i].cpu().detach()],
        x_ticks=[sentences[i][0]],
        y_ticks=[sentences[i][0]],
        fig_size=(20, 10),
        cmap=cmap,
        annot=True,
        fmt=".2f",
    )
    fig_smha_head_weight.suptitle("Self-Attention Weight %d" % (i + 1))
    fig_smha_head_weight.tight_layout()
    fig_smha_head_weight.savefig(output_dir / "smha_weight")


"""+++++++++++++++++++++++++++++++++++++++++++++++
@@@ Decoder 第一层推理
+++++++++++++++++++++++++++++++++++++++++++++++"""

# Self-Attention
out = model.transformer.decoder.layers[0]._sa_block(tgt_pos_emb, tgt_mask, tgt_padding_mask)
out = model.transformer.decoder.layers[0].norm1(tgt_pos_emb + out)
# 获取 transformer.decoder 第一层权重信息
input_proj_weight = model.transformer.decoder.layers[0].multihead_attn.in_proj_weight  # q, k, v 映射的 weight
input_proj_bias = model.transformer.decoder.layers[0].multihead_attn.in_proj_bias  # q, k, v 映射的 bias
out_proj_weight = model.transformer.decoder.layers[0].multihead_attn.out_proj.weight  # concat 部分的 weight
out_proj_bias = model.transformer.decoder.layers[0].multihead_attn.out_proj.bias  # concat 部分的 bias

# 由于 q 和 k,v 输入不同，因此将输入权重进行拆分
E = memory.size(-1)
w_q, w_kv = input_proj_weight.split([E, E * 2])
b_q, b_kv = input_proj_bias.split([E, E * 2])
# 对输入矩阵进行线性映射，将为到多头输入举证中
q = F.linear(out, w_q, b_q)
k, v = F.linear(memory, w_kv, b_kv).chunk(2, dim=-1)
# 将 q, k, v 分成多个头，形状为 (batch * num_heads, src_len, embed_dim // num_heads)
q = q.contiguous().view(tgt_len, bsz * n_heads, head_dim).transpose(0, 1)
k = k.contiguous().view(k.shape[0], bsz * n_heads, head_dim).transpose(0, 1)
v = v.contiguous().view(v.shape[0], bsz * n_heads, head_dim).transpose(0, 1)

# 生成 attn_mask
key_padding_mask = (
    src_padding_mask.view(bsz, 1, 1, src_len).expand(-1, n_heads, -1, -1).reshape(bsz * n_heads, 1, src_len)
)
attn_mask = key_padding_mask
new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
new_attn_mask.masked_fill_(attn_mask, float("-inf"))
attn_mask = new_attn_mask
# MHA 计算
mha_output, mha_output_weights = scaled_dot_product_attention(q, k, v, attn_mask)
mha_output_weights = mha_output_weights.view(bsz, n_heads, tgt_len, src_len)

"""++++++++++++++++++++++++++++++++++
@@@ Multi-Head Attention mask 可视化
++++++++++++++++++++++++++++++++++"""

attn_mask_vis = attn_mask.view(bsz, n_heads, -1, src_len)
fig_mha_mask, ax_mha_mask = heatmap(
    [attn_mask_vis[i, 0].cpu() for i in range(len(data_idx))],
    x_ticks=([sent[0] for sent in sentences], 90),
    n_rows=5,
    fig_size=(16, 8),
    cbar=False,
    annot=True,
    fmt=".0f",
    cmap=cmap,
)
fig_mha_mask.suptitle("Multi-Head Attention Mask")
fig_mha_mask.tight_layout()
fig_mha_mask.savefig(output_dir / "mha_mask.png")

"""+++++++++++++++++++++++++++++++++++++++++++++++
@@@ Multi-Head Attention 多头注意力权重 可视化
+++++++++++++++++++++++++++++++++++++++++++++++"""

fig_mha_head_weights, ax_mha_head_weights = heatmap(
    [mha_output_weights[0, i, :, :].cpu().detach() for i in range(n_heads)],
    x_ticks=([sentences[0][0] for _ in range(n_heads)], 90),
    y_ticks=([sentences[0][1][:-1] for _ in range(n_heads)], 0),
    n_rows=2,
    n_cols=4,
    titles=["head-%d" % (i + 1) for i in range(n_heads)],
    fig_size=(20, 12),
    cbar=True,
    annot=False,
    fmt=".2f",
    cmap=cmap,
)
fig_mha_head_weights.suptitle("Multi-Head Attention Weights")
fig_mha_head_weights.tight_layout()
fig_mha_head_weights.savefig(output_dir / "mha_head_weights.png")

"""+++++++++++++++++++++++++++++++++++++++++++++++
@@@ Multi-Head Attention 多头平均注意力权重 可视化
+++++++++++++++++++++++++++++++++++++++++++++++"""

fig_mha_weight, ax_mha_weight = heatmap(
    [(mha_output_weights.sum(dim=1) / n_heads)[0].cpu().detach()],
    x_ticks=([sentences[0][0]], 90),
    y_ticks=([sentences[0][1][:-1]], 0),
    fig_size=(16, 10),
    cbar=True,
    annot=True,
    fmt=".2f",
    cmap=cmap,
)
fig_mha_weight.suptitle("Multi-Head Attention Weight (Average)")
fig_mha_weight.tight_layout()
fig_mha_weight.savefig(output_dir / "mha_weight.png")

"""+++++++++++++++++++++++++++++++++++++++++++++++
@@@ MultiHead-Attention 推理结果 可视化
+++++++++++++++++++++++++++++++++++++++++++++++"""

mha_output = mha_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
mha_output = F.linear(mha_output, out_proj_weight, out_proj_bias)
mha_output = mha_output.view(tgt_len, bsz, mha_output.size(1))

fig_mha_output, ax_mha_output = heatmap(
    [mha_output[:, 0, :].cpu().detach().reshape(-1, d_model)[:, :]],
    y_ticks=([sentences[0][1][:-1]], 0),
    fig_size=(28, 10),
    cbar=True,
    annot=True,
    fmt=".2f",
    cmap=cmap,
)
fig_mha_output.suptitle("Multi-Head Attention Output")
fig_mha_output.tight_layout()
fig_mha_output.savefig(output_dir / "mha_output.png")
