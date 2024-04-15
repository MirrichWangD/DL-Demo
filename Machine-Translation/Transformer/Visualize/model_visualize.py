# -*- coding: utf-8 -*-
"""
+++++++++++++++++++++++++++++++++++
@ File        : model_visualize.py
@ Time        : 2024/04/09 22:06:23
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (Conda)
+++++++++++++++++++++++++++++++++++
...
+++++++++++++++++++++++++++++++++++
"""

# 导入基础模块
from collections import OrderedDict
from pathlib import Path
import warnings
import re

# 导入依赖模块
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 导入torch相关模块
import torch
import torch.nn.functional as F
from torchtext.vocab import vocab
from torchtext.data import get_tokenizer

# 导入自定义模块
from constants import BOS_IDX, EOS_IDX
from model import Transformer, create_mask, collate_fn
from utils import heatmap, heatmaps, token_heatmap

# 设置忽略warning信息
warnings.filterwarnings("ignore")
# 设置绘图参数
plt.rcParams["font.family"] = ["Microsoft YaHei"]  # 使用微软雅黑字体

"""++++++++++++++++++
@@@ Config
++++++++++++++++++"""

# 模型参数
n_enc_layers = 6
n_dec_layers = 6
n_heads = 8
d_model = 64
d_ff = 256
dropout = 0.1
max_len = 40
resume = "../Tab-Separator/ckpt/en-zh/e6d6h8dm64df256ml40/best_model.pth"

# 数据参数
data_dir = "../Tab-Separator/data/eng-zh.txt"
src_dict = "../Tab-Separator/ckpt/en-zh/e6d6h8dm64df256ml40/words/src_dict.txt"
tgt_dict = "../Tab-Separator/ckpt/en-zh/e6d6h8dm64df256ml40/words/tgt_dict.txt"

cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)

# 设置训练硬件
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 获取输出目录
output_dir = Path("imgs")
output_dir.mkdir(exist_ok=True)

"""++++++++++++++++++++
@@@ Load Data
++++++++++++++++++++"""

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
# 转移至运算设备
src, tgt = src.to(device), tgt.to(device)
# 创建对应的mask
tgt_input = tgt[:-1, :]
src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

"""+++++++++++++++++++
@@@ Token 可视化
+++++++++++++++++++"""

fig_src, ax_src = token_heatmap(
    [src[:, i:i+1].cpu() for i in range(len(data_idx))],
    [sent[0] for sent in sentences],
    ncols=5,
    figsize=(8, 8),
    cbar=False,
    annot=True,
    fmt="d",
    cmap=cmap,
)
fig_src.suptitle("Source Token (en)")
fig_src.tight_layout()
fig_src.savefig(output_dir / "data_src_token.png")

fig_tgt, ax_tgt = token_heatmap(
    [tgt[:, i:i+1].cpu() for i in range(len(data_idx))],
    [sent[1] for sent in sentences],
    ncols=5,
    figsize=(8, 8),
    cbar=False,
    annot=True,
    fmt="d",
    cmap=cmap,
)
fig_tgt.suptitle("Target Token (zh)")
fig_tgt.tight_layout()
fig_tgt.savefig(output_dir / "data_tgt_token.png")

"""++++++++++++++++
@@@ Build Model
++++++++++++++++"""

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

"""++++++++++++++++++++++++++++++
@@@ Positional Encoding 可视化
++++++++++++++++++++++++++++++"""

fig_pe, ax_pe = heatmap(
    model.positional_encoding.pos_embedding.reshape(-1, d_model).cpu().detach(),
    figsize=(28, 10),
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
    src_emb.cpu().detach()[:, 0, :].reshape(-1, d_model),
    ([], sentences[0][0]),
    figsize=(28, 10),
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
    src_pos_emb.cpu().detach()[:, 0, :].reshape(-1, d_model),
    ([], sentences[0][0]),
    figsize=(28, 10),
    cbar=True,
    annot=True,
    fmt=".2f",
    cmap=cmap,
)
fig_src_pos_emb.suptitle("Source Positional Embeddings")
fig_src_pos_emb.tight_layout()
fig_src_pos_emb.savefig(output_dir / "emb_src_pos.png")

"""+++++++++++++++++++++++++++++++++++++++++++++++
@@@ encoder 第一层 Self-Attention 推理
+++++++++++++++++++++++++++++++++++++++++++++++"""

# 计算每个头的维度
src_len, bsz, embed_dim = src_pos_emb.shape
tgt_len, _, _ = tgt_pos_emb.shape
head_dim = embed_dim // n_heads
# 获取 transformer.encoder 第一层权重信息
input_proj_weight = model.transformer.encoder.layers[0].self_attn.in_proj_weight  # q, k, v 映射的 weight
input_proj_bias = model.transformer.encoder.layers[0].self_attn.in_proj_bias  # q, k, v 映射的 bias
out_proj_weight = model.transformer.encoder.layers[0].self_attn.out_proj.weight  # concat 部分的 weight
out_proj_bias = model.transformer.encoder.layers[0].self_attn.out_proj.bias  # concat 部分的 bias

# 以下推理片段节选自 torch.nn.functional.multi_head_attention_forward
# 将 q, k, v 分成多个头，形状为 (batch * num_heads, src_len, embed_dim // num_heads)
q, k, v = F.linear(src_emb, input_proj_weight, input_proj_bias).chunk(3, dim=-1)
q = q.contiguous().view(src_len, bsz * n_heads, head_dim).transpose(0, 1)
k = k.contiguous().view(k.shape[0], bsz * n_heads, head_dim).transpose(0, 1)
v = v.contiguous().view(v.shape[0], bsz * n_heads, head_dim).transpose(0, 1)

"""+++++++++++++++++++++++++++++++++++++++++++++++
@@@ 多头 q, k, v 可视化
+++++++++++++++++++++++++++++++++++++++++++++++"""

q_vis = [i[0].cpu().detach() for i in q.chunk(n_heads)]
# k_vis = [i[0].cpu().detach() for i in k.chunk(n_heads)]
# v_vis = [i[0].cpu().detach() for i in v.chunk(n_heads)]
fig_smha_q, ax_smha_q = heatmaps(
    q_vis,
    ([], sentences[0][0]),
    nums=(2, 4),
    titles=["head-%d" % (i + 1) for i in range(n_heads)],
    figsize=(20, 12),
    cbar=True,
    annot=True,
    fmt=".2f",
    cmap=cmap,
)
fig_smha_q.suptitle("Self Multi-Head Attention Query")
fig_smha_q.tight_layout()
fig_smha_q.savefig(output_dir / "smha_query.png")

"""+++++++++++++++++++++++++++
@@@ Self-Attention mask
+++++++++++++++++++++++++++"""

# merge key padding and attention masks
attn_mask = src_mask.unsqueeze(0)
key_padding_mask = (
    src_padding_mask.view(bsz, 1, 1, src_len).expand(-1, n_heads, -1, -1).reshape(bsz * n_heads, 1, src_len)
)
attn_mask = attn_mask.logical_or(key_padding_mask)

# convert mask to float
new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
new_attn_mask.masked_fill_(attn_mask, float("-inf"))
attn_mask = new_attn_mask

"""+++++++++++++++++++++++++++++
@@@ Self-Attention mask 可视化
+++++++++++++++++++++++++++++"""

attn_mask_vis = attn_mask.view(bsz, n_heads, src_len ,src_len)

fig_smha_mask, ax_smha_mask = heatmaps(
    [attn_mask_vis[i, 0].cpu() for i in range(len(data_idx))],
    (sentences[0][0], sentences[0][0]),
    nums=(1, 5),
    figsize=(40, 10),
    cbar=False,
    annot=True,
    fmt=".0f",
    cmap=cmap,
)
fig_smha_mask.suptitle("Self-Attention Mask")
fig_smha_mask.tight_layout()
fig_smha_mask.savefig(output_dir / "smha_mask.png")


"""++++++++++++++
@@@ 点乘注意力计算
++++++++++++++"""
# Softmax(Q·K)/sqrt(d_model//n_heads)，可以跳转到 torch.nn.functional 的 4816 行查看
# 采用了 torch.baddbmm 和torch.bmm 的方式，将 Multi-Head 和 batch 合在一起的方式进行矩阵乘法
attn_output, attn_output_weights = F._scaled_dot_product_attention(q, k, v, attn_mask, 0.0)
attn_output_weights = attn_output_weights.view(bsz, n_heads, src_len, src_len)

"""+++++++++++++++++++++++++++++++++++++++++++++++
@@@ 对 Self-Attention 的注意力权重进行可视化
+++++++++++++++++++++++++++++++++++++++++++++++"""

attn_output_weights_vis = [attn_output_weights[0, i, :, :].cpu().detach() for i in range(n_heads)]

fig_smha_head_weights, ax5_attn_head_weights = heatmaps(
    attn_output_weights_vis,
    (sentences[0][0], sentences[0][0]),
    nums=(2, 4),
    titles=["head-%d" % (i + 1) for i in range(n_heads)],
    figsize=(20, 12),
    cbar=True,
    annot=False,
    fmt=".2f",
    cmap=cmap,
)
fig_smha_head_weights.suptitle("Self Multi-head Attention Weights")
fig_smha_head_weights.tight_layout()
fig_smha_head_weights.savefig(output_dir / "smha_head_weights.png")

"""+++++++++++++++++++++++++++++++++++++++++++++++
@@@ 对 Self-Attention 的多头注意力权重取平均后进行可视化
+++++++++++++++++++++++++++++++++++++++++++++++"""

attn_output_weights = attn_output_weights.sum(dim=1) / n_heads

fig_smha_weight, ax_smha_weight = heatmap(
    attn_output_weights[0].cpu().detach(),
    (sentences[0][0], sentences[0][0]),
    figsize=(20, 10),
    cbar=True,
    annot=True,
    fmt=".2f",
    cmap=cmap,
)
fig_smha_weight.suptitle("Self Multi-head Attention Weight (Average)")
fig_smha_weight.tight_layout()
fig_smha_weight.savefig(output_dir / "smha_weight.png")

"""+++++++++++++++++++++++++++++++++++++++++++++++
@@@ 对 Self-Attention 推理的最终结果进行可视化
+++++++++++++++++++++++++++++++++++++++++++++++"""

# concat 后映射到 d_model
attn_output = attn_output.transpose(0, 1).contiguous().view(src_len * bsz, embed_dim)
attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
attn_output = attn_output.view(src_len, bsz, attn_output.size(1))

fig_smha_output, ax_smha_output = heatmap(
    attn_output[:, 0, :].cpu().detach().reshape(-1, d_model),
    ([], sentences[0][0]),
    figsize=(28, 10),
    cbar=True,
    annot=True,
    fmt=".2f",
    cmap=cmap,
)
fig_smha_output.suptitle("Self Multi-head Attention Output")
fig_smha_output.tight_layout()
fig_smha_output.savefig(output_dir / "smha_output.png")

"""+++++++++++++++++++++++++++++++++++++++++++++++
@@@ 整个 encoder 推理
+++++++++++++++++++++++++++++++++++++++++++++++"""

memory = model.transformer.encoder(src_pos_emb, src_mask, src_padding_mask)

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

# 将 MHA 的权重进行拆分
E = memory.size(-1)
w_q, w_kv = input_proj_weight.split([E, E * 2])
b_q, b_kv = input_proj_bias.split([E, E * 2])
# 将 q, k, v 分成多个头，形状为 (batch * num_heads, src_len, embed_dim // num_heads)
q = F.linear(out, w_q, b_q)
k, v = F.linear(memory, w_kv, b_kv).chunk(2, dim=-1)
q = q.contiguous().view(tgt_len, bsz * n_heads, head_dim).transpose(0, 1)
k = k.contiguous().view(k.shape[0], bsz * n_heads, head_dim).transpose(0, 1)
v = v.contiguous().view(v.shape[0], bsz * n_heads, head_dim).transpose(0, 1)

# 制作 mask
key_padding_mask = (
    src_padding_mask.view(bsz, 1, 1, src_len).expand(-1, n_heads, -1, -1).reshape(bsz * n_heads, 1, src_len)
)
attn_mask = key_padding_mask
new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
new_attn_mask.masked_fill_(attn_mask, float("-inf"))
attn_mask = new_attn_mask
# MHA 计算
mha_output, mha_output_weights = F._scaled_dot_product_attention(q, k, v, attn_mask, 0.0)
mha_output_weights = mha_output_weights.view(bsz, n_heads, tgt_len, src_len)

"""++++++++++++++++++++++++++++++++++
@@@ Multi-Head Attention mask 可视化
++++++++++++++++++++++++++++++++++"""

attn_mask_vis = attn_mask.view(bsz, n_heads, tgt_len, src_len)

fig_mha_mask, ax_mha_mask = heatmaps(
    [attn_mask_vis[i, 0].cpu() for i in range(len(data_idx))],
    (sentences[0][0], sentences[0][1]),
    nums=(1, 5),
    figsize=(40, 10),
    cbar=False,
    annot=True,
    fmt=".0f",
    cmap=cmap,
)
fig_mha_mask.suptitle("Multi-Head Attention Mask")
fig_mha_mask.tight_layout()
fig_mha_mask.savefig(output_dir / "mha_mask.png")

"""+++++++++++++++++++++++++++++++++++++++++++++++
@@@ 对 MHA 的注意力权重进行可视化
+++++++++++++++++++++++++++++++++++++++++++++++"""

mha_out_weights_vis = [mha_output_weights[0, i, :, :].cpu().detach() for i in range(n_heads)]

fig_mha_head_weights, ax_mha_head_weights = heatmaps(
    mha_out_weights_vis,
    (sentences[0][0], sentences[0][1][:-1]),
    nums=(2, 4),
    titles=["head-%d" % (i + 1) for i in range(n_heads)],
    figsize=(20, 12),
    cbar=True,
    annot=False,
    fmt=".2f",
    cmap=cmap,
)
fig_mha_head_weights.suptitle("Multi-Head Attention Weights")
fig_mha_head_weights.tight_layout()
fig_mha_head_weights.savefig(output_dir / "mha_head_weights.png")

"""+++++++++++++++++++++++++++++++++++++++++++++++
@@@ 对 MHA 的多头注意力权重取平均后进行可视化
+++++++++++++++++++++++++++++++++++++++++++++++"""

mha_output_weights = mha_output_weights.sum(dim=1) / n_heads

fig_mha_weight, ax_mha_weight = heatmap(
    mha_output_weights[0].cpu().detach(),
    (sentences[0][0], sentences[0][1][:-1]),
    figsize=(28, 10),
    cbar=True,
    annot=True,
    fmt=".2f",
    cmap=cmap,
)
fig_mha_weight.suptitle("Multi-Head Attention Weight (Average)")
fig_mha_weight.tight_layout()
fig_mha_weight.savefig(output_dir / "mha_weight.png")

"""+++++++++++++++++++++++++++++++++++++++++++++++
# 对 MultiHead-Attention 推理的最终结果进行可视化
+++++++++++++++++++++++++++++++++++++++++++++++"""

mha_output = mha_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
mha_output = F.linear(mha_output, out_proj_weight, out_proj_bias)
mha_output = mha_output.view(tgt_len, bsz, mha_output.size(1))

fig_mha_output, ax_mha_output = heatmap(
    mha_output[:, 0, :].cpu().detach().reshape(-1, d_model),
    ([], sentences[0][1][:-1]),
    figsize=(28, 10),
    cbar=True,
    annot=True,
    fmt=".2f",
    cmap=cmap,
)
fig_mha_output.suptitle("Multi-Head Attention Output")
fig_mha_output.tight_layout()
fig_mha_output.savefig(output_dir / "mha_output.png")
