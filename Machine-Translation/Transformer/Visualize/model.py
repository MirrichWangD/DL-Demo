# -*- coding: utf-8 -*-
"""
+++++++++++++++++++++++++++++++++++
@ File        : model.py
@ Time        : 2024/4/15 3:27
@ Author      : Mirrich Wang
@ Version     : Python 3.x.x (Conda)
+++++++++++++++++++++++++++++++++++
...
+++++++++++++++++++++++++++++++++++
"""

import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from constants import PAD_IDX


"""+++++++++++++++++++++
@@@ 模型搭建
+++++++++++++++++++++"""


class PositionalEncoding(nn.Module):
    """位置编码层"""

    def __init__(self, emb_size: int, dropout: float, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, emb_size))
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
        max_len: int,
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
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

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
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )
        return self.generator(outs)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor, src_padding_mask: torch.Tensor):
        out = self.positional_encoding(self.src_tok_emb(src))
        return self.transformer.encoder(out, mask=src_mask, src_key_padding_mask=src_padding_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)


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
