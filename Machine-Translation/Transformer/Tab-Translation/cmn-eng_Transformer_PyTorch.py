# -*- encoding: utf-8 -*-
"""
+++++++++++++++++++++++++++++++++++
    @ File        : cmn-eng_Transformer_PyTorch.py
    @ Time        : 2022/10/10 18:49
    @ Author      : Mirrich Wang
    @ Version     : Python 3.8.10 (Anaconda)
+++++++++++++++++++++++++++++++++++
基于《深度学习原理与PyTorch实战（第2版）》的 2017 年《Attention is All You Need》文章中 Translation2019zh_Transformer_PyTorch 架构
的 PyTorch 源码，训练数据集采用 Tatoeba Project 的 Mandarin Chinese - English 中英双语数据集，nltk英文分词
https://www.manythings.org/anki/，其中一共有 29155 条句子，英文字典长度为 7968 + 4，中文为 2850 + 4
"""

# 导入基本模块
import time
import copy
import math
from collections import Counter

# 导入依赖模块
import opencc
import nltk
import numpy as np

# 导入 torch 相关模块
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

"""==============================
@@@ 模型结构及搭建
=============================="""


def clones(module, num):
    """
    clones 工具函数
    基于 torch.nn.xxx 快速获得 num 个相同结构不同参数的层
    Args:
        module: torch.nn.xxx 传入层或 nn.Module
        num: int 复制个数

    Returns:

    """
    return nn.ModuleList(list(map(lambda _: copy.deepcopy(module), range(num))))


class PositionalEncoding(nn.Module):
    """实现 Positional Encoding（位置编码）"""

    def __init__(self, d_model, dropout, max_len=5000):
        """
        生成词向量相同维度的位置编码矩阵，采取对数正余弦不同频率函数
        Args:
            d_model: int 向量维度
            dropout: float 拆线比例
            max_len: int 词最大长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        计算位置编码和输入张量矩阵的矩阵加法
        Args:
            x: torch.Tensor 输入张量矩阵

        Returns:

        """
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return self.dropout(x)


def attention(query, key, value, mask=None, dropout=None):
    """
    计算 Scaled Dot-Product Attention
    Args:
        query: torch.Tensor 查询矩阵
        key: torch.Tensor 键矩阵
        value: torch.Tensor 值矩阵
        mask: torch.Tensor 掩码矩阵
        dropout: float 拆线比例

    Returns:

    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # 计算 Q·K^T / sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # (Q·K^T + M) / sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)  # softmax(Q·K^T + M / sqrt(d_k))
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn  # softmax((Q·K^T + M) / sqrt(d_k))


class MultiHeadedAttention(nn.Module):
    """多头注意力实现"""

    def __init__(self, h, d_model, dropout=0.1):
        """
        计算多头注意力矩阵
        Args:
            h: int 头的数量
            d_model: int 向量维度
            dropout: float 拆线比例
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 3)  ###
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        前向传播计算
        Args:
            query: torch.Tensor 查询矩阵
            key: torch.Tensor 键矩阵
            value: torch.Tensor 值矩阵
            mask: torch.Tensor 掩码矩阵

        Returns:

        """
        if mask is not None:
            # 相同的掩码矩阵应用在共 h 个头
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model = h * d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class LayerNorm(nn.Module):
    """实现层归一化"""

    def __init__(self, features, eps=1e-6):
        """
        初始化层归一化所需参数
        Args:
            features: torch.Tensor 传入输入矩阵
            eps: float 误差变动
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        """
        前向传播计算
        Args:
            x: torch.Tensor 传入需要层归一化的张量矩阵

        Returns:

        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    残差连接，后跟层范数。
    注意，为了代码的简单性，规范是第一个而不是最后一个。
    """

    def __init__(self, size, dropout):
        """
        构造上一步输入然后进行残差连接和层归一化
        Args:
            size:
            dropout:
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        将剩余连接应用到具有相同大小的任何子层
        Args:
            x: 输入特征
            sublayer: 残差连接和层归一化之前的层，如 FFN 或 MultiHead Attention

        Returns:

        """
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    """实现前馈神经网络层"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        构造指定向量维度和权重维度的前馈神经网络
        Args:
            d_model: int 向量维度
            d_ff: int 权重维度
            dropout: 拆线比例
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """计算 ReLU(X·W_1 + b_1)W_2 + b_2"""
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    """编码器由自注意（Self-Attention）和前馈（FFN）组成，定义如下"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        构造单个编码块
        Args:
            size: int 向量维度 d_model
            self_attn: MultiHeadedAttention 多头注意力
            feed_forward: PositionwiseFeedForward 前馈神经网络
            dropout: float 拆线比例
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """按照原文图 1 进行连接。"""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """生成 N 个编码块的编码器"""

    def __init__(self, layer, N):
        """
        构造 N 个编码块
        Args:
            layer: EncoderLayer 编码块
            N: int 生成 N 个编码块
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """依次在每个层中传递输入"""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


def subsequent_mask(size):
    """
    屏蔽后续位置的掩码
    Args:
        size: int 词最大长度 max_len

    Returns:

    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(subsequent_mask) == 0


class DecoderLayer(nn.Module):
    """解码器由自注意力（Self-Attention）、编码器的注意力（src-Attention）和前馈网络（FFN）组成"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        构造单个编码块
        Args:
            size: int 向量维度 d_model
            self_attn: MultiHeadedAttention 多头自注意力
            src_attn: MultiHeadedAttention 多头注意力
            feed_forward: PositionwiseFeedForward FFN
            dropout: float 拆线比例
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """按照原文图 1 进行连接。"""
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    """带掩码操作的 N 层解码块"""

    def __init__(self, layer, N):
        """
        构造解码器
        Args:
            layer: DecoderLayer 单个解码块
            N: int 解码块的数量
        """
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Generator(nn.Module):
    """定义标准线性+ softmax生成步骤。"""

    def __init__(self, d_model, vocab):
        """

        Args:
            d_model: int 向量维度
            vocab: int 词字典数量
        """
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class EncoderDecoder(nn.Module):
    """标准的编码器-解码器架构"""

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        生成整个模型结构
        Args:
            encoder: Encoder 编码器
            decoder: Decoder 解码器
            src_embed: Embeddings 输入词嵌入
            tgt_embed: Embeddings 输出词嵌入
            generator: Generator 输出区块
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# -------------------------------------------------------- #
# Embedding
# 词嵌入
# -------------------------------------------------------- #


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # x = x.long()
        return self.lut(x) * math.sqrt(self.d_model)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )
    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


"""=================================
@@@ 模型损失函数、优化器
===================================="""


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)  # ?
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1).type(torch.int64), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(
        model.src_embed[0].d_model,
        2,
        4000,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
    )


def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        print(batch.src.shape, batch.trg.shape, batch.src_mask.shape, batch.trg_mask.shape)
        break
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" % (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


##################################### 数据迭代器构造 #####################################


def Traditional2Simplified(sentence):
    """
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    """
    cc = opencc.OpenCC("t2s")
    sentence = cc.convert(sentence)
    return sentence


def batch_padding(batch, max_len, padding_idx=0):
    for i, sent in enumerate(batch):
        padding_len = max_len - len(sent)
        if padding_len > 0:
            sent.extend([padding_idx] * padding_len)
        else:
            batch[i] = batch[i][:max_len]
    return batch


def load_data(filename, max_len=5000):
    """
    根据 Anki 中、英文数据集进行英->中数据集处理
    Args:
        filename:
        max_len:

    Returns:

    """
    src_token_map = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
    tgt_token_map = src_token_map.copy()
    file = open(filename, encoding="utf-8")
    src_word, tgt_word = [], []
    for line in file:
        src_sent, tgt_sent = line.rstrip("\n").split("\t")[:2]
        tgt_sent = Traditional2Simplified(tgt_sent)
        src_word.append(" ".join(nltk.word_tokenize(src_sent)).split())
        tgt_word.append(" ".join(tgt_sent).split())
    src_counter = Counter(np.concatenate(src_word))
    tgt_counter = Counter(np.concatenate(tgt_word))
    src_counter_sort = dict(sorted(src_counter.items(), key=lambda i: i[1], reverse=True))
    tgt_counter_sort = dict(sorted(tgt_counter.items(), key=lambda i: i[1], reverse=True))
    for word, i in zip(src_counter_sort.keys(), range(len(src_counter_sort))):
        src_token_map[word] = i + 4
    for word, i in zip(tgt_counter_sort.keys(), range(len(tgt_counter_sort))):
        tgt_token_map[word] = i + 4

    src = torch.tensor(
        batch_padding([[1] + [src_token_map[i] for i in line] + [2] for line in src_word], max_len),
        dtype=torch.int,
    )
    tgt = torch.tensor(
        batch_padding([[1] + [tgt_token_map[i] for i in line] + [2] for line in tgt_word], max_len),
        dtype=torch.int,
    )

    return src, tgt, src_token_map, tgt_token_map


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def real_data_gen(src, tgt, batch_size=32, cuda=True):
    batch_en = []
    batch_zh = []

    for src_word, tgt_word in zip(src, tgt):
        batch_en.append(src_word.tolist())
        batch_zh.append(tgt_word.tolist())

        if len(batch_en) % batch_size == 0:
            src = torch.tensor(batch_en, dtype=torch.int).long()
            tgt = torch.tensor(batch_zh, dtype=torch.int).long()
            if cuda:
                src = src.cuda()
                tgt = tgt.cuda()

            batch_en = []
            batch_zh = []
            yield Batch(src, tgt, 0)
    else:
        src = src.long()
        tgt = tgt.long()
        if cuda:
            src = src.cuda()
            tgt = tgt.cuda()
        yield Batch(src, tgt, 0)


# greedy decode
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    # ys代表目前已生成的序列，最初为仅包含一个起始符的序列，不断将预测结果追加到序列最后
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory,
            src_mask,
            Variable(ys),
            Variable(subsequent_mask(ys.size(1)).type_as(src.data)),
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


if __name__ == "__main__":
    # ------------------------------- #
    # Config
    # ------------------------------- #

    cuda = True  # 是否使用 CUDA
    epochs = 40  # 训练世纪
    max_len = 40  # 词最大长度
    batch_size = 32  # 批数量

    # ------------------------------- #
    # 加载数据，划分训练集和验证集
    # ------------------------------- #

    # 加载数据并且转换成词索引
    src, tgt, en_token_map, zh_token_map = load_data("../data/cmn-eng/cmn.txt", max_len)
    print(list(en_token_map.items())[:15])
    print(list(zh_token_map.items())[:15])
    print("src:", src.shape)
    print("target:", tgt.shape)
    en_word_map = dict(zip(en_token_map.values(), en_token_map.keys()))
    zh_word_map = dict(zip(zh_token_map.values(), zh_token_map.keys()))
    print(src[0])
    # 划分训练集和验证集
    np.random.seed(666)
    indices = np.arange(0, src.shape[0])
    np.random.shuffle(indices)
    train_indices = indices[: int(0.99 * src.shape[0])]
    val_indices = indices[int(0.99 * src.shape[0]) :]
    train_src = src[train_indices]
    train_tgt = tgt[train_indices]
    val_src = src[val_indices]
    val_tgt = tgt[val_indices]
    # 构造迭代器
    train_iter = real_data_gen(train_src, train_tgt, batch_size)
    val_iter = real_data_gen(val_src, val_tgt, batch_size=1)

    vocab_en, vocab_zh = len(en_token_map), len(zh_token_map)  # 获取词典数量

    print("vocab_en:", vocab_en)
    print("vocab_zh:", vocab_zh)

    # ------------------------------- #
    # 初始化模型、损失函数和优化器
    # ------------------------------- #

    criterion = LabelSmoothing(size=vocab_zh, padding_idx=0, smoothing=0.0)
    model = make_model(vocab_en, vocab_zh)
    model_opt = NoamOpt(
        model.src_embed[0].d_model,
        1,
        400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
    )

    if cuda:
        model.cuda()

    # ------------------------------- #
    # 模型训练
    # ------------------------------- #

    # for epoch in range(epochs):
    #     print(("Epoch %i" % (epoch + 1)).center(50, "="))
    #     model.train()
    #     run_epoch(real_data_gen(train_src, train_tgt, batch_size), model,
    #               SimpleLossCompute(model.generator, criterion, model_opt))
    #     # model.eval()
    #     # print(run_epoch(val_iter, model,
    #     #                 SimpleLossCompute(model.generator, criterion, None)))
    #
    # torch.save(model.state_dict(), f"final_weights.pth")  # 模型权重保存

    # ------------------------------- #
    # 模型预测
    # ------------------------------- #

    for batch in val_iter:
        print(batch.src.shape, batch.src_mask.shape)
        pred_result = greedy_decode(model, batch.src, batch.src_mask, max_len=max_len, start_symbol=1)
        print("=" * 50)
        print(pred_result)
        print("pred:", " ".join([zh_word_map[int(i)] for i in pred_result[0]]))
        print("real:", " ".join([zh_word_map[int(i)] for i in batch.trg[0]]))

    # src = Variable(torch.LongTensor([[25, 26, 27, 28, 4]])).cuda()
    # src_mask = Variable(torch.ones(1, 1, 5)).cuda()
    # pred_result = greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=1)
    # print(pred_result)
    # print(" ".join([zh_dict_reverse[int(i)] for i in pred_result[0]]))
