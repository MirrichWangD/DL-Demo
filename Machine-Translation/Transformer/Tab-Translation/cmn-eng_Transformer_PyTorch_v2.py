# -*- coding: UTF-8 -*-
"""
+++++++++++++++++++++++++++++++++++
@ File        : cmn-eng_Transformer_PyTorch_v2.py
@ Time        : 2022/10/10 18:49
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (Conda)
+++++++++++++++++++++++++++++++++++
简要概述：
    基于《深度学习原理与 PyTorch 实战（第2版）》的 2017 年《Attention is All You Need》文章中 Translation2019zh_Transformer_PyTorch 架构
    的 PyTorch 源码，训练数据集采用 Tatoeba Project 的 Mandarin Chinese - English 中英双语数据集，nltk英文分词
    https://www.manythings.org/anki/，其中一共有 29155 条句子，英文字典长度为 7968 + 4，中文为 2850 + 4
    ！v2改动：增加了全局变量，使用 torchtext 进行词汇表构建
硬件环境：
    12th Gen Intel(R) Core(TM) i7-12700H   2.70 GHz
    NVIDIA GeForce RTX 3060 6GB
    CUDA 11.6.1 + CUDNN 8.3.02
依赖模块 (* 表示可以安装最新版本）：
    torch       1.12.1+cu116
    torchtext   0.13.1
    nltk        3.8.1* (需要进行 download 相关包：>>import nltk;nltk.download()）
+++++++++++++++++++++++++++++++++++
"""

# 导入基本库
from collections import Counter, OrderedDict
import warnings
import time
import copy
import math
import os

# 导入依赖库
import nltk
import numpy as np

# 导入Torch相关库
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchtext

# 忽略警告信息
warnings.filterwarnings("ignore")

"""++++++++++++++++++++
@@@ Config
++++++++++++++++++++"""

# 设置硬件设备
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# Token信息
TOKEN = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3

np.random.seed(666)

"""++++++++++++++++++++++
@@@ 模型结构及搭建
++++++++++++++++++++++"""


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
    """ 实现 Positional Encoding（位置编码）"""

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
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        计算位置编码和输入张量矩阵的矩阵加法
        Args:
            x: torch.Tensor 输入张量矩阵

        Returns:

        """
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
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
    """ 多头注意力实现 """

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
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class LayerNorm(nn.Module):
    """ 实现层归一化 """

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
    """ 实现前馈神经网络层 """

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
        """ 计算 ReLU(X·W_1 + b_1)W_2 + b_2"""
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    """ 编码器由自注意（Self-Attention）和前馈（FFN）组成，定义如下 """

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
        """ 按照原文图 1 进行连接。 """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """ 生成 N 个编码块的编码器 """

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
        """ 依次在每个层中传递输入 """
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
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class DecoderLayer(nn.Module):
    """ 解码器由自注意力（Self-Attention）、编码器的注意力（src-Attention）和前馈网络（FFN）组成 """

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
        """ 按照原文图 1 进行连接。 """
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    """ 带掩码操作的 N 层解码块 """

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
    """ 定义标准线性+ softmax生成步骤。 """

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
    """ 标准的编码器-解码器架构 """

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
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Embeddings(nn.Module):
    """ 词嵌入层 """

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # x = x.long()
        return self.lut(x) * math.sqrt(self.d_model)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, resume=None):
    """ Helper: Construct a model from hyperparameters. """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )
    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    if resume is not None:
        model.load_state_dict(torch.load(resume))

    return model


"""+++++++++++++++++++++++++++++++++++++
@@@ 模型损失函数、优化器、预测贪婪解码函数
+++++++++++++++++++++++++++++++++++++"""


class LabelSmoothing(nn.Module):
    """ Implement label smoothing. """

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
    """ A simple loss compute and train function. """

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm


class NoamOpt:
    """ Optim wrapper that implements rate. """

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """ Update parameters and rate """
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """ Implement `lrate` above """
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


def greedy_decode(model, src, src_mask, max_len, start_symbol=BOS_TOKEN):
    """
    贪婪匹配解码函数

    Args:
        model: torch.nn.Module 传入模型
        src: torch.Tensor 批词索引向量，形状必须是[1, max_len]
        src_mask: torch.Tensor 批词索引mask向量，形状必须是[1, max_len]
        max_len: int 词索引向量最大长度
        start_symbol: int 词索引开始符号索引，<BOS>: 1

    Returns:

    """
    memory = model.encode(src, src_mask)
    # ys代表目前已生成的序列，最初为仅包含一个起始符的序列，不断将预测结果追加到序列最后
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


"""++++++++++++++++++++ 
@@@ 数据集构建
++++++++++++++++++++"""


def Traditional2Simplified(sentence: str) -> str:
    """
    将str类型的变量sentence中繁体字转为简体字

    Args:
        sentence: str 包含繁体字的字符串

    Returns:

    """
    sentence = Converter('zh-hans').convert(sentence)
    return sentence


def batch_padding(batch, max_len, padding_idx=PAD_TOKEN):
    """
    批次填充函数，将根据batch的最大长度和max_len进行对应裁剪或padding

    Args:
        batch: torch.Tensor 输入词索引向量
        max_len: int 词向量最大长度
        padding_idx: int 填充字符串的TOKEN

    Returns:

    """
    for i, sent in enumerate(batch):
        padding_len = max_len - len(sent)
        if padding_len > 0:
            sent.extend([padding_idx] * padding_len)
        else:
            batch[i] = batch[i][:max_len]
    return batch


def load_data(file_path, src_func=nltk.word_tokenize, tgt_func=lambda x: " ".join(x).split()):
    """
    根据 Anki 中、英文数据集进行英->中数据集处理

    Args:
        file_path: str <src>\t<tgt>\n 格式的数据文档
        src_func: function 输入语言转换函数，默认是nltk分词
        tgt_func: function 输出语言转换函数，默认是单字符划分

    Returns:

    """
    file = open(file_path, encoding="utf-8")
    src_word, tgt_word = [], []
    for line in file:
        src_sent, tgt_sent = line.rstrip("\n").split('\t')[:2]
        tgt_sent = Traditional2Simplified(tgt_sent)
        # 分词
        src_word.append(src_func(src_sent))
        tgt_word.append(tgt_func(tgt_sent))
    # 统计词频并排序
    src_counter = sorted(Counter(np.concatenate(src_word)).items(), key=lambda i: i[1], reverse=True)
    tgt_counter = sorted(Counter(np.concatenate(tgt_word)).items(), key=lambda i: i[1], reverse=True)

    src_vocab = torchtext.vocab.vocab(OrderedDict(src_counter), specials=TOKEN)
    tgt_vocab = torchtext.vocab.vocab(OrderedDict(tgt_counter), specials=TOKEN)

    with open(os.path.join(os.path.dirname(file_path), "src_dict.txt"), "w+", encoding="utf-8") as f:
        f.writelines(map(lambda i: i + "\n", src_vocab.get_itos()))

    with open(os.path.join(os.path.dirname(file_path), "tgt_dict.txt"), "w+", encoding="utf-8") as f:
        f.writelines(map(lambda i: i + "\n", tgt_vocab.get_itos()))

    return src_word, tgt_word, src_vocab, tgt_vocab


class TranslationDataset(Dataset):
    """ 机器翻译数据集 """

    def __init__(self, data, src_vocab, tgt_vocab, max_len):
        """

        Args:
            data: list 句子列表
            src_vocab: dict 对应语言token字典
            src_vocab: torchtext.vocab.vocab.Vocab
            tgt_vocab: torchtext.vocab.vocab.Vocab
            max_len int 最大句子产长度
        """
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __getitem__(self, idx):
        src_word, tgt_word = self.data[idx]
        src_token = torch.tensor(
            batch_padding([[BOS_TOKEN] + self.src_vocab.lookup_indices(src_word) + [EOS_TOKEN]], self.max_len),
            dtype=torch.int
        )
        tgt_token = torch.tensor(
            batch_padding([[BOS_TOKEN] + self.tgt_vocab.lookup_indices(tgt_word) + [EOS_TOKEN]], self.max_len),
            dtype=torch.int
        )
        src_mask = (src_token != PAD_TOKEN).unsqueeze(-2)
        tgt_mask = (tgt_token[:, :-1] != PAD_TOKEN).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt_token[:, :-1].size(-1)).type_as(tgt_mask.data))

        return src_token[0], tgt_token[0], src_mask[0], tgt_mask[0]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    # ------------------------------- #
    # 训练参数设置
    # ------------------------------- #

    start_epoch = 13  # 开始训练世纪
    epochs = 100  # 训练世纪
    max_len = 40  # 词最大长度
    batch_size = 32  # 批数量
    # resume = "checkpoints/Epoch-13.pth"  # 断点权重
    resume = None

    # ------------------------------- #
    # 加载数据，划分训练集和验证集
    # ------------------------------- #

    # 加载分词句句子和token字典
    en_words, zh_words, en_vocab, zh_vocab = load_data("./data/eng-cmn.txt")  # en_words和zh_words等长度且一一对应
    print("src - Vocab Length: %d, Token: %s" % (len(en_vocab), en_vocab.get_itos()[:15]))
    print("target - Vocab Length: %d, Token: %s" % (len(zh_vocab), zh_vocab.get_itos()[:15]))
    # 划分训练集和验证集，取出1000条验证，100条测试
    indices = np.arange(0, len(en_words))
    np.random.shuffle(indices)

    # 构造数据集
    words = list(map(lambda i: (en_words[i], zh_words[i]), range(len(en_words))))
    dataset = TranslationDataset(words, en_vocab, zh_vocab, max_len)
    # 构造数据跌打第七
    train_iterator = DataLoader(dataset, batch_size=batch_size, num_workers=0, sampler=indices[:-1100])
    val_iterator = DataLoader(dataset, batch_size=batch_size, num_workers=0, sampler=indices[-1100:-100])
    test_iterator = DataLoader(dataset, batch_size=batch_size, num_workers=0, sampler=indices[-100:])

    # ------------------------------- #
    # 初始化模型、损失函数和优化器
    # ------------------------------- #

    criterion = LabelSmoothing(size=len(zh_vocab), padding_idx=0, smoothing=0.0)
    model = make_model(len(en_vocab), len(zh_vocab), resume=resume).to(DEVICE)
    optimizer = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    # ------------------------------- #
    # 模型训练和验证
    # ------------------------------- #

    train_loss_compute = SimpleLossCompute(model.generator, criterion, optimizer)
    val_loss_compute = SimpleLossCompute(model.generator, criterion, None)
    for epoch in range(start_epoch, epochs):
        print(" Starting Training ".center(50, "#"))
        model.train()
        total_loss = 0
        acc = list()
        st = time.time()
        for step, (src, tgt, src_mask, tgt_mask) in enumerate(train_iterator):
            num_tokens = (tgt[:, 1:] != PAD_TOKEN).data.sum()
            # 运算硬件转移
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            src_mask, tgt_mask = src_mask.to(DEVICE), tgt_mask.to(DEVICE)
            # 模型推理
            out = model(src, tgt[:, :-1], src_mask, tgt_mask)
            loss = train_loss_compute(out, tgt[:, 1:], num_tokens)  # 计算损失

            # 记录一个epoch中的损失和token数量
            total_loss += loss
            # 打印训练记录
            if step % 50 == 0:
                print("Epoch: %d/%d\t Step: %d/%d\t Loss: %.4f\t Time: %fs" % (
                    epoch + 1, epochs, step + 1, len(train_iterator), loss / num_tokens, time.time() - st
                ))

        model.eval()
        print(" Start Evaluating ".center(50, "#"))
        val_total_loss = 0
        val_st = time.time()
        for step, (src, tgt, src_mask, tgt_mask) in enumerate(val_iterator):
            num_tokens = (tgt[:, 1:] != PAD_TOKEN).data.sum()
            # 运算硬件转移
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            src_mask, tgt_mask = src_mask.to(DEVICE), tgt_mask.to(DEVICE)

            out = model(src, tgt[:, :-1], src_mask, tgt_mask)
            loss = val_loss_compute(out, tgt[:, 1:], num_tokens)

            val_total_loss += loss

            if step % 10 == 0:
                print("Evaluating Step: %d/%d, Loss: %.4f, Tokens per Sec: %fs" % (
                    step + 1, len(val_iterator), loss / num_tokens, time.time() - val_st
                ))

        torch.save(model.state_dict(), f"checkpoints/Epoch-{epoch + 1}.pth")  # 模型权重保存

    # ------------------------------- #
    # 模型预测
    # ------------------------------- #

    for src, tgt, src_mask, tgt_mask in test_iterator:
        src, src_mask = src.to(DEVICE), src_mask.to(DEVICE)
        src_loc = torch.nonzero(src == EOS_TOKEN)[0]
        pred = greedy_decode(model, src, src_mask, max_len=max_len)
        pred_loc = torch.nonzero(pred == EOS_TOKEN)[0]
        print("=" * 50)
        # print(pred_result)
        print("pred:", " ".join(zh_vocab.lookup_tokens(pred[0][:src_loc[1] + 2].data.tolist())))
        print("real:", " ".join(zh_vocab.lookup_tokens(tgt[0][:pred_loc[1] + 1].data.tolist())))
