# -*- coding: utf-8 -*-
"""+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # 机器翻译的神经网络实现
# 本节课我们讲述了利用Transformer网络实现法－英机器翻译。
# 整个代码包括了数据预处理、Transformer网络两个部分组成。
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"""

# 导入模块
from io import open
import unicodedata
import string
import re
import random


# Pytorch必备的模块
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as DataSet


# 绘图所用的包
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import copy
import math
from torch.autograd import Variable

# 判断本机是否有支持的GPU
use_cuda = torch.cuda.is_available()

# 即时绘图
# get_ipython().run_line_magic('matplotlib', 'inline')


# # 一、数据准备
# 从硬盘读取语料文件，进行基本的预处理
# 读取平行语料库
# 英＝法
ger_eng = pd.read_table("data/ger_eng.txt", header=None, sep="\t")
english = ger_eng[0]
german = ger_eng[1]


# 定义两个特殊符号，分别对应句子头和句子尾
SOS_token = 0
EOS_token = 1


# 定义一个语言类，方便进行自动的建立、词频的统计等
# 在这个对象中，最重要的是两个字典：word2index，index2word
# 故名思议，第一个字典是将word映射到索引，第二个是将索引映射到word
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        # 在语言中添加一个新句子，句子是用空格隔开的一组单词
        # 将单词切分出来，并分别进行处理
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        # 插入一个单词，如果单词已经在字典中，则更新字典中对应单词的频率
        # 同时建立反向索引，可以从单词编号找到单词
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
# 将unicode编码转变为ascii编码
def unicodeToAscii(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


# 把输入的英文字符串转成小写
def normalizeEngString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# 对输入的单词对做过滤，保证每句话的单词数不能超过MAX_LENGTH
def filterPair(p):
    return len(p[0].split(" ")) < MAX_LENGTH and len(p[1].split(" ")) < MAX_LENGTH


# 输入一个句子，输出一个单词对应的编码序列
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(" ")]


# 和上面的函数功能类似，不同在于输出的序列等长＝MAX_LENGTH
def indexFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    for i in range(MAX_LENGTH - len(indexes)):
        indexes.append(EOS_token)
    return indexes


def subsequent_mask(size):
    attn_shape = (size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(subsequent_mask) == 0


# 从一个词对到下标
def indexFromPair(pair):
    input_variable = indexFromSentence(input_lang, pair[0])
    input_mask = np.expand_dims((np.array(input_variable) != 1), -2)
    target_variable = indexFromSentence(output_lang, pair[1])
    target_mask = np.array(target_variable) != 1
    target_mask = target_mask & (subsequent_mask(len(target_variable)).to(torch.int32)).numpy()
    return (input_variable, input_mask, target_variable, target_mask)


# 从一个列表到句子
def SentenceFromList(lang, lst):
    result = [lang.index2word[i] for i in lst if i != EOS_token]
    result = " ".join(result)
    return result


# 计算准确度的函数
def rightness(predictions, labels):
    """计算预测错误率的函数，其中predictions是模型给出的一组预测结果，batch_size行num_classes列的矩阵，labels是数据之中的正确答案"""
    pred = torch.max(predictions.data, 1)[1]  # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
    rights = int(pred.eq(labels.data).sum())  # 将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
    return rights, len(labels)  # 返回正确的数量和这一次一共比较了多少元素


# 处理数据形成训练数据
# 设置句子的最大长度
MAX_LENGTH = 10

# 对英文做标准化处理
pairs = [[normalizeEngString(ger), normalizeEngString(eng)] for ger, eng in zip(german, english)]

# 对句子对做过滤，处理掉那些超过MAX_LENGTH长度的句子
input_lang = Lang("German")
output_lang = Lang("English")
pairs = [pair for pair in pairs if filterPair(pair)]
print("有效句子对：", len(pairs))

# 建立两个字典（中文的和英文的）
for pair in pairs:
    input_lang.addSentence(pair[0])
    output_lang.addSentence(pair[1])

print("总单词数:")  # 有效句子对： 96464
print(input_lang.name, input_lang.n_words)  # French 17106
print(output_lang.name, output_lang.n_words)  # English 10426


# 形成训练集，首先，打乱所有句子的顺序
random_idx = np.random.permutation(range(len(pairs)))
pairs = [pairs[i] for i in random_idx]

# 将语言转变为单词的编码构成的序列
pairs = [indexFromPair(pair) for pair in pairs]

# 形成训练集、校验集和测试集
valid_size = len(pairs) // 10
if valid_size > 10000:
    valid_size = 10000
pp = pairs
pairs = pairs[:-valid_size]
valid_pairs = pp[-valid_size : -valid_size // 2]
test_pairs = pp[-valid_size // 2 :]

# 利用PyTorch的dataset和dataloader对象，将数据加载到加载器里面，并且自动分批

batch_size = 32  # 一批包含32个数据记录，这个数字越大，系统在训练的时候，每一个周期处理的数据就越多，这样处理越快，但总的数据量会减少

print("训练记录：", len(pairs))  # 训练记录： 86818
print("校验记录：", len(valid_pairs))  # 校验记录： 4823
print("测试记录：", len(test_pairs))  # 测试记录： 4823


# 形成训练对列表，用于喂给train_dataset
pairs_X = [pair[0] for pair in pairs]
pairs_X_mask = [pair[1] for pair in pairs]
pairs_Y = [pair[2] for pair in pairs]
pairs_Y_mask = [pair[3] for pair in pairs]
valid_X = [pair[0] for pair in valid_pairs]
valid_X_mask = [pair[1] for pair in valid_pairs]
valid_Y = [pair[2] for pair in valid_pairs]
valid_Y_mask = [pair[3] for pair in valid_pairs]
test_X = [pair[0] for pair in test_pairs]
test_X_mask = [pair[1] for pair in test_pairs]
test_Y = [pair[2] for pair in test_pairs]
test_Y_mask = [pair[3] for pair in test_pairs]


# 形成训练集
train_dataset = DataSet.TensorDataset(
    torch.LongTensor(pairs_X),
    torch.Tensor(pairs_X_mask),
    torch.LongTensor(pairs_Y),
    Variable(torch.Tensor(pairs_Y_mask)),
)
# 形成数据加载器
train_loader = DataSet.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


# 校验数据
valid_dataset = DataSet.TensorDataset(
    torch.LongTensor(valid_X),
    torch.Tensor(valid_X_mask),
    torch.LongTensor(valid_Y),
    Variable(torch.Tensor(valid_Y_mask)),
)
valid_loader = DataSet.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# 测试数据
test_dataset = DataSet.TensorDataset(
    torch.LongTensor(test_X),
    torch.Tensor(test_X_mask),
    torch.LongTensor(test_Y),
    Variable(torch.Tensor(test_Y_mask)),
)
test_loader = DataSet.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)


""" # # 二、构建Transformer网络  """


# 构建位置编码
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
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
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return self.dropout(x)


# 构建嵌入层
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# 构建注意力机制
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


# 构建多头注意力机制
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
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


# 构建层归一化
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# 构建前馈神经网络
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 构建编码器
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


# 构建解码器
class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


# 构建编码器-解码器
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.generator(self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask))

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# 构建Transformer网络
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
    # for p in model.parameters():
    #    if p.dim() > 1:
    #        nn.init.xavier_uniform_(p)
    return model


"""-----------------
# 开始训练过程
--------------------"""
# 定义网络架构
max_length = MAX_LENGTH
model = make_model(input_lang.n_words, output_lang.n_words)

if use_cuda:
    model = model.cuda()

learning_rate = 0.0001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

criterion = nn.NLLLoss()

num_epoch = 1
"""
# 开始训练周期循环
plot_losses = []
for epoch in range(num_epoch):
    # 将网络置于训练状态，让dropout工作
    model.train()
    print_loss_total = 0
    # 对训练数据进行循环
    for i, data in enumerate(train_loader):
        input_variable = data[0].cuda() if use_cuda else data[0]
        # input_variable的大小：batch_size, length_seq
        input_mask = data[1].cuda() if use_cuda else data[1]
        # input_mask的大小：batch_size, 1, length_seq
        target_variable = data[2].cuda() if use_cuda else data[2]
        # target_variable的大小：batch_size, length_seq
        target_mask = data[3].cuda() if use_cuda else data[3]
        # target_mask的大小：batch_size, length_seq, length_seq
        
        # 清空梯度
        optimizer.zero_grad()

        loss = 0

        # 网络开始工作
        outputs = model(input_variable, target_variable, input_mask, target_mask)
        # outputs的大小：batch_size, length_seq, output_lang.n_words

        # 计算损失函数
        for di in range(MAX_LENGTH - 1):
            loss += criterion(outputs[:, di, :], target_variable[:, di])
        
        # 反向传播开始
        loss.backward()
        loss = loss.cpu() if use_cuda else loss
        # 开始梯度下降
        optimizer.step()
        print_loss_total += loss.data.numpy()

    print_loss_avg = print_loss_total / len(train_loader)
        
    valid_loss = 0
    rights = []
    # 将网络的training设置为False，以便关闭dropout
    model.eval()
    
    #对所有的校验数据做循环
    for data in valid_loader:
        input_variable = data[0].cuda() if use_cuda else data[0]
        # input_variable的大小：batch_size, length_seq
        input_mask = data[1].cuda() if use_cuda else data[1]
        # input_mask的大小：batch_size, 1, length_seq
        target_variable = data[2].cuda() if use_cuda else data[2]
        # target_variable的大小：batch_size, length_seq
        target_mask = data[3].cuda() if use_cuda else data[3]
        # target_mask的大小：batch_size, length_seq, length_seq

        loss = 0
        outputs = model(input_variable, target_variable, input_mask, target_mask)
        # outputs的大小：batch_size, length_seq, output_lang.n_words

        # 开始预测
        for di in range(MAX_LENGTH - 1):
            right = rightness(outputs[:, di, :], target_variable[:, di])
            rights.append(right)
            loss += criterion(outputs[:, di, :], target_variable[:, di])
        loss = loss.cpu() if use_cuda else loss
        valid_loss += loss.data.numpy()
    # 计算平均损失、准确率等指标并打印输出
    right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
    print('进程：%d%% 训练损失：%.4f，校验损失：%.4f，词正确率：%.2f%%' % (epoch * 1.0 / num_epoch * 100, 
                                                    print_loss_avg,
                                                    valid_loss / len(valid_loader),
                                                    100.0 * right_ratio))
    plot_losses.append([print_loss_avg, valid_loss / len(valid_loader), right_ratio])
    torch.save(model, 'model-final.mdl')


# 绘制统计指标曲线图
torch.save(model, 'model-final.mdl')
"""
model = torch.load("model-final.mdl")
"""
a = [i[0] for i in plot_losses]
b = [i[1] for i in plot_losses]
c = [i[2] * 100 for i in plot_losses]
plt.plot(a, '-', label = 'Training Loss')
plt.plot(b, ':', label = 'Validation Loss')
plt.plot(c, '.', label = 'Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss & Accuracy')
plt.legend()
plt.savefig('31.jpg', dpi=600)
"""


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    # ys代表目前已生成的序列，最初为仅包含一个起始符的序列，不断将预测结果追加到序列最后
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    outputs = []
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).unsqueeze(0).cuda())
        prob = model.generator(out[:, -1])
        outputs.append(prob.unsqueeze(0).unsqueeze(0))
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return torch.cat(outputs, 1), ys


# 从测试集中随机挑选20个句子来测试翻译的结果
indices = np.random.choice(range(len(test_X)), 20)
for ind in indices:
    data = [test_X[ind]]
    data_mask = [test_X_mask[ind]]
    target = [test_Y[ind]]
    print(data[0])
    print(SentenceFromList(input_lang, data[0]))
    input_variable = torch.LongTensor(data).cuda() if use_cuda else torch.LongTensor(data)
    # input_variable的大小：batch_size, length_seq
    input_mask = torch.Tensor(data_mask).to(torch.bool).cuda() if use_cuda else torch.Tensor(data_mask).to(torch.bool)
    # input_mask的大小：batch_size, 1, length_seq
    target_variable = torch.LongTensor(target).cuda() if use_cuda else torch.LongTensor(target)
    # target_variable的大小：batch_size, length_seq
    outputs, output_sentence = greedy_decode(model, input_variable, input_mask, max_length, 0)
    # outputs的大小：batch_size, length_seq, output_lang.n_words

    # Without teacher forcing: use its own predictions as the next input
    rights = []
    for di in range(MAX_LENGTH - 1):
        right = rightness(outputs[:, di, :], target_variable[:, di])
        rights.append(right)
    sentence = SentenceFromList(output_lang, output_sentence.cpu().numpy().reshape(-1).tolist())
    standard = SentenceFromList(output_lang, target[0])
    print("机器翻译：", sentence)
    print("标准翻译：", standard)
    # 输出本句话的准确率
    right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
    print("词准确率：", 100.0 * right_ratio)
    print("\n")


# 通过几个特殊的句子翻译，考察注意力机制关注的情况
input_sentence = "elle est trop petit ."
data = np.array([indexFromSentence(input_lang, input_sentence)])

input_variable = torch.LongTensor(data).cuda() if use_cuda else torch.LongTensor(data)
# input_variable的大小：batch_size, length_seq
input_mask = torch.Tensor(data_mask).to(torch.bool).cuda() if use_cuda else torch.Tensor(data_mask).to(torch.bool)
# input_mask的大小：batch_size, 1, length_seq
target_variable = torch.LongTensor(target).cuda() if use_cuda else torch.LongTensor(target)
# target_variable的大小：batch_size, length_seq

loss = 0
outputs, output_sentence = greedy_decode(model, input_variable, input_mask, max_length, 0)
# outputs的大小：batch_size, length_seq, output_lang.n_words

decoder_attentions = torch.zeros(max_length, max_length)
for di in range(MAX_LENGTH - 1):
    right = rightness(outputs[:, di, :], target_variable[:, di])
    rights.append(right)
sentence = SentenceFromList(output_lang, output_sentence.numpy().reshape(-1).tolist())
print("机器翻译：", sentence)
print("\n")


# 将每一步存储的注意力权重组合到一起就形成了注意力矩阵，绘制为图
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(model.decoder.layers[-1].src_attn.attn[0, 0].detach().numpy(), cmap="bone")
fig.colorbar(cax)

# 设置坐标轴
ax.set_xticklabels([""] + input_sentence.split(" ") + ["<EOS>"], rotation=90)
ax.set_yticklabels([""] + sentence.split(" "))

# 在标度上展示单词
import matplotlib.ticker as ticker

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
plt.savefig("32.jpg", dpi=600)
plt.show()


index = np.random.choice(range(len(test_X)), 1)[0]
data = [test_X[index]]
data_mask = [test_X_mask[index]]
target = [test_Y[index]]
target_mask = [test_Y_mask[index]]
data = np.array([indexFromSentence(input_lang, input_sentence)])

input_variable = torch.LongTensor(data).cuda() if use_cuda else torch.LongTensor(data)
# input_variable的大小：batch_size, length_seq
input_mask = torch.Tensor(data_mask).to(torch.bool).cuda() if use_cuda else torch.Tensor(data_mask).to(torch.bool)
# input_mask的大小：batch_size, 1, length_seq
target_variable = torch.LongTensor(target).cuda() if use_cuda else torch.LongTensor(target)
# target_variable的大小：batch_size, length_seq
target_mask = torch.Tensor(target_mask).to(torch.bool).cuda() if use_cuda else torch.Tensor(target_mask).to(torch.bool)
# target_mask的大小：batch_size, length_seq, length_seq

outputs = model(input_variable, target_variable, input_mask, target_mask)
# outputs的大小：batch_size, length_seq, output_lang.n_words

output_sentence = outputs.argmax(2)
sentence = SentenceFromList(output_lang, output_sentence.numpy().reshape(-1).tolist())
print("机器翻译：", sentence)
print("\n")


# 将每一步存储的注意力权重组合到一起就形成了注意力矩阵，绘制为图
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(model.decoder.layers[-1].src_attn.attn[0, 0].detach().numpy(), cmap="bone")
fig.colorbar(cax)

# 设置坐标轴
ax.set_xticklabels([""] + input_sentence.split(" ") + ["<EOS>"], rotation=90)
ax.set_yticklabels([""] + sentence.split(" "))

# 在标度上展示单词
import matplotlib.ticker as ticker

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.savefig("33.jpg", dpi=600)
plt.show()
