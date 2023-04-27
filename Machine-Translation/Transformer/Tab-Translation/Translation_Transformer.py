# -*- coding: utf-8 -*-
"""++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # 机器翻译的神经网络实现
# 本节课我们讲述了利用Transformer网络实现机器翻译，来自帮助文档。
# 损失函数使用交叉熵函数，评价指标使用准确率
# 整个代码包括了数据预处理、Transformer网络两个部分组成。

Created on Tue Apr 25 01:03:05 2023
@author: 陈焯辉
1. 数据处理部分：
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
2. 网络搭建部分：
https://pytorch.org/tutorials/beginner/transformer_tutorial.html
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"""

# 导入模块
import re
import math
import time
import torch
import sklearn
import argparse
import unicodedata
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt

# 参数设置
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
parser = argparse.ArgumentParser(description='Transformer')
parser.add_argument('--lang1', type=str, default='fre', help='法语')
parser.add_argument('--lang2', type=str, default='eng', help='英语')
parser.add_argument('--reverse', type=bool, default=False, help='False为法-英，True为英-法')
parser.add_argument('--val_size', type=float, default=0.1, help='验证集比例')
parser.add_argument('--test_size', type=float, default=0.1, help='测试集比例')
parser.add_argument('--D', type=int, default=512, help='模型维数')
parser.add_argument('--H', type=int, default=8, help='多头自注意力机制头数')
parser.add_argument('--N', type=int, default=6, help='编码器与解码器的层数')
parser.add_argument('--L', type=int, default=10, help='句子最大长度')
parser.add_argument('--d_ff', type=int, default=2048, help='前馈神经网络的隐藏神经元个数')
parser.add_argument('--d_qkv', type=int, default=64, help='QKV向量的维数')
parser.add_argument('--dropout', type=float, default=0.1, help='剪枝层剪枝比例')
parser.add_argument('--epochs', type=int, default=1, help='训练周期数')
parser.add_argument('--batch_size', type=int, default=32, help='批训练大小')
args = parser.parse_args()
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 选择CPU或GPU

"""------------------------------
@@@    载入翻译数据集与数据准备
------------------------------"""
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: '<pad>', 1: '<bos>', 2: '<eos>', 3: '<unk>'}
        self.n_words = 4  # 统计index2word

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# 将 Unicode 字符串转换为纯 ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# 去停用词
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r'([.!?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    return s


def readLangs(lang1, lang2, reverse=False):
    lines = open('data/%s_%s.txt' % (lang1, lang2), encoding='utf-8'). \
        read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        src_lang = Lang(lang2)
        tgt_lang = Lang(lang1)
    else:
        src_lang = Lang(lang1)
        tgt_lang = Lang(lang2)
    return src_lang, tgt_lang, pairs


eng_prefixes = (
    'i am ', 'i m ',
    'he is', 'he s ',
    'she is', 'she s ',
    'you are', 'you re ',
    'we are', 'we re ',
    'they are', 'they re '
)


def filterPair(p):
    return len(p[0].split(' ')) < args.L - 1 and \
        len(p[1].split(' ')) < args.L - 1


def filterPairs(pairs):
    pairs = [pair for pair in pairs if len(pair) == 2]
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    src_lang, tgt_lang, pairs = readLangs(lang1, lang2, reverse)
    print('加载 %s 个句子对' % len(pairs))
    pairs = filterPairs(pairs)
    print('筛选出 %s 个句子对' % len(pairs))
    for pair in pairs:
        src_lang.addSentence(pair[0])
        tgt_lang.addSentence(pair[1])
    print('统计单词：')
    print(src_lang.name, src_lang.n_words)
    print(tgt_lang.name, tgt_lang.n_words)
    return src_lang, tgt_lang, pairs


def indexesFromSentence(lang, sentence):
    indexes = [BOS_IDX]
    indexes.extend([lang.word2index[word] for word in sentence.split(' ')])
    indexes.append(EOS_IDX)
    for i in range(args.L - len(indexes)):
        indexes.append(PAD_IDX)
    return indexes


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=args.device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_mask = torch.zeros((src.shape[1], src.shape[1]), device=args.device).type(torch.bool)
    tgt_mask = generate_square_subsequent_mask(tgt.shape[1])

    src_pad_mask = (src == PAD_IDX)
    tgt_pad_mask = (tgt == PAD_IDX)
    return src_mask, tgt_mask, src_pad_mask, tgt_pad_mask


src_lang, tgt_lang, pairs = prepareData(args.lang1, args.lang2, args.reverse)
src_vocab = src_lang.n_words
tgt_vocab = tgt_lang.n_words

src = [indexesFromSentence(src_lang, pair[0]) for pair in pairs]
src = torch.LongTensor(np.array(src))
tgt = [indexesFromSentence(tgt_lang, pair[1]) for pair in pairs]
tgt = torch.LongTensor(np.array(tgt))
src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(src, tgt)

dataset = torch.utils.data.TensorDataset(src, tgt, src_pad_mask, tgt_pad_mask)
# 拆分训练集、验证集与测试集
indices = list(range(len(dataset)))
indices = sklearn.utils.shuffle(indices, random_state=seed)  # 随机打乱
indices_train = indices[int(len(dataset) * (args.test_size + args.val_size)):]
indices_val = indices[:int(len(dataset) * args.val_size)]
indices_test = indices[int(len(dataset) * args.val_size): int(len(dataset) * (args.test_size + args.val_size))]
train_db = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size,
                                       shuffle=False, sampler=indices_train)
val_db = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size,
                                     shuffle=False, sampler=indices_val)
test_db = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size,
                                      shuffle=False, sampler=indices_test)
print('train data: src:', (len(indices_train), args.L), 'trg:', (len(indices_train), args.L))
print('val   data: src:', (len(indices_val), args.L), 'trg:', (len(indices_val), args.L))
print('test  data: src:', (len(indices_test), args.L), 'trg:', (len(indices_test), args.L))

"""------------------------------
@@@     搭建Transformer模型
------------------------------"""


# 位置编码器
class PositionalEncoding(nn.Module):
    def __init__(self, L=5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, args.D, 2) * math.log(10000) / args.D)
        pos = torch.arange(0, L).reshape(L, 1)
        self.pos_embedding = torch.zeros((L, args.D)).to(args.device)
        self.pos_embedding[:, 0::2] = torch.sin(pos * den)
        self.pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.pos_embedding = self.pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, embedding):
        return self.dropout(embedding + self.pos_embedding[:embedding.size(0), :])


# 嵌入层
class Embedding(nn.Module):
    def __init__(self, vocab_size):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, args.D)

    def forward(self, tokens):
        return self.embedding(tokens) * math.sqrt(args.D)


# Transformer网络
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.src_embed = Embedding(src_vocab)
        self.tgt_embed = Embedding(tgt_vocab)
        self.positional_encoding = PositionalEncoding()
        self.transformer = nn.Transformer(d_model=args.D, nhead=args.H, num_encoder_layers=args.N,
                                          num_decoder_layers=args.N, dim_feedforward=args.d_ff,
                                          dropout=args.dropout, batch_first=True)
        self.generator = nn.Linear(args.D, tgt_vocab)
        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask):
        src_emb = self.positional_encoding(self.src_embed(src))
        tgt_emb = self.positional_encoding(self.tgt_embed(tgt))
        transformer = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                       src_padding_mask, tgt_padding_mask, src_padding_mask)
        return self.generator(transformer)

    def encode(self, src, src_mask):
        src_emb = self.positional_encoding(self.src_embed(src))
        return self.transformer.encoder(src_emb, src_mask)

    def decode(self, tgt, memory, tgt_mask):
        tgt_emb = self.positional_encoding(self.tgt_embed(tgt))
        return self.transformer.decoder(tgt_emb, memory, tgt_mask)


model = Transformer().to(args.device)

"""------------------------------
@@@          模型训练
------------------------------"""
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

LOSS = []
LOSS_VAL = []
ACCURACY = []
ACCURACY_VAL = []
t1 = time.time()
for epoch in range(args.epochs):
    Loss = []
    total = 0
    correct = 0
    with tqdm(total=len(train_db), desc='Epoch {}/{}'.format(epoch + 1, args.epochs)) as pbar:
        for step, (src, tgt, src_pad_mask, tgt_pad_mask) in enumerate(train_db):
            model.train()
            src = src.to(args.device)
            tgt = tgt.to(args.device)
            src_pad_mask = src_pad_mask.to(args.device)
            tgt_pad_mask = tgt_pad_mask.to(args.device)
            output = model(src, tgt, src_mask.to(args.device), tgt_mask.to(args.device), src_pad_mask, tgt_pad_mask)
            # 损失函数
            CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
            loss = CrossEntropyLoss(output.view((output.shape[0] * output.shape[1], output.shape[2])), tgt.reshape(-1))
            Loss.append(float(loss))
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()  # 一步随机梯度下降算法
            # 计算准确率
            pred = torch.argmax(output, 2)
            for i in range(output.shape[0]):
                L = (tgt[i, :] != PAD_IDX).sum()
                total += L
                correct += (pred[i, :L] == tgt[i, :L]).sum()
            pbar.set_postfix({'loss': '%.4f' % np.mean(Loss),
                              'accuracy': '%.2f' % ((correct / total) * 100) + '%'})

            # 验证
            model.eval()
            if step == len(train_db) - 1:
                Loss_val = []
                total_val = 0
                correct_val = 0
                for src, tgt, src_pad_mask, tgt_pad_mask in val_db:
                    src = src.to(args.device)
                    tgt = tgt.to(args.device)
                    src_pad_mask = src_pad_mask.to(args.device)
                    tgt_pad_mask = tgt_pad_mask.to(args.device)
                    output = model(src, tgt, src_mask.to(args.device), tgt_mask.to(args.device), src_pad_mask,
                                   tgt_pad_mask)
                    # 损失函数
                    CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
                    loss_val = CrossEntropyLoss(output.view((output.shape[0] * output.shape[1], output.shape[2])),
                                                tgt.reshape(-1))
                    Loss_val.append(float(loss_val))
                    # 计算准确率
                    pred = torch.argmax(output, 2)
                    for i in range(output.shape[0]):
                        L = (tgt[i, :] != PAD_IDX).sum()
                        total_val += L
                        correct_val += (pred[i, :L] == tgt[i, :L]).sum()
                pbar.set_postfix({'loss': '%.4f' % np.mean(Loss),
                                  'val_loss': '%.4f' % np.mean(Loss_val),
                                  'accuracy': '%.2f' % ((correct / total).item() * 100) + '%',
                                  'val_accuracy': '%.2f' % ((correct_val / total_val).item() * 100) + '%'})
            pbar.update(1)
        LOSS.append(np.mean(Loss))
        LOSS_VAL.append(np.mean(Loss_val))
        ACCURACY.append((correct / total).item())
        ACCURACY_VAL.append((correct_val / total_val).item())
        torch.save(model.state_dict(), 'model_epoch%s.pth' % (epoch + 1))  # 保存模型权重
train_history = {'loss': LOSS, 'val_loss': LOSS_VAL,
                 'accuracy': ACCURACY, 'val_accuracy': ACCURACY_VAL}
t2 = time.time()
times = t2 - t1
print('Time taken: %d seconds' % times)


# 绘制拟合曲线
def show_train_history(type_str, train_type, val_type):
    plt.figure(dpi=200)
    plt.plot(train_history[train_type])
    plt.plot(train_history[val_type])
    plt.ylabel(type_str)
    plt.xlabel('训练次数')
    plt.legend(['训练集', '验证集'], loc='best')
    plt.show()


show_train_history('准确率/%', 'accuracy', 'val_accuracy')
show_train_history('损失值', 'loss', 'val_loss')

"""------------------------------
@@@          模型测试
------------------------------"""
Loss = []
total = 0
correct = 0
model.eval()
with tqdm(total=len(test_db)) as pbar:
    for src, tgt, src_pad_mask, tgt_pad_mask in test_db:
        src = src.to(args.device)
        tgt = tgt.to(args.device)
        src_pad_mask = src_pad_mask.to(args.device)
        tgt_pad_mask = tgt_pad_mask.to(args.device)
        output = model(src, tgt, src_mask.to(args.device), tgt_mask.to(args.device), src_pad_mask, tgt_pad_mask)
        # 损失函数
        CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        loss = CrossEntropyLoss(output.view((output.shape[0] * output.shape[1], output.shape[2])), tgt.reshape(-1))
        Loss.append(float(loss))
        # 计算准确率
        pred = torch.argmax(output, 2)
        for i in range(output.shape[0]):
            L = (tgt[i, :] != PAD_IDX).sum()
            total += L
            correct += (pred[i, :L] == tgt[i, :L]).sum()
        pbar.set_postfix({'loss': '%.4f' % np.mean(Loss),
                          'accuracy': '%.2f' % ((correct / total) * 100) + '%'})
        pbar.update(1)

"""------------------------------
@@@          模型预测
------------------------------"""


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(args.device)
    src_mask = src_mask.to(args.device)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(args.device)
    for i in range(max_len - 1):
        memory = memory.to(args.device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(1)).type(torch.bool)).to(args.device)
        out = model.decode(ys, memory, tgt_mask)
        # out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == EOS_IDX:
            break
    return ys


def translate(model, src_sentence):
    model.eval()  # torch.Size([32, 10]) torch.Size([10, 10])
    src = torch.LongTensor(indexesFromSentence(src_lang, src_sentence)).unsqueeze(0)
    src_mask = (torch.zeros(src.shape[1], src.shape[1])).type(torch.bool)
    tgt_tokens = greedy_decode(model, src, src_mask, max_len=src.shape[1] + 5, start_symbol=BOS_IDX).flatten()
    tgt_sentence = ' '.join(map(lambda tkn: tgt_lang.index2word[int(tkn)], tgt_tokens)).strip()
    return tgt_sentence


src_sentence = 'vous etes fort serviable .'
print(translate(model, src_sentence))
