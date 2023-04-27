# -*- coding: UTF-8 -*-
"""
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@ File        : Tatoeba_Transformer_PyTorch_bleu.py
@ Time        : 2023/4/25 11:25
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (Conda)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
数据集：https://www.manythings.org/anki/
模型参考文档：https://pytorch.org/tutorials/beginner/translation_transformer.html?highlight=transformer%20translation
简要概述：
    暂时只实验了中英翻译，./data/eng-zh.txt 繁中转换成了简中（参考 utils/cmn-zh.py）。该代码采用 Spacy 3.5 语言模型生成tokenizer，
    因此需要提前安装对应语言模块，如 英文-简中 翻译，需要安装 zh_core_web_sm 和 en_core_web_sm，这些均可以在下方 SPACY 变量进行调整
    ！改动：代码训练、验证过程中计算的准确率调整为BLEU计算
实验设备：
    CPU: 10th Gen Intel(R) Core(TM) i5-10400
    GPU:  Nvidia GeForce RTX2060-Super 8GB
依赖模块（带*为可以最新版本）：
    torch       1.12.1+cu116
    torchtext   0.13.1
    spacy       3.5.0
    matplotlib  3.6.2*
    pandas      1.5.1*
    tqdm        4.64.1*
+++++++++++++++++++++++++++++++++++
"""

# 导入基础模块
from collections import OrderedDict
from pathlib import Path
import warnings
import argparse
import math
import time
import json

# 导入依赖模块
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 导入torch相关模块
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, BatchSampler
from torchtext.vocab import build_vocab_from_iterator, vocab
from torchtext.data import get_tokenizer, bleu_score

# 设置忽略warning信息
warnings.filterwarnings("ignore")

"""++++++++++++++++++++++++
@@@ 全局变量
++++++++++++++++++++++++"""

LANGUAGE = {}
# 定义特殊TOKEN
PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]
# 定义 spacy 语言模型库，用于分词，该部分可以自行增加
# 注意！运行时请确保输入的 src_lang 和 tgt_lang 能够在此查询到相对应的 Spacy 语言模块，否则会构造数据集时报错
SPACY = {
    "de": "de_core_news_sm",
    "en": "en_core_web_sm",
    "zh": "zh_core_web_sm"
}

"""+++++++++++++++++++++++++
@@@ 变量对象定义
+++++++++++++++++++++++++"""


def get_args_parser():
    """ 设置对象解释器 """
    parser = argparse.ArgumentParser('Transformer Arguments', add_help=False)

    # 随机数种子
    parser.add_argument("--seed", default=666, type=int)

    # 数据集参数
    parser.add_argument("--data_dir", default="./data/eng-zh.txt", type=str, help="表格间隔符的txt数据文件地址")
    parser.add_argument("--src_lang", default="en", type=str, help="输入语言，如en：英语")
    parser.add_argument("--tgt_lang", default="zh", type=str, help="输出语言，如zh：简中")
    parser.add_argument("--src_dict", default=None, type=str, help="输入语言词典txt文件")
    parser.add_argument("--tgt_dict", default=None, type=str, help="输出语言词典txt文件")
    parser.add_argument("--min_freq", default=1, type=str, help="词频最少数量，小于该数将不会加载进词汇中")
    parser.add_argument("--val_size", default=1000, type=int, help="验证数据量")
    parser.add_argument("--test_size", default=100, type=int, help="测试数据量")

    # 模型部分参数
    parser.add_argument("--n_enc_layers", default=6, type=int, help="Encoder编码器层数")
    parser.add_argument("--n_dec_layers", default=6, type=int, help="Decoder解码器层数")
    parser.add_argument("--n_heads", default=8, type=int, help="多头注意力头的数量")
    parser.add_argument("--d_model", default=512, type=int, help="Embedding嵌入层维数")
    parser.add_argument("--d_ff", default=2048, type=int, help="FFN维数")
    parser.add_argument("--dropout", default=.1, type=float, help="LN层Dropout比例")
    parser.add_argument("--resume", default=None, type=str, help="导入模型权重路径")

    # 训练参数
    parser.add_argument("--epochs", default=100, type=int, help="训练轮数")
    parser.add_argument("--lr", default=0.0001, type=float, help="学习率")
    parser.add_argument("--batch_size", default=64, type=int, help="批数量")
    parser.add_argument("--n_workers", default=0, type=int, help="读取数据进程数，使用交互式窗口运行时请设置为0")
    parser.add_argument("--device", default="cuda:0", type=str, help="运算设备")
    parser.add_argument("--output", default="output/eng-zh", type=str, help="训练结果保存路径")

    return parser


"""++++++++++++++++++++++
@@@ 数据集构建
++++++++++++++++++++++"""


class TranslationDataset(Dataset):
    """ 机器翻译数据集 """

    def __init__(self,
                 file_path: str,
                 src_lang: str = "en",
                 tgt_lang: str = "zh",
                 src_dict: str = None,
                 tgt_dict: str = None):
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
        if src_dict is not None:
            with open(src_dict, encoding="utf-8") as f:
                src_words = list(map(lambda i: i.strip(), f.readlines()))
            self.src_vocab = vocab(OrderedDict(zip(src_words, [10] * len(src_words))))
        else:
            self.src_vocab = build_vocab_from_iterator(self.src_sentences, 1, specials=SPECIALS)
        if tgt_dict is not None:
            with open(tgt_dict, encoding="utf-8") as f:
                tgt_words = list(map(lambda i: i.strip(), f.readlines()))
            self.tgt_vocab = vocab(OrderedDict(zip(tgt_words, [10] * len(tgt_words))))
        else:
            self.tgt_vocab = build_vocab_from_iterator(self.tgt_sentences, 1, specials=SPECIALS)

        # 设置默认字符为 <unk> 的索引
        self.src_vocab.set_default_index(UNK_IDX)
        self.tgt_vocab.set_default_index(UNK_IDX)

    def __len__(self):
        """ 数据集整体长度 """
        return self.length

    def __repr__(self):
        """ 字符串可视化显示数据集信息 """
        return " Dataset Info ".center(50, "=") + "\n" + \
            "| %-21s | %-22s |\n" % ("size", self.length) + \
            "| %-21s | %-22s |\n" % (f"src vocab: {self.src_lang}", len(self.src_vocab)) + \
            "| %-21s | %-22s |\n" % (f"tgt vocab: {self.tgt_lang}", len(self.tgt_vocab)) + "=" * 50

    def __getitem__(self, idx):
        """ 根据索引 idx 获取 src、tgt 的 tokens """
        # 通过 vocab 获取 token，并且前后插入起始、终止符号
        src = [BOS_IDX] + self.src_vocab.lookup_indices(self.src_sentences[idx]) + [EOS_IDX]
        tgt = [BOS_IDX] + self.tgt_vocab.lookup_indices(self.tgt_sentences[idx]) + [EOS_IDX]
        return torch.tensor(src), torch.tensor(tgt)

"""++++++++++++++++++++++
@@@ Mask处理
++++++++++++++++++++++"""


def generate_square_subsequent_mask(sz: int) -> Tensor:
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
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src: Tensor, tgt: Tensor) -> object:
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


"""+++++++++++++++++++++
@@@ 模型搭建
+++++++++++++++++++++"""


class PositionalEncoding(nn.Module):
    """ 位置编码层 """

    def __init__(self, emb_size: int, dropout: float, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    """ 词向量嵌入层 """

    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Seq2Seq Network
class Transformer(nn.Module):
    """ Transformer Seq2Seq模型 """

    def __init__(self,
                 num_enc_layers: int,
                 num_dec_layers: int,
                 d_model: int,
                 n_head: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Transformer, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=n_head,
                                          num_encoder_layers=num_enc_layers,
                                          num_decoder_layers=num_dec_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        # 输出全连接层：由于使用的 nn.CrossEntropyLoss 因此不需要作 log_softmax 处理
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)


"""++++++++++++++++++++++
@@@ 计算BLEU Score
++++++++++++++++++++++"""


def get_bleu_score(tgt, pred, vocab):
    candidate_corpus = []
    references_corpus = []
    for i in range(tgt.shape[1]):
        L = (tgt[:, i] != PAD_IDX).sum()
        candidate_corpus.append(vocab.lookup_tokens(pred[:, :L].flatten().tolist()))
        references_corpus.append([vocab.lookup_tokens(tgt[:, :L].flatten().tolist())])

    return bleu_score(candidate_corpus, references_corpus)


"""++++++++++++++++++++++
@@@ Greedy Decode和预测
++++++++++++++++++++++"""


# function to generate output sequence using greedy algorithm
def greedy_decode(model: nn.Module,
                  src: Tensor,
                  src_mask: Tensor,
                  max_len: int,
                  start_symbol: int) -> Tensor:
    """
    预测所使用的贪婪解码算法
    Args:
        model: torch.nn.Module 模型对象
        src: torch.Tensor 输入词向量 (S_src, 1)
        src_mask: torch.Tensor 输入词向量 mask (S_src, S_src)
        max_len: int 预测最大句子长度，通常时 S_src + 5
        start_symbol: int 起始符索引，默认为 2
        device: torch.device 运算设备

    Returns:
        Tensor(S_pred, 1)
    """
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(src.device)
    for i in range(max_len - 1):
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(src.device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


def translate(model: torch.nn.Module, src_sentence: str, tokenizer, vocab, device) -> str:
    """

    Args:
        model: torch.nn.Module 模型对象
        src_sentence: str 待翻译句子
        tokenizer: functools.partial 分词器
        vocab: torchtext.vocab.vocab.Vocab torchtext生成的词汇对象
        device: torch.device 运算设备

    Returns:
        str
    """
    model.eval()
    src = tokenizer(src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX, device=device).flatten()
    return " ".join(vocab.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    # ------------------------ #
    # Config
    # ------------------------ #

    # 设置随机数种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # 将输入和翻译语言构造索引
    LANGUAGE[args.src_lang] = 0
    LANGUAGE[args.tgt_lang] = 1
    if args.src_lang not in SPACY.keys() or args.tgt_lang not in SPACY.keys():
        raise "请确保 spacy 模块中是否包含设置的输入、输出语言！"
    # 设置开始轮数
    start_epoch = -1
    # 设置训练硬件
    args.device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    # 获取输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------ #
    # Load Data
    # ------------------------ #

    # 构建数据集
    dataset = TranslationDataset(file_path=args.data_dir, src_lang=args.src_lang, tgt_lang=args.tgt_lang,
                                 src_dict=args.src_dict, tgt_dict=args.tgt_dict)
    print(dataset)
    # 保存 vocab
    with open(output_dir / "src_dict.txt", "w+", encoding="utf-8") as f:
        f.writelines(map(lambda i: i + "\n", dataset.src_vocab.get_itos()))
    with open(output_dir / "tgt_dict.txt", "w+", encoding="utf-8") as f:
        f.writelines(map(lambda i: i + "\n", dataset.tgt_vocab.get_itos()))

    # 划分训练、验证、测试集，分批
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    train_sampler = BatchSampler(indices[:-(args.val_size + args.test_size)], args.batch_size, False)
    val_sampler = BatchSampler(indices[-(args.val_size + args.test_size):-args.test_size], args.batch_size, False)
    test_sampler = BatchSampler(indices[-args.test_size:], batch_size=1, drop_last=False)

    # 生成数据迭代器
    train_db = DataLoader(dataset, batch_sampler=train_sampler, num_workers=args.n_workers, collate_fn=collate_fn)
    val_db = DataLoader(dataset, batch_sampler=val_sampler, num_workers=args.n_workers, collate_fn=collate_fn)
    test_db = DataLoader(dataset, batch_sampler=test_sampler, num_workers=args.n_workers, collate_fn=collate_fn)

    # ------------------------ #
    # Build Model
    # ------------------------ #

    model = Transformer(args.n_enc_layers,
                        args.n_dec_layers,
                        args.d_model,
                        args.n_heads,
                        len(dataset.src_vocab),
                        len(dataset.tgt_vocab),
                        args.d_ff,
                        args.dropout).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    # 设置断点训练
    if args.resume:
        resume = torch.load(args.resume)
        start_epoch = resume["epoch"]
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])

    # ---------------------------- #
    # Training and Evaluate Model
    # ---------------------------- #

    print("Start training...")
    BEST_BLEU = 0.  # 记录最优准确率的 epoch
    # 记录每个 step 的损失喝准确率
    LOSS = {"train": [[] for _ in range(args.epochs)], "val": [[] for _ in range(args.epochs)]}
    BLEU = {"train": [[] for _ in range(args.epochs)], "val": [[] for _ in range(args.epochs)]}

    st = time.time()
    for epoch in range(start_epoch + 1, args.epochs):
        with tqdm(total=len(train_db), desc=f"Epoch: {epoch + 1}/{args.epochs}") as pbar:  # 训练进度条
            model.train()
            for step, (src, tgt) in enumerate(train_db):
                # 转移张量至指定运算硬件上
                src, tgt = src.to(args.device), tgt.to(args.device)
                tgt_input = tgt[:-1, :]  # 前移一位，不需要终止符

                # 生成 mask
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

                # 计算每个词的概率
                logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

                optimizer.zero_grad()  # 初始化优化器梯度

                # 计算损失
                tgt_out = tgt[1:, :]  # 前移一位，不需要起始符
                loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
                loss.backward()  # 反向传播

                optimizer.step()  # 更新模型参数

                # 计算 BLEU
                pred = torch.argmax(F.log_softmax(logits, -1), -1)
                bleu = get_bleu_score(tgt_out, pred, dataset.tgt_vocab)

                # 记录损失、准确率
                LOSS["train"][epoch].append(loss.item())
                BLEU["train"][epoch].append(bleu)
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({"loss": f"{np.mean(LOSS['train'][epoch]).item():.4f}",
                                  "bleu_score": f"{np.mean(BLEU['train'][epoch]).item() * 100:.2f}%"})

            # 训练完一个 epoch 后进行验证
            model.eval()
            for step, (src, tgt) in enumerate(val_db):
                src, tgt = src.to(args.device), tgt.to(args.device)
                tgt_input = tgt[:-1, :]

                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

                logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

                tgt_out = tgt[1:, :]
                loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

                # 计算 BLEU
                pred = torch.argmax(F.log_softmax(logits, -1), -1)
                bleu = get_bleu_score(tgt_out, pred, dataset.tgt_vocab)

                # 记录验证损失、准确率
                LOSS["val"][epoch].append(loss.item())
                BLEU["val"][epoch].append(bleu)
                # 更新进度条
                pbar.set_postfix({"loss": f"{np.mean(LOSS['train'][epoch]).item():.4f}",
                                  "bleu_score": f"{np.mean(BLEU['train'][epoch]).item() * 100:.2f}%",
                                  "loss_val": f"{np.mean(LOSS['val'][epoch]).item():.4f}",
                                  "bleu_score_val": f"{np.mean(BLEU['val'][epoch]).item() * 100:.2f}%"})

            # 判断验证数据平均准确率是否大于最优准确率，若更高则保存 best_model.pth
            if np.mean(BLEU["val"][epoch]).item() > BEST_BLEU:
                BEST_BLEU = np.mean(BLEU["val"][epoch]).item()
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch},
                           output_dir / "best_model.pth")

    et = time.time()
    print("Time taken: %ds" % (et - st))

    # 保存最终的模型权重
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": args.epochs},
               output_dir / "final_model.pth")
    print("模型权重保存到：%s" % output_dir)
    # 保存损失、准确率
    print("训练、验证损失和准确率保存到：%s" % (output_dir / "metrics.json"))
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"loss": LOSS, "bleu": BLEU}, f)

    # ------------------------ #
    # Visualization
    # ------------------------ #

    # 绘制损失图
    plt.figure()
    plt.plot(np.mean(LOSS["train"], 1), label="Train")
    plt.plot(np.mean(LOSS["val"], 1), label="Val")
    plt.legend(loc="best")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(output_dir / "loss.png")
    # 绘制精准率图
    plt.figure()
    plt.plot(np.mean(BLEU["train"], 1), label="Train")
    plt.plot(np.mean(BLEU["val"], 1), label="Val")
    plt.legend(loc="best")
    plt.xlabel("Epoch")
    plt.ylabel("Bleu")
    plt.tight_layout()
    plt.savefig(output_dir / "bleu.png")

    # ------------------------ #
    # Evaluating Model
    # ------------------------ #

    # 读取模型权重
    # model.load_state_dict(torch.load(output_dir / "model.pth"))
    print("Start Evaluate...")
    model.eval()
    with tqdm(total=len(val_db), desc="Evaluating") as pbar:
        for step, (src, tgt) in enumerate(val_db):
            src, tgt = src.to(args.device), tgt.to(args.device)
            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

            logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

            tgt_out = tgt[1:, :]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

            # 计算 BLEU
            pred = torch.argmax(F.log_softmax(logits, -1), -1)
            bleu = get_bleu_score(tgt_out, pred, dataset.tgt_vocab)

            pbar.update(1)
            pbar.set_postfix({"loss_val": f"{loss.item():.4f}",
                              "bleu_val": f"{bleu * 100:.2f}%"})

    # ------------------------ #
    # Prediction
    # ------------------------ #

    print("Start Testing...")
    for src, tgt in test_db:
        src = src.to(args.device)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(args.device)
        tgt_tokens = greedy_decode(model, src, src_mask, num_tokens + 5, BOS_IDX)

        print("原始翻译：", " ".join(dataset.tgt_vocab.lookup_tokens(tgt.flatten().tolist())))
        print("模型翻译：", " ".join(dataset.tgt_vocab.lookup_tokens(tgt_tokens.flatten().tolist())))
        print()
