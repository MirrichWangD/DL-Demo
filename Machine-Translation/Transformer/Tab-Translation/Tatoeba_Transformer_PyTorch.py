# -*- coding: UTF-8 -*-
"""
+++++++++++++++++++++++++++++++++++
@ File        : Tatoeba_Transformer_PyTorch.py
@ Time        : 2023/4/25 11:25
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (Conda)
+++++++++++++++++++++++++++++++++++
数据集：https://www.manythings.org/anki/
模型参考文档：https://pytorch.org/tutorials/beginner/translation_transformer.html?highlight=transformer%20translation
简要概述：
    暂时只实验了中英翻译，./data/eng-zh.txt 繁中转换成了简中（参考 utils/cmn-zh.py）。该代码采用 Spacy 3.5 语言模型生成tokenizer，
    因此需要提前安装对应语言模块，如 英文-简中 翻译，需要安装 zh_core_web_sm 和 en_core_web_sm，这些均可以在下方 SPACY 变量进行调整
    ！第一次改动：为整体代码增加 checkpoint 存储，该类型文件读取出来时 {"model": ..., "optimizer": ..., "epoch": ...} 字典
    ！第二次改动：计算准确率部分调整成 map(lambda ...) 形式，加快准确率计算
    ！第三次改动：增加测试推理结果时保存 <原始翻译>\t<模型翻译>\n 的行文本结果文件 test.txt
    ！第四次改动：增加  mode（设置脚本训练/验证推理模式） 和 max_len（分词后最大长度）传入参数，将 __main__ 脚本全部封装进 main() 函数
实验设备：
    CPU:    10th Gen Intel(R) Core(TM) i5-10400 2.9 GHz
    GPU:    Nvidia GeForce RTX2060-Super 8GB
    CUDA:   11.6 + CUDNN 8.3.02
依赖模块（带*为可以最新版本）：
    torch       1.12.1+cu116
    torchtext   0.13.1 (必须 >= 0.10.0)
    spacy       3.5.0 (语言模型帮助文档链接：https://spacy.io/models）
    matplotlib  3.6.2*
    pandas      1.5.1*
    tqdm        4.64.1*
实验记录：(max_len: 40; epochs: 40)
    eng-zh: 英文-简中 (batch_size: 128)
        size: 29155
        src vocab (en): 7899
        tgt vocab (zh): 11332
        Epoch: 40/40: 100% 220/220 [00:50<00:00,  4.32it/s, loss=0.2639, acc=93.44%, loss_val=2.5564, acc_val=65.00%]
        Evaluating: 100%|██████████| 8/8 [00:00<00:00, 14.45it/s, loss_val=2.5120, acc_val=65.41%]
    fre-eng: 法语-英文 (batch_size: 128)
        size: 134594
        src vocab (fr): 22760
        tgt vocab (en): 14441
        Epoch: 40/40: 100% 1043/1043 [03:40<00:00,  4.73it/s, loss=0.1917, acc=95.18%, loss_val=1.1674, acc_val=83.47%]
        Evaluating: 100%|██████████| 8/8 [00:00<00:00, 15.79it/s, loss_val=1.0950, acc_val=83.56%]
    eng-deu: 英文-德语 (batch_size: 64)
        size: 255816
        src vocab (de): 18971
        tgt vocab (en): 41032
        Epoch: 40/40: 100% 3980/3980 [12:53<00:00,  5.14it/s, loss=0.9235, acc=86.41%, loss_val=1.5359, acc_val=78.17%]
        Evaluating: 100%|██████████| 16/16 [00:01<00:00, 14.33it/s, loss_val=1.2981, acc_val=78.83%]
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
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, BatchSampler
from torchtext.vocab import build_vocab_from_iterator, vocab
from torchtext.data import get_tokenizer

# 设置忽略warning信息
warnings.filterwarnings("ignore")

"""++++++++++++++++++++++++
@@@ 全局变量
++++++++++++++++++++++++"""

# 定义特殊TOKEN
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

"""+++++++++++++++++++++++++
@@@ 变量对象定义
    - 运行代码前请先查阅该部分，配置个人训练数据集参数、模型参数和训练参数再运行
    - 提示：如果是使用 Spyder、PyCharm、VS Code 等IDE内核运行的，请将n_workers设置为0！
    - 预测时请设置 mode="eval"，resume、src_dict、tgt_dict 的默认值（default）
+++++++++++++++++++++++++"""


def get_args_parser():
    """设置对象解释器"""
    parser = argparse.ArgumentParser("Transformer Arguments", add_help=False)

    # 随机数种子
    parser.add_argument("--seed", default=666, type=int)

    # 数据集参数
    parser.add_argument("--data_dir", default="./data/eng-deu.txt", type=str, help="表格间隔符的txt数据文件地址")
    parser.add_argument("--src_lang", default="en", type=str, help="输入语言，如en：英语")
    parser.add_argument("--tgt_lang", default="de", type=str, help="输出语言，如zh：简zh")
    parser.add_argument("--src_dict", default=None, type=str, help="预测时输入语言词典txt文件")
    parser.add_argument("--tgt_dict", default=None, type=str, help="预测时输出语言词典txt文件")
    parser.add_argument("--max_len", default=40, type=int, help="句子向量最大长度")
    parser.add_argument("--min_freq", default=1, type=str, help="词频最少数量，小于该数将不会加载进词汇中")
    parser.add_argument("--val_size", default=1000, type=int, help="验证数据量")
    parser.add_argument("--test_size", default=100, type=int, help="测试数据量")

    # 模型部分参数
    parser.add_argument("--n_enc_layers", default=6, type=int, help="Encoder编码器层数")
    parser.add_argument("--n_dec_layers", default=6, type=int, help="Decoder解码器层数")
    parser.add_argument("--n_heads", default=8, type=int, help="多头注意力头的数量")
    parser.add_argument("--d_model", default=512, type=int, help="Embedding嵌入层维数")
    parser.add_argument("--d_ff", default=2048, type=int, help="FFN维数")
    parser.add_argument("--dropout", default=0.1, type=float, help="LN层Dropout比例")
    parser.add_argument("--resume", default=None, type=str, help="导入模型权重路径")

    # 训练参数
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "eval"],
        help="脚本模式，train：训练；eval：验证",
    )
    parser.add_argument("--epochs", default=40, type=int, help="训练轮数")
    parser.add_argument("--lr", default=0.0001, type=float, help="学习率")
    parser.add_argument("--batch_size", default=64, type=int, help="批数量")
    parser.add_argument("--n_workers", default=0, type=int, help="DataLoader读取数据开启进程数")
    parser.add_argument("--device", default="cuda:0", type=str, help="运算设备")
    parser.add_argument("--output", default="output/eng-deu", type=str, help="训练结果保存路径")

    return parser


"""++++++++++++++++++++++
@@@ 数据集构建
++++++++++++++++++++++"""


class TranslationDataset(Dataset):
    """机器翻译数据集"""

    def __init__(
        self,
        file_path: str,
        src_lang: str,
        tgt_lang: str,
        src_dict: str = None,
        tgt_dict: str = None,
        max_len: int = 256,
        min_freq: int = 1,
    ):
        """

        Args:
            file_path (str): <src>\t<tgt>\n 一行的txt文件地址
            src_lang (str): 原始数据列索引
            tgt_lang (str): 翻译数据列索引
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
        for src, tgt in tqdm(texts[[0, 1]].values, total=self.length, desc="Tokenizing Data"):
            self.src_sentences.append(self.src_tokenizer(src)[:max_len])
            self.tgt_sentences.append(self.tgt_tokenizer(tgt)[:max_len])

        # 构建词汇表对象
        if src_dict is not None:
            with open(src_dict, encoding="utf-8") as fp:
                src_words = list(map(lambda i: i.strip("\n"), fp.readlines()))
            self.src_vocab = vocab(OrderedDict(zip(src_words, [10] * len(src_words))))
        else:
            self.src_vocab = build_vocab_from_iterator(self.src_sentences, min_freq, specials=SPECIALS)
        if tgt_dict is not None:
            with open(tgt_dict, encoding="utf-8") as fp:
                tgt_words = list(map(lambda i: i.strip("\n"), fp.readlines()))
            self.tgt_vocab = vocab(OrderedDict(zip(tgt_words, [10] * len(tgt_words))))
        else:
            self.tgt_vocab = build_vocab_from_iterator(self.tgt_sentences, min_freq, specials=SPECIALS)

        # 设置默认字符为 <unk> 的索引
        self.src_vocab.set_default_index(UNK_IDX)
        self.tgt_vocab.set_default_index(UNK_IDX)

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
@@@ Mask处理
++++++++++++++++++++++"""


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """
    生成方形后续掩码
    Args:
        sz (int): 生成方形掩码的尺寸

    Returns:
        torch.tensor: 返回对应的掩码方阵
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
        src (torch.Tensor): 输入词索引向量 (S_src, N)
        tgt (torch.Tensor): 输出词索引向量 (S_tgt, N)

    Returns:
        torch.Tensor(S_src, S_src), torch.Tensor(S_tgt, S_tgt), torch.Tensor(N, S_src), torch.Tensor(N, S_tgt)
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
        batch (iter): 批数据 (src_token(S_src), tgt_token(S_tgt))

    Returns:
        torch.Tensor(S_src, N), torch.Tensor(S_tgt, N)
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
    """位置编码层"""

    def __init__(self, emb_size: int, dropout: float, max_len: int = 5000):
        """
        位置编码层构造函数
        Args:
            emb_size (int): 嵌入维度
            dropout (float): 线性层 Dropout 比例
            max_len (int): 最大词长度
        """
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
        """
        词嵌入层构造方法
        Args:
            vocab_size (int): 词典长度
            emb_size (int): 嵌入维度
        """
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
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        """
        Transformer 构造方法
        Args:
            num_enc_layers (int): Encoder Block 数量
            num_dec_layers (int): Decoder Block 数量
            d_model (int): 词向量维度
            n_head (int): 多头注意力的头数量
            src_vocab_size (int): 输入词汇表长度
            tgt_vocab_size (int): 输出词汇表长度
            dim_feedforward (int): 前馈网络维度
            dropout (float): 前馈网络 Dropout 比例
        """
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
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

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

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)


"""++++++++++++++++++++++
@@@ Greedy Decode和预测
++++++++++++++++++++++"""


# function to generate output sequence using greedy algorithm
def greedy_decode(
    model: nn.Module,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
) -> torch.Tensor:
    """
    预测所使用的贪婪解码算法
    Args:
        model (torch.nn.Module): 模型对象
        src (torch.Tensor): 输入词向量 (S_src, 1)
        src_mask (torch.Tensor): 输入词向量 mask (S_src, S_src)
        max_len (int): 预测最大句子长度，通常时 S_src + 5
        start_symbol (int): 起始符索引，默认为 2

    Returns:
        torch.Tensor(S_pred, 1)
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


def main(args):
    # ------------------------ #
    # Config
    # ------------------------ #

    # 设置随机数种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.src_lang not in SPACY.keys() or args.tgt_lang not in SPACY.keys():
        raise "请确保 spacy 模块中是否包含设置的输入、输出语言！"
    # 设置开始轮数
    start_epoch = -1
    # 设置训练硬件
    args.device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    # 获取输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (output_dir / "words").mkdir(parents=True, exist_ok=True)

    # ------------------------ #
    # Load Data
    # ------------------------ #

    # 构建数据集
    dataset = TranslationDataset(
        file_path=args.data_dir,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        src_dict=args.src_dict,
        tgt_dict=args.tgt_dict,
        max_len=args.max_len,
        min_freq=args.min_freq,
    )
    print(dataset)

    # 划分训练、验证、测试集，分批
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    train_sampler = BatchSampler(indices[: -(args.val_size + args.test_size)], args.batch_size, False)
    val_sampler = BatchSampler(
        indices[-(args.val_size + args.test_size) : -args.test_size],
        args.batch_size,
        False,
    )
    test_sampler = BatchSampler(indices[-args.test_size :], batch_size=1, drop_last=False)

    # 生成数据迭代器
    train_db = DataLoader(
        dataset,
        batch_sampler=train_sampler,
        num_workers=args.n_workers,
        collate_fn=collate_fn,
    )
    val_db = DataLoader(
        dataset,
        batch_sampler=val_sampler,
        num_workers=args.n_workers,
        collate_fn=collate_fn,
    )
    test_db = DataLoader(
        dataset,
        batch_sampler=test_sampler,
        num_workers=args.n_workers,
        collate_fn=collate_fn,
    )

    # ------------------------ #
    # Build Model
    # ------------------------ #

    model = Transformer(
        args.n_enc_layers,
        args.n_dec_layers,
        args.d_model,
        args.n_heads,
        len(dataset.src_vocab),
        len(dataset.tgt_vocab),
        args.d_ff,
        args.dropout,
    ).to(args.device)

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

    if args.mode == "train":
        print("Start training...")
        # 保存 vocab
        with open(output_dir / "words" / "src_dict.txt", "w+", encoding="utf-8") as fp:
            fp.writelines(map(lambda i: i + "\n", dataset.src_vocab.get_itos()))
        with open(output_dir / "words" / "tgt_dict.txt", "w+", encoding="utf-8") as fp:
            fp.writelines(map(lambda i: i + "\n", dataset.tgt_vocab.get_itos()))
        # 记录每个 step 的损失喝准确率
        train_history = {
            "loss": {
                "train": [[] for _ in range(args.epochs)],
                "val": [[] for _ in range(args.epochs)],
            },
            "acc": {
                "train": [[] for _ in range(args.epochs)],
                "val": [[] for _ in range(args.epochs)],
            },
            "best_acc": 0.0,
        }

        st = time.time()
        for epoch in range(start_epoch + 1, args.epochs):
            # 初始化准确率计算变量
            correct, total = {"train": 0, "val": 0}, {"train": 0, "val": 0}

            model.train()
            with tqdm(total=len(train_db), desc=f"Epoch: {epoch + 1}/{args.epochs}") as pbar:  # 训练进度条
                for step, (src, tgt) in enumerate(train_db):
                    # 转移张量至指定运算硬件上
                    src, tgt = src.to(args.device), tgt.to(args.device)
                    tgt_input = tgt[:-1, :]  # 删除最后一位，为了匹配最后预测的形状

                    # 生成 mask
                    (
                        src_mask,
                        tgt_mask,
                        src_padding_mask,
                        tgt_padding_mask,
                    ) = create_mask(src, tgt_input)

                    # 计算每个词的概率
                    logits = model(
                        src,
                        tgt_input,
                        src_mask,
                        tgt_mask,
                        src_padding_mask,
                        tgt_padding_mask,
                        src_padding_mask,
                    )

                    optimizer.zero_grad()  # 初始化优化器梯度

                    # 计算损失
                    tgt_out = tgt[1:, :]  # 前移一位，不需要起始符
                    loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
                    loss.backward()  # 反向传播

                    optimizer.step()  # 更新模型参数

                    # 计算准确率
                    pred = torch.argmax(F.log_softmax(logits, -1), -1)
                    L = (tgt_out != PAD_IDX).sum(0)
                    total["train"] += L.sum().item()
                    correct["train"] += sum(
                        map(
                            lambda i: (pred[: L[i], i] == tgt_out[: L[i], i]).sum().item(),
                            range(pred.shape[1]),
                        )
                    )

                    # 记录损失、准确率
                    train_history["loss"]["train"][epoch].append(loss.item())
                    train_history["acc"]["train"][epoch].append(correct["train"] / total["train"])
                    # 更新进度条
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "loss": f"{np.mean(train_history['loss']['train'][epoch]).item():.4f}",
                            "acc": f"{np.mean(train_history['acc']['train'][epoch]).item() * 100:.2f}%",
                        }
                    )

                # 训练完一个 epoch 后进行验证
                model.eval()
                for step, (src, tgt) in enumerate(val_db):
                    src, tgt = src.to(args.device), tgt.to(args.device)
                    tgt_input = tgt[:-1, :]

                    (
                        src_mask,
                        tgt_mask,
                        src_padding_mask,
                        tgt_padding_mask,
                    ) = create_mask(src, tgt_input)

                    logits = model(
                        src,
                        tgt_input,
                        src_mask,
                        tgt_mask,
                        src_padding_mask,
                        tgt_padding_mask,
                        src_padding_mask,
                    )

                    tgt_out = tgt[1:, :]
                    loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

                    # 计算准确率
                    pred = torch.argmax(F.log_softmax(logits, -1), -1)
                    L = (tgt_out != PAD_IDX).sum(0)
                    total["val"] += L.sum().item()
                    correct["val"] += sum(
                        map(
                            lambda i: (pred[: L[i], i] == tgt_out[: L[i], i]).sum().item(),
                            range(pred.shape[1]),
                        )
                    )

                    # 记录验证损失、准确率
                    train_history["loss"]["val"][epoch].append(loss.item())
                    train_history["acc"]["val"][epoch].append(correct["val"] / total["val"])
                    # 更新进度条
                    pbar.set_postfix(
                        {
                            "loss": f"{np.mean(train_history['loss']['train'][epoch]).item():.4f}",
                            "acc": f"{np.mean(train_history['acc']['train'][epoch]).item() * 100:.2f}%",
                            "loss_val": f"{np.mean(train_history['loss']['val'][epoch]).item():.4f}",
                            "acc_val": f"{np.mean(train_history['acc']['val'][epoch]).item() * 100:.2f}%",
                        }
                    )

                # 判断验证数据平均准确率是否大于最优准确率，若更高则保存 best_model.pth
                if np.mean(train_history["acc"]["val"][epoch]).item() > train_history["best_acc"]:
                    train_history["best_acc"] = np.mean(train_history["acc"]["val"][epoch]).item()
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                        },
                        output_dir / "best_model.pth",
                    )

        et = time.time()
        print("Time taken: %ds" % (et - st))

        # 保存最终的模型权重
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": args.epochs,
            },
            output_dir / "final_model.pth",
        )
        print("模型权重保存到：%s" % output_dir)
        # 保存损失、准确率
        print("训练、验证损失和准确率保存到：%s" % (output_dir / "metrics" / "train_history.json"))
        with open(output_dir / "metrics" / "train_history.json", "w", encoding="utf-8") as f:
            json.dump(train_history, f)

        # 可视化训练过程（损失、准确率）
        ## 绘制损失图
        fig, ax = plt.subplots(1, 1, dpi=200)
        ax.plot(np.mean(train_history["loss"]["train"], 1), label="Train")
        ax.plot(np.mean(train_history["loss"]["val"], 1), label="Val")
        ax.legend(loc="best")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        fig.tight_layout()
        fig.savefig(output_dir / "metrics" / "loss.png")
        ## 绘制精准率图
        fig, ax = plt.subplots(1, 1, dpi=200)
        ax.plot(np.mean(train_history["acc"]["train"], 1), label="Train")
        ax.plot(np.mean(train_history["acc"]["val"], 1), label="Val")
        ax.legend(loc="best")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        fig.tight_layout()
        fig.savefig(output_dir / "metrics" / "accuracy.png")

    # ------------------------ #
    # Evaluating Model
    # ------------------------ #

    print("Start Evaluate...")
    correct, total = 0, 0
    model.eval()
    with tqdm(total=len(val_db), desc="Evaluating") as pbar:
        for step, (src, tgt) in enumerate(val_db):
            src, tgt = src.to(args.device), tgt.to(args.device)
            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

            logits = model(
                src,
                tgt_input,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                src_padding_mask,
            )

            tgt_out = tgt[1:, :]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

            # 计算准确率
            pred = torch.argmax(F.log_softmax(logits, -1), -1)
            L = (tgt_out != PAD_IDX).sum(0)
            total += L.sum().item()
            correct += sum(
                map(
                    lambda i: (pred[: L[i], i] == tgt_out[: L[i], i]).sum(),
                    range(pred.shape[1]),
                )
            )
            pbar.update(1)
            pbar.set_postfix(
                {
                    "loss_val": f"{loss.item():.4f}",
                    "acc_val": f"{(correct / total) * 100:.2f}%",
                }
            )

    # ------------------------ #
    # Prediction
    # ------------------------ #

    print("Start Testing...")
    with open(output_dir / "test.txt", "a", encoding="utf-8") as f:
        for src, tgt in test_db:
            src = src.to(args.device)
            num_tokens = src.shape[0]
            src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(args.device)
            tgt_tokens = greedy_decode(model, src, src_mask, num_tokens + 5, BOS_IDX)

            tgt_sentence = " ".join(dataset.tgt_vocab.lookup_tokens(tgt.flatten()[1:-1].tolist()))
            pred_sentence = " ".join(dataset.tgt_vocab.lookup_tokens(tgt_tokens.flatten()[1:-1].tolist()))

            f.write(f"{tgt_sentence}\t{pred_sentence}\n")

            print(
                f"原始翻译：{tgt_sentence}",
            )
            print(f"模型翻译：{pred_sentence}")
            print()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
