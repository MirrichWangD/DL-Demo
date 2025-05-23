# -*- coding: utf-8 -*-
"""
+++++++++++++++++++++++++++++++++++
@ File        : Seq2Seq-Attention.py
@ Time        : 2022/6/14 13:48
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (Conda)
+++++++++++++++++++++++++++++++++++
课程：“自然语言处理” 结课论文相关程序
数据集：https://www.manythings.org/anki/ （英-繁中）
修正：更新 torchtext 模块后调整旧版 Field 为 vocab相关类
运行记录：
    ================== Dataset Info ==================
    | size                  | 29155                  |
    | src vocab: en         | 7899                   |
    | tgt vocab: zh         | 11332                  |
    ==================================================
    Epoch 1/10: 100%|█| 220/220 [00:40<00:00,  5.45Batch/s, loss=4.7801, PPL=119.1161, val_loss=4.2438, val_PPL=69.6708]
    Epoch 2/10: 100%|█| 220/220 [00:44<00:00,  4.89Batch/s, loss=3.4289, PPL=30.8428, val_loss=3.6193, val_PPL=37.3105]
    Epoch 3/10: 100%|█| 220/220 [00:49<00:00,  4.41Batch/s, loss=2.6441, PPL=14.0704, val_loss=3.4348, val_PPL=31.0248]
    Epoch 4/10: 100%|█| 220/220 [00:50<00:00,  4.33Batch/s, loss=2.1623, PPL=8.6910, val_loss=3.4123, val_PPL=30.3339]
    Epoch 5/10: 100%|█| 220/220 [00:50<00:00,  4.40Batch/s, loss=1.8734, PPL=6.5104, val_loss=3.4155, val_PPL=30.4314]
    Epoch 6/10: 100%|█| 220/220 [00:54<00:00,  4.00Batch/s, loss=1.6949, PPL=5.4458, val_loss=3.4066, val_PPL=30.1639]
    Epoch 7/10: 100%|█| 220/220 [00:53<00:00,  4.10Batch/s, loss=1.5763, PPL=4.8371, val_loss=3.4197, val_PPL=30.5612]
    Epoch 8/10: 100%|█| 220/220 [00:57<00:00,  3.84Batch/s, loss=1.4365, PPL=4.2059, val_loss=3.5257, val_PPL=33.9764]
    Epoch 9/10: 100%|█| 220/220 [00:50<00:00,  4.36Batch/s, loss=1.3514, PPL=3.8629, val_loss=3.5341, val_PPL=34.2640]
    Epoch 10/10: 100%|█| 220/220 [00:49<00:00,  4.45Batch/s, loss=1.2740, PPL=3.5752, val_loss=3.5696, val_PPL=35.5009]
    测试损失:3.2864
+++++++++++++++++++++++++++++++++++
"""

# 导入基础包
import os
import json
import random

# 导入依赖包
import opencc
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

# 导入深度学习相关包
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, BatchSampler
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data import get_tokenizer

"""===============
@@@ Settings
==============="""

# 全局配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 运算设备
SEED = 666  # 随机数种子

# 训练参数
EPOCHS = 10  # 训练周期
SAVE_EACH_EPOCH = 2  # 保存权重周期
BATCH_SIZE = 128  # 批训练大小
TRAIN_SIZE = 0.8  # 训练集比例
VAL_SIZE = 1000  # 验证数据量
TEST_SIZE = 100  # 测试数据量

# 模型参数
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

# 数据参数
PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3  # 特殊 Token
SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]  # 特殊符号定义

# 随机模块设置随机数种子
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

"""===============
@@@ 数据集构建
==============="""


class TranslateDataset(Dataset):
    """机器翻译数据集"""

    def __init__(
        self,
        file_path: str = "./data/cmn.txt",
    ):
        """
        构造机器翻译数据集

        Args:
            file_path (str, optional): \t 间隔符数据文本文件路径. Defaults to "./data/cmn.txt".
        """
        super().__init__()
        self.src_sentences = []
        self.tgt_sentences = []

        cc = opencc.OpenCC("t2s")  # 初始化 繁体转简体
        data = pd.read_table(file_path, header=None)  # 读取数据
        self.length = len(data)

        # 通过 spacy 获取语言分词模型
        self.src_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
        self.tgt_tokenizer = get_tokenizer("spacy", language="zh_core_web_sm")

        # 迭代原始数据，进行分词处理
        for src, tgt in data[[0, 1]].values:
            self.src_sentences.append(self.src_tokenizer(src))
            self.tgt_sentences.append(self.tgt_tokenizer(cc.convert(tgt)))

        # 构建 vocab 对象
        self.src_vocab = build_vocab_from_iterator(self.src_sentences, 1, SPECIALS)
        self.src_vocab.set_default_index(UNK_IDX)

        self.tgt_vocab = build_vocab_from_iterator(self.tgt_sentences, 1, SPECIALS)
        self.tgt_vocab.set_default_index(UNK_IDX)

    def __len__(self):
        """数据集长度"""
        return self.length

    def __repr__(self):
        """字符串可视化显示数据集信息"""
        return (
            " Dataset Info ".center(50, "=")
            + "\n"
            + "| %-21s | %-22s |\n" % ("size", self.length)
            + "| %-21s | %-22s |\n" % ("src vocab: en", len(self.src_vocab))
            + "| %-21s | %-22s |\n" % ("tgt vocab: zh", len(self.tgt_vocab))
            + "=" * 50
        )

    def __getitem__(self, idx):
        """根据idx获取单条数据"""
        src = [BOS_IDX] + self.src_vocab.lookup_indices(self.src_sentences[idx]) + [EOS_IDX]
        tgt = [BOS_IDX] + self.tgt_vocab.lookup_indices(self.tgt_sentences[idx]) + [EOS_IDX]

        return torch.tensor(src), torch.tensor(tgt)


def collate_fn(batch):
    """
    DataLoader 批获取数据时进行的处理函数

    Args:
        batch (iterator): iter 批数据 (src_token(S_src), tgt_token(S_tgt))

    Returns:
        tuple: Tensor(S_src, N), Tensor(S_tgt, N)
    """
    src_batch, tgt_batch = [], []
    for src_token, tgt_token in batch:
        src_batch.append(src_token)
        tgt_batch.append(tgt_token)
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)

    return src_batch, tgt_batch


"""===================
@@@ 模型建立
==================="""


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        src = [src_len, batch_size]
        """
        src = src.transpose(0, 1)  # src = [batch_size, src_len]
        embedded = self.dropout(self.embedding(src)).transpose(0, 1)  # embedded = [src_len, batch_size, emb_dim]

        # enc_output = [src_len, batch_size, hid_dim * num_directions]
        # enc_hidden = [n_layers * num_directions, batch_size, hid_dim]
        enc_output, enc_hidden = self.rnn(embedded)  # if h_0 is not give, it will be set 0 acquiescently

        # enc_hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # enc_output are always from the last layer

        # enc_hidden [-2, :, : ] is the last of the forwards RNN
        # enc_hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        # encoder RNNs fed through a linear layer
        # s = [batch_size, dec_hid_dim]
        s = torch.tanh(self.fc(torch.cat((enc_hidden[-2, :, :], enc_hidden[-1, :, :]), dim=1)))

        return enc_output, s


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, s, enc_output):
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]

        # batch_size = enc_output.shape[1]
        src_len = enc_output.shape[0]

        # repeat decoder hidden state src_len times
        # s = [batch_size, src_len, dec_hid_dim]
        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        s = s.unsqueeze(1).repeat(1, src_len, 1)
        enc_output = enc_output.transpose(0, 1)

        # energy = [batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim=2)))

        # attention = [batch_size, src_len]
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input, s, enc_output):
        # dec_input = [batch_size]
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]

        dec_input = dec_input.unsqueeze(1)  # dec_input = [batch_size, 1]

        embedded = self.dropout(self.embedding(dec_input)).transpose(0, 1)  # embedded = [1, batch_size, emb_dim]

        # a = [batch_size, 1, src_len]
        a = self.attention(s, enc_output).unsqueeze(1)

        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        enc_output = enc_output.transpose(0, 1)

        # c = [1, batch_size, enc_hid_dim * 2]
        c = torch.bmm(a, enc_output).transpose(0, 1)

        # rnn_input = [1, batch_size, (enc_hid_dim * 2) + emb_dim]
        rnn_input = torch.cat((embedded, c), dim=2)

        # dec_output = [src_len(=1), batch_size, dec_hid_dim]
        # dec_hidden = [n_layers * num_directions, batch_size, dec_hid_dim]
        dec_output, dec_hidden = self.rnn(rnn_input, s.unsqueeze(0))

        # embedded = [batch_size, emb_dim]
        # dec_output = [batch_size, dec_hid_dim]
        # c = [batch_size, enc_hid_dim * 2]
        embedded = embedded.squeeze(0)
        dec_output = dec_output.squeeze(0)
        c = c.squeeze(0)

        # pred = [batch_size, output_dim]
        pred = self.fc_out(torch.cat((dec_output, c, embedded), dim=1))

        return pred, dec_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, DEVICE):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.DEVICE = DEVICE

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]
        # teacher_forcing_ratio is probability to use teacher forcing

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.DEVICE)

        # enc_output is all hidden states of the input sequence, back and forwards
        # s is the final forward and backward hidden states, passed through a linear layer
        enc_output, s = self.encoder(src)

        # first input to the decoder is the <SOS> tokens
        dec_input = trg[0, :]

        for t in range(1, trg_len):
            # insert dec_input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            dec_output, s = self.decoder(dec_input, s, enc_output)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = dec_output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = dec_output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            dec_input = trg[t] if teacher_force else top1

        return outputs


"""==========================
@@@ 模型训练、验证执行函数
=========================="""


def process(model, iterator, criterion, optimizer=None, pbar=None):
    """模型训练/验证推理函数（单个epoch）

    传入 optimizer 时则训练模型，否则验证模型

    Args:
        model (nn.Module): 模型对象
        iterator (DataLoader): 数据迭代器
        criterion (nn.modules.loss): 损失函数对象
        optimizer (torch.optim, optional): 优化器对象. Defaults to None.
        pbar (tqdm, optional): 进度条对象. Defaults to None.

    Returns:
        float: 训练/验证在 iterator 条件下的平均损失
    """
    if optimizer is None:
        model.eval()
    else:
        model.train()
    epoch_loss = 0
    for i, (src, trg) in enumerate(iterator):
        # 转移张量至运算设备
        src = src.to(DEVICE)
        trg = trg.to(DEVICE)

        # 模型推理
        pred = model(src, trg)

        # 修改预测张量形状
        pred_dim = pred.shape[-1]
        trg = trg[1:].reshape(-1)
        pred = pred[1:].view(-1, pred_dim)
        # 计算损失
        loss = criterion(pred, trg)
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 更新进度条
            pbar.set_postfix({"loss": f"{epoch_loss / (i + 1) :.4f}", "PPL": f"{np.exp(epoch_loss / (i + 1)) :.4f}"})
            pbar.update(1)

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


"""=========================
@@@ 模型测试和句子翻译
========================="""


def translation(model, iterator, dataset, ckpt=False):
    """词向量翻译成文本函数

    Args:
        model (nn.Module): 模型对象
        iterator (DataLoader): 数据迭代器
        dataset (Dataset): 数据集对象
        ckpt (Bool, optional): 是否使用训练最后的权重进行验证，否则使用最佳权重. Defaults to False.
    """
    if ckpt:
        model.load_state_dict(torch.load("ckpt/lastest_ckpt.pt"))
    else:
        model.load_state_dict(torch.load("ckpt/best_ckpt.pt"))
    # 文本翻译测试
    model.eval()
    for src, tgt in iterator:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        pred = model(src, tgt, 0).argmax(-1)
        for i in range(src.shape[1]):
            print()
            src_doc = " ".join(dataset.src_vocab.lookup_tokens(src[:, i].flatten().tolist()))
            trg_doc = " ".join(dataset.tgt_vocab.lookup_tokens(tgt[:, i].flatten().tolist()))
            pred_doc = " ".join(dataset.tgt_vocab.lookup_tokens(pred[:, i].flatten().tolist()))

            print("input:", src_doc)
            print("target:", trg_doc)
            print("pred:", pred_doc)
            print()


"""============
@@@ 主函数
============"""


def main():
    if not os.path.exists("ckpt"):
        os.mkdir("ckpt")
    # 构造数据集对象（数据预处理）
    dataset = TranslateDataset()
    print(dataset)

    # 划分训练、验证、测试集，分批
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    train_sampler = BatchSampler(indices[: -(VAL_SIZE + TEST_SIZE)], BATCH_SIZE, False)
    val_sampler = BatchSampler(
        indices[-(VAL_SIZE + TEST_SIZE) : -TEST_SIZE],
        BATCH_SIZE,
        False,
    )
    test_sampler = BatchSampler(indices[-TEST_SIZE:], batch_size=1, drop_last=False)

    # 生成数据迭代器
    train_db = DataLoader(
        dataset,
        batch_sampler=train_sampler,
        num_workers=0,
        collate_fn=collate_fn,
    )
    valid_db = DataLoader(
        dataset,
        batch_sampler=val_sampler,
        num_workers=0,
        collate_fn=collate_fn,
    )
    test_db = DataLoader(
        dataset,
        batch_sampler=test_sampler,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # 初始化模型
    INPUT_DIM = len(dataset.src_vocab)
    OUTPUT_DIM = len(dataset.tgt_vocab)

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    # 初始化损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 模型训练
    best_valid_loss = float("inf")
    LOSS = {"train": [], "vaild": []}
    for epoch in range(EPOCHS):
        with tqdm(train_db, desc=f"Epoch {epoch + 1}/{EPOCHS}", unit="Batch") as pbar:
            train_loss = process(model, train_db, criterion, optimizer, pbar)
            valid_loss = process(model, valid_db, criterion, pbar=pbar)
            pbar.set_postfix(
                {
                    "loss": f"{train_loss :.4f}",
                    "PPL": f"{np.exp(train_loss) :.4f}",
                    "val_loss": f"{valid_loss :.4f}",
                    "val_PPL": f"{np.exp(valid_loss) :.4f}",
                }
            )
            LOSS["train"].append(train_loss)
            LOSS["vaild"].append(valid_loss)
        if not (epoch + 1) % SAVE_EACH_EPOCH:
            torch.save(model.state_dict(), f"ckpt/epoch_{epoch + 1}_ckpt.pt")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "ckpt/best_ckpt.pt")

    # 保存最终模型权重
    torch.save(model.state_dict(), "ckpt/lastest_ckpt.pt")
    # 保存期间模型损失
    with open("ckpt/metrics.json", "w+") as fp:
        json.dump(LOSS, fp)

    # 使用测试数据对模型进行测试
    test_loss = process(model, test_db, criterion)
    print(f"测试损失:{test_loss :.4f}")
    # 将测试数据集进行逐一翻译
    translation(model, test_db, dataset)

    # 对训练过程损失进行可视化
    matplotlib.rcParams["font.family"] = ["Times New Roman"]
    plt.figure(figsize=(6, 4))
    plt.plot(LOSS["train"], "b-", label="Train")
    plt.plot(LOSS["vaild"], "r-", label="Valid")
    plt.legend(loc="upper right")
    plt.title("Train History")
    plt.xlabel("Epoch")
    plt.xticks([1, 2, 3, 4, 5])
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"ckpt/Epoch-{EPOCHS}.png")


if __name__ == "__main__":
    main()
