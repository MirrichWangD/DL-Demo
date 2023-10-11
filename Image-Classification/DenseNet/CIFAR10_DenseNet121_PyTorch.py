# -*- coding: utf-8 -*-
"""
+++++++++++++++++++++++++++++++++++
@ File        : CIFAR10_DenseNet121_PyTorch.py
@ Time        : 2023/10/07 14:38:17
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (Conda)
+++++++++++++++++++++++++++++++++++
参考：https://zhuanlan.zhihu.com/p/490175600
+++++++++++++++++++++++++++++++++++
"""

import os
import time
import pickle
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import torchsummary

"""==================
@@@ Settings
=================="""

# Global Constant
N_WORKERS = 0
BATCH_SIZE = 256
EPOCHS = 10
VAL_SIZE = 5000
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


"""===========================
@@@ Dataset and Processing
+++++++++++++++++++++++++++"""


def load_meta(file: str):
    """
    _summary_

    Args:
        file (str): _description_
    """
    with open(file, "rb") as fp:
        d = pickle.load(fp, encoding="iso-8859-1")
    return d


class CIFAR10(Dataset):
    name = "CIFAR10"
    base_folder = "cifar-10-batches-py"

    n_channels = 3
    n_labels = 10
    img_size = [32, 32]
    train_size = 50000
    test_size = 10000
    label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    train_list = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    test_list = ["test_batch"]

    def __init__(self, root: str, train: bool = True) -> None:
        super().__init__()

        self.data = []
        self.labels = []

        if train:
            file_list = self.train_list
            shape = [-1, self.n_channels] + self.img_size
        else:
            file_list = self.test_list
            shape = [-1, self.n_channels] + self.img_size

        for file in file_list:
            meta = load_meta(os.path.join(root, self.base_folder, file))
            self.data.extend(meta["data"].reshape(shape))
            self.labels.extend(meta["labels"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> tuple:
        return torch.tensor(self.data[idx]).type(torch.FloatTensor), torch.tensor(self.labels[idx]).type(torch.LongTensor)


"""====================
@@@ Modeling Build
===================="""


class DenseNet121(nn.Module):
    """DenseNet121"""

    def __init__(
        self,
        growth_rate: int = 32,
        n_layers=(6, 12, 24, 16),
        n_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        n_classes: int = 1000,
    ) -> None:
        super(DenseNet121, self).__init__()
        # input
        self.features = nn.Sequential(
            OrderedDict(
                [
                    # DenseNet 原文使用(3, 224, 224)图像，不适合CIFAR10的(3, 32, 32)图像，因此调整该层。
                    ("conv0", nn.Conv2d(3, n_init_features, kernel_size=3, stride=1, padding=1, bias=False))
                    # ("conv0", nn.Conv2d(3, n_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    # ("norm0", nn.BatchNorm2d(n_init_features)),
                    # ("relu0", nn.ReLU(inplace=True)),
                    # ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        # Denseblocks
        self.blocks = nn.ModuleDict()
        n_features = n_init_features
        for i, n_layer in enumerate(n_layers):
            # Each Denseblock
            for j in range(n_layer):
                layer = nn.Sequential(
                    nn.BatchNorm2d(n_features + j * growth_rate),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(n_features + j * growth_rate, bn_size * growth_rate, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(bn_size * growth_rate),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.Dropout(p=drop_rate),
                )
                self.blocks.add_module("denselayer%d_%d" % (i + 1, j + 1), layer)

            n_features = n_features + n_layer * growth_rate
            if i != len(n_layers) - 1:
                transition_layer = nn.Sequential(
                    nn.BatchNorm2d(n_features),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(n_features, n_features // 2, kernel_size=1, stride=1, bias=False),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                )
                self.blocks.add_module("transition%d" % (i + 1), transition_layer)
                n_features //= 2

        # Final Classifier -> Linear Layer
        self.classifier = nn.Linear(n_features, n_classes)

        self.drop_rate = drop_rate

    def forward(self, x: Tensor) -> Tensor:
        features = [self.features(x)]

        # Dense Block forward
        for name, layer in self.blocks.items():
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
            if name.startswith("transition") and not name.endswith("3"):
                features = [new_features]

        out = F.relu(torch.cat(features, 1), inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


"""===================================
@@@ Modeling Training / Evaluating
==================================="""


def process(model, data_iterator, criterion, optimizer=None, pbar=None):
    if optimizer:
        model.train()
    else:
        model.eval()
    loss, acc = [], []
    for step, (x, y) in enumerate(data_iterator):
        x, y = x.to(DEVICE), y.to(DEVICE)
        output = model(x)
        pred = torch.softmax(output, dim=1).argmax(dim=1)

        lossi = criterion(output, y)
        loss.append(lossi.item())
        acc.append(((pred == y).sum() / y.shape[0]).item())
        if optimizer:
            optimizer.zero_grad()
            lossi.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_postfix({"loss": f"{np.mean(loss):.4f}", "acc": f"{np.mean(acc) * 100:.2f}%"})

    return loss, acc


"""=============
@@@ Main
============="""


def main():
    model = DenseNet121(12, (5, 10, 20), n_classes=10, drop_rate=0.1).to(DEVICE)
    torchsummary.summary(model, input_size=(3, 32, 32))

    train_data = CIFAR10(root="./data", train=True)
    test_data = CIFAR10(root="./data", train=False)

    indices = torch.randperm(len(train_data))
    train_db = DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=N_WORKERS, sampler=indices[:-VAL_SIZE])
    valid_db = DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=N_WORKERS, sampler=indices[-VAL_SIZE:])
    test_db = DataLoader(test_data, batch_size=1, num_workers=N_WORKERS)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.3, momentum=0.9, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    st = time.time()
    for epoch in range(EPOCHS):
        with tqdm(total=len(train_db), desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
            train_loss, train_acc = process(model, train_db, criterion, optimizer, pbar=pbar)
            valid_loss, valid_acc = process(model, valid_db, criterion, pbar=pbar)

            pbar.set_postfix(
                {
                    "loss": f"{np.mean(train_loss):.4f}",
                    "acc": f"{np.mean(train_acc) * 100:.2f}%",
                    "val_loss": f"{np.mean(valid_loss):.4f}",
                    "val_acc": f"{np.mean(valid_acc):.2f}%",
                }
            )
    et = time.time()
    print("Time Taken: %fs" % (et - st))


if __name__ == "__main__":
    main()
