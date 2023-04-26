# -*- encoding: utf-8 -*-
"""
    @File        : ViT_CIFAR100_PyTorch.py
    @Time        : 2022/9/26 10:56
    @Author      : Mirrich Wang 
    @Version     : Python 3.8.8 (Anaconda)
    @Description : ViT 使用 CIFAR100 数据集进行图像分类案例，本机10个EPOCH，
                   loss=3.6136, accuracy=14.52%, val_loss=3.6165, val_accuracy=14.62%
"""

import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
from vit_pytorch import ViT
from torchsummary import summary
from torchvision import transforms
from torchvision.datasets import ImageFolder, CIFAR100

import matplotlib.pyplot as plt

from tqdm import tqdm


# --------------- 模型操作自定义类 -------------------- #


class VisionTransformer(object):
    _default = {
        "cuda": torch.cuda.is_available(),  # 自动判断是否使用 CUDA

        "epochs": 20,  # 训练世纪

        "save_epochs": 5,  # 每隔若干个周期保存模型权重（未写相关代码）

        "save_dir": "./logs"  # 模型权重保存路径
    }

    def __init__(self, model, criterion, optimizer, **args):
        """
        对模型进行相关操作类
        Args:
            model: 传入 nn.Module 类型的架构
            criterion: 损失函数
            optimizer: 优化器
            **args: 将按照 xx=""来赋予类 attribute，目前更改有效的未 "cuda", "epochs", "save_epochs", "save_dir"
        """
        self.__dict__.update(self._default)
        for k, v in args.items():
            self.__setattr__(k, v)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        if self.cuda:
            self.model.cuda()

    def predict(self, data):
        self.model.eval()
        if self.cuda():
            data = data.cuda()
        output = self.model(data)

    def eval(self, data):
        self.model.eval()
        correct, total = 0, 0
        loss = []
        for x, y in data:
            if self.cuda:
                x, y = x.cuda(), y.cuda()
                output = self.model(x)
                loss.append(self.criterion(output, y).item())
                total += y.shape[0]
                correct += torch.eq(output.argmax(-1), y).sum().item()
        return torch.mean(torch.tensor(loss)).item(), correct / total

    def train(self, data, valid_data=None):
        LOSS, ACCURACY = [], []
        VAL_LOSS, VAL_ACCURACY = [], []
        for epoch in range(self.epochs):
            correct, total = 0, 0
            loss = []
            with tqdm(total=len(data), desc=f"Epoch {epoch + 1}/{self.epochs}") as pbar:
                for x, y in data:
                    if self.cuda:
                        x, y = x.cuda(), y.cuda()
                    self.model.train()
                    output = self.model(x)
                    loss_step = self.criterion(output, y)
                    loss.append(loss_step.item())

                    self.optimizer.zero_grad()
                    loss_step.backward()
                    self.optimizer.step()

                    total += y.shape[0]
                    correct += torch.eq(output.argmax(-1), y).sum().item()

                    pbar.update(1)
                    pbar.set_postfix({"loss": "%.4f" % loss[-1],
                                      "accuracy": "%.2f%%" % (correct / total * 100)})

                LOSS.append(torch.mean(torch.tensor(loss)).item())
                ACCURACY.append(correct / total)
                if valid_data:
                    val_loss, val_accuracy = self.eval(valid_data)
                    VAL_LOSS.append(val_loss)
                    VAL_ACCURACY.append(val_accuracy)
                    pbar.set_postfix({"loss": "%.4f" % LOSS[-1],
                                      "accuracy": "%.2f%%" % (ACCURACY[-1] * 100),
                                      "val_loss": "%.4f" % VAL_LOSS[-1],
                                      "val_accuracy": "%.2f%%" % (VAL_ACCURACY[-1] * 100)})

        torch.save(self.model.state_dict(), os.path.join(self.save_dir, "CIFAR100_ViT_weights.pth"))  # 模型权重保存

        return {"loss": LOSS, "accuracy": ACCURACY, "val_loss": VAL_LOSS, "val_accuracy": VAL_ACCURACY}


if __name__ == "__main__":
    # ----------------- config ------------------- #

    data_dir = "../../datasets/CIFAR100"  # 数据集根目录

    save_dir = "../../logs"  # 权重保存路径

    image_size = (32, 32)  # 图像大小

    num_channels = 3  # 图片通道数 RGB

    num_classes = 100  # 类别数量

    num_workers = 0  # DataLoader 进程数

    valid_split = 0.1  # 校验集占训练集的比例

    batch_size = 32  # 批大小

    epochs = 10  # 训练世纪

    cuda = True  # 是否使用 CUDA

    # ---------------- 加载数据 ------------------ #

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(image_size)]
    )
    # train_data = ImageFolder(root=os.path.join(path, 'train'), transform=transform)
    # test_data = ImageFolder(root=os.path.join(path, 'test'), transform=transform)
    # 构造数据集，需要注意在 data_dir 下若使用 CIFAR100数据集，请放置 “cifar-100-python.tar.gz" 和 "cifar-100-python" 两个文件
    train_data = CIFAR100(root=data_dir + "/raw", transform=transform)
    test_data = CIFAR100(root=data_dir + "/raw", transform=transform, train=False)

    # 训练 data 划分成 train 和 validation
    valid_size = int(len(train_data) * valid_split)
    indices = np.arange(len(train_data))
    np.random.shuffle(indices)
    indices_train = indices[:-valid_size].tolist()
    indices_valid = indices[-valid_size:].tolist()
    # 构造迭代器
    train_db = data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        sampler=indices_train,
        num_workers=num_workers,
    )
    valid_db = data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        sampler=indices_valid,
        num_workers=num_workers,
    )
    test_db = data.DataLoader(
        dataset=test_data, batch_size=batch_size, num_workers=num_workers
    )
    print(f"Train: (%i, %i, %i, %i)" % (len(train_data), num_channels, *image_size))
    print(f"Valid: (%i, %i, %i, %i)" % (len(indices_valid), num_channels, *image_size))
    print(f"Test: (%i, %i, %i, %i)" % (len(test_data), num_channels, *image_size))

    # ---------------- 构建模型 ------------------ #

    vit = ViT(
        image_size=image_size[0],
        patch_size=4,
        num_classes=100,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
    )
    summary(vit, input_size=(num_channels, *image_size), device="cpu")

    # --------------- 模型训练 ----------------- #
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(vit.parameters())
    model = VisionTransformer(vit, criterion, optimizer, epochs=epochs)

    st = time.time()
    train_history = model.train(train_db, valid_db)
    et = time.time()
    print("Time Taken: %.4f" % (et - st))

    # -------------- 模型测试 ------------------ #

    val_loss, val_acc = model.eval(test_db)
    print("Valid Loss: %.4f, Accuracy: %.2f%%" % (val_loss, val_acc * 100))

    # -------------- 训练过程可视化 ------------- #

    plt.plot(train_history["loss"], "-o", label="Train")
    plt.plot(train_history["val_loss"], "-o", label="Valid")
    plt.title("ViT Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.plot(train_history["accuracy"], "-o", label="Train")
    plt.plot(train_history["val_accuracy"], "-o", label="Valid")
    plt.title("ViT Training Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
