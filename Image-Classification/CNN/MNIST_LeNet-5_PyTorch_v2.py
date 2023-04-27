# -*- coding: utf-8 -*-
"""
+++++++++++++++++++++++++++++++++++
@ File        : MNIST_LeNet-5_PyTorch_v2.py
@ Time        : 2022/8/10 9:36
@ Author      : Mirrich Wang
@ Version     : Python 3.8.8 (Anaconda3)
+++++++++++++++++++++++++++++++++++
PyTorch 框架的 CNN 使用 MNIST 手写数字进行训练
重构了代码结构：
1): 在if __name__ == "__main__":下进行一系列操作，能够使用num_workers加快数据读取
2): 将数据加载、模型构建、模型训练、验证使用了类进行封装
+++++++++++++++++++++++++++++++++++
"""

# 导入基础模块
import os
import time
import numpy as np

# 导入 torch 相关
import torch
import torch.nn as nn
import torchsummary
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import ImageFolder, MNIST

# 导入可视化模块
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条

'''
====================
@@@ 数据加载
====================
'''


def load_data(root, batch_size=4, num_workers=0, valid_split=.2, method="image"):
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为张量
        # transforms.Grayscale(num_output_channels=num_channels),  # 转换成灰度单通道图像
    ])

    if method == "image":
        # 采取 ImageFolder 方式读取数据集（该方法可能较慢）
        train_data = ImageFolder(root=os.path.join(root, 'train'), transform=transform)
        test_data = ImageFolder(root=os.path.join(root, 'test'), transform=transform)
    elif method == "ubyte":
        root = root.split("MNIST")[0]
        # 采取Torchvision.datasets.MNIST的方式加载数据集
        train_data = MNIST(root=root, train=True, transform=transform)
        test_data = MNIST(root=root, train=False, transform=transform)
    else:
        raise f"No {method} method."

    if valid_split:
        # 训练数据根据 valid_split 划分训练、校验集
        valid_size = int(len(train_data) * valid_split)
        indices = np.arange(len(train_data))
        np.random.shuffle(indices)
        indices_train = indices[:-valid_size].tolist()
        indices_valid = indices[-valid_size:].tolist()

        # 构造数据迭代器
        train_db = data.DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=indices_train
        )
        valid_db = data.DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=indices_valid
        )
        test_db = data.DataLoader(
            dataset=test_data,
            num_workers=num_workers,
            batch_size=batch_size
        )

        return train_db, valid_db, test_db
    else:
        train_db = data.DataLoader(dataset=train_data, batch_size=batch_size)
        test_db = data.DataLoader(dataset=test_data, batch_size=batch_size)

        return train_db, test_db


'''
====================
@@@ 模型搭建
====================
'''


class Net(nn.Module):
    """ CNN 卷积网络在 MNIST 28x28 手写数字灰色图像上应用版本 """

    def __init__(self):
        super(Net, self).__init__()
        # 卷积层 #
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,  # 输入通道由 3 调账为 1
                               kernel_size=5, stride=1, padding=2)  # padding 使得模型与原文提供的 32x32 结构保持不变
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
                               kernel_size=5, stride=1)
        # 池化层 #
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层 #
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # 激活函数 #
        self.relu = nn.ReLU()

    def forward(self, x):
        # 卷积层 1 #
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool1(out)
        # 卷积层 2 #
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool2(out)
        # 全连接层 #
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)

        return out


'''
====================
@@@ 模型操作自定义类
====================
'''


class LeNet5(object):
    """ CNN 模型训练、验证相关类 """

    # 默认配置
    cuda = True  # 是否使用 CUDA
    epochs = 20  # 训练世纪
    model_path = f"LeNet-5_{epochs}s.pth"  # 模型权重保存名字

    def __init__(self, model, criterion, optimizer, args):
        for k, v in args.items():
            self.__setattr__(k, v)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        # 根据 CUDA 使用情况，将模型转移至 GPU
        if self.cuda:
            self.model = self.model.cuda()

        if os.path.exists(model_path):
            model_parameters = torch.load(model_path)
            self.model.load_state_dict(model_parameters)
        else:
            print("Model weights is not found and will be retrained.")

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

                    pbar.set_postfix({"loss": "%.4f" % np.mean(loss),
                                      "accuracy": "%.2f%%" % (correct / total * 100)})
                    pbar.update(1)

                LOSS.append(np.mean(loss))
                ACCURACY.append(correct / total)
                if valid_data:
                    val_loss, val_accuracy = self.eval(valid_data)
                    VAL_LOSS.append(val_loss)
                    VAL_ACCURACY.append(val_accuracy)
                    pbar.set_postfix({"loss": "%.4f" % LOSS[-1],
                                      "accuracy": "%.2f%%" % (ACCURACY[-1] * 100),
                                      "val_loss": "%.4f" % VAL_LOSS[-1],
                                      "val_accuracy": "%.2f%%" % (VAL_ACCURACY[-1] * 100)})

        torch.save(self.model.state_dict(), self.model_path)  # 模型权重保存
        print(f"Model weights is saved in {self.model_path}")

        return {"loss": LOSS, "accuracy": ACCURACY, "val_loss": VAL_LOSS, "val_accuracy": VAL_ACCURACY}


if __name__ == "__main__":
    # ----------------- config ------------------- #

    # 绘图参数
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 使用新罗马字体
    # 数据集
    data_path = r".\data\MNIST"
    batch_size = 128  # 批次大小
    num_workers = 4  # 读取图片进程数
    valid_split = .0  # 在训练集上划出验证集的尺寸
    # 图像配置
    image_size = [28, 28]
    num_channels = 1
    # 训练配置
    epochs = 20  # 训练世纪
    cuda = torch.cuda.is_available()  # 是否使用 CUDA
    model_path = rf".\output\MNIST_LeNet5_{epochs}.pth"  # 模型权重保存位置

    # 训练配置封装为字典
    model_config = {
        "epochs": epochs,
        "cuda": cuda,
        "model_path": model_path
    }

    # ---------------- 加载数据和模型 ------------------ #

    # 加载数据
    train_db, test_db = load_data(data_path, batch_size, num_workers, valid_split=valid_split, method="ubyte")
    # 加载模型
    net = Net()
    torchsummary.summary(net, input_size=(num_channels, *image_size), device="cpu")  # 采用 keras 的方式顺序打印模型结构

    # --------------- 模型训练 ----------------- #

    criterion = nn.CrossEntropyLoss()  # 设置损失函数
    optimizer = torch.optim.Adam(net.parameters())  # 配置优化器
    lenet5 = LeNet5(net, criterion, optimizer, model_config)

    st = time.time()
    train_history = lenet5.train(train_db, test_db)
    et = time.time()
    print('Time Taken: %d seconds' % (et - st))  # 1622

    # --------------- 模型验证 ----------------- #

    print("Testing model by Test's data...")
    valid_result = lenet5.eval(test_db)
    print(valid_result)

    # --------------- 训练结果可视化 ----------------- #

    keys = ['loss', 'accuracy']
    for k in keys:
        plt.plot(range(1, model_config["epochs"] + 1), train_history[k], label="Train")
        plt.plot(range(1, model_config["epochs"] + 1), train_history[f'val_{k}'], label="Val")
        plt.legend()
        plt.title(f'MNIST CNN Train & Valid {k.title()}')
        plt.xlabel('Epoch')
        plt.ylabel(k.title())
        plt.grid(True)
        plt.show()
