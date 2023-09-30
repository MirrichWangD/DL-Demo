# -*- coding: UTF-8 -*-
"""
+++++++++++++++++++++++++++++++++++
@ File        : MNIST_LeNet-5_PyTorch.py
@ Time        : 2023/4/25 11:25
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (Conda)
+++++++++++++++++++++++++++++++++++
PyTorch 框架的 CNN 使用 MNIST 手写数字进行训练
+++++++++++++++++++++++++++++++++++
"""

# 导入基本模块
import os
import json
import time

# 导入依赖模块
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

# 导入torch相关模块
import torch
import torch.nn as nn
import torchsummary
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import MNIST


"""++++++++++++++++++++++
@@@ Settings
+++++++++++++++++++++++++"""

# 基础配置
data_path = "./data"  # 数据集路径
save_dir = "./output/MNIST_LeNet-5"  # 模型权重保存路径
num_channels = 1  # 图像通道数
image_size = 28  # 图像尺寸
num_workers = 0  # 读取图片进程数
valid_split = 0.2  # 在训练集上划出验证集的尺寸0
# 训练配置
epochs = 20  # 周期
batch_size = 512  # 批次大小
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用设备
mpl.rcParams["font.sans-serif"] = ["Times New Roman"]  # 使用新罗马字体

"""++++++++++++++++++++++
@@@ 数据预处理
++++++++++++++++++++++"""

transform = transforms.Compose(
    [
        transforms.ToTensor(),  # 转换为张量
        # transforms.Grayscale(num_output_channels=num_channels),  # 转换成灰度单通道图像 （使用ImageFolder请取消注释）
    ]
)
# 读取图片为数据集
train_data = MNIST(root=data_path, train=True, transform=transform, download=True)
test_data = MNIST(root=data_path, train=False, transform=transform, download=True)
# 训练 data 划分成 train 和 validation
valid_size = int(len(train_data) * valid_split)
indices = np.arange(len(train_data))
np.random.shuffle(indices)
# 构造迭代器
train_db = data.DataLoader(dataset=train_data, batch_size=batch_size, sampler=indices[:-valid_size])
val_db = data.DataLoader(dataset=train_data, batch_size=batch_size, sampler=indices[-valid_size:])
test_db = data.DataLoader(dataset=test_data, batch_size=batch_size)
print("Train: (%i, %i, %i, %i)" % (len(indices[:-valid_size]), num_channels, image_size, image_size))
print("Valid: (%i, %i, %i, %i)" % (len(indices[-valid_size:]), num_channels, image_size, image_size))
print("Test: (%i, %i, %i, %i)" % (len(test_data), num_channels, image_size, image_size))


"""++++++++++++++++++
@@@ 模型搭建
++++++++++++++++++"""


class Net(nn.Module):
    """CNN 卷积网络在 MNIST 28x28 手写数字上应用的修改版"""

    def __init__(self):
        super(Net, self).__init__()
        # 卷积层 #
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2
        )  # padding 使得模型与原文提供的 32x32 结构保持不变
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
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


net = Net()  # 初始化模型并加载到GPU
net.to(device)
torchsummary.summary(net, input_size=(num_channels, image_size, image_size))  # 采用 keras 的方式顺序打印模型结构

"""+++++++++++++++++++++++
@@@ 模型训练和验证
+++++++++++++++++++++++"""

criterion = nn.CrossEntropyLoss()  # 设置损失函数
optimizer = torch.optim.Adam(net.parameters())  # 配置优化器

# 初始化损失和准确率
train_history = {
    "loss": {"train": [[] for _ in range(epochs)], "val": [[] for _ in range(epochs)]},
    "acc": {"train": [[] for _ in range(epochs)], "val": [[] for _ in range(epochs)]},
}

st = time.time()
for epoch in range(epochs):
    with tqdm(total=len(train_db), desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
        for step, (x, y) in enumerate(train_db):
            net.train()  # 标记模型开始训练，此时权重可变
            x, y = x.to(device), y.to(device)  # 转移张量至 GPU
            output = net(x)  # 将 x 送进模型进行推导

            # 计算损失
            loss = criterion(output, y)  # 计算交叉熵损失
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 一步随机梯度下降算法

            # 计算准确率
            prediction = torch.softmax(output, dim=1).argmax(dim=1)  # 将预测值转换成标签

            # 记录损失和准确率
            train_history["loss"]["train"][epoch].append(loss.item())
            train_history["acc"]["train"][epoch].append(((prediction == y).sum() / y.shape[0]).item())

            # 进度条状态更新
            pbar.update(1)
            pbar.set_postfix(
                {
                    "loss": "%.4f" % np.mean(train_history["loss"]["train"][epoch]),
                    "acc": "%.2f%%" % (np.mean(train_history["acc"]["train"][epoch]) * 100),
                }
            )

        # 每一个 epoch 训练结束后进行验证
        net.eval()  # 标记模型开始验证，此时权重不可变
        for x, y in val_db:
            x, y = x.to(device), y.to(device)
            output = net(x)
            loss_val = criterion(output, y).item() / len(val_db)

            prediction = torch.softmax(output, dim=1).argmax(dim=-1)
            # 记录验证损失和准确率
            train_history["loss"]["val"][epoch].append(loss_val)
            train_history["acc"]["val"][epoch].append(((prediction == y).sum() / y.shape[0]).item())

            # 更新进度条
            pbar.set_postfix(
                {
                    "loss": "%.4f" % np.mean(train_history["loss"]["train"][epoch]),
                    "acc": "%.2f%%" % (np.mean(train_history["acc"]["train"][epoch]) * 100),
                    "val_loss": "%.4f" % np.mean(train_history["loss"]["val"][epoch]),
                    "val_acc": "%.2f%%" % (np.mean(train_history["acc"]["val"][epoch]) * 100),
                }
            )

et = time.time()
time.sleep(0.1)
print("Time Taken: %d seconds" % (et - st))  # 69

"""+++++++++++++++++++
@@@ 模型测试
+++++++++++++++++++"""

print("Test data in model...")
correct, total, loss = 0, 0, 0
per_time = []  # 计算每个
net.eval()
with tqdm(total=len(test_db)) as pbar:
    for step, (x, y) in enumerate(test_db):
        x, y = x.to(device), y.to(device)
        st = time.perf_counter()
        output = net(x)
        torch.cuda.synchronize()
        et = time.perf_counter()
        per_time.append(et - st)
        loss += float(criterion(output, y)) / len(test_db)
        prediction = torch.softmax(output, dim=1).argmax(dim=1)
        correct += int((prediction == y).sum())
        total += y.shape[0]

        pbar.update(1)
        pbar.set_postfix(
            {
                "loss": "%.4f" % loss,
                "accuracy": "%.2f%%" % (correct / total * 100),
                "per_time": "%.4fs" % (et - st),
            }
        )

print("Time Per-Image Taken: %f seconds" % np.mean(per_time))  # 0.000984
print("FPS: %f" % (1.0 / (np.sum(per_time) / len(per_time))))  # 1015.759509

"""+++++++++++++++++++
@@@ 模型权重保存和加载
+++++++++++++++++++"""

# 创建输出目录
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# 保存模型权重
torch.save(net.state_dict(), os.path.join(save_dir, "final_model.pth"))
# 保存训练过程记录
with open(os.path.join(save_dir, "metrics.json"), "w+", encoding="utf-8") as fp:
    json.dump(train_history, fp)
# 读取权重
net.load_state_dict(torch.load(os.path.join(save_dir, "final_model.pth")))

"""+++++++++++++++++++
@@@ 训练和测试结果可视化
+++++++++++++++++++"""


def plot_train_history(history, num_epoch=epochs):
    """
    对训练结果的可视化
    Args:
        history (dict): 训练结果字典（包含 loss 和 accuracy 键）
        num_epoch (int): 展示周期数量（默认为 epochs）

    Returns:

    """
    keys = ["loss", "acc"]
    for k in keys:
        plt.plot(range(1, num_epoch + 1), np.mean(history[k]["train"][: num_epoch + 1], -1))
        plt.plot(range(1, num_epoch + 1), np.mean(history[k]["val"][: num_epoch + 1], -1))
        plt.legend(labels=["Train", "Val"])
        plt.title(f"MNIST LeNet-5 Train & Valid {k.title()}")
        plt.xlabel("Epoch")
        plt.ylabel(k.title())
        plt.grid(True)
        plt.show()


plot_train_history(train_history, epochs)
