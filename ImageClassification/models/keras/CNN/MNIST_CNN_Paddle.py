# -*- encoding: utf-8 -*-",
"""
+++++++++++++++++++++++++++++++++++
@ File        : MNIST_CNN_Paddle.py
@ Time        : 2023/3/7 10:23
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (Conda)
+++++++++++++++++++++++++++++++++++

+++++++++++++++++++++++++++++++++++
"""

# 导入基本库
from pathlib import Path
from PIL import Image
import warnings
import tqdm
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入Paddle相关库
from paddle.vision import RandomRotation, RandomHorizontalFlip, CenterCrop, Normalize
import paddle
import paddle.nn as nn

warnings.filterwarnings("ignore")
"""
++++++++++++++++++++
@@@ 自定义数据集类
++++++++++++++++++++
"""


class MyDataset(paddle.io.Dataset):
    def __init__(self, data_dir, transform=None, ext=".png"):
        # 初始化数据集，加载图片路径和标签
        super(MyDataset, self).__init__()
        self.label_names = list(map(lambda i: int(i), os.listdir(data_dir)))
        self.data_list = []
        for file_path in Path(data_dir).glob(f"*/*{ext}"):
            label = int(os.path.split(file_path.parent)[1])
            self.data_list.append([file_path, label])

        self.transform = transform

    def __getitem__(self, index):
        # 根据索引读取图像
        image_path, label = self.data_list[index]
        # PIL.Image 读取图片方式
        image = Image.open(image_path)
        image = np.asarray(image, dtype="float32")
        # OpenCV 读取图片方式
        # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype("float32")
        if self.transform:
            image = self.transform(image)
        label = int(label)
        return image, label

    def __len__(self):
        return len(self.data_list)


"""
+++++++++++++++++++++++++++
@@@ 路径、数据增广、数据集设置
+++++++++++++++++++++++++++
"""

# 批次数量
bs = 64
# 读取数据进程数量
# 注意！Windows和MacOS不支持多进程读取
num_workers = 4
# 数据集存放路径，根据 Image-processing_v2.py生成的数字为目录的图像文件夹
data_dir = r"..\data\MNIST"

# 制作数据增广组合
# transform = paddle.vision.transforms.Compose([
#     RandomRotation(30),
#     RandomHorizontalFlip(1),
#     CenterCrop(1000)
# ])

transform = Normalize(mean=[127.5], std=[127.5], data_format="HWC")
# 根据路径形成训练集和测试集
train_mnist = MyDataset(f"{data_dir}/train", transform=transform)
test_mnist = MyDataset(f"{data_dir}/test", transform=transform)

# 生成训练集数据迭代器
train_loader = paddle.io.DataLoader(train_mnist, batch_size=bs, shuffle=True, num_workers=num_workers, drop_last=True)
# 生成测试集数据迭代器
test_loader = paddle.io.DataLoader(test_mnist, batch_size=bs, shuffle=True)
# 自定义批采样器
# bs = paddle.io.BatchSampler(train_mnist, batch_size=batch_size, shuffle=shuffle, drop_last=True)
# train_loader_self = paddle.io.DataLoader(train_mnist, batch_sampler=bs, num_workers=num_workers)


"""
++++++++++++++++++++++++
@@@ 模型搭建
++++++++++++++++++++++++
"""


class MyNet(paddle.nn.Layer):
    def __init__(self, num_classes=10):
        super(MyNet, self).__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2D(1, 6, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(2, 2),
            nn.Conv2D(6, 16, 5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2D(2, 2),
        )

        if self.num_classes > 0:
            self.head = paddle.nn.Sequential(nn.Linear(400, 120), nn.Linear(120, 84), nn.Linear(84, self.num_classes))

    def forward(self, inputs):
        x = self.features(inputs)

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.head(x)

        return x


# nn.Sequential()方法
model = nn.Sequential(
    nn.Conv2D(1, 6, 3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2D(2, 2),
    nn.Conv2D(6, 16, 5, stride=1, padding=0),
    nn.ReLU(),
    nn.MaxPool2D(2, 2),
    nn.Flatten(),
    nn.Linear(400, 120),
    nn.Linear(120, 84),
    nn.Linear(84, 10),
)

"""
++++++++++++++++++++
@@@ 模型训练和权重保存
++++++++++++++++++++
"""

# 必须在这个模式下运行脚本，否则多线程读取数据
# 若非__main__模式，请将上述代码的num_workers调整为0
# 注意！Windows和MacOS不支持多进程读取
if __name__ == "__main__":
    # 训练轮数
    epochs = 10
    # 间隔若干个周期进行验证和模型权重保存
    snapshot_epoch = 2
    # 模型权重保存路径
    save_dir = Path("output")
    save_dir.mkdir(exist_ok=True)
    # 初始化模型、优化器、损失函数
    model = MyNet(num_classes=10)
    optim = paddle.optimizer.Adam(parameters=model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    train_history = {"loss": [], "acc": [], "loss_test": [], "acc_test": []}
    for epoch in range(1, epochs + 1):
        train_history["loss"].append(list())
        train_history["acc"].append(list())
        with tqdm.tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{epochs}") as pbar:
            for step_id, (x, y) in enumerate(train_loader):
                y = paddle.unsqueeze(paddle.cast(y, dtype="int64"), axis=1)
                # 标记模型训练状态，参数开始变化
                model.train()
                predicts = model(x)
                loss = loss_fn(predicts, y)
                acc = paddle.metric.accuracy(predicts, y)
                loss.backward()  # 反向传播
                # 优化器迭代更新模型权重
                optim.step()
                optim.clear_grad()
                # 记录准确率和损失
                train_history["loss"][-1].append(loss.item())
                train_history["acc"][-1].append(acc.item())
                # 进度条更新
                pbar.update(1)
                pbar.set_postfix({"acc": f"{acc.item():.4f}", "loss": f"{loss.item():.4f}"})

            # 论次数达到指定次数时进行验证
            if (epoch % snapshot_epoch) == 0:
                train_history["loss_test"].append(list())
                train_history["acc_test"].append(list())
                # 计算过去训练历史中测试精度
                acc_max = max(train_history["acc_test"], key=lambda i: np.mean(i)) if train_history["acc_test"] else 0
                # 标记模型验证状态，参数锁定不变化
                model.eval()
                with tqdm.tqdm(total=len(test_loader), desc="Evaluating...") as pbar1:
                    for step_id, (x, y) in enumerate(test_loader):
                        y = paddle.unsqueeze(paddle.cast(y, dtype="int64"), axis=1)
                        predicts = model(x)
                        loss = loss_fn(predicts, y)
                        acc = paddle.metric.accuracy(predicts, y)
                        train_history["loss_test"][-1].append(loss.item())
                        train_history["acc_test"][-1].append(acc.item())
                        pbar1.update(1)
                        pbar1.set_postfix(
                            {
                                "acc_test": f"{acc.item():.4f}",
                                "loss_test": f"{loss.item():.4f}",
                            }
                        )

                    if np.mean(train_history["acc_test"][-1]) > np.mean(acc_max):
                        paddle.save(model.state_dict(), str(save_dir / "best_model.pdparams"))
                        paddle.save(optim.state_dict(), str(save_dir / "best_model.pdopt"))

                    # 保存权重和优化器参数
                    paddle.save(model.state_dict(), str(save_dir / f"{epoch}.pdparams"))  # 动态图参数
                    paddle.save(optim.state_dict(), str(save_dir / f"{epoch}.pdopt"))

    # 保存模型最后权重
    paddle.save(model.state_dict(), str(save_dir / "final_model.pdparams"))
    paddle.save(optim.state_dict(), str(save_dir / "final_model.pdopt"))

    (save_dir / "loss").mkdir(exist_ok=True)
    (save_dir / "acc").mkdir(exist_ok=True)
    # 可视化训练过程的损失、准确率变化
    for epoch_id in range(epochs):
        # 绘制损失
        plt.figure(figsize=(20, 5))
        plt.plot(train_history["loss"][epoch_id], "o-")
        plt.title(f"Epoch:{epoch_id + 1} Loss")
        plt.savefig(save_dir / f"loss/Epoch{epoch_id + 1}.jpg")

        # 绘制准确率
        plt.figure(figsize=(20, 5))
        plt.plot(train_history["acc"][epoch_id], "o-")
        plt.title(f"Epoch:{epoch_id + 1} Acc.")
        plt.savefig(save_dir / f"acc/Epoch{epoch_id + 1}.jpg")

    for key in ["loss", "loss_test", "acc", "acc_test"]:
        table = pd.DataFrame(train_history[key])
        table.index.name = "Epoch"
        table.insert(0, "mean", table.mean(axis=1))
        table.index = table.index + 1
        if "test" in key:
            table.index = table.index * snapshot_epoch
        table.to_csv(save_dir / f"Train_{key}.csv")

"""
++++++++++++++++
@@@ 模型验证和评估
++++++++++++++++
"""

save_dir = Path("output")
# 初始化模型，并设置为eval模式
model = MyNet(10)
model.eval()
# 加载模型权重
model_params = paddle.load(str(save_dir / "best_model.pdparams"))
model.set_state_dict(model_params)
# 获取图像矩阵和标签，预测标签
img, label = test_mnist[0]

pred = model(paddle.Tensor(img).unsqueeze(1))[0].argmax()
print(f"raw label: {label}, pred label: {pred.item()}")
