# -*- coding: utf-8 -*-
"""
+++++++++++++++++++++++++++++++++++
@ File        : CIFAR10_DenseNet121_PyTorch.py
@ Time        : 2024/01/17 16:02:35
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (Conda)
+++++++++++++++++++++++++++++++++++
参考：
- https://www.bilibili.com/video/BV1yt4y1e7sZ/
- https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/train_multi_GPU
========================================================
启动命令：
- 旧版：python -m torch.distributed.launch --nproc_per_node=2 --use_env CIFAR10_DenseNet121_PyTorch.py
- 新版：torchrun --nproc_per_node=2 CIFAR10_DenseNet121_PyTorch.py
========================================================
模块版本：
    CUDA        11.6
    torch       1.12.1+cu116
    torchvision 0.13.1+cu116
========================================================
实验设备：单机两张 NVIDIA GeForce RTX 3080 10GB
+++++++++++++++++++++++++++++++++++
"""

# 导入基础模块
import os
import sys
import tempfile
import argparse
from collections import OrderedDict

# 导入进度条模块
from tqdm import tqdm

# 导入 torch 相关模块
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入优化器相关模块
import torch.optim as optim

# 导入分布式训练相关模块
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

# 导入数据集相关模块
from torchvision import transforms
from torchvision.datasets import CIFAR10


"""=========================
@@@ 启动参数初始化
========================="""


def init_parse_args():
    """外部标识符参数初始化函数"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", type=str, default="./data")  # 数据集所在目录文件
    parser.add_argument("--download", type=bool, default=True)  # 是否下载数据
    parser.add_argument("--output", type=str, default="./weights/CIFAR10_DenseNet121_PyTorch")  # 文件输出路径

    parser.add_argument("--epochs", type=int, default=30)  # 训练周期
    parser.add_argument("--batch-size", type=int, default=256)  # 训练批次大小

    # 优化器参数
    parser.add_argument("--lr", type=float, default=0.3)  # 基础学习率
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--weights", type=str, default="", help="initial weights path")  # 模型预训练权重
    parser.add_argument("--freeze-layers", type=bool, default=False)  # 是否需要冻结除输出部分 FC 层的参数

    # 分布式配置
    parser.add_argument("--dist-backend", default="nccl", type=str, help="please use 'gloo' on windows platform")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--num-workers", default=4, type=int)  # 每个进程数据迭代器的线程数
    parser.add_argument("--syncBN", type=bool, default=True)  # 是否启用SyncBatchNorm
    # 以下参数不需要更改，会根据 nproc_per_node 自动设置
    parser.add_argument("--device", default="cuda", help="device id (i.e. 0 or 0,1 or cpu)")
    parser.add_argument("--world-size", default=4, type=int, help="number of distributed processes")

    return parser.parse_args()


"""========================================
@@@ 分布式运行环境初始化
========================================"""


def init_distributed_mode(args):
    """初始化分布式模式
    注：一个进程对应一块 GPU

    - 如果是"多机多卡"的机器：
        - WORLD_SIZE 代表所有机器中使用的进程（GPU）数量
        - RANK 代表当前机器/进程节点，如 0 为主机
        - LOCAL_RANK 表示当前机器中第几个进程
    - 如果是"单机多卡"的机器：
        - WORLD_SIZE 代表有几块 GPU
        - RANK 和 LOCAL_RANK 代表第几块 GPU

    Args:
        args (argparse.Namespace): 启动参数解释器对象
    """
    # torch.distributed.launch 会根据 nproc_per_node 自动设置环境参数
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        return

    torch.cuda.set_device(args.gpu)  # 对当前进程指定使用的 GPU
    print("| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True)
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    dist.barrier()  # 等待每个GPU都运行完这个地方以后再继续


def reduce_value(value, average=True):
    """根据 world_size 对值进行减少

    Args:
        value (tensor, int, float): 要平均计算的值
        average (bool, optional): 是否取平均. Defaults to True.

    Returns:
        value 相同类型
    """
    world_size = dist.get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value


"""====================
@@@ 数据集构建
===================="""


def load_data(data_path: str, is_download: bool = True):
    """读取训练、测试数据

    Args:
        data_path (str): 数据集存储路径
        is_download (bool, optional): 是否自动下载数据集（若 data_path 不存在）. Defaults to True.

    Returns:
        train, test: 返回训练、测试数据集对象
    """
    transform = {
        "train": transforms.Compose(
            [
                # transforms.RandomHorizontalFlip(),  # 随机水平翻转
                # transforms.RandomRotation(10),  # 随机旋转
                # transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                # transforms.Resize((32, 32)),  # 图像大小调整为 (w,h)=(32，32)
                transforms.ToTensor(),  # 将图像转换为张量 Tensor
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
        "test": transforms.Compose(
            [
                # transforms.Resize((32, 32)),  # 图像大小调整为 (w,h)=(32，32)
                transforms.ToTensor(),  # 将图像转换为张量 Tensor
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
    }

    # 读取 CIFAR10 数据集
    train = CIFAR10(root=data_path, train=True, transform=transform["train"], download=is_download)
    test = CIFAR10(root=data_path, train=False, transform=transform["test"], download=is_download)

    return train, test


"""====================
@@@ 模型构建
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
        """DenseNet构造函数

        Args:
            growth_rate (int, optional): k通道增长倍率. Defaults to 32.
            n_layers (tuple, optional): 每一层DenseBlock的数量. Defaults to (6, 12, 24, 16).
            n_init_features (int, optional): 特征卷积层输出通道数. Defaults to 64.
            bn_size (int, optional): DenseLayer中BN层尺寸. Defaults to 4.
            drop_rate (float, optional): drouput比例. Defaults to 0.
            n_classes (int, optional): 输出层神经元数量，类别数. Defaults to 1000.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


"""=========================
@@@ 模型训练、验证
========================="""


def train_one_epoch(model, optimizer, data_loader, device, rank, epoch_info):
    """模型训练一个周期函数

    Args:
        model (nn.module): 模型架构对象
        optimizer: 优化器对象
        data_loader: 数据迭代器
        device: 运算设备
        rank (int): 进程序号
        epoch_info (str): 周期字符串信息

    Returns:
        mean_loss: 当前周期下模型训练损失平均值
    """
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    # 存储损失、样本预测正确数量和总数量
    mean_loss = torch.zeros(1).to(device)

    optimizer.zero_grad()

    if rank == 0:
        data_loader = tqdm(data_loader, bar_format="{l_bar}{bar:10}{r_bar}")

    for step, (images, labels) in enumerate(data_loader):
        # 模型推理
        pred = model(images.to(device))
        # 计算损失
        loss = criterion(pred, labels.to(device))
        loss.backward()
        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses
        # 计算显存
        mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)

        # 在进程 0 中打印平均 loss
        if rank == 0:
            data_loader.set_description(("%11s" * 2 + "%11.4g") % (epoch_info, mem, mean_loss.item()))

        if not torch.isfinite(loss):
            print("WARNING: non-finite loss, ending training ", loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, device):
    """模型验证函数

    Args:
        model (nn.module): 模型对象
        data_loader: 数据迭代器
        device: 运算设备

    Returns:
        mean_loss, correct: 验证损失, 类别预测正确数量
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    # 存储损失、样本预测正确数量
    mean_loss = torch.zeros(1).to(device)
    correct = torch.zeros(1).to(device)

    for step, (images, labels) in enumerate(data_loader):
        # 模型推理
        pred = model(images.to(device))
        # 计算损失
        loss = criterion(pred, labels.to(device))
        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses
        # 计算预测正确数量和类别总数量
        pred = torch.max(pred, dim=1)[1]
        correct += torch.eq(pred, labels.to(device)).sum()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    correct = reduce_value(correct, average=False)

    return mean_loss.item(), correct.item()


def main(args):
    """脚本主函数"""
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    # 初始化各进程环境
    init_distributed_mode(args=args)

    # 获取当前进程下的相关参数
    rank = args.rank
    device = torch.device(args.device)
    batch_size = args.batch_size
    weights_path = args.weights
    args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增
    checkpoint_path = ""

    if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print(args)
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter(args.output)
        if os.path.exists(args.output) is False:
            os.makedirs(args.output)

    # 根据数据集路径获取训练和测试数据
    train_data, test_data = load_data(args.data_path)
    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    val_sampler = torch.utils.data.distributed.DistributedSampler(test_data)

    # 将样本索引每 batch_size 个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)

    # 分配读取数据的线程数
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, args.num_workers])  # number of workers
    if rank == 0:
        print("Using {} dataloader workers every process".format(nw))
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_sampler=train_batch_sampler,
        pin_memory=True,
        num_workers=nw,
    )

    val_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=nw,
    )

    # 初始化 DenseNet121 对象
    model = DenseNet121(12, (5, 10, 20), n_classes=10, drop_rate=0.1).to(device)

    # 如果存在预训练权重则载入
    if os.path.exists(weights_path):
        weights_dict = torch.load(weights_path, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(load_weights_dict, strict=False)
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
        if rank == 0:
            torch.save(model.state_dict(), checkpoint_path)

        dist.barrier()
        # 这里注意，一定要指定 map_location 参数，否则会导致第一块 GPU 占用更多资源
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)
    else:
        # 只有训练带有 BN 结构的网络时使用 SyncBatchNorm 才有意义
        if args.syncBN:
            # 使用 SyncBatchNorm 后训练会更耗时
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # 转为 DDP 模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # 初始化优化器
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if rank == 0:
        print(("%11s" * 3) % ("Epoch", "GPU_mem", "Loss"))

    best_loss = 1e9  # 记录最佳损失
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        # 模型训练和验证
        train_loss = train_one_epoch(model, optimizer, train_loader, device, rank, f"{epoch+1}/{args.epochs}")
        val_loss, correct = evaluate(model, val_loader, device, rank, f"{epoch+1}/{args.epochs}")

        # tensorboard 记录训练过程
        if rank == 0:
            tb_writer.add_scalar("loss", train_loss, epoch)
            tb_writer.add_scalar("loss_val", val_loss, epoch)
            tb_writer.add_scalar("accuracy", correct / val_sampler.total_size, epoch)
            tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.module.state_dict(), f"{args.output}/model-best.pth")

    # 保存模型最后的权重，删除临时缓存文件
    if rank == 0:
        torch.save(model.module.state_dict(), f"{args.output}/model-final.pth")
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    # 关闭所有进程，释放计算机资源
    dist.destroy_process_group()


if __name__ == "__main__":
    opt = init_parse_args()
    main(opt)
