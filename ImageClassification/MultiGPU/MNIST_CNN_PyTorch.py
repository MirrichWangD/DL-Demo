# -*- coding: UTF-8 -*-
"""
++++++++++++++++++++++++++++++++++++++
@ File        : MNIST_CNN_PyTorch.py
@ Time        : 2024/1/15 14:51
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (Conda)
++++++++++++++++++++++++++++++++++++++
参考：
- https://www.bilibili.com/video/BV1yt4y1e7sZ/
- https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/train_multi_GPU
========================================================
启动命令：
- 旧版：python -m torch.distributed.launch --nproc_per_node=2 --use_env MNIST_CNN_PyTorch.py
- 新版：torchrun --nproc_per_node=2 MNIST_CNN_PyTorch.py
========================================================
模块版本：
    CUDA        11.6
    torch       1.12.1+cu116
    torchvision 0.13.1+cu116
========================================================
实验设备：单机两张 NVIDIA GeForce RTX 3080 10GB
++++++++++++++++++++++++++++++++++++++
"""
# 导入基础模块
import os
import sys
import tempfile
import argparse

# 导入进度条模块
from tqdm import tqdm

# 导入 torch 相关模块
import torch
import torch.nn as nn

# 导入优化器相关模块
import torch.optim as optim

# 导入分布式训练相关模块
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

# 导入数据集相关模块
from torchvision import transforms
from torchvision.datasets import MNIST


"""=========================
@@@ 启动参数初始化
========================="""


def init_parse_args():
    parser = argparse.ArgumentParser()
    # 数据集所在根目录
    parser.add_argument("--data-path", type=str, default="./data")
    # 文件输出路径
    parser.add_argument("--output", type=str, default="./weights/MNIST_CNN_PyTorch")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    # 优化器参数
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    # 分布式配置
    parser.add_argument("--dist-backend", default="nccl", type=str, help="please use 'gloo' on windows platform")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--num-workers", default=4, type=int)
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
@@@ 模型构建
===================="""


class CNN(nn.Module):
    """CNN 卷积网络在 MNIST 28x28 手写数字上应用的修改版"""

    def __init__(self):
        super(CNN, self).__init__()
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


"""=========================
@@@ 模型训练、验证
========================="""


def train_one_epoch(model, optimizer, data_loader, device, rank, epoch_info):
    """模型训练一个周期函数

    Args:
        model (nn.module): 模型对象
        optimizer: 优化器对象
        data_loader: 数据迭代器
        device: 运算设备
        epoch_info (str): 周期字符串信息

    Returns:
        float: 当前周期下模型训练损失平均值
    """
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
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
        mean_loss, sum_num: 验证损失, 类别预测正确数量
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    # 存储损失、样本预测正确数量
    mean_loss = torch.zeros(1).to(device)
    sum_num = torch.zeros(1).to(device)

    for step, (images, labels) in enumerate(data_loader):
        pred = model(images.to(device))

        loss = criterion(pred, labels.to(device))
        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    sum_num = reduce_value(sum_num, average=False)

    return mean_loss.item(), sum_num.item()


def main(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    # 初始化各进程环境
    init_distributed_mode(args=args)

    # 获取当前进程下的相关参数
    rank = args.rank
    device = torch.device(args.device)
    batch_size = args.batch_size

    if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print(args)
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter(args.output)
        if os.path.exists(args.output) is False:
            os.makedirs(args.output)

    ###
    transform = transforms.Compose([transforms.ToTensor()])  # 转换为张量
    # 读取 MNIST 数据集
    train_data = MNIST(root=args.data_path, train=True, transform=transform, download=True)
    test_data = MNIST(root=args.data_path, train=False, transform=transform, download=True)

    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    val_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    ###

    # 将样本索引每 batch_size 个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)

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
    # 实例化模型
    model = CNN().to(device)

    # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
    checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
    if rank == 0:
        torch.save(model.state_dict(), checkpoint_path)

    dist.barrier()
    # 这里注意，一定要指定 map_location 参数，否则会导致第一块 GPU 占用更多资源
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 转为 DDP 模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # 初始化优化器
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr * args.world_size, momentum=args.momentum, weight_decay=args.weight_decay)
    if rank == 0:
        print(("%11s" * 3) % ("Epoch", "GPU_mem", "Loss"))

    best_loss = 1e9  # 记录最佳损失
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        # 模型训练和验证
        train_loss = train_one_epoch(model, optimizer, train_loader, device, rank, f"{epoch+1}/{args.epochs}")
        val_loss, correct = evaluate(model=model, data_loader=val_loader, device=device)

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
