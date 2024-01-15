# -*- coding: UTF-8 -*-
"""
++++++++++++++++++++++++++++++++++++++
@ File        : MNIST_CNN_PyTorch_DataDistributed.py
@ Time        : 2024/1/15 14:51
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (Conda)
++++++++++++++++++++++++++++++++++++++
参考：https://zhuanlan.zhihu.com/p/403339155

CMD: python -m torch.distributed.launch --nproc_per_node=2 --use_env MNIST_CNN_PyTorch_DataDistributed.py

两种GPU训练方法：DataParallel 和 DistributedDataParallel：
- DataParallel 是单进程多线程的，仅仅能工作在单机中。而 DistributedDataParallel 是多进程的，可以工作在单机或多机器中。
- DataParallel 通常会慢于 DistributedDataParallel 。所以目前主流的方法是 DistributedDataParallel。

启动方式：
- torch.distributed.launch: cmd: "python -m torch.distributed.launch --help" (-m: run library module as a script)
- torch.multiprocessing
++++++++++++++++++++++++++++++++++++++
"""
import os
import sys
import math
import tempfile
import argparse

import torch
import torch.distributed as dist

import torch.nn as nn

import torch.utils.data as data

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms
from torchvision.datasets import MNIST

from tqdm import tqdm


"""========================================
@@@ 分布式环境初始化
========================================"""


def init_distributed_mode(args):
    # 如果是多机多卡的机器，WORLD_SIZE代表使用的机器数，RANK对应第几台机器
    # 如果是单机多卡的机器，WORLD_SIZE代表有几块GPU，RANK和LOCAL_RANK代表第几块GPU
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)  # 对当前进程指定使用的GPU
    args.dist_backend = "nccl"  # 通信后端，nvidia GPU推荐使用NCCL
    print("| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True)
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    dist.barrier()  # 等待每个GPU都运行完这个地方以后再继续


def cleanup():
    dist.destroy_process_group()


def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value


"""=========================
@@@ 模型训练、验证
========================="""


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data

        pred = model(images.to(device))

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        # 在进程0中打印平均loss
        if is_main_process():
            data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

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
    model.eval()

    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)

    # 在进程0中打印验证进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    sum_num = reduce_value(sum_num, average=False)

    return sum_num.item()

"""====================
@@@ 模型构建
===================="""


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


def main(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    # 初始化各进程环境
    init_distributed_mode(args=args)

    rank = args.rank
    device = torch.device(args.device)
    batch_size = args.batch_size
    # weights_path = args.weights
    args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增
    checkpoint_path = ""

    if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print(args)
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter()
        if os.path.exists("./weights") is False:
            os.makedirs("./weights")

    ###
    data_path = "./data"
    transform = transforms.Compose([transforms.ToTensor()])  # 转换为张量
    # 读取图片为数据集
    train_data = MNIST(root=args.data_path, train=True, transform=transform, download=True)
    test_data = MNIST(root=args.data_path, train=False, transform=transform, download=True)

    # train_info, val_info, num_classes = read_split_data(args.data_path)
    # train_images_path, train_images_label = train_info
    # val_images_path, val_images_label = val_info

    # check num_classes
    # assert args.num_classes == num_classes, "dataset num_classes: {}, input {}".format(args.num_classes,
    #                                                                                    num_classes)

    # data_transform = {
    #     "train": transforms.Compose([transforms.RandomResizedCrop(224),
    #                                  transforms.RandomHorizontalFlip(),
    #                                  transforms.ToTensor(),
    #                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    #     "val": transforms.Compose([transforms.Resize(256),
    #                                transforms.CenterCrop(224),
    #                                transforms.ToTensor(),
    #                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    #
    # # 实例化训练数据集
    # train_data_set = MyDataSet(images_path=train_images_path,
    #                            images_class=train_images_label,
    #                            transform=data_transform["train"])
    #
    # # 实例化验证数据集
    # val_data_set = MyDataSet(images_path=val_images_path,
    #                          images_class=val_images_label,
    #                          transform=data_transform["val"])

    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    val_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    ###

    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
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
    model = Net().to(device)

    # 如果存在预训练权重则载入
    # if os.path.exists(weights_path):
    #     weights_dict = torch.load(weights_path, map_location=device)
    #     load_weights_dict = {k: v for k, v in weights_dict.items() if model.state_dict()[k].numel() == v.numel()}
    #     model.load_state_dict(load_weights_dict, strict=False)
    # else:
    #     checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
    #     # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
    #     if rank == 0:
    #         torch.save(model.state_dict(), checkpoint_path)
    #
    #     dist.barrier()
    #     # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
    #     model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 是否冻结权重
    # if args.freeze_layers:
    #     for name, para in model.named_parameters():
    #         # 除最后的全连接层外，其他权重全部冻结
    #         if "fc" not in name:
    #             para.requires_grad_(False)
    # else:
    #     # 只有训练带有BN结构的网络时使用SyncBatchNorm采用意义
    #     if args.syncBN:
    #         # 使用SyncBatchNorm后训练会更耗时
    #         model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=0.005)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        mean_loss = train_one_epoch(
            model=model, optimizer=optimizer, data_loader=train_loader, device=device, epoch=epoch
        )

        scheduler.step()

        sum_num = evaluate(model=model, data_loader=val_loader, device=device)
        acc = sum_num / val_sampler.total_size

        if rank == 0:
            print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
            tags = ["loss", "accuracy", "learning_rate"]
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            tb_writer.add_scalar(tags[1], acc, epoch)
            tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

            torch.save(model.module.state_dict(), "./weights/model-{}.pth".format(epoch))

    # 删除临时缓存文件
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.1)
    # 是否启用SyncBatchNorm
    parser.add_argument("--syncBN", type=bool, default=True)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument("--data-path", type=str, default="./data")

    # resnet34 官方权重下载地址
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    parser.add_argument("--weights", type=str, default="resNet34.pth", help="initial weights path")
    parser.add_argument("--freeze-layers", type=bool, default=False)
    # 不要改该参数，系统会自动分配
    parser.add_argument("--device", default="cuda", help="device id (i.e. 0 or 0,1 or cpu)")
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument("--world-size", default=4, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    opt = parser.parse_args()

    main(opt)

