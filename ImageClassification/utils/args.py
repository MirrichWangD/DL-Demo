import argparse


def init_parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'infer'])
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--optimizer', type=str)

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default='./runs')

    return parser.parse_args()


# TODO
def init_distributed_parse_args():
    parser = argparse.ArgumentParser()
    # 数据集所在根目录
    parser.add_argument('--data-path', type=str, default='./data')
    # 文件输出路径
    parser.add_argument('--output', type=str, default='./weights/MNIST_CNN_PyTorch')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    # 优化器参数
    parser.add_argument('--lr', type=float, default=0.3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)

    # 分布式配置
    parser.add_argument('--dist-backend', default='nccl', type=str, help="please use 'gloo' on windows platform")
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--num-workers', default=4, type=int)
    # 以下参数不需要更改，会根据 nproc_per_node 自动设置
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--world-size', default=4, type=int, help='number of distributed processes')

    return parser.parse_args()
