# -*- encoding: utf-8 -*-",
"""
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@ File        : Image-processing.py
@ Time        : 2022/10/16 23:49
@ Author      : Mirrich Wang
@ Version     : Python 3.8.10 (基于 PyTorch 的 Anaconda3 虚拟环境)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
模仿 torchvision 利用 pickle、struct、numpy 等基础模块解析图像分类数据集。
将陆续收集各种 图像分类 数据集，从而制作成类进行相关预处理
目前已有：
    - 灰白单通道图片：MNIST、FashionMNIST (28x28, 10分类)
    - 彩色三通道: CIFAR10、CIFAR100 (32x32, 10、100分类)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

# 导入基础模块
import os
import pickle
import struct
import numpy as np
from PIL import Image

# 导入相关可视化模块
import matplotlib.pyplot as plt  # 绘图
from tqdm import tqdm  # 进度条


"""+++++++++++++++++++
@@@ 自定义工具函数
++++++++++++++++++++++"""


def load_meta(file):
    """读取 meta 格式的数据文件

    Args:
        file: 文件路径的字符串

    Returns:
        dict

    """
    with open(file, "rb") as fo:
        d = pickle.load(fo, encoding="iso-8859-1")
    return d


def load_byte(file, cache=">IIII", dtp=np.uint8):
    """读取 ubyte 格式数据

    Args:
        file: 文件路径的字符串
        cache: 缓存字符
        dtp: 矩阵类型

    Returns:
        np.array

    """
    iter_num = cache.count("I") * 4
    with open(file, "rb") as f:
        magic = struct.unpack(cache, f.read(iter_num))
        data = np.fromfile(f, dtype=dtp)
    return data


"""+++++++++++++++++++
@@@ 构造相关数据集类
++++++++++++++++++++++"""


class Mnist:
    """自定义 MNIST 相关操作的类"""

    name = "MNIST"  # 数据集名字

    num_channels = 1  # 图片通道数
    num_labels = 10  # 标签数量

    image_size = [28, 28]  # 图片尺寸

    train_size = 60000  # 训练数据大小
    test_size = 10000  # 测试数据大小

    label_names = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
    ]  # 标签的字符串信息

    base_folder = ""  # 原始文件目录

    # 资源信息
    mirrors = [
        "http://yann.lecun.com/exdb/mnist/",
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
    ]

    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
    ]

    is_label_name = False  # 保存图片时，是否使用标签名字命名目录

    def __init__(self, root=".", vis_sample=False):
        """类构造方法

        通过 root 路径下的 raw 源文件目录，获取源文件中数据集信息
        Args:
            root: 数据集路径
            vis_sample: 是否查看样本图片

        """
        print(self.__repr__())

        self.root = root
        self.raw_dir = os.path.join(self.root, "raw", self.base_folder)  # 存放原始文件路径

        assert os.path.exists(self.raw_dir)

        # 是否需要展示部分数据集图片
        if vis_sample:
            self.plot_sample()

    def __repr__(self):
        """返回数据集字符串信息"""
        channel_info = "(RGB)" if self.num_channels == 3 else "(Gray)"
        train_rate = self.train_size / (self.train_size + self.test_size)
        test_rate = self.test_size / (self.train_size + self.test_size)
        info = {
            "Dataset Name": f"{self.name}",
            "Num labels": f"{self.num_labels}",
            "Num channels": f"{self.num_channels} {channel_info}",
            "Image size": f"{self.image_size}",
            "Train size": f"{self.train_size} ({train_rate :.2%})",
            "Test size": f"{self.test_size} ({test_rate :.2%})",
        }
        s = (
            " Info ".center(50, "=")
            + "\n"
            + "\n".join(["| %-21s | %-22s |" % (k, v) for k, v in info.items()])
            + "\n"
            + "=" * 50
        )
        return s

    def load_data(self):
        """通过源文件加载数据"""
        # 读取 ubyte 源文件
        train_data = load_byte(os.path.join(self.raw_dir, "train-images-idx3-ubyte"))
        test_data = load_byte(os.path.join(self.raw_dir, "t10k-images-idx3-ubyte"))
        train_label = load_byte(os.path.join(self.raw_dir, "train-labels-idx1-ubyte"), ">II")
        test_label = load_byte(os.path.join(self.raw_dir, "t10k-labels-idx1-ubyte"), ">II")
        # 将 (size, 784) 的矩阵重新定制为 (size, 28, 28)
        train_data = train_data.reshape([-1] + self.image_size)
        test_data = test_data.reshape([-1] + self.image_size)

        return (train_data, train_label), (test_data, test_label)

    def plot_sample(self, num=32):
        """绘制样本图片方法

        Args:
            num: 绘制图片数量，取8的倍数

        Returns:
            None

        """
        assert num % 8 == 0
        rows = int(num / 8)
        cols = 8
        # 随机抽取32张图片
        ids = np.random.randint(0, self.test_size, num)
        (_, _), (data, labels) = self.load_data()
        # 创建画布
        fig, ax = plt.subplots(rows, cols, tight_layout=True)
        for i, idx in enumerate(ids):
            # 计算子图位置
            r, c = int(i / 8), i % 8
            # 显示图片和标签
            ax[r][c].imshow(data[idx], cmap="gray")
            ax[r][c].set_title(f"<{self.label_names[labels[idx]]}>", fontsize=7)
            # 设置空坐标刻度（关闭坐标轴）
            ax[r][c].set_xticks([])
            ax[r][c].set_yticks([])
        plt.show()

    def get_categories(self, path=None):
        """获取数据集标签信息

        若有传入路径，那么将保存到路径下的 <name>.names 制表间隔符的文本文件
        Args:
            path: 保存目录字符串

        Returns:
            若无传入 path 则返回标签信息列表

        """
        if path:
            save_dir = os.path.join(path, f"{self.name}.names")
            print(f'Label information save to "{save_dir}"')
            with open(save_dir, "w+") as f:
                for i, category in enumerate(self.label_names):
                    f.write(f"{i}\t{category}\n")
        else:
            return self.label_names

    def save_data(self, path, img_ext=".png"):
        """保存矩阵到本地路径的方法

        按照 path/<subset>/<label> 来分类保存图片信息，一般 subset 指 train、valid或test
        Args:
            path: 图片保存路径
            img_ext: 图片格式、后缀名，默认为 .png，可用 .jpg、.bmp

        """
        # 加载数据和文件名
        (train_data, train_labels), (test_data, test_labels) = self.load_data()
        train_files, test_files = self._generate_filenames()
        print(f'Convert np.array -> PIL.Image and save to "{path}"')
        # 将训练和测试数据通过 np.concatenate 拼接在一起
        data = np.concatenate([train_data, test_data])
        files = np.concatenate([train_files, test_files])
        labels = np.concatenate([train_labels, test_labels])
        # 判断是否根据标签字符串来命名分类目录，否则采取标签索引命名
        if self.is_label_name:
            labels = list(map(lambda i: self.label_names[i], labels))
        with tqdm(total=len(data)) as pbar:
            for i, (data, label, file) in enumerate(zip(data, labels, files)):
                # 判断索引是否在训练大小内来制作保存目录字符串
                if i < self.train_size:
                    save_dir = os.path.join(path, "train", str(label))
                else:
                    save_dir = os.path.join(path, "test", str(label))
                # 假如没有目录则新创建
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                # 加载矩阵到 PIL.Image 中
                img = Image.fromarray(data)
                img.save(os.path.join(save_dir, file) + img_ext)  # 保存图片到路径
                pbar.update(1)
            pbar.set_postfix_str("Save Completed!")

    def _generate_filenames(self):
        """生成图片文件名方法"""
        # 加载图像标签
        train_label = load_byte(self.raw_dir + "\\train-labels-idx1-ubyte", ">II")
        test_label = load_byte(self.raw_dir + "\\t10k-labels-idx1-ubyte", ">II")
        # 初始化图像索引字典和文件列表
        train_id = dict(zip(range(self.num_labels), [0] * self.num_labels))
        test_id = train_id.copy()
        train_files, test_files = [], []
        for i, label in enumerate(np.concatenate([train_label, test_label])):
            if i < self.train_size:
                train_files.append("train_%i_%i" % (label, train_id[label]))
                train_id[label] += 1
            else:
                test_files.append("test_%i_%i" % (label, test_id[label]))
                test_id[label] += 1
        return train_files, test_files


class FashionMnist(Mnist):
    """继承 Mnist 类的 FashionMNIST 数据集相关操作类"""

    name = "FashionMNIST"

    label_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]  # 标签名字


class Cifar10(Mnist):
    """继承 Mnist 类的 CIFAR10 数据集相关操作类"""

    name = "CIFAR10"

    num_channels = 3
    num_labels = 10
    image_size = [32, 32]
    train_size = 50000
    test_size = 10000
    label_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    base_folder = "cifar-10-batches-py"

    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"

    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]

    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    is_label_name = True

    def load_data(self):
        """重写读取数据方法，读取 meta 格式文件"""
        # 通过 meta 加载数据
        label_key = self.meta["key"].replace("_names", "") + "s"  # 获取 meta 数据中标签的键
        train_meta = list(map(lambda i: load_meta(self.raw_dir + "/%s" % i[0]), self.train_list))
        test_meta = list(map(lambda i: load_meta(self.raw_dir + "/%s" % i[0]), self.test_list))
        # 将训练数据列表中矩阵数据进行提取并拼接起来
        train_data = np.concatenate(list(map(lambda i: i["data"], train_meta)))
        train_labels = np.concatenate(list(map(lambda i: i[label_key], train_meta)))
        # 提取测试数据
        test_data = np.concatenate(list(map(lambda i: i["data"], test_meta)))
        test_labels = np.concatenate(list(map(lambda i: i[label_key], test_meta)))
        # 将 (size, 3072) 图像转换成 (size, 32, 32, 3)
        train_data = train_data.reshape([-1, self.num_channels] + self.image_size).transpose(0, 2, 3, 1)
        test_data = test_data.reshape([-1, self.num_channels] + self.image_size).transpose(0, 2, 3, 1)

        return (train_data, train_labels), (test_data, test_labels)

    def get_categories(self, path=None):
        """重写获取标签方法，从 meta 文件中获取标签信息"""
        meta = load_meta(os.path.join(self.raw_dir, self.meta["filename"]))
        self.label_names = meta[self.meta["key"]]
        if path:
            save_dir = os.path.join(path, self.name + ".names")
            print(f"Label information save to {save_dir}")
            with open(save_dir, "w+") as f:
                for i, category in enumerate(self.label_names):
                    f.write(f"{i}\t{category}\n")
        else:
            return self.label_names

    def _generate_filenames(self):
        """重写生成图片文件名方法，从 meta 文件中获取文件名"""
        # 加载数据
        train_meta = list(map(lambda i: load_meta(self.raw_dir + "/%s" % i[0]), self.train_list))
        test_meta = list(map(lambda i: load_meta(self.raw_dir + "/%s" % i[0]), self.test_list))
        # 列表文件名拼接在一起
        train_files = np.concatenate(list(map(lambda i: i["filenames"], train_meta)))
        test_files = np.concatenate(list(map(lambda i: i["filenames"], test_meta)))

        return train_files, test_files


class Cifar100(Cifar10):
    """继承 Cifar10 类的 CIFAR100 数据集相关操作类"""

    name = "CIFAR100"
    num_labels = 100

    label_names = [
        "apple",
        "aquarium_fish",
        "baby",
        "bear",
        "beaver",
        "bed",
        "bee",
        "beetle",
        "bicycle",
        "bottle",
        "bowl",
        "boy",
        "bridge",
        "bus",
        "butterfly",
        "camel",
        "can",
        "castle",
        "caterpillar",
        "cattle",
        "chair",
        "chimpanzee",
        "clock",
        "cloud",
        "cockroach",
        "couch",
        "crab",
        "crocodile",
        "cup",
        "dinosaur",
        "dolphin",
        "elephant",
        "flatfish",
        "forest",
        "fox",
        "girl",
        "hamster",
        "house",
        "kangaroo",
        "keyboard",
        "lamp",
        "lawn_mower",
        "leopard",
        "lion",
        "lizard",
        "lobster",
        "man",
        "maple_tree",
        "motorcycle",
        "mountain",
        "mouse",
        "mushroom",
        "oak_tree",
        "orange",
        "orchid",
        "otter",
        "palm_tree",
        "pear",
        "pickup_truck",
        "pine_tree",
        "plain",
        "plate",
        "poppy",
        "porcupine",
        "possum",
        "rabbit",
        "raccoon",
        "ray",
        "road",
        "rocket",
        "rose",
        "sea",
        "seal",
        "shark",
        "shrew",
        "skunk",
        "skyscraper",
        "snail",
        "snake",
        "spider",
        "squirrel",
        "streetcar",
        "sunflower",
        "sweet_pepper",
        "table",
        "tank",
        "telephone",
        "television",
        "tiger",
        "tractor",
        "train",
        "trout",
        "tulip",
        "turtle",
        "wardrobe",
        "whale",
        "willow_tree",
        "wolf",
        "woman",
        "worm",
    ]

    base_folder = "cifar-100-python"

    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"

    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }


if __name__ == "__main__":
    root = "F:/DeepLearning/Image-Classification (图像分类)/data/"  # 数据集根目录

    mnist = Mnist(root=root + "MNIST", vis_sample=True)
    mnist.get_categories(root + "MNIST")
    # mnist.save_data(root + "MNIST")

    fashion_mnist = FashionMnist(root=root + "FashionMNIST", vis_sample=True)
    fashion_mnist.get_categories(root + "FashionMNIST")
    # fashion_mnist.save_data(root + "FashionMNIST")

    cifar10 = Cifar10(root=root + "CIFAR10", vis_sample=True)
    cifar10.get_categories(root + "CIFAR10")
    # cifar10.save_data(root + "CIFAR10")

    cifar100 = Cifar100(root=root + "CIFAR100", vis_sample=True)
    cifar100.get_categories(root + "CIFAR100")
    # cifar100.save_data(root + "CIFAR100")
