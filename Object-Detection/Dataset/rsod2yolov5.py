# -*- encoding: utf-8 -*-
"""
+++++++++++++++++++++++++++++++++++++++++
@ File        : RSOD2YOLOv5.py
@ Time        : 2022/8/1 14:56
@ Author      : Mirrich Wang
@ Version     : Python 3.9.12 (Conda)
+++++++++++++++++++++++++++++++++++++++++
RSOD 的 txt 标注转换成 YOLOv5 格式
其中，RSOD 976张图片中没有划分训练、校验、测试集，40 张 playground 未标注图片作为测试集，
剩下的936张图片8:2划分训练、校验集，因此最终的数量为748:188:40
RSOD 的 边界框信息为 [x_min, y_min, x_max, y_max]
YOLOv5 格式计算公式：
    x = ((x_min+x_max)/2)/W
    y = ((y_min+y_max)/2)/H
    w = (x_max-x_min)/2
    h = (y_max-y_min)/2
W、H 为图像宽度和高度，x, y, w, h ∈ [0, 1]
+++++++++++++++++++++++++++++++++++++++++
"""

# 导入基础模块
import os
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image

# 导入可视化模块
from tqdm import tqdm

"""
=====================
@@@ 自定义工具函数
=====================
"""


def xyxy2xywh(w, h, x1, y1, x2, y2):
    """左上角、右下角坐标转换函数

    [x_min, y_min, x_max, y_max] 转换成YOLOv5格式 [x, y, w, h]
    Args:
        w: (int) 图片宽度 height
        h: (int) 图片高度 width
        x1: (float) 左上角横坐标
        y1: (float) 左上角纵坐标
        x2: (float) 右下角横坐标
        y2: (float) 右下角纵坐标

    Returns:
        (list) 转换成YOLOv5格式的 [x, y, w, h]

    """
    x = ((x1 + x2) / 2) / w  # x center
    y = ((y1 + y2) / 2) / h  # y center
    w = (x2 - x1) / w  # width
    h = (y2 - y1) / h  # height
    try:
        x = np.where(x > 0, x, 0)
        x = np.where(x < 1, x, 1)
        y = np.where(y > 0, y, 0)
        y = np.where(y < 1, y, 1)
        w = np.where(w > 0, w, 0)
        w = np.where(w < 1, w, 1)
        h = np.where(h > 0, h, 0)
        h = np.where(h < 1, h, 1)
    except:
        pass
    return [float(x), float(y), float(w), float(h)]


"""
=====================
@@@ Config
=====================
"""

train_size = .8  # 训练集比例
root = r'F:\DeepLearning\Object-Detection (目标检测)\data\RSOD'  # 数据集根目录
raw_file = "raw"  # 存放解压源文件的目录，若为根目录填"."
splits = ['train', 'val', 'test']  # 子集
label_names = ["aircraft", "oiltank", "overpass", "playground"]  # 类别名字
pd.DataFrame(label_names).to_csv(os.path.join(root, "RSOD_Categories.csv"), index=False)
class_to_idx = dict(zip(label_names, range(len(label_names))))

"""
=====================
@@@ 获取子集文件名
=====================
"""

# 使用 glob 模块查询所有文件名
img_files = glob(os.path.join(root, raw_file, r"*\JPEGImages\*.jpg"))
txt_files = glob(os.path.join(root, raw_file, r"*\Annotation\labels\*.txt"))
# 剔除路径和拓展名
img_files = list(map(lambda i: os.path.splitext(os.path.basename(i))[0], img_files))
txt_files = list(map(lambda i: os.path.splitext(os.path.basename(i))[0], txt_files))
test_files = list(set(img_files) - set(txt_files))
# 按照8:2获取训练、校验集文件名
N = len(txt_files)
indices = np.arange(len(txt_files))
np.random.shuffle(indices)  # 打乱原始索引
train_indices = indices[:int(N * train_size)]
valid_indices = indices[int(N * train_size):]
# 获取文件名
train_files = list(map(lambda i: txt_files[i], train_indices))
valid_files = list(map(lambda i: txt_files[i], valid_indices))
# 封装成字典
files = dict(zip(splits, [train_files, valid_files, test_files]))

"""
=====================
@@@ 转换YOLOv5格式
=====================
"""

for split in splits:
    # 创建子集文件夹
    os.makedirs(os.path.join(root, 'images', split))
    if split != "test":
        os.makedirs(os.path.join(root, 'labels', split))
        os.makedirs(os.path.join(root, "annotations", split))
    for i, file in tqdm(enumerate(files[split]), total=len(files[split]), desc=split.title()):
        label = file.split("_")[0]  # 获取标签名字
        # 读取图片
        img = Image.open(os.path.join(root, raw_file, label, 'JPEGImages', file + '.jpg'))
        # 判断是否为测试集，若为测试集则不读取标注文件
        if split != "test":
            # 读取 xml 文件并另存为
            xml_file = open(os.path.join(root, raw_file, label, "Annotation", "xml", f"{file}.xml"))
            open(os.path.join(root, "annotations", split, f"{file}.xml"), "w+").write(xml_file.read())
            # 读取 txt 文件
            anno = pd.read_table(os.path.join(root, raw_file, label, r'Annotation\labels', f'{file}.txt'), header=None)
            # 将字符串类别名字转换成索引
            labels = list(map(lambda i: class_to_idx[i], anno[1]))
            bboxes = anno.values[:, 2:]
            new_bboxes = pd.DataFrame([[j, *xyxy2xywh(*img.size, *bbox)] for j, bbox in zip(labels, bboxes)])
            # 保存 YOLOv5 格式标注和图片
            new_bboxes.to_csv(os.path.join(root, 'labels', split, f'{file}.txt'), sep='\t', index=False, header=False)
        img.save(os.path.join(root, 'images', split, f'{file}.jpg'))

