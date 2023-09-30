# -*- encoding: utf-8 -*-
"""
+++++++++++++++++++++++++++++++++++++++++
@ File        : COCO2YOLOv5.py
@ Time        : 2022/8/1 16:56
@ Author      : Mirrich Wang
@ Version     : Python 3.9.12 (Conda)
+++++++++++++++++++++++++++++++++++++++++
COCO 的 json 标注格式 -> YOLOv5 格式
其中，COCO 的边界框信息为[x, y, w, h]
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
import json
import numpy as np
import pandas as pd

# 导入可视化模块
from tqdm import tqdm

"""
=====================
@@@ 自定义工具函数
=====================
"""


def load_json(file):
    """读取 json 格式文件函数

    Args:
        file: [str] 文件目录字符串

    Returns:
        dict

    """
    with open(file, "r") as f:
        data = json.load(f)
    return data


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

# 设置相关参数
root = r"G:\datasets\COCO"  # 数据集根目录
save_dir = r"G:\Object-Detection (目标检测)\data\COCO"  # 保存目录
raw_files = "raw"  # 存放原始文件目录
years = ["2014", "2017"]  # 数据集年份
splits = ["train", "val"]  # 数据集划分
# 制作标签映射，COCO数据集中标签索引非连续
label_idx = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
]
col_map = dict(zip(label_idx, range(len(label_idx))))

"""
=========================
@@@ COCO格式转YOLOv5格式
=========================
"""

for year in years:
    for split in splits:
        # 创建子集文件夹
        os.makedirs(os.path.join(save_dir, "labels", f"{split}{year}"), exist_ok=True)
        # 读取标注json文件
        instances = load_json(os.path.join(root, "annotations", f"instances_{split}{year}.json"))
        # 获取当前年份和子集的图片信息，id和图片名字
        images = pd.DataFrame(instances["images"])[["id", "file_name", "height", "width"]]
        # 获取标注信息，图片id、标签id和标注框
        annotation = pd.DataFrame(instances["annotations"])[["image_id", "category_id", "bbox"]]
        categories = pd.DataFrame(instances["categories"])
        if not os.path.exists(os.path.join(save_dir, "COCO_Categories.csv")):
            categories.to_csv(os.path.join(save_dir, "COCO_categories.csv"), index=False)
        for img_id, img_name, H, W in tqdm(images.values, total=len(images), desc=split + year):
            # 获取对应图片产生的txt文件完整路径
            txt_file = os.path.join(
                save_dir,
                "labels",
                f"{split}{year}",
                os.path.splitext(img_name)[0] + ".txt",
            )
            # if os.path.exists(txt_file):
            #     continue
            bbox = annotation[annotation["image_id"] == img_id]["bbox"]  # 获取指定图片的标注信息
            # 获取指定图片的标签信息
            labels = annotation[annotation["image_id"] == img_id]["category_id"].apply(lambda i: col_map[i])
            new_bbox = [[label, *xyxy2xywh(W, H, x, y, x + w, y + h)] for label, (x, y, w, h) in zip(labels, bbox)]
            # 标注文件保存到txt文件中
            pd.DataFrame(new_bbox).to_csv(txt_file, index=False, header=False, sep="\t")
