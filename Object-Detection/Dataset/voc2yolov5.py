# -*- encoding: utf-8 -*-
"""
+++++++++++++++++++++++++++++++++++++++++
@ File        : VOC2YOLOv5.py
@ Time        : 2022/7/16 16:23
@ Author      : Mirrich Wang
@ Version     : Python 3.9.12 (Conda)
+++++++++++++++++++++++++++++++++++++++++
VOC 的 xml 标注格式 -> YOLOv5 格式
其中，VOC 的边界框信息为[x_min, y_min, x_max, y_max]
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
import xml.dom.minidom
import pandas as pd
import numpy as np

# 导入可视化模块
from PIL import Image
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


def xml2txt(file, label_names):
    """xml标注转换成YOLO-V5格式

    Args:
        file: (str) xml 文件
        label_names: (str[]) 标签列表

    Returns:
        (list) YOLO-V5格式的 bbox

    """
    # 打开xml文档
    dom = xml.dom.minidom.parse(file)
    # 得到文档元素对象
    root = dom.documentElement
    name = root.getElementsByTagName('filename')[0].firstChild.data.split(".")[0]
    w = root.getElementsByTagName('width')[0].firstChild.data.split(".")[0]
    h = root.getElementsByTagName('height')[0].firstChild.data.split(".")[0]

    class_to_ind = dict(zip(label_names, range(len(label_names))))

    data = root.getElementsByTagName('object')
    new = []
    for doc in data:
        sens = doc.getElementsByTagName('name')[0].firstChild.data
        xmin = doc.getElementsByTagName('xmin')[0].firstChild.data
        ymin = doc.getElementsByTagName('ymin')[0].firstChild.data
        xmax = doc.getElementsByTagName('xmax')[0].firstChild.data
        ymax = doc.getElementsByTagName('ymax')[0].firstChild.data
        xywh = xyxy2xywh(float(w), float(h), float(xmin), float(ymin), float(xmax), float(ymax))
        new.append([class_to_ind[sens], xywh[0], xywh[1], xywh[2], xywh[3]])
    return new


"""
=====================
@@@ Config
=====================
"""

years = ["2007", "2012"]  # 数据集年份
splits = ["train", "val", "test"]  # 数据集划分
root = r"G:\datasets\VOC"  # 数据集根目录
save_dir = r"G:\Object-Detection (目标检测)\data\VOC"  # 保存YOLOv5标注目录
raw_file = r"raw\VOCdevkit"  # 存放源文件解压后目录
# VOC 的 20 类标签
label_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

"""
=====================
@@@ VOC xml 标注转换成YOLOv5格式
=====================
"""

# 获取年份 year
for year in years:
    anno_path = os.path.join(root, raw_file, "VOC" + year, "Annotations")  # 获取Annotation目录
    img_path = os.path.join(root, raw_file, "VOC" + year, "JPEGImages")  # 获取图片目录
    # 获取划分集文件名称 split (train, val, test)
    for split in splits:
        set_file = os.path.join(root, raw_file, "VOC" + year, "ImageSets", "Main", split + ".txt")
        image_set = pd.read_table(set_file, header=None, dtype=str)[0]

        # 获得图片和标签保存地址，将YOLOv5格式的.txt标注文件保存到保存目录，原始标注和图片保存到原始数据集路径
        save_label_path = os.path.join(save_dir, "labels", split + year)  # 将YOLOv5格式的标签保存至目录中
        save_anno_path = os.path.join(root, "annotations", split + year)
        save_img_path = os.path.join(root, "images", split + year)
        # 创建文件夹
        os.makedirs(save_img_path, exist_ok=True)
        if (split, year) != ("test", "2012"):
            # 由于2017年的test没有标注文件，因此不需要创建标注目录
            os.makedirs(save_label_path, exist_ok=True)
            os.makedirs(save_anno_path, exist_ok=True)

        # 获取图片和标注文件名字
        for img_name in tqdm(image_set, total=len(image_set), desc=split + year):
            img = Image.open(os.path.join(img_path, img_name + ".jpg"))
            img.save(os.path.join(save_img_path, img_name + ".jpg"))
            # VOC2012 测试图片没有 annotations，因此需要跳过
            if (split, year) != ("test", "2012"):
                # 读取 xml 文件并且转换成 YOLO-V5 数据格式
                xml_file = os.path.join(anno_path, img_name + ".xml")
                txt_file = os.path.join(save_label_path, img_name + ".txt")
                bbox = pd.DataFrame(xml2txt(xml_file, label_names))
                # 保存标注文件
                bbox.to_csv(txt_file, index=False, header=False, sep="\t")
                # 保存xml文件
                open(os.path.join(save_anno_path, img_name + ".xml"), "w+").write(open(xml_file).read())
