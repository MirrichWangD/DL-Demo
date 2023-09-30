# -*- encoding: utf-8 -*-",
"""
+++++++++++++++++++++++++++++++++++
@ File        : ppocr2coco.py
@ Time        : 2023/1/11 14:17
@ Author      : Mirrich Wang
@ Version     : Python 3.x.x (env)
+++++++++++++++++++++++++++++++++++
个人用：山东智能审核项目
标注数据集->COCO格式用
其中qr类（二维码）的标注格式和PPOCRLabel不同
是单独的JSON文件标注，一张图片一个标注文件
随机划分训练、验证集，比例为 9:!
+++++++++++++++++++++++++++++++++++
"""

import os
import json
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

# 设置随机数种子
np.random.seed(667)
# 生成训练集、验证集索引
indices = np.arange(0, 30)
np.random.shuffle(indices)
train_indices = indices[:20]
val_indices = indices[20:]

"""
=====================
@@@ Config
=====================
"""

# COCO 格式的licenses、info键内容
licenses = [{"name": "", "id": 0, "url": ""}]
info = {
    "contributor": "",
    "date_created": "",
    "description": "",
    "url": "",
    "version": "",
    "year": "",
}

root_dir = Path(r"E:\Documents\datasets\ShanDong\ppdet")
output_dir = Path(r"E:\Documents\datasets\ShanDong\ppdet\demo_data_v3")
raw_dir = root_dir / "raw"
subset = ["train", "val"]  # 子集划分
train_size = 0.9  # 训练集比例
width = 3  # 图片名字长度

# 创建输出目录的相关文件夹
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
output_dir.mkdir(exist_ok=True)
(output_dir / "annotations").mkdir(exist_ok=True)
(output_dir / "visualize").mkdir(exist_ok=True)
for s in subset:
    (output_dir / s).mkdir(exist_ok=True)

# 标签处理
labels_dirs = ["police", "weapon", "qiang", "tanke"]  # 图片和Label.txt文件夹
label_nums = [-1, -1, -1, -1]  # -1表示选取全部图片
label_names = list(
    map(
        lambda i: i.strip().split("\t")[0],
        open(root_dir / "label_names.txt", encoding="utf-8").readlines(),
    )
)  # 读取标签
label_dict = dict(zip(label_names, range(1, len(label_names) + 1)))  # 标签->ID

# 生成COCO格式的"categories"键内容
categories = []
for name, id in label_dict.items():
    categories.append({"id": id, "name": name, "supercategory": ""})
print(categories)

"""
==================
@@@ GT 转换为 COCO
==================
"""

images, annotations = [], []
id, image_id = 0, 0

# 读取ppocrlabel的标注
total = 0
for label_num, label in zip(label_nums, labels_dirs):
    ppocr_labels = open(raw_dir / label / "Label.txt", encoding="utf-8").readlines()
    print(label, len(ppocr_labels))
    total += len(ppocr_labels)
    for i, line in enumerate(ppocr_labels):
        if label_num != -1 and i == label_num:
            break
        image_id += 1
        file_path, labels = line.strip().split("\t")  # 获取Label.txt的一行内容
        file_name = f"%0{width}d.jpg" % image_id  # 重新生成图片名字
        gts = json.loads(labels)  # 解析标注

        # 读取图片信息
        img = Image.open(root_dir / "raw" / file_path)
        w, h = img.size

        images.append(
            {
                "id": image_id,
                "width": w,
                "height": h,
                "file_name": file_name,
                "raw_file_name": os.path.basename(file_path),  # 原始图片路径
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "data_captured": 0,
            }
        )
        for gt in gts:
            id += 1
            # 获取标注信息
            category = gt["transcription"]
            points = np.array(gt["points"])
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]

            annotations.append(
                {
                    "id": id,
                    "image_id": image_id,
                    "category_id": label_dict[category],
                    "area": bbox[2] * bbox[3],
                    "bbox": bbox,
                    "iscrowd": 0,
                    "attributes": {"occluded": False, "rotation": 0.0},
                }
            )
print(total)

# 读取二维码图片的json标注
for i, anno_file in enumerate((root_dir / "raw/qr").glob("*.json")):
    image_id += 1
    with open(root_dir / "raw/qr" / anno_file) as f:
        gt = json.load(f)
    # 获取图片信息
    file_path = gt["imagePath"].name
    file_name = f"qr_{i + 1}.jpg"
    img = Image.open(root_dir / "raw/qr" / file_path)
    h, w = gt["imageHeight"], gt["imageWidth"]
    images.append(
        {
            "id": image_id,
            "width": w,
            "height": h,
            "file_name": file_name,
            "raw_file_name": file_path,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0,
        }
    )
    for anno in gt["shapes"]:
        id += 1
        category = anno["label"].lower()
        points = anno["points"]
        x_min, y_min = points[0]
        x_max, y_max = points[1]
        bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
        annotations.append(
            {
                "id": id,
                "image_id": image_id,
                "category_id": label_dict[category],
                "segmentations": [],
                "area": bbox[-1] * bbox[-2],
                "bbox": bbox,
                "iscrowd": 0,
                "attributes": {"occluded": False, "rotation": 0.0},
            }
        )

"""
==================
@@@ 保存 JSON 标注
==================
"""

indices = np.arange(0, len(images))
np.random.shuffle(indices)
train_indices = indices[: int(0.9 * len(indices))]
val_indices = indices[int(0.9 * len(indices)) :]

result = dict(
    zip(
        subset,
        [
            {
                "licenses": licenses,
                "info": info,
                "categories": categories,
                "images": [],
                "annotations": [],
            }
            for _ in range(len(subset))
        ],
    )
)

for idx in indices:
    image = images[idx]
    file_path = image.pop("raw_file_name")
    label_dir = file_path.split("_")[0]
    img = Image.open(root_dir / "raw" / label_dir / file_path).convert("RGB")
    if idx in train_indices:
        img.save(output_dir / "train" / image["file_name"])
        result["train"]["images"].append(image)
    else:
        img.save(output_dir / "val" / image["file_name"])
        result["val"]["images"].append(image)
    for anno in annotations:
        if anno["image_id"] == image["id"]:
            if idx in train_indices:
                result["train"]["annotations"].append(anno)
            else:
                result["val"]["annotations"].append(anno)

for s in subset:
    with open(output_dir / f"annotations/instances_{s}.json", "w+", encoding="utf-8") as f:
        json.dump(result[s], f, indent=4, ensure_ascii=False)

"""
=================
@@@ 可视化
=================
"""

for s in subset:
    images = result[s]["images"]
    annotations = result[s]["annotations"]
    for image in images:
        img = Image.open(output_dir / s / image["file_name"])
        img_draw = ImageDraw.Draw(img)
        for anno in annotations:
            if anno["image_id"] == image["id"]:
                color = tuple(np.random.randint(0, 256, 3))
                x, y, w, h = anno["bbox"]
                img_draw.line(
                    [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)],
                    fill=color,
                    width=2,
                )
        img.save(output_dir / "visualize" / image["file_name"])
        # break
    # break
