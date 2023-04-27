# -*- coding: utf-8 -*-
"""
+++++++++++++++++++++++++++++++++++
@ File        : coco_concat.py
@ Time        : 2023/01/31 09:21:27
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (pyenv)
+++++++++++++++++++++++++++++++++++
拼接多个 COCO 格式数据集
特点：
    - 图片重命名
    - 按照类别标签字符串排序 -> 分配类别ID
+++++++++++++++++++++++++++++++++++
"""

import os
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from PIL import Image

# COCO相关基础变量
licenses = [{"name": "", "id": 0, "url": ""}]
info = {"contributor": "", "date_created": "", "description": "", "url": "", "version": "", "year": ""}
# 随机种子
np.random.seed(666)

"""
+++++++++++++++++
@@@ Config
+++++++++++++++++
"""

# 相关路径设置
root = Path(r"E:\Documents\datasets\sdznsh\ppdet")
raw_dir = "test_data_2"
output_dir = "demo_data_v6"
# 创建输出文件夹
(root / output_dir).mkdir(exist_ok=True)
(root / output_dir / "annotations").mkdir(exist_ok=True)

# 需要拼接的文件夹名字和子集
new_dirs = [
    {"new_dir": "demo_smoke", "subset": ["train2017", "val2017"], "num": [-1, -1], "shuffle": True},
    {"new_dir": "demo_logo", "subset": ["train2017", "val2017"], "num": [2000, 500], "shuffle": True},
]

# 集合名字
raw_subset = ["train", "val"]

# 格式化字符串变量
anno_file = "instances_%s.json"
image_file = "%06d.jpg"

anno_id, image_id = 0, 0
dst_image_id = defaultdict(dict)  # 图片原始ID -> 新ID字典
labels, category_ids = list(), dict()
images, annotations = defaultdict(list), defaultdict(list)

"""
+++++++++++++++++++
@@@ 拼接COCO格式标注
+++++++++++++++++++
"""

for s in raw_subset:
    (root / output_dir / s).mkdir(exist_ok=True)
    with open(root / raw_dir / "annotations" / (anno_file % s)) as f:
        instances = json.load(f)
    # 添加类别标签
    if s == "train":
        for category in instances["categories"]:
            labels.append(category["name"])
            category_ids[category["id"]] = category["name"]
    # 修改标注图片ID
    for image in instances["images"]:
        image_id += 1
        dst_image_id[raw_dir][image["id"]] = image_id
        image["id"] = image_id
        image["raw_file_name"] = f"{raw_dir}/{s}/" + image["file_name"]
        image["file_name"] = image_file % image_id
        image["width"] = int(image["width"])
        image["height"] = int(image["height"])
        images[s].append(image)
    # 修改标注ID
    for anno in instances["annotations"]:
        anno_id += 1
        anno["id"] = anno_id
        anno["image_id"] = dst_image_id[raw_dir][anno["image_id"]]
        anno["category_name"] = category_ids[anno["category_id"]]
        annotations[s].append(anno)

    print(f"[Done] raw {s}'s images: {len(images[s])}, annotations: {len(annotations[s])}")

# 处理待融合的新COCO格式数据集
for line in new_dirs:
    new_dir = line["new_dir"]
    subset = line["subset"]
    num = line["num"]
    is_shuffle = line["shuffle"]
    for i in range(len(subset)):
        # 读取COCO格式的标注
        with open(root / new_dir / "annotations" / (anno_file % subset[i])) as f:
            instances = json.load(f)
        print(
            f"[Doing] {new_dir}/{subset[i]}, images: {len(instances['images'])}, anotations: {len(instances['annotations'])}"
        )
        if i == 0:
            for category in instances["categories"]:
                labels.append(category["name"].replace("1", ""))
        # 处理"images"
        image_ids = []
        if is_shuffle:
            np.random.shuffle(instances["images"])
        for image in instances["images"]:
            image_ids.append(image["id"])
            image_id += 1
            dst_image_id[new_dir][image["id"]] = image_id
            image["id"] = image_id
            image["raw_file_name"] = f"{new_dir}/{subset[i]}/" + image["file_name"]
            image["file_name"] = image_file % image_id
            image["license"] = 0
            image["flickr_url"] = ""
            image["coco_url"] = ""
            image["data_captured"] = 0
            images[raw_subset[i]].append(image)
            if len(image_ids) >= num[i] != -1:
                break
        # 处理"annotations"
        for anno in instances["annotations"]:
            if anno["image_id"] in image_ids:
                anno_id += 1
                anno["id"] = anno_id
                anno["image_id"] = dst_image_id[new_dir][anno["image_id"]]
                anno["category_name"] = instances["categories"][0]["name"].replace("1", "")
                annotations[raw_subset[i]].append(anno)

label_idx = dict(zip(sorted(set(labels)), range(1, len(labels) + 1)))

# 转换标注的类别ID
print(f"[Doing] convert category id for annotations...")
for k, annotation in annotations.items():
    for anno in annotation:
        category_name = anno.pop("category_name")
        anno["category_id"] = label_idx[category_name]
    print(f"[Done] annotations: {k}, num: {len(annotation)}")

# 转换图片名字和ID
print(f"[Doing] convert image id for images...")
for k, image in images.items():
    for img in image:
        raw_file = img.pop("raw_file_name")
        img_raw = Image.open(root / raw_file).convert("RGB")
        img_raw.save(root / output_dir / k / img["file_name"])
        # break
    # break
    print(f"[Done] images: {k}, num: {len(image)}")

# 保存标注为JSON文件
print(f"[Doing] Saving JSON annotation files...")
# 制作COCO格式的标签，supercategory为空
categories = []
print("[Info] Categories:")
for label, idx in label_idx.items():
    print(f"{idx} - {label}")
    categories.append({"id": idx, "name": label, "supercategory": ""})
# 保存类别标签
with open(root / output_dir / "label_names.txt", "w+") as f:
    f.write("\n".join(sorted(labels)))
# 输出JSON文件
for s in raw_subset:
    with open(root / output_dir / f"annotations/instances_{s}.json", "w+") as f:
        json.dump(
            {
                "licenses": licenses,
                "info": info,
                "categories": categories,
                "images": images[s],
                "annotations": annotations[s],
            },
            f,
            indent=4,
            ensure_ascii=False,
        )
