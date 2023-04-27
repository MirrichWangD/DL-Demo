# -*- encoding: utf-8 -*-",
"""
+++++++++++++++++++++++++++++++++++
@ File        : labelme2coco.py
@ Time        : 2023/1/30 9:53
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (Conda)
+++++++++++++++++++++++++++++++++++
labelme 标签格式转 COCO 格式
+++++++++++++++++++++++++++++++++++
"""

import os
import json
import shutil
import numpy as np
from collections import defaultdict
from pathlib import Path
from PIL import Image

"""
++++++++++++++
@@@ Config
++++++++++++++
"""

# COCO 格式的licenses、info键内容
licenses = [
    {
        "name": "",
        "id": 0,
        "url": ""
    }
]
info = {
    "contributor": "",
    "date_created": "",
    "description": "",
    "url": "",
    "version": "",
    "year": ""
}

subset = ["train", "val"]
train_size = .9

# 设置根目录和输出目录文件夹
root_dir = r"E:\Documents\datasets\sdznsh\ppdet\raw"
output_dir = r"E:\Documents\datasets\sdznsh\ppdet\test_data_2"

# 进行目录IO处理
root_dir = Path(root_dir)
output_dir = Path(output_dir)
# 创建输出目录
output_dir.mkdir(exist_ok=True)
(output_dir / "annotations").mkdir(exist_ok=True)
for s in subset:
    (output_dir / s).mkdir(exist_ok=True)

labels_dirs = ["bazooka", "police", "chariot", "gun", "knife", "pistol", "grenade", "warcraft", "warship"]
label_nums = [-1, -1, -1, -1, -1, -1, -1, -1, -1]
label_names = list(map(
    lambda i: i.strip().split("\t")[0], open(root_dir / ".." / "label_names.txt", encoding="utf-8").readlines())
)  # 读取标签
label_dict = dict(zip(sorted(label_names), range(1, len(label_names) + 1)))  # 标签->ID

# 生成COCO格式的"categories"键内容
print("[INFO] Categories:")
categories = []
for name, id in label_dict.items():
    print(f"{id} - {name}")
    categories.append({
        "id": id,
        "name": name,
        "supercategory": ""
    })

"""
==================
@@@ GT 转换为 COCO
==================
"""

images, annotations = defaultdict(list), defaultdict(list)
id, image_id = 0, 0

# 读取ppocrlabel的标注
for label_num, label_dir in zip(label_nums, labels_dirs):
    correct = 0

    label_files = list((root_dir / label_dir).glob("*.json"))
    num = len(label_files)

    indices = np.arange(0, num)
    np.random.shuffle(indices)
    train_indices = indices[:int(train_size * num)]
    val_indices = indices[int(train_size * num):]
    print(f"[INFO] {label_dir} Train: {train_indices.shape[0]}, Val: {val_indices.shape[0]}")

    for i, file in enumerate(label_files):
        if label_num != -1 and correct == label_num:
            break
        correct += 1
        image_id += 1
        with open(file) as f:
            labels = json.load(f)
        file_path = labels["imagePath"]
        file_name = f"%06d.jpg" % image_id
        h, w = int(labels["imageHeight"]), int(labels["imageWidth"])

        img = Image.open(root_dir / label_dir / os.path.basename(file_path)).convert("RGB")

        image = {
            "id": image_id,
            "width": w,
            "height": h,
            "file_name": file_name,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "data_captured": 0
        }

        anns = []
        for gt in labels["shapes"]:
            id += 1
            category = gt["label"]
            points = np.array(gt["points"])
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
            anns.append({
                "id": id,
                "image_id": image_id,
                "category_id": label_dict[category],
                "area": bbox[2] * bbox[3],
                "bbox": bbox,
                "iscrowd": 0,
                "attributes": {"occluded": False, "rotation": 0.}
            })

        if i in train_indices:
            img.save(output_dir / "train" / file_name)
            images["train"].append(image)
            annotations["train"].extend(anns)
        else:
            img.save(output_dir / "val" / file_name)
            images["val"].append(image)
            annotations["val"].extend(anns)


"""
==================
@@@ 保存 JSON 标注
==================
"""

for s in subset:
    with open(output_dir / f"annotations/instances_{s}.json", "w+", encoding="utf-8") as f:
        json.dump({
            "licenses": licenses,
            "info": info,
            "categories": categories,
            "images": images[s],
            "annotations": annotations[s]
        }, f, indent=4, ensure_ascii=False)
