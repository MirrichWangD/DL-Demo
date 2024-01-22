# -*- coding: utf-8 -*-
"""
+++++++++++++++++++++++++++++++++++
@ File        : coco_visualize.py
@ Time        : 2023/02/01 09:15:45
@ Author      : Mirrich Wang
@ Version     : Python 3.x.x (env)
+++++++++++++++++++++++++++++++++++
根据COCO标注绘制边界框
+++++++++++++++++++++++++++++++++++
"""

import json
import warnings
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageDraw

warnings.filterwarnings("ignore")

"""
++++++++++++++++++
@@@ Config
++++++++++++++++++
"""

colors = np.array(sns.color_palette(n_colors=11)) * 255
subset = ["Train", "Eval", "Test"]
# 目录操作
root = Path(r"D:\Workspaces\projects\app_aiam_sdznsh2023\ai\ai_core\train_manage\datasets\det\test_detection")
(root / "visualize").mkdir(exist_ok=True)

"""
++++++++++++++++
@@@ 可视化操作
++++++++++++++++
"""

label_idx = dict()
for s in subset:
    (root / "visualize" / s).mkdir(exist_ok=True)
    with open(root / "annotations" / f"instances_{s}.json") as f:
        instances = json.load(f)
    # 获取标注中的标签类别、图片信息、标注信息
    categories = instances["categories"]
    images = instances["images"]
    annotations = instances["annotations"]

    if s == subset[0]:
        for category in categories:
            print("{} - {}".format(category["id"], category["name"]))
            label_idx[category["id"]] = category["name"]

    # 统计标注中所有标签的数量
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.countplot(x="category_id", data=pd.DataFrame(annotations))
    for p in ax.patches:
        ax.annotate(
            f"\n{p.get_height():.0f}",
            (p.get_x() + 0.2, p.get_height() + 8),
            color="black",
            size=8,
        )
    plt.title(f"{root.name} - {s}")
    plt.xticks(range(len(label_idx)), list(label_idx.values()), rotation=15)
    plt.savefig(root / f"visualize/{s}.jpg")

    with tqdm(total=len(images), desc=f"{s}, annotations: {len(annotations)}") as pbar:
        for i, image in enumerate(images):
            # 获取图片信息
            image_id = image["id"]
            file_name = image["file_name"]
            img = Image.open(root / s / file_name).convert("RGB")
            img_draw = ImageDraw.Draw(img)
            # 绘制标注的边界框和标签类别
            for anno in annotations:
                if anno["image_id"] == image_id:
                    # 获取标注信息
                    category_id = anno["category_id"]
                    label = label_idx[category_id]
                    x, y, w, h = anno["bbox"]
                    # 绘制矩形框
                    color = tuple(np.uint8(colors[category_id - 1]))
                    img_draw.rectangle((x, y, x + w, y + h), outline=color, width=2)
                    # 绘制文本文本框
                    tw, th = img_draw.textsize(label)
                    img_draw.rectangle((x + 1, y - th, x + tw + 1, y), fill=color)
                    img_draw.text((x + 1, y - th), label, fill=(255, 255, 255))

            img.save(root / "visualize" / s / file_name)
            pbar.update(1)
