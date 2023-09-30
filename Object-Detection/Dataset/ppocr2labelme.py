# -*- encoding: utf-8 -*-",
"""
+++++++++++++++++++++++++++++++++++
@ File        : ppocr2labelme.py
@ Time        : 2023/1/17 15:13
@ Author      : Mirrich Wang
@ Version     : Python 3.x.x (env)
+++++++++++++++++++++++++++++++++++

+++++++++++++++++++++++++++++++++++
"""

import os
import json
import numpy as np
from PIL import Image
from pathlib import Path

version = "5.1.1"  # labelme 版本
label_dirs = ["police", "gun", "chariot", "weapon"]

for label_dir in label_dirs:
    root_dir = Path(rf"E:\Documents\datasets\ShanDong\ppdet\raw\{label_dir}")

    (root_dir / "labels").mkdir(exist_ok=True)

    ppocr_labels = list(map(lambda i: i.strip().split("\t"), open(root_dir / "Label.txt").readlines()))

    for file, annos in ppocr_labels:
        file_name = os.path.splitext(os.path.basename(file))[0]
        img = Image.open(root_dir / f"{file_name}.jpg")
        w, h = img.size
        annos = json.loads(annos)
        shapes = []
        for anno in annos:
            points = np.array(anno["points"])
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            shapes.append(
                {
                    "label": anno["transcription"],
                    "points": [
                        [float(x_min), float(y_min)],
                        [float(x_max), float(y_max)],
                    ],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {},
                }
            )
        with open(root_dir / "labels" / f"{file_name}.json", "w+", encoding="utf-8") as f:
            json.dump(
                {
                    "version": version,
                    "flags": {},
                    "shapes": shapes,
                    "imagePath": os.path.join("../../../../../scripts", f"{file_name}.jpg"),
                    "imageData": None,
                    "imageHeight": h,
                    "imageWidth": w,
                },
                f,
                indent=4,
                ensure_ascii=False,
            )
        # break
