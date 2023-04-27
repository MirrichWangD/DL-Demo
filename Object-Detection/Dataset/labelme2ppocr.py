# -*- encoding: utf-8 -*-",
"""
+++++++++++++++++++++++++++++++++++
@ File        : labelme2ppocr.py
@ Time        : 2023/1/13 10:34
@ Author      : Mirrich Wang
@ Version     : Python 3.x.x (env)
+++++++++++++++++++++++++++++++++++
QR 1994张数据集
json -> ppocr Label.txt
+++++++++++++++++++++++++++++++++++
"""

import os
import json
from PIL import Image
from pathlib import Path

root_dir = Path(r"E:\Documents\datasets\ShanDong\ppdet\raw\qr")
img_dir = root_dir / "imgs"
anno_dir = root_dir / "labels"

result = []
for i, anno_file in enumerate(anno_dir.glob("*.json")):
    with open(anno_file) as f:
        gts = json.load(f)
    file_name = f"qr_{i + 1}.jpg"
    img = Image.open(img_dir / gts["imagePath"])
    img.save(root_dir / file_name)
    annos = []
    for gt in gts["shapes"]:
        pt1, pt2 = gt["points"]
        annos.append({
            "transcription": "qr",
            "points": [[int(pt1[0]), int(pt1[1])],
                       [int(pt2[0]), int(pt1[1])],
                       [int(pt2[0]), int(pt2[1])],
                       [int(pt1[0]), int(pt2[1])]],
            "difficult": "false",
        })
    result.append(f"qr/{file_name}\t{json.dumps(annos)}\n")

with open(root_dir / "Label.txt", "w+", encoding="utf-8") as f:
    f.writelines(result)
