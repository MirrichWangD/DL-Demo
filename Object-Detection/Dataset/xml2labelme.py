# -*- encoding: utf-8 -*-",
"""
+++++++++++++++++++++++++++++++++++
@ File        : xml2labelme.py
@ Time        : 2023/1/30 10:36
@ Author      : Mirrich Wang
@ Version     : Python 3.x.x (env)
+++++++++++++++++++++++++++++++++++
XML格式（VOCData）转换为 LabelMe 格式
+++++++++++++++++++++++++++++++++++
"""

import os
import json
import xml.dom.minidom
from pathlib import Path
from tqdm import tqdm
from PIL import Image

"""
++++++++++++++
@@@ Config
++++++++++++++
"""

version = "5.1.1"  # 版本
# 路径操作
root_dir = Path(r"E:\Documents\datasets\sdznsh\ppdet\raw\pistol")
output_dir = root_dir
# (output_dir / "labels").mkdir(exist_ok=True)

"""
++++++++++++++++++
@@@ 遍历文件并转换
++++++++++++++++++
"""

files = list((root_dir / "xmls").glob("*"))

with tqdm(total=len(files)) as pbar:
    for file in files:
        dom = xml.dom.minidom.parse(str(file))
        # 得到文档元素对象
        root = dom.documentElement
        name = root.getElementsByTagName("filename")[0].firstChild.data.split(".")[0]
        w = root.getElementsByTagName("width")[0].firstChild.data.split(".")[0]
        h = root.getElementsByTagName("height")[0].firstChild.data.split(".")[0]
        # 获取目标物体
        data = root.getElementsByTagName("object")
        shapes = []
        for doc in data:
            # 获取标签
            label = doc.getElementsByTagName("name")[0].firstChild.data
            # 获取左上角、右下角坐标
            x1 = float(doc.getElementsByTagName("xmin")[0].firstChild.data)
            y1 = float(doc.getElementsByTagName("ymin")[0].firstChild.data)
            x2 = float(doc.getElementsByTagName("xmax")[0].firstChild.data)
            y2 = float(doc.getElementsByTagName("ymax")[0].firstChild.data)
            # 添加规范格式单条数据
            shapes.append(
                {
                    "label": label,
                    "points": [[x1, y1], [x2, y2]],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {},
                }
            )
        # 保存单张图片JSON文件
        with open(output_dir / f"{name}.json", "w+", encoding="utf-8") as f:
            json.dump(
                {
                    "version": version,
                    "flags": {},
                    "shapes": shapes,
                    "imagePath": f"{name}.jpg",
                    "imageData": None,
                    "imageHeight": h,
                    "imageWidth": w,
                },
                f,
                indent=4,
                ensure_ascii=False,
            )
        pbar.update(1)
        pbar.set_postfix_str(file.name)
