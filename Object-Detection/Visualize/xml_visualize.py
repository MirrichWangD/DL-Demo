# -*- encoding: utf-8 -*-",
"""
+++++++++++++++++++++++++++++++++++
@ File        : xml_visualize.py
@ Time        : 2023/2/16 14:24
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (Conda)
+++++++++++++++++++++++++++++++++++

+++++++++++++++++++++++++++++++++++
"""

from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw
import os
import xml.dom.minidom
import warnings
import numpy as np
import seaborn as sns

warnings.filterwarnings("ignore")

"""
+++++++++++++++++++++++++
@@@ Config
+++++++++++++++++++++++++
"""

colors = np.array(sns.color_palette(n_colors=11)) * 255
root_dir = Path(r"D:\Downloads\抽烟打电话喝水\train\smoke1")
output_dir = root_dir / "visualize"
output_dir.mkdir(exist_ok=True)

"""
+++++++++++++++++++++++
@@@ 根据XML文件可视化图片
+++++++++++++++++++++++
"""

files = list((root_dir / "outputs").glob("*"))

with tqdm(total=len(files)) as pbar:
    for file in files:
        dom = xml.dom.minidom.parse(str(file))
        # 得到文档元素对象
        root = dom.documentElement
        path = os.path.basename(root.getElementsByTagName('path')[0].firstChild.data)
        # 获取目标物体
        data = root.getElementsByTagName('item')
        try:
            img = Image.open(root_dir / path).convert("RGB")
            img_draw = ImageDraw.Draw(img)
        except:
            continue
        for doc in data:
            # 获取标签
            label = doc.getElementsByTagName('name')[0].firstChild.data
            # 获取左上角、右下角坐标
            x1 = float(doc.getElementsByTagName('xmin')[0].firstChild.data)
            y1 = float(doc.getElementsByTagName('ymin')[0].firstChild.data)
            x2 = float(doc.getElementsByTagName('xmax')[0].firstChild.data)
            y2 = float(doc.getElementsByTagName('ymax')[0].firstChild.data)

            color = tuple(np.random.randint(0, 256, 3, dtype=np.uint8).tolist())
            img_draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
            # 绘制文本文本框
            tw, th = img_draw.textsize(label)
            img_draw.rectangle((x1 + 1, y1 - th, x1 + tw + 1, y1), fill=color)
            img_draw.text((x1 + 1, y1 - th), label, fill=(255, 255, 255))
        img.save(output_dir / path)
        pbar.update(1)
