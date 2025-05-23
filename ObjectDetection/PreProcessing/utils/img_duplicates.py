# -*- encoding: utf-8 -*-",
"""
+++++++++++++++++++++++++++++++++++
@ File        : img_duplicates.py
@ Time        : 2023/1/13 11:15
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (Conda)
+++++++++++++++++++++++++++++++++++
图像去重脚本，依据为MD5码
+++++++++++++++++++++++++++++++++++
"""

import os
import hashlib
from pathlib import Path
from tqdm import tqdm
from PIL import Image

"""
++++++++++++++++++
@@@ 自定义函数
++++++++++++++++++
"""


def get_md5(file):
    """获取文件md5码"""
    with open(file, "rb") as f:
        md5 = hashlib.md5(f.read())
    md5_values = md5.hexdigest()
    return md5_values


"""
+++++++++++++++
@@@ Config
+++++++++++++++
"""

# 图片根路径
root = Path(r"E:\Documents\datasets\sdznsh\ppdet\raw\bazooka")
# 输出路径
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
# 是否删除源图片
is_delete = False

"""
+++++++++++++++++
@@@ 图片名 -> MD5
+++++++++++++++++
"""

md5s = []
total = 0
files = list(root.glob("*"))
with tqdm(total=len(files)) as pbar:
    for file in files:
        md5 = get_md5(file)
        if md5 not in md5s:
            if os.path.splitext(file)[-1] not in [".jpg", ".png", ".jfif"]:
                pbar.update(1)
                continue
            img = Image.open(file).convert("RGB")
            img.save(output_dir / "{}.jpg".format(md5))
            md5s.append(md5)
        else:
            if is_delete:
                os.remove(file)
            total += 1
        pbar.update(1)
        pbar.set_postfix({"Total": total})
