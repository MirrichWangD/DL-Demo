# -*- encoding: utf-8 -*-",
"""
+++++++++++++++++++++++++++++++++++
@ File        : coco2mysql.py
@ Time        : 2023/3/2 11:16
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (Conda)
+++++++++++++++++++++++++++++++++++

+++++++++++++++++++++++++++++++++++
"""

from sqlalchemy import create_engine
from pathlib import Path
import pandas as pd
import json
import pymysql

# 数据集根目录
root = Path(r"E:\Pictures\datasets\projects\sdznsh\detection")
version = "v3"
subset = ["train", "val"]
engine = create_engine("mysql+pymysql://root:root@localhost:3306/sdznsh")

anns = list()

for s in subset:
    with open(root / f"annotations/{version}/instances_{s}.json", encoding="utf-8") as f:
        instances = json.load(f)
    categories = pd.DataFrame(instances["categories"])
    images = pd.DataFrame(instances["images"])
    annotations = pd.DataFrame(instances["annotations"])
    for ann in instances["annotations"]:
        x1, y1, w, h = ann["bbox"]
        x2 = x1 + w
        y2 = y1 + h
        anns.append(
            {
                "id": ann["id"],
                "filename": str(root / s / version / images[images["id"] == ann["image_id"]]["file_name"].item()),
                "points": ",".join([str(x1), str(y1), str(x2), str(y2)]),
                "label": categories[categories["id"] == ann["category_id"]]["name"].item(),
            }
        )
anns = pd.DataFrame(sorted(anns, key=lambda i: i["id"]))
anns.to_sql("test_detection", index=False, con=engine, if_exists="replace")

engine.connect()

data = pd.read_sql("SELECT * FROM test_detection", con=engine)
