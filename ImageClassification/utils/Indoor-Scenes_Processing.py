# -*- encoding: utf-8 -*-",
"""
+++++++++++++++++++++++++++++++++++
@ File        : Indoor-Scenes_Processing.py
@ Time        : 2023/3/14 11:39
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (Conda)
+++++++++++++++++++++++++++++++++++

+++++++++++++++++++++++++++++++++++
"""

from sqlalchemy import create_engine
from pathlib import Path
import pandas as pd
import os

root = Path(r"E:\Pictures\datasets\Indoor-Scenes\raw\indoorCVPR_09\Images")
engine = create_engine("mysql+pymysql://root:root@localhost:3306/sdznsh")
result = list()
labels = sorted(os.listdir(root))

for i, img_path in enumerate(root.glob("*/*")):
    label = img_path.parts[-2]
    result.append({"id": i + 1, "filename": str(img_path), "label": label})

pd.DataFrame(result).to_sql("test_classification", index=False, con=engine, if_exists="replace")
