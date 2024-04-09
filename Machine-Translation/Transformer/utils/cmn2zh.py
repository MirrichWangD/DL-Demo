# -*- coding: UTF-8 -*-
"""
+++++++++++++++++++++++++++++++++++
@ File        : cmn2zh.py
@ Time        : 2023/4/26 17:17
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (Conda)
+++++++++++++++++++++++++++++++++++
繁体中文转简体程序
+++++++++++++++++++++++++++++++++++
"""

import opencc
import pandas as pd

cc = opencc.OpenCC("t2s")
data = pd.read_table("../Tab-Separator/data/eng-cmn.txt", header=None)
data[1] = data[1].apply(lambda i: cc.convert(i))
data.to_csv("../Tab-Separator/data/eng-zh.txt", header=None, index=False, sep="\t")
