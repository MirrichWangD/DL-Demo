# -*- coding: UTF-8 -*-
"""
+++++++++++++++++++++++++++++++++++
@ File        : cmn-zh.py
@ Time        : 2023/4/26 17:17
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (Conda)
+++++++++++++++++++++++++++++++++++

+++++++++++++++++++++++++++++++++++
"""

import opencc
import pandas as pd

cc = opencc.OpenCC("t2s")
data = pd.read_table("../data/cmn-eng.txt", header=None)
data[1] = data[1].apply(lambda i: cc.convert(i))
data.to_csv("../data/eng-zh.txt", header=None, index=False, sep="\t")
