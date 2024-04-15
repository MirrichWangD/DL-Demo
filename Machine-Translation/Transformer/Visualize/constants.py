# -*- coding: utf-8 -*-
"""
+++++++++++++++++++++++++++++++++++
@ File        : constants.py
@ Time        : 2024/4/15 上午3:34
@ Author      : Mirrich Wang
@ Version     : Python 3.x.x (Conda)
+++++++++++++++++++++++++++++++++++
...
+++++++++++++++++++++++++++++++++++
"""

# 定义特殊 TOKEN
PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]
# 定义 spacy 语言模型库，用于分词，该部分可以自行增加
# 注意！运行时请确保输入的 src_lang 和 tgt_lang 能够在此查询到相对应的 Spacy 语言模块，否则会构造数据集时报错
SPACY = {
    "de": "de_core_news_sm",  # German 德语
    "en": "en_core_web_sm",  # English 英语
    "zh": "zh_core_web_sm",  # Chinese 简中
    "fr": "fr_core_news_sm",  # French 法语
}
