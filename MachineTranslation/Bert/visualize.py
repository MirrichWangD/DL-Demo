# -*- coding: utf-8 -*-
"""
+++++++++++++++++++++++++++++++++++
@ File        : visualize.py
@ Time        : 2024/4/15 下午6:56
@ Author      : Mirrich Wang
@ Version     : Python 3.x.x (Conda)
+++++++++++++++++++++++++++++++++++
...
+++++++++++++++++++++++++++++++++++
"""


import os.path

from matplotlib import pyplot as plt
from transformers import AutoTokenizer, TFAutoModel
import numpy as np
import tensorflow as tf
import pandas as pd

# pip install seaborn -i https://pypi.tuna.tsinghua.edu.cn/simple
import seaborn as sns

# Token化工具
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# BERT模型，并加载预训练权重
model = TFAutoModel.from_pretrained("bert-base-uncased")

# 数据输入 -> Token化 -> 模型输入
sentence = "How Attention works in Deep Learning: understanding the attention mechanism in sequence models"
inputs = tokenizer(sentence, return_tensors="tf")

# 经过BERT模型，进行Attention计算
outputs = model(**inputs, output_attentions=True)


# 打印
def attention_plot(
    attention,
    x_texts,
    y_texts=None,
    figsize=(15, 10),
    annot=False,
    figure_path="./figures",
    figure_name="attention_weight.png",
):
    plt.clf()
    fig, ax = plt.subplots(figsize=figsize)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(
        attention,
        cbar=True,
        cmap="RdBu_r",
        annot=annot,
        square=True,
        fmt=".2f",
        annot_kws={"size": 10},
        yticklabels=y_texts,
        xticklabels=x_texts,
        xtick_rotation=90,
    )
    if os.path.exists(figure_path) is False:
        os.makedirs(figure_path)
    plt.savefig(os.path.join(figure_path, figure_name))
    plt.close()


# 模型输出：
print("模型输出： ")
print("outputs:", list(outputs.keys()))
for k, v in outputs.items():

    if k == "attentions":
        # Token ids
        ids = np.array(inputs["input_ids"][0], dtype=np.int)

        # 明字
        texts = tokenizer.convert_ids_to_tokens(ids)

        # [Batch_Size, Heads, Sequence_Length, Sequence_Length]
        attentions = v[-1]
        heads = attentions.shape[1]

        for ii in range(heads):
            # 显示第ii个Head的Attention
            attention_plot(
                attentions[0, ii, :, :],
                annot=True,
                x_texts=texts,
                y_texts=texts,
                figsize=(15, 15),
                figure_path="./figures",
                figure_name="bert_attention_weight_head_{}.png".format(ii + 1),
            )

            # Attention 归一化
            attentions_norm = tf.math.l2_normalize(attentions, axis=-1)
            attention_plot(
                attentions_norm[0, ii, :, :],
                x_texts=texts,
                y_texts=texts,
                annot=True,
                figsize=(15, 15),
                figure_path="./figures",
                figure_name="bert_attention_weight_head_{}_norm.png".format(ii + 1),
            )

            # 显示第ii个Head的Attention，除了SEP
            attention_plot(
                attentions[0, ii, :-1, :-1],
                annot=True,
                x_texts=texts[:-1],
                y_texts=texts[:-1],
                figsize=(15, 15),
                figure_path="./figures",
                figure_name="bert_attention_weight_head_{}_no_SEP.png".format(ii + 1),
            )

            # Attention 归一化，除了SEP
            attentions_norm = tf.math.l2_normalize(attentions[:, :, :-1, :-1], axis=-1)
            attention_plot(
                attentions_norm[0, ii, :, :],
                annot=True,
                x_texts=texts[:-1],
                y_texts=texts[:-1],
                figsize=(15, 15),
                figure_path="./figures",
                figure_name="bert_attention_weight_head_{}_no_SEP_norm.png".format(ii + 1),
            )

            # ==============================================================
            # SUM
            # ==============================================================
            # 按Heads取和
            attention_sum = tf.reduce_sum(attentions, axis=1)

            # 显示Attention
            attention_plot(
                attention_sum[0, :, :],
                annot=True,
                x_texts=texts,
                y_texts=texts,
                figsize=(15, 15),
                figure_path="./figures",
                figure_name="bert_attention_weight_sum.png",
            )

            # Attention 归一化
            attentions_norm = tf.math.l2_normalize(attention_sum, axis=-1)
            attention_plot(
                attentions_norm[0, :, :],
                x_texts=texts,
                y_texts=texts,
                annot=True,
                figsize=(15, 15),
                figure_path="./figures",
                figure_name="bert_attention_weight_sum_norm.png",
            )

            # 显示Attention, 除了SEP
            attention_plot(
                attention_sum[0, :-1, :-1],
                annot=True,
                x_texts=texts[:-1],
                y_texts=texts[:-1],
                figsize=(15, 15),
                figure_path="./figures",
                figure_name="bert_attention_weight_sum_no_SEP.png",
            )

            # Attention 归一化
            attentions_norm = tf.math.l2_normalize(attention_sum[:, :-1, :-1], axis=-1)
            attention_plot(
                attentions_norm[0, :, :],
                x_texts=texts[:-1],
                y_texts=texts[:-1],
                annot=True,
                figsize=(15, 15),
                figure_path="./figures",
                figure_name="bert_attention_weight_sum_no_SEP_norm.png",
            )

            # ==============================================================
            # MEAN
            # ==============================================================
            # 按Heads取均值
            attention_mean = tf.reduce_mean(attentions, axis=1)

            # 显示Attention
            attention_plot(
                attention_mean[0, :, :],
                annot=True,
                x_texts=texts,
                y_texts=texts,
                figsize=(15, 15),
                figure_path="./figures",
                figure_name="bert_attention_weight_mean.png",
            )

            # Attention 归一化
            attentions_norm = tf.math.l2_normalize(attention_mean, axis=-1)
            attention_plot(
                attentions_norm[0, :, :],
                x_texts=texts,
                y_texts=texts,
                annot=True,
                figsize=(15, 15),
                figure_path="./figures",
                figure_name="bert_attention_weight_mean_norm.png",
            )

            # 显示Attention, 除了SEP
            attention_plot(
                attention_mean[0, :-1, :-1],
                annot=True,
                x_texts=texts[:-1],
                y_texts=texts[:-1],
                figsize=(15, 15),
                figure_path="./figures",
                figure_name="bert_attention_weight_mean_no_SEP.png",
            )

            # Attention 归一化
            attentions_norm = tf.math.l2_normalize(attention_mean[:, :-1, :-1], axis=-1)
            attention_plot(
                attentions_norm[0, :, :],
                x_texts=texts[:-1],
                y_texts=texts[:-1],
                annot=True,
                figsize=(15, 15),
                figure_path="./figures",
                figure_name="bert_attention_weight_mean_no_SEP_norm.png",
            )
