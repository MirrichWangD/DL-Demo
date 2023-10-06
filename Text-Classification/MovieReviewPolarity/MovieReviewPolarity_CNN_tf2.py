# -*- coding: utf-8 -*-
"""
+++++++++++++++++++++++++++++++++++
@ File        : MovieReviewPolarity_CNN_tf2.py
@ Time        : 2022/4/25 10:47
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (Conda)
+++++++++++++++++++++++++++++++++++
use polarity MovieReview dataset to CNN
+++++++++++++++++++++++++++++++++++
"""


import re
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer


# 使用read_files()函数读取ReviewPolarity下的txt_sentoken文件目录
def read_files():
    path = "./data/ReviewPolarity/txt_sentoken"
    file_list = []

    positive_path = path + "/pos/"
    for f in os.listdir(positive_path):
        file_list += [positive_path + f]

    negative_path = path + "/neg/"
    for f in os.listdir(negative_path):
        file_list += [negative_path + f]

    print("read all files:", len(file_list))

    all_labels = [1] * len(os.listdir(positive_path)) + [0] * len(os.listdir(negative_path))

    all_texts = []
    for fi in file_list:
        with open(fi, encoding="utf8") as file_input:
            all_texts += [" ".join([i.rstrip("\n") for i in file_input.readlines()])]

    return all_labels, all_texts


labels, texts = read_files()
data = list(zip(texts, labels))
test_size = 0.2
train_text, test_text, y_train, y_test = train_test_split(texts, labels, test_size=test_size, random_state=666)


# 导入的数据
# y_train, train_text = read_files("train")
# y_test, test_text = read_files("test")

print(len(train_text))
print(len(test_text))

# 建立token，即用训练的1800评价文字产生一个字典，
# 只取排序后的前10000名英文单词进入字典（完整为41394）
token = Tokenizer()
token.fit_on_texts(train_text)

# 使用token字典，将“影评文字”转为“数字列表”
# 将每一篇文章的文字转换一连串的数字，只有在字典中的文字才会被转换为数字
x_train_seq = token.texts_to_sequences(train_text)
x_test_seq = token.texts_to_sequences(test_text)

# 让转换后的数字长度相同400
x_train = sequence.pad_sequences(x_train_seq, maxlen=400)
x_test = sequence.pad_sequences(x_test_seq, maxlen=400)

# vocab_size大约为4000  这里如果采用自动导入数据，有88587（可以用num_words =4000）
vocab_size = np.max([np.max(x_train[i]) for i in range(x_train.shape[0])]) + 1

"""----------------------+
# 建立卷积神经网络CNN模型
-------------------------+
"""

model = tf.keras.Sequential()

# 嵌入层 (字典vocab_size=4000，另外将每一个数字映射到64维的向量空间中去)
model.add(tf.keras.layers.Embedding(vocab_size, 64, input_length=400))

# 卷积层1，池tf.keras.layers.化层1，剪枝25%
model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.25))

# 卷积层2，池化层2，剪枝25%
model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.25))

# 平坦层（输入层）
model.add(tf.keras.layers.Flatten())

# 隐藏层1，有64个神经元
model.add(tf.keras.layers.Dense(64, activation="relu"))

# 隐藏层2，有32个神经元
model.add(tf.keras.layers.Dense(32, activation="relu"))

# 输出层，一个神经元，用Sigmoid函数作激活函数，预测 0，1变量的概率。
# 最后输出 0或者1的概率。
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

# 模型框架摘要
model.summary()

# # 训练模型 epochs=10
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 训练前，要将list：y_train和y_test转化为numpy.ndarray
type(x_train)
type(y_train)
y_train = np.array(y_train)

type(x_test)
type(y_test)
y_test = np.array(y_test)

# 开始训练
import time

t1 = time.time()
train_history = model.fit(x_train, y_train, batch_size=256, epochs=100, verbose=2, validation_split=0.2)
t2 = time.time()
CNNReviewPolarity = float(t2 - t1)
print("Time taken: {} seconds".format(CNNReviewPolarity))


# 可视化准确率与损失值
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title("Train History")
    plt.ylabel(train)
    plt.xlabel("Epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


show_train_history(train_history, "accuracy", "val_accuracy")
show_train_history(train_history, "loss", "val_loss")

# # 评估模型的准确率
#  epochs=10：  CPU：秒  %    GPU 秒  %
#  epochs=30：  GPU 8.2869秒  100%
# 15/15 - 0s - loss: 3.7722e-06 - accuracy: 1.0000
# - val_loss: 2.0682 - val_accuracy: 0.7222
scores = model.evaluate(x_test, y_test, verbose=1)
print(scores[1])

# # 预测概率
probility = model.predict(x_test)
print(probility[:10])

for p in probility[100:110]:
    print(p)

# # 预测结果
predict = np.argmax(model.predict(x_test), axis=1)
print(predict[:10])

predict_classes = predict.reshape(-1)
print(predict_classes[:10])

# # 创建一个函数，查看预测结果
SentimentDict = {1: "正面的", 0: "负面的"}


def display_test_Sentiment(i):
    print(test_text[i][:1000], "...")
    print("标签label:", SentimentDict[y_test[i]], "预测结果:", SentimentDict[predict_classes[i]])


display_test_Sentiment(2)
print(predict_classes[100:110])
display_test_Sentiment(102)
display_test_Sentiment(104)
