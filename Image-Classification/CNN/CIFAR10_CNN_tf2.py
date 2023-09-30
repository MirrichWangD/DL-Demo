# -*- coding: utf-8 -*-
"""
第06讲  卷积神经网络识别CIFAR-10图像
CIFAR-10数据集介绍
http://www.cs.toronto.edu/~kriz/cifar.html

@ Deep Learning Using Python
@ Recognize CIFAR-10 by CNN（卷积神经网络）
@ TensorFlow 2.0 版本
@ keras 看作 TensorFlow 的一个子模块
==============================================="""

# 导入需要的模块
# 导入TensorFlow模块，keras作为tensorflow的子模块：tf.keras
import time
import numpy as np
import tensorflow as tf

"""--------------------------
载入CIFAR10图像数据集与数据准备
-----------------------------"""
# 导入CIFAR10图像数据
cifar10 = tf.keras.datasets.cifar10
(x_img_train, y_label_train), (x_img_test, y_label_test) = cifar10.load_data()

# 查看数据维度
print("train data:", "images:", x_img_train.shape, " labels:", y_label_train.shape)
print("test  data:", "images:", x_img_test.shape, " labels:", y_label_test.shape)

# 定义标签字典（将文字映射到数字）
label_dict = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

# 查看前5个images
import matplotlib.pyplot as plt


def plot_images_labels_prediction(images, labels, prediction, idx, num=5):
    fig = plt.gcf()
    fig.set_size_inches(8, 10)
    if num > 25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(images[idx], cmap="binary")

        title = str(i) + "," + label_dict[labels[i][0]]
        if len(prediction) > 0:
            title += "=>" + label_dict[prediction[i]]

        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()


plot_images_labels_prediction(x_img_train, y_label_train, [], 0)

"""-------------------
数据标准化（归一化）
----------------------"""
x_img_train_normalize = x_img_train.astype("float32") / 255.0
x_img_test_normalize = x_img_test.astype("float32") / 255.0

# 标签映射（OneHot）
y_label_train_OneHot = tf.keras.utils.to_categorical(y_label_train)
y_label_test_OneHot = tf.keras.utils.to_categorical(y_label_test)

# 查看标签维度
y_label_test_OneHot.shape

"""----------------------------
@@@     建立卷积神经网络模型
----------------------------"""
model = tf.keras.Sequential()

"""
建立卷积层1与池化层1
加入Dropout（0.25）的功能是每次训练迭代时，
会随机地在神经网络中放弃25%的神经元，以免过度拟合。
"""

"""+++++++++++++++
卷积层1与池化层1
filters=32 设置随机产生32个滤镜（32个）
kernel_size=(3,3) 每个滤镜大小为3x3
input_shape=(32, 32,3) 代表图像大小为32x32，3代表彩色图像，代表三色RGB值
activation='relu'代表设置的激活函数：ReLU
padding='same'设置卷积运算产生的卷积图像大小不变
"""
model.add(
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        input_shape=(32, 32, 3),
        activation="relu",
        padding="same",
    )
)
# 避免过度拟合
model.add(tf.keras.layers.Dropout(rate=0.25))

"""池化层1
pool_size=(2, 2)执行第一次缩减采样，将32x32的图像缩小为16x16的图像
缩小后图像任然为32个。
"""
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

"""++++++++++++++++++
建立卷积层2与池化层2
卷积层2
执行第2次卷积运算，将原来32个图像转为64个图像，图片大小为16x16
"""
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"))

##避免过度拟合
model.add(tf.keras.layers.Dropout(0.25))

"""池化层2
pool_size=(2, 2)执行第一次缩减采样，将16x16的图像缩小为8x8的图像
缩小后图像为64个。
"""
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

"""+++++++++++++++++++++++++++++++++++++
建立神经网络(平坦层、隐藏层、输出层) """
# 平坦层
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(rate=0.25))

# 隐藏层（1024个神经元）
model.add(tf.keras.layers.Dense(1024, activation="relu"))
model.add(tf.keras.layers.Dropout(rate=0.25))

# 输出层（十种分类）
model.add(tf.keras.layers.Dense(10, activation="softmax"))

# 查看卷积神经网络摘要
print(model.summary())

"""------------------------------------------
  开始训练模型： 训练epochs=10周期 
  CPU需要17分钟
  GPU 30秒
---------------------------------------------"""
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

t1 = time.time()
train_history = model.fit(
    x_img_train_normalize,
    y_label_train_OneHot,
    validation_split=0.2,
    epochs=10,
    batch_size=128,
    verbose=1,
)
t2 = time.time()
CNNfit = float(t2 - t1)
print("Time taken: {} seconds".format(CNNfit))


# 描绘拟合曲线
def show_train_history(train_acc, test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title("Train History")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()


# show_train_history('acc','val_acc') 1.x
show_train_history("accuracy", "val_accuracy")
show_train_history("loss", "val_loss")

# 评估模型的准确率（测试集）:
# epochs=10周期，CPU运行，准确率：71.95%
# epochs=10周期，GPU运行 30秒，准确率：73.04%
# epochs=30周期，GPU运行 85秒，准确率：74.10%
# epochs=60周期，GPU运行 355秒，准确率：74.02%

scores = model.evaluate(x_img_test_normalize, y_label_test_OneHot, verbose=0)
scores[1]

# 预测
prediction = model.predict_classes(x_img_test_normalize)
prediction[:10]

# 查看预测结果
label_dict = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(8, 10)
    if num > 25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(images[idx], cmap="binary")

        title = str(i) + "," + label_dict[labels[i][0]]
        if len(prediction) > 0:
            title += "=>" + label_dict[prediction[i]]

        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()


plot_images_labels_prediction(x_img_test, y_label_test, prediction, 0, 10)

# 查看预测概率
Predicted_Probability = model.predict(x_img_test_normalize)


def show_Predicted_Probability(y, prediction, x_img, Predicted_Probability, i):
    print("label:", label_dict[y[i][0]], "predict:", label_dict[prediction[i]])
    plt.figure(figsize=(2, 2))
    plt.imshow(np.reshape(x_img_test[i], (32, 32, 3)))
    plt.show()
    for j in range(10):
        print(label_dict[j] + " Probability:%1.9f" % (Predicted_Probability[i][j]))


show_Predicted_Probability(y_label_test, prediction, x_img_test, Predicted_Probability, 0)

show_Predicted_Probability(y_label_test, prediction, x_img_test, Predicted_Probability, 3)

# 模糊矩阵 confusion matrix
prediction.shape
y_label_test.shape
y_label_test
y_label_test.reshape(-1)

import pandas as pd

print(label_dict)
pd.crosstab(y_label_test.reshape(-1), prediction, rownames=["label"], colnames=["predict"])

print(label_dict)

import os

os.makedirs("./out/Cifar10")
""" 保存模型的结构和权重"""
# Save model to JSON
model_json = model.to_json()
with open("./out/Cifar10/cifarCnnModelnew.json", "w+") as json_file:
    json_file.write(model_json)

## Tensorflow 2.8.0 had removed model.to_yaml()
# # Save Model to YAML
# model_yaml = model.to_yaml()
#
# with open("./out/18DL/SaveModel/cifarCnnModelnew.yaml", "w") as yaml_file:
#     yaml_file.write(model_yaml)


# Save Weight to h5
model.save_weights("./out/Cifar10/cifarCnnModelnew.h5")
print("Saved model to disk")
