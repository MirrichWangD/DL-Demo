# -*- coding: utf-8 -*-
"""
    @Author        ：Mirrich Wang
    @Created       ：2022/5/16 11:18
    @Description   ：
"""

import time
import numpy as np
import tensorflow as tf

cifar100 = tf.keras.datasets.cifar100
(X_img_train, y_label_train), (X_img_test, y_label_test) = cifar100.load_data()

# 数据归一化
X_img_train_normalize = X_img_train.astype("float32") / 255.0
X_img_test_normalize = X_img_test.astype("float32") / 255.0

# 标签OneHot化
y_label_train_OneHot = tf.keras.utils.to_categorical(y_label_train)
y_label_test_OneHot = tf.keras.utils.to_categorical(y_label_test)

""""@@@ 网络块 ResNet152 """
# 建立卷积神经网络模型（ResNet152）
# 注意：原输入224X224，现调整为：32X32。 输出为1000，调整为：10
num_layers = [3, 8, 36, 3]  # 卷积层个数
inputs = tf.keras.layers.Input([32, 32, 3])

# Block1
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding="same")(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding="same")(x)
# Block2
for i in range(num_layers[0]):
    x0 = x
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    if i == 0:
        x0 = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=1)(x0)
        x0 = tf.keras.layers.BatchNormalization()(x0)
    x += x0
# Block3
for i in range(num_layers[1]):
    x0 = x
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    if i == 0:
        x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=2, padding="same")(x)
    else:
        x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    if i == 0:
        x0 = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), strides=2)(x0)
        x0 = tf.keras.layers.BatchNormalization()(x0)
    x += x0
# Block4
for i in range(num_layers[2]):
    x0 = x
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    if i == 0:
        x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding="same")(x)
    else:
        x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=(1, 1), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    if i == 0:
        x0 = tf.keras.layers.Conv2D(filters=1024, kernel_size=(1, 1), strides=2)(x0)
        x0 = tf.keras.layers.BatchNormalization()(x0)
    x += x0
# Block5
for i in range(num_layers[3]):
    x0 = x
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    if i == 0:
        x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=2, padding="same")(x)
    else:
        x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=2048, kernel_size=(1, 1), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    if i == 0:
        x0 = tf.keras.layers.Conv2D(filters=2048, kernel_size=(1, 1), strides=2)(x0)
        x0 = tf.keras.layers.BatchNormalization()(x0)
    x += x0

# 建立神经网络(平坦层、输出层)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(100, activation="softmax")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
# 输出模型列表，需要调整近6000万的参数，Total params: 58,391,434
print(model.summary())

# epochs = 1  # 时间：2分钟  训练集准确率：29.38%   测试集准确率：13.28%
epochs = 50  # 时间：分钟  训练集准确率：%   测试集准确率：%
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

t1 = time.time()
train_history = model.fit(
    X_img_train_normalize,
    y_label_train_OneHot,
    validation_split=0.2,
    epochs=epochs,
    batch_size=128,
    verbose=1,
)
t2 = time.time()
CNNResNet152 = float(t2 - t1)
print("Time taken: {} seconds".format(CNNResNet152))
