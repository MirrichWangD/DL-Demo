import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from models.keras import alexnet

image_size = (224, 224)
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
# 导入CIFAR10图像数据
cifar10 = tf.keras.datasets.cifar10
(X_img_train, y_label_train), (X_img_test, y_label_test) = cifar10.load_data()

# 数据归一化
X_img_train_normalize = X_img_train.astype("float32") / 255.0
X_img_test_normalize = X_img_test.astype("float32") / 255.0

# 标签OneHot化
y_label_train_OneHot = tf.keras.utils.to_categorical(y_label_train)
y_label_test_OneHot = tf.keras.utils.to_categorical(y_label_test)

model = alexnet(num_classes=10)
model.build(input_shape=(None, 224, 224, 3))
model.summary(expand_nested=True)
# criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# epochs = 50
# model.compile(loss=criterion, optimizer="adam", metrics=["accuracy"])

# t1 = time.time()
# train_history = model.fit(X_img_train_normalize,
#                           y_label_train_OneHot,
#                           validation_split=0.2,
#                           epochs=epochs,
#                           batch_size=128,
#                           verbose=1)
# t2 = time.time()
# print("Time taken: {} seconds".format(float(t2 - t1)))
