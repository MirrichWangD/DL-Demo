# -*- encoding: utf-8 -*-
"""
+++++++++++++++++++++++++++++++++++
@ File        : MNIST_LeNet-5_tf2.py
@ Time        : 2022/11/09 15:16:58
@ Author      : Mirrich Wang
@ Version     : Python 3.7.9 (env)
+++++++++++++++++++++++++++++++++++
TensorFlow 框架的 CNN 使用 MNIST 手写数字进行训练
+++++++++++++++++++++++++++++++++++
"""

# 导入基础模块
import time
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# 导入可视化模块
import matplotlib.pyplot as plt


# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
"""
    默认为0：输出所有log信息
    设置为1：进一步屏蔽INFO信息
    设置为2：进一步屏蔽WARNING信息
    设置为3：进一步屏蔽ERROR信息'
"""

"""
==================
@@@ 数据加载
==================
"""


# mnist = tf.keras.datasets.mnist()
# mnist = Mnist(root='../../datasets/MNIST', vis_sample=True)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape)
# print(x_test.shape)
x_train = x_train.reshape((60000, 28, 28, 1)) / 255
x_test = x_test.reshape((10000, 28, 28, 1)) / 255
y_train = tf.keras.utils.to_categorical(y_train)  # 转one-hot
y_test = tf.keras.utils.to_categorical(y_test)  # 转one-hot
# 初始化模型训练的参数
epochs = 32  # 训练批次
batch_size = 128  # 数据分批数量
val_size = 0.2  # 验证集比例

"""
==================
@@@ 模型搭建
==================
"""

LeNet5 = tf.keras.Sequential(
    [  # 网络容器
        tf.keras.layers.Conv2D(6, kernel_size=3, strides=1),  # 第一个卷积层, 6 个 3x3 卷积核
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),  # 高宽各减半的池化层
        tf.keras.layers.ReLU(),  # 激活函数
        tf.keras.layers.Conv2D(16, kernel_size=3, strides=1),  # 第二个卷积层, 16 个 3x3 卷积核
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),  # 高宽各减半的池化层
        tf.keras.layers.ReLU(),  # 激活函数
        tf.keras.layers.Flatten(),  # 打平层，方便全连接层处理
        tf.keras.layers.Dense(120, activation="relu"),  # 全连接层，120 个节点
        tf.keras.layers.Dense(84, activation="relu"),  # 全连接层，84 节点
        tf.keras.layers.Dense(10),  # 全连接层，10 个节点
    ]
)

LeNet5.build(input_shape=(None, 28, 28, 1))  # 构造网络
LeNet5.summary()  # 打印网络的每层参数

optimizer = tf.keras.optimizers.Adam()  # 采用Adam方法
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
LeNet5.compile(loss=loss_object, optimizer=optimizer, metrics=["accuracy"])  # 模型设置损失函数和调试器

"""
=================
@@@ 模型训练和验证
=================
"""

t1 = time.time()
train_history = LeNet5.fit(x=x_train, y=y_train, validation_split=val_size, epochs=epochs, batch_size=512)  # 模型训练
t2 = time.time()
print(f"Time taken:{t2 - t1}s")
LeNet5.evaluate(x_test, y_test)

"""
==================
@@@ 训练过程可视化
==================
"""


def plot_train_history(history_dict):
    plt.plot(history_dict["accuracy"])
    plt.plot(history_dict["val_accuracy"])
    plt.title("Accuracy")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend(["train accuracy", "val accuracy"], loc="lower right")
    plt.show()

    plt.plot(history_dict["loss"])
    plt.plot(history_dict["val_loss"])
    plt.title("Loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(["train loss", "val loss"], loc="upper right")
    plt.show()


plot_train_history(train_history.history)
