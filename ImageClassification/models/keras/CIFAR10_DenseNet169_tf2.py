# -*- coding: utf-8 -*-

"""
第05讲 卷积神经网络识别CIFAR-10图像
DenseNet169 网络识别 CIFAR-10数据集
======================================="""

# 导入TensorFlow模块，keras作为tensorflow的子模块: tf.keras
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


""""@@@ 数据块 """
# 导入CIFAR10图像数据
cifar10 = tf.keras.datasets.cifar10
(X_img_train, y_label_train), (X_img_test, y_label_test) = cifar10.load_data()

# 数据归一化
X_img_train_normalize = X_img_train.astype("float32") / 255.0
X_img_test_normalize = X_img_test.astype("float32") / 255.0

# 标签OneHot化
y_label_train_OneHot = tf.keras.utils.to_categorical(y_label_train)
y_label_test_OneHot = tf.keras.utils.to_categorical(y_label_test)


""""@@@ 网络块 DenseNet169 """
# 建立卷积神经网络模型（DenseNet169）
# 注意：原输入224X224，现调整为：32X32。 输出为1000，调整为：10
num_layers = [6, 12, 32, 32]  # 卷积层个数
inputs = tf.keras.layers.Input([32, 32, 3])

# 卷积层与池化层
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding="same")(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding="same")(x)
# Block1
for i in range(num_layers[0]):
    if i == 0:
        x0 = x
        x1 = x
    else:
        x1 = tf.concat([x0, x], axis=-1)
    x = tf.keras.layers.BatchNormalization()(x1)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same")(x)
# 过渡层1
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding="same")(x)
x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2)(x)
# Block2
for i in range(num_layers[1]):
    if i == 0:
        x0 = x
        x1 = x
    else:
        x1 = tf.concat([x0, x], axis=-1)
    x = tf.keras.layers.BatchNormalization()(x1)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same")(x)
# 过渡层2
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), padding="same")(x)
x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2)(x)
# Block3
for i in range(num_layers[2]):
    if i == 0:
        x0 = x
        x1 = x
    else:
        x1 = tf.concat([x0, x], axis=-1)
    x = tf.keras.layers.BatchNormalization()(x1)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same")(x)
# 过渡层3
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), padding="same")(x)
x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2)(x)
# Block4
for i in range(num_layers[3]):
    if i == 0:
        x0 = x
        x1 = x
    else:
        x1 = tf.concat([x0, x], axis=-1)
    x = tf.keras.layers.BatchNormalization()(x1)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same")(x)


# 建立神经网络(平坦层、输出层)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
# 输出模型列表，需要调整近700万的参数，Total params: 6,944,906
print(model.summary())


""""@@@ 训练模型 """
epochs = 1  # 时间：1分钟  训练集准确率：36.15%   测试集准确率：19.17%
# epochs=50 #时间：分钟  训练集准确率：%   测试集准确率：%
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
CNNDenseNet169 = float(t2 - t1)
print("Time taken: {} seconds".format(CNNDenseNet169))


""""@@@ 评估模型的准确率 """


def show_train_history(train_acc, test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title("Train History")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()


show_train_history("accuracy", "val_accuracy")
show_train_history("loss", "val_loss")

# 测试集准确率：19.90%
scores = model.evaluate(X_img_test_normalize, y_label_test_OneHot, verbose=0)
scores[1]


""""@@@ 进行预测 """
prediction = model.predict(X_img_test_normalize)
prediction = np.argmax(prediction, axis=1)

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
    fig.set_size_inches(12, 14)
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


plot_images_labels_prediction(X_img_test_normalize, y_label_test, prediction, 0, 10)


# 查看预测概率
Predicted_Probability = model.predict(X_img_test_normalize)


def show_Predicted_Probability(X_img, Predicted_Probability, i):
    plt.figure(figsize=(2, 2))
    plt.imshow(np.reshape(X_img_test[i], (32, 32, 3)))
    plt.show()
    for j in range(10):
        print(label_dict[j] + " Probability:%1.9f" % (Predicted_Probability[i][j]))


show_Predicted_Probability(X_img_test, Predicted_Probability, 0)


""""@@@ save model 储存模型结构和权重 """
model_json = model.to_json()
open("./out/Cifar10/DenseNet169_architecture.json", "w").write(model_json)
model.save_weights("./out/Cifar10/DenseNet169_weights.h5", overwrite=True)
