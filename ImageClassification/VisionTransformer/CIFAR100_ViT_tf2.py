# -*- coding: utf-8 -*-
"""
    @Author        ：Mirrich Wang
    @Created       ：2022/5/15 16:39
    @Description   ：
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

"""+++++++++++++++++++++++++++++
    @ Settings
++++++++++++++++++++++++++++++"""

epochs = 50  # 训练次数
batch_size = 64  # 训练批次大小
image_size = 72  # 图形改变大小
patch_size = 16  # 输入图片拆分的块大小
num_patches = (image_size // patch_size) ** 2  # 拆分的块数量（14*14 = 196）
projection_dim = 64  # 向量长度
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Transformer layers的大小
transformer_layers = 8
mlp_head_units = [2048, 1024]  # 输出部分的MLP全连接层的大小

"""+++++++++++++++++++++++++++++
    @ 读取数据
++++++++++++++++++++++++++++++"""

# 类别数
num_classes = 100
# 数据大小
input_shape = (32, 32, 3)
# 读取cifar100数据集
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

print(f"x_train 大小: {x_train.shape} - y_train 大小: {y_train.shape}")
print(f"x_test 大小: {x_test.shape} - y_test 大小: {y_test.shape}")

"""+++++++++++++++++++++++++++++
    @ 读取数据
++++++++++++++++++++++++++++++"""
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 64  # 训练批次大小
num_epochs = 100  # 训练周期
image_size = 224  # 改变图形大小
patch_size = 16  # 输入图片拆分的块大小
num_patches = (image_size // patch_size) ** 2  # 拆分的块数量
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Transformer layers的大小
transformer_layers = 8
mlp_head_units = [2048, 1024]  # 输出部分的MLP全连接层的大小

# 图像增强
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.Normalization(),
        layers.experimental.preprocessing.Resizing(image_size, image_size),
        # layers.experimental.preprocessing.RandomFlip("horizontal"),
        # layers.experimental.preprocessing.RandomRotation(factor=0.02),
        # layers.experimental.preprocessing.RandomZoom(
        #     height_factor=0.2, width_factor=0.2
        # ),
    ],
    name="data_augmentation",
)


def mlp(x, hidden_units, dropout_rate):
    model = tf.keras.Sequential()
    for units in hidden_units:
        model.add(layers.Dense(units, activation=tf.nn.gelu))
        model.add(layers.Dropout(dropout_rate))
        # x = layers.Dense(units, activation=tf.nn.gelu)(x)
        # x = layers.Dropout(dropout_rate)(x)
    return model(x)


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


plt.figure(figsize=(4, 4))
image = x_train[np.random.choice(range(x_train.shape[0]))]
plt.imshow(image.astype("uint8"))
plt.axis("off")
plt.show()

resized_image = tf.image.resize(tf.convert_to_tensor([image]), size=(image_size, image_size))
patches = Patches(patch_size)(resized_image)
print(f"图片大小: {image_size} X {image_size}")
print(f"切块大小e: {patch_size} X {patch_size}")
print(f"每个图对应的切块大小: {patches.shape[1]}")
print(f"每个块对应的元素: {patches.shape[-1]}")

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy().astype("uint8"))
    plt.axis("off")
plt.show()


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    # 这里call后需要定义get_config函数，命名自拟，文章3.9中给出
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    # 数据增强
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # 创建多个Transformer encoding 块
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # 创建多头自注意力机制 multi-head attention layer，这里经过测试Tensorflow2.5可用
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        # Skip connection.
        x2 = layers.Add()([attention_output, encoded_patches])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # 增加MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # 输出分类.
    logits = layers.Dense(num_classes)(features)
    # 构建
    model = keras.Model(inputs=inputs, outputs=logits)
    model.summary()
    return model


# tfa.方法可替换为adam
def run_experiment(model):
    model.compile(
        # 下述可直接替换为  optimizer='adam',
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = "tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history


vit_classifier = create_vit_classifier()
history = run_experiment(vit_classifier)
