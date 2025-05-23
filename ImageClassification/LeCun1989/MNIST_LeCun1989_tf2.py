import torch
import numpy as np
import tensorflow as tf

model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(12, kernel_size=5, strides=2, padding="same"),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dense(10),
    ]
)

model.build(input_shape=(None, 16, 16, 1))
model.summary()

x_train, y_train = torch.load("./lecun1989-repro-master/train1989.pt")
x_test, y_test = torch.load("./lecun1989-repro-master/test1989.pt")
x_train = np.array(x_train).reshape(7291, 16, 16, 1)
x_test = np.array(x_test).reshape(2007, 16, 16, 1)
y_train = (np.array(y_train) + 1) / 2
y_test = (np.array(y_test) + 1) / 2

optimizer = tf.keras.optimizers.SGD()
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
model.fit(x_train, y_train, epochs=23, validation_split=0, batch_size=128)
