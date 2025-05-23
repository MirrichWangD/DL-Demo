from tensorflow import keras


def alexnet(num_classes: int = 1000, dropout: float = 0.5, **kwargs):
    model = keras.Sequential(name='AlexNet')
    model.add(
        keras.Sequential(
            [
                keras.layers.Conv2D(64, kernel_size=11, strides=4, padding='same'),
                keras.layers.ReLU(),
                keras.layers.MaxPooling2D((3, 3), strides=2),
                keras.layers.Conv2D(192, kernel_size=5, padding='same'),
                keras.layers.ReLU(),
                keras.layers.MaxPooling2D((3, 3), strides=2),
                keras.layers.Conv2D(384, kernel_size=3, padding='same'),
                keras.layers.ReLU(),
                keras.layers.Conv2D(256, kernel_size=3, padding='same'),
                keras.layers.ReLU(),
                keras.layers.Conv2D(256, kernel_size=3, padding='same'),
                keras.layers.ReLU(),
                keras.layers.MaxPooling2D((3, 3), strides=2),
            ],
            name='features',
        )
    )
    model.add(keras.layers.AveragePooling2D((6, 6), name='avgpool'))
    model.add(
        keras.Sequential(
            [
                keras.layers.Flatten(),
                keras.layers.Dropout(dropout),
                keras.layers.Dense(4096),
                keras.layers.ReLU(),
                keras.layers.Dropout(dropout),
                keras.layers.Dense(4096),
                keras.layers.ReLU(),
                keras.layers.Dense(num_classes),
            ],
            name='classifier',
        )
    )

    return model


if __name__ == '__main__':
    model = alexnet()
    model.build((None, 224, 224, 3))
    model.summary(expand_nested=True)
