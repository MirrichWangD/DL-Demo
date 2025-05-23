from tensorflow import keras


def lecun1989(num_classes: int = 10):
    return keras.Sequential(
        [
            keras.layers.Conv2D(12, kernel_size=5, strides=2, padding='same'),
            keras.layers.MaxPooling2D(pool_size=2, strides=2),
            keras.layers.ReLU(),
            keras.layers.Flatten(),
            keras.layers.Dense(30),
            keras.layers.ReLU(),
            keras.layers.Dense(num_classes),
        ],
        name='LeCun1989-CNN',
    )
