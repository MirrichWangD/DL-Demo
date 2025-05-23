from tensorflow import keras


class AlexNet(keras.Model):

    def __init__(self, num_classes: int = 1000, dropout: float = 0.5):
        super().__init__()
        self.features = keras.Sequential(
            [
                keras.layers.Conv2D(64, kernel_size=11, strides=4, padding="same"),
                keras.layers.ReLU(),
                keras.layers.MaxPooling2D((3, 3), strides=2),
                keras.layers.Conv2D(192, kernel_size=5, padding="same"),
                keras.layers.ReLU(),
                keras.layers.MaxPooling2D((3, 3), strides=2),
                keras.layers.Conv2D(384, kernel_size=3, padding="same"),
                keras.layers.ReLU(),
                keras.layers.Conv2D(256, kernel_size=3, padding="same"),
                keras.layers.ReLU(),
                keras.layers.Conv2D(256, kernel_size=3, padding="same"),
                keras.layers.ReLU(),
                keras.layers.MaxPooling2D((3, 3), strides=2),
            ],
            name="features",
        )
        self.avgpool = keras.layers.AveragePooling2D((6, 6))
        self.classifier = keras.Sequential( 
                keras.layers.Flatten(),
                keras.layers.Dropout(dropout),
                keras.layers.Dense(4096),
                keras.layers.ReLU(),
                keras.layers.Dropout(dropout),
                keras.layers.Dense(4096),
                keras.layers.ReLU(),
                keras.layers.Dense(num_classes),
            ]
        )

    def build(self, input_shape):
        self.call(keras.Input(input_shape[1:]))
        super().build(input_shape)

    def call(self, inputs):
        x = self.features(inputs)
        x = self.avgpool(x)
        x = self.classifier(x)

        return x


def alexnet(**kwargs):
    model = AlexNet(**kwargs)

    return model


model = alexnet()
model.build(input_shape=(None, 224, 224, 3))
model.summary(expand_nested=True)
