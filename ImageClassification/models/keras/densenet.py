from typing import Tuple

from tensorflow import keras


def _dense_layer(name: str, growth_rate: int, bn_size: int, drop_rate: float) -> keras.Sequential:
    layer = keras.Sequential(
        [
            keras.layers.BatchNormalization(name='norm1'),
            keras.layers.ReLU(name='relu1'),
            keras.layers.Conv2D(
                bn_size * growth_rate, kernel_size=1, use_bias=False, kernel_initializer='he_normal', name='conv1'
            ),
            keras.layers.BatchNormalization(name='norm2'),
            keras.layers.ReLU(name='relu2'),
            keras.layers.Conv2D(
                growth_rate, kernel_size=3, padding='same', use_bias=False, kernel_initializer='he_normal', name='conv2'
            ),
        ],
        name=name,
    )

    if drop_rate > 0:
        layer.add(keras.layers.Dropout(drop_rate, name='dropout'))

    return layer


def _dense_block(name: str, num_layers: int, bn_size: int, growth_rate: int, drop_rate: float) -> keras.Sequential:
    block = keras.Sequential(name=name)
    for i in range(num_layers):
        layer = _dense_layer(f'denselayer{i + 1}', growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        block.add(layer)

    return block


def _transition(name: str, num_out_features: int) -> keras.Sequential:
    transition = keras.Sequential(
        [
            keras.layers.BatchNormalization(name='norm'),
            keras.layers.ReLU(name='relu'),
            keras.layers.Conv2D(
                num_out_features, kernel_size=1, use_bias=False, kernel_initializer='he_normal', name='conv'
            ),
            keras.layers.AveragePooling2D((2, 2), strides=2, name='pool'),
        ],
        name=name,
    )

    return transition


def densenet(
    name: str = 'DenseNet',
    growth_rate: int = 32,
    block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
    num_init_features: int = 64,
    bn_size: int = 4,
    drop_rate: float = 0,
    num_classes: int = 1000,
) -> keras.Sequential:
    model = keras.Sequential(name=name)

    # First Convolution
    features = keras.Sequential(
        [
            keras.layers.Conv2D(
                num_init_features,
                kernel_size=7,
                strides=2,
                padding='same',
                use_bias=False,
                kernel_initializer='he_normal',
                name='conv0',
            ),
            keras.layers.BatchNormalization(name='norm0'),
            keras.layers.ReLU(name='relu'),
            keras.layers.MaxPooling2D(3, strides=2, padding='same', name='pool0'),
        ],
        name='features',
    )

    # Each denseblock
    num_features = num_init_features
    for i, num_layers in enumerate(block_config):
        block = _dense_block(
            f'denseblock{i + 1}', num_layers=num_layers, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate
        )
        features.add(block)
        num_features = num_features + num_layers * growth_rate
        if i < len(block_config) - 1:
            trans = _transition(f'transition{i + 1}', num_features // 2)
            features.add(trans)

    # Final batch norm
    features.add(keras.layers.BatchNormalization(name='norm5'))

    # Linear layer
    classifier = keras.Sequential(
        [
            keras.layers.Flatten(name='flatten'),
            keras.layers.Dense(num_classes, name='classifier'),
        ],
        name='classifier',
    )

    model.add(features)
    model.add(keras.layers.GlobalAveragePooling2D(name='globalavgpool'))
    model.add(classifier)

    return model


def dense121(**kwargs) -> keras.Sequential:
    return densenet('DenseNet-121', 32, (6, 12, 24, 16), 64, *kwargs)


def dense161(**kwargs) -> keras.Sequential:
    return densenet('DenseNet-161', 48, (6, 12, 36, 24), 96, *kwargs)


def dense169(**kwargs) -> keras.Sequential:
    return densenet('DenseNet-169', 32, (6, 12, 32, 32), 64, *kwargs)


def dense201(**kwargs) -> keras.Sequential:
    return densenet('DenseNet-201', 32, (6, 12, 48, 32), 64, *kwargs)


def dense269(**kwargs) -> keras.Sequential:
    return densenet('DenseNet-269', 32, (6, 12, 64, 48), 64, *kwargs)
