from typing import Dict, List, Tuple, Type, Union, cast

from tensorflow import keras

cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False, init_weights: bool = True) -> keras.Sequential:
    init = 'he_normal' if init_weights else 'glorot_uniform'
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [keras.layers.MaxPooling2D(2, strides=2)]
        else:
            v = cast(int, v)
            conv2d = keras.layers.Conv2D(v, kernel_size=3, padding='same', kernel_initializer=init)
            if batch_norm:
                layers += [conv2d, keras.layers.BatchNormalization(), keras.layers.ReLU()]
            else:
                layers += [conv2d, keras.layers.ReLU()]
    return keras.Sequential(layers, name='features')


def vgg(
    name: str, features: keras.Sequential, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
) -> keras.Sequential:
    init = keras.initializers.RandomNormal(0, 0.01) if init_weights else 'glorot_uniform'
    model = keras.Sequential(name=name)
    classifier = keras.Sequential(
        [
            keras.layers.Flatten(),
            keras.layers.Dense(4096, kernel_initializer=init),
            keras.layers.ReLU(),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(4096, kernel_initializer=init),
            keras.layers.ReLU(),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(num_classes, kernel_initializer=init),
        ],
        name='classifier',
    )

    model.add(features)
    model.add(keras.layers.GlobalAveragePooling2D(name='avgpool'))
    model.add(classifier)

    return model


def vgg11(init_weights: bool = True, **kwargs) -> keras.Sequential:
    return vgg('VGG-11', make_layers(cfgs['A'], init_weights=init_weights), init_weights=init_weights, **kwargs)


def vgg13(init_weights: bool = True, **kwargs) -> keras.Sequential:
    return vgg('VGG-13', make_layers(cfgs['B'], init_weights=init_weights), init_weights=init_weights, **kwargs)


def vgg13_bn(init_weights: bool = True, **kwargs) -> keras.Sequential:
    return vgg(
        'VGG-13 (BN)',
        make_layers(cfgs['B'], init_weights=init_weights, batch_norm=True),
        init_weights=init_weights,
        **kwargs,
    )


def vgg16(init_weights: bool = True, **kwargs) -> keras.Sequential:
    return vgg('VGG-16', make_layers(cfgs['D'], init_weights=init_weights), init_weights=init_weights, **kwargs)


def vgg16_bn(init_weights: bool = True, **kwargs) -> keras.Sequential:
    return vgg(
        'VGG-16 (BN)',
        make_layers(cfgs['D'], init_weights=init_weights, batch_norm=True),
        init_weights=init_weights,
        **kwargs,
    )


def vgg19(init_weights: bool = True, **kwargs) -> keras.Sequential:
    return vgg('VGG-19', make_layers(cfgs['E'], init_weights=init_weights), init_weights=init_weights, **kwargs)


def vgg19_bn(init_weights: bool = True, **kwargs) -> keras.Sequential:
    return vgg(
        'VGG-19 (BN)',
        make_layers(cfgs['E'], init_weights=init_weights, batch_norm=True),
        init_weights=init_weights,
        **kwargs,
    )
