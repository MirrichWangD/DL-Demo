from typing import Optional, Type, Union, List

from tensorflow import keras


def conv3x3(planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> keras.layers.Conv2D:
    """3x3 convolution with padding"""
    return keras.layers.Conv2D(
        planes,
        kernel_size=3,
        strides=stride,
        padding="same",
        groups=groups,
        use_bias=False,
        dilation_rate=dilation,
        kernel_initializer="he_normal",
    )


def conv1x1(planes: int, stride: int = 1) -> keras.layers.Conv2D:
    """1x1 convolution"""
    return keras.layers.Conv2D(planes, kernel_size=1, strides=stride, use_bias=False, kernel_initializer="he_normal")


class BasicBlock(keras.layers.Layer):
    expansion: int = 1

    def __init__(
        self,
        planes: int,
        stride: int = 1,
        downsample: Type[Union[keras.Sequential, keras.layers.Layer]] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer=None,
        zero_init_residual: bool = False,
    ) -> keras.layers.Layer:
        super().__init__()
        if zero_init_residual:
            norm_gamma_initializer = "zeros"
        else:
            norm_gamma_initializer = "ones"
        if norm_layer is None:
            norm_layer = keras.layers.BatchNormalization
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(planes, stride)
        self.bn1 = norm_layer()
        self.relu = keras.layers.ReLU()
        self.conv2 = conv3x3(planes)
        self.bn2 = norm_layer(gamma_initializer=norm_gamma_initializer)
        self.downsample = downsample
        self.stride = stride

    def call(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(keras.layers.Layer):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        planes: int,
        stride: int = 1,
        downsample: Type[Union[keras.Sequential, keras.layers.Layer]] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Type[Union[keras.Sequential, keras.layers.Layer]] = None,
        zero_init_residual: bool = False,
    ) -> keras.layers.Layer:
        super().__init__()
        if zero_init_residual:
            norm_gamma_initializer = "zeros"
        else:
            norm_gamma_initializer = "ones"

        if norm_layer is None:
            norm_layer = keras.layers.BatchNormalization
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(width)
        self.bn1 = norm_layer()
        self.conv2 = conv3x3(width, stride, groups, dilation)
        self.bn2 = norm_layer()
        self.conv3 = conv1x1(planes * self.expansion)
        self.bn3 = norm_layer(gamma_initializer=norm_gamma_initializer)
        self.relu = keras.layers.ReLU()
        self.downsample = downsample
        self.stride = stride

    def call(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(keras.Model):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Type[Union[keras.Sequential, keras.layers.Layer]] = None,
    ) -> keras.Model:
        super().__init__()
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        self.zero_init_residual = zero_init_residual

        if norm_layer is None:
            norm_layer = keras.layers.BatchNormalization
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = keras.layers.Conv2D(
            self.inplanes, kernel_size=7, strides=2, padding="same", use_bias=False, kernel_initializer="he_normal"
        )
        self.bn1 = norm_layer()
        self.relu = keras.layers.ReLU()
        self.maxpool = keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same")

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.avgpool = keras.layers.GlobalAveragePooling2D()

        self.fc = keras.Sequential([keras.layers.Flatten(), keras.layers.Dense(num_classes)])

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> keras.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = keras.Sequential([conv1x1(planes * block.expansion, stride), norm_layer()])

        layers = []
        layers.append(
            block(
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                self.zero_init_residual,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    zero_init_residual=self.zero_init_residual,
                )
            )

        return keras.Sequential(layers)

    def build(self, input_shape):
        self.call(keras.Input(input_shape[1:]))
        super().build(input_shape)

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x


def resnet18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    return model


def resnet34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    return model


def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    return model


def resnet101(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

    return model


def resnet152(**kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

    return model
