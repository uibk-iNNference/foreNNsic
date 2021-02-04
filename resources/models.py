from tensorflow.keras.models import Sequential, Model
import tensorflow.keras.layers as layers
from tensorflow.keras.regularizers import l2


def build_mnist_cnn():
    ret = Sequential(name='mnist_cnn')
    ret.add(layers.Conv2D(32, input_shape=(28, 28, 1), kernel_size=3, activation='relu', name="conv1"))
    ret.add(layers.MaxPooling2D(pool_size=2, name="pool1"))
    ret.add(layers.Conv2D(64, kernel_size=3, activation='relu', name="conv2"))
    ret.add(layers.MaxPooling2D(pool_size=2, name="pool2"))
    ret.add(layers.Flatten(name="flatten"))
    ret.add(layers.Dense(10, activation='softmax', name="softmax"))

    return ret


def build_mnist_mlp():
    ret = Sequential(name='mnist_mlp')
    ret.add(layers.Input((28, 28, 1)))
    ret.add(layers.Flatten())
    ret.add(layers.Dense(1000, activation='relu'))
    ret.add(layers.Dense(1000, activation='relu'))
    ret.add(layers.Dense(10, activation='softmax'))

    return ret


def build_min_conv_layer(filters=1, kernel_size=1):
    ret = Sequential(name=f'min_conv_layer_{filters}_{kernel_size}')
    ret.add(layers.Conv2D(filters, input_shape=(28, 28, 1), kernel_size=kernel_size, name="conv1"))
    ret.add(layers.Conv2D(filters, kernel_size=kernel_size, name="conv2"))
    ret.add(layers.Flatten())
    ret.add(layers.Dense(10, name="softmax"))

    return ret


def build_single_layer():
    ret = Sequential(name='single_layer')
    ret.add(layers.Flatten(name="flatten", input_shape=(28, 28, 1)))
    ret.add(layers.Dense(10, activation='linear', name='dense'))

    return ret


def build_single_conv_layer():
    ret = Sequential(name='single_conv_layer')
    ret.add(layers.Conv2D(32, input_shape=(28, 28, 1), kernel_size=3, activation='relu', name="conv"))
    ret.add(layers.Dense(10, activation='softmax', name="softmax"))

    return ret


def get_resnet_layer(inputs,
                     num_filters=16,
                     kernel_size=3,
                     strides=1,
                     activation='relu',
                     batch_normalization=True,
                     conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = layers.Conv2D(num_filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding='same',
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
    else:
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
        x = conv(x)
    return x


def get_resnet_v1(input_shape, depth, num_classes=10, name='unnamed'):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = layers.Input(shape=input_shape)
    x = get_resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = get_resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 strides=strides)
            y = get_resnet_layer(inputs=y,
                                 num_filters=num_filters,
                                 activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = get_resnet_layer(inputs=x,
                                     num_filters=num_filters,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=None,
                                     batch_normalization=False)
            x = layers.add([x, y])
            x = layers.Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = layers.AveragePooling2D(pool_size=7)(x)
    # x = AveragePooling2D(pool_size=8)(x)
    y = layers.Flatten()(x)
    outputs = layers.Dense(num_classes,
                           activation='softmax',
                           kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs, name=name)
    return model


def build_fmnist(shape=(28, 28, 1), classes=(10), name='fmnist'):
    return get_resnet_v1(shape, 20, classes, name)


def build_cifar_10(shape=(32, 32, 3), classes=10, name='cifar10'):
    return get_resnet_v1(shape, 32, classes, name)
