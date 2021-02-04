from enum import Enum

import tensorflow.keras as keras
import numpy as np
import os


class FrequencyType(Enum):
    HORIZONTAL = 0
    VERTICAL = 1
    DIAGONAL = 2


class InputType(Enum):
    DEFAULT = 0
    ARTIFICIAL = 1
    BOUNDARY = 2
    BOUNDARY_BATCH = 3


def load_boundary(model_type, input_type: InputType):
    if input_type == InputType.BOUNDARY:
        suffix = ''
    elif input_type == InputType.BOUNDARY_BATCH:
        suffix = '_batch'
    path = os.path.join('boundaries', f"{model_type}{suffix}.npy")
    ret = np.load(path)
    return ret


def preprocess_data(x_train, y_train, x_test, y_test, shape, classes, test_only=False):
    x_test = (x_test / 255).reshape(
        -1, shape[0], shape[1], shape[2]).astype(np.float32)
    y_test = keras.utils.to_categorical(y_test, classes)

    if not test_only:
        x_train = (x_train / 255).reshape(
            -1, shape[0], shape[1], shape[2]).astype(np.float32)
        y_train = keras.utils.to_categorical(y_train, classes)

        return (x_train, y_train), (x_test, y_test)
    return x_test, y_test


def get_mnist_data(test_only=False):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    return preprocess_data(x_train, y_train, x_test, y_test, (28, 28, 1), classes=10, test_only=test_only)


def get_single_mnist_test_sample(index=42, include_label=False):
    x_test, y_test = get_mnist_data(test_only=True)
    ret_slice = slice(index, index + 1)
    if include_label:
        return x_test[ret_slice], y_test[ret_slice]
    return x_test[ret_slice]


def get_cifar_10_data(test_only=False):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    return preprocess_data(x_train, y_train, x_test, y_test, shape=(32, 32, 3), classes=10, test_only=test_only)


def get_single_cifar_10_test_sample(index=42, include_label=False):
    x_test, y_test = get_cifar_10_data(test_only=True)
    ret_slice = slice(index, index + 1)
    if include_label:
        return x_test[ret_slice], y_test[ret_slice]
    return x_test[ret_slice]


def get_fmnist_data(test_only=False):
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    return preprocess_data(x_train, y_train, x_test, y_test, shape=(28, 28, 1), classes=10, test_only=test_only)


def get_single_fmnist_test_sample(index=42, include_label=False):
    x_test, y_test = get_fmnist_data(test_only=True)
    ret_slice = slice(index, index + 1)
    if include_label:
        return x_test[ret_slice], y_test[ret_slice]
    return x_test[ret_slice]


def get_artificial_samples():
    shape = (1, 28, 28, 1)
    zero_image = np.zeros(shape, dtype=np.float32)
    half_image = np.full(shape, 0.5, dtype=np.float32)
    one_image = np.full(shape, 1, dtype=np.float32)

    horizontal_gradient = build_frequency_artificial(
        shape, period=4, frequency_type=FrequencyType.HORIZONTAL)
    vertical_gradient = build_frequency_artificial(
        shape, period=4, frequency_type=FrequencyType.VERTICAL)
    diagonal_gradient = build_frequency_artificial(
        shape, period=4, frequency_type=FrequencyType.DIAGONAL)
    checkerboard = build_frequency_artificial(
        shape, period=2, frequency_type=FrequencyType.DIAGONAL)

    vertical_gradient_1 = build_frequency_artificial(
        shape, period=1, frequency_type=FrequencyType.VERTICAL)
    vertical_gradient_2 = build_frequency_artificial(
        shape, period=2, frequency_type=FrequencyType.VERTICAL)
    vertical_gradient_3 = build_frequency_artificial(
        shape, period=3, frequency_type=FrequencyType.VERTICAL)
    vertical_gradient_4 = build_frequency_artificial(
        shape, period=4, frequency_type=FrequencyType.VERTICAL)
    vertical_gradient_5 = build_frequency_artificial(
        shape, period=5, frequency_type=FrequencyType.VERTICAL)
    vertical_gradient_6 = build_frequency_artificial(
        shape, period=6, frequency_type=FrequencyType.VERTICAL)
    vertical_gradient_14 = build_frequency_artificial(
        shape, period=14, frequency_type=FrequencyType.VERTICAL)
    vertical_gradient_15 = build_frequency_artificial(
        shape, period=15, frequency_type=FrequencyType.VERTICAL)
    vertical_gradient_28 = build_frequency_artificial(
        shape, period=28, frequency_type=FrequencyType.VERTICAL)

    horizontal_gradient_1 = build_frequency_artificial(
        shape, period=1, frequency_type=FrequencyType.HORIZONTAL)
    horizontal_gradient_2 = build_frequency_artificial(
        shape, period=2, frequency_type=FrequencyType.HORIZONTAL)
    horizontal_gradient_3 = build_frequency_artificial(
        shape, period=3, frequency_type=FrequencyType.HORIZONTAL)
    horizontal_gradient_4 = build_frequency_artificial(
        shape, period=4, frequency_type=FrequencyType.HORIZONTAL)
    horizontal_gradient_5 = build_frequency_artificial(
        shape, period=5, frequency_type=FrequencyType.HORIZONTAL)
    horizontal_gradient_6 = build_frequency_artificial(
        shape, period=6, frequency_type=FrequencyType.HORIZONTAL)
    horizontal_gradient_14 = build_frequency_artificial(
        shape, period=14, frequency_type=FrequencyType.HORIZONTAL)
    horizontal_gradient_15 = build_frequency_artificial(
        shape, period=15, frequency_type=FrequencyType.HORIZONTAL)
    horizontal_gradient_28 = build_frequency_artificial(
        shape, period=28, frequency_type=FrequencyType.HORIZONTAL)

    diagonal_gradient_1 = build_frequency_artificial(
        shape, period=1, frequency_type=FrequencyType.DIAGONAL)
    diagonal_gradient_2 = build_frequency_artificial(
        shape, period=2, frequency_type=FrequencyType.DIAGONAL)
    diagonal_gradient_3 = build_frequency_artificial(
        shape, period=3, frequency_type=FrequencyType.DIAGONAL)
    diagonal_gradient_4 = build_frequency_artificial(
        shape, period=4, frequency_type=FrequencyType.DIAGONAL)
    diagonal_gradient_5 = build_frequency_artificial(
        shape, period=5, frequency_type=FrequencyType.DIAGONAL)
    diagonal_gradient_6 = build_frequency_artificial(
        shape, period=6, frequency_type=FrequencyType.DIAGONAL)
    diagonal_gradient_14 = build_frequency_artificial(
        shape, period=14, frequency_type=FrequencyType.DIAGONAL)
    diagonal_gradient_15 = build_frequency_artificial(
        shape, period=15, frequency_type=FrequencyType.DIAGONAL)
    diagonal_gradient_28 = build_frequency_artificial(
        shape, period=28, frequency_type=FrequencyType.DIAGONAL)

    return np.concatenate(
        [zero_image, half_image, one_image, horizontal_gradient,
         vertical_gradient, diagonal_gradient, checkerboard, vertical_gradient_1, vertical_gradient_2,
         vertical_gradient_3, vertical_gradient_4, vertical_gradient_5, vertical_gradient_6,
         vertical_gradient_14, vertical_gradient_15, vertical_gradient_28, horizontal_gradient_1, horizontal_gradient_2,
         horizontal_gradient_3, horizontal_gradient_4, horizontal_gradient_5, horizontal_gradient_6,
         horizontal_gradient_14, horizontal_gradient_15, horizontal_gradient_28, diagonal_gradient_1,
         diagonal_gradient_2,
         diagonal_gradient_3, diagonal_gradient_4, diagonal_gradient_5, diagonal_gradient_6,
         diagonal_gradient_14, diagonal_gradient_15, diagonal_gradient_28])


def build_frequency_artificial(shape: tuple, period=4, frequency_type=FrequencyType.HORIZONTAL):
    assert (len(shape) == 4)
    ret = np.empty(shape)
    frequency = 1 / period

    if frequency_type == FrequencyType.HORIZONTAL:
        for j in range(shape[2]):
            ret[0, :, j, :] = np.cos(j * frequency * 2 * np.pi)
    elif frequency_type == FrequencyType.VERTICAL:
        for i in range(shape[1]):
            ret[0, i] = np.cos(i * frequency * 2 * np.pi)
    elif frequency_type == FrequencyType.DIAGONAL:
        for i in range(shape[1]):
            for j in range(shape[2]):
                ret[0, i, j] = np.cos(frequency * 2 * np.pi * (i + j))

    return ret.astype(np.float32)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.figure()
    sample = build_frequency_artificial(
        (1, 28, 28, 1), period=10, frequency_type=FrequencyType.DIAGONAL)[0]
    plt.imshow(sample, cmap='gray')
    plt.show()
