# Python RNG
import random
random.seed(42)

# Numpy RNG
import numpy as np
np.random.seed(42)

# TF RNG
from tensorflow.python.framework import random_seed
random_seed.set_seed(42)

from resources import data, storage, models
import argparse


def train(model, x_train, y_train, x_test, y_test, batch_size, epochs):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    score = model.evaluate(x_test, y_test, verbose=0)

    print(f"Test loss: {score[0]}")
    print(f"Test accuracy: {score[1]}")

    print("Saving weights...")
    storage.save_weights(model)


def main(model_type):
    model_type = model_type.lower()

    if model_type == 'mnist':
        print("Training mnist")
        (x_train, y_train), (x_test, y_test) = data.get_mnist_data()
        model = models.build_mnist_cnn()
        train(model, x_train, y_train, x_test, y_test, batch_size=128, epochs=5)

    elif model_type == 'mnist_mlp':
        print("Training mnist_mlp")
        (x_train, y_train), (x_test, y_test) = data.get_mnist_data()
        model = models.build_mnist_mlp()
        train(model, x_train, y_train, x_test, y_test, batch_size=128, epochs=5)

    elif model_type == 'single_layer':
        print("Training single layer")
        model = models.build_single_layer()
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print("Saving weights...")
        storage.save_weights(model)

    elif model_type == 'min_conv_layer_1x1':
        print("Training single layer")
        model = models.build_min_conv_layer(kernel_size=1)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print("Saving weights...")
        storage.save_weights(model)

    elif model_type.startswith('min_conv_layer'):
        model_conv = model_type.replace('min_conv_layer_', '').split('_')
        model_conv = list(map(int, model_conv))

        print(f"Training minimum conv layer with {model_conv[0]} filters and kernel size {model_conv[1]}")
        model = models.build_min_conv_layer(filters=model_conv[0], kernel_size=model_conv[1])
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print("Saving weights...")
        storage.save_weights(model)

    elif model_type == 'single_conv_layer':
        print("Training single conv layer")
        model = models.build_single_conv_layer()
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print("Saving weights...")
        storage.save_weights(model)

    elif model_type == 'cifar10':
        print(f"Training cifar10")

        (x_train, y_train), (x_test, y_test) = data.get_cifar_10_data()
        model = models.build_cifar_10()
        train(model, x_train, y_train, x_test, y_test, batch_size=128, epochs=150)

    elif model_type == 'fmnist':
        print(f"Training fmnist")

        (x_train, y_train), (x_test, y_test) = data.get_fmnist_data()
        model = models.build_fmnist()
        train(model, x_train, y_train, x_test, y_test, batch_size=128, epochs=150)

    else:
        raise RuntimeError(f"Unknown model type {model_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a model used for evaluation")
    parser.add_argument("model_type", type=str, help="One of {'mnist', 'mnist_mlp', 'single_layer', 'cifar10', "
                                                     "'fmnist'}; the type of model to train")

    args = parser.parse_args()
    main(args.model_type)
