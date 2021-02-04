import tensorflow as tf
from os.path import join
from os import makedirs
import argparse

from resources import storage, models, utils


def save_as_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    model_dir = join('tflite_models', utils.get_cleaned_hostname())
    makedirs(model_dir, exist_ok=True)

    destination_path = join(model_dir, f"{model.name}.tflite")
    with open(destination_path, 'wb') as target_file:
        target_file.write(tflite_model)


def main(model_type):
    model_type = model_type.lower()
    if model_type == 'mnist':
        model = models.build_mnist_cnn()
    elif model_type == 'single_layer':
        model = models.build_single_layer()
    elif model_type == 'single_conv_layer':
        model = models.build_min_conv_layer()
    elif model_type == 'mnist_mlp':
        model = models.build_mnist_mlp()
    elif model_type == 'cifar10':
        model = models.build_cifar_10()
    elif model_type == 'fmnist':
        model = models.build_fmnist()
    else:
        raise RuntimeError(f"Unknown model type {model_type}")

    storage.load_weights(model)
    save_as_tflite(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert model to tflite")
    parser.add_argument("model_type", type=str, default="mnist", help="The model type to convert")

    args = parser.parse_args()
    main(args.model_type)
