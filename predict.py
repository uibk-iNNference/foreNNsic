import argparse
import os

import numpy as np
import tensorflow as tf

from resources import config as cfg, data, storage, models, utils
from resources.data import InputType

tf.compat.v1.enable_eager_execution()


def _get_mnist_inputs(input_type: InputType):
    if input_type == InputType.DEFAULT:
        return [(data.get_single_mnist_test_sample(42), '')]
    elif input_type == InputType.BOUNDARY:
        return [(data.load_boundary('mnist', input_type), '_boundary')]
    elif input_type == InputType.ARTIFICIAL:
        return zip(data.get_artificial_samples(), cfg.ARTIFICIAL_KEYS)
    else:
        raise SystemError(f"Unknown input type {input_type.name}")


def _get_fmnist_inputs(input_type: InputType):
    if input_type == InputType.DEFAULT:
        return [(data.get_single_fmnist_test_sample(42), '')]
    elif input_type == InputType.BOUNDARY:
        return [(data.load_boundary('fmnist', input_type), '_boundary')]
    elif input_type == InputType.ARTIFICIAL:
        return zip(data.get_artificial_samples(), cfg.ARTIFICIAL_KEYS)
    else:
        raise SystemError(f"Unknown input type {input_type.name}")


def _get_cifar10_inputs(input_type: InputType):
    if input_type == InputType.DEFAULT:
        return [(data.get_single_cifar_10_test_sample(), '')]
    elif input_type == InputType.BOUNDARY:
        return [(data.load_boundary('cifar10', input_type), '_boundary')]
    else:
        raise SystemError(f"Unhandled input type {input_type.name}")


def predict_individual(model_type: str, input_type: InputType):
    model_type = model_type.lower()
    if model_type == 'mnist':
        model = models.build_mnist_cnn()
        inputs = _get_mnist_inputs(input_type)
    elif model_type == 'single_layer':
        model = models.build_single_layer()
        inputs = _get_mnist_inputs(input_type)
    elif model_type.startswith('min_conv_layer'):
        model_conv = model_type.replace('min_conv_layer_', '').split('_')
        model_conv = list(map(int, model_conv))
        model = models.build_min_conv_layer(
            filters=model_conv[0], kernel_size=model_conv[1])
        if input_type == InputType.ARTIFICIAL:
            inputs = _get_mnist_inputs(input_type)
        else:
            inputs = zip(data.build_frequency_artificial((1, 28, 28, 1)), [''])
    elif model_type == 'single_conv_layer':
        model = models.build_single_layer()
        inputs = _get_mnist_inputs(input_type)
    elif model_type == 'mnist_mlp':
        model = models.build_mnist_mlp()
        inputs = _get_mnist_inputs(input_type)
    elif model_type == 'cifar10':
        model = models.build_cifar_10()
        inputs = _get_cifar10_inputs(input_type)
    elif model_type == 'fmnist':
        model = models.build_fmnist()
        inputs = _get_fmnist_inputs(input_type)
    else:
        raise RuntimeError(f"Unknown model type {model_type}")

    storage.load_weights(model)

    utils.ensure_prediction_directory(verbose=True)

    for i, (input, suffix) in enumerate(inputs):
        if len(input.shape) == 3:
            correct_input_shape = (1,) + input.shape
        else:
            correct_input_shape = input.shape

        input = np.reshape(input, correct_input_shape)
        prediction = model(input)

        tf_version = tf.__version__
        target_filename = os.path.join(
            cfg.PREDICTIONS_DIR,
            utils.get_cleaned_hostname(),
            cfg.FILE_TEMPLATE.format(
                f"{tf_version}_{model_type}{suffix}")
        )
        np.save(target_filename, prediction)


def predict_lite(model_type: str, input_type: InputType, conversion_host):
    model_type = model_type.lower()
    file_name = model_type
    if model_type == 'mnist':
        file_name = 'mnist_cnn'
        inputs = _get_mnist_inputs(input_type)
    elif model_type == 'single_layer':
        inputs = _get_mnist_inputs(input_type)
    elif model_type == 'single_conv_layer':
        inputs = _get_mnist_inputs(input_type)
    elif model_type == 'mnist_mlp':
        inputs = _get_mnist_inputs(input_type)
    elif model_type == 'cifar10':
        inputs = _get_cifar10_inputs(input_type)
    elif model_type == 'fmnist':
        inputs = _get_fmnist_inputs(input_type)
    else:
        raise RuntimeError(f"Unknown model type {model_type}")

    model_dir = os.path.join('tflite_models', conversion_host)
    interpreter = tf.lite.Interpreter(
        model_path=os.path.join(model_dir, f"{file_name}.tflite"))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i, (input, suffix) in enumerate(inputs):
        if len(input.shape) == 3:
            correct_input_shape = (1,) + input.shape
        else:
            correct_input_shape = input.shape

        input = np.reshape(input, correct_input_shape)

        interpreter.set_tensor(input_details[0]['index'], input)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        tf_version = tf.__version__
        target_filename = os.path.join(
            cfg.PREDICTIONS_DIR,
            utils.get_cleaned_hostname(),
            cfg.FILE_TEMPLATE.format(
                f"lite_{conversion_host}_{tf_version}_{model_type}{suffix}")
        )
        np.save(target_filename, prediction)


def predict_boundary_batch(model_type: str):
    model_type = model_type.lower()
    if model_type == 'mnist':
        model = models.build_mnist_cnn()
    elif model_type == 'fmnist':
        model = models.build_fmnist()
    elif model_type == 'cifar10':
        model = models.build_cifar_10()
    else:
        raise SystemError(f"Unhandled model type {model_type}")

    storage.load_weights(model)
    inputs = data.load_boundary(model_type, InputType.BOUNDARY_BATCH)

    utils.ensure_prediction_directory()

    predictions = np.empty((inputs.shape[0], 10))
    for i in range(inputs.shape[0]):
        current_input = inputs[i:i + 1]
        current_prediction = model(current_input)
        predictions[i] = current_prediction

    target_filename = os.path.join(
        cfg.PREDICTIONS_DIR,
        utils.get_cleaned_hostname(),
        cfg.FILE_TEMPLATE.format(f"{tf.__version__}_{model_type}_boundary_batch"))
    np.save(target_filename, predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Generate predictions")
    # parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument("--input", default="clean",
                        help="One of {clean, boundary, boundary-batch, artificial}. The inputs to use")
    parser.add_argument("model_type", type=str, nargs='?',
                        default='mnist', help="The model type to use (default: mnist)")
    parser.add_argument("--lite", metavar="host", dest="conversion_host",
                        help="Predict using TensorFlow Lite model created on [host]")
    args = parser.parse_args()

    if args.input == "clean":
        input_type = InputType.DEFAULT
    elif args.input == "artificial":
        input_type = InputType.ARTIFICIAL
    elif args.input == "boundary":
        input_type = InputType.BOUNDARY
    elif args.input == "boundary-batch":
        input_type = InputType.BOUNDARY_BATCH
    else:
        raise SystemError(f"Unknown input type {args.input}")

    if input_type == InputType.BOUNDARY_BATCH:
        predict_boundary_batch(args.model_type)
    elif args.conversion_host is not None:
        predict_lite(args.model_type, input_type,
                     conversion_host=args.conversion_host)
    else:
        predict_individual(args.model_type, input_type)
