from os.path import join

import numpy as np

from resources import config as cfg


def save_weights(model, weight_dir="weights", verbose=False):
    for layer in model.layers:
        if verbose:
            print(f"Saving weights of layer {layer.name}")
        weights = layer.get_weights()
        for i, weight in enumerate(weights):
            target_file_name = join(weight_dir, f"{model.name}_{layer.name}_{i}")
            np.save(target_file_name, weight)


def load_weights(model, weight_dir="weights", verbose=False):
    for layer in model.layers:
        if verbose:
            print(f"Loading weights of layer {layer.name}")

        new_weights = []
        for i in range(len(layer.get_weights())):
            target_file_name = join(weight_dir, f"{model.name}_{layer.name}_{i}.npy")
            new_weights.append(np.load(target_file_name))

        layer.set_weights(new_weights)


def load_prediction(host):
    return np.load(
        join(
            cfg.PREDICTIONS_DIR,
            cfg.FILE_TEMPLATE.format(host)
        )
    )


def load_gpu_prediction(host):
    return np.load(
        join(
            cfg.PREDICTIONS_DIR,
            cfg.GPU_FILE_TEMPLATE.format(host)
        )
    )
