import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from resources import config, data


def gradients(x, y, pred, gradients):
    """Visualize gradients for sample along with sample itself
    Arguments:
        x {np.ndarray} -- Input sample
        y {np.ndarray} -- True label as one-hot encoding
        pred {np.ndarray} -- Model predictions
        gradients {np.ndarray} -- dloss/dx
    """
    fig = plt.figure()
    fig.suptitle(f"True label: {np.argmax(y)}, Prediction: {np.argmax(pred)}, Confidence: {np.max(pred)}")
    plt.subplot(121)
    if len(gradients.shape) == 4:
        vis_gradients = gradients[0, :, :, 0]
    else:
        vis_gradients = gradients
    plt.imshow(vis_gradients, cmap='coolwarm')
    plt.title("Gradients")

    plt.subplot(122)
    if x.shape[3] == 1:
        plt.imshow(x[0, :, :, 0], cmap='gray')
    elif x.shape[3] == 3:
        plt.imshow(x[0])
    plt.title("Sample")
    plt.show()


def save_artificial_samples():
    artificials = data.get_artificial_samples()

    for i, sample in enumerate(artificials):
        sample = (sample * 255).astype(np.int8)
        sample = sample[:, :, 0]
        im = Image.fromarray(sample, mode="L")
        im.convert("RGB").save(os.path.join('data', f"{config.ARTIFICIAL_KEYS[i][1:]}.png"))


def prediction(x, y, pred):
    """Visualize prediction
    Arguments:
        x {np.ndarray} -- Input sample
        y {np.ndarray} -- True label as one-hot encoding
        pred {np.ndarray} -- Model predictions
    """
    sorted_predictions = np.sort(pred, axis=None)
    fig = plt.figure()
    fig.suptitle(
        f"True label: {np.argmax(y)}, Prediction: {np.argmax(pred)}, Confidence diff: {sorted_predictions[-1] - sorted_predictions[-2]}")

    if x.shape[3] == 1:
        plt.imshow(x[0, :, :, 0], cmap='gray')
    elif x.shape[3] == 3:
        plt.imshow(x[0])
    plt.show()
