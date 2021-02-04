import os
import re
import socket
from glob import glob

import numpy as np

from resources import config as cfg, storage


def compare_predictions(host_a, host_b, similar_sufficient=False, use_gpu_if_available=False):
    if use_gpu_if_available and host_a in cfg.GPU_HOSTS:
        prediction_a = storage.load_gpu_prediction(host_a)
    else:
        prediction_a = storage.load_prediction(host_a)

    if use_gpu_if_available and host_b in cfg.GPU_HOSTS:
        prediction_b = storage.load_gpu_prediction(host_b)
    else:
        prediction_b = storage.load_prediction(host_b)

    if similar_sufficient:
        np.testing.assert_almost_equal(prediction_a, prediction_b)
    else:
        np.testing.assert_equal(prediction_a, prediction_b)


def ensure_prediction_directory(hostname=None, verbose=False):
    if hostname is None:
        hostname = get_cleaned_hostname()
    target_dir = os.path.join(cfg.PREDICTIONS_DIR, hostname)
    if verbose:
        print(f"Ensuring predictions directory for host {hostname} exists")
    os.makedirs(target_dir, exist_ok=True)


def get_cleaned_hostname():
    hostname = socket.gethostname()
    if hostname in cfg.CLUSTER_NODES:
        hostname = "cluster"
    return hostname


def get_tf_versions():
    tf_versions = []

    environment_files = glob("environments/tf-*.yml")
    for file in environment_files:
        basename = os.path.basename(file)
        m = re.search(r'tf-(.+)\.yml', basename)
        tf_versions.append(m.group(1))

    return tf_versions
