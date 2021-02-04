import io
from glob import glob
from os.path import join

import numpy as np
from fabric import Connection

from resources import config as cfg


def upload_weights_to_remote(model, host, weight_dir, verbose=False):
    conn = Connection(host, user=cfg.HOSTS[host])

    for layer in model.layers:
        if verbose:
            print(f"Uploading weights of layer {layer.name}")
        target_file_name = join(weight_dir, f"{model.name}_{layer.name}.npz")
        weights = layer.get_weights()
        buffer = io.BytesIO()
        np.savez(buffer, weights)
        conn.put(buffer, join(cfg.PROJECT_DIR, target_file_name))


def update_remote_scripts(host, user, verbose=False):
    if verbose:
        print(f"Updating host {host}")
    conn = Connection(host, user, connect_timeout=2)
    files = glob("*.py")
    for file in files:
        if verbose:
            print(f"Uploading file {file}")
        conn.put(file, cfg.PROJECT_DIR)


def upload_inputs(host, user, verbose=False):
    conn = Connection(host, user, connect_timeout=2)
    files = glob("inputs/*.npy")
    for file in files:
        if verbose:
            print(f"Uploading input file {file}")
        conn.run(f"mkdir -p {cfg.PROJECT_DIR}/inputs")
        conn.put(file, f"{cfg.PROJECT_DIR}/inputs/")
