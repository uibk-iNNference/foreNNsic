# instructions for using this configuration file are marked with #!
import os

TMP_FILE = "prediction_tmp.npy"
FILE_TEMPLATE = "prediction_{}.npy"
ARTIFICIAL_KEYS = ['_zero_image', '_half_image', '_one_image', '_horizontal_gradient', '_vertical_gradient',
                   '_diagonal_gradient', '_checkerboard', '_vertical_gradient_1', '_vertical_gradient_2',
                   '_vertical_gradient_3', '_vertical_gradient_4', '_vertical_gradient_5', '_vertical_gradient_6',
                   '_vertical_gradient_14', '_vertical_gradient_15', '_vertical_gradient_28', '_horizontal_gradient_1',
                   '_horizontal_gradient_2',
                   '_horizontal_gradient_3', '_horizontal_gradient_4', '_horizontal_gradient_5',
                   '_horizontal_gradient_6', '_horizontal_gradient_14', '_horizontal_gradient_15',
                   '_horizontal_gradient_28', '_diagonal_gradient_1', '_diagonal_gradient_2', '_diagonal_gradient_3',
                   '_diagonal_gradient_4', '_diagonal_gradient_5', '_diagonal_gradient_6', '_diagonal_gradient_14',
                   '_diagonal_gradient_15', '_diagonal_gradient_28']
GPU_FILE_TEMPLATE = "prediction_{}_gpu.npy"
ENVIRONMENT = "foreNNsic"  #! name of the conda environment
PROJECT_DIR = "Projects/forennsic"  #! the root folder of this repository relative to your home folder
PREDICTIONS_DIR = "predictions"
FULL_PREDICTIONS_DIR = f"{PROJECT_DIR}/{PREDICTIONS_DIR}"
USER = "forennsic"  #! we used the same user for all remote machines

HOSTS = {
    #! this dictionary maps from hostnames to usernames
}

#! if you classify on a cluster, this can help deal with user and name configuration
CLUSTER_NODES = [
    "headnode"
]
for i in range(10):
    CLUSTER_NODES.append(f"gc{i}")

for n in CLUSTER_NODES:
    HOSTS[n] = HOSTS["cluster"]

REMOTE_HOSTS = [
    #! we distinguish between cloud hosts and remote hosts for historical reasons, used in get_remote_predictions
]

#! we didn't use it in the paper, but only some hosts may have a GPU available
GPU_HOSTS = [
]

CLOUD_HOSTS = [
    #! our configuration for the cloud instances
]

DISPLAY_NAMES = {
    #! used for cleaned up names shown in the output graphics
}


def get_display_name(host: str):
    try:
        return DISPLAY_NAMES[host]
    except IndexError:
        return host


SSH_TIMEOUT = 2
