import os
import sys
import argparse

from fabric import Connection

from resources import config as cfg, utils
from resources.data import InputType


def get_remote_prediction(connection, host, model, input_type=InputType.DEFAULT, conversion_host=None):
    tf_versions = utils.get_tf_versions()

    if input_type == InputType.BOUNDARY:
        input_parameter = 'boundary'
    elif input_type == InputType.BOUNDARY_BATCH:
        input_parameter = 'boundary-batch'
    elif input_type == InputType.ARTIFICIAL:
        input_parameter = 'artificial'
    else:
        input_parameter = 'clean'

    if conversion_host is not None:
        lite_suffix = f"--lite {conversion_host}"
        lite_prefix = f"lite_{conversion_host}_"
    else:
        lite_suffix = ''
        lite_prefix = ''

    for tf_version in tf_versions:
        command = f'python predict.py --input {input_parameter} {model} {lite_suffix}'
        print(f"\n\nRunning command {command} on host {host}")
        with connection.cd(cfg.PROJECT_DIR):
            with connection.prefix(f"conda activate foreNNsic-cpu-{tf_version}"):
                connection.run(command)

        files = []
        if input_type == InputType.BOUNDARY:
            files.append(cfg.FILE_TEMPLATE.format(
                f"{lite_prefix}{tf_version}_{model}_boundary"))
        elif input_type == InputType.BOUNDARY_BATCH:
            files.append(cfg.FILE_TEMPLATE.format(
                f"{lite_prefix}{tf_version}_{model}_boundary_batch"))
        elif input_type == InputType.ARTIFICIAL:
            for key in cfg.ARTIFICIAL_KEYS:
                files.append(cfg.FILE_TEMPLATE.format(
                    f"{lite_prefix}{tf_version}_{model}{key}"))
        else:
            files.append(cfg.FILE_TEMPLATE.format(lite_prefix + tf_version + '_' + model))

        for filename in files:
            remote_filename = f"{cfg.FULL_PREDICTIONS_DIR}/{host}/{filename}"
            local_filename = os.path.join(cfg.PREDICTIONS_DIR, host, filename)
            print(f"Loading file {remote_filename} from host {host}")
            connection.get(remote_filename, local_filename)


def main():
    parser = argparse.ArgumentParser("Predict on remote machines")
    parser.add_argument("--remote", action="store_true", default=False)
    parser.add_argument("--cloud", action="store_true", default=False)
    parser.add_argument("--models", type=str, nargs='+', default=["mnist"],
                        help="The model type to use on the remote machine (default: mnist)")
    args = parser.parse_args()

    hosts = []
    connect_args = None

    if args.remote:
        print("Predicting on remote hosts")
        hosts.extend(cfg.REMOTE_HOSTS)

    if args.cloud:
        print("Predicting on cloud hosts")
        hosts.extend(cfg.CLOUD_HOSTS)

    for host in hosts:
        utils.ensure_prediction_directory(host, True)

        user = cfg.HOSTS[host]
        connection = Connection(host, user, connect_kwargs=connect_args)
        for model in args.models:
            get_remote_prediction(connection, host, model)
            if model in ['mnist', 'single_layer', 'mnist_mlp', 'fmnist', 'min_conv_layer_1_3']:
                get_remote_prediction(
                    connection, host, model, InputType.ARTIFICIAL)
            if model in ['mnist', 'cifar10', 'fmnist']:
                for conversion_host in cfg.REMOTE_HOSTS:
                    get_remote_prediction(connection, host, model, input_type=InputType.DEFAULT,
                                          conversion_host=conversion_host)

                get_remote_prediction(
                    connection, host, model, InputType.BOUNDARY)
                get_remote_prediction(
                    connection, host, model, InputType.BOUNDARY_BATCH)


if __name__ == "__main__":
    main()
