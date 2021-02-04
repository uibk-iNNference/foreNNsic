import argparse
import os

import numpy as np
from jinja2 import Environment, FileSystemLoader
from prettytable import PrettyTable

import resources.config as cfg
from resources import data
from resources.data import InputType


class SingleEvaluation(object):
    """
    Contains all data of a single boundary evaluation
    """

    def __init__(self, dataset: str, correct_label: int, linf_diff: float, l2_diff: float,
                 peak_signal_noise_ratio: list, rows: list):
        self.dataset = dataset
        self.correct_label = correct_label
        self.linf_diff = linf_diff
        self.l2_diff = l2_diff
        self.peak_signal_noise_ratio = peak_signal_noise_ratio
        self.rows = rows


class BatchEvaluation(object):
    """
    Contains all data of a batch evaluation
    """

    def __init__(self, dataset: str, hosts: list, rows: list):
        self.dataset = dataset
        self.hosts = hosts
        self.rows = rows


def evaluate_single(model_type: str, hosts: list, tf_version: str) -> SingleEvaluation:
    if model_type == 'mnist':
        x, y = data.get_single_mnist_test_sample(include_label=True)
    elif model_type == 'fmnist':
        x, y = data.get_single_fmnist_test_sample(include_label=True)
    elif model_type == 'cifar10':
        x, y = data.get_single_cifar_10_test_sample(include_label=True)
    else:
        raise SystemError(f'Unknown model type {model_type}')

    x_adv = data.load_boundary(model_type, InputType.BOUNDARY)

    rows = {}
    all_sorted_predictions = np.empty((len(hosts), 10))
    for i, host in enumerate(hosts):
        prediction_path = os.path.join(cfg.PREDICTIONS_DIR, host, cfg.FILE_TEMPLATE.format(
            f"{tf_version}_{model_type}_boundary"))
        predictions = np.load(prediction_path)

        predicted_class = np.argmax(predictions)
        sorted_predictions = np.sort(predictions, axis=None)
        highest_confidence = sorted_predictions[-1]
        second_confidence = sorted_predictions[-2]
        display_name = cfg.get_display_name(host)

        rows[display_name] = {'predicted class': predicted_class,
                              'highest confidence': highest_confidence,
                              'second confidence': second_confidence,
                              'difference': highest_confidence - second_confidence}

        all_sorted_predictions[i] = sorted_predictions

    unique_sorted = np.unique(all_sorted_predictions, axis=0)
    if len(unique_sorted) == 1:
        print("ATTENTION: All predictions are identical after sorting\n\n")

    difference = x - x_adv
    linf_diff = np.max(np.abs(difference))
    flat_diff = difference.flatten()
    l2_diff = np.sqrt(np.dot(flat_diff, flat_diff))

    mean_squared_error = np.mean(np.square(difference))
    # max_i^2 is simply 1 in our case
    peak_signal_noise_ratio = 10 * np.log10(1 / mean_squared_error)

    return SingleEvaluation(model_type, int(np.argmax(y)), linf_diff, l2_diff, peak_signal_noise_ratio, rows)


def generate_single_table(model_type: str, hosts: list, tf_version: str):
    evaluation = evaluate_single(model_type, hosts, tf_version)

    output = PrettyTable()
    header = ['Host', 'Prediction', 'Confidence',
              'Second Confidence', 'Difference']
    output.field_names = header

    for host, results in evaluation.rows.items():
        row = [host] + list(results.values())
        output.add_row(row)

    print(output)
    print()
    print(f"infinity norm Diff = {evaluation.linf_diff}")
    print(f"L2 norm Diff = {evaluation.l2_diff}")
    print(f"PSNR = {evaluation.peak_signal_noise_ratio}")


def generate_single_tex(model_type: str, hosts: list, tf_version: str):
    evaluation = evaluate_single(model_type, hosts, tf_version)

    env = Environment(loader=FileSystemLoader('./templates'), block_start_string='@@',
                      block_end_string='@@', variable_start_string='@=', variable_end_string='=@', trim_blocks=True)
    template = env.get_template('single_boundary.tex')
    rendered_template = template.render(rows=evaluation.rows)
    print(rendered_template)


def evaluate_batch(model_type: str, hosts: list, tf_version: str):
    all_labels = {}
    for host in hosts:
        key = f"{host}-{tf_version}"
        prediction_path = os.path.join(cfg.PREDICTIONS_DIR, host, cfg.FILE_TEMPLATE.format(
            f"{tf_version}_{model_type}_boundary_batch"))
        current_predictions = np.load(prediction_path)
        labels = np.argmax(current_predictions, axis=1)
        all_labels[host] = labels

    all_ratios = {}
    for host in hosts:
        current_ratios = []
        current_labels = all_labels[host]
        for other_host in hosts:
            other_labels = all_labels[other_host]
            difference_ratio = np.mean(current_labels != other_labels)
            current_ratios.append(difference_ratio)

        all_ratios[cfg.get_display_name(host)] = current_ratios

    return BatchEvaluation(model_type, hosts, all_ratios)


def generate_batch_table(model_type: str, hosts: list, tf_version: str):
    evaluation = evaluate_batch(model_type, hosts, tf_version)

    table = PrettyTable()

    header = [''] + [cfg.get_display_name(host) for host in hosts]
    table.field_names = header

    for host, results in evaluation.rows.items():
        row = [host] + results
        table.add_row(row)

    print(table)


def generate_batch_tex(model_type: str, hosts: list, tf_version: str):
    evaluation = evaluate_batch(model_type, hosts, tf_version)

    env = Environment(loader=FileSystemLoader('./templates'), block_start_string='@@',
                      block_end_string='@@', variable_start_string='@=', variable_end_string='=@', trim_blocks=True)
    template = env.get_template('batch_boundary.tex')
    display_hosts = [cfg.get_display_name(host) for host in hosts]
    rendered_template = template.render(
        hosts=display_hosts, rows=evaluation.rows)
    print(rendered_template)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Evaluate the effectiveness of our \"boundary\" samples")
    parser.add_argument(
        "model_type", help="One of {mnist, fmnist, cifar10}, the model type to print evaluation for")
    parser.add_argument("--hosts", nargs='+', default=cfg.REMOTE_HOSTS,
                        help="hosts to print evaluation for")
    parser.add_argument("--tf-version", default="2.3.0",
                        help="TensorFlow version (default: 2.3.0)")
    parser.add_argument("--batch", default=False, action="store_true",
                        help="Evaluate a batch of boundarys (disables visualization)")
    parser.add_argument("--tex", action="store_true",
                        default=False, help="Generate LaTeX output")

    args = parser.parse_args()
    if args.tex:
        if args.batch:
            generate_batch_tex(args.model_type, args.hosts, args.tf_version)
        else:
            generate_single_tex(args.model_type, args.hosts, args.tf_version)
    else:
        if args.batch:
            generate_batch_table(args.model_type, args.hosts, args.tf_version)
        else:
            generate_single_table(args.model_type, args.hosts, args.tf_version)
