from resources import utils, config as cfg
import itertools
from os.path import join
import argparse
from jinja2 import Environment, FileSystemLoader
import json

import numpy as np
from prettytable import PrettyTable


class Comparison(object):
    """
    Contains all the data required for rendering a comparison table
    """

    def __init__(self, model_types: list, tf_versions: list, inputs: list, rows: dict, labels: dict, stats: dict):
        self.model_types = model_types
        self.tf_versions = tf_versions
        self.inputs = inputs
        self.rows = rows
        self.labels = labels
        self.stats = stats


def generate_comparison(model_types: list, hosts: list, tf_versions: list, inputs: list):
    rows = {}

    def insert_value(host, value):
        try:
            rows[host].append(value)
        except KeyError:
            rows[host] = [value]

    labels = {}
    stats = {}
    for model_type in model_types:
        model_labels = {}
        model_stats = {}
        for tf_version in tf_versions:
            version_labels = {}
            version_stats = {}
            for input_suffix in inputs:
                input_labels = {}
                suffix = f"_{input_suffix}" if len(input_suffix) > 0 else ''
                unique_predictions = []
                filename = f"{tf_version}_{model_type}{suffix}"
                for host in hosts:
                    prediction_path = join(
                        cfg.PREDICTIONS_DIR, host, cfg.FILE_TEMPLATE.format(filename))
                    prediction = np.load(prediction_path)

                    candidate_class = 0
                    for unique_prediction in unique_predictions:
                        if np.all(prediction == unique_prediction):
                            break
                        candidate_class += 1

                    if candidate_class == len(unique_predictions):
                        unique_predictions.append(prediction)

                    try:
                        display_name = cfg.DISPLAY_NAMES[host]
                    except KeyError:
                        display_name = host

                    insert_value(display_name, candidate_class)

                # add labels
                for i, prediction in enumerate(unique_predictions):
                    input_labels[i] = int(np.argmax(prediction))
                version_labels[input_suffix] = input_labels

                # calculate stats
                all_pairs = list(itertools.combinations(unique_predictions, 2))
                differences = np.empty((len(all_pairs), 10))

                for i, (left, right) in enumerate(all_pairs):
                    if np.all(left == right):
                        continue
                    differences[i] = np.abs(left - right)
                differences = differences.flatten()
                mean = 0.0 if not len(differences) else np.mean(differences)
                version_stats[input_suffix] = {'min': np.min(differences, initial=0), 'max': np.max(
                    differences, initial=0), 'mean': mean}

            model_labels[tf_version] = version_labels
            model_stats[tf_version] = version_stats

        labels[model_type] = model_labels
        stats[model_type] = model_stats

    return Comparison(model_types, tf_versions, inputs, rows, labels, stats)


def generate_table(model_types, hosts, tf_versions, inputs, display_labels=False, display_stats=False):
    comparison = generate_comparison(
        model_types, hosts, tf_versions, inputs)

    table = PrettyTable()
    table.header = False

    # generate header, version, and input row
    header_row = ['']
    version_row = ['']
    input_row = ['']
    for model_type in model_types:
        header_row.append(model_type)
        for i, tf_version in enumerate(tf_versions):
            version_row.append(tf_version)
            for j, input_suffix in enumerate(inputs):
                if i + j > 0:
                    header_row.append('')
                if j > 0:
                    version_row.append('')
                input_row.append(input_suffix if len(
                    input_suffix) > 0 else 'default')

    table.add_row(header_row)
    table.add_row(version_row)
    table.add_row(input_row)

    # content
    for key, row in comparison.rows.items():
        new_row = [key]
        for item in row:
            new_row.append(item)
        table.add_row(new_row)

    print(table)

    if display_labels:
        print("Labels:")
        print(json.dumps(comparison.labels, indent=2))

    if display_stats:
        print("Stats about differences between unique predictions:")
        print(json.dumps(comparison.stats, indent=2))


def generate_tex(model_types, hosts, tf_versions, inputs):
    # header line
    env = Environment(loader=FileSystemLoader('./templates'), block_start_string='@@',
                      block_end_string='@@', variable_start_string='@=', variable_end_string='=@', trim_blocks=True)
    template = env.get_template('equivalence_classes.tex')
    comparison = generate_comparison(
        model_types, hosts, tf_versions, inputs)

    # simplify gradient names
    inputs = [i.replace('vertical_gradient_', '') for i in inputs]
    inputs = [i.replace('horizontal_gradient_', '') for i in inputs]
    inputs = [i.replace('diagonal_gradient_', '') for i in inputs]

    # escape special characters
    model_types = [m.replace('_', '\_') for m in model_types]
    inputs = [i.replace('_', '\_') for i in inputs]

    rendered_template = template.render(
        model_types=model_types, tf_versions=tf_versions, inputs=inputs, rows=comparison.rows)
    print(rendered_template)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generate comparison table")
    parser.add_argument('--hosts', default=None, nargs='+',
                        help='List of hosts to compare')
    parser.add_argument(
        '--versions', default=['2.3.0'], nargs='+', help='List of TF versions to compare')
    parser.add_argument('--inputs', default=[''], nargs='+',
                        help='Optional input specification (default: [""])')
    parser.add_argument(
        '--model_types', nargs='+',
        default=['mnist', 'single_layer', 'single_conv_layer', 'min_conv_layer_1x1', 'min_conv_layer_2x2', 'mnist_mlp',
                 'cifar10', 'fmnist'],
        help="One of {mnist, mnist_mlp, single_layer, single_conv_layer, cifar10, fmnist}. The model to compare for.")
    parser.add_argument("--stats", action="store_true", default=False,
                        help="Add stats for prediction differences per model")
    parser.add_argument("--labels", action="store_true", default=False,
                        help="Add class labels for equivalence classes per model")
    parser.add_argument("--tex", default=False, action="store_true",
                        help="Change output to LaTeX")

    args = parser.parse_args()

    if args.tex:
        generate_tex(args.model_types, args.hosts,
                     args.versions, args.inputs)
    else:
        generate_table(model_types=args.model_types, hosts=args.hosts, tf_versions=args.versions,
                       inputs=args.inputs, display_labels=args.labels, display_stats=args.stats)
