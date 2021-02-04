import argparse
from os.path import join

import numpy as np
from jinja2 import Environment, FileSystemLoader
from prettytable import PrettyTable

from resources import config as cfg


class LiteComparison(object):
    """
    Comparison for TF lite predictions
    """

    def __init__(self, model_types, tf_versions, inputs, conversion_hosts, rows):
        self.model_types = model_types
        self.tf_versions = tf_versions
        self.inputs = inputs
        self.conversion_hosts = conversion_hosts
        self.rows = rows


def generate_comparison(model_types: list, hosts: list, tf_versions: list, inputs: list, conversion_hosts: list):
    rows = {}

    def insert_value(host, value):
        try:
            rows[host].append(value)
        except KeyError:
            rows[host] = [value]

    for model_type in model_types:
        for execution_host in hosts:
            for tf_version in tf_versions:
                for input_suffix in inputs:
                    suffix = f"_{input_suffix}" if len(
                        input_suffix) > 0 else ''
                    unique_predictions = []
                    for conversion_host in conversion_hosts:
                        filename = f"lite_{conversion_host}_{tf_version}_{model_type}{suffix}"
                        prediction_path = join(
                            cfg.PREDICTIONS_DIR, execution_host, cfg.FILE_TEMPLATE.format(filename))
                        prediction = np.load(prediction_path)

                        candidate_class = 0
                        for unique_prediction in unique_predictions:
                            if np.all(prediction == unique_prediction):
                                break
                            candidate_class += 1

                        if candidate_class == len(unique_predictions):
                            unique_predictions.append(prediction)

                        insert_value(cfg.get_display_name(
                            conversion_host), candidate_class)

    return LiteComparison(model_types, tf_versions, inputs, conversion_hosts, rows)


def generate_lite_table(model_types, hosts, tf_versions, inputs, conversion_hosts):
    comparison = generate_comparison(
        model_types, hosts, tf_versions, inputs, conversion_hosts)

    table = PrettyTable()
    table.header = False

    # generate header, version, and input row
    header_row = ['Model']
    execution_host_row = ['Executed on']
    version_row = ['TF Version']
    input_row = ['Input']

    for model_type in model_types:
        header_row.append(model_type)
        for i, conversion_host in enumerate(hosts):
            execution_host_row.append(cfg.get_display_name(conversion_host))
            for j, tf_version in enumerate(tf_versions):
                version_row.append(tf_version)
                for k, input_suffix in enumerate(inputs):
                    input_row.append(input_suffix if len(
                        input_suffix) > 0 else 'default')

                    if i + j + k > 0:
                        header_row.append('')
                    if j + k > 0:
                        execution_host_row.append('')
                    if k > 0:
                        version_row.append('')

    table.add_row(header_row)
    table.add_row(execution_host_row)
    table.add_row(version_row)
    table.add_row(input_row)

    # content
    for key, row in comparison.rows.items():
        new_row = [key]
        for item in row:
            new_row.append(item)
        table.add_row(new_row)

    lines = table.get_string().splitlines()
    print("\n".join(lines[:4]))
    print(lines[-1])
    print("\n".join(table.get_string().splitlines()[4:]))


def generate_lite_tex(model_types, hosts, tf_versions, inputs, conversion_hosts):
    # header line
    env = Environment(loader=FileSystemLoader('./templates'), block_start_string='@@',
                      block_end_string='@@', variable_start_string='@=', variable_end_string='=@', trim_blocks=True)
    template = env.get_template('lite_classes.tex')
    comparison = generate_comparison(
        model_types, hosts, tf_versions, inputs, conversion_hosts)

    # simplify gradient names
    inputs = [i.replace('vertical_gradient_', '') for i in inputs]
    inputs = [i.replace('horizontal_gradient_', '') for i in inputs]
    inputs = [i.replace('diagonal_gradient_', '') for i in inputs]

    # escape special characters
    model_types = [m.replace('_', '\_') for m in model_types]
    inputs = [i.replace('_', '\_') for i in inputs]

    cleaned_execution_hosts = [cfg.get_display_name(host) for host in hosts]
    cleaned_conversion_hosts = [cfg.get_display_name(conversion_host) for conversion_host in conversion_hosts]

    rendered_template = template.render(
        model_types=model_types,
        inputs=inputs,
        execution_hosts=cleaned_execution_hosts,
        conversion_hosts=cleaned_conversion_hosts,
        rows=comparison.rows)
    print(rendered_template)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate comparison table")
    parser.add_argument('--hosts', default=cfg.REMOTE_HOSTS, nargs='+',
                        help='List of hosts to compare')
    parser.add_argument(
        '--versions', default=['2.3.0'], nargs='+', help='List of TF versions to compare')
    parser.add_argument('--inputs', default=[''], nargs='+',
                        help='Optional input specification (default: [""])')
    parser.add_argument(
        '--model_types', nargs='+',
        default=['mnist', 'cifar10', 'fmnist'],
        help="One of {mnist, mnist_mlp, single_layer, single_conv_layer, cifar10, fmnist}. The model to compare for.")
    parser.add_argument("--conversion_hosts", nargs='+', help="Hosts for which to use TF lite converted models")
    parser.add_argument("--tex", default=False,
                        action="store_true", help="Render LaTeX output")

    args = parser.parse_args()

    if args.tex:
        generate_lite_tex(args.model_types, args.hosts,
                          args.versions, args.inputs, args.conversion_hosts)
    else:
        generate_lite_table(args.model_types, args.hosts,
                            args.versions, args.inputs, args.conversion_hosts)
