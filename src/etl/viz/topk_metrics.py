import os
import sys
import json
import argparse
import numpy as np


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_metric", type=str)
    parser.add_argument(
        "-m", "--metric_field", type=str, required=True,
        help="Field to extract to sort metrics (dot notation accepted)."
    )
    parser.add_argument(
        "-t", "--topk", type=int, default=3,
        help="Number of top metrics to keep."
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output file to store top metrics"
    )
    return parser.parse_args()


def field_access(data, key):
    aux = data
    for k in key.split("."):
        aux = aux[k]
    return aux


def main(input_metric, metric_field, topk, output):
    data = json.load(open(input_metric, "r"))
    data_dir = os.path.dirname(input_metric)

    if output is None:
        output = os.path.join(data_dir, "topk.json")

    topk_mets = {}
    keys = list(data.keys())

    vals = [field_access(elem, metric_field) for elem in data.values()]
    topk_indices = np.argsort(vals, axis=None)[::-1][:topk].tolist()
    for index in topk_indices:
        index_key = keys[index]
        topk_mets[index_key] = data[index_key]

    with open(output, "w") as fout:
        fout.write(json.dumps(topk_mets) + "\n")


if __name__ == "__main__":
    main(**vars(parse_flags()))

