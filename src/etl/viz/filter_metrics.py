import os
import json
import argparse


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_metric", type=str)
    parser.add_argument(
        "-f", "--filter_fields", required=True, nargs="+",
        help="Fields to extract from the metrics file."
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output file to store top metrics"
    )
    return parser.parse_args()


def main(input_metric, filter_fields, output):
    data = json.load(open(input_metric, "r"))
    data_dir = os.path.dirname(input_metric)

    if output is None:
        output = os.path.join(data_dir, "filtered.json")

    filtered_mets = {}
    keys = list(data.keys())
    filtered = [field for field in keys if field in filter_fields]

    for field in filtered:
        filtered_mets[field] = data[field]

    with open(output, "w") as fout:
        fout.write(json.dumps(filtered_mets) + "\n")


if __name__ == "__main__":
    main(**vars(parse_flags()))
