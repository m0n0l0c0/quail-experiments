import json
import argparse


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files", metavar="files", type=str, nargs="+",
        help="Files to print in tabulated form"
    )
    parser.add_argument(
        "-d", "--digits", type=int, required=False, default=4,
        help="Number of digits to round floats in report"
    )
    return parser.parse_args()


def print_file(file_name, digits):
    report = ""
    data = json.load(open(file_name, "r"))
    width = max([len(cn) for cn in data] + [digits])
    row = "{:>{width}s}\t{:>9.{digits}f}\n"
    for key, value in data.items():
        report += row.format(key, value, width=width, digits=digits)
    return report


def main(files, digits):
    for file_name in files:
        print(print_file(file_name, digits))


if __name__ == "__main__":
    main(**vars(parse_flags()))
