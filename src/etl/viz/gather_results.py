import os
import sys
import json
import argparse

from pygit2 import Repository
from pathlib import Path

base_path = os.path.dirname(os.path.dirname(__file__))
src_root = os.path.dirname(base_path)

sys.path.append(os.path.join(src_root, "processing"))
sys.path.append(os.path.join(src_root, "etl", "embeddings"))


from hyperp_utils import load_params  # noqa: E402
from classification_report import classification_report  # noqa: E402
from classify.classification import get_features_from_object  # noqa: E402


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--scores_file", type=str, required=False,
        help="Scores file to gather results from"
    )
    parser.add_argument(
        "-c", "--commit_msg", type=str, required=False,
        help="Message from commits to filter"
    )
    parser.add_argument(
        "-d", "--digits", type=int, required=False, default=4,
        help="Number of digits to round floats in report"
    )
    parser.add_argument(
        "-o", "--output_file", type=str, required=False, default=None,
        help="Output file to store results"
    )
    parser.add_argument(
        "--git_tape", action="store_true", required=False,
        help="Whether to do the git tape or print the results "
        "(commits must be contiguous)"
    )
    parser.add_argument(
        "--filter_pipeline", type=str, required=False, default=None,
        help="Filter pipeline results based on a string"
    )
    parser.add_argument(
        "-p", "--print_report", action="store_true", required=False,
        help="Whether to print the classification report or not"
    )
    args = parser.parse_args()
    if args.git_tape and (not args.commit_msg or not args.scores_file):
        raise ValueError(
            "When doing a `git tape` you need a `commit message`!"
        )

    if not args.git_tape and not args.scores_file:
        raise ValueError(
            "When printing scores you must pass a `score_file` to print"
        )
    return args


def simple_report(data, digits, tabulated=False):
    report = ""
    width = max([len(cn) for cn in data] + [digits])
    row = "{:>{width}s}" + ("\t" if tabulated else "") + "{:>9.{digits}f}\n"
    for key, value in data.items():
        report += row.format(key, value, width=width, digits=digits)
    return report


def print_single_results(pipeline, features, data, digits):
    print(f"Pipeline:\t{pipeline}")
    print(f"Features:\t{', '.join(features)}")
    try:
        report = classification_report(data, digits, tabulated=True)
    except Exception as e:
        report = simple_report(data, digits, tabulated=True)
    finally:
        print(report)


def gather_data(lookup_table, scores_file, digits=2, print_report=False):
    params = load_params("params.yaml")
    data = json.load(open(scores_file))
    features = get_features_from_object(params, allow_all_feats=False)
    pipeline = params["classification"]["pipeline"]
    lookup_entry = "_".join([pipeline, *features])
    if lookup_entry not in lookup_table:
        lookup_table[lookup_entry] = data
        if print_report:
            print_single_results(pipeline, features, data, digits)

    return lookup_table


def save_data(lookup_table, output_file):
    if output_file is not None:
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        with open(output_file, "w") as fout:
            json.dump(fp=fout, obj=lookup_table)


def git_tape(scores_file, commit_msg, digits, output_file, print_report):
    lookup_table = {}
    repo = Repository(".")
    lookup_table = gather_data(lookup_table, scores_file, digits=digits)
    start_found = False
    for commit in repo.walk(repo.head.target):
        base_cmd = f"git checkout {commit.id}"
        if not start_found:
            base_cmd += " 1>/dev/null"
            base_cmd += " 2>/dev/null"
        else:
            base_cmd += " 2>&1 >/dev/null"
        os.system(base_cmd)
        if commit.message.strip() == commit_msg:
            if not start_found:
                print(f"Found starting commit at {commit.id}")
            start_found = True
        else:
            if not start_found and commit.message.strip() != commit_msg:
                continue
            elif start_found:
                break
        lookup_table = gather_data(
            lookup_table, scores_file,
            digits=digits, print_report=print_report
        )

    save_data(lookup_table, output_file)


def print_results(scores_file, digits, filter_pipeline):
    all_data = json.load(open(scores_file, "r"))
    for key in sorted(all_data):
        data = all_data[key]
        name_split = key.split("_")
        pipeline = name_split[0]
        if filter_pipeline is not None and (
            pipeline.strip().lower() != filter_pipeline.strip().lower()
        ):
            continue
        features = name_split[1:]
        print_single_results(pipeline, features, data, digits)


if __name__ == '__main__':
    args = parse_flags()
    fn_args = dict(
        scores_file=args.scores_file,
        digits=args.digits,
    )
    if args.git_tape:
        fn_args.update(
            commit_msg=args.commit_msg,
            output_file=args.output_file,
            print_report=args.print_report,
        )
        git_tape(**fn_args)
    else:
        fn_args.update(filter_pipeline=args.filter_pipeline)
        print_results(**fn_args)
