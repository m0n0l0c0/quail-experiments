import os
import sys
import json
import argparse

from pygit2 import Repository
from pathlib import Path

sys.path.append("../../processing")
sys.path.append("../embeddings/classify")

from hyperp_utils import load_params  # noqa: E402
from classification_report import classification_report  # noqa: E402
from classification import get_features_from_object  # noqa: E402


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
        "-o", "--output_dir", type=str, required=False, default=None,
        help="Output dir to store data"
    )
    parser.add_argument(
        "--git_tape", action="store_true", required=False,
        help="Whether to do the git tape or print the results"
    )
    parser.add_argument(
        "--filter_pipeline", type=str, required=False, default=None,
        help="Filter pipeline results based on a string"
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


def print_single_results(pipeline, features, data, digits):
    print(f"Pipeline:\t{pipeline}")
    print(f"Features:\t{', '.join(features)}")
    print(classification_report(data, digits, tabulated=True))


def print_data(lookup_table, scores_file, digits=2):
    params = load_params("params.yaml")
    data = json.load(open(scores_file))
    features = get_features_from_object(params, allow_all_feats=False)
    pipeline = params["classification"]["pipeline"]
    lookup_entry = "_".join([pipeline, *features])
    if lookup_entry not in lookup_table:
        lookup_table[lookup_entry] = data
        print_single_results(pipeline, features, data, digits)

    return lookup_table


def save_data(lookup_table, output_dir):
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        file_path = str(output_path.joinpath("results.json"))
        with open(file_path, "w") as fout:
            json.dump(fp=fout, obj=lookup_table)


def git_tape(scores_file, commit_msg, digits, output_dir):
    lookup_table = {}
    repo = Repository(".")
    lookup_table = print_data(lookup_table, scores_file, digits=digits)
    for commit in repo.walk(repo.head.target):
        os.system(f"git checkout {commit.id} 2>&1 >/dev/null")
        if commit.message.strip() != commit_msg:
            continue
        lookup_table = print_data(lookup_table, scores_file, digits=digits)

    save_data(lookup_table, output_dir)


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
            output_dir=args.output_dir,
        )
        git_tape(**fn_args)
    else:
        fn_args.update(filter_pipeline=args.filter_pipeline)
        print_results(**fn_args)
