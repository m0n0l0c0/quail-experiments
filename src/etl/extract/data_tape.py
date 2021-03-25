import os
import sys
import argparse

from shutil import copyfile
from pygit2 import Repository
from pathlib import Path
from os.path import basename, dirname

base_path = dirname(dirname(__file__))
src_root = dirname(base_path)

sys.path.append(os.path.join(src_root, "processing"))
sys.path.append(os.path.join(src_root, "etl", "embeddings", "classify"))

from hyperp_utils import load_params  # noqa: E402
from classification import get_features_from_object  # noqa: E402
from classification import get_data_path_from_features  # noqa: E402


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_file", type=str, required=True,
        help="File to extract along the git tape"
    )
    parser.add_argument(
        "-c", "--commit_msg", type=str, required=True,
        help="Message from commits to filter"
    )
    parser.add_argument(
        "--dvc_stage", type=str, required=False, default=None,
        help="Dvc target to checkout when git taping"
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=False, default=None,
        help="Output dir to store data"
    )
    return parser.parse_args()


def gather_data(lookup_table, data_file, output_dir):
    params = load_params("params.yaml")
    features = get_features_from_object(params, allow_all_feats=False)
    pipeline = params["classification"]["pipeline"]
    lookup_entry = "_".join([pipeline, *features])
    if lookup_entry not in lookup_table:
        lookup_table.append(lookup_entry)
        output_dir = Path(output_dir).joinpath(
            pipeline, get_data_path_from_features()
        )
        save_data(data_file, output_dir)

    return lookup_table


def save_data(data_file, output_dir):
    output_dir.mkdir(exist_ok=False, parents=True)
    file_name = basename(data_file)
    data_src = str(data_file)
    params_src = "params.yaml"
    data_dst = str(output_dir.joinpath(file_name))
    params_dst = str(output_dir.joinpath("params.yaml"))
    print(f"Saving '{data_src}': '{data_dst}'")
    copyfile(data_src, data_dst)
    print(f"Saving '{params_src}': '{params_dst}'")
    copyfile(params_src, params_dst)


def git_tape(data_file, commit_msg, output_dir, dvc_stage):
    lookup_table = []
    repo = Repository(".")
    lookup_table = gather_data(lookup_table, data_file, output_dir)
    for commit in repo.walk(repo.head.target):
        os.system(f"git checkout {commit.id} 2>&1 >/dev/null")
        if commit.message.strip() != commit_msg:
            continue
        dvc_stage = "" if dvc_stage is None else dvc_stage
        os.system(f"dvc checkout {dvc_stage} 2>&1 >/dev/null")
        lookup_table = gather_data(lookup_table, data_file, output_dir)


if __name__ == '__main__':
    git_tape(**vars(parse_flags()))
