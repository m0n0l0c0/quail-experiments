import argparse

from pathlib import Path
from sklearn.model_selection import ParameterGrid

from hyperp_utils import (
    load_params,
    write_params,
    update_params,
    combination_to_params,
    validate_combination,
)

features = {
    # "pipeline": ["logreg", "mlp"],
    "normalization": [False, True],
    "oversample": [False, True],
    "embeddings": [False, True],
    "logits": [False, True],
    "context": [False, True],
    "question": [False, True],
    "endings": [False, True]
}


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_path", required=True, default=None, type=str,
        help="File containing the dataset"
    )
    parser.add_argument(
        "-f", "--features", required=False, type=str, nargs="*", default=None,
        help="Features used to train the classifier"
    )
    return parser.parse_args()


def load_index(index_file):
    index = Path(index_file)
    if not index.exists():
        index = 0
    else:
        index = int(index.read_text())

    return index


def write_index(index, index_file):
    index_file = Path(index_file)
    index_file.write_text(str(index))


def get_combination(args, features, index):
    comb = None
    grid = ParameterGrid(features)
    if index >= len(grid):
        return comb, index

    comb = grid[index]
    comb["pipeline"] = "mlp"
    while not validate_combination(args, comb):
        index += 1
        if index >= len(grid):
            comb = None
            break
        comb = grid[index]
        comb["pipeline"] = "mlp"

    return combination_to_params(comb), index


def main(args):
    params_file = "./params.yaml"
    index_file = "./sweep.index"
    end_index_file = "./end.index"
    index = load_index(index_file)
    comb, index = get_combination(args, features, index)
    if comb is None or index is None:
        write_index(index, end_index_file)
    else:
        params = update_params(comb, load_params(params_file))
        write_params(params, params_file)
        index += 1
        write_index(index, index_file)


if __name__ == "__main__":
    main(parse_flags())
