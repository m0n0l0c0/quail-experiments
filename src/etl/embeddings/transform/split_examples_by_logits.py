import os
import sys
import argparse

base_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_path)
sys.path.append(os.path.join(base_path, "classify"))
sys.path.append(os.path.join(base_path, "extract"))

from extract_embeddings import save_data  # noqa: E402
from classification import (
    get_dataset,
    get_x_y_from_dict,
    normalize_dataset
)  # noqa: E402


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_path", type=str, required=True,
        help="Path to embeddings dataset"
    )
    parser.add_argument(
        "-o", "--output_path", type=str, required=True,
        help="Output path to store the splitted dataset"
    )
    return parser.parse_args()


def split_data(dataset):
    feature_set = [key for key in dataset.keys() if key not in ["labels"]]
    n_choices = None
    for feature in feature_set:
        shape = dataset[feature].shape
        if n_choices is None:
            n_choices = shape[1]
        new_shape = [-1, *shape[2:]]
        dataset[feature] = dataset[feature].reshape(new_shape)

    if "labels" in dataset:
        dataset["labels"] = dataset["labels"].repeat(n_choices)
    return dataset


def main(data_path, output_path):
    output_dir = os.path.dirname(output_path)
    output_name = os.path.basename(output_path)
    dataset = split_data(get_dataset(data_path))

    save_data(
        output_dir,
        output_name,
        **dataset,
    )


if __name__ == "__main__":
    args = parse_flags()
    main(**vars(args))
