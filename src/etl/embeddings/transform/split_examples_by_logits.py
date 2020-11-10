import os
import sys
import argparse

base_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_path)
sys.path.append(os.path.join(base_path, "classify"))
sys.path.append(os.path.join(base_path, "extract"))

from extract_embeddings import save_data  # noqa: E402
from classification import get_dataset  # noqa: E402


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
    flat_features = []
    if "embeddings" in dataset and len(dataset["embeddings"].shape) < 3:
        raise ValueError(
            "This dataset is already flatten, cannot split by example!"
        )

    n_choices = None
    for feature in dataset.keys():
        shape = dataset[feature].shape
        if len(shape) < 2:
            flat_features.append(feature)
            continue
        elif n_choices is None:
            n_choices = shape[1]
        new_shape = [-1, *shape[2:]]
        dataset[feature] = dataset[feature].reshape(new_shape)

    for feature in flat_features:
        dataset[feature] = dataset[feature].repeat(n_choices)

    return dataset


def main(data_path, output_path):
    output_dir = os.path.dirname(output_path)
    output_name = os.path.splitext(os.path.basename(output_path))[0]
    dataset = split_data(get_dataset(data_path))

    save_data(
        output_dir,
        output_name,
        **dataset,
    )


if __name__ == "__main__":
    args = parse_flags()
    main(**vars(args))
