import os
import sys
import argparse

base_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_path)
sys.path.append(os.path.join(base_path, "classify"))

from dataset import get_dataset, save_data  # noqa: E402
from dataset_class import Dataset  # noqa: E402


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--embeddings_path", required=True, type=str,
        help="Path to the embeddings dataset"
    )
    parser.add_argument(
        "-d", "--data_path", required=True, type=str,
        help="Path to dataset of text lengths"
    )
    parser.add_argument(
        "--scatter_dataset", action="store_true",
        help="Whether to store the dataset scattered across multiple files or "
        "in a single file"
    )
    parser.add_argument(
        "-o", "--output_dir", required=False, type=str,
        help="Output path to store embeddings and data file"
    )
    return parser.parse_args()


def single_dataset_merge(embeddings_path, data_path):
    embeddings_data = get_dataset(embeddings_path)
    data = get_dataset(data_path)
    embeddings_data.update(data)
    return embeddings_data


def scatter_dataset_merge(embeddings_path, data_path, output_dir):
    embeddings_dataset = Dataset(data_path=embeddings_path)
    extra_dataset = get_dataset(data_path)
    embeddings_dataset.add_features(
        extra_dataset, in_place=False, data_dir=output_dir
    )


def main(embeddings_path, data_path, scatter_dataset, output_dir):
    output_name = "embeddings_with_lengths"
    if not scatter_dataset:
        embeddings_data = single_dataset_merge(embeddings_path, data_path)
        save_data(output_dir, output_name, **embeddings_data)
    else:
        embeddings_data = scatter_dataset_merge(
            embeddings_path, data_path, output_dir
        )


if __name__ == "__main__":
    args = parse_flags()
    main(**vars(args))
