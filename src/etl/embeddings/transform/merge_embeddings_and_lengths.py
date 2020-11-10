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
        "-e", "--embeddings_path", required=True, type=str,
        help="Path to the embeddings dataset"
    )
    parser.add_argument(
        "-d", "--data_path", required=True, type=str,
        help="Path to dataset of text lengths"
    )
    parser.add_argument(
        "-o", "--output_path", required=False, type=str,
        help="Output path to store embeddings and data file"
    )
    return parser.parse_args()


def main(embeddings_path, data_path, output_path):
    output_dir = os.path.dirname(output_path)
    output_name = os.path.splitext(os.path.basename(output_path))
    embeddings_data = get_dataset(embeddings_path)
    data = get_dataset(data_path)
    embeddings_data.update(data)
    save_data(
        output_dir,
        output_name,
        **embeddings_data
    )


if __name__ == "__main__":
    args = parse_flags()
    main(**vars(args))
