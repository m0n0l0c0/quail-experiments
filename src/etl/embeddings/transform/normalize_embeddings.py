import os
import sys
import argparse

from sklearn.decomposition import PCA

base_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_path)
sys.path.append(os.path.join(base_path, "classify"))

from dataset_class import Dataset  # noqa: E402
from dataset import (  # noqa: E402
    get_dataset,
    get_x_y_from_dict,
    normalize_dataset,
    save_data,
)


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_path", type=str, required=True,
        help="Path to embeddings dataset"
    )
    parser.add_argument(
        "-n", "--n_components", type=int, required=False, default=50,
        help="Number of components to reduce dimensionality"
    )
    parser.add_argument(
        "--normalize", action="store_true",
        help="Wheter to normalize dataset before applying PCA"
    )
    parser.add_argument(
        "--scatter_dataset", action="store_true",
        help="Whether to store the dataset scattered across multiple files or "
        "in a single file"
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, type=str,
        help="Output dir to store embeddings and data file"
    )
    return parser.parse_args()


def single_dataset_normalize(data_path, normalize, n_components):
    dataset = get_dataset(data_path)
    feature_set = ["embeddings"]

    if normalize:
        dataset = normalize_dataset(dataset, feature_set)

    X, y = get_x_y_from_dict(dataset, features=feature_set)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    dataset["embeddings"] = X_pca
    return dataset


def scatter_dataset_normalize(data_path, normalize, n_components):
    feature_set = ["embeddings"]
    dataset = Dataset(data_path=data_path)
    if normalize:
        dataset = dataset.normalize_dataset_by_features(features=feature_set)

    pca = PCA(n_components=n_components)
    X_data = dataset.gather(features=feature_set)["X"]
    embeddings = X_data[feature_set[0]]
    embeddings = embeddings.reshape(embeddings.shape[0], -1)
    X_pca = pca.fit_transform(embeddings)
    return dataset, X_pca


def main(data_path, n_components, normalize, scatter_dataset, output_dir):
    if not scatter_dataset:
        dataset = single_dataset_normalize(data_path, normalize, n_components)
        save_data(
            output_dir,
            f"{n_components}_comps",
            **dataset,
        )
    else:
        dataset, X_pca = scatter_dataset_normalize(
            data_path, normalize, n_components
        )
        dataset.add_features(
            dict(embeddings=X_pca), in_place=False, data_dir=output_dir
        )


if __name__ == "__main__":
    args = parse_flags()
    main(**vars(args))
