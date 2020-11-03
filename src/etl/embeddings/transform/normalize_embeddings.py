import argparse

from sklearn.decomposition import PCA
from extract_embeddings import save_data
from classify.classification import (
    get_dataset,
    get_splits,
    get_x_y_from_dict,
    get_loggers,
    normalize_dataset
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
        "-o", "--output_dir", type=str, required=True,
        help="Output directory path to store the normalized dataset"
    )
    return parser.parse_args()


def main(data_path, n_components, normalize, output_dir):
    out_data_name = f"pca_{n_components}_comps"
    dataset = get_dataset(data_path)
    feature_set = ["embeddings"]
    untouched_features = {
        key: dataset[key]
        for key in dataset.keys()
        if key not in feature_set
    }
    
    if normalize:
        dataset = normalize_dataset(dataset, feature_set)

    X, y = get_x_y_from_dict(dataset, features=feature_set)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    dataset["embeddings"] = X_pca

    save_data(
        data_path,
        out_data_name,
        embeddings=X_pca,
        **untouched_features,
    )


if __name__ == "__main__":
    args = parse_flags()
    main(**vars(args))
