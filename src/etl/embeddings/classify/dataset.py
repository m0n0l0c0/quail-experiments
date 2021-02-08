import os
import pickle
import numpy as np

from pathlib import Path
from itertools import combinations
from collections import Counter
from sklearn.model_selection import train_test_split

"""DEPRECATED"""
# separate normalization
NORM_FEATS = [["embeddings", "logits"], ["contexts", "question", "endings"]]
DEFAULT_FEATS = [["embeddings", "logits", "contexts", "question", "endings"]]


def get_dataset(data_path):
    dataset = pickle.load(open(data_path, "rb"))
    return dataset


def get_splits(dataset, test_size=0.25):
    data_keys = [key for key in dataset.keys() if key not in ["labels"]]
    train_dict, test_dict = {}, {}
    for key in data_keys:
        X_train, X_test, y_train, y_test = train_test_split(
            dataset[key], dataset["labels"], test_size=test_size
        )
        train_dict.update(**{key: X_train})
        test_dict.update(**{key: X_test})

        if "lebels" not in train_dict:
            train_dict.update(labels=y_train)
            test_dict.update(labels=y_test)

    return (train_dict, test_dict)


def get_x_y_from_dict(set_dict, **kwargs):
    X_set = None
    y_set = set_dict["labels"]
    if kwargs.get("features", None) is None:
        kwargs["features"] = ["embeddings"]

    if "features" in kwargs:
        for feature in kwargs["features"]:
            feat_values = set_dict[feature]
            feat_shape = feat_values.shape
            if len(feat_shape) > 2:
                if not kwargs.get("dont_reshape", False):
                    feat_values = feat_values.reshape(feat_shape[0], -1)
            elif len(feat_shape) == 1:
                if not kwargs.get("dont_reshape", False):
                    feat_values = feat_values.reshape(feat_shape[0], 1)
            if X_set is None:
                X_set = feat_values
            else:
                X_set = np.concatenate([X_set, feat_values], axis=1)

    if "cast" in kwargs:
        X_set = X_set.astype(kwargs["cast"])
        y_set = y_set.astype(kwargs["cast"])

    return X_set, y_set


def get_dataset_class_proportions(train_dict):
    y_train = train_dict["labels"]
    props = list(Counter(y_train).values())
    max_nof, min_nof = max(props), min(props)
    return round(max_nof / min_nof)


def get_flat_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def get_unique_features(features):
    return list(Counter(get_flat_list(features)).keys())


def get_normalized_dataset(data_path, features):
    return normalize_dataset_by_features(get_dataset(data_path), features)


def sweep_features(features):
    feature_combinations = []
    flat_features = get_flat_list(features)
    for i in range(1, len(flat_features) + 1):
        for combination in combinations(flat_features, i):
            feature_combinations.append(list(combination))

    return feature_combinations


def normalize_dataset_by_features(dataset, features):
    all_features = get_unique_features(features)
    for norm_group in NORM_FEATS:
        to_norm = [feat for feat in norm_group if feat in all_features]
        if len(to_norm) > 0:
            dataset = normalize_dataset(dataset, to_norm)

    return dataset, features


def normalize_dataset(dataset, features):
    all_data = None
    shapes = []
    for feature in features:
        feat_values = dataset[feature]
        feat_shape = feat_values.shape
        if len(feat_shape) > 2:
            feat_values = feat_values.reshape(feat_shape[0], -1)
        elif len(feat_shape) == 1:
            feat_values = feat_values.reshape(feat_shape[0], 1)

        shapes.append((feat_shape, feat_values.shape))

        if all_data is None:
            all_data = feat_values
        else:
            all_data = np.concatenate([all_data, feat_values], axis=1)

    # mean along rows
    all_data = (all_data.T - all_data.mean(axis=1)).T
    all_data = (all_data.T / all_data.std()).T

    for shape, feature in zip(shapes, features):
        orig_shape, reshaped = shape
        dataset[feature] = all_data[:, :reshaped[-1]]
        dataset[feature] = dataset[feature].reshape(orig_shape)

    return dataset


def save_data(output_dir, prefix, single_items=False, **kwargs):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if single_items:
        for key, value in kwargs.items():
            fpath = os.path.join(output_dir, f"{prefix}_{key}.pkl")
            with open(fpath, "wb") as fout:
                pickle.dump(value, fout)

    fpath = os.path.join(output_dir, f"{prefix}_data.pkl")
    with open(fpath, "wb") as fout:
        pickle.dump(kwargs, fout)
