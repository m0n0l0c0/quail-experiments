import pickle
import random
import argparse

from collections import Counter

from autogoal.search import PESearch
from autogoal.ml.metrics import accuracy

from utils import get_loggers
from balanced_pipeline import get_pipeline
from balanced_sampling import balanced_resample
from dataset import (
    get_dataset,
    get_splits,
    get_x_y_from_dict,
)


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_path", required=True, type=str,
        help="Path to the dataset"
    )
    parser.add_argument(
        "-ts", "--test_size", required=False, type=float, default=0.33,
        help="The percentage of examples to use for test"
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, type=str,
        help="Output directory to store the models"
    )
    return parser.parse_args()


def get_dataset_rounds(train_dict):
    y_train = train_dict["labels"]
    props = list(Counter(y_train).values())
    max_nof, min_nof = max(props), min(props)
    return round(max_nof / min_nof)


def make_fn(train_dict, test_dict, feature_set, score_fn):
    X_test, y_test = get_x_y_from_dict(test_dict, features=feature_set)
    X_train, y_train = get_x_y_from_dict(train_dict, features=feature_set)
    X_train, y_train = balanced_resample(
        seed=random.choice(list(range(1000))),
        X_train=X_train,
        y_train=y_train,
    )

    def fitness(pipeline):
        y_pred = pipeline.fit(X_train, y_train).predict(X_test)
        return score_fn(y_pred, y_test)

    return fitness


def setup_pipeline(train_dict, test_dict, feature_set, score_metric):
    pipeline = get_pipeline(log_grammar=True)
    fitness_fn = make_fn(train_dict, test_dict, feature_set, score_metric)
    return PESearch(
        pipeline,
        fitness_fn,
        pop_size=50,
        selection=5,
        evaluation_timeout=1800,
        memory_limit=64 * (1024**3),
        early_stop=True,
    )


def main(args):
    loggers = get_loggers(args.output_dir)
    dataset = get_dataset(args.data_path)
    test_features = [
        ["embeddings"],
        ["embeddings", "logits"]
    ]

    train_dict, test_dict = get_splits(dataset, test_size=args.test_size)
    train_epochs = get_dataset_rounds(train_dict)
    for i, feature_set in enumerate(test_features):
        pipeline = setup_pipeline(train_dict, test_dict, feature_set, accuracy)
        for epoch in range(train_epochs):
            best_pipe, score = pipeline.run(50, logger=loggers)
            print(f"Epoch ({epoch}): {score}\nPipe {best_pipe}")
            classifier_fname = f"{args.output_dir}/classifier_{i}_{epoch}.pkl"
            with open(classifier_fname, "wb") as fout:
                pickle.dump(best_pipe, fout)


if __name__ == "__main__":
    args = parse_flags()
    main(args)
