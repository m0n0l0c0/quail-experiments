import os
import torch
import pickle
import argparse

from pathlib import Path

from autogoal.search import PESearch
from autogoal.ml.metrics import accuracy

from utils import get_loggers, save_args
from mlp_classifier import (
    get_pipeline,
    MLPClassifier,
    get_hidden_size,
    eval_classifier,
    train_classifier,
)
from dataset import (
    get_splits,
    get_normalized_dataset,
    sweep_features,
    DEFAULT_FEATS,
)

GPU_DEVICE = None


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_path", required=True, type=str,
        help="Path to the dataset"
    )
    parser.add_argument(
        "-bs", "--batch_size", required=False, type=int, default=1024,
        help="The batch size for train/predict"
    )
    parser.add_argument(
        "--lr", required=False, type=float, default=0.01,
        help="Learning rate for optimization algorithm"
    )
    parser.add_argument(
        "--epochs", required=False, type=int, default=50,
        help="Training epochs for MLP classifier"
    )
    parser.add_argument(
        "-ts", "--test_size", required=False, type=float, default=0.33,
        help="The percentage of examples to use for test"
    )
    parser.add_argument(
        "-g", "--gpu", required=False, default=0, type=int,
        help="GPU to use (default to 0)"
    )
    parser.add_argument(
        "-f", "--features", required=False, type=str, nargs="*", default=None,
        help=f"Features used to train the classifier (default: "
        f"{DEFAULT_FEATS})"
    )
    parser.add_argument(
        "-sf", "--sweep_features", action="store_true",
        help="Try all possible combinations of features"
    )
    parser.add_argument(
        "-a", "--autogoal", action="store_true",
        help="Whether to perform hyper search with autogoal (dafault False)"
    )
    parser.add_argument(
        "-i", "--iterations", required=False, type=int, default=100,
        help="Iterations to run for autogoal training"
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, type=str,
        help="Output directory to store the models"
    )
    return parser.parse_args()


def save_classifier(classifier, output_dir, feature_set):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    features_str = f"classifier_{'_'.join(feature_set)}.pkl"
    classifier_fname = os.path.join(output_dir, features_str)
    with open(classifier_fname, "wb") as fout:
        pickle.dump(classifier, fout)
    print(f"Saved model to: {classifier_fname}")


def make_fn(
    args,
    train_dict,
    test_dict,
    feature_set,
    score_fn,
):
    train_data = dict(
        train_epochs=args.epochs,
        train_dict=train_dict,
        test_dict=test_dict,
        feature_set=feature_set,
        batch_size=args.batch_size,
        score_fn=score_fn,
        print_result=False,
    )

    eval_data = dict(
        test_dict=test_dict,
        feature_set=feature_set,
        batch_size=args.batch_size,
        score_fn=score_fn,
        print_result=False,
    )

    def fitness(pipeline):
        train_classifier(pipeline.classifier, **train_data)
        return eval_classifier(pipeline.classifier, **eval_data)

    return fitness


def setup_pipeline(args, train_dict, test_dict, feature_set, score_fn):
    pipeline = get_pipeline(log_grammar=True)
    fitness_fn = make_fn(
        args,
        train_dict,
        test_dict,
        feature_set,
        score_fn,
    )
    return PESearch(
        pipeline,
        fitness_fn,
        pop_size=5,
        selection=2,
        evaluation_timeout=1800,
        memory_limit=64 * (1024**3),
        early_stop=False,
    )


def autogoal_train(args, train_dict, test_dict, features, score_fn):
    loggers = get_loggers(args.output_dir)
    for feature_set in features:
        pipeline = setup_pipeline(
            args,
            train_dict,
            test_dict,
            feature_set,
            score_fn,
        )
        best_pipe, score = pipeline.run(args.iterations, logger=loggers)
        print(f"Pipe {best_pipe}")
        save_classifier(best_pipe, args.output_dir, feature_set)


def std_train(args, train_dict, test_dict, features, score_fn):
    for feature_set in features:
        print(f"Training with features: {feature_set}")
        hidden_size = get_hidden_size(train_dict, feature_set)
        classifier = MLPClassifier(lr=args.lr)
        classifier.initialize(hidden_size, device=GPU_DEVICE)
        train_data = dict(
            train_epochs=args.epochs,
            train_dict=train_dict,
            test_dict=test_dict,
            feature_set=feature_set,
            batch_size=args.batch_size,
            score_fn=score_fn,
        )
        test_data = dict(
            test_dict=test_dict,
            feature_set=feature_set,
            batch_size=args.batch_size,
            score_fn=score_fn,
        )
        train_classifier(classifier, **train_data)
        eval_classifier(classifier, **test_data)
        save_classifier(classifier, args.output_dir, feature_set)


def main(args):
    global GPU_DEVICE
    print(f"Loading data from {args.data_path}")
    GPU_DEVICE = torch.device("cuda", index=args.gpu)
    features = [args.features] if args.features is not None else DEFAULT_FEATS
    if args.sweep_features:
        features = sweep_features(features)

    dataset, features = get_normalized_dataset(args.data_path, features)
    train_dict, test_dict = get_splits(dataset, test_size=args.test_size)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_fn = autogoal_train if args.autogoal else std_train
    train_fn(args, train_dict, test_dict, features, accuracy)
    save_args(args, args.output_dir)


if __name__ == "__main__":
    args = parse_flags()
    main(args)
