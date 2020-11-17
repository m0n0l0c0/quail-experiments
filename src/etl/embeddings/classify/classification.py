import os
import random
import argparse

from pathlib import Path

from sklearn.metrics import f1_score, accuracy_score

from balanced_sampling import balanced_resample
from utils import get_loggers, save_args
from pipeline import get_pipeline, save_pipeline, pipeline_map
from dataset import (
    get_splits,
    get_x_y_from_dict,
    get_normalized_dataset,
    sweep_features,
    DEFAULT_FEATS,
)

from autogoal.utils import Gb
from autogoal.search import PESearch
from autogoal.ml import AutoML
from autogoal.kb import (
    MatrixContinuousDense,
    CategoricalVector,
)

arg_to_metric_map = {
    "accuracy": accuracy_score,
    "f1": f1_score,
}


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--popsize", type=int, default=75)
    parser.add_argument("--selection", type=int, default=5)
    parser.add_argument("--early_stop", type=int, default=False)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument(
        "--memory", type=int, default=32,
        help="Allowed memory in GB"
    )
    parser.add_argument(
        "-d", "--data_path", required=True, default=None, type=str,
        help="File containing the dataset"
    )
    parser.add_argument(
        "-a", "--autogoal", action="store_true",
        help="Whether to auto-discover the best pipeline or try whith the "
        "provided one"
    )
    parser.add_argument(
        "-p", "--pipeline", required=False, default="full", type=str,
        help="Pipeline to use (unused when `autogoal=True`).\nOptions: "
        f"{list(pipeline_map.keys())}, default=`full`"
    )
    parser.add_argument(
        "-m", "--metric", required=False, default="accuracy", type=str,
        help="Metric to train/evaluate classifiers"
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
        "-ts", "--test_size", required=False, type=float, default=0.33,
        help="The percentage of examples to use for test"
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, type=str,
        help="Output directory to store k-top best pipelines and logs"
    )
    return parser.parse_args()


def get_automl(args, score_metric):
    return AutoML(
        input=MatrixContinuousDense(),
        output=CategoricalVector(),
        search_algorithm=PESearch,
        search_iterations=args.iterations,
        score_metric=score_metric,
        search_kwargs=dict(
            pop_size=args.popsize,
            selection=args.selection,
            evaluation_timeout=args.timeout,
            memory_limit=args.memory * Gb,
            early_stop=args.early_stop,
        ),
    )


def make_balanced_fn(train_dict, test_dict, feature_set, score_metric):
    X_test, y_test = get_x_y_from_dict(test_dict, features=feature_set)
    X_train, y_train = get_x_y_from_dict(train_dict, features=feature_set)
    X_train, y_train = balanced_resample(
        seed=random.choice(list(range(1000))),
        X_train=X_train,
        y_train=y_train,
    )

    def fitness(pipeline):
        y_pred = pipeline.fit(X_train, y_train).predict(X_test)
        return score_metric(y_pred, y_test)

    return fitness


def make_fn(train_dict, test_dict, feature_set, score_metric):
    X_train, y_train = get_x_y_from_dict(train_dict, features=feature_set)
    X_test, y_test = get_x_y_from_dict(test_dict, features=feature_set)

    def fitness(pipeline):
        y_pred = pipeline.fit(X_train, y_train).predict(X_test)
        return score_metric(y_test, y_pred)

    return fitness


def setup_pipeline(args, train_dict, test_dict, feature_set, score_metric):
    pipeline = get_pipeline(pipe_type=args.pipeline, log_grammar=True)
    if args.balanced:
        fitness_fn = make_balanced_fn(train_dict, test_dict, feature_set, score_metric)
    else:
        fitness_fn = make_fn(train_dict, test_dict, feature_set, score_metric)

    return PESearch(
        pipeline,
        fitness_fn,
        pop_size=args.popsize,
        selection=args.selection,
        evaluation_timeout=args.timeout,
        memory_limit=args.memory * Gb,
        early_stop=args.early_stop,
    )


def setup_automl(args, train_dict, test_dict, feature_set, score_metric):
    classifier = get_automl(args, score_metric)
    fitness_fn = make_fn(train_dict, test_dict, feature_set, score_metric)
    return dict(classifier=classifier, fitness_fn=fitness_fn)


def fit_classifier(args, classifier):
    loggers = get_loggers(args.output_dir)
    if isinstance(classifier, dict):
        classifier, fitness_fn = classifier.values()
        score = fitness_fn(classifier)
        best_pipe = classifier.best_pipeline_.sampler_.replay()
    else:
        best_pipe, score = classifier.run(args.iterations, logger=loggers)

    print(f"End of training, best score {score}\nPipe: {best_pipe}")
    return best_pipe


def save_classifier(classifier, output_dir, feature_set):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    features_str = f"classifier_{'_'.join(feature_set)}.pkl"
    classifier_fname = os.path.join(output_dir, features_str)
    save_pipeline(classifier, classifier_fname)
    print(f"Saved best pipeline to: {classifier_fname}")


def train_classifier(args, train_dict, test_dict, features):
    for feature_set in features:
        print(f"Training with features: {feature_set}")
        if args.autogoal:
            setup_fn = setup_automl
        else:
            setup_fn = setup_pipeline

        classifier = setup_fn(
            args,
            train_dict,
            test_dict,
            feature_set,
            arg_to_metric_map[args.metric],
        )
        best_pipeline = fit_classifier(args, classifier)
        save_classifier(best_pipeline, args.output_dir, feature_set)


def main(args):
    print(f"Loading data from {args.data_path}")
    features = [args.features] if args.features is not None else DEFAULT_FEATS
    if args.sweep_features:
        features = sweep_features(features)

    dataset, features = get_normalized_dataset(args.data_path, features)
    train_dict, test_dict = get_splits(dataset, test_size=args.test_size)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_classifier(args, train_dict, test_dict, features)
    save_args(args, args.output_dir)


if __name__ == "__main__":
    args = parse_flags()
    main(args)
