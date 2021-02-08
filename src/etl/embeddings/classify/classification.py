import os
import yaml
import json
import pickle
import random
import argparse
import numpy as np

from pathlib import Path
from functools import partial
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

from utils import get_loggers, save_args
from pipeline import get_pipeline, save_pipeline, pipeline_map
from dataset_class import Dataset
from balanced_sampling import balanced_resample

from mlp_classifier import eval_classifier as mlp_evaluation
from mlp_classification import setup_gpu_device
from mlp_classification import std_train as mlp_std_train
from mlp_classification import autogoal_train as mlp_autogoal_train

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
    "weighted_accuracy": balanced_accuracy_score,
    "f1": f1_score,
    "weighted_f1": partial(f1_score, average="weighted"),
}


def merge_with_params_file(parser, args):
    params_file = Path(os.getcwd())/"params.yaml"

    if not params_file.exists():
        return args

    params = {}
    file_params = yaml.safe_load(open(params_file, 'r'))

    if "classification" in file_params:
        params.update(**file_params["classification"])
    if "multi_layer" in file_params["classification"]:
        params.update(**file_params["classification"]["multi_layer"])

    for key, value in params.items():
        if (
            args.__contains__(key) and
            parser.get_default(key) == args.__getattribute__(key)
        ):
            print(f"Overriding from 'params.yaml': {key}={value}")
            args.__setattr__(key, value)

    return args


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
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
        "-t", "--train", required=False, action="store_true"
    )
    parser.add_argument(
        "-e", "--eval", required=False, action="store_true"
    )
    parser.add_argument(
        "--scatter_dataset", action="store_true",
        help="Whether to store the dataset scattered across multiple files or "
        "in a single file"
    )
    parser.add_argument(
        "-a", "--autogoal", action="store_true",
        help="Whether to auto-discover the best pipeline/hyperparams or try "
        "whith the provided ones"
    )
    parser.add_argument(
        "-p", "--pipeline", required=False, default="full", type=str,
        help="Pipeline to use (unused when `autogoal=True`).\nOptions: "
        f"{list(pipeline_map.keys())}, default=`full`"
    )
    parser.add_argument(
        "-m", "--metric", required=False, default="accuracy", type=str,
        help="Metric to train/evaluate classifiers.\n Options: "
        f"{list(arg_to_metric_map.keys())}"
    )
    parser.add_argument(
        "-f", "--features", required=False, type=str, nargs="*", default=None,
        help=f"Features used to train the classifier (default: "
        f"{DEFAULT_FEATS})"
    )
    parser.add_argument(
        "-ts", "--test_size", required=False, type=float, default=0.33,
        help="The percentage of examples to use for test"
    )
    parser.add_argument(
        "--seed", required=False, type=float, default=None,
        help="Seed to use when splitting datasets (used for reproducibility)"
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, type=str,
        help="Output directory to store k-top best pipelines and logs"
    )
    parser.add_argument(
        "--metrics_dir", required=False, type=str,
        help="Output directory to store classification report as json"
    )
    parser.add_argument(
        "-sf", "--sweep_features", action="store_true",
        help="Try all possible combinations of features"
    )
    parser.add_argument(
        "--mlp", required=False, action="store_true",
        help="Whether to train a MLP classifier or a pipeline"
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
        "-g", "--gpu", required=False, default=0, type=int,
        help="GPU to use (default to 0), used only with mlp=True"
    )
    args = merge_with_params_file(parser, parser.parse_args())
    if not args.train and not args.eval:
        raise ValueError(
            "You must either train or evaluate a model"
        )
    return args


def make_balanced_fn(train_dict, test_dict, feature_set, score_fn):
    if isinstance(train_dict, Dataset):
        raise ValueError(
            "Cannot do balanced sampling on a Dataset instance, load raw data"
            " or balance outside classification process"
        )
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


def make_fn(train_dict, test_dict, feature_set, score_fn):
    if isinstance(train_dict, Dataset):
        X_train, y_train = train_dict.get_x_y_from_dict(features=feature_set)
        X_test, y_test = test_dict.get_x_y_from_dict(features=feature_set)
    else:
        X_train, y_train = get_x_y_from_dict(train_dict, features=feature_set)
        X_test, y_test = get_x_y_from_dict(test_dict, features=feature_set)

    def fitness(pipeline):
        pipeline.fit(X_train, y_train)

        X_test_list = X_test
        y_test_list = y_test
        if isinstance(X_test, Dataset):
            X_test_list = np.array(X_test.to_list())
            y_test_list = np.array(y_test.to_list())

        y_pred = pipeline.predict(X_test_list)
        return score_fn(y_test_list, y_pred)

    return fitness


def setup_pipeline(args, train_dict, test_dict, feature_set, score_fn):
    pipeline = get_pipeline(pipe_type=args.pipeline, log_grammar=True)
    if args.balanced:
        maker = make_balanced_fn
    else:
        maker = make_fn

    fitness_fn = maker(train_dict, test_dict, feature_set, score_fn)

    classifier = PESearch(
        pipeline,
        fitness_fn,
        pop_size=args.popsize,
        selection=args.selection,
        evaluation_timeout=args.timeout,
        memory_limit=args.memory * Gb,
        early_stop=args.early_stop,
        random_state=args.seed,
    )
    return dict(classifier=classifier, fitness_fn=fitness_fn)


def setup_automl(args, train_dict, test_dict, feature_set, score_fn):
    classifier = AutoML(
        input=MatrixContinuousDense(),
        output=CategoricalVector(),
        search_algorithm=PESearch,
        search_iterations=args.iterations,
        score_metric=score_fn,
        random_state=args.seed,
        search_kwargs=dict(
            pop_size=args.popsize,
            selection=args.selection,
            evaluation_timeout=args.timeout,
            memory_limit=args.memory * Gb,
            early_stop=args.early_stop,
        ),
    )
    fitness_fn = make_fn(train_dict, test_dict, feature_set, score_fn)
    return dict(classifier=classifier, fitness_fn=fitness_fn)


def fit_classifier(args, classifier):
    loggers = get_loggers(args.output_dir)
    classifier, fitness_fn = classifier.values()
    if isinstance(classifier, PESearch):
        classifier, _ = classifier.run(args.iterations, logger=loggers)

    score = fitness_fn(classifier)

    print(f"End of training, best score {score}\nPipe: {classifier}")
    return classifier


def save_classifier(classifier, output_dir, feature_set):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    features_str = f"classifier_{'_'.join(feature_set)}.pkl"
    classifier_fname = os.path.join(output_dir, features_str)
    save_pipeline(classifier, classifier_fname)
    print(f"Saved best pipeline to: {classifier_fname}")


def train_classifier(args, train_dict, test_dict, features, score_fn):
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
            score_fn,
        )
        best_pipeline = fit_classifier(args, classifier)
        save_classifier(best_pipeline, args.output_dir, feature_set)


def eval_classifier(args, train_dict, test_dict, features, score_fn):
    if not args.mlp:
        raise ValueError(
            "Only MLP classifiers evaluation is implemented by now"
        )

    if args.metrics_dir is not None:
        Path(args.metrics_dir).mkdir(exist_ok=True, parents=True)

    for feature_set in features:
        print(f"Evaluating with features: {feature_set}")
        classifier_name = f"classifier_{'_'.join(feature_set)}"
        classifier_path = os.path.join(
            args.output_dir, f"{classifier_name}.pkl"
        )
        classifier = pickle.load(open(classifier_path, "rb"))

        setup_gpu_device(args.gpu)
        test_data = dict(
            test_dict=test_dict,
            feature_set=feature_set,
            batch_size=args.batch_size,
            score_fn=score_fn,
            print_result=True,
            return_y=True,
        )
        _, y_test, y_pred = mlp_evaluation(classifier, **test_data)

        if args.metrics_dir is not None:
            report = classification_report(y_test, y_pred, output_dict=True)
            report_path = os.path.join(
                args.metrics_dir, f"{classifier_name}.json"
            )
            print(f"Writing evalution to {report_path}")
            with open(report_path, "w") as fout:
                fout.write(json.dumps(report) + "\n")


def main(args):
    print(f"Loading data from {args.data_path}")
    features = [args.features] if args.features is not None else DEFAULT_FEATS
    if args.sweep_features:
        features = sweep_features(features)

    if args.scatter_dataset:
        dataset = Dataset(data_path=args.data_path)
        train_dict, test_dict = dataset.get_splits(
            test_size=args.test_size, random_state=args.seed
        )
    else:
        dataset, features = get_normalized_dataset(args.data_path, features)
        train_dict, test_dict = get_splits(
            dataset, test_size=args.test_size, random_state=args.seed
        )

    score_fn = arg_to_metric_map[args.metric]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.train:
        train_fn = train_classifier
        if args.mlp:
            setup_gpu_device(args.gpu)
            train_fn = mlp_autogoal_train if args.autogoal else mlp_std_train

        train_fn(args, train_dict, test_dict, features, score_fn)
        save_args(args, args.output_dir)

    if args.eval:
        eval_classifier(args, train_dict, test_dict, features, score_fn)


if __name__ == "__main__":
    args = parse_flags()
    main(args)
