import os
import yaml
import json
import argparse

from pathlib import Path
from functools import partial
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from autogoal.search import PESearch

from pipeline import pipeline_map
from dataset_class import Dataset
from dataset_utils import DEFAULT_FEATS

from classifier_setup import (
    setup_gpu_device,
    setup_pipeline,
    get_fitness_fn,
)
from dataset import (
    get_splits,
    get_normalized_dataset,
    sweep_features,
)
from utils import (
    get_loggers,
    save_args,
    save_classifier,
    load_classifier,
)

arg_to_metric_map = {
    "accuracy": accuracy_score,
    "weighted_accuracy": balanced_accuracy_score,
    "f1": f1_score,
    "weighted_f1": partial(f1_score, average="weighted"),
}


def get_params(throw=True):
    params = None
    params_file = Path(os.getcwd()).absolute().joinpath("params.yaml")
    if params_file.exists():
        params = yaml.safe_load(open(params_file, "r"))

    if params is None and throw:
        raise RuntimeError("No params file found to search for features!")

    return params


def merge_with_params_file(parser, args):
    params = {}
    file_params = get_params(throw=False)
    if file_params is None:
        return args

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
        "--no_scatter_dataset", action="store_true",
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


def fit_classifier(args, classifier):
    loggers = get_loggers(args.output_dir)
    classifier, fitness_fn = classifier.values()
    if isinstance(classifier, PESearch):
        classifier, _ = classifier.run(args.iterations, logger=loggers)

    score = fitness_fn(classifier)

    print(f"End of training, best score {score}\nPipe: {classifier}")
    return classifier


# In the new setup, we don't use feature sets
# dvc exp will take care of comparing models accross feature_sets
def train_classifier(args, train_dict, test_dict, features, score_fn):
    print(f"Training with features: {features}")
    classifier = setup_pipeline(
        args,
        train_dict,
        test_dict,
        features,
        score_fn,
    )
    best_pipeline = fit_classifier(args, classifier)
    save_classifier(best_pipeline, args.output_dir)


def eval_score_fn(y_test, y_pred):
    print(classification_report(y_test, y_pred))
    return classification_report(y_test, y_pred, output_dict=True)


def eval_classifier(args, train_dict, test_dict, features, score_fn):
    if args.pipeline == "mlp":
        setup_gpu_device(args.gpu)

    print(f"Evaluating with features: {features}")
    classifier = load_classifier(args.output_dir)
    args.train = False
    fitness_fn = get_fitness_fn(
        args,
        train_dict,
        test_dict,
        features,
        eval_score_fn
    )

    # ToDo := Save a smaller report for DVC
    report = fitness_fn(classifier)
    if args.metrics_dir is not None:
        Path(args.metrics_dir).mkdir(exist_ok=True, parents=True)
        report_path = os.path.join(
            args.metrics_dir, "scores.json"
        )
        print(f"Writing evalution to {report_path}")
        with open(report_path, "w") as fout:
            fout.write(json.dumps(report) + "\n")


def get_data_path_from_features(args):
    params = get_params()
    features = params["features"]
    data_path = args.data_path
    prefix = ""
    suffix = ""
    if features.get("oversample", False):
        prefix += "oversample_"
    if features.get("normalization", False):
        prefix += "normalized_"

    # exception: oversample_embeddings contains the data in
    # oversample_embeddings/embeddings
    prefix += "embeddings"

    if features.get("text_length", False):
        suffix += "_with_lengths"

    dataset_name = f"{prefix}{suffix}"
    if dataset_name == "oversample_embeddings":
        dataset_name = os.path.join(dataset_name, "embeddings")

    data_path = os.path.join(data_path, dataset_name)
    return data_path


def get_features_from_params(args):
    if args.features is not None:
        features = [feat for feat in args.features if feat in DEFAULT_FEATS]
    else:
        params = get_params()
        params_feats = params["features"]
        features = [
            feat for feat, value in params_feats.items()
            if isinstance(value, bool) and value is not False
        ]

    return features


def main(args):
    data_path = get_data_path_from_features(args)
    print(f"Loading data from {data_path}")
    features = get_features_from_params(args)
    if not len(features):
        raise RuntimeError("No features found to work")

    if args.sweep_features:
        raise ValueError(
            "Sweep features is deprecated by now."
        )
        features = sweep_features(features)

    if args.no_scatter_dataset:
        dataset, features = get_normalized_dataset(data_path, features)
        train_dict, test_dict = get_splits(
            dataset, test_size=args.test_size, random_state=args.seed
        )
    else:
        dataset = Dataset(data_path=data_path)
        train_dict, test_dict = dataset.get_splits(
            test_size=args.test_size, random_state=args.seed
        )

    score_fn = arg_to_metric_map[args.metric]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.train:
        if args.pipeline == "mlp":
            setup_gpu_device(args.gpu)

        train_classifier(args, train_dict, test_dict, features, score_fn)
        save_args(args, args.output_dir)

    if args.eval:
        eval_classifier(args, train_dict, test_dict, features, score_fn)


if __name__ == "__main__":
    args = parse_flags()
    main(args)
