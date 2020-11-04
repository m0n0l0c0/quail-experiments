import os
import pickle
import argparse
import numpy as np

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from pipeline import get_pipeline, save_pipeline

from autogoal.ml.metrics import accuracy
from autogoal.utils import Gb
from autogoal.ml import AutoML
from autogoal.kb import (
    MatrixContinuousDense,
    CategoricalVector,
)
from autogoal.search import (
    PESearch,
    ConsoleLogger,
    MemoryLogger,
    ProgressLogger,
    Logger,
)


class CustomLogger(Logger):
    def error(self, e: Exception, solution):
        out_file = os.path.join(
            self.get_prefix(), "embeddings_classifier_errors.log"
        )
        if e and solution:
            with open(out_file, "a") as fp:
                fp.write(f"solution={repr(solution)}\nerror={e}\n\n")

    def update_best(self, new_best, new_fn, *args):
        out_file = os.path.join(
            self.get_prefix(), "embeddings_classifier.log"
        )
        with open(out_file, "a") as fp:
            fp.write(f"solution={repr(new_best)}\nfitness={new_fn}\n\n")

    def set_prefix(self, prefix):
        self.prefix = prefix

    def get_prefix(self):
        prefix = ""
        if hasattr(self, "prefix"):
            prefix = self.prefix
        return prefix


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--popsize", type=int, default=75)
    parser.add_argument("--selection", type=int, default=5)
    parser.add_argument(
        "--memory", type=int, default=32,
        help="Allowed memory in GB"
    )
    parser.add_argument(
        "-d", "--data_path", required=False, default=None, type=str,
        help="File containing the dataset"
    )
    parser.add_argument(
        "-o", "--output_dir", required=False, type=str,
        help="Output directory to store k-top best pipelines and logs"
    )
    parser.add_argument(
        "-a", "--autogoal", action="store_true",
        help="Whether to auto-discover the best pipeline or try whith the "
        "provided one"
    )
    return parser.parse_args()


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


def get_loggers(output_dir):
    logger = MemoryLogger()
    custom = CustomLogger()
    custom.set_prefix(output_dir)
    return [
        ProgressLogger(),
        ConsoleLogger(),
        custom,
        logger
    ]


def get_automl(args):
    return AutoML(
        input=MatrixContinuousDense(),
        output=CategoricalVector(),
        search_algorithm=PESearch,
        search_iterations=args.iterations,
        score_metric=accuracy,
        search_kwargs=dict(
            pop_size=args.popsize,
            selection=args.selection,
            evaluation_timeout=args.timeout,
            memory_limit=args.memory * Gb,
            early_stop=10,
        ),
    )


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


def setup_pipeline(train_dict, test_dict, score_metric):
    pipeline = get_pipeline(log_grammar=True)
    fitness_fn = make_fn(train_dict, test_dict, score_metric)
    return PESearch(
        pipeline,
        fitness_fn,
        pop_size=args.popsize,
        selection=args.selection,
        evaluation_timeout=args.timeout,
        memory_limit=args.memory * Gb,
        early_stop=True,
    )


def make_fn(train_dict, test_dict, score_fn):
    X_train, y_train = get_x_y_from_dict(train_dict)
    X_test, y_test = get_x_y_from_dict(test_dict)

    def fitness_fn(pipeline):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        return score_fn(y_test, y_pred)

    return fitness_fn


def setup_output_dir(output_dir):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def fit_classifier(args, classifier, train_dict, test_dict, loggers):
    if isinstance(classifier, AutoML):
        X_train, y_train = get_x_y_from_dict(train_dict)
        X_test, y_test = get_x_y_from_dict(test_dict)
        classifier.fit(X_train, y_train, logger=loggers)
        score = classifier.score(X_test, y_test)
        best_pipe = classifier.best_pipeline_.sampler_.replay()
    else:
        best_pipe, score = classifier.run(args.iterations, logger=loggers)

    print(f"End of training, best score {score}\nPipe: {best_pipe}")
    return best_pipe


def save_classifier(best_pipeline, output_dir):
    pipeline_file = os.path.join(args.output_dir, "best_pipeline.pkl")
    save_pipeline(best_pipeline, pipeline_file)
    print(f"Saved best pipeline to: {pipeline_file}")


def main(args):
    setup_output_dir(args.output_dir)
    test_size = 0.33
    dataset = get_dataset(args.data_path)
    train_dict, test_dict = get_splits(dataset, test_size)

    loggers = get_loggers(args.output_dir)

    if args.autogoal:
        classifier = get_automl(args)
    else:
        classifier = setup_pipeline(train_dict, test_dict, accuracy)

    best_pipeline = fit_classifier(
        args,
        classifier,
        train_dict,
        test_dict,
        loggers
    )

    save_classifier(best_pipeline, args.output_dir)


if __name__ == "__main__":
    args = parse_flags()
    main(args)
