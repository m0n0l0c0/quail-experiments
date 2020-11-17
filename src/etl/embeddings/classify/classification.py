import os
import argparse

from pathlib import Path

from utils import get_loggers
from pipeline import get_pipeline, save_pipeline
from dataset import (
    get_splits,
    get_dataset,
    get_x_y_from_dict,
)

from autogoal.ml.metrics import accuracy
from autogoal.utils import Gb
from autogoal.ml import AutoML
from autogoal.kb import (
    MatrixContinuousDense,
    CategoricalVector,
)
from autogoal.search import PESearch


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
