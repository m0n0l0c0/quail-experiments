import os

from pickle import Pickler, Unpickler
from pathlib import Path
from autogoal.search import (
    ConsoleLogger,
    MemoryLogger,
    ProgressLogger,
    Logger,
)


class CustomLogger(Logger):
    def error(self, e: Exception, solution):
        out_file = os.path.join(
            self.get_prefix(), "classifier_errors.log"
        )
        if e and solution:
            with open(out_file, "a") as fp:
                fp.write(f"solution={repr(solution)}\nerror={e}\n\n")

    def update_best(self, new_best, new_fn, *args):
        out_file = os.path.join(
            self.get_prefix(), "classifier.log"
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


def get_save_load_name(feature_set):
    fname = "classifier"
    if feature_set is not None:
        fname = f"{fname}_{'_'.join(feature_set)}"

    fname += ".pkl"
    return fname


def save_args(args, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    args_fname = os.path.join(output_dir, "training_args.pkl")
    Pickler(open(args_fname, "wb")).dump(args)
    print(f"Saved args to: {args_fname}")


def save_classifier(classifier, output_dir, feature_set=None):
    fname = get_save_load_name(feature_set)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    classifier_fname = os.path.join(output_dir, fname)
    Pickler(open(classifier_fname, "wb")).dump(classifier)
    print(f"Saved classifier to: {classifier_fname}")


def load_classifier(output_dir, feature_set=None):
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        raise RuntimeError(
            "Requested to load a classifier from an invalid path "
            f"`{output_dir}`"
        )

    classifier_fname = output_dir
    if output_dir_path.is_dir():
        fname = get_save_load_name(feature_set)
        classifier_fname = os.path.join(output_dir, fname)
    elif output_dir_path.is_file():
        classifier_fname = output_dir
    classifier = Unpickler(open(classifier_fname, "rb")).load()
    print(f"Loaded classifier from: {classifier_fname}")
    return classifier
