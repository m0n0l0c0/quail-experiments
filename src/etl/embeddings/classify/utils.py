import os
import pickle

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


def save_args(args, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    args_fname = os.path.join(output_dir, "training_args.pkl")
    with open(args_fname, "wb") as fout:
        pickle.dump(args, fout)
    print(f"Saved args to: {args_fname}")
