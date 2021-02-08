from pickle import Pickler, Unpickler
from sklearn.pipeline import Pipeline as SkPipeline

from autogoal.grammar import (
    Union,
    generate_cfg,
)
from classifiers import (
    LR,
    SGD,
    KNN,
    NoOp,
    NORMALIZERS,
    CLASSIFIERS,
    RandomForest,
    MinMaxScaler,
)


class FullPipeline(SkPipeline):
    def __init__(
        self,
        normalizer: Union("Normalize", *NORMALIZERS),
        classifier: Union("Classifier", *CLASSIFIERS),
    ):
        self.normalizer = normalizer
        self.classifier = classifier
        super(FullPipeline, self).__init__([
            ("norm", normalizer),
            ("class", classifier),
        ])


class LogRegPipeline(SkPipeline):
    def __init__(
        self,
        normalizer: Union("Normalize", NoOp, MinMaxScaler),
        classifier: Union("Classifier", LR),
    ):
        self.normalizer = normalizer
        self.classifier = classifier
        super(LogRegPipeline, self).__init__([
            ("norm", normalizer),
            ("class", classifier),
        ])


class TreePipeline(SkPipeline):
    def __init__(
        self,
        normalizer: Union("Normalize", NoOp, MinMaxScaler),
        classifier: Union("Classifier", RandomForest, SGD, KNN),
    ):
        self.normalizer = normalizer
        self.classifier = classifier
        super(TreePipeline, self).__init__([
            ("norm", normalizer),
            ("class", classifier),
        ])


def get_pipeline(pipe_type="full", log_grammar=True):
    grammar = generate_cfg(pipeline_map[pipe_type])
    if log_grammar:
        print(grammar)
    return grammar


def save_pipeline(pipe, file_path):
    Pickler(open(file_path, "wb")).dump(pipe)


def load_pipeline(file_path):
    Unpickler(open(file_path, "rb")).load()


pipeline_map = {
    "full": FullPipeline,
    "logreg": LogRegPipeline,
    "tree": TreePipeline,
}
