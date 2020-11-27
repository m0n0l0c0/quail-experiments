import pickle

from sklearn.pipeline import Pipeline as SkPipeline

from autogoal.ml import AutoML
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
    with open(file_path, 'wb') as fout:
        if isinstance(pipe, AutoML):
            pipe.save(fout)
        else:
            pickle.dump(pipe, fout)


def load_pipeline(file_path, autogoal_pipe=True):
    with open(file_path, 'rb') as fin:
        if not autogoal_pipe:
            pipeline = pickle.load(fin)
        else:
            pipeline = AutoML()
            pipeline.load(fin)

    return pipeline


pipeline_map = {
    "full": FullPipeline,
    "logreg": LogRegPipeline,
    "tree": TreePipeline,
}
