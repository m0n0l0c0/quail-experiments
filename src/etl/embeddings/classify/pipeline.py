import pickle

from classifiers import (
    # DECOMPOSERS,
    CLASSIFIERS
)
from sklearn.pipeline import Pipeline as SkPipeline

from autogoal.ml import AutoML
from autogoal.grammar import (
    Union,
    generate_cfg,
)


class Pipeline(SkPipeline):
    def __init__(
        self,
        # decomposer: Union("Decomposer", *DECOMPOSERS),
        classifier: Union("Classifier", *CLASSIFIERS),
    ):
        # self.decomposer = decomposer
        self.classifier = classifier

        # super().__init__(
        #     [("decomp", decomposer), ("class", classifier), ]
        # )

        super().__init__(
            [("class", classifier), ]
        )


def get_pipeline(log_grammar=True):
    grammar = generate_cfg(Pipeline)
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
