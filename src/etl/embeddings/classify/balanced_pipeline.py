from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.ensemble import RandomForestClassifier as SkRandomForestClassifier
from sklearn.preprocessing import MinMaxScaler as SkMinMaxScaler
from autogoal.grammar import Discrete, Union, generate_cfg
from autogoal.contrib.sklearn import DecisionTreeClassifier as DT
from classifiers import (
    NoOp,
    SVDWithoutLogits,
    SVD,
    SGD,
    KNN
)


class RandomForest(SkRandomForestClassifier):
    def __init__(
        self,
        n_estimators: Discrete(100, 200),
    ):
        super(RandomForest, self).__init__(n_estimators=n_estimators)
        self.n_estimators = n_estimators


class MinMaxScaler(SkMinMaxScaler):
    def __init__(self):
        super(MinMaxScaler, self).__init__(feature_range=(0, 1))


class Pipeline(SkPipeline):
    def __init__(
        self,
        normalizer: Union("Normalize", NoOp, MinMaxScaler),
        decomposer: Union("Decomposer", SVD, SVDWithoutLogits),
        classifier: Union("Classifier", RandomForest, DT, SGD, KNN),
    ):
        self.normalizer = normalizer
        self.decomposer = decomposer
        self.classifier = classifier
        super(Pipeline, self).__init__([
            ("norm", normalizer),
            ("decomp", decomposer),
            ("class", classifier),
        ])


def get_pipeline(log_grammar=True):
    grammar = generate_cfg(Pipeline)
    if log_grammar:
        print(grammar)
    return grammar
