from sklearn.preprocessing import MinMaxScaler as SkMinMaxScaler
from sklearn.ensemble import RandomForestClassifier as SkRandomForestClassifier
from autogoal.contrib.sklearn import DecisionTreeClassifier as DT
from autogoal.contrib.sklearn import (
    TruncatedSVD,
    ComplementNB,
    SVC,
    LogisticRegression,
    OneClassSVM,
    SGDClassifier,
    KNeighborsClassifier,
)
from autogoal.grammar import (
    Boolean,
    Categorical,
    Continuous,
    Discrete,
)

sgd_avail_losses = [
    'epsilon_insensitive',
    'hinge',
    'huber',
    'log',
    'modified_huber',
    'perceptron',
    'squared_epsilon_insensitive',
    'squared_hinge',
    'squared_loss',
]


# normalizers
class MinMaxScaler(SkMinMaxScaler):
    def __init__(self):
        super(MinMaxScaler, self).__init__(feature_range=(0, 1))


# decomposers
class NoOp:
    def fit_transform(self, X, y=None):
        return X

    def transform(self, X, y=None):
        return X

    def __repr__(self):
        return "NoOp()"


class SVD(TruncatedSVD):
    def __init__(
        self,
        n: Discrete(50, 200),
        n_iter: Discrete(1, 10),
        tol: Continuous(0.0, 1.0)
    ):
        super(SVD, self).__init__(n_components=n, n_iter=n_iter, tol=tol)
        self.n = n
        self.n_iter = n_iter
        self.tol = tol


class SVDWithoutLogits(TruncatedSVD):
    def __init__(
        self,
        n: Discrete(50, 200),
        n_iter: Discrete(1, 10),
        tol: Continuous(0.0, 1.0),
    ):
        super(SVDWithoutLogits, self).__init__(
            n_components=n, n_iter=n_iter, tol=tol
        )
        self.n = n
        self.n_iter = n_iter
        self.tol = tol
        self.n_logits = 3

    def fit_transform(self, X, y=None):
        X_reshaped = X[:, :-self.n_logits]
        return super().fit_transform(X_reshaped, y)

    def transform(self, X):
        X_reshaped = X[:, :-self.n_logits]
        return super().transform(X_reshaped)


# classifiers
class LR(LogisticRegression):
    def __init__(
        self,
        penalty: Categorical("l1", "l2"),
        reg: Continuous(0.1, 10),
    ):
        super(LR, self).__init__(
            penalty=penalty,
            C=reg,
            solver="liblinear",
        )
        self.penalty = penalty
        self.reg = reg


class RandomForest(SkRandomForestClassifier):
    def __init__(
        self,
        n_estimators: Discrete(100, 200),
    ):
        super(RandomForest, self).__init__(n_estimators=n_estimators)
        self.n_estimators = n_estimators


class SVM(SVC):
    def __init__(
        self,
        kernel: Categorical("rbf", "linear", "poly"),
        reg: Continuous(0.1, 10)
    ):
        super(SVM, self).__init__(C=reg, kernel=kernel)
        self.kernel = kernel
        self.reg = reg


class ComplBN(ComplementNB):
    def __init__(self, fit_prior: Boolean(), norm: Boolean()):
        super(ComplBN, self).__init__(fit_prior=fit_prior, norm=norm)
        self.fit_prior = fit_prior
        self.norm = norm


class OCSVM(OneClassSVM):
    def __init__(
        self,
        kernel: Categorical('linear', 'poly', 'rbf', 'sigmoid'),
        degree: Discrete(min=1, max=5),
        gamma: Categorical('auto', 'scale'),
        coef0: Continuous(min=-0.992, max=0.992),
        shrinking: Boolean(),
        cache_size: Discrete(min=1, max=399),
    ):
        super(OCSVM, self).__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            cache_size=cache_size,
        )
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.cache_size = cache_size


class SGD(SGDClassifier):
    def __init__(
        self,
        loss: Categorical(*sgd_avail_losses),
        penalty: Categorical('elasticnet', 'l1', 'l2'),
        l1_ratio: Continuous(min=0.001, max=0.999),
        fit_intercept: Boolean(),
        tol: Continuous(min=-0.005, max=0.001),
        shuffle: Boolean(),
        epsilon: Continuous(min=-0.992, max=0.993),
        learning_rate: Categorical('optimal'),
        eta0: Continuous(min=-0.992, max=0.992),
        power_t: Continuous(min=-4.995, max=4.991),
        early_stopping: Boolean(),
        validation_fraction: Continuous(min=0.006, max=0.993),
        n_iter_no_change: Discrete(min=1, max=9),
        average: Boolean(),
    ):
        super(SGD, self).__init__(
            loss=loss,
            penalty=penalty,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            tol=tol,
            shuffle=shuffle,
            epsilon=epsilon,
            learning_rate=learning_rate,
            eta0=eta0,
            power_t=power_t,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            average=average,
        )
        self.loss = loss
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.shuffle = shuffle
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.average = average


class KNN(KNeighborsClassifier):
    def __init__(
        self,
        n_neighbors: Discrete(min=1, max=9),
        weights: Categorical('distance', 'uniform'),
        algorithm: Categorical('auto', 'ball_tree', 'brute', 'kd_tree'),
        leaf_size: Discrete(min=1, max=59),
        p: Discrete(min=1, max=3),
        metric: Categorical('minkowski'),
    ):
        super(KNN, self).__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
        )
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric


NORMALIZERS = [
    NoOp,
    MinMaxScaler,
]

DECOMPOSERS = [
    NoOp,
    SVD,
    SVDWithoutLogits,
]

CLASSIFIERS = [
    LR,
    RandomForest,
    SVM,
    DT,
    ComplBN,
    OCSVM,
    SGD,
    KNN,
]
