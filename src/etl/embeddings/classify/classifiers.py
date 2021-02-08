import torch
import numpy as np
import torch.nn as nn

from dataset_class import Dataset
from sklearn.ensemble import RandomForestClassifier as SkRandomForestClassifier
from sklearn.preprocessing import MinMaxScaler as SkMinMaxScaler
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


# Utility classes to stream a Dataset from dataset_class
# avoids loading all datapoints in RAM
class PartialEstimateOnDataset(object):
    def fit(self, X, y=None):
        if not isinstance(X, Dataset):
            return super().fit_transform(X, y)

        iter_args = dict(x=True, y=True, return_dict=False, batch_size=100)
        for _X, _y in X.iter(**iter_args):
            super().partial_fit(_X, _y)

        return self

    def fit_transform(self, X, y=None):
        if not isinstance(X, Dataset):
            return super().fit_transform(X, y)

        self.fit(X, y)
        return self.transform(X, y)

    def transform(self, X, y=None):
        if not isinstance(X, Dataset):
            return super().transform(X)

        samples = []
        iter_args = dict(x=True, y=False, return_dict=False, batch_size=100)
        for batch in X.iter(**iter_args):
            samples.append(super().transform(batch))

        return np.concatenate(samples)


class EstimateOnDataset(object):
    def _cast(self, data):
        if data is None or not isinstance(data, Dataset):
            return data

        return np.array(data.to_list())

    def fit(self, X, y=None):
        if not isinstance(X, Dataset):
            # just in case of supervised learning
            return super().fit(X, self._cast(y))

        _X = self._cast(X)
        _y = self._cast(y)
        super().fit(_X, _y)
        return self

    def fit_transform(self, X, y=None):
        if not isinstance(X, Dataset):
            return super().fit_transform(X, y)

        # avoid loading everything to memory again...
        _X = self._cast(X)
        _y = self._cast(y)
        self.fit(_X, _y)
        return self.transform(_X, _y)

    def transform(self, X, y=None):
        if not isinstance(X, Dataset):
            return super().transform(X)

        return super().transform(self._cast(X))


# normalizers
class MinMaxScaler(PartialEstimateOnDataset, SkMinMaxScaler):
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


# No partial_fit:
#   - SVD
#   - LogisticRegresion
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
class LR(EstimateOnDataset, LogisticRegression):
    def __init__(
        self,
        penalty: Categorical("l1", "l2"),
        reg: Continuous(0.1, 10),
    ):
        super(LR, self).__init__(
            penalty=penalty,
            C=reg,
            solver="liblinear",
            fit_intercept=False,
            multi_class="auto",
            dual=False,
        )
        self.penalty = penalty
        self.reg = reg


class RandomForest(EstimateOnDataset, SkRandomForestClassifier):
    def __init__(
        self,
        n_estimators: Discrete(100, 200),
    ):
        super(RandomForest, self).__init__(n_estimators=n_estimators)
        self.n_estimators = n_estimators


class SVM(EstimateOnDataset, SVC):
    def __init__(
        self,
        kernel: Categorical("rbf", "linear", "poly"),
        reg: Continuous(0.1, 10)
    ):
        super(SVM, self).__init__(C=reg, kernel=kernel)
        self.kernel = kernel
        self.reg = reg


class ComplBN(EstimateOnDataset, ComplementNB):
    def __init__(self, fit_prior: Boolean(), norm: Boolean()):
        super(ComplBN, self).__init__(fit_prior=fit_prior, norm=norm)
        self.fit_prior = fit_prior
        self.norm = norm


class OCSVM(EstimateOnDataset, OneClassSVM):
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


class SGD(EstimateOnDataset, SGDClassifier):
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
        validation_fraction: Continuous(min=0.1, max=0.8),
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


class KNN(EstimateOnDataset, KNeighborsClassifier):
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


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size=768, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_1 = nn.Linear(input_size, self.hidden_size)
        self.layer_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.layer_out = nn.Linear(self.hidden_size, 1)

        for layer in (self.layer_1, self.layer_2, self.layer_out):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.batchnorm1 = nn.BatchNorm1d(self.hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(self.hidden_size)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.dropout1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout2(x)
        x = self.layer_out(x)

        return x


class MLPClassifier():
    def __init__(
        self,
        mlp_hidden_size: Discrete(64, 256) = 256,
        mlp_dropout: Continuous(0.0, 0.3) = 0.3,
        lr: Continuous(0.00005, 0.01) = 0.01,
    ):
        self.mlp_hidden_size = mlp_hidden_size
        self.mlp_dropout = mlp_dropout
        self.lr = lr
        self.is_initialized = False
        self.score_fn = None

    def initialize(self, input_size, device=None, score_fn=None):
        self.input_size = input_size
        if device is None:
            device = torch.device("cuda", index=0)
        self.device = device
        self.model = MLP(
            self.input_size,
            hidden_size=self.mlp_hidden_size,
            dropout=self.mlp_dropout
        )
        self.model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-5
        )
        self.is_initialized = True
        if score_fn is not None:
            self.set_score_fn(score_fn)

    def _setup_input(self, in_data):
        if not torch.is_tensor(in_data):
            in_data = torch.as_tensor(in_data, dtype=torch.float32)
        return in_data.to(self.device)

    def binary_acc(self, y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum / y_test.shape[0]
        acc = torch.round(acc * 100)

        return acc.item()

    def set_score_fn(self, score_fn):
        self.score_fn = score_fn

    def score(self, y_pred, y_test):
        res = None
        if self.score_fn is None:
            res = self.binary_acc(y_pred, y_test)
        else:
            preds = torch.round(torch.sigmoid(y_pred))
            preds = preds.detach().cpu().numpy()
            y = y_test.cpu().numpy()
            res = self.score_fn(preds, y)
        return res

    def fit(self, X_train, y_train):
        X_train = self._setup_input(X_train)
        y_train = self._setup_input(y_train).unsqueeze(1)
        y_pred = self.model(X_train)
        loss = self.criterion(y_pred, y_train)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_value = loss.item()
        score_value = self.score(y_pred, y_train)
        return loss_value, score_value

    def predict(self, X_test, y=None):
        with torch.no_grad():
            X_test = self._setup_input(X_test)
            y_pred = torch.round(torch.sigmoid(self.model(X_test)))
            y_pred = y_pred.cpu().numpy()
        return y_pred


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
    ComplBN,
    OCSVM,
    SGD,
    KNN,
]
