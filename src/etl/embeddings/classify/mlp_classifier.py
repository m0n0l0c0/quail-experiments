import torch
import numpy as np
import torch.nn as nn

from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline as SkPipeline

from dataset_class import Dataset
from dataset import get_x_y_from_dict, get_dataset_class_proportions
from balanced_sampling import balanced_sampling_iter
from torch.utils.data.dataset import Dataset as TorchDataset
from torch.utils.data.dataloader import DataLoader
from autogoal.grammar import (
    Discrete,
    Union,
    Continuous,
    generate_cfg,
)


default_device = torch.device("cuda", index=1)


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
        self.score = self.binary_acc

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

    def score_fn_wrapper(self, score_fn):
        def fn(y_pred, y_test):
            preds = torch.round(torch.sigmoid(y_pred))
            preds = preds.detach().cpu().numpy()
            y = y_test.cpu().numpy()
            return score_fn(preds, y)

        return fn

    def set_score_fn(self, score_fn):
        self.score = self.score_fn_wrapper(score_fn)

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


class EmbeddingsDataset(TorchDataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if self.y is None:
            return self.X[index]
        return (self.X[index], self.y[index])


class MLPPipeline(SkPipeline):
    def __init__(self, classifier: Union("Classification", MLPClassifier)):
        self.classifier = classifier
        super(MLPPipeline, self).__init__([("class", classifier)])


def std_train_classifier(
    classifier,
    train_epochs,
    train_dict,
    test_dict,
    feature_set,
    batch_size,
    score_fn,
    print_result=True,
):
    from mlp_classification import GPU_DEVICE
    dataset_rounds = get_dataset_class_proportions(train_dict)
    if not classifier.is_initialized:
        hidden_size = get_hidden_size(train_dict, feature_set)
        classifier.initialize(hidden_size, device=GPU_DEVICE)

    classifier.set_score_fn(score_fn)
    for epoch in range(1, train_epochs + 1):
        classifier.model.train()
        epoch_loss = 0
        epoch_acc = 0
        for X_train, y_train in balanced_sampling_iter(
            dataset_rounds,
            train_dict,
            feature_set,
            dont_reshape=False
        ):
            embed_dataset = EmbeddingsDataset(X_train, y_train)
            train_loader = DataLoader(
                dataset=embed_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            for batch in train_loader:
                X_train, y_train = batch
                loss, acc = classifier.fit(X_train, y_train)
                epoch_loss += (loss / dataset_rounds)
                epoch_acc += (acc / dataset_rounds)

        if print_result:
            eval_acc, y_test, y_preds = eval_classifier(
                classifier,
                test_dict,
                feature_set,
                batch_size,
                score_fn=score_fn,
                print_result=False,
                return_y=True
            )
            status = f"Epoch {epoch+0:03}: | "
            status += f"Loss: {epoch_loss/len(train_loader):.5f} | "
            status += f"Acc: {epoch_acc/len(train_loader):.3f} | "
            status += f"Eval Acc: {eval_acc:.3f} | "
            print(status)

            if epoch % 20 == 0:
                print(classification_report(y_test, y_preds))


def std_eval_classifier(
    classifier,
    test_dict,
    feature_set,
    batch_size,
    score_fn,
    print_result=True,
    return_y=False,
):
    y_preds_list = None
    classifier.model.eval()
    X_test, y_test = get_x_y_from_dict(
        test_dict, features=feature_set, cast=np.float32
    )
    embed_dataset = EmbeddingsDataset(X_test)
    test_loader = DataLoader(
        dataset=embed_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    for batch in test_loader:
        y_preds = classifier.predict(batch)
        if y_preds_list is None:
            y_preds_list = y_preds
        else:
            y_preds_list = np.concatenate([y_preds_list, y_preds], axis=0)

    y_preds_list = y_preds_list.squeeze()
    if print_result:
        print(classification_report(y_test, y_preds_list))

    output = score_fn(y_test, y_preds_list)
    if return_y:
        output = (output, y_test, y_preds_list)

    return output


def scatter_train_classifier(
    classifier,
    train_epochs,
    train_dict,
    test_dict,
    feature_set,
    batch_size,
    score_fn,
    print_result=True,
):
    from mlp_classification import GPU_DEVICE
    if not classifier.is_initialized:
        hidden_size = get_hidden_size(train_dict, feature_set)
        classifier.initialize(hidden_size, device=GPU_DEVICE)

    classifier.set_score_fn(score_fn)
    train_dict.prepare_for_train(feature_set)
    for epoch in range(1, train_epochs + 1):
        classifier.model.train()
        epoch_loss = 0
        epoch_acc = 0

        train_loader = DataLoader(
            dataset=train_dict,
            batch_size=batch_size,
            shuffle=True
        )
        for batch in train_loader:
            X_train, y_train = batch
            loss, acc = classifier.fit(X_train, y_train)
            epoch_loss += loss
            epoch_acc += acc

        if print_result:
            eval_acc, y_test, y_preds = scatter_eval_classifier(
                classifier,
                test_dict,
                feature_set,
                batch_size,
                score_fn=score_fn,
                print_result=False,
                return_y=True
            )
            status = f"Epoch {epoch+0:03}: | "
            status += f"Loss: {epoch_loss/len(train_loader):.5f} | "
            status += f"Acc: {epoch_acc/len(train_loader):.3f} | "
            status += f"Eval Acc: {eval_acc:.3f} | "
            print(status)

            if epoch % 20 == 0:
                print(classification_report(y_test, y_preds))


def scatter_eval_classifier(
    classifier,
    test_dict,
    feature_set,
    batch_size,
    score_fn,
    print_result=True,
    return_y=False,
):
    y_preds_list = None
    classifier.model.eval()
    X_test, y_test = test_dict.get_x_y_from_dict(features=feature_set)
    X_test.prepare_for_eval(feature_set)
    y_test = y_test.to_list()
    test_loader = DataLoader(
        dataset=X_test,
        batch_size=batch_size,
        shuffle=False
    )
    for batch in test_loader:
        y_preds = classifier.predict(batch)
        if y_preds_list is None:
            y_preds_list = y_preds
        else:
            y_preds_list = np.concatenate([y_preds_list, y_preds], axis=0)

    y_preds_list = y_preds_list.squeeze()
    if print_result:
        print(classification_report(y_test, y_preds_list))

    output = score_fn(y_test, y_preds_list)
    if return_y:
        output = (output, y_test, y_preds_list)

    return output


def train_classifier(classifier, **kwargs):
    if isinstance(kwargs["train_dict"], Dataset):
        train_fn = scatter_train_classifier
    else:
        train_fn = std_train_classifier

    return train_fn(classifier, **kwargs)


def eval_classifier(classifier, **kwargs):
    if isinstance(kwargs["test_dict"], Dataset):
        eval_fn = scatter_eval_classifier
    else:
        eval_fn = std_eval_classifier

    return eval_fn(classifier, **kwargs)


def get_hidden_size(train_dict, features=None):
    if isinstance(train_dict, Dataset):
        sample = train_dict.first()
        return sample.shape[-1]
    else:
        X_train, _ = get_x_y_from_dict(train_dict, features=features)
        return X_train.reshape(X_train.shape[0], -1).shape[-1]


def get_pipeline(log_grammar=True):
    grammar = generate_cfg(MLPPipeline)
    if log_grammar:
        print(grammar)
    return grammar
