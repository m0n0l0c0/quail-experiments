import torch
import torch.nn as nn

from sklearn.pipeline import Pipeline as SkPipeline

from torch.utils.data.dataset import Dataset
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
        lr: Continuous(0.0001, 0.01) = 0.01,
    ):
        self.mlp_hidden_size = mlp_hidden_size
        self.mlp_dropout = mlp_dropout
        self.lr = lr
        self.device = torch.device("cuda", index=1)
        self.is_initialized = False

    def _initialize(self, input_size):
        self.input_size = input_size
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

    def _setup_input(self, in_data):
        if not torch.is_tensor(in_data):
            in_data = torch.as_tensor(in_data, dtype=torch.float32)
        return in_data.to(self.device)

    def binary_acc(self, y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum/y_test.shape[0]
        acc = torch.round(acc * 100)

        return acc

    def score(self, y_pred, y_test):
        return self.binary_acc(y_pred, y_test)

    def fit(self, X_train, y_train):
        X_train = self._setup_input(X_train)
        y_train = self._setup_input(y_train).unsqueeze(1)
        y_pred = self.model(X_train)
        loss = self.criterion(y_pred, y_train)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_value = loss.item()
        acc_value = self.binary_acc(y_pred, y_train).item()
        return loss_value, acc_value

    def predict(self, X_test, y=None):
        with torch.no_grad():
            X_test = self._setup_input(X_test)
            y_pred = torch.round(torch.sigmoid(self.model(X_test)))
            y_pred = y_pred.cpu().numpy()
        return y_pred


class EmbeddingsDataset(Dataset):
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


def get_pipeline(log_grammar=True):
    grammar = generate_cfg(MLPPipeline)
    if log_grammar:
        print(grammar)
    return grammar
