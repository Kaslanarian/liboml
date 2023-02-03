import torch
from torch import nn
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


class OGDBinaryClassifier(ClassifierMixin):

    def __init__(self, init_lr: float = 1.) -> None:
        super().__init__()
        self.init_lr = init_lr

    def fit(self, X: np.ndarray, y: np.ndarray):
        if len(np.unique(y)) != 2:
            raise ValueError(
                "OGDBinaryClassifier only supports binary classification!")
        self.label_binarizer = LabelBinarizer(neg_label=-1).fit(y)
        y = self.label_binarizer.transform(y)

        X, y = torch.FloatTensor(X), torch.FloatTensor(y)
        T = X.shape[0]

        self.coef_ = nn.Linear(X.shape[1], 1)
        optimizer = torch.optim.SGD(self.coef_.parameters(), lr=self.init_lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1 / np.sqrt(epoch + 1),
        )

        for t in range(T):
            xt, yt = X[t], y[t]
            loss = torch.log(1 + torch.exp(-yt * self.coef_(xt)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        return self

    @torch.no_grad()
    def decision_function(self, X):
        return self.coef_(torch.FloatTensor(X)).numpy()

    def predict(self, X):
        pred = self.decision_function(X)
        pred[pred < 0] = -1
        pred[pred >= 0] = 1
        return self.label_binarizer.inverse_transform(pred)


class OGDClassifier(ClassifierMixin):

    def __init__(self, init_lr: float = 1.) -> None:
        super().__init__()
        self.init_lr = init_lr

    def fit(self, X, y):
        self.label_encoder = OneHotEncoder(sparse=False).fit(np.c_[y])
        y = self.label_encoder.transform(np.c_[y])

        X, y = torch.FloatTensor(X), torch.FloatTensor(y)
        T = X.shape[0]

        self.coef_ = nn.Linear(X.shape[1], y.shape[1])
        optimizer = torch.optim.SGD(self.coef_.parameters(), lr=self.init_lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1 / np.sqrt(epoch + 1),
        )
        for t in range(T):
            xt, yt = X[t], y[t]
            loss = -(torch.log_softmax(self.coef_(xt), dim=0) * yt).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        return self

    @torch.no_grad()
    def decision_function(self, X):
        return self.coef_(torch.FloatTensor(X)).numpy()

    def predict(self, X):
        pred = self.decision_function(X)
        return self.label_encoder.inverse_transform(pred)
