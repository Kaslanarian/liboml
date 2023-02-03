import torch
from torch import nn
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import OneHotEncoder

from ..util import mlp


class MLPClassifier(ClassifierMixin):

    def __init__(
        self,
        *hidden_args,
        activation=nn.ReLU,
        lr: float = .1,
    ) -> None:
        super().__init__()
        self.hidden_args = hidden_args
        self.activation = activation
        self.lr = lr

    def fit(self, X, y):
        self.label_encoder = OneHotEncoder(sparse=False).fit(np.c_[y])
        y = self.label_encoder.transform(np.c_[y])

        X, y = torch.FloatTensor(X), torch.FloatTensor(y)
        T = X.shape[0]

        layer_args = [
            X.shape[1],
            *self.hidden_args,
            y.shape[1],
        ]
        self.model = mlp(*layer_args, activation=self.activation)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        for t in range(T):
            xt, yt = X[t], y[t]
            loss = -(torch.log_softmax(self.model(xt), dim=0) * yt).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return self

    @torch.no_grad()
    def decision_function(self, X):
        return torch.log_softmax(
            self.model(torch.FloatTensor(X)),
            dim=0,
        ).numpy()

    def predict(self, X):
        pred = self.decision_function(X)
        return self.label_encoder.inverse_transform(pred)
