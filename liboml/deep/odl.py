import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import OneHotEncoder


class HedgeNet(nn.Module):

    def __init__(self, *layer_args, activation=nn.ReLU, beta=0.99) -> None:
        super().__init__()
        self.layer_args = layer_args
        self.classifier_list = nn.ModuleList([
            nn.Linear(n_layer, layer_args[-1]) for n_layer in layer_args[:-1]
        ])  # len(layer_args) - 1
        self.linear_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layer_args[i], layer_args[i + 1]),
                activation(),
            ) for i in range(len(layer_args) - 2)
        ])  # len(layer_args) - 2
        self.alpha = torch.ones(len(self.layer_args) -
                                1) / (len(self.layer_args) - 1)
        self.beta = beta

    def forward(self, x):
        output_list = []
        for i in range(len(self.linear_list)):
            output_list.append(self.classifier_list[i](x))
            x = self.linear_list[i](x)

        output_list.append(self.classifier_list[-1](x))
        return output_list

    def weighted_sum(self, outputs):
        return sum(
            [self.alpha[i] * outputs[i] for i in range(len(self.alpha))])

    def loss(self, x, y):
        outputs = self.forward(x)
        output = self.weighted_sum(outputs)
        with torch.no_grad():
            loss = torch.tensor(
                [-(torch.log_softmax(p, dim=0) * y).sum() for p in outputs])
        self.alpha = self.alpha * self.beta**loss
        self.alpha = self.alpha / self.alpha.sum()
        return -(torch.log_softmax(output, dim=0) * y).sum()


class ODLClassifier(ClassifierMixin):

    def __init__(
        self,
        *hidden_args,
        activation=nn.ReLU,
        lr: float = 1.,
        beta=0.99,
    ) -> None:
        super().__init__()
        self.hidden_args = hidden_args
        self.activation = activation
        self.lr = lr
        self.beta = beta

    def fit(self, X, y):
        self.label_encoder = OneHotEncoder().fit(np.c_[y])
        y = self.label_encoder.transform(np.c_[y])

        X, y = torch.FloatTensor(X), torch.FloatTensor(y)
        T = X.shape[0]

        layer_args = [
            X.shape[1],
            *self.hidden_args,
            y.shape[1],
        ]
        self.model = HedgeNet(
            *layer_args,
            activation=self.activation,
            beta=self.beta,
        )
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        for t in range(T):
            xt, yt = X[t], y[t]
            loss = self.model.loss(xt, yt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return self

    @torch.no_grad()
    def decision_function(self, X):
        outputs = self.model(torch.FloatTensor(X))
        return torch.log_softmax(self.model.weighted_sum(outputs)).numpy()

    def predict(self, X):
        pred = self.decision_function(X)
        return self.label_encoder.inverse_transform(pred)
