import torch
from torch import nn
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import OneHotEncoder

from .odl import HedgeNet
from ..util import mlp


class ODLAE(nn.Module):

    def __init__(
        self,
        n_classes: int,
        *encoder_args,
        activation=nn.ReLU,
        beta1=0.99,
        beta2=0.99,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.encoder_args = encoder_args
        self.decoder_args = encoder_args[::-1]
        self.activation = activation
        self.beta1 = beta1
        self.beta2 = beta2
        self.alpha1, self.alpha2 = 0.5, 0.5

        self.encoder = HedgeNet(
            *[*encoder_args, n_classes],
            activation=activation,
            beta=beta1,
        )
        self.decoder = mlp(*self.decoder_args, activation=activation)

    def forward(self, x):
        output_list = []
        for i in range(len(self.encoder.linear_list)):
            output_list.append(self.encoder.classifier_list[i](x))
            x = self.encoder.linear_list[i](x)

        output_list.append(self.encoder.classifier_list[-1](x))
        recon_x = self.decoder(x)
        return output_list, recon_x

    def loss(self, x, y):
        outputs, recon_x = self(x)
        output = self.encoder.weighted_sum(outputs)
        with torch.no_grad():
            loss = torch.tensor(
                [-(torch.log_softmax(p, dim=0) * y).sum() for p in outputs])
        self.encoder.alpha = self.encoder.alpha * self.beta1**loss
        self.encoder.alpha = self.encoder.alpha / self.encoder.alpha.sum()

        l_re = (x - recon_x).square().sum()
        l_pre = -(torch.log_softmax(output, dim=0) * y).sum()
        term1 = self.alpha1 * self.beta2**l_re.item()
        term2 = self.alpha2 * self.beta2**l_pre.item()
        self.alpha1 = term1 / (term1 + term2)
        self.alpha2 = 1 - self.alpha1
        return self.alpha1 * l_re + self.alpha2 * l_pre


class ODLAEClassifier(ClassifierMixin):

    def __init__(
        self,
        *hidden_args,
        activation=nn.ReLU,
        init_lr: float = 1.,
        beta1=0.99,
        beta2=0.99,
    ) -> None:
        super().__init__()
        self.hidden_args = hidden_args
        self.activation = activation
        self.init_lr = init_lr
        self.beta1 = beta1
        self.beta2 = beta2

    def fit(self, X, y):
        self.label_encoder = OneHotEncoder(sparse=False).fit(y)
        y = self.label_encoder.transform(y)

        X, y = torch.FloatTensor(X), torch.FloatTensor(y)
        T = X.shape[0]

        layer_args = [
            X.shape[1],
            *self.hidden_args,
        ]
        self.model = ODLAE(
            y.shape[1],
            *layer_args,
            activation=self.activation,
            beta1=self.beta1,
            beta2=self.beta2,
        )
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.init_lr)
        for t in range(T):
            xt, yt = X[t], y[t]
            loss = self.model.loss(xt, yt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return self

    @torch.no_grad()
    def decision_function(self, X):
        outputs = self.model(torch.FloatTensor(X))[0]
        return torch.log_softmax(
            self.model.encoder.weighted_sum(outputs),
            dim=1,
        ).numpy()

    def predict(self, X):
        pred = self.decision_function(X)
        return self.label_encoder.inverse_transform(pred)
