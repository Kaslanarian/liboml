from sklearn.preprocessing import LabelBinarizer

import numpy as np
from typing import Optional

from .perceptron import KernelPerceptron
from ..util import sigmoid


class KernelOGD(KernelPerceptron):

    def __init__(
        self,
        lr: float = 0.1,
        lambda_: float = 1.,
        window_size: Optional[int] = None,
        kernel: str = 'rbf',
        degree: float = 3,
        gamma: str = 'scale',
        coef0: float = 0,
    ) -> None:
        super().__init__(
            lr,
            lambda_,
            window_size,
            kernel,
            degree,
            gamma,
            coef0,
        )

    def fit(self, X, y):
        if len(np.unique(y)) != 2:
            raise ValueError("KernelOGD only supports binary classification!")
        self.label_binarizer = LabelBinarizer(neg_label=-1).fit(y)
        y = self.label_binarizer.transform(y)
        l, self.n_features = X.shape
        self.kernel_func = self.register_kernel(X.std())

        if self.window_size:
            window_size = min(self.window_size, l)
        else:
            window_size = l
        self.alpha = np.zeros(window_size)
        self.bias = 0.

        for t in range(1, window_size):
            ft = self.kernel_func(X[t:t + 1],
                                  X[:t]) @ self.alpha[:t] + self.bias
            self.alpha[:t] *= (1 - self.lambda_ * self.lr)
            self.alpha[t] = self.lr * y[t] * sigmoid(y[t] * ft)
            self.bias += self.lr * y[t]

        for t in range(window_size, l):
            ft = np.matmul(
                self.kernel_func(X[t:t + 1], X[t - window_size:t]),
                self.alpha,
            ) + self.bias
            self.alpha[:-1] = self.alpha[1:] * (1 - self.lambda_ * self.lr)
            self.alpha[-1] = self.lr * y[t] * sigmoid(y[t] * ft)
            self.bias += self.lr * y[t]
    
        self.budget = X[-window_size:]
        return self
