from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelBinarizer

import numpy as np
from functools import partial
from typing import Optional


class KernelPerceptron(ClassifierMixin):

    def __init__(
        self,
        lr: float = 0.1,
        lambda_: float = 0.,
        window_size: Optional[int] = None,
        kernel: str = 'rbf',
        degree: float = 3,
        gamma: str = 'scale',
        coef0: float = 0,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.lambda_ = lambda_
        self.window_size = window_size
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def fit(self, X, y):
        if len(np.unique(y)) != 2:
            raise ValueError(
                "KernelPerceptron only supports binary classification!")
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

        # t = 0
        for t in range(1, window_size):
            ft = self.kernel_func(X[t:t + 1],
                                  X[:t]) @ self.alpha[:t] + self.bias
            self.alpha[:t] *= (1 - self.lambda_ * self.lr)
            if y[t] * ft <= 0:
                self.alpha[t] = y[t] * self.lr
                self.bias += self.lr * y[t]

        for t in range(window_size, l):
            ft = np.matmul(
                self.kernel_func(X[t:t + 1], X[t - window_size:t]),
                self.alpha,
            ) + self.bias
            self.alpha[:-1] = self.alpha[1:] * (1 - self.lambda_ * self.lr)
            if y[t] * ft < 0:
                self.alpha[-1] = y[t] * self.lr
                self.bias += self.lr * y[t]
            else:
                self.alpha[-1] = 0.

        self.budget = X[-window_size:]
        return self

    def decision_function(self, X):
        sv = self.alpha != 0.
        return self.kernel_func(X, self.budget[sv]) @ self.alpha[sv] + self.bias

    def predict(self, X):
        score = self.decision_function(X)
        score[score >= 0] = 1
        score[score != 1] = -1
        return self.label_binarizer.inverse_transform(score)

    def register_kernel(self, std: float):
        '''注册核函数
        
        Parameters
        ----------
        std : 输入数据的标准差，用于rbf='scale'的情况
        '''
        if type(self.gamma) == str:
            gamma = {
                'scale': 1 / (self.n_features * std),
                'auto': 1 / self.n_features,
            }[self.gamma]
        else:
            gamma = self.gamma
        return {
            "linear":
            self.lin_kernel,
            "poly":
            partial(
                self.poly_kernel,
                degree=self.degree,
                gamma=gamma,
                coef0=self.coef0,
            ),
            "rbf":
            partial(
                self.rbf_kernel,
                gamma=gamma,
            ),
            "sigmoid":
            partial(
                self.sigmoid_kernel,
                gamma=gamma,
                coef0=self.coef0,
            ),
        }[self.kernel]

    @staticmethod
    def lin_kernel(x, y):
        return np.matmul(x, y.T)

    @staticmethod
    def poly_kernel(x, y, degree, gamma, coef0):
        return (gamma * np.matmul(x, y.T) + coef0)**degree

    @staticmethod
    def rbf_kernel(x, y, gamma):
        x2 = np.sum(x**2, -1, keepdims=True)
        y2 = np.sum(y**2, -1)
        xy = np.matmul(x, y.T)
        return np.exp(-gamma * (x2 + y2 - 2 * xy))

    @staticmethod
    def sigmoid_kernel(x, y, gamma, coef0):
        return np.tanh(gamma * np.matmul(x, y.T) + coef0)
