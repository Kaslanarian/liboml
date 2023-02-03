import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelBinarizer


class Perceptron(ClassifierMixin):

    def __init__(self, lr: float = 1.) -> None:
        super().__init__()
        self.lr = lr

    def fit(self, X: np.ndarray, y: np.ndarray):
        if len(np.unique(y)) != 2:
            raise ValueError("Perceptron only supports binary classification!")
        self.label_binarizer = LabelBinarizer(neg_label=-1).fit(y)
        y = self.label_binarizer.transform(y)
        X = np.concatenate(X, np.ones((X.shape[0], 1)), axis=1)

        self.coef_ = np.zeros(X.shape[1])

        T = X.shape[0]
        for t in range(T):
            xt, yt = X[t], y[t]
            if yt * self.decision_function(xt) <= 0:
                self.coef_ += self.lr * yt * xt

        return self

    def decision_function(self, x):
        return x @ self.coef_[:-1] + self.coef_[-1]

    def predict(self, X):
        pred = self.decision_function(X)
        pred[pred < 0] = -1
        pred[pred >= 0] = 1
        return self.label_binarizer.inverse_transform(pred)
