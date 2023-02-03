import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelBinarizer


class PAClassifier(ClassifierMixin):

    def __init__(self, C: float = 1., type: int = 0) -> None:
        super().__init__()
        self.C = C
        self.type = type
        if self.type not in {0, 1, 2}:
            raise ValueError("'type' parameter can only be 0, 1, or 2!")
        self.__calculate_step_size = [
            self.__calculate_step_size0,
            self.__calculate_step_size1,
            self.__calculate_step_size2,
        ][self.type]

    def fit(self, X: np.ndarray, y: np.ndarray):
        if len(np.unique(y)) != 2:
            raise ValueError(
                "PAClassifier only supports binary classification!")
        self.label_binarizer = LabelBinarizer(neg_label=-1).fit(y)
        y = self.label_binarizer.transform(y)

        X = np.concatenate(X, np.ones((X.shape[0], 1)), axis=1)
        norm_X = np.square(X).sum(1)
        self.coef_ = np.zeros(X.shape[1])
        T = X.shape[0]
        for t in range(T):
            xt, yt = X[t], y[t]
            lt = max(0, 1 - yt * (self.coef_ @ xt))
            tau = self.__calculate_step_size(lt, norm_X[t])
            self.coef_ += tau * yt * xt

        return self

    def decision_function(self, x):
        return x @ self.coef_[:-1] + self.coef_[-1]

    def predict(self, X):
        pred = self.decision_function(X)
        pred[pred < 0] = -1
        pred[pred >= 0] = 1
        return self.label_binarizer.inverse_transform(pred)

    def __calculate_step_size0(self, loss, norm2):
        return loss / norm2

    def __calculate_step_size1(self, loss, norm2):
        return min(self.C, loss / norm2)

    def __calculate_step_size2(self, loss, norm2):
        return loss / (norm2 + 1 / (2 * self.C))
