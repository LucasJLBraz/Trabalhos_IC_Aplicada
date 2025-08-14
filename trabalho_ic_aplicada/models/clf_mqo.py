# trabalho_ic_aplicada/models/clf_mqo.py
import numpy as np

__all__ = ["LeastSquaresClassifier"]

class LeastSquaresClassifier:
    """
    Classificador linear de MQ (multi-classe) com L2 opcional.
    Inspirado no seu ridge MQO para regressão, estendido p/ one-hot.  # ver refs
    """
    def __init__(self, l2: float = 0.0):
        self.l2 = l2
        self.W_ = None  # (d+1) x C

    @staticmethod
    def _add_bias(X):
        return np.c_[np.ones((X.shape[0], 1)), X]

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int | None = None):
        Xb = self._add_bias(X)
        C = n_classes if n_classes is not None else int(y.max()) + 1
        Y = np.zeros((X.shape[0], C)); Y[np.arange(X.shape[0]), y.astype(int)] = 1.0
        if self.l2 > 0:
            A = Xb.T @ Xb + self.l2 * np.eye(Xb.shape[1])
            B = Xb.T @ Y
            self.W_ = np.linalg.solve(A, B)
        else:
            self.W_ = np.linalg.pinv(Xb) @ Y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xb = self._add_bias(X)
        scores = Xb @ self.W_
        return np.argmax(scores, axis=1)
