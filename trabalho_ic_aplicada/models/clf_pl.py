# trabalho_ic_aplicada/models/clf_pl.py
import numpy as np
from .optim import make_optimizer

__all__ = ["SoftmaxRegression"]

def _softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    e = np.exp(Z); return e / e.sum(axis=1, keepdims=True)

class SoftmaxRegression:
    """
    Perceptron Logístico (softmax) com GD/otimizadores.
    """
    def __init__(self, lr=1e-2, epochs=200, l2=0.0, opt="sgd"):
        self.lr = lr; self.epochs = epochs; self.l2 = l2; self.opt_name = opt
        self.W_ = None  # (d+1) x C
        self.loss_history_ = []

    @staticmethod
    def _add_bias(X): return np.c_[np.ones((X.shape[0], 1)), X]

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int | None = None):
        Xb = self._add_bias(X)
        n, d1 = Xb.shape
        C = n_classes if n_classes is not None else int(y.max()) + 1
        Y = np.zeros((n, C)); Y[np.arange(n), y.astype(int)] = 1.0
        self.W_ = np.zeros((d1, C))
        opt = make_optimizer(self.opt_name, lr=self.lr)
        params = {"W": self.W_}
        for _ in range(self.epochs):
            Z = Xb @ self.W_
            P = _softmax(Z)
            # grad + L2
            grad_W = (Xb.T @ (P - Y)) / n + self.l2 * self.W_
            opt(params, {"W": grad_W})
            self.W_ = params["W"]
            # loss
            ce = -np.sum(Y * np.log(P + 1e-12)) / n + 0.5 * self.l2 * np.sum(self.W_ * self.W_)
            self.loss_history_.append(float(ce))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xb = self._add_bias(X)
        return _softmax(Xb @ self.W_)
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xb = self._add_bias(X)
        return _softmax(Xb @ self.W_)
