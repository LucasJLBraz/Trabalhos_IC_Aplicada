# trabalho_ic_aplicada/models/clf_mlp.py
import numpy as np
from .optim import make_optimizer

__all__ = ["MLPClassifier"]

def _sigmoid(x): return 1/(1+np.exp(-np.clip(x, -50, 50)))
def _dsigmoid(y): return y*(1-y)
def _tanh(x): return np.tanh(x)
def _dtanh(y): return 1 - y*y
def _relu(x): return np.maximum(0.0, x)
def _drelu(y): return (y > 0).astype(float)
def _leaky(x, a=0.01): return np.where(x>0, x, a*x)
def _dleaky(y, a=0.01): return np.where(y>0, 1.0, a)
def _relu6(x): return np.minimum(np.maximum(0.0, x), 6.0)
def _drelu6(y): return ((y>0)&(y<6)).astype(float)

def _swish(x):
    return x * _sigmoid(x)

def _dswish(y, x):
    # y = x * sigmoid(x)
    sig = _sigmoid(x)
    return sig + y * (1.0 - sig)

_ACT = {
    "tanh": (_tanh, _dtanh, False),
    "sigmoid": (_sigmoid, _dsigmoid, False),
    "relu": (_relu, _drelu, False),
    "leaky_relu": (_leaky, _dleaky, False),
    "relu6": (_relu6, _drelu6, False),
    "swish": (_swish, None, True),  # precisa de x para derivada
}

def _softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(axis=1, keepdims=True)

class MLPClassifier:
    """
    MLP 1H/2H para classificação (softmax cross-entropy).
    Ativações: tanh, sigmoid, relu, leaky_relu, relu6, swish.
    Otimizadores: sgd, momentum, nesterov, rmsprop, adam.
    """
    def __init__(self, hidden=(64,), activation="relu", lr=1e-2, epochs=200, l2=0.0, opt="sgd", seed=None, clip_grad=5.0):
        self.hidden = tuple(hidden)
        self.activation = activation
        self.lr = lr; self.epochs = epochs; self.l2 = l2; self.opt_name = opt
        self.seed = seed
        self.clip_grad = float(clip_grad)
        self.params_ = None
        self.loss_history_ = []

    def _init_params(self, d, C):
        rng = np.random.default_rng(self.seed)
        sizes = [d] + list(self.hidden) + [C]
        W, b = [], []
        for i in range(len(sizes)-1):
            fan_in, fan_out = sizes[i], sizes[i+1]
            scale = np.sqrt(2.0 / max(1, fan_in))  # He simples
            W.append(rng.normal(0.0, scale, size=(fan_in, fan_out)))
            b.append(np.zeros((1, fan_out)))
        self.params_ = {"W": W, "b": b}

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int | None = None):
        n, d = X.shape
        C = n_classes if n_classes is not None else int(y.max())+1
        self._init_params(d, C)
        f_act, f_dact, needs_x = _ACT[self.activation]
        opt = make_optimizer(self.opt_name, lr=self.lr)
        Y = np.zeros((n, C)); Y[np.arange(n), y.astype(int)] = 1.0

        clip = self.clip_grad

        for _ in range(self.epochs):
            # FORWARD
            A = [X]; Zs = []
            for i in range(len(self.params_["W"]) - 1):
                Z = A[-1] @ self.params_["W"][i] + self.params_["b"][i]; Zs.append(Z)
                A.append(f_act(Z))
            Z = A[-1] @ self.params_["W"][-1] + self.params_["b"][-1]; Zs.append(Z)
            P = _softmax(Z)

            # LOSS
            ce = -np.sum(Y * np.log(P + 1e-12)) / n
            reg = 0.5 * self.l2 * sum((W**2).sum() for W in self.params_["W"])
            self.loss_history_.append(float(ce + reg))

            # BACKWARD
            grads = {"W":[None]*len(self.params_["W"]), "b":[None]*len(self.params_["b"])}
            dZ = (P - Y) / n
            grads["W"][-1] = A[-1].T @ dZ + self.l2 * self.params_["W"][-1]
            grads["b"][-1] = dZ.sum(axis=0, keepdims=True)
            dA = dZ @ self.params_["W"][-1].T

            for i in range(len(self.params_["W"]) - 2, -1, -1):
                if needs_x:  # swish: deriv depende de x (pré-ativação)
                    dAct = _dswish(A[i+1], Zs[i])
                else:
                    dAct = _ACT[self.activation][1](A[i+1])
                dZ = dA * dAct
                grads["W"][i] = A[i].T @ dZ + self.l2 * self.params_["W"][i]
                grads["b"][i] = dZ.sum(axis=0, keepdims=True)
                if i > 0:
                    dA = dZ @ self.params_["W"][i].T

            # Gradient clipping (element-wise)
            if np.isfinite(clip) and clip > 0:
                for i in range(len(grads["W"])):
                    np.clip(grads["W"][i], -clip, clip, out=grads["W"][i])
                    np.clip(grads["b"][i], -clip, clip, out=grads["b"][i])

            # UPDATE
            opt(self.params_, grads)

        return self

    def _forward(self, X):
        A = X
        f_act = _ACT[self.activation][0]
        for i in range(len(self.params_["W"]) - 1):
            A = f_act(A @ self.params_["W"][i] + self.params_["b"][i])
        Z = A @ self.params_["W"][-1] + self.params_["b"][-1]
        return _softmax(Z)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
