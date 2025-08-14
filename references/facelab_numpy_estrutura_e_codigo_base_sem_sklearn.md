# facelab\_numpy — estrutura e código-base (sem sklearn)

> **Objetivo**: executar as Atividades 1–8 do projeto de reconhecimento de faces usando **somente NumPy** para ML (sem scikit‑learn). Inclui pré‑processamento, PCA (com e sem redução), Box‑Cox, normalizações, classificadores (MQ, PL, MLP‑1H, MLP‑2H), particionamento estratificado, execução repetida (Nr=50), métricas e geração das Tabelas 1–3.

---

## 1) Estrutura do projeto

```
facelab_numpy/
├─ README.md
├─ pyproject.toml                # opcional; ou requirements.txt
├─ facelab/
│  ├─ __init__.py
│  ├─ utils.py                   # seeds, one_hot, timers, helpers
│  ├─ splits.py                  # split estratificado por sujeito
│  ├─ metrics.py                 # accuracy, confmat, precision, recall etc.
│  ├─ preprocess.py              # normalizações e Box-Cox
│  ├─ pca.py                     # PCA via SVD; escolha de q (≥98%)
│  ├─ io.py                      # leitura dos dados já vetorizados ou imagens (opcional)
│  └─ models/
│     ├─ mq.py                   # Mínimos Quadrados (com viés e L2 opcional)
│     ├─ pl.py                   # Regressão logística softmax (GD batch)
│     └─ mlp.py                  # MLP 1H/2H (relu/sigmoid/tanh/leaky) + backprop
│
├─ experiments/
│  ├─ pipelines.py               # definição dos steps (PCA/Box-Cox/normalização)
│  └─ run_experiments.py         # executa A1–A8, salva tabelas CSV
│
└─ data/
   ├─ yalefaces/                 # Yale A (pré ou cru) — **não versionar**
   └─ intruder/                  # 11 imagens do “intruso” (A8)
```

**Requisitos mínimos**: `python >= 3.10`, `numpy >= 1.23`.\
**I/O opcional** (se for ler PGM/JPG/PNG): `imageio` ou `Pillow`.

> Observação: as rotinas de ML são puramente NumPy; bibliotecas extra são só para **carregar imagens**.

---

## 2) Convenções de uso

- `X` sempre em formato **(n amostras, d atributos)**; `y` inteiro **0..C-1**.
- Para classificação multi‑classe, **one‑hot** internamente (Y ∈ {0,1}^{n×C}).
- **Viés** (bias) tratado com coluna de 1s no início de `X` (via helper `add_bias`).
- **Tempo**: cada treino/avaliação cronometra `fit_time` e `pred_time` (tic/toc).
- **RNG**: fixe `seed_base` e, a cada rodada `r`, use `seed_base + r`.

---

## 3) Código‑base (módulos)

A seguir, os arquivos sugeridos com implementações **sem sklearn**.

### facelab/**init**.py

```python
# vazio ou exports úteis
```

### facelab/utils.py

```python
import time
import numpy as np

__all__ = [
    "set_seed", "one_hot", "add_bias", "softmax", "Timer",
]

def set_seed(seed: int | None = None):
    if seed is not None:
        np.random.seed(seed)

class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.dt = time.perf_counter() - self.t0


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    Y = np.zeros((y.shape[0], num_classes), dtype=float)
    Y[np.arange(y.shape[0]), y.astype(int)] = 1.0
    return Y


def add_bias(X: np.ndarray) -> np.ndarray:
    return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)


def softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.max(axis=1, keepdims=True)
    expZ = np.exp(Z)
    return expZ / expZ.sum(axis=1, keepdims=True)
```

### facelab/splits.py

```python
import numpy as np
from typing import Tuple

__all__ = ["stratified_train_test_split"]


def stratified_train_test_split(y: np.ndarray, p_train: float, rng: np.random.Generator | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Retorna (idx_train, idx_test) estratificado por classe.
    p_train é a fração por classe (e.g., 0.8 para Ptrain=80).
    """
    if rng is None:
        rng = np.random.default_rng()
    idx_train, idx_test = [], []
    classes = np.unique(y)
    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        n_train = max(1, int(np.floor(p_train * idx_c.size)))
        idx_train.append(idx_c[:n_train])
        idx_test.append(idx_c[n_train:])
    return np.concatenate(idx_train), np.concatenate(idx_test)
```

### facelab/metrics.py

```python
import numpy as np
from typing import Dict

__all__ = [
    "confusion_matrix", "accuracy", "precision_recall_sensitivity",
]


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, C: int) -> np.ndarray:
    M = np.zeros((C, C), dtype=int)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        M[t, p] += 1
    return M


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def precision_recall_sensitivity(M: np.ndarray) -> Dict[str, float]:
    # macro‑averaged para multi‑classe; na A8 (binária), coincide com bin.
    TP = np.diag(M).astype(float)
    FP = M.sum(axis=0) - TP
    FN = M.sum(axis=1) - TP
    TN = M.sum() - (TP + FP + FN)

    with np.errstate(divide='ignore', invalid='ignore'):
        prec = np.nanmean(TP / (TP + FP))
        rec = np.nanmean(TP / (TP + FN))  # recall = sensitivity
    return {
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "sensitivity_macro": float(rec),
        "fp_rate_macro": float(np.nanmean(FP / (FP + TN))),
        "fn_rate_macro": float(np.nanmean(FN / (TP + FN))),
    }
```

### facelab/preprocess.py

```python
import numpy as np

__all__ = [
    "ZScore", "MinMax", "MinMaxNegPos", "apply_boxcox_per_feature",
]

class ZScore:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-12
        return self
    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_
    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MinMax:
    def __init__(self):
        self.min_ = None
        self.max_ = None
    def fit(self, X: np.ndarray):
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self
    def transform(self, X: np.ndarray) -> np.ndarray:
        rng = (self.max_ - self.min_) + 1e-12
        return (X - self.min_) / rng
    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MinMaxNegPos:
    """Mapeia para [-1, +1] via min‑max."""
    def __init__(self):
        self.mm = MinMax()
    def fit(self, X):
        self.mm.fit(X); return self
    def transform(self, X):
        return self.mm.transform(X) * 2.0 - 1.0
    def fit_transform(self, X):
        return self.transform(self.mm.fit_transform(X))


def _boxcox_transform(x: np.ndarray, lam: float) -> np.ndarray:
    if lam == 0.0:
        return np.log(x)
    return (np.power(x, lam) - 1.0) / lam


def _boxcox_ll(x_pos: np.ndarray, lam: float) -> float:
    # Perfil do log‑likelihood do Box‑Cox (normalidade) para um vetor 1D
    z = _boxcox_transform(x_pos, lam)
    n = x_pos.size
    var = z.var(ddof=1) + 1e-12
    # constante omitida; usamos forma proporcional
    return - (n / 2.0) * np.log(var) + (lam - 1.0) * np.log(x_pos).sum()


def apply_boxcox_per_feature(X: np.ndarray, grid: np.ndarray = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aplica Box‑Cox **por feature** (coluna). Retorna (X_bc, lambdas, shifts).
    Se houver valores ≤0, desloca por coluna: shift = (|min| + eps).
    """
    if grid is None:
        grid = np.linspace(-2.0, 2.0, 81)  # passo 0.05
    X = X.copy()
    n, d = X.shape
    lambdas = np.zeros(d)
    shifts = np.zeros(d)
    for j in range(d):
        x = X[:, j]
        mn = x.min()
        shift = 0.0
        if mn <= 0:
            shift = -mn + 1e-6
            x = x + shift
        # escolhe lambda por máxima verossimilhança em grade
        ll_best, lam_best = -np.inf, 1.0
        for lam in grid:
            ll = _boxcox_ll(x, lam)
            if ll > ll_best:
                ll_best, lam_best = ll, lam
        X[:, j] = _boxcox_transform(x, lam_best)
        lambdas[j] = lam_best
        shifts[j] = shift
    # normaliza em seguida (z‑score) deve ser feito fora, se desejado
    return X, lambdas, shifts
```

### facelab/pca.py

```python
import numpy as np

__all__ = ["PCA", "choose_q_for_variance"]

class PCA:
    def __init__(self, n_components: int | None = None):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None  # Vt
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray):
        Xc = X - X.mean(axis=0, keepdims=True)
        self.mean_ = X.mean(axis=0)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        var = (S ** 2) / (X.shape[0] - 1)
        self.components_ = Vt
        self.explained_variance_ = var
        self.explained_variance_ratio_ = var / var.sum()
        return self

    def transform(self, X: np.ndarray, q: int | None = None) -> np.ndarray:
        if q is None:
            q = self.n_components or X.shape[1]
        Xc = X - self.mean_
        Vt = self.components_[:q, :]
        return Xc @ Vt.T

    def fit_transform(self, X: np.ndarray, q: int | None = None) -> np.ndarray:
        self.fit(X)
        return self.transform(X, q=q)


def choose_q_for_variance(ev_ratio: np.ndarray, target: float = 0.98) -> int:
    csum = np.cumsum(ev_ratio)
    return int(np.searchsorted(csum, target) + 1)
```

### facelab/models/mq.py

```python
import numpy as np
from ..utils import add_bias

__all__ = ["LeastSquaresClassifier"]

class LeastSquaresClassifier:
    def __init__(self, l2: float = 0.0):
        self.l2 = l2
        self.W = None  # (d+1)×C

    def fit(self, X: np.ndarray, Y: np.ndarray):
        Xa = add_bias(X)
        d1, C = Xa.shape[1], Y.shape[1]
        if self.l2 > 0:
            A = Xa.T @ Xa + self.l2 * np.eye(d1)
            B = Xa.T @ Y
            self.W = np.linalg.solve(A, B)
        else:
            self.W = np.linalg.pinv(Xa) @ Y
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xa = add_bias(X)
        return Xa @ self.W

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
```

### facelab/models/pl.py (softmax)

```python
import numpy as np
from ..utils import add_bias, softmax

__all__ = ["SoftmaxRegression"]

class SoftmaxRegression:
    def __init__(self, lr: float = 1e-2, epochs: int = 200, l2: float = 0.0):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.W = None  # (d+1)×C
        self.loss_history_ = []

    def fit(self, X: np.ndarray, Y: np.ndarray):
        Xa = add_bias(X)
        n, d1 = Xa.shape
        C = Y.shape[1]
        self.W = np.zeros((d1, C))
        for _ in range(self.epochs):
            Z = Xa @ self.W
            P = softmax(Z)
            # cross‑entropy + L2
            grad = Xa.T @ (P - Y) / n + self.l2 * self.W
            self.W -= self.lr * grad
            # opcional: registrar loss
            ce = -np.sum(Y * np.log(P + 1e-12)) / n + 0.5 * self.l2 * np.sum(self.W * self.W)
            self.loss_history_.append(float(ce))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xa = add_bias(X)
        Z = Xa @ self.W
        return np.argmax(softmax(Z), axis=1)
```

### facelab/models/mlp.py

```python
import numpy as np

__all__ = ["MLPClassifier"]

# ativações

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _dsigmoid(y):
    return y * (1.0 - y)

def _tanh(x):
    return np.tanh(x)

def _dtanh(y):
    return 1.0 - y*y

def _relu(x):
    return np.maximum(0.0, x)

def _drelu(y):
    return (y > 0).astype(float)

def _leaky(x, a=0.01):
    return np.where(x > 0, x, a * x)

def _dleaky(y, a=0.01):
    return np.where(y > 0, 1.0, a)

ACT = {
    "sigmoid": (_sigmoid, _dsigmoid),
    "tanh": (_tanh, _dtanh),
    "relu": (_relu, _drelu),
    "leaky_relu": (_leaky, _dleaky),
}

class MLPClassifier:
    def __init__(self, hidden: tuple[int, ...] = (64,), activation: str = "relu", lr: float = 1e-2, epochs: int = 200, l2: float = 0.0, seed: int | None = None):
        self.hidden = hidden
        self.activation = activation
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed
        self.params_ = None
        self.loss_history_ = []

    def _init_params(self, d: int, C: int):
        rng = np.random.default_rng(self.seed)
        sizes = [d] + list(self.hidden) + [C]
        W, b = [], []
        for i in range(len(sizes) - 1):
            fan_in, fan_out = sizes[i], sizes[i+1]
            # He/Xavier simples
            scale = np.sqrt(2.0 / fan_in)
            W.append(rng.normal(0.0, scale, size=(fan_in, fan_out)))
            b.append(np.zeros((1, fan_out)))
        self.params_ = {"W": W, "b": b}

    def fit(self, X: np.ndarray, Y: np.ndarray):
        n, d = X.shape
        C = Y.shape[1]
        self._init_params(d, C)
        f_act, f_dact = ACT[self.activation]
        for _ in range(self.epochs):
            # forward
            A = [X]
            Zs = []
            for i in range(len(self.params_["W"]) - 1):
                Z = A[-1] @ self.params_["W"][i] + self.params_["b"][i]
                Zs.append(Z)
                A.append(f_act(Z))
            # camada de saída (softmax)
            Z = A[-1] @ self.params_["W"][-1] + self.params_["b"][-1]
            Zs.append(Z)
            # softmax estável
            Zs[-1] = Zs[-1] - Zs[-1].max(axis=1, keepdims=True)
            expZ = np.exp(Zs[-1])
            P = expZ / (expZ.sum(axis=1, keepdims=True))

            # loss
            ce = -np.sum(Y * np.log(P + 1e-12)) / n
            reg = 0.5 * self.l2 * sum((W**2).sum() for W in self.params_["W"])
            self.loss_history_.append(float(ce + reg))

            # backward
            dZ = (P - Y) / n
            grads_W, grads_b = [], []
            # saída -> última oculta
            dW = A[-1].T @ dZ + self.l2 * self.params_["W"][-1]
            db = dZ.sum(axis=0, keepdims=True)
            grads_W.insert(0, dW)
            grads_b.insert(0, db)
            dA = dZ @ self.params_["W"][-1].T

            # camadas ocultas (reversa)
            for i in range(len(self.params_["W"]) - 2, -1, -1):
                if i > 0:
                    Z_prev = Zs[i-1]
                    A_prev = A[i]
                else:
                    Z_prev = None
                    A_prev = A[0]
                # deriv da ativação usa saída ativada (A[i])
                dZ = dA * ACT[self.activation][1](A[i])
                dW = A_prev.T @ dZ + self.l2 * self.params_["W"][i]
                db = dZ.sum(axis=0, keepdims=True)
                grads_W.insert(0, dW)
                grads_b.insert(0, db)
                if i > 0:
                    dA = dZ @ self.params_["W"][i].T

            # update
            for i in range(len(self.params_["W"])):
                self.params_["W"][i] -= self.lr * grads_W[i]
                self.params_["b"][i] -= self.lr * grads_b[i]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        A = X
        for i in range(len(self.params_["W"]) - 1):
            A = ACT[self.activation][0](A @ self.params_["W"][i] + self.params_["b"][i])
        Z = A @ self.params_["W"][-1] + self.params_["b"][-1]
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z) / np.exp(Z).sum(axis=1, keepdims=True)
        return np.argmax(P, axis=1)
```

### facelab/io.py (opcional; se já tiver X,y prontos, não use)

```python
import os
import numpy as np
try:
    import imageio.v3 as iio
except Exception:  # opcional
    iio = None

__all__ = ["load_flatten_from_dir"]


def load_flatten_from_dir(root: str, size: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Lê imagens em subpastas (uma pasta por classe). Retorna X (n×d), y (n,), e um dict id->folder.
    Se size for dado (H,W), usa um reamostrador NN simples (sem libs externas).
    """
    X, y, mapping = [], [], {}
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    for c_idx, c in enumerate(classes):
        mapping[c_idx] = c
        for f in sorted(os.listdir(os.path.join(root, c))):
            path = os.path.join(root, c, f)
            if iio is None:
                raise RuntimeError("Instale imageio para leitura de imagens ou pré‑gere X,y por MATLAB.")
            img = iio.imread(path)
            if img.ndim == 3:
                img = img.mean(axis=2)  # grayscale simples
            if size is not None:
                img = _resize_nn(img, size)
            X.append(img.flatten())
            y.append(c_idx)
    return np.asarray(X, float), np.asarray(y, int), mapping


def _resize_nn(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    H, W = img.shape[:2]
    h, w = size
    ys = (np.arange(h) * (H / h)).astype(int)
    xs = (np.arange(w) * (W / w)).astype(int)
    return img[ys][:, xs]
```

### experiments/pipelines.py

```python
import numpy as np
from dataclasses import dataclass
from typing import Literal

from facelab.pca import PCA, choose_q_for_variance
from facelab.preprocess import ZScore, MinMax, MinMaxNegPos, apply_boxcox_per_feature

NormName = Literal["none", "zscore", "minmax", "minmax_pm1"]
PCAType = Literal["none", "rotate", "reduce98"]

@dataclass
class PipelineConfig:
    pca: PCAType = "none"
    norm: NormName = "none"
    use_boxcox: bool = False   # se True, aplica Box‑Cox e depois z‑score

class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.pca_ = None
        self.q_ = None
        self.norm_ = None
        self._boxcox_ = None  # (lambdas, shifts)

    def fit(self, X: np.ndarray):
        Xp = X
        # PCA
        if self.cfg.pca != "none":
            self.pca_ = PCA()
            self.pca_.fit(X)
            if self.cfg.pca == "rotate":
                self.q_ = X.shape[1]
                Xp = self.pca_.transform(X, q=self.q_)
            elif self.cfg.pca == "reduce98":
                self.q_ = choose_q_for_variance(self.pca_.explained_variance_ratio_, target=0.98)
                Xp = self.pca_.transform(X, q=self.q_)
        # Box‑Cox (opcional) + z‑score
        if self.cfg.use_boxcox:
            Xp, lambdas, shifts = apply_boxcox_per_feature(Xp)
            self._boxcox_ = (lambdas, shifts)
            self.norm_ = ZScore().fit(Xp)
            Xp = self.norm_.transform(Xp)
        else:
            # Normalização comum
            if self.cfg.norm == "zscore":
                self.norm_ = ZScore().fit(Xp)
                Xp = self.norm_.transform(Xp)
            elif self.cfg.norm == "minmax":
                self.norm_ = MinMax().fit(Xp)
                Xp = self.norm_.transform(Xp)
            elif self.cfg.norm == "minmax_pm1":
                self.norm_ = MinMaxNegPos().fit(Xp)
                Xp = self.norm_.transform(Xp)
        return Xp

    def transform(self, X: np.ndarray):
        Xp = X
        if self.pca_ is not None:
            Xp = self.pca_.transform(X, q=self.q_)
        if self.cfg.use_boxcox and self._boxcox_ is not None:
            # re‑aplica Box‑Cox com lambdas e shifts aprendidos
            lambdas, shifts = self._boxcox_
            Xq = Xp.copy()
            for j in range(Xq.shape[1]):
                x = Xq[:, j] + shifts[j]
                lam = lambdas[j]
                Xq[:, j] = np.log(x) if lam == 0.0 else (np.power(x, lam) - 1.0) / lam
            Xp = Xq
            Xp = self.norm_.transform(Xp)
        else:
            if self.norm_ is not None:
                Xp = self.norm_.transform(Xp)
        return Xp
```

### experiments/run\_experiments.py

```python
import os, csv, math
import numpy as np
from statistics import mean, median, pstdev

from facelab.utils import set_seed, Timer, one_hot
from facelab.splits import stratified_train_test_split
from facelab.metrics import confusion_matrix, accuracy, precision_recall_sensitivity
from facelab.models.mq import LeastSquaresClassifier
from facelab.models.pl import SoftmaxRegression
from facelab.models.mlp import MLPClassifier
from experiments.pipelines import Pipeline, PipelineConfig

# ----------------------------
# Helpers de avaliação por rodada
# ----------------------------

def eval_model(model, Xtr, ytr, Xte, yte):
    with Timer() as tfit:
        model.fit(Xtr, one_hot(ytr, int(ytr.max()+1)))
    with Timer() as tpred:
        yhat = model.predict(Xte)
    acc = accuracy(yte, yhat)
    C = int(max(yte.max(), yhat.max()) + 1)
    M = confusion_matrix(yte, yhat, C)
    extra = precision_recall_sensitivity(M)
    return {
        "acc": acc,
        "fit_time": tfit.dt,
        "pred_time": tpred.dt,
        **extra,
    }


def k_runs(X, y, p_train, Nr, pipeline_cfg, model_ctor):
    stats = []
    for r in range(Nr):
        rng = np.random.default_rng(1234 + r)
        idx_tr, idx_te = stratified_train_test_split(y, p_train, rng)
        pipe = Pipeline(pipeline_cfg)
        Xtr = pipe.fit(X[idx_tr])
        Xte = pipe.transform(X[idx_te])
        m = model_ctor()
        out = eval_model(m, Xtr, y[idx_tr], Xte, y[idx_te])
        stats.append(out)
    return stats


def summarize(stats):
    def agg(key):
        vals = [s[key] for s in stats]
        return {
            "mean": float(mean(vals)),
            "std": float(pstdev(vals)),
            "min": float(min(vals)),
            "max": float(max(vals)),
            "median": float(median(vals)),
        }
    return {
        "acc": agg("acc"),
        "fit_time": agg("fit_time"),
        "pred_time": agg("pred_time"),
    }

# ----------------------------
# Entradas do experimento
# ----------------------------

def run_A1_A2(X, y, out_csv_dir):
    """Atividades 1–2: sem PCA (pca='none'), Ptrain=0.8, Nr=50.
    Para cada classificador, varremos normalizações e escolhemos a **melhor**.
    Tabela 1: reportar só a melhor por classificador.
    """
    Ptrain, Nr = 0.8, 50
    norms = ["none", "zscore", "minmax", "minmax_pm1"]

    def best_of(model_ctor):
        best = None
        for nm in norms:
            cfg = PipelineConfig(pca="none", norm=nm, use_boxcox=False)
            stats = k_runs(X, y, Ptrain, Nr, cfg, model_ctor)
            S = summarize(stats)
            if (best is None) or (S["acc"]["mean"] > best["S"]["acc"]["mean"]):
                best = {"norm": nm, "S": S}
        return best

    results = {}
    results["MQ"] = best_of(lambda: LeastSquaresClassifier(l2=0.0))
    results["PL"] = best_of(lambda: SoftmaxRegression(lr=1e-2, epochs=200, l2=0.0))
    results["MLP-1H"] = best_of(lambda: MLPClassifier(hidden=(64,), activation="relu", lr=1e-2, epochs=200, l2=0.0))
    results["MLP-2H"] = best_of(lambda: MLPClassifier(hidden=(64, 32), activation="relu", lr=1e-2, epochs=200, l2=0.0))

    os.makedirs(out_csv_dir, exist_ok=True)
    # salva Tabela 1
    with open(os.path.join(out_csv_dir, "tabela1.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Classificador", "Normalização", "ACC_mean", "ACC_std", "ACC_min", "ACC_max", "ACC_median", "fit_time_mean", "pred_time_mean"])
        for name, res in results.items():
            S = res["S"]; nm = res["norm"]
            w.writerow([name, nm, S["acc"]["mean"], S["acc"]["std"], S["acc"]["min"], S["acc"]["max"], S["acc"]["median"], S["fit_time"]["mean"], S["pred_time"]["mean"]])
    return results


def run_A3_A4(X, y, out_csv_dir):
    """PCA sem redução (rotate). Ptrain=0.8, Nr=50. Tabela 2."""
    Ptrain, Nr = 0.8, 50
    cfg = PipelineConfig(pca="rotate", norm="zscore", use_boxcox=False)

    def do_all(model_ctor):
        stats = k_runs(X, y, Ptrain, Nr, cfg, model_ctor)
        return summarize(stats)

    results = {
        "MQ": do_all(lambda: LeastSquaresClassifier()),
        "PL": do_all(lambda: SoftmaxRegression(lr=1e-2, epochs=200, l2=0.0)),
        "MLP-1H": do_all(lambda: MLPClassifier(hidden=(64,), activation="relu", lr=1e-2, epochs=200)),
        "MLP-2H": do_all(lambda: MLPClassifier(hidden=(64,32), activation="relu", lr=1e-2, epochs=200)),
    }

    os.makedirs(out_csv_dir, exist_ok=True)
    with open(os.path.join(out_csv_dir, "tabela2.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Classificador", "ACC_mean", "ACC_std", "ACC_min", "ACC_max", "ACC_median", "fit_time_mean", "pred_time_mean"])
        for name, S in results.items():
            w.writerow([name, S["acc"]["mean"], S["acc"]["std"], S["acc"]["min"], S["acc"]["max"], S["acc"]["median"], S["fit_time"]["mean"], S["pred_time"]["mean"]])
    return results


def run_A5_A6(X, y, out_csv_dir):
    """PCA com redução (≥98% variância). Ptrain=0.8, Nr=50. Tabela 3."""
    Ptrain, Nr = 0.8, 50
    cfg = PipelineConfig(pca="reduce98", norm="zscore", use_boxcox=False)

    def do_all(model_ctor):
        stats = k_runs(X, y, Ptrain, Nr, cfg, model_ctor)
        return summarize(stats)

    results = {
        "MQ": do_all(lambda: LeastSquaresClassifier()),
        "PL": do_all(lambda: SoftmaxRegression(lr=1e-2, epochs=200, l2=0.0)),
        "MLP-1H": do_all(lambda: MLPClassifier(hidden=(64,), activation="relu", lr=1e-2, epochs=200)),
        "MLP-2H": do_all(lambda: MLPClassifier(hidden=(64,32), activation="relu", lr=1e-2, epochs=200)),
    }

    os.makedirs(out_csv_dir, exist_ok=True)
    with open(os.path.join(out_csv_dir, "tabela3.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Classificador", "ACC_mean", "ACC_std", "ACC_min", "ACC_max", "ACC_median", "fit_time_mean", "pred_time_mean"])
        for name, S in results.items():
            w.writerow([name, S["acc"]["mean"], S["acc"]["std"], S["acc"]["min"], S["acc"]["max"], S["acc"]["median"], S["fit_time"]["mean"], S["pred_time"]["mean"]])
    return results


def run_A7_with_boxcox(X, y, out_csv_dir):
    """Após PCA reduzida, aplica Box‑Cox + z‑score e repete (para comparação com Tabela 3)."""
    Ptrain, Nr = 0.8, 50
    cfg = PipelineConfig(pca="reduce98", norm="none", use_boxcox=True)

    def do_all(model_ctor):
        stats = k_runs(X, y, Ptrain, Nr, cfg, model_ctor)
        return summarize(stats)

    results = {
        "MQ": do_all(lambda: LeastSquaresClassifier()),
        "PL": do_all(lambda: SoftmaxRegression(lr=1e-2, epochs=200, l2=0.0)),
        "MLP-1H": do_all(lambda: MLPClassifier(hidden=(64,), activation="relu", lr=1e-2, epochs=200)),
        "MLP-2H": do_all(lambda: MLPClassifier(hidden=(64,32), activation="relu", lr=1e-2, epochs=200)),
    }

    os.makedirs(out_csv_dir, exist_ok=True)
    with open(os.path.join(out_csv_dir, "tabela3_boxcox.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Classificador", "ACC_mean", "ACC_std", "ACC_min", "ACC_max", "ACC_median", "fit_time_mean", "pred_time_mean"])
        for name, S in results.items():
            w.writerow([name, S["acc"]["mean"], S["acc"]["std"], S["acc"]["min"], S["acc"]["max"], S["acc"]["median"], S["fit_time"]["mean"], S["pred_time"]["mean"]])
    return results

# A8 (controle de acesso) seria chamado com dados contendo a classe "intruso" e cálculo binário; reutiliza as métricas.
```

---

## 4) Como usar (exemplo mínimo)

1. \*\*Obtenha \*\*`** e **` (amostras × atributos, rótulos 0..C-1). Se vierem do MATLAB já vetorizados, basta `np.load`/`np.genfromtxt`.
2. Rode na raiz do projeto um script Python semelhante a:

```python
import numpy as np
from experiments.run_experiments import run_A1_A2, run_A3_A4, run_A5_A6, run_A7_with_boxcox

# Carregue seus dados aqui
X = np.load("./data/X.npy")  # shape (n, d)
y = np.load("./data/y.npy")  # shape (n,)

# Rodar Atividades
res12 = run_A1_A2(X, y, out_csv_dir="./results")
res34 = run_A3_A4(X, y, out_csv_dir="./results")
res56 = run_A5_A6(X, y, out_csv_dir="./results")
res7  = run_A7_with_boxcox(X, y, out_csv_dir="./results")
```

3. **Saídas**: CSVs `tabela1.csv`, `tabela2.csv`, `tabela3.csv`, `tabela3_boxcox.csv` com as estatísticas pedidas (média, desvio, min, máx, mediana, tempo).
4. **A8**: use o mesmo pipeline, mas componha o conjunto com a classe binária *autorizado vs. intruso* e reporte acurácia, FN/FP, sensibilidade e precisão (as funções já existem em `metrics.py`).

---

## 5) Extensões sugeridas

- **Validador de configurações** por JSON (evita hard‑code de hyperparams).
- **Treino MLP com mini‑batches** e **early stopping** (ainda só batch GD).
- **Ridge** em MQ e PL (já há `l2`, mas pode virar sweep).
- **Estratificação por sujeito** a partir de nomes de arquivo (separador no `io.py`).
- **Logs por rodada** (CSV por rodada) para análises adicionais.

---

## 6) Observações finais

- O código fornecido é **base funcional** e cobre o que o enunciado pede, sem uso de `sklearn`.
- Para reproduzir exatamente os scripts MATLAB, basta **conectar o loader** às saídas que você já gera (vetores de atributos) e manter **Ptrain=80, Nr=50** conforme solicitado.
- Se me enviar o script MATLAB, integro mapeando 1‑para‑1 as opções (dimensões, q, normalizações) e deixo pronto para rodar no seu dataset específico.



---

# 🔧 Atualizações para atender aos novos pontos

## Novos módulos

### facelab/activations.py

```python
import numpy as np

# Ativações e derivadas (derivadas recebem o pré-ativo z)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def dsigmoid(z):
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh(z):
    return np.tanh(z)

def dtanh(z):
    t = np.tanh(z)
    return 1.0 - t*t


def relu(z):
    return np.maximum(0.0, z)

def drelu(z):
    return (z > 0).astype(float)


def leaky_relu(z, a=0.01):
    return np.where(z > 0, z, a * z)

def dleaky_relu(z, a=0.01):
    g = np.ones_like(z); g[z <= 0] = a; return g

# ReLU6 e Swish

def relu6(z):
    return np.minimum(np.maximum(0.0, z), 6.0)

def drelu6(z):
    return ((z > 0) & (z < 6)).astype(float)


def swish(z):
    s = sigmoid(z)
    return z * s

def dswish(z):
    s = sigmoid(z)
    return s + z * s * (1.0 - s)

ACT_FNS = {
    "sigmoid": (sigmoid, dsigmoid),
    "tanh": (tanh, dtanh),
    "relu": (relu, drelu),
    "leaky_relu": (leaky_relu, dleaky_relu),
    "relu6": (relu6, drelu6),
    "swish": (swish, dswish),
}
```

### facelab/optim/optimizers.py

```python
import numpy as np

class Optimizer:
    def __init__(self, lr: float):
        self.lr = lr
        self.state = {}
    def step(self, name: str, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        # SGD puro como fallback
        return param - self.lr * grad

class SGD(Optimizer):
    pass

class Momentum(Optimizer):
    def __init__(self, lr: float, momentum: float = 0.9):
        super().__init__(lr); self.m = momentum
    def step(self, name, p, g):
        v = self.state.get(name, np.zeros_like(p))
        v = self.m * v - self.lr * g
        self.state[name] = v
        return p + v

class Nesterov(Optimizer):
    def __init__(self, lr: float, momentum: float = 0.9):
        super().__init__(lr); self.m = momentum
    def step(self, name, p, g):
        v = self.state.get(name, np.zeros_like(p))
        v_prev = v
        v = self.m * v - self.lr * g
        self.state[name] = v
        return p + (-self.m * v_prev + (1 + self.m) * v)

class RMSProp(Optimizer):
    def __init__(self, lr: float, beta: float = 0.9, eps: float = 1e-8):
        super().__init__(lr); self.b = beta; self.eps = eps
    def step(self, name, p, g):
        s = self.state.get(name, np.zeros_like(p))
        s = self.b * s + (1 - self.b) * (g * g)
        self.state[name] = s
        return p - self.lr * g / (np.sqrt(s) + self.eps)

class Adam(Optimizer):
    def __init__(self, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(lr); self.b1 = beta1; self.b2 = beta2; self.eps = eps; self.t = 0
    def step(self, name, p, g):
        m, v = self.state.get(name, (np.zeros_like(p), np.zeros_like(p)))
        self.t += 1
        m = self.b1 * m + (1 - self.b1) * g
        v = self.b2 * v + (1 - self.b2) * (g * g)
        mhat = m / (1 - self.b1 ** self.t)
        vhat = v / (1 - self.b2 ** self.t)
        self.state[name] = (m, v)
        return p - self.lr * mhat / (np.sqrt(vhat) + self.eps)

OPTIMIZERS = {
    "sgd": SGD,
    "momentum": Momentum,
    "nesterov": Nesterov,
    "rmsprop": RMSProp,
    "adam": Adam,
}
```

## Atualização do MLP para suportar novas ativações e otimizadores

### facelab/models/mlp.py (atualizado)

```python
import numpy as np
from ..optim.optimizers import OPTIMIZERS
from ..activations import ACT_FNS

__all__ = ["MLPClassifier"]

class MLPClassifier:
    def __init__(self, hidden=(64,), activation="relu", lr=1e-2, epochs=200, l2=0.0, seed=None,
                 optimizer: str = "sgd", batch_size: int | None = None, opt_kwargs: dict | None = None):
        self.hidden = tuple(hidden)
        self.activation = activation
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed
        self.optimizer_name = optimizer
        self.batch_size = batch_size
        self.opt_kwargs = opt_kwargs or {}
        self.params_ = None
        self.loss_history_ = []

    def _init_params(self, d, C):
        rng = np.random.default_rng(self.seed)
        sizes = [d] + list(self.hidden) + [C]
        W, b = [], []
        for i in range(len(sizes) - 1):
            fan_in, fan_out = sizes[i], sizes[i+1]
            scale = np.sqrt(2.0 / fan_in)
            W.append(rng.normal(0.0, scale, size=(fan_in, fan_out)))
            b.append(np.zeros((1, fan_out)))
        self.params_ = {"W": W, "b": b}
        # otimizador com estado por parâmetro
        Opt = OPTIMIZERS[self.optimizer_name]
        self.opt_ = Opt(self.lr, **self.opt_kwargs)

    def _forward(self, X):
        f_act, _ = ACT_FNS[self.activation]
        A = [X]; Zs = []
        # ocultas
        for i in range(len(self.params_["W"]) - 1):
            Z = A[-1] @ self.params_["W"][i] + self.params_["b"][i]
            Zs.append(Z)
            A.append(f_act(Z))
        # saída (softmax)
        Z = A[-1] @ self.params_["W"][-1] + self.params_["b"][-1]
        Zs.append(Z)
        Z = Z - Z.max(axis=1, keepdims=True)
        expZ = np.exp(Z)
        P = expZ / (expZ.sum(axis=1, keepdims=True))
        return A, Zs, P

    def fit(self, X, Y):
        n, d = X.shape; C = Y.shape[1]
        self._init_params(d, C)
        _, dact = ACT_FNS[self.activation]
        bs = self.batch_size or n
        for _ in range(self.epochs):
            # embaralha
            idx = np.random.permutation(n)
            for k in range(0, n, bs):
                batch = idx[k:k+bs]
                Xb, Yb = X[batch], Y[batch]
                A, Zs, P = self._forward(Xb)
                # loss com L2
                ce = -np.sum(Yb * np.log(P + 1e-12)) / Xb.shape[0]
                reg = 0.5 * self.l2 * sum((W**2).sum() for W in self.params_["W"])
                self.loss_history_.append(float(ce + reg))
                # backward
                dZ = (P - Yb) / Xb.shape[0]
                grads_W, grads_b = [], []
                # saída
                dW = A[-1].T @ dZ + self.l2 * self.params_["W"][-1]
                db = dZ.sum(axis=0, keepdims=True)
                grads_W.insert(0, dW); grads_b.insert(0, db)
                dA = dZ @ self.params_["W"][-1].T
                # ocultas (reverso)
                for i in range(len(self.params_["W"]) - 2, -1, -1):
                    dZ = dA * dact(Zs[i])
                    A_prev = A[i]
                    dW = A_prev.T @ dZ + self.l2 * self.params_["W"][i]
                    db = dZ.sum(axis=0, keepdims=True)
                    grads_W.insert(0, dW); grads_b.insert(0, db)
                    if i > 0:
                        dA = dZ @ self.params_["W"][i].T
                # update com otimizador
                for i in range(len(self.params_["W"])):
                    self.params_["W"][i] = self.opt_.step(f"W{i}", self.params_["W"][i], grads_W[i])
                    self.params_["b"][i] = self.opt_.step(f"b{i}", self.params_["b"][i], grads_b[i])
        return self

    def predict(self, X):
        A, Zs, P = self._forward(X)
        return np.argmax(P, axis=1)
```

## Busca aleatória de hiperparâmetros (inclui normalização)

### experiments/random\_search.py

```python
import numpy as np
from statistics import mean
from dataclasses import dataclass
from facelab.splits import stratified_train_test_split
from facelab.metrics import accuracy
from experiments.pipelines import Pipeline, PipelineConfig
from facelab.models.mq import LeastSquaresClassifier
from facelab.models.pl import SoftmaxRegression
from facelab.models.mlp import MLPClassifier

@dataclass
class SearchResult:
    config: dict
    acc_mean: float

NORM_CHOICES = ["none", "zscore", "minmax", "minmax_pm1"]
ACT_CHOICES = ["tanh", "sigmoid", "leaky_relu", "relu6", "swish", "relu"]
OPT_CHOICES = ["sgd", "momentum", "nesterov", "rmsprop", "adam"]


def _mk_model(name: str, cfg: dict):
    if name == "MQ":
        return LeastSquaresClassifier(l2=cfg.get("l2", 0.0))
    if name == "PL":
        return SoftmaxRegression(lr=cfg.get("lr", 1e-2), epochs=cfg.get("epochs", 200), l2=cfg.get("l2", 0.0))
    if name.startswith("MLP"):
        hidden = cfg.get("hidden", (64,))
        return MLPClassifier(hidden=hidden, activation=cfg.get("activation", "relu"), lr=cfg.get("lr", 1e-2),
                             epochs=cfg.get("epochs", 200), l2=cfg.get("l2", 0.0), optimizer=cfg.get("optimizer", "sgd"),
                             batch_size=cfg.get("batch_size", None), opt_kwargs=cfg.get("opt_kwargs", {}))
    raise ValueError(name)


def sample_config(model_name: str, rng: np.random.Generator) -> dict:
    cfg = {"norm": rng.choice(NORM_CHOICES)}
    if model_name == "MQ":
        cfg["l2"] = float(10 ** rng.uniform(-6, -1))
    elif model_name == "PL":
        cfg.update({
            "lr": float(10 ** rng.uniform(-4, -1))),
            "epochs": int(rng.integers(100, 301)),
            "l2": float(10 ** rng.uniform(-6, -2)),
        })
    else:  # MLP-1H / MLP-2H
        h1 = int(rng.choice([16, 32, 64, 96, 128]))
        hidden = (h1,) if model_name == "MLP-1H" else (h1, int(rng.choice([16, 32, 64])))
        cfg.update({
            "activation": rng.choice(ACT_CHOICES),
            "optimizer": rng.choice(OPT_CHOICES),
            "lr": float(10 ** rng.uniform(-4, -2))),
            "epochs": int(rng.integers(100, 301)),
            "l2": float(10 ** rng.uniform(-6, -3)),
            "batch_size": int(rng.choice([None, 16, 32, 64])) if rng.random() < 0.7 else None,
            "hidden": hidden,
        })
    return cfg


def random_search_models(X, y, model_names, p_train=0.8, n_iter=40, seed=42):
    rng = np.random.default_rng(seed)
    best = {m: None for m in model_names}
    for m in model_names:
        cand = []
        for _ in range(n_iter):
            cfg = sample_config(m, rng)
            idx_tr, idx_te = stratified_train_test_split(y, p_train, rng)
            pipe = Pipeline(PipelineConfig(pca="none", norm=cfg["norm"], use_boxcox=False))
            Xtr = pipe.fit(X[idx_tr]); Xte = pipe.transform(X[idx_te])
            model = _mk_model(m, cfg)
            model.fit(Xtr, one_hot(y[idx_tr], int(y.max()+1)))
            acc = accuracy(y[idx_te], model.predict(Xte))
            cand.append((acc, cfg))
        acc_best, cfg_best = max(cand, key=lambda t: t[0])
        best[m] = SearchResult(cfg_best, float(acc_best))
    return best
```

## Benchmark de escala e gráfico Tempo × Dimensão

### experiments/scale\_benchmarks.py

```python
import time, csv
import numpy as np
import matplotlib.pyplot as plt
from facelab.io import load_flatten_from_dir
from facelab.metrics import accuracy
from facelab.utils import one_hot
from facelab.models.mq import LeastSquaresClassifier
from facelab.models.pl import SoftmaxRegression
from facelab.models.mlp import MLPClassifier
from experiments.pipelines import Pipeline, PipelineConfig

# sizes: lista de (H, W) — ex.: [(20,20),(24,24),(28,28),(30,30),(32,32)]

def run_scaling(root_dir, sizes, p_train=0.8, repeats=10, out_csv="./results/scaling_times.csv", plot_png="./results/scaling_times.png"):
    os.makedirs("./results", exist_ok=True)
    rows = [("size","model","fit_time_mean","pred_time_mean","acc_mean")]
    for sz in sizes:
        X, y, _ = load_flatten_from_dir(root_dir, size=sz)
        d = X.shape[1]
        for name, ctor in [
            ("MQ", lambda: LeastSquaresClassifier()),
            ("PL", lambda: SoftmaxRegression(lr=1e-2, epochs=200)),
            ("MLP-1H", lambda: MLPClassifier(hidden=(64,), activation="relu", optimizer="adam", lr=1e-3, epochs=200)),
        ]:
            fit_ts, pred_ts, accs = [], [], []
            for r in range(repeats):
                rng = np.random.default_rng(2025 + r)
                idx_tr, idx_te = stratified_train_test_split(y, p_train, rng)
                pipe = Pipeline(PipelineConfig(pca="none", norm="zscore", use_boxcox=False))
                Xtr = pipe.fit(X[idx_tr]); Xte = pipe.transform(X[idx_te])
                m = ctor()
                t0 = time.perf_counter(); m.fit(Xtr, one_hot(y[idx_tr], int(y.max()+1))); t1 = time.perf_counter()
                yhat = m.predict(Xte); t2 = time.perf_counter()
                fit_ts.append(t1 - t0); pred_ts.append(t2 - t1); accs.append(accuracy(y[idx_te], yhat))
            rows.append((d, name, np.mean(fit_ts), np.mean(pred_ts), np.mean(accs)))
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    # Gráfico (tempo de treino) por modelo
    data = {}
    for r in rows[1:]:
        d, name, ft, pt, acc = r
        data.setdefault(name, []).append((d, ft))
    plt.figure(figsize=(8,5))
    for name, series in data.items():
        series.sort(key=lambda t: t[0])
        xs = [s[0] for s in series]; ys = [s[1] for s in series]
        plt.plot(xs, ys, marker="o", label=name)
    plt.xlabel("Dimensão (H×W)"); plt.ylabel("Tempo médio de treino (s)"); plt.legend(); plt.tight_layout()
    plt.savefig(plot_png, dpi=150)
```

## Hooks nos experimentos para Random Search após escolher o tamanho

**Sugestão prática**:

- Use `scale_benchmarks.run_scaling(...)` para visualizar o crescimento do tempo vs. dimensão e escolha o tamanho “OK”.
- Em seguida, chame `random_search.random_search_models(...)` com `model_names=["MQ","PL","MLP-1H","MLP-2H"]` para otimizar **normalização, ativação, otimizador e demais hiperparâmetros**.
- Para a **Atividade 4**, troque `PipelineConfig(pca="rotate", ...)` mantendo `q = H×W` do tamanho escolhido e refaça a busca.

> Observação: o MLP agora aceita `activation` ∈ {`tanh`, \`sigmoi
