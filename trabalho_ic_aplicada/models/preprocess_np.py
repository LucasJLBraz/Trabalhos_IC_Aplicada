# trabalho_ic_aplicada/models/preprocess_np.py
import numpy as np
from dataclasses import dataclass

__all__ = [
    "ZScore", "MinMax01", "MinMaxNegPos", "apply_norm",
]

class ZScore:
    def __init__(self):
        self.mu_ = None
        self.sd_ = None
    def fit(self, X: np.ndarray):
        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0) + 1e-12
        return self
    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu_) / self.sd_
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

class MinMax01:
    def __init__(self):
        self.mn_ = None
        self.mx_ = None
    def fit(self, X: np.ndarray):
        self.mn_ = X.min(axis=0)
        self.mx_ = X.max(axis=0)
        return self
    def transform(self, X: np.ndarray) -> np.ndarray:
        den = (self.mx_ - self.mn_) + 1e-12
        return (X - self.mn_) / den
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

class MinMaxNegPos:
    def __init__(self):
        self._mm = MinMax01()
    def fit(self, X: np.ndarray):
        self._mm.fit(X); return self
    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._mm.transform(X) * 2.0 - 1.0
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.transform(self._mm.fit_transform(X))

@dataclass
class NormSpec:
    name: str  # "none" | "zscore" | "minmax" | "minmax_pm1"

def apply_norm(Xtr: np.ndarray, Xte: np.ndarray, spec: NormSpec):
    if spec.name == "none":
        return Xtr, Xte, None
    if spec.name == "zscore":
        n = ZScore().fit(Xtr)
    elif spec.name == "minmax":
        n = MinMax01().fit(Xtr)
    elif spec.name == "minmax_pm1":
        n = MinMaxNegPos().fit(Xtr)
    else:
        raise ValueError(f"normalização desconhecida: {spec.name}")
    return n.transform(Xtr), n.transform(Xte), n
