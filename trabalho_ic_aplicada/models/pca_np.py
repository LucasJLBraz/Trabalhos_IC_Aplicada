# trabalho_ic_aplicada/models/pca_np.py
import numpy as np

__all__ = ["PCA_np"]

class PCA_np:
    """
    PCA via SVD. Modo 'rotate' (sem redução): q = d original.
    """
    def __init__(self, q: int | None = None):
        self.q = q
        self.mu_ = None
        self.Vt_ = None
        self.ev_ratio_ = None

    def fit(self, X: np.ndarray):
        self.mu_ = X.mean(axis=0)
        Xc = X - self.mu_
        # SVD compacto
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        var = (S**2) / (X.shape[0] - 1)
        self.Vt_ = Vt
        self.ev_ratio_ = var / var.sum()
        return self

    def transform(self, X: np.ndarray, q: int | None = None) -> np.ndarray:
        if q is None:
            q = self.q if self.q is not None else X.shape[1]
        Xc = X - self.mu_
        Vt = self.Vt_[:q, :]
        return Xc @ Vt.T

    def fit_transform(self, X: np.ndarray, q: int | None = None) -> np.ndarray:
        self.fit(X)
        return self.transform(X, q=q)
