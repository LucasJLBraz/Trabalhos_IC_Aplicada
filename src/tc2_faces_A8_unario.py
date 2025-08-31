# src/tc2_faces_A8_unario.py
import os
import time
import csv
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

from trabalho_ic_aplicada.dataset_faces import build_face_dataset
from trabalho_ic_aplicada.models.pca_np import PCA_np
from trabalho_ic_aplicada.models.reg_mlp_class import MLPRegressor

# =========================
# CONFIG
# =========================
DATA_ROOT = "./data/raw/Kit_projeto_FACES"
SCALES = [(30, 30)]
RESULTS_DIR = "./results/TC2/"
N_SAMPLES_RS = 30
K_SELECT_EVAL = 10
N_REPEATS_BEST = 50

# =========================
# Funções de Pré-processamento (com correção de bug)
# =========================
def _boxcox_transform(x_pos: np.ndarray, lam: float) -> np.ndarray:
    x_safe = np.clip(x_pos, 1e-9, None) # BUG FIX: Garante que a base é estritamente positiva
    if lam == 0.0:
        return np.log(x_safe)
    return (np.power(x_safe, lam) - 1.0) / lam

def _boxcox_ll(x_pos: np.ndarray, lam: float) -> float:
    x_safe = np.clip(x_pos, 1e-9, None) # BUG FIX: Garante que o log não recebe zero/negativo
    z = _boxcox_transform(x_safe, lam)
    n = x_safe.size
    var = z.var(ddof=1) + 1e-12
    return - (n/2.0)*np.log(var) + (lam - 1.0)*np.log(x_safe).sum()

def fit_boxcox_then_zscore(X: np.ndarray, grid=None):
    if grid is None: grid = np.linspace(-2.0, 2.0, 81)
    n, d = X.shape
    Xbc = np.empty_like(X, dtype=float)
    lambdas = np.zeros(d); shifts = np.zeros(d)
    for j in range(d):
        x = X[:, j]
        mn = x.min()
        shift = 0.0
        if mn <= 0:
            shift = -mn + 1e-6
        x_pos = x + shift
        best_ll, best_lam = -np.inf, 1.0
        for lam in grid:
            ll = _boxcox_ll(x_pos, lam)
            if ll > best_ll:
                best_ll, best_lam = ll, lam
        Xbc[:, j] = _boxcox_transform(x_pos, best_lam)
        lambdas[j] = best_lam
        shifts[j] = shift
    mu  = Xbc.mean(axis=0)
    std = Xbc.std(axis=0) + 1e-12
    Xn  = (Xbc - mu) / std
    # BUG FIX: Checa se há NaNs após a transformação
    if not np.all(np.isfinite(Xn)):
        raise ValueError("NaN ou Inf detectado após o pré-processamento.")
    return Xn, (lambdas, shifts, mu, std)

def transform_boxcox_then_zscore(X: np.ndarray, params):
    lambdas, shifts, mu, std = params
    Xbc = np.empty_like(X, dtype=float)
    for j in range(X.shape[1]):
        x_pos = X[:, j] + shifts[j]
        lam = lambdas[j]
        Xbc[:, j] = _boxcox_transform(x_pos, lam)
    Xn = (Xbc - mu) / std
    return Xn

# =========================
# Métricas e Avaliação
# =========================
def binary_metrics(y_true_bin, y_pred_bin):
    TP = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
    TN = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
    FP = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
    FN = np.sum((y_true_bin == 1) & (y_pred_bin == 0))
    acc = (TP + TN) / max(1, len(y_true_bin))
    tpr = TP / max(1, TP + FN)
    ppv = TP / max(1, TP + FP)
    fnr = FN / max(1, TP + FN)
    fpr = FP / max(1, FP + TN)
    f1 = 2 * ppv * tpr / max(1e-9, ppv + tpr)
    return {"acc": acc, "tpr": tpr, "ppv": ppv, "fnr": fnr, "fpr": fpr, "f1_score": f1}

def find_optimal_threshold(scores_normal, scores_anomaly):
    y_true = np.concatenate([np.zeros(len(scores_normal)), np.ones(len(scores_anomaly))])
    all_scores = np.concatenate([scores_normal, scores_anomaly])
    best_f1, best_thresh = -1, 0
    for threshold in np.unique(all_scores):
        y_pred = (all_scores >= threshold).astype(int)
        metrics = binary_metrics(y_true, y_pred)
        if metrics["f1_score"] > best_f1:
            best_f1, best_thresh = metrics["f1_score"], threshold
    return best_thresh, best_f1

def summarize_runs(run_list):
    from statistics import mean, stdev, median
    agg = {}
    if not run_list: return agg
    keys = run_list[0].keys()
    for k in keys:
        vals = [r[k] for r in run_list]
        agg[k+"_mean"] = mean(vals)
        agg[k+"_std"] = stdev(vals) if len(vals) > 1 else 0
        agg[k+"_median"] = median(vals)
    return agg

# =========================
# Modelos e Samplers
# =========================
class PCAAnomalyDetector:
    def __init__(self, q_dim):
        self.q_dim = q_dim
        self.pca = PCA_np(q=self.q_dim)
    def fit(self, X): self.pca.fit(X); return self
    def predict_scores(self, X):
        X_proj = self.pca.transform(X)
        X_recon = self.pca.inverse_transform(X_proj)
        return np.mean((X - X_recon)**2, axis=1)

class AutoencoderAnomalyDetector:
    def __init__(self, **kwargs): self.params = kwargs; self.model = MLPRegressor(**kwargs)
    def fit(self, X): self.model.fit(X, X); return self
    def predict_scores(self, X): return np.mean((X - self.model.predict(X))**2, axis=1)

class SklearnAnomalyWrapper:
    def __init__(self, model_ctor, params): self.params = params; self.model = model_ctor(**params)
    def fit(self, X): self.model.fit(X); return self
    def predict_scores(self, X): return -self.model.decision_function(X)

class Autoencoder1HSampler:
    def __init__(self, q_dim): self.q_dim = q_dim
    def __call__(self, rng):
        middle_dim = int(rng.choice([self.q_dim * f for f in [0.25, 0.5, 0.75]]))
        return {"hidden": (middle_dim,), "activation": "tanh", "lr": float(rng.choice([0.005, 0.01])),
                "epochs": 200, "l2": 1e-4, "opt": "nesterov", "clip_grad": 5.0}
    def to_model(self, p): return AutoencoderAnomalyDetector(**p)

class Autoencoder2HSampler:
    def __init__(self, q_dim): self.q_dim = q_dim
    def __call__(self, rng):
        h1_dim = int(rng.choice([self.q_dim * f for f in [0.75, 1.25]]))
        middle_dim = int(rng.choice([self.q_dim * f for f in [0.25, 0.5]]))
        return {"hidden": (h1_dim, middle_dim, h1_dim), "activation": "tanh", "lr": float(rng.choice([0.005, 0.01])),
                "epochs": 200, "l2": 1e-4, "opt": "nesterov", "clip_grad": 5.0}
    def to_model(self, p): return AutoencoderAnomalyDetector(**p)

class OneClassSVMSampler:
    def __call__(self, rng):
        return {"nu": float(rng.choice([0.01, 0.05, 0.1, 0.15])), "gamma": float(rng.choice([0.01, 0.1, 1, 10]))}
    def to_model(self, p): return SklearnAnomalyWrapper(OneClassSVM, p)

class IsolationForestSampler:
    def __call__(self, rng):
        return {"n_estimators": int(rng.choice([50, 100, 200])), "random_state": 42}
    def to_model(self, p): return SklearnAnomalyWrapper(IsolationForest, p)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    X, y, idx2class = build_face_dataset(DATA_ROOT, size=SCALES[0], load_intruders=True)
    intruder_class_id = [k for k, v in idx2class.items() if not v.lower().startswith("subject")][0]
    X_auth, X_intruder = X[y != intruder_class_id], X[y == intruder_class_id]

    pca_pre = PCA_np(); pca_pre.fit(X_auth)
    q_star = int(np.searchsorted(np.cumsum(pca_pre.ev_ratio_), 0.99) + 1)
    print(f"Dimensão original: {X.shape[1]}, Dimensão reduzida (q*={q_star} para 99% da variância)")

    samplers = {
        "PCA_Baseline": (lambda r: {}, lambda p: PCAAnomalyDetector(q_dim=q_star)), # No-op sampler
        "AE_1H": (Autoencoder1HSampler(q_star), lambda p: AutoencoderAnomalyDetector(**p)),
        "AE_2H": (Autoencoder2HSampler(q_star), lambda p: AutoencoderAnomalyDetector(**p)),
        "OneClassSVM": (OneClassSVMSampler(), lambda p: SklearnAnomalyWrapper(OneClassSVM, p)),
        "IsolationForest": (IsolationForestSampler(), lambda p: SklearnAnomalyWrapper(IsolationForest, p))
    }
    all_final_results = []

    for name, (sampler_func, model_ctor_func) in samplers.items():
        print(f"--- Fase 1: Otimizando modelo: {name} ---")
        rng_search = np.random.default_rng(2025)
        best_params = {}; best_f1 = -1

        # O baseline de PCA não tem hiperparâmetros para buscar
        is_baseline = name == "PCA_Baseline"
        current_n_samples = 1 if is_baseline else N_SAMPLES_RS

        for s in range(current_n_samples):
            params = sampler_func(rng_search)
            f1_scores = []
            for k in range(K_SELECT_EVAL):
                shuffled_auth = rng_search.permutation(len(X_auth))
                tr_auth_idx, val_auth_idx = shuffled_auth[:int(0.7*len(X_auth))], shuffled_auth[int(0.7*len(X_auth)):int(0.85*len(X_auth))]
                shuffled_intruder = rng_search.permutation(len(X_intruder))
                val_intr_idx = shuffled_intruder[:int(0.5*len(X_intruder))]

                X_auth_tr, X_auth_val = X_auth[tr_auth_idx], X_auth[val_auth_idx]
                X_intr_val = X_intruder[val_intr_idx]

                # Pipeline
                pca_search = PCA_np(q=q_star); X_auth_tr_p = pca_search.fit_transform(X_auth_tr)
                X_auth_val_p = pca_search.transform(X_auth_val); X_intr_val_p = pca_search.transform(X_intr_val)
                X_auth_tr_n, bx_params = fit_boxcox_then_zscore(X_auth_tr_p)
                X_auth_val_n = transform_boxcox_then_zscore(X_auth_val_p, bx_params); X_intr_val_n = transform_boxcox_then_zscore(X_intr_val_p, bx_params)

                model = model_ctor_func(params).fit(X_auth_tr_n)
                scores_auth_val = model.predict_scores(X_auth_val_n)
                scores_intruder_val = model.predict_scores(X_intr_val_n)
                
                _, f1 = find_optimal_threshold(scores_auth_val, scores_intruder_val)
                f1_scores.append(f1)

            avg_f1 = np.mean(f1_scores)
            if avg_f1 > best_f1:
                best_f1, best_params = avg_f1, params
        
        print(f"Melhores parâmetros para {name}: {best_params} (F1 Val: {best_f1:.3f})")

        print(f"--- Fase 2: Avaliando melhor modelo: {name} ---")
        final_runs = []
        for i in range(N_REPEATS_BEST):
            rng_final = np.random.default_rng(9090 + i)
            shuffled_auth = rng_final.permutation(len(X_auth))
            tr_idx, te_idx = shuffled_auth[:int(0.85*len(X_auth))], shuffled_auth[int(0.85*len(X_auth)):]
            X_auth_tr, X_auth_te = X_auth[tr_idx], X_auth[te_idx]
            X_intruder_te = X_intruder # Usamos todos os intrusos para o teste final

            pca_final = PCA_np(q=q_star); X_auth_tr_p = pca_final.fit_transform(X_auth_tr)
            X_auth_tr_n, bx_params_final = fit_boxcox_then_zscore(X_auth_tr_p)

            t0 = time.time()
            model_final = model_ctor_func(best_params).fit(X_auth_tr_n)
            fit_time = time.time() - t0

            train_scores = model_final.predict_scores(X_auth_tr_n)
            threshold = np.percentile(train_scores, 95)

            X_auth_te_p = pca_final.transform(X_auth_te); X_auth_te_n = transform_boxcox_then_zscore(X_auth_te_p, bx_params_final)
            X_intruder_te_p = pca_final.transform(X_intruder_te); X_intruder_te_n = transform_boxcox_then_zscore(X_intruder_te_p, bx_params_final)

            t1 = time.time()
            scores_auth_te = model_final.predict_scores(X_auth_te_n)
            scores_intruder_te = model_final.predict_scores(X_intruder_te_n)
            pred_time = time.time() - t1

            y_true = np.concatenate([np.zeros(len(scores_auth_te)), np.ones(len(scores_intruder_te))])
            y_pred = (np.concatenate([scores_auth_te, scores_intruder_te]) >= threshold).astype(int)
            
            metrics = binary_metrics(y_true, y_pred)
            metrics["fit_time"] = fit_time; metrics["pred_time"] = pred_time
            final_runs.append(metrics)

        summary = summarize_runs(final_runs)
        summary["model"] = name; summary["best_params"] = str(best_params)
        all_final_results.append(summary)

    out_csv_path = os.path.join(RESULTS_DIR, "tabela4_intruso_unario.csv")
    if all_final_results:
        fieldnames = list(all_final_results[0].keys())
        with open(out_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames); writer.writeheader(); writer.writerows(all_final_results)
        print(f"[OK] Resultados da abordagem unária salvos em: {out_csv_path}")