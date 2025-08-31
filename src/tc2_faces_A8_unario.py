# src/tc2_faces_A8_unario.py
import os
import time
import csv
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

from trabalho_ic_aplicada.dataset_faces import build_face_dataset
from trabalho_ic_aplicada.models.pca_np import PCA_np
from trabalho_ic_aplicada.models.reg_mlp import MLPRegressor

# =========================
# CONFIG
# =========================
DATA_ROOT = "./data/raw/Kit_projeto_FACES"
SCALES = [(30, 30)]
RESULTS_DIR = "./results/TC2/"

# Configs da busca e avaliação
N_SAMPLES_RS = 30   # Amostras da busca aleatória (reduzida)
K_SELECT_EVAL = 10  # Repetições para avaliar cada candidato da busca
N_REPEATS_BEST = 50 # Repetições finais para o melhor modelo

# =========================
# Funções de Pré-processamento (Idênticas às de A8)
# =========================
def _boxcox_transform(x_pos: np.ndarray, lam: float) -> np.ndarray:
    if lam == 0.0: return np.log(x_pos)
    return (np.power(x_pos, lam) - 1.0) / lam

def _boxcox_ll(x_pos: np.ndarray, lam: float) -> float:
    z = _boxcox_transform(x_pos, lam)
    n = x_pos.size
    var = z.var(ddof=1) + 1e-12
    return - (n/2.0)*np.log(var) + (lam - 1.0)*np.log(x_pos).sum()

def fit_boxcox_then_zscore(X: np.ndarray, grid=None):
    if grid is None: grid = np.linspace(-2.0, 2.0, 81)
    n, d = X.shape
    Xbc = np.empty_like(X, dtype=float)
    lambdas = np.zeros(d); shifts = np.zeros(d)
    for j in range(d):
        x = X[:, j]
        mn = x.min(); shift = 0.0
        if mn <= 0: shift = -mn + 1e-6
        x_pos = x + shift
        best_ll, best_lam = -np.inf, 1.0
        for lam in grid:
            ll = _boxcox_ll(x_pos, lam)
            if ll > best_ll: best_ll, best_lam = ll, lam
        Xbc[:, j] = _boxcox_transform(x_pos, best_lam)
        lambdas[j] = best_lam; shifts[j] = shift
    mu  = Xbc.mean(axis=0); std = Xbc.std(axis=0) + 1e-12
    Xn  = (Xbc - mu) / std
    return Xn, (lambdas, shifts, mu, std)

def transform_boxcox_then_zscore(X: np.ndarray, params):
    lambdas, shifts, mu, std = params
    Xbc = np.empty_like(X, dtype=float)
    for j in range(X.shape[1]):
        x_pos = X[:, j] + shifts[j]
        lam   = lambdas[j]
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
# Modelos e Samplers para Random Search
# =========================
class AutoencoderAnomalyDetector:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.model = MLPRegressor(**kwargs)
    def fit(self, X): self.model.fit(X, X); return self
    def predict_scores(self, X): return np.mean((X - self.model.predict(X))**2, axis=1)

class SklearnAnomalyWrapper:
    def __init__(self, model_ctor, params):
        self.params = params
        self.model = model_ctor(**params)
    def fit(self, X): self.model.fit(X); return self
    def predict_scores(self, X): return -self.model.decision_function(X)

class AutoencoderSampler:
    def __init__(self, q_dim):
        self.q_dim = q_dim
    def __call__(self, rng):
        middle_dim = int(rng.choice([self.q_dim/4, self.q_dim/2, self.q_dim*3/4]))
        return {
            "hidden": (self.q_dim, middle_dim, self.q_dim),
            "activation": "tanh",
            "lr": float(rng.choice([0.005, 0.01, 0.02])),
            "epochs": 200,
            "l2": float(rng.choice([1e-5, 1e-4, 1e-3])),
            "opt": "adam",
            "clip_grad": 5.0
        }
    def to_model(self, p): return AutoencoderAnomalyDetector(**p)

class OneClassSVMSampler:
    def __call__(self, rng):
        return {"nu": float(rng.choice([0.01, 0.05, 0.1, 0.15, 0.2])), "gamma": float(rng.choice([0.01, 0.1, 1, 10]))}
    def to_model(self, p): return SklearnAnomalyWrapper(OneClassSVM, p)

class IsolationForestSampler:
    def __call__(self, rng):
        return {
            "n_estimators": int(rng.choice([50, 100, 200])),
            "max_samples": float(rng.choice([0.5, 0.75, 1.0])),
            "contamination": "auto",
            "random_state": 42
        }
    def to_model(self, p): return SklearnAnomalyWrapper(IsolationForest, p)

# =========================
# Protocolo Experimental Unário
# =========================
def eval_once_anomaly(X_auth, X_intruder, q, model_ctor, seed=42):
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(X_auth))
    split_idx = int(0.8 * len(X_auth))
    train_indices, test_indices = shuffled_indices[:split_idx], shuffled_indices[split_idx:]
    X_auth_tr, X_auth_te = X_auth[train_indices], X_auth[test_indices]

    pca = PCA_np(q=q); X_auth_tr_p = pca.fit_transform(X_auth_tr)
    X_auth_tr_n, bx_params = fit_boxcox_then_zscore(X_auth_tr_p)

    model = model_ctor().fit(X_auth_tr_n)

    X_auth_te_p = pca.transform(X_auth_te); X_auth_te_n = transform_boxcox_then_zscore(X_auth_te_p, bx_params)
    X_intruder_p = pca.transform(X_intruder); X_intruder_n = transform_boxcox_then_zscore(X_intruder_p, bx_params)

    scores_auth = model.predict_scores(X_auth_te_n)
    scores_intruder = model.predict_scores(X_intruder_n)

    threshold, f1 = find_optimal_threshold(scores_auth, scores_intruder)
    return {"f1_score": f1, "threshold": threshold}

def select_best_by_random_search(X_auth, X_intruder, q, sampler, n_samples, k_eval, seed_base=2025):
    rng = np.random.default_rng(seed_base)
    best_params, best_score = None, -1

    for s in range(n_samples):
        params = sampler(rng)
        evals = [eval_once_anomaly(X_auth, X_intruder, q, lambda: sampler.to_model(params), seed=seed_base + s*100 + k) for k in range(k_eval)]
        score = np.mean([e["f1_score"] for e in evals])
        if score > best_score:
            best_score, best_params = score, params
    return best_params

def eval_best_over_repeats(X_auth, X_intruder, q, model_ctor, n_repeats, seed_base=9090):
    runs = []
    for i in range(n_repeats):
        # Re-implementa a lógica de eval_once_anomaly para capturar todas as métricas e tempos
        rng = np.random.default_rng(seed_base + i)
        shuffled_indices = rng.permutation(len(X_auth))
        split_idx = int(0.8 * len(X_auth))
        train_indices, test_indices = shuffled_indices[:split_idx], shuffled_indices[split_idx:]
        X_auth_tr, X_auth_te = X_auth[train_indices], X_auth[test_indices]

        pca = PCA_np(q=q); X_auth_tr_p = pca.fit_transform(X_auth_tr)
        X_auth_tr_n, bx_params = fit_boxcox_then_zscore(X_auth_tr_p)

        t0 = time.perf_counter()
        model = model_ctor().fit(X_auth_tr_n)
        fit_time = time.perf_counter() - t0

        X_auth_te_p = pca.transform(X_auth_te); X_auth_te_n = transform_boxcox_then_zscore(X_auth_te_p, bx_params)
        X_intruder_p = pca.transform(X_intruder); X_intruder_n = transform_boxcox_then_zscore(X_intruder_p, bx_params)

        t1 = time.perf_counter()
        scores_auth = model.predict_scores(X_auth_te_n)
        scores_intruder = model.predict_scores(X_intruder_n)
        pred_time = time.perf_counter() - t1

        threshold, _ = find_optimal_threshold(scores_auth, scores_intruder)
        y_true = np.concatenate([np.zeros(len(scores_auth)), np.ones(len(scores_intruder))])
        y_pred = (np.concatenate([scores_auth, scores_intruder]) >= threshold).astype(int)
        
        metrics = binary_metrics(y_true, y_pred)
        metrics["fit_time"] = fit_time
        metrics["pred_time"] = pred_time
        runs.append(metrics)
    return summarize_runs(runs)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X, y, idx2class = build_face_dataset(DATA_ROOT, size=SCALES[0], load_intruders=True)
    intruder_class_id = [k for k, v in idx2class.items() if not v.lower().startswith("subject")][0]
    X_auth, X_intruder = X[y != intruder_class_id], X[y == intruder_class_id]

    pca = PCA_np(); pca.fit(X_auth)
    q_star = int(np.searchsorted(np.cumsum(pca.ev_ratio_), 0.98) + 1)
    print(f"Dimensão original: {X.shape[1]}, Dimensão reduzida (q*): {q_star}")

    samplers = {
        "AutoencoderMLP": AutoencoderSampler(q_star),
        "OneClassSVM": OneClassSVMSampler(),
        "IsolationForest": IsolationForestSampler()
    }

    all_results = []
    for name, sampler in samplers.items():
        print(f"--- Otimizando modelo: {name} ---")
        best_params = select_best_by_random_search(X_auth, X_intruder, q_star, sampler, N_SAMPLES_RS, K_SELECT_EVAL)
        print(f"Melhores parâmetros para {name}: {best_params}")

        print(f"--- Avaliando melhor modelo: {name} ---")
        summary = eval_best_over_repeats(X_auth, X_intruder, q_star, lambda: sampler.to_model(best_params), N_REPEATS_BEST)
        
        summary["model"] = name
        summary["best_params"] = str(best_params)
        all_results.append(summary)

    out_csv_path = os.path.join(RESULTS_DIR, "tabela4_intruso_unario.csv")
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(out_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"[OK] Resultados da abordagem unária salvos em: {out_csv_path}")