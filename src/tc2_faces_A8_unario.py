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
# Funções de Pré-processamento (Idênticas)
# =========================
def fit_and_apply_pipeline(X_train, X_val, X_test, var_retained=0.99):
    # 1. PCA
    pca = PCA_np()
    pca.fit(X_train)
    q_dim = int(np.searchsorted(np.cumsum(pca.ev_ratio_), var_retained) + 1)
    
    X_train_p = pca.transform(X_train, q=q_dim)
    X_val_p = pca.transform(X_val, q=q_dim)
    X_test_p = pca.transform(X_test, q=q_dim)

    # 2. Box-Cox + Z-Score
    # (Funções _boxcox_transform, _boxcox_ll, fit_boxcox_then_zscore, transform_boxcox_then_zscore são necessárias aqui)
    # Para simplicidade, vamos assumir que elas estão definidas neste escopo.
    X_train_n, bx_params = fit_boxcox_then_zscore(X_train_p)
    X_val_n = transform_boxcox_then_zscore(X_val_p, bx_params)
    X_test_n = transform_boxcox_then_zscore(X_test_p, bx_params)
    
    return X_train_n, X_val_n, X_test_n, q_dim

# (Cole as funções de pré-processamento de A8 aqui para o script ser autocontido)
# ... (fit_boxcox_then_zscore, etc.)

# =========================
# Métricas e Avaliação
# =========================
# (Cole as funções binary_metrics e summarize_runs aqui)

# =========================
# Modelos e Samplers
# =========================
# (Cole as classes de Wrapper e Samplers aqui)

def find_optimal_threshold(scores_auth, scores_intruder):
    """
    Encontra o limiar ótimo (threshold) que maximiza o F1-score na validação.
    Retorna (threshold, f1_score).
    """
    all_scores = np.concatenate([scores_auth, scores_intruder])
    y_true = np.concatenate([np.zeros(len(scores_auth)), np.ones(len(scores_intruder))])
    best_f1 = -1
    best_thresh = None
    for thresh in np.unique(all_scores):
        y_pred = (all_scores >= thresh).astype(int)
        metrics = binary_metrics(y_true, y_pred)
        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_thresh = thresh
    return best_thresh, best_f1

# =========================
# Protocolo Experimental Robusto
# =========================
def run_hyperparameter_search(X_auth, X_intruder, sampler, q_dim, n_samples, k_eval, seed_base=2025):
    # ... (Implementação da busca com conjunto de validação)
    pass

def run_final_evaluation(X_auth, X_intruder, model_ctor, q_dim, n_repeats, seed_base=9090):
    # ... (Implementação da avaliação final com conjunto de teste e limiar por percentil)
    pass

# =========================
# MAIN (Lógica completa a ser inserida)
# =========================
if __name__ == "__main__":
    # Este é um placeholder. O código completo será gerado na próxima etapa.
    print("Estrutura do script criada. O código completo será gerado a seguir.")

# --- Implementação completa das funções auxiliares ---

def _boxcox_transform(x_pos: np.ndarray, lam: float) -> np.ndarray:
    if lam == 0.0: return np.log(x_pos)
    return (np.power(x_pos, lam) - 1.0) / lam

def _boxcox_ll(x_pos: np.ndarray, lam: float) -> float:
    z = _boxcox_transform(x_pos, lam); n = x_pos.size; var = z.var(ddof=1) + 1e-12
    return - (n/2.0)*np.log(var) + (lam - 1.0)*np.log(x_pos).sum()

def fit_boxcox_then_zscore(X: np.ndarray, grid=None):
    if grid is None: grid = np.linspace(-2.0, 2.0, 81)
    n, d = X.shape; Xbc = np.empty_like(X, dtype=float); lambdas = np.zeros(d); shifts = np.zeros(d)
    for j in range(d):
        x = X[:, j]; mn = x.min(); shift = 0.0
        if mn <= 0: shift = -mn + 1e-6
        x_pos = x + shift; best_ll, best_lam = -np.inf, 1.0
        for lam in grid:
            ll = _boxcox_ll(x_pos, lam)
            if ll > best_ll: best_ll, best_lam = ll, lam
        Xbc[:, j] = _boxcox_transform(x_pos, best_lam); lambdas[j] = best_lam; shifts[j] = shift
    mu  = Xbc.mean(axis=0); std = Xbc.std(axis=0) + 1e-12; Xn  = (Xbc - mu) / std
    return Xn, (lambdas, shifts, mu, std)

def transform_boxcox_then_zscore(X: np.ndarray, params):
    lambdas, shifts, mu, std = params; Xbc = np.empty_like(X, dtype=float)
    for j in range(X.shape[1]):
        x_pos = X[:, j] + shifts[j]; lam = lambdas[j]; Xbc[:, j] = _boxcox_transform(x_pos, lam)
    Xn = (Xbc - mu) / std
    return Xn

def binary_metrics(y_true_bin, y_pred_bin):
    TP = np.sum((y_true_bin == 1) & (y_pred_bin == 1)); TN = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
    FP = np.sum((y_true_bin == 0) & (y_pred_bin == 1)); FN = np.sum((y_true_bin == 1) & (y_pred_bin == 0))
    acc = (TP + TN) / max(1, len(y_true_bin)); tpr = TP / max(1, TP + FN); ppv = TP / max(1, TP + FP)
    fnr = FN / max(1, TP + FN); fpr = FP / max(1, FP + TN); f1 = 2 * ppv * tpr / max(1e-9, ppv + tpr)
    return {"acc": acc, "tpr": tpr, "ppv": ppv, "fnr": fnr, "fpr": fpr, "f1_score": f1}

def summarize_runs(run_list):
    from statistics import mean, stdev, median
    agg = {};
    if not run_list: return agg
    keys = run_list[0].keys()
    for k in keys:
        vals = [r[k] for r in run_list]; agg[k+"_mean"] = mean(vals)
        agg[k+"_std"] = stdev(vals) if len(vals) > 1 else 0; agg[k+"_median"] = median(vals)
    return agg

class AutoencoderAnomalyDetector:
    def __init__(self, **kwargs): self.params = kwargs; self.model = MLPRegressor(**kwargs)
    def fit(self, X): self.model.fit(X, X); return self
    def predict_scores(self, X): return np.mean((X - self.model.predict(X))**2, axis=1)

class SklearnAnomalyWrapper:
    def __init__(self, model_ctor, params): self.params = params; self.model = model_ctor(**params)
    def fit(self, X): self.model.fit(X); return self
    def predict_scores(self, X): return -self.model.decision_function(X)

class AutoencoderSampler:
    def __init__(self, q_dim): self.q_dim = q_dim
    def __call__(self, rng):
        middle_dim = int(rng.choice([self.q_dim/4, self.q_dim/2, self.q_dim*3/4]))
        return {
            "hidden": (self.q_dim, middle_dim, self.q_dim),
            "activation": "tanh", "lr": float(rng.choice([0.005, 0.01])),
            "epochs": 200, "l2": float(rng.choice([1e-5, 1e-4])),
            "opt": "nesterov", "clip_grad": 5.0 }
    def to_model(self, p): return AutoencoderAnomalyDetector(**p)

class OneClassSVMSampler:
    def __call__(self, rng):
        return {"nu": float(rng.choice([0.01, 0.05, 0.1, 0.15])), "gamma": float(rng.choice([0.01, 0.1, 1, 10]))}
    def to_model(self, p): return SklearnAnomalyWrapper(OneClassSVM, p)

class IsolationForestSampler:
    def __call__(self, rng):
        return {"n_estimators": int(rng.choice([50, 100, 200])), "random_state": 42}
    def to_model(self, p): return SklearnAnomalyWrapper(IsolationForest, p)

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    X, y, idx2class = build_face_dataset(DATA_ROOT, size=SCALES[0], load_intruders=True)
    intruder_class_id = [k for k, v in idx2class.items() if not v.lower().startswith("subject")][0]
    X_auth, X_intruder = X[y != intruder_class_id], X[y == intruder_class_id]

    pca = PCA_np(); pca.fit(X_auth)
    q_star = int(np.searchsorted(np.cumsum(pca.ev_ratio_), 0.99) + 1)
    print(f"Dimensão original: {X.shape[1]}, Dimensão reduzida (q*={q_star} para 99% da variância)")

    samplers = {
        "AutoencoderMLP": AutoencoderSampler(q_star),
        "OneClassSVM": OneClassSVMSampler(),
        "IsolationForest": IsolationForestSampler()
    }
    all_final_results = []

    for name, sampler in samplers.items():
        print(f"--- Fase 1: Otimizando modelo: {name} ---")
        rng_search = np.random.default_rng(2025)
        best_params = None; best_f1 = -1

        for s in range(N_SAMPLES_RS):
            params = sampler(rng_search)
            f1_scores = []
            for k in range(K_SELECT_EVAL):
                # Divisão para a busca: 70% treino, 30% val
                shuffled_indices = rng_search.permutation(len(X_auth))
                train_idx, val_idx = shuffled_indices[:int(0.7*len(X_auth))], shuffled_indices[int(0.7*len(X_auth)):]
                X_auth_tr, X_auth_val = X_auth[train_idx], X_auth[val_idx]
                
                # Pipeline fit no treino, transform no treino e val
                pca_search = PCA_np(q=q_star); X_auth_tr_p = pca_search.fit_transform(X_auth_tr)
                X_auth_val_p = pca_search.transform(X_auth_val)
                X_intruder_p = pca_search.transform(X_intruder)
                X_auth_tr_n, bx_params_search = fit_boxcox_then_zscore(X_auth_tr_p)
                X_auth_val_n = transform_boxcox_then_zscore(X_auth_val_p, bx_params_search)
                X_intruder_n = transform_boxcox_then_zscore(X_intruder_p, bx_params_search)

                model = sampler.to_model(params).fit(X_auth_tr_n)
                scores_auth_val = model.predict_scores(X_auth_val_n)
                scores_intruder_val = model.predict_scores(X_intruder_n)
                
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
            shuffled_auth = rng_final.permutation(len(X_auth)); shuffled_intruder = rng_final.permutation(len(X_intruder))
            
            # Divisão final: 80% treino, 20% teste
            tr_auth_idx, te_auth_idx = shuffled_auth[:int(0.8*len(X_auth))], shuffled_auth[int(0.8*len(X_auth)):]
            X_auth_tr, X_auth_te = X_auth[tr_auth_idx], X_auth[te_auth_idx]
            X_intruder_te = X_intruder # Usamos todos os intrusos para o teste final

            pca_final = PCA_np(q=q_star); X_auth_tr_p = pca_final.fit_transform(X_auth_tr)
            X_auth_tr_n, bx_params_final = fit_boxcox_then_zscore(X_auth_tr_p)

            t0 = time.time()
            model_final = sampler.to_model(best_params).fit(X_auth_tr_n)
            fit_time = time.time() - t0

            # Limiar pelo percentil 95 dos erros de treino
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
