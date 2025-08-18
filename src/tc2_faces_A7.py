# src/tc2_faces_A7.py
import os, time, csv
import numpy as np
import matplotlib.pyplot as plt

from trabalho_ic_aplicada.dataset_faces import load_scales_from_root
from trabalho_ic_aplicada.models.pca_np import PCA_np
from trabalho_ic_aplicada.models.clf_mqo import LeastSquaresClassifier
from trabalho_ic_aplicada.models.clf_pl import SoftmaxRegression
from trabalho_ic_aplicada.models.clf_mlp import MLPClassifier

# =========================
# CONFIG
# =========================
DATA_ROOT       = "./data/raw/Kit_projeto_FACES"
SCALES          = [(20,20), (30,30), (40,40)]  # use a(s) mesma(s) de A5/A6
SELECT_SCALE_ID = -1
RESULTS_DIR     = "./results"
BASELINE_T3_CSV = os.path.join(RESULTS_DIR, "tabela3.csv")  # gerado na A6

# Busca e avaliação (iguais à A6)
N_SAMPLES_RS   = 50    # amostras no random search por modelo
K_SELECT_EVAL  = 10# repetições por candidato na seleção (estabilidade)
N_REPEATS_BEST = 50    # repetições finais para tabela

# =========================
# Utilidades
# =========================
def train_test_split_stratified(y: np.ndarray, ptrain=0.8, rng=None):
    if rng is None: rng = np.random.default_rng()
    classes = np.unique(y)
    idx_tr, idx_te = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        ntr = max(1, int(np.floor(ptrain*len(idx))))
        idx_tr.append(idx[:ntr]); idx_te.append(idx[ntr:])
    return np.concatenate(idx_tr), np.concatenate(idx_te)

def confusion_matrix(y_true, y_pred, C):
    M = np.zeros((C, C), dtype=int)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        M[t, p] += 1
    return M

def compute_metrics(y_true, y_pred, C):
    acc = float((y_true == y_pred).mean())
    M = confusion_matrix(y_true, y_pred, C)
    TP = np.diag(M).astype(float)
    FP = M.sum(axis=0) - TP
    FN = M.sum(axis=1) - TP
    with np.errstate(divide='ignore', invalid='ignore'):
        prec_c = TP / (TP + FP)
        rec_c  = TP / (TP + FN)
        f1_c   = 2*prec_c*rec_c / (prec_c + rec_c)
    prec = float(np.nanmean(prec_c))
    rec  = float(np.nanmean(rec_c))
    f1   = float(np.nanmean(f1_c))
    return {"acc": acc, "precision_macro": prec, "recall_macro": rec, "f1_macro": f1}

# =========================
# Box-Cox (implementado aqui para não tocar módulos)
# =========================
def _boxcox_transform(x_pos: np.ndarray, lam: float) -> np.ndarray:
    if lam == 0.0:
        return np.log(x_pos)
    return (np.power(x_pos, lam) - 1.0) / lam

def _boxcox_ll(x_pos: np.ndarray, lam: float) -> float:
    z = _boxcox_transform(x_pos, lam)
    n = x_pos.size
    var = z.var(ddof=1) + 1e-12
    return - (n / 2.0) * np.log(var) + (lam - 1.0) * np.log(x_pos).sum()

def fit_boxcox_then_zscore(X: np.ndarray, grid=None):
    """
    Ajusta Box-Cox por feature no conjunto de treino e, em seguida, z-score
    no espaço transformado. Retorna X_norm, (lambdas, shifts, mean, std).
    """
    if grid is None:
        grid = np.linspace(-2.0, 2.0, 81)  # passo 0.05
    n, d = X.shape
    Xbc = np.empty_like(X, dtype=float)
    lambdas = np.zeros(d)
    shifts  = np.zeros(d)
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
        shifts[j]  = shift
    # z-score no espaço transformado
    mu  = Xbc.mean(axis=0)
    std = Xbc.std(axis=0) + 1e-12
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
# Avaliação (PCA reduzida + Box-Cox + z-score)
# =========================
def eval_once_boxcox(X, y, q, model_ctor, seed=42):
    rng = np.random.default_rng(seed)
    tr, te = train_test_split_stratified(y, 0.8, rng)
    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]

    # PCA (fit no treino), redução para q
    pca = PCA_np(q=q)
    Xtr_p = pca.fit_transform(Xtr, q=q)
    Xte_p = pca.transform(Xte, q=q)

    # Box-Cox + z-score (fit no treino)
    Xtr_n, bx_params = fit_boxcox_then_zscore(Xtr_p)
    Xte_n = transform_boxcox_then_zscore(Xte_p, bx_params)

    # Treina / pred
    model = model_ctor()
    t0 = time.perf_counter()
    model.fit(Xtr_n, ytr, n_classes=int(y.max())+1)
    fit_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    yhat = model.predict(Xte_n)
    pred_time = time.perf_counter() - t1

    C = int(y.max()) + 1
    met = compute_metrics(yte, yhat, C)
    met["fit_time"]   = fit_time
    met["pred_time"]  = pred_time
    met["total_time"] = fit_time + pred_time
    return met

def summarize_runs(run_list, keys=("acc","precision_macro","recall_macro","f1_macro","fit_time","pred_time","total_time")):
    from statistics import mean, median, pstdev
    agg = {}
    for k in keys:
        vals = [r[k] for r in run_list]
        agg[k+"_mean"]   = float(mean(vals))
        agg[k+"_std"]    = float(pstdev(vals))
        agg[k+"_min"]    = float(min(vals))
        agg[k+"_max"]    = float(max(vals))
        agg[k+"_median"] = float(median(vals))
    return agg

# =========================
# Random Search (mesmo de A6, mas sem normalização extra)
# =========================
class MQSampler:
    def __call__(self, rng):
        return {"l2": float(rng.choice([0.0, 1e-4, 1e-3, 1e-2, 1e-1]))}
    def to_model(self, p): return LeastSquaresClassifier(l2=p["l2"])

class PLSampler:
    def __call__(self, rng):
        return {
            "lr":     float(rng.choice([5e-3, 1e-2, 2e-2])),
            "epochs": int(rng.choice([100, 200, 300])),
            "l2":     float(rng.choice([0.0, 1e-4, 1e-3])),
            "opt":    str(rng.choice(["sgd","momentum","nesterov","rmsprop","adam"]))
        }
    def to_model(self, p): return SoftmaxRegression(lr=p["lr"], epochs=p["epochs"], l2=p["l2"], opt=p["opt"])

class MLP1HSampler:
    def __call__(self, rng):
        acts   = ["tanh","sigmoid","relu","leaky_relu","relu6","swish"]
        h1     = [16, 32, 64, 128, 256, 512]
        return {
            "hidden":     (int(rng.choice(h1)),),
            "activation": str(rng.choice(acts)),
            "lr":         float(rng.choice([5e-3, 1e-2, 2e-2])),
            "epochs":     int(rng.choice([150, 200, 300])),
            "l2":         float(rng.choice([0.0, 1e-4, 1e-3])),
            "opt":        str(rng.choice(["sgd","momentum","nesterov","rmsprop","adam"])),
            "clip_grad":  float(rng.choice([2.0, 5.0, 10.0])),
        }
    def to_model(self, p):
        return MLPClassifier(hidden=p["hidden"], activation=p["activation"], lr=p["lr"],
                             epochs=p["epochs"], l2=p["l2"], opt=p["opt"], clip_grad=p["clip_grad"])

class MLP2HSampler:
    def __call__(self, rng):
        acts   = ["tanh","sigmoid","relu","leaky_relu","relu6","swish"]
        h1     = [16, 32, 64, 128, 256]
        h2     = [16, 32, 64, 128, 256]
        return {
            "hidden":     (int(rng.choice(h1)), int(rng.choice(h2))),
            "activation": str(rng.choice(acts)),
            "lr":         float(rng.choice([5e-3, 1e-2, 2e-2])),
            "epochs":     int(rng.choice([150, 200, 300])),
            "l2":         float(rng.choice([0.0, 1e-4, 1e-3])),
            "opt":        str(rng.choice(["sgd","momentum","nesterov","rmsprop","adam"])),
            "clip_grad":  float(rng.choice([2.0, 5.0, 10.0])),
        }
    def to_model(self, p):
        return MLPClassifier(hidden=p["hidden"], activation=p["activation"], lr=p["lr"],
                             epochs=p["epochs"], l2=p["l2"], opt=p["opt"], clip_grad=p["clip_grad"])

def select_best_by_random_search(X, y, q, sampler, n_samples=N_SAMPLES_RS, k_select=K_SELECT_EVAL, seed_base=777):
    rng = np.random.default_rng(seed_base)
    best = None
    for s in range(n_samples):
        params = sampler(rng)
        reps = []
        for k in range(k_select):
            out = eval_once_boxcox(X, y, q, model_ctor=lambda: sampler.to_model(params), seed=seed_base + s*100 + k)
            reps.append(out)
        score = float(np.mean([r["acc"] for r in reps]))
        if (best is None) or (score > best["score"]):
            best = {"params": params, "score": score}
    return best

def eval_best_over_repeats(X, y, q, best, sampler, n_repeats=N_REPEATS_BEST, seed_base=9090):
    runs = []
    for r in range(n_repeats):
        out = eval_once_boxcox(X, y, q, model_ctor=lambda: sampler.to_model(best["params"]), seed=seed_base + r)
        runs.append(out)
    agg = summarize_runs(runs)
    return runs, agg

# =========================
# Comparativo com baseline (Tabela 3 da A6)
# =========================
def load_baseline_table3(path):
    base = {}
    if not os.path.isfile(path):
        return base
    import csv
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            base[row["Model"]] = row
    return base

def save_comparative_csv(baseline, boxcox_rows, out_csv):
    # baseline: dict Model -> row (strings)
    # boxcox_rows: list of dicts (num types)
    with open(out_csv, "w", newline="") as f:
        cols = ["Model",
                "acc_mean_base","acc_mean_boxcox","acc_delta",
                "f1_mean_base","f1_mean_boxcox","f1_delta",
                "fit_time_mean_base","fit_time_mean_boxcox","fit_time_delta"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in boxcox_rows:
            model = r["Model"]
            b = baseline.get(model)
            if b is None:
                continue
            def f(x):
                try: return float(x)
                except: return np.nan
            acc_b  = f(b.get("acc_mean",""))
            f1_b   = f(b.get("f1_mean",""))
            fit_b  = f(b.get("fit_time_mean",""))
            acc_bc = r["acc_mean"]
            f1_bc  = r["f1_mean"]
            fit_bc = r["fit_time_mean"]
            w.writerow({
                "Model": model,
                "acc_mean_base": acc_b, "acc_mean_boxcox": acc_bc, "acc_delta": (acc_bc - acc_b) if np.isfinite(acc_b) else np.nan,
                "f1_mean_base":  f1_b,  "f1_mean_boxcox":  f1_bc,  "f1_delta":  (f1_bc - f1_b)   if np.isfinite(f1_b)  else np.nan,
                "fit_time_mean_base": fit_b, "fit_time_mean_boxcox": fit_bc, "fit_time_delta": (fit_bc - fit_b) if np.isfinite(fit_b) else np.nan
            })

def save_comparative_figure(baseline, boxcox_rows, out_png):
    # barplot simples: acc_mean (base vs boxcox)
    models = []
    acc_base, acc_box = [], []
    for r in boxcox_rows:
        m = r["Model"]
        b = baseline.get(m)
        if b is None:
            continue
        models.append(m)
        try:
            acc_base.append(float(b.get("acc_mean","nan")))
        except:
            acc_base.append(np.nan)
        acc_box.append(r["acc_mean"])
    x = np.arange(len(models))
    w = 0.35
    plt.figure()
    plt.bar(x - w/2, acc_base, width=w, label="Sem Box-Cox (A6)")
    plt.bar(x + w/2, acc_box,  width=w, label="Com Box-Cox (A7)")
    plt.xticks(x, models)
    plt.ylabel("Acurácia média")
    plt.title("Comparativo A6 vs A7 (acurácia média)")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

# =========================
# MAIN — A7
# =========================
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Carrega dataset(s) e escolhe a mesma escala da A6
    datasets = load_scales_from_root(DATA_ROOT, SCALES)
    scale_label, X_sel, y_sel = datasets[SELECT_SCALE_ID]

    # Recupera q escolhido na A5 (ou calcula de novo se não existir)
    q_file = os.path.join(RESULTS_DIR, "pca_q_98.txt")
    if os.path.isfile(q_file):
        with open(q_file, "r") as f:
            lines = f.read().strip().splitlines()
        # última linha deve estar no formato: q (...): <int>
        try:
            q_star = int(lines[-1].split(":")[-1].strip())
        except Exception:
            q_star = None
    else:
        q_star = None

    if q_star is None:
        # fallback: calcula de novo
        pca = PCA_np()
        pca.fit(X_sel)
        evr = pca.ev_ratio_
        q_star = int(np.searchsorted(np.cumsum(evr), 0.98) + 1)

    print(f"[INFO] Escala selecionada: {scale_label} | d={X_sel.shape[1]} | q*={q_star} (≥98%)")

    # Random search por modelo com pipeline PCA reduzida + Box-Cox + z-score
    models = {
        "MQ":      MQSampler(),
        "PL":      PLSampler(),
        "MLP-1H":  MLP1HSampler(),
        "MLP-2H":  MLP2HSampler(),
    }

    rows_bc = []
    for name, sampler in models.items():
        best = select_best_by_random_search(X_sel, y_sel, q_star, sampler)
        runs, agg = eval_best_over_repeats(X_sel, y_sel, q_star, best, sampler)

        P = best["params"]
        row = {
            "Scale": scale_label, "q": q_star, "Model": name,
            "Norm": "boxcox+zscore",  # fixo neste pipeline
            "Opt": P.get("opt",""), "Act": P.get("activation",""),
            "Hidden": str(P.get("hidden","")), "LR": P.get("lr",""),
            "Epochs": P.get("epochs",""), "L2": P.get("l2",""),
            "acc_mean": agg["acc_mean"], "acc_std": agg["acc_std"], "acc_min": agg["acc_min"], "acc_max": agg["acc_max"], "acc_median": agg["acc_median"],
            "precision_mean": agg["precision_macro_mean"], "recall_mean": agg["recall_macro_mean"], "f1_mean": agg["f1_macro_mean"],
            "fit_time_mean": agg["fit_time_mean"], "pred_time_mean": agg["pred_time_mean"], "total_time_mean": agg["total_time_mean"],
        }
        rows_bc.append(row)

    # Tabela 3 (versão Box-Cox)
    cols = ["Scale","q","Model","Norm","Opt","Act","Hidden","LR","Epochs","L2",
            "acc_mean","acc_std","acc_min","acc_max","acc_median",
            "precision_mean","recall_mean","f1_mean",
            "fit_time_mean","pred_time_mean","total_time_mean"]
    with open(os.path.join(RESULTS_DIR, "tabela3_boxcox.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows_bc: w.writerow(r)
    print(f"[OK] A7: results -> {os.path.join(RESULTS_DIR, 'tabela3_boxcox.csv')}")

    # Comparativo com baseline (A6)
    baseline = load_baseline_table3(BASELINE_T3_CSV)
    comp_csv = os.path.join(RESULTS_DIR, "tabela3_comparativo_A7.csv")
    save_comparative_csv(baseline, rows_bc, comp_csv)
    print(f"[OK] Comparativo salvo em {comp_csv}")

    comp_png = os.path.join(RESULTS_DIR, "comparativo_A7_acc.png")
    save_comparative_figure(baseline, rows_bc, comp_png)
    print(f"[OK] Figura comparativa salva em {comp_png}")

    print("A7 finalizado.")
