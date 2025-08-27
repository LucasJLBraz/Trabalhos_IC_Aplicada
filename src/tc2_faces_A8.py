# src/tc2_faces_A8.py
import os, time, csv
import numpy as np

from trabalho_ic_aplicada.dataset_faces import build_face_dataset
from trabalho_ic_aplicada.models.pca_np import PCA_np
from trabalho_ic_aplicada.models.clf_mqo import LeastSquaresClassifier
from trabalho_ic_aplicada.models.clf_pl import SoftmaxRegression
from trabalho_ic_aplicada.models.clf_mlp import MLPClassifier

# =========================
# CONFIG (iguais aos seus)
# =========================
DATA_ROOT       = "./data/raw/Kit_projeto_FACES"
SCALES          = [(20,20), (30,30), (40,40)]
SELECT_SCALE_ID = -1
RESULTS_DIR     = "./results"

N_SAMPLES_RS   = 60
K_SELECT_EVAL  = 10
N_REPEATS_BEST = 50

# =========================
# Box-Cox + z-score (mantemos aqui para não mexer nos módulos)
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
    if grid is None: grid = np.linspace(-2.0, 2.0, 81)  # passo 0.05
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
# Métricas binárias (intruso = classe positiva)
# =========================
def binary_metrics(y_true_bin, y_pred_bin):
    y_true_bin = y_true_bin.astype(int); y_pred_bin = y_pred_bin.astype(int)
    TP = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    TN = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
    FP = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    FN = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
    acc = (TP + TN) / max(1, TP + TN + FP + FN)
    tpr = TP / max(1, TP + FN)      # sensibilidade (intruso detectado)
    ppv = TP / max(1, TP + FP)      # precisão (PPV) para intruso
    fnr = 1 - tpr                   # taxa de falsos negativos (intruso permitido)
    fpr = FP / max(1, FP + TN)      # taxa de falsos positivos (autorizado negado)
    return {"acc": acc, "tpr": tpr, "ppv": ppv, "fnr": fnr, "fpr": fpr}

def summarize_runs(run_list, keys=("acc","fnr","fpr","tpr","ppv","fit_time","pred_time","total_time")):
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
# Split estratificado por sujeito (multiclasse), depois agregamos intruso vs autorizado
# =========================
def train_test_split_stratified(y: np.ndarray, ptrain=0.8, rng=None):
    if rng is None: rng = np.random.default_rng()
    classes = np.unique(y)
    tr, te = [], []
    for c in classes:
        idx = np.where(y==c)[0]
        rng.shuffle(idx)
        ntr = max(1, int(np.floor(ptrain*len(idx))))
        tr.append(idx[:ntr]); te.append(idx[ntr:])
    return np.concatenate(tr), np.concatenate(te)

# =========================
# Samplers (seu espaço de busca)
# =========================
class MQSampler:
    def __call__(self, rng):
        return {"l2": float(rng.choice([0.0, 1e-4, 1e-3, 1e-2, 1e-1]))}
    def to_model(self, p): return LeastSquaresClassifier(l2=p["l2"])

class PLSampler:
    def __call__(self, rng):
        return {"lr": float(rng.choice([5e-3, 1e-2, 2e-2])),
                "epochs": int(rng.choice([100, 200, 300])),
                "l2": float(rng.choice([0.0, 1e-4, 1e-3])),
                "opt": str(rng.choice(["sgd","momentum","nesterov","rmsprop","adam"]))}
    def to_model(self, p): return SoftmaxRegression(lr=p["lr"], epochs=p["epochs"], l2=p["l2"], opt=p["opt"])

class MLP1HSampler:
    def __call__(self, rng):
        acts = ["tanh","sigmoid","relu","leaky_relu","relu6","swish"]
        h1   = [16,32,64,128,256,512]
        return {"hidden": (int(rng.choice(h1)),),
                "activation": str(rng.choice(acts)),
                "lr": float(rng.choice([5e-3, 1e-2, 2e-2])),
                "epochs": int(rng.choice([150, 200, 300])),
                "l2": float(rng.choice([0.0, 1e-4, 1e-3])),
                "opt": str(rng.choice(["sgd","momentum","nesterov","rmsprop","adam"])),
                "clip_grad": float(rng.choice([2.0, 5.0, 10.0]))}
    def to_model(self, p):
        return MLPClassifier(hidden=p["hidden"], activation=p["activation"], lr=p["lr"],
                             epochs=p["epochs"], l2=p["l2"], opt=p["opt"], clip_grad=p["clip_grad"])

class MLP2HSampler:
    def __call__(self, rng):
        acts = ["tanh","sigmoid","relu","leaky_relu","relu6","swish"]
        h    = [16,32,64,128,256,512]
        return {"hidden": (int(rng.choice(h)), int(rng.choice(h))),
                "activation": str(rng.choice(acts)),
                "lr": float(rng.choice([5e-3, 1e-2, 2e-2])),
                "epochs": int(rng.choice([150, 200, 300])),
                "l2": float(rng.choice([0.0, 1e-4, 1e-3])),
                "opt": str(rng.choice(["sgd","momentum","nesterov","rmsprop","adam"])),
                "clip_grad": float(rng.choice([2.0, 5.0, 10.0]))}
    def to_model(self, p):
        return MLPClassifier(hidden=p["hidden"], activation=p["activation"], lr=p["lr"],
                             epochs=p["epochs"], l2=p["l2"], opt=p["opt"], clip_grad=p["clip_grad"])

# =========================
# Eval: PCA(q) -> Box-Cox -> z-score -> classificador
# Seleção pelo F1 da classe intruso (equilíbrio entre FN e FP)
# =========================
def eval_once_intruder(X, y, intruder_id, q, model_ctor, seed=42):
    rng = np.random.default_rng(seed)
    tr, te = train_test_split_stratified(y, 0.8, rng)
    Xtr, Xte = X[tr], X[te]; ytr, yte = y[tr], y[te]

    # PCA reduzida
    pca = PCA_np(q=q)
    Xtr_p = pca.fit_transform(Xtr, q=q)
    Xte_p = pca.transform(Xte, q=q)

    # Box-Cox + z-score
    Xtr_n, bx = fit_boxcox_then_zscore(Xtr_p)
    Xte_n = transform_boxcox_then_zscore(Xte_p, bx)

    # Treino / predição
    model = model_ctor()
    t0 = time.perf_counter()
    model.fit(Xtr_n, ytr, n_classes=int(y.max())+1)
    fit_time = time.perf_counter() - t0
    t1 = time.perf_counter()
    yhat = model.predict(Xte_n)
    pred_time = time.perf_counter() - t1

    # Binário: intruso = 1; autorizados = 0
    y_true_bin = (yte == intruder_id).astype(int)
    y_pred_bin = (yhat == intruder_id).astype(int)
    m = binary_metrics(y_true_bin, y_pred_bin)
    m["fit_time"] = fit_time; m["pred_time"] = pred_time; m["total_time"] = fit_time + pred_time
    # F1 para seleção (classe positiva = intruso)
    prec = m["ppv"]; rec = m["tpr"]
    m["f1_intruso"] = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
    return m

def select_best_by_random_search(X, y, intruder_id, q, sampler, n_samples=N_SAMPLES_RS, k_select=K_SELECT_EVAL, seed_base=20258):
    rng = np.random.default_rng(seed_base)
    best = None
    for s in range(n_samples):
        params = sampler(rng)
        reps = []
        for k in range(k_select):
            out = eval_once_intruder(X, y, intruder_id, q, model_ctor=lambda: sampler.to_model(params), seed=seed_base + s*100 + k)
            reps.append(out)
        score = float(np.mean([r["f1_intruso"] for r in reps]))  # critério: F1 do intruso
        if (best is None) or (score > best["score"]):
            best = {"params": params, "score": score}
    return best

def eval_best_over_repeats(X, y, intruder_id, q, best, sampler, n_repeats=N_REPEATS_BEST, seed_base=9090):
    runs = []
    for r in range(n_repeats):
        out = eval_once_intruder(X, y, intruder_id, q, model_ctor=lambda: sampler.to_model(best["params"]), seed=seed_base + r)
        runs.append(out)
    agg = summarize_runs(runs)
    return runs, agg

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Carrega escala escolhida e obtém mapeamento de classes
    scale = SCALES[SELECT_SCALE_ID]
    X, y, idx2class = build_face_dataset(DATA_ROOT, size=scale)
    # Descobre id do intruso: qualquer classe cujo nome NÃO comece por "subject"
    intruder_ids = [k for k,v in idx2class.items() if not v.lower().startswith("subject")]
    if len(intruder_ids) != 1:
        raise RuntimeError(f"Esperava 1 classe de intruso; encontrei {len(intruder_ids)}: {[idx2class[i] for i in intruder_ids]}")
    intruder_id = intruder_ids[0]
    print(f"[INFO] Escala {scale[0]}x{scale[1]} | d={X.shape[1]} | classe intruso = {idx2class[intruder_id]} (id={intruder_id})")

    # Recupera q* (≥98%) salvo na A5, ou recalcula
    q_file = os.path.join(RESULTS_DIR, "pca_q_98.txt")
    if os.path.isfile(q_file):
        lines = open(q_file).read().strip().splitlines()
        try: q_star = int(lines[-1].split(":")[-1].strip())
        except Exception: q_star = None
    else:
        q_star = None
    if q_star is None:
        pca = PCA_np(); pca.fit(X)
        q_star = int(np.searchsorted(np.cumsum(pca.ev_ratio_), 0.98) + 1)
    print(f"[INFO] q* (≥98%): {q_star}")

    # Espaço de busca
    models = {
        "MQ":      MQSampler(),
        "PL":      PLSampler(),
        "MLP-1H":  MLP1HSampler(),
        "MLP-2H":  MLP2HSampler(),
    }

    rows = []
    for name, sampler in models.items():
        best = select_best_by_random_search(X, y, intruder_id, q_star, sampler)
        runs, agg = eval_best_over_repeats(X, y, intruder_id, q_star, best, sampler)
        P = best["params"]
        row = {
            "Scale": f"{scale[0]}x{scale[1]}", "q": q_star, "Model": name,
            "Hidden": str(P.get("hidden","")), "Act": P.get("activation",""),
            "Opt": P.get("opt",""), "LR": P.get("lr",""), "Epochs": P.get("epochs",""), "L2": P.get("l2",""), "Clip": P.get("clip_grad",""),
            # métricas (média/disp.)
            "acc_mean": agg["acc_mean"], "acc_std": agg["acc_std"], "acc_min": agg["acc_min"], "acc_max": agg["acc_max"], "acc_median": agg["acc_median"],
            "fnr_mean": agg["fnr_mean"], "fnr_std": agg["fnr_std"], "fpr_mean": agg["fpr_mean"], "fpr_std": agg["fpr_std"],
            "tpr_mean": agg["tpr_mean"], "ppv_mean": agg["ppv_mean"],
            "fit_time_mean": agg["fit_time_mean"], "pred_time_mean": agg["pred_time_mean"], "total_time_mean": agg["total_time_mean"],
        }
        rows.append(row)

    cols = ["Scale","q","Model","Hidden","Act","Opt","LR","Epochs","L2","Clip",
            "acc_mean","acc_std","acc_min","acc_max","acc_median",
            "fnr_mean","fnr_std","fpr_mean","fpr_std","tpr_mean","ppv_mean",
            "fit_time_mean","pred_time_mean","total_time_mean"]

    out_csv = os.path.join(RESULTS_DIR, "tabela4_intruso.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"[OK] A8: resultados -> {out_csv}")
