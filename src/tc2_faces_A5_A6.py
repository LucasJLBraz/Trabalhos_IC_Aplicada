# src/tc2_faces_A5_A6.py
import os, time, csv, json
import numpy as np
import matplotlib.pyplot as plt

from trabalho_ic_aplicada.dataset_faces import load_scales_from_root
from trabalho_ic_aplicada.models.preprocess_np import apply_norm, NormSpec
from trabalho_ic_aplicada.models.pca_np import PCA_np
from trabalho_ic_aplicada.models.clf_mqo import LeastSquaresClassifier
from trabalho_ic_aplicada.models.clf_pl import SoftmaxRegression
from trabalho_ic_aplicada.models.clf_mlp import MLPClassifier

# =========================
# CONFIG
# =========================
DATA_ROOT       = "./data/raw/Kit_projeto_FACES"
SELECT_SCALE_ID = -1   # use a “escala OK” (ex.: última da lista)
VAR_TARGET      = 0.98 # ≥ 98% da variância

SCALES    = [(30,30)]  # compare as que quiser
N_SAMPLES_RS   = 200   # nº de amostras da busca aleatória por modelo
K_SELECT_EVAL  = 10   # repetições por candidato na seleção (trade-off tempo/estabilidade)
N_REPEATS_BEST = 50   # repetições finais para estatísticas da Tabela (pedidas no enunciado)
RESULTS_DIR    = "./results/TC2/"

# =========================
# Utils
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
    # macro
    prec = float(np.nanmean(prec_c))
    rec  = float(np.nanmean(rec_c))
    f1   = float(np.nanmean(f1_c))
    return {"acc": acc, "precision_macro": prec, "recall_macro": rec, "f1_macro": f1}

def eval_once(X, y, model_ctor, norm_name="none", use_pca=False, pca_q=None, seed=42):
    rng = np.random.default_rng(seed)
    tr, te = train_test_split_stratified(y, 0.8, rng)
    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]

    # PCA (fit no treino, transform no teste) — evita vazamento
    if use_pca:
        pca = PCA_np(q=pca_q)
        Xtr = pca.fit_transform(Xtr, q=pca_q)
        Xte = pca.transform(Xte, q=pca_q)

    Xtr, Xte, _ = apply_norm(Xtr, Xte, NormSpec(norm_name))

    model = model_ctor()
    t0 = time.perf_counter()
    model.fit(Xtr, ytr, n_classes=int(y.max())+1)
    fit_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    yhat = model.predict(Xte)
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
# Samplers (iguais ao A1–A4, mas aqui só reusamos)
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
        h1     = [16, 32, 64, 128, 256]
        return {
            "hidden":     (int(rng.choice(h1)),),
            "activation": str(rng.choice(acts)),
            "lr":         float(rng.choice([5e-3, 1e-2, 2e-2])),
            "epochs":     int(rng.choice([150, 200, 300])),
            "l2":         float(rng.choice([0.0, 1e-4, 1e-3])),
            "opt":        str(rng.choice(["sgd","momentum","nesterov","rmsprop","adam"])),
            "clip_grad":  float(rng.choice([2.0, 5.0, 10.0]))
        }
    def to_model(self, p):
        return MLPClassifier(hidden=p["hidden"], activation=p["activation"], lr=p["lr"],
                             epochs=p["epochs"], l2=p["l2"], opt=p["opt"], clip_grad=p["clip_grad"])

class MLP2HSampler:
    def __call__(self, rng):
        acts   = ["tanh","sigmoid","relu","leaky_relu","relu6","swish"]
        h1     = [32, 64, 128, 256]
        h2     = [16, 32, 64, 128]
        return {
            "hidden":     (int(rng.choice(h1)), int(rng.choice(h2))),
            "activation": str(rng.choice(acts)),
            "lr":         float(rng.choice([5e-3, 1e-2, 2e-2])),
            "epochs":     int(rng.choice([150, 200, 300])),
            "l2":         float(rng.choice([0.0, 1e-4, 1e-3])),
            "opt":        str(rng.choice(["sgd","momentum","nesterov","rmsprop","adam"])),
            "clip_grad":  float(rng.choice([2.0, 5.0, 10.0]))
        }
    def to_model(self, p):
        return MLPClassifier(hidden=p["hidden"], activation=p["activation"], lr=p["lr"],
                             epochs=p["epochs"], l2=p["l2"], opt=p["opt"], clip_grad=p["clip_grad"])

def select_best_by_random_search(
    X, y, sampler, use_pca=False, pca_q=None, n_samples=N_SAMPLES_RS,
    norm_space=("none","zscore","minmax","minmax_pm1"), k_select=K_SELECT_EVAL, seed_base=2025
):
    rng = np.random.default_rng(seed_base)
    best = None
    for s in range(n_samples):
        params = sampler(rng)
        norm   = rng.choice(norm_space)
        reps = []
        for k in range(k_select):
            out = eval_once(X, y,
                            model_ctor=lambda: sampler.to_model(params),
                            norm_name=norm, use_pca=use_pca, pca_q=pca_q,
                            seed=seed_base + s*100 + k)
            reps.append(out)
            
        score = np.mean([r["f1_macro"] for r in reps])

        # score = np.mean([r["acc"] for r in reps])
        if (best is None) or (score > best["score"]):
            best = {"params": params, "norm": norm, "score": float(score)}
    return best

def eval_best_over_repeats(X, y, best, sampler, use_pca=False, pca_q=None, n_repeats=N_REPEATS_BEST, seed_base=9090):
    runs = []
    for r in range(n_repeats):
        out = eval_once(
            X, y,
            model_ctor=lambda: sampler.to_model(best["params"]),
            norm_name=best["norm"], use_pca=use_pca, pca_q=pca_q,
            seed=seed_base + r
        )
        runs.append(out)
    agg = summarize_runs(runs)
    return runs, agg

# =========================
# A5 — Curva da variância explicada (da A3) e escolha de q
# =========================
def compute_and_save_explained_variance_figure(X, outpath_png, var_target=0.98):
    pca = PCA_np()
    pca.fit(X)
    evr = pca.ev_ratio_
    csum = np.cumsum(evr)
    q = int(np.searchsorted(csum, var_target) + 1)

    # figura
    plt.figure()
    x = np.arange(1, len(evr)+1)
    plt.plot(x, csum, marker=".", linewidth=1)
    plt.axhline(var_target, linestyle="--")
    plt.axvline(q, linestyle="--")
    plt.title(f"Variância explicada acumulada (q*={q} para ≥ {int(var_target*100)}%)")
    plt.xlabel("Componentes principais")
    plt.ylabel("Variância explicada acumulada")
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(outpath_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath_png, dpi=150)
    plt.close()

    return q, csum

# =========================
# MAIN — A5–A6
# =========================
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- Carrega a mesma(s) escala(s) usadas antes e escolhe a “OK”
    datasets = load_scales_from_root(DATA_ROOT, SCALES)
    scale_label, X_sel, y_sel = datasets[SELECT_SCALE_ID]
    d = X_sel.shape[1]

    # ---- (A5) Curva de variância explicada acumulada (da A3) e escolha de q≥98%
    fig_path = os.path.join(RESULTS_DIR, "pca_variance_explained_A3.png")
    q_star, csum = compute_and_save_explained_variance_figure(X_sel, fig_path, var_target=VAR_TARGET)
    print(f"[OK] Figura de variância explicada salva em {fig_path}")
    # salva q (para Questão 5)
    with open(os.path.join(RESULTS_DIR, "pca_q_98.txt"), "w") as f:
        f.write(f"Escala: {scale_label}\n")
        f.write(f"d (original): {d}\n")
        f.write(f"q (>= {int(VAR_TARGET*100)}%): {q_star}\n")

    # ---- (A6) RS com PCA reduzida (q = q_star) e 50 repetições dos melhores
    models = {
        "MQ":      MQSampler(),
        "PL":      PLSampler(),
        "MLP-1H":  MLP1HSampler(),
        "MLP-2H":  MLP2HSampler(),
    }

    rows_t3 = []
    for name, sampler in models.items():
        best = select_best_by_random_search(X_sel, y_sel, sampler, use_pca=True, pca_q=q_star)
        runs, agg = eval_best_over_repeats(X_sel, y_sel, best, sampler, use_pca=True, pca_q=q_star)

        P = best["params"]
        row = {
            "Scale": scale_label, "q": q_star, "Model": name, "Norm": best["norm"],
            "Opt": P.get("opt",""), "Act": P.get("activation",""),
            "Hidden": str(P.get("hidden","")), "LR": P.get("lr",""),
            "Epochs": P.get("epochs",""), "L2": P.get("l2",""),
            "acc_mean": agg["acc_mean"], "acc_std": agg["acc_std"], "acc_min": agg["acc_min"], "acc_max": agg["acc_max"], "acc_median": agg["acc_median"],
            "precision_mean": agg["precision_macro_mean"], "recall_mean": agg["recall_macro_mean"], "f1_mean": agg["f1_macro_mean"],
            "fit_time_mean": agg["fit_time_mean"], "pred_time_mean": agg["pred_time_mean"], "total_time_mean": agg["total_time_mean"],
            "__internal_best_obj": best
        }
        rows_t3.append(row)

    cols = ["Scale","q","Model","Norm","Opt","Act","Hidden","LR","Epochs","L2",
            "acc_mean","acc_std","acc_min","acc_max","acc_median",
            "precision_mean","recall_mean","f1_mean",
            "fit_time_mean","pred_time_mean","total_time_mean"]
    with open(os.path.join(RESULTS_DIR, "tabela3.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows_t3:
            row_to_write = r.copy()
            del row_to_write['__internal_best_obj']
            w.writerow(row_to_write)
    print(f"[OK] A6 (PCA reduzida): results -> {os.path.join(RESULTS_DIR, 'tabela3.csv')}")

    # # ---- (Extra) Plot Matriz de Confusão para o melhor modelo da A6 ----
    # def plot_confusion_matrix(M, classes, title, outpath):
    #     import seaborn as sns
    #     plt.figure(figsize=(8, 8))
    #     sns.heatmap(M, annot=True, fmt="d", cbar=False, cmap="Blues",
    #                 xticklabels=classes, yticklabels=classes)
    #     plt.title(title)
    #     plt.xlabel("Predito")
    #     plt.ylabel("Verdadeiro")
    #     plt.tight_layout()
    #     os.makedirs(os.path.dirname(outpath), exist_ok=True)
    #     plt.savefig(outpath, dpi=150)
    #     plt.close()
    #
    # sampler_map = {"MQ": MQSampler(), "PL": PLSampler(), "MLP-1H": MLP1HSampler(), "MLP-2H": MLP2HSampler()}
    # C = int(y_sel.max()) + 1
    # classes_labels = [f"S{i+1}" for i in range(C)]
    # rng_plot = np.random.default_rng(1234)
    # tr, te = train_test_split_stratified(y_sel, 0.8, rng_plot)
    # Xtr, Xte, ytr, yte = X_sel[tr], X_sel[te], y_sel[tr], y_sel[te]
    #
    # best_row_a6 = max(rows_t3, key=lambda r: r['acc_mean'])
    # model_name_a6 = best_row_a6['Model']
    # best_obj_a6 = best_row_a6['__internal_best_obj']
    # sampler_a6 = sampler_map[model_name_a6]
    # model_a6 = sampler_a6.to_model(best_obj_a6['params'])
    #
    # pca = PCA_np(q=q_star)
    # Xtr_p = pca.fit_transform(Xtr)
    # Xte_p = pca.transform(Xte)
    # Xtr_a6, Xte_a6, _ = apply_norm(Xtr_p, Xte_p, NormSpec(best_obj_a6["norm"]))
    #
    # model_a6.fit(Xtr_a6, ytr, n_classes=C)
    # yhat_a6 = model_a6.predict(Xte_a6)
    # M_a6 = confusion_matrix(yte, yhat_a6, C)
    # cm_path_a6 = os.path.join(RESULTS_DIR, f"cm_A6_{model_name_a6}.png")
    # plot_confusion_matrix(M_a6, classes_labels, f"Matriz de Confusão - A6 (PCA Reduzido) / {model_name_a6}", cm_path_a6)
    # print(f"[OK] Figura Matriz de Confusão (A6) salva em {cm_path_a6}")
    #
    # print("A5–A6 finalizado.")
