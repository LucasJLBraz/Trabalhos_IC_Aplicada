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
SCALES          = [(30,30)]
SELECT_SCALE_ID = -1
RESULTS_DIR     = "./results/TC2/"

N_SAMPLES_RS   = 200
K_SELECT_EVAL  = 10
N_REPEATS_BEST = 50

# =========================
# Box-Cox + z-score (mantemos aqui para não mexer nos módulos)
# =========================
def _boxcox_transform(x_pos: np.ndarray, lam: float) -> np.ndarray:
    """
    Transforms the input data using the Box-Cox transformation.

    Box-Cox transformation is a statistical technique used to stabilize the variance
    and make the data more closely follow a normal distribution. The transformation
    is controlled by the parameter `lam`. When `lam` is 0, the natural logarithm
    is applied instead of the Box-Cox formula.

    :param x_pos: A numpy array containing the input data to be transformed.
        All values in the array must be positive.
    :param lam: The Box-Cox transformation parameter. Determines the
        power to which the input is raised before scaling.
    :return: Transformed numpy array after applying the Box-Cox transformation.
    """
    if lam == 0.0: return np.log(x_pos)
    return (np.power(x_pos, lam) - 1.0) / lam

def _boxcox_ll(x_pos: np.ndarray, lam: float) -> float:
    """

    """
    z = _boxcox_transform(x_pos, lam)
    n = x_pos.size
    var = z.var(ddof=1) + 1e-12
    return - (n/2.0)*np.log(var) + (lam - 1.0)*np.log(x_pos).sum()

def fit_boxcox_then_zscore(X: np.ndarray, grid=None):
    """
    Transform the input data by applying the Box-Cox transformation followed by z-score normalization.

    The function first applies the Box-Cox transformation to stabilize variance and make the data
    more normally distributed. If the data contains non-positive values, a small positive shift is added
    to make all values positive. The optimal Box-Cox parameter (`lambda`) is determined by maximizing
    the log-likelihood over a given grid of candidate values. After the transformation, the data is
    normalized using z-score standardization by subtracting the mean and dividing by the standard deviation.

    :param X: The input data as a 2D array of shape (n_samples, n_features).
    :param grid: An optional 1D array representing the grid of lambda values to search for the best
        Box-Cox transformation parameter. If None, a default grid in the range [-2.0, 2.0] with
        step size 0.05 is used.
    :return: A tuple containing the normalized data and a tuple with the transformation parameters:
        lambdas (optimal Box-Cox lambdas for each feature), shifts (applied shifts for positivity),
        mu (means of transformed data for each feature), std (standard deviations post transformation).
    """
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
    """
    Transforms the input data using the Box-Cox transformation followed by Z-score
    normalization. The transformation adjusts the input data based on provided
    parameters including shifts, lambda values for the Box-Cox transformation,
    and the normalization parameters (mean and standard deviation). The resulting
    data is normalized to follow standard normal distribution per feature.

    :param X: Array of shape (n_samples, n_features) containing the input data to
              be transformed.
    :type X: np.ndarray
    :param params: A tuple containing transformation parameters:
                   - `lambdas`: List or array of lambda parameters for Box-Cox
                     transformation for each feature.
                   - `shifts`: List or array of shifts to add for each feature
                     (ensuring positive values).
                   - `mu`: Array of means of the features for Z-score normalization.
                   - `std`: Array of standard deviations of the features for Z-score
                     normalization.

    :return: Array of transformed data after applying Box-Cox transformation and
             Z-score normalization, with the same shape as the input array `X`.
    :rtype: np.ndarray
    """
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
    """
    Calculates a dictionary of binary classification metrics including accuracy,
    true positive rate (TPR), precision (PPV), false negative rate (FNR),
    and false positive rate (FPR) based on the provided true and predicted
    binary arrays.

    :param y_true_bin: Ground truth binary values, where 0 represents the
        absence of a condition and 1 represents the presence of a condition
        (e.g., a positive detection or the presence of an intruder).
    :type y_true_bin: numpy.ndarray

    :param y_pred_bin: Predicted binary values, where 0 represents the predicted
        absence and 1 represents the predicted presence of the condition.
    :type y_pred_bin: numpy.ndarray

    :return: A dictionary containing the following keys:
             - "acc": Accuracy of the predictions (TP + TN) / Total.
             - "tpr": True Positive Rate, also known as sensitivity.
             - "ppv": Positive Predictive Value, also known as precision.
             - "fnr": False Negative Rate, the complement of TPR.
             - "fpr": False Positive Rate, ratio of false positives.
    :rtype: dict
    """
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
    """
    Summarizes performance metrics and timing statistics for a list of runs, calculating
    mean, standard deviation, minimum, maximum, and median for each selected key.

    :param run_list: List of dictionaries, where each dictionary corresponds to the metrics
        of a single run. Each dictionary must contain the specified keys.
    :type run_list: list[dict]
    :param keys: Tuple of keys to extract from the dictionaries for summarization. The default
        keys include common metrics such as "acc" (accuracy), "fnr" (false negative rate),
        "fpr" (false positive rate), "tpr" (true positive rate), "ppv" (positive predictive value),
        and timing metrics like "fit_time", "pred_time", and "total_time".
    :type keys: tuple[str]
    :return: A dictionary containing the aggregated statistics (mean, standard deviation, minimum,
        maximum, and median) for each specified key with suffixes denoting the statistic type
        (e.g., "_mean", "_std", "_min", "_max", "_median").
    :rtype: dict
    """
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
    """
    Splits an array into stratified training and testing indices based on
    class distribution. The function ensures that the training and testing
    sets contain approximately the same proportion of samples for each
    class as the original dataset.

    :param y: Array of class labels (target variable).
    :type y: np.ndarray
    :param ptrain: Proportion of data to be included in the training set.
        Defaults to 0.8.
    :type ptrain: float
    :param rng: Random number generator instance. Defaults to
        `numpy.random.default_rng()`.
    :type rng: numpy.random.Generator, optional
    :return: A tuple containing two arrays: training indices and testing
        indices, both stratified by class.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
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
    """
    MQSampler is a callable class used for generating hyperparameters and converting them into
    a model configuration.

    This class provides functionality to sample hyperparameters using a random generator instance
    and map the sampled parameters into a model-specific configuration.

    """
    def __call__(self, rng):
        return {"l2": float(rng.choice([0.0, 1e-4, 1e-3, 1e-2, 1e-1]))}
    def to_model(self, p): return LeastSquaresClassifier(l2=p["l2"])

class PLSampler:
    """

    """
    def __call__(self, rng):
        return {"lr": float(rng.choice([5e-3, 1e-2, 2e-2])),
                "epochs": int(rng.choice([100, 200, 300])),
                "l2": float(rng.choice([0.0, 1e-4, 1e-3])),
                "opt": str(rng.choice(["sgd","momentum","nesterov","rmsprop","adam"]))}
    def to_model(self, p): return SoftmaxRegression(lr=p["lr"], epochs=p["epochs"], l2=p["l2"], opt=p["opt"])

class MLP1HSampler:
    """
    A class responsible for sampling hyperparameters for a single hidden layer
    Multi-Layer Perceptron (MLP) model and converting the sampled parameters
    into an actual model instance.

    This class provides methods to randomly sample a set of hyperparameters for
    training a neural network and to create a model instance configured with
    the sampled parameters. The purpose of this class is to facilitate
    experiments with different configurations of MLP hyperparameters.

    :ivar acts: List of available activation functions for the MLP model.
    :type acts: list[str]
    :ivar h1: List of possible sizes for the MLP hidden layer.
    :type h1: list[int]
    """
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
    """
    Represents a configurable sampler for generating hyperparameters of a Multi-Layer Perceptron (MLP) model.

    This class provides methods to generate random hyperparameters for MLPs, including hidden
    layers, activation functions, learning rate, number of training epochs, regularization (L2),
    optimizer type, and gradient clipping value. It also provides a method to create an MLP classifier
    model using the generated hyperparameters.

    :ivar activation_choices: List of available activation functions.
    :type activation_choices: list[str]
    :ivar hidden_layer_sizes: List of available options for the number of neurons in each hidden layer.
    :type hidden_layer_sizes: list[int]
    :ivar learning_rates: List of available learning rates.
    :type learning_rates: list[float]
    :ivar epochs_choices: List of available options for the number of training epochs.
    :type epochs_choices: list[int]
    :ivar l2_regularization: List of available L2 regularization values.
    :type l2_regularization: list[float]
    :ivar optimizers: List of available optimizer types.
    :type optimizers: list[str]
    :ivar gradient_clipping_values: List of available gradient clipping values.
    :type gradient_clipping_values: list[float]
    """
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
    """
    Evaluates a machine learning model on a dataset, factoring in preprocessing steps such
    as PCA for dimensionality reduction and normalization (Box-Cox and z-score transformation),
    with particular emphasis on detecting a specific "intruder" class.

    The function performs the following operations:
    1. Stratified train-test split of the dataset.
    2. Reduction of data dimensionality using PCA up to the specified number of components.
    3. Application of Box-Cox transformation and z-score normalization.
    4. Training of the specified machine learning model on the preprocessed training data.
    5. Computation of predictions on the test set.
    6. Evaluation of classification metrics, including binary metrics for detecting the "intruder"
       class, as well as benchmark times (model fitting, prediction, total time).

    :param X: Dataset/features, represented as a NumPy array of shape (n_samples, n_features).
    :param y: Labels corresponding to the dataset, represented as a NumPy array of shape
        (n_samples,).
    :param intruder_id: Integer (or equivalent label) identifying the "intruder" category/class
        that needs to be detected in the data.
    :param q: Number of principal components to retain during PCA transformation.
    :param model_ctor: Callable that instantiates and returns the machine learning model to be
        evaluated.
    :param seed: Random seed for reproducibility. Defaults to 42.
    :return: Dictionary containing binary classification metrics, as well as time metrics for
        model fitting, prediction, and total time.
    """
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
    """
    Performs a random search over a parameter space to select the best model configuration
    based on the F1 score of the intruder. This function evaluates several parameter
    combinations using the provided sampler and selects the one that achieves the best
    performance.

    The selection process involves sampling a set of parameters, evaluating the model on
    multiple runs using these parameters, and calculating the average score. The parameters
    that yield the highest F1 score for the intruder are returned as the best configuration.

    :param X: Input data to be used for evaluation.
    :type X: Any
    :param y: Target labels corresponding to the input data.
    :type y: Any
    :param intruder_id: Identifier for the intruder class.
    :type intruder_id: int or str
    :param q: Arbitrary additional parameter required for evaluation.
    :type q: Any
    :param sampler: Callable that generates random hyperparameters for the model.
    :type sampler: Callable
    :param n_samples: Number of random configurations to sample. Defaults to `N"""
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
    """
    Evaluates the performance of the best configuration over multiple repeated runs using a given sampler and intruder setup.

    The method executes the evaluation process for `n_repeats` times with a unique seed for each iteration to ensure variability
    among runs. It essentially utilizes the provided sampler and `best` parameters to create a model, runs evaluations against
    the intruder scenarios, and aggregates the results.

    :param X: Input data used as the feature set for evaluation.
    :type X: Any
    :param y: Target labels corresponding to the feature set X.
    :type y: Any
    :param intruder_id: Identifier specifying the intruder for the evaluation.
    :type intruder_id: Any
    :param q: Query or specific configuration for the evaluation runs.
    :type q: Any
    :param best: Dictionary containing the best parameters to configure the model.
    :type best: dict
    :param sampler: Object capable of creating a model using provided parameters.
    :type sampler: Any
    :param n_repeats: Number of times the evaluation is repeated. Default is N_REPEATS_BEST.
    :type n_repeats: int
    :param seed_base: Seed value used as a base to generate unique seeds for each iteration. Default is 9090.
    :type seed_base: int
    :return: A tuple containing all runs' outputs and their aggregated summary.
    :rtype: tuple
    """
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

    # Carrega escala escolhida e obtém mapeamento de classes, ativando a flag para intrusos
    scale = SCALES[SELECT_SCALE_ID]
    X, y, idx2class = build_face_dataset(DATA_ROOT, size=scale, load_intruders=True)
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
