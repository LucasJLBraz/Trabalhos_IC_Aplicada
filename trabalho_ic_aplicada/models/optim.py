# trabalho_ic_aplicada/models/optim.py
import numpy as np
from typing import Any, Dict

__all__ = ["make_optimizer"]

# ------- helpers para estruturas em árvore (dict/list de ndarrays) --------
def _tree_like(a: Any, ref: Any):
    """Cria um zero-like com a mesma estrutura (list/ndarray) de ref."""
    if isinstance(ref, list):
        return [np.zeros_like(x) for x in ref]
    elif isinstance(ref, np.ndarray):
        return np.zeros_like(ref)
    else:
        raise TypeError(f"Tipo de parâmetro não suportado em optimizer: {type(ref)}")

def _tree_op_inplace(params: Any, update: Any, op):
    """Aplica op(param, update) in-place sobre a mesma estrutura."""
    if isinstance(params, list):
        for i in range(len(params)):
            op(params[i], update[i])
    elif isinstance(params, np.ndarray):
        op(params, update)
    else:
        raise TypeError(f"Tipo de parâmetro não suportado: {type(params)}")

def _sub_inplace(a: np.ndarray, b: np.ndarray):
    a -= b

def _add_inplace(a: np.ndarray, b: np.ndarray):
    a += b

def _tree_axpy(dest: Any, a: float, x: Any):
    """dest += a * x (in-place) na mesma estrutura."""
    if isinstance(dest, list):
        for i in range(len(dest)):
            dest[i] += a * x[i]
    else:
        dest += a * x

def _tree_copy(x: Any):
    if isinstance(x, list):
        return [xi.copy() for xi in x]
    return x.copy()

def _tree_mul_scalar(x: Any, s: float):
    if isinstance(x, list):
        return [xi * s for xi in x]
    return x * s

def _tree_div_eps(x: Any, y: Any, eps: float):
    if isinstance(x, list):
        return [x[i] / (np.sqrt(y[i]) + eps) for i in range(len(x))]
    return x / (np.sqrt(y) + eps)

def _tree_square(x: Any):
    if isinstance(x, list):
        return [xi * xi for xi in x]
    return x * x

def _tree_zeros_like(x: Any):
    return _tree_like(x, x)

# ------------------------ otimizadores ------------------------
def make_optimizer(name: str, lr: float = 1e-2, **kw):
    """
    Retorna uma função step(params: Dict[str, Any], grads: Dict[str, Any]) -> None
    que atualiza params in-place. Suporta estruturas com listas de ndarrays.
    """
    name = name.lower()

    if name == "sgd":
        def step(params: Dict[str, Any], grads: Dict[str, Any]):
            for k in params:
                # params[k] -= lr * grads[k]
                if isinstance(params[k], list):
                    for i in range(len(params[k])):
                        params[k][i] -= lr * grads[k][i]
                else:
                    params[k] -= lr * grads[k]
        return step

    if name == "momentum":
        mu = kw.get("beta", 0.9)
        v: Dict[str, Any] = {}
        def step(params, grads):
            for k in params:
                if k not in v:
                    if isinstance(params[k], list):
                        v[k] = [np.zeros_like(p) for p in params[k]]
                    else:
                        v[k] = np.zeros_like(params[k])
                # v = mu*v + grad
                if isinstance(params[k], list):
                    for i in range(len(params[k])):
                        v[k][i] = mu * v[k][i] + grads[k][i]
                        params[k][i] -= lr * v[k][i]
                else:
                    v[k] = mu * v[k] + grads[k]
                    params[k] -= lr * v[k]
        return step

    if name == "nesterov":
        mu = kw.get("beta", 0.9)
        v: Dict[str, Any] = {}
        def step(params, grads):
            for k in params:
                if k not in v:
                    if isinstance(params[k], list):
                        v[k] = [np.zeros_like(p) for p in params[k]]
                    else:
                        v[k] = np.zeros_like(params[k])
                # NAG (Sutskever-style): v_prev = v; v = mu*v - lr*grad; param += -mu*v_prev + (1+mu)*v
                if isinstance(params[k], list):
                    v_prev = [vi.copy() for vi in v[k]]
                    for i in range(len(params[k])):
                        v[k][i] = mu * v[k][i] - lr * grads[k][i]
                        params[k][i] += -mu * v_prev[i] + (1 + mu) * v[k][i]
                else:
                    v_prev = v[k].copy()
                    v[k] = mu * v[k] - lr * grads[k]
                    params[k] += -mu * v_prev + (1 + mu) * v[k]
        return step

    if name == "rmsprop":
        rho = kw.get("rho", 0.9); eps = 1e-8
        s: Dict[str, Any] = {}
        def step(params, grads):
            for k in params:
                if k not in s:
                    if isinstance(params[k], list):
                        s[k] = [np.zeros_like(p) for p in params[k]]
                    else:
                        s[k] = np.zeros_like(params[k])
                if isinstance(params[k], list):
                    for i in range(len(params[k])):
                        s[k][i] = rho * s[k][i] + (1 - rho) * (grads[k][i] ** 2)
                        params[k][i] -= lr * grads[k][i] / (np.sqrt(s[k][i]) + eps)
                else:
                    s[k] = rho * s[k] + (1 - rho) * (grads[k] ** 2)
                    params[k] -= lr * grads[k] / (np.sqrt(s[k]) + eps)
        return step

    if name == "adam":
        b1 = kw.get("beta1", 0.9); b2 = kw.get("beta2", 0.999); eps = 1e-8
        t = {"t": 0}
        m: Dict[str, Any] = {}
        v: Dict[str, Any] = {}
        def step(params, grads):
            t["t"] += 1
            for k in params:
                # inicializa estados com a mesma estrutura
                if k not in m:
                    if isinstance(params[k], list):
                        m[k] = [np.zeros_like(p) for p in params[k]]
                        v[k] = [np.zeros_like(p) for p in params[k]]
                    else:
                        m[k] = np.zeros_like(params[k])
                        v[k] = np.zeros_like(params[k])
                if isinstance(params[k], list):
                    for i in range(len(params[k])):
                        m[k][i] = b1 * m[k][i] + (1 - b1) * grads[k][i]
                        v[k][i] = b2 * v[k][i] + (1 - b2) * (grads[k][i] ** 2)
                        mhat = m[k][i] / (1 - b1 ** t["t"])
                        vhat = v[k][i] / (1 - b2 ** t["t"])
                        params[k][i] -= lr * mhat / (np.sqrt(vhat) + eps)
                else:
                    m[k] = b1 * m[k] + (1 - b1) * grads[k]
                    v[k] = b2 * v[k] + (1 - b2) * (grads[k] ** 2)
                    mhat = m[k] / (1 - b1 ** t["t"])
                    vhat = v[k] / (1 - b2 ** t["t"])
                    params[k] -= lr * mhat / (np.sqrt(vhat) + eps)
        return step

    raise ValueError(f"Otimizador desconhecido: {name}")
