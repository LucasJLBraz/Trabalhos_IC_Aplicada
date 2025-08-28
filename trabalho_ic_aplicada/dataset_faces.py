# trabalho_ic_aplicada/dataset_faces.py
import os
import re
import numpy as np

try:
    import imageio.v3 as iio  # leve e lê PGM/JPG/PNG
except Exception as e:
    raise RuntimeError(
        "Instale imageio para leitura de imagens: pip install imageio"
    ) from e

__all__ = ["build_face_dataset", "load_scales_from_root"]

# -----------------------------
# Utilidades de imagem
# -----------------------------
def _to_gray(arr: np.ndarray) -> np.ndarray:
    # Se RGB, converte para cinza por média (simples e suficiente p/ Yale A)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return arr.astype(np.float64)

def _resize_nn(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    H, W = img.shape[:2]
    h, w = size
    ys = (np.arange(h) * (H / h)).astype(int)
    xs = (np.arange(w) * (W / w)).astype(int)
    return img[ys][:, xs]

def _read_pgm_fallback(path: str) -> np.ndarray:
    """Leitor PGM (P2/P5) simplificado, para arquivos sem extensão."""
    with open(path, "rb") as f:
        data = f.read()
    # separa header do corpo
    # remove comentários
    def _tokens(b):
        # transforma em texto só para parsear cabeçalho de forma robusta
        txt = []
        i = 0
        while i < len(b):
            if b[i:i+1] == b'#':
                # pular até fim da linha
                while i < len(b) and b[i:i+1] not in (b'\n', b'\r'):
                    i += 1
            else:
                txt.append(b[i:i+1])
            i += 1
        return b"".join(txt)

    clean = _tokens(data)
    parts = clean.split()
    if len(parts) < 4:
        raise ValueError("PGM inválido.")
    magic = parts[0]
    if magic not in (b"P2", b"P5"):
        raise ValueError("Apenas P2/P5 suportado no fallback.")
    w = int(parts[1]); h = int(parts[2]); maxval = int(parts[3])
    header_len = len(b" ".join(parts[:4])) + 1  # +1 por espaço/linha
    if magic == b"P2":
        # ASCII: pixels seguem em texto após o cabeçalho
        pix = np.fromstring(clean[header_len:].decode("ascii"), sep=" ", dtype=float)
    else:
        # Binário: encontre o offset real do início dos dados (após \n do maxval)
        # localiza a sequência do cabeçalho original (com comentários removidos já)
        # fallback: procura primeira ocorrência do maxval como bytes na versão 'clean'
        # e pula um byte de whitespace
        # Para ser mais robusto, reparse o arquivo original:
        with open(path, "rb") as f:
            # lê linha mágica
            f.readline()
            # pula comentários e lê dims
            line = f.readline()
            while line.startswith(b'#'):
                line = f.readline()
            # pode já conter w h ou só um
            dims = line.split()
            while len(dims) < 2:
                dims += f.readline().split()
            # lê maxval
            line = f.readline()
            while line.startswith(b'#'):
                line = f.readline()
            # agora começa o binário
            raw = f.read()
        dtype = np.uint8 if maxval < 256 else ">u2"
        pix = np.frombuffer(raw, dtype=dtype).astype(float)
    if pix.size != w * h:
        # às vezes há quebras extras; tente truncar/ajustar
        pix = pix[: w*h]
    img = pix.reshape((h, w))
    if maxval > 0:
        img = img * (255.0 / maxval)
    return img

def _imread_any(path: str) -> np.ndarray:
    # tenta com imageio; se falhar (sem extensão, plugin), usa fallback PGM
    try:
        return iio.imread(path)
    except Exception:
        return _read_pgm_fallback(path)

# -----------------------------
# Builders
# -----------------------------
def _build_from_subfolders(root_dir: str, size: tuple[int, int]):
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    if not classes:
        return None  # deixa o caller decidir o plano B (flat)
    X_list, y_list = [], []
    idx2class = {}
    for cid, cname in enumerate(classes):
        idx2class[cid] = cname
        cpath = os.path.join(root_dir, cname)
        files = sorted([f for f in os.listdir(cpath) if not f.startswith(".")])
        for f in files:
            path = os.path.join(cpath, f)
            if not os.path.isfile(path):
                continue
            try:
                img = _imread_any(path)
            except Exception:
                continue
            g = _to_gray(img)
            g = _resize_nn(g, size)
            X_list.append(g.flatten())
            y_list.append(cid)
    if not X_list:
        raise ValueError(f"Nenhuma imagem válida encontrada nas subpastas de {root_dir}.")
    X = np.vstack(X_list)
    y = np.asarray(y_list, dtype=int)
    return X, y, idx2class

_SUBJECT_RE = re.compile(r"^subject(\d+)", re.IGNORECASE)
_INTRUDER_RE = re.compile(r"^intruso", re.IGNORECASE)

def _build_from_flat(root_dir: str, size: tuple[int, int], load_intruders: bool = False):
    files = sorted([f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))])
    
    X_list, y_list = [], []
    class_map = {}  # str subjectNN -> int id

    for f in files:
        if f.startswith("."):
            continue

        m_subj = _SUBJECT_RE.match(f)
        m_intr = _INTRUDER_RE.match(f) if load_intruders else None

        cname = None
        if m_subj:
            cname = f"subject{int(m_subj.group(1)):02d}"
        elif m_intr:
            cname = "intruder"
        else:
            continue  # Ignora arquivos que não correspondem a nenhum padrão

        if cname not in class_map:
            class_map[cname] = len(class_map)
        cid = class_map[cname]

        try:
            path = os.path.join(root_dir, f)
            img = _imread_any(path)
            g = _to_gray(img)
            g = _resize_nn(g, size)
            X_list.append(g.flatten())
            y_list.append(cid)
        except Exception:
            continue

    if not X_list:
        raise ValueError(f"Nenhuma imagem válida lida em {root_dir} (formato flat).")

    X = np.vstack(X_list)
    y = np.asarray(y_list, dtype=int)
    idx2class = {v: k for k, v in class_map.items()}
    return X, y, idx2class

def build_face_dataset(root_dir: str, size: tuple[int, int], load_intruders: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Suporta:
      1) subpastas por classe (root_dir/sujeitoX/arquivos...)
      2) diretório flat com nomes estilo Yale A: subjectNN.<condicao>
    Se `load_intruders` for True, também carrega arquivos com padrão `intrusoXX`.
    """
    # A lógica de subpastas não precisa mudar, pois intrusos geralmente estão no modo flat.
    sub = _build_from_subfolders(root_dir, size)
    if sub is not None:
        # Esta parte é uma simplificação; se intrusos pudessem estar em subpastas,
        # a lógica precisaria ser mais complexa. Para este projeto, está OK.
        return sub
    return _build_from_flat(root_dir, size, load_intruders=load_intruders)

def load_scales_from_root(root_dir: str, sizes: list[tuple[int,int]], load_intruders: bool = False):
    """
    Constrói datasets para várias escalas: [(label, X, y), ...]
    """
    datasets = []
    for (h, w) in sizes:
        X, y, _ = build_face_dataset(root_dir, (h, w), load_intruders=load_intruders)
        datasets.append((f"{h}x{w}", X, y))
    return datasets

