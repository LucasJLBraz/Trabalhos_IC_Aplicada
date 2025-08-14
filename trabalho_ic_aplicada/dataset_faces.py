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

def _build_from_flat(root_dir: str, size: tuple[int, int]):
    files = sorted([f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))])
    # filtra óbvios não-imagem
    skip_suffixes = {".m", ".txt", ".dat", ".zip", ".eps", ".wav", ".jpg", ".jpeg"}  # jpg real entra via imageio no caminho feliz; eps/wav/zip não
    paths = []
    for f in files:
        if f.startswith("."):
            continue
        if any(f.lower().endswith(suf) for suf in skip_suffixes):
            # exceção: queremos permitir .jpg reais — mas já filtramos acima. Ajuste se precisar.
            if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".pgm",".ppm",".pbm")):
                pass
            else:
                continue
        # aceita nomes tipo subject01.centerlight (sem extensão)
        m = _SUBJECT_RE.match(f)
        if m is None:
            # ainda pode ser imagem normal com extensão; deixa imageio decidir
            pass
        paths.append(os.path.join(root_dir, f))

    if not paths:
        raise ValueError(f"Nenhum arquivo de imagem elegível encontrado em {root_dir}.")

    X_list, y_list = [], []
    idx2class = {}
    class_map = {}  # str subjectNN -> int id
    for path in paths:
        base = os.path.basename(path)
        m = _SUBJECT_RE.match(base)
        if m is None:
            # tenta assim mesmo (pode ser PNG/JPG etc em um flat set)
            try:
                img = _imread_any(path)
            except Exception:
                continue
            cname = "unknown"
        else:
            cname = f"subject{int(m.group(1)):02d}"
        if cname not in class_map:
            class_map[cname] = len(class_map)
        cid = class_map[cname]
        try:
            img = _imread_any(path)
        except Exception:
            continue
        g = _to_gray(img)
        g = _resize_nn(g, size)
        X_list.append(g.flatten())
        y_list.append(cid)

    if not X_list:
        raise ValueError(f"Nenhuma imagem válida lida em {root_dir} (formato flat).")

    X = np.vstack(X_list)
    y = np.asarray(y_list, dtype=int)
    # monta idx2class ordenado por id
    idx2class = {v: k for k, v in class_map.items()}
    return X, y, idx2class

def build_face_dataset(root_dir: str, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Suporta:
      1) subpastas por classe (root_dir/sujeitoX/arquivos...)
      2) diretório flat com nomes estilo Yale A: subjectNN.<condicao>
    """
    sub = _build_from_subfolders(root_dir, size)
    if sub is not None:
        return sub
    # se não há subpastas, tenta formato flat
    return _build_from_flat(root_dir, size)

def load_scales_from_root(root_dir: str, sizes: list[tuple[int,int]]):
    """
    Constrói datasets para várias escalas: [(label, X, y), ...]
    """
    datasets = []
    for (h, w) in sizes:
        X, y, _ = build_face_dataset(root_dir, (h, w))
        datasets.append((f"{h}x{w}", X, y))
    return datasets
