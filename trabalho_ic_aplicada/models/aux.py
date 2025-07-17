import numpy as np
from scipy import stats



def error_rit_rate(y_test:np.ndarray, y_pred:np.ndarray, porcentagem:float = 0.20) -> float:
    """"
    Calcula quantas previsões estão dentro de uma porcentagem do valor real.

    Args:
        y_test (np.ndarray): Valores reais.
        y_pred (np.ndarray): Valores previstos pelo modelo.
        porcentagem (float): Porcentagem de tolerância para o erro.

    Returns:
        float: Proporção de previsões dentro da porcentagem do valor real.
    """

    if len(y_test) != len(y_pred):
        raise ValueError("y_test e y_pred devem ter o mesmo tamanho.")

    # Calcula o erro absoluto
    erro_absoluto = np.abs(y_test - y_pred)

    # Calcula o limite de erro baseado na porcentagem
    limite_erro = porcentagem * np.abs(y_test)

    # Verifica quantas previsões estão dentro do limite de erro
    dentro_limite = erro_absoluto <= limite_erro

    # Calcula a proporção de previsões dentro do limite
    proporcao = np.mean(dentro_limite).astype(float)

    return proporcao

def validacao_cruzada_kfold(X: np.ndarray, y: np.ndarray, k: int = 10) -> list:
    """
    Realiza validação cruzada k-fold. Pega os dados de entrada e os divide em k folds, retornando os
    indices de treino e teste para cada fold.

    Args:
        X (np.ndarray): Matriz de características.
        y (np.ndarray): Vetor de rótulos.
        k (int): Número de folds.

    Returns:
        tuple[np.ndarray, np.ndarray]: Dois arrays contendo os índices de treino e teste para cada fold.
    """

    n_amostras = X.shape[0] # Número de amostras

    # Pega uma porção dos dados para treino | (total / k) * (k - 1) || E teste | (total / k);
    # Faz isso k vezes, cada vez pegando uma porção diferente para teste, uma janela deslizante.

    indices = np.arange(n_amostras)  # Cria um array de índices
    np.random.shuffle(indices)  # Embaralha os índices para garantir aleatoriedade
    fold_size = n_amostras // k  # Tamanho de cada fold

    folds = []  # Lista para armazenar os folds
    for i in range(k):
        start = i * fold_size

        # Garante que o último fold pegue o restante
        if i < k - 1:
            end = start + fold_size
        else:
            end = n_amostras

        test_indices = indices[start:end]  # Índices de teste para este fold
        train_indices = np.concatenate((indices[:start], indices[end:]))  # Índices de treino para este fold
        folds.append((train_indices, test_indices))  # Armazena os índices de treino e teste

    return folds  # Retorna a lista de folds com índices de treino e teste



def plot_disperssao_hist_residuo(y_train: np.ndarray, y_train_pred: np.ndarray, y_test: np.ndarray, y_test_pred: np.ndarray, title: str) -> None:
    """
    Plota a dispersão dos plots e histogramas dos resíduos para os conjuntos de treino e teste.

    Args:
        y_train (np.ndarray): Valores reais do conjunto de treino.
        y_train_pred (np.ndarray): Valores previstos pelo modelo no conjunto de treino.
        y_test (np.ndarray): Valores reais do conjunto de teste.
        y_test_pred (np.ndarray): Valores previstos pelo modelo no conjunto de teste.
        title (str): Titulo do plot.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Calcula os resíduos
    residuos_train = y_train - y_train_pred
    residuos_test = y_test - y_test_pred

    # Cria a figura e os eixos
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    plt.title(title)
    # Gráfico de dispersão dos resíduos do treino
    axs[0, 0].scatter(y_train, y_train_pred, alpha=0.5)
    # axs[0, 0].axhline(0, color='red', linestyle='--')
    axs[0, 0].set_title('Dispersão do Real X Previsto - Treino')
    axs[0, 0].set_xlabel('Valores Real - Treino')
    axs[0, 0].set_ylabel('Valores Previsto - Treino')

    # Ajusta os eixos para melhor visualização- ambos os eixos devem ter o mesmo limite
    min_val = min(np.min(y_train), np.min(y_train_pred))
    max_val = max(np.max(y_train), np.max(y_train_pred))

    axs[0, 0].set_xlim(min_val, max_val)
    axs[0, 0].set_ylim(min_val, max_val)

    # Histograma dos resíduos do treino
    sns.histplot(residuos_train, bins=30, alpha=0.7, kde=True, ax=axs[0, 1])
    axs[0, 1].set_title('Histograma dos Resíduos - Treino')
    axs[0, 1].set_xlabel('Resíduos - Treino')
    axs[0, 1].set_ylabel('Frequência')

    # Gráfico de dispersão dos resíduos do teste
    axs[1, 0].scatter(y_test, y_test_pred, alpha=0.5)
    # axs[1, 0].axhline(0, color='red', linestyle='--')
    axs[1, 0].set_title('Dispersão do Real X Previsto - Teste')
    axs[1, 0].set_xlabel('Valores Real - Treino')
    axs[1, 0].set_ylabel('Valores Previsto - Treino')

    # Ajusta os eixos para melhor visualização- ambos os eixos devem ter o mesmo limite
    # min_val = min(np.min(y_test), np.min(y_test_pred))
    # max_val = max(np.max(y_test), np.max(y_test_pred))

    axs[1, 0].set_xlim(min_val, max_val)
    axs[1, 0].set_ylim(min_val, max_val)

    #
    # axs[1, 0].set_xlim(0, 80)
    # axs[1, 0].set_ylim(0, 80)

    # Histograma dos resíduos do teste
    sns.histplot(residuos_test, bins=30, alpha=0.7, kde=True, ax=axs[1, 1])
    axs[1, 1].set_title('Histograma dos Resíduos - Teste')
    axs[1, 1].set_xlabel('Resíduos - Teste')
    axs[1, 1].set_ylabel('Frequência')



    # Ajusta o layout
    plt.tight_layout()
    plt.show()


# Função para calcular métricas de avaliação
def calcular_metricas(y_true, y_pred, aux):
    residuos = y_true - y_pred
    eqm = np.mean(residuos ** 2)
    reqm = np.sqrt(eqm)
    r2 = 1 - (np.sum(residuos ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    hit_20 = aux.error_rit_rate(y_true, y_pred, 0.20)
    hit_10 = aux.error_rit_rate(y_true, y_pred, 0.10)
    return eqm, reqm, r2, hit_20, hit_10, residuos

# Função para imprimir as métricas com média e desvio padrão
def imprimir_metricas(eqms, reqms, r2s, hits_20, hits_10):
    print(f"Erro Quadrático Médio (EQM): {np.mean(eqms):.4f} ± {np.std(eqms):.4f}")
    print(f"Raiz do Erro Quadrático Médio (REQM): {np.mean(reqms):.4f} ± {np.std(reqms):.4f}")
    print(f"Hit rate 20%: {np.mean(hits_20):.4f} ± {np.std(hits_20):.4f}")
    print(f"Hit rate 10%: {np.mean(hits_10):.4f} ± {np.std(hits_10):.4f}\n")

# Função para análise dos resíduos (shapiro + gráfico)
def analisar_residuos(y_train, y_pred_train, y_test, y_pred_test, aux, titulo=''):
    aux.plot_disperssao_hist_residuo(y_train, y_pred_train, y_test, y_pred_test, titulo)

    res_train = y_train - y_pred_train
    res_test = y_test - y_pred_test

    shapiro_treino = stats.shapiro(res_train)
    shapiro_teste = stats.shapiro(res_test)

    print("\n--- Análise de Gaussianidade dos Resíduos ---")
    print(f"Shapiro-Wilk (Treino): Estatística={shapiro_treino.statistic:.4f}, p-valor={shapiro_treino.pvalue:.4f}")
    print(f"Shapiro-Wilk (Teste): Estatística={shapiro_teste.statistic:.4f}, p-valor={shapiro_teste.pvalue:.4f}")

    if shapiro_treino.pvalue < 0.05:
        print("Resíduos de treino NÃO seguem distribuição normal.")
    else:
        print("Resíduos de treino seguem distribuição normal.")

# Função para imprimir correlações
def imprimir_correlacoes(corrs_treino, corrs_teste, r2s):
    print("--- Coeficientes de Correlação (Real vs. Previsto) ---")
    print(f"Correlação Média (Treino): {np.mean(corrs_treino):.4f} ± {np.std(corrs_treino):.4f}")
    print(f"Correlação Média (Teste): {np.mean(corrs_teste):.4f} ± {np.std(corrs_teste):.4f}")
    print(f"Coeficiente de Determinação (R²): {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")




def normalizar_zscore(X: np.ndarray) -> np.ndarray:
    """
    Normaliza os dados usando o método Z-score.

    Args:
        X (np.ndarray): Matriz de características a ser normalizada.

    Returns:
        np.ndarray: Matriz normalizada.
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("X deve ser um array numpy.")

    # Calcula a média e o desvio padrão
    media = np.mean(X, axis=0)
    desvio_padrao = np.std(X, axis=0)

    # Evita divisão por zero
    desvio_padrao[desvio_padrao == 0] = 1

    # Normaliza os dados
    X_normalizado = (X - media) / desvio_padrao

    return X_normalizado


def normalizar_minmax(X: np.ndarray) -> np.ndarray:
    """
    Normaliza os dados usando o método Min-Max.

    Args:
        X (np.ndarray): Matriz de características a ser normalizada.

    Returns:
        np.ndarray: Matriz normalizada.
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("X deve ser um array numpy.")

    # Calcula o mínimo e máximo
    minimo = np.min(X, axis=0)
    maximo = np.max(X, axis=0)

    # Evita divisão por zero
    range_valor = maximo - minimo
    range_valor[range_valor == 0] = 1

    # Normaliza os dados
    X_normalizado = (X - minimo) / range_valor

    return X_normalizado

def normalizar_robusto(X: np.ndarray) -> np.ndarray:
    """
    Normaliza os dados usando o método Robusto.

    Args:
        X (np.ndarray): Matriz de características a ser normalizada.

    Returns:
        np.ndarray: Matriz normalizada.
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("X deve ser um array numpy.")

    # Calcula a mediana e o intervalo interquartil
    mediana = np.median(X, axis=0)
    q75, q25 = np.percentile(X, [75, 25], axis=0)
    intervalo_interquartil = q75 - q25

    # Evita divisão por zero
    intervalo_interquartil[intervalo_interquartil == 0] = 1

    # Normaliza os dados
    X_normalizado = (X - mediana) / intervalo_interquartil

    return X_normalizado
