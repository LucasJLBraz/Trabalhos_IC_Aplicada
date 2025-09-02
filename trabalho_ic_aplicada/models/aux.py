import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def calcular_metricas_normalizadas(y_pred_testes, y_reais_testes):
    """
    Calcula o valor médio do output e o RMSE normalizado pelo valor médio para múltiplos folds.
    
    Args:
        y_pred_testes: lista de arrays, onde cada array contém as predições de um fold
        y_reais_testes: lista de arrays, onde cada array contém os valores reais de um fold
        
    Returns:
        valor_medio: média das predições
        valor_medio_std: desvio padrão das médias
        rmse_normalizado: média dos RMSEs normalizados
        rmse_normalizado_std: desvio padrão dos RMSEs normalizados
    """
    valores_medios = []
    rmses_normalizados = []
    
    for y_pred, y_real in zip(y_pred_testes, y_reais_testes):
        valor_medio = np.mean(y_pred)
        # Calcula RMSE: raiz quadrada da média das diferenças quadráticas
        rmse = np.sqrt(np.mean((y_real - y_pred) ** 2))
        rmse_normalizado = rmse / max(abs(valor_medio), 1e-10)
        
        valores_medios.append(valor_medio)
        rmses_normalizados.append(rmse_normalizado)
    
    # Calcula médias e desvios padrão
    valor_medio_final = np.mean(valores_medios)
    valor_medio_std = np.std(valores_medios)
    rmse_normalizado_final = np.mean(rmses_normalizados)
    rmse_normalizado_std = np.std(rmses_normalizados)



    # print(f"Avg. house price: {valor_medio_final:.4f} ± {valor_medio_std:.4f}")
    # print(f"RSME/Avg. house price: {eqm_normalizado_final:.4f} ± {eqm_normalizado_std:.4f}")
    print(f"Avg. house price: {valor_medio_final:.4f} ± {valor_medio_std:.4f}")
    print(f"RSME/Avg. house price: {rmse_normalizado_final:.4f} ± {rmse_normalizado_std:.4f}")

    return valor_medio_final, valor_medio_std, rmse_normalizado_final, rmse_normalizado_std


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



def plot_disperssao_hist_residuo(y_train: np.ndarray, y_train_pred: np.ndarray, 
                                y_test: np.ndarray, y_test_pred: np.ndarray, 
                                title: str) -> None:
    """
    Plota a dispersão dos plots e histogramas dos resíduos para os conjuntos de treino e teste.
    Inclui linha de tendência e linha x=y para comparação.

    Args:
        y_train (np.ndarray): Valores reais do conjunto de treino.
        y_train_pred (np.ndarray): Valores previstos pelo modelo no conjunto de treino.
        y_test (np.ndarray): Valores reais do conjunto de teste.
        y_test_pred (np.ndarray): Valores previstos pelo modelo no conjunto de teste.
        title (str): Titulo do plot.
    """
    # Calcula os resíduos
    residuos_train = y_train - y_train_pred
    residuos_test = y_test - y_test_pred

    # Cria a figura e os eixos
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    plt.title(title)

    # Configuração comum para os plots de dispersão
    def config_scatter_plot(ax, x, y, title_suffix):
        # Scatter plot
        ax.scatter(x, y, alpha=0.5)
        
        # Linha x=y (diagonal ideal)
        min_val = min(np.min(x), np.min(y))
        max_val = max(np.max(x), np.max(y))
        ax.plot([min_val, max_val], [min_val, max_val], 
                'gray', linestyle='--', alpha=0.8, label='x=y')
        
        # Linha de tendência
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "r-", alpha=0.8, label='Tendência')
        
        # Configurações do plot
        ax.set_title(f'Dispersão do Real X Previsto - {title_suffix}')
        ax.set_xlabel(f'Valores Real - {title_suffix}')
        ax.set_ylabel(f'Valores Previsto - {title_suffix}')
        ax.legend()
        
        # Ajusta os limites
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

    # Plot de dispersão - Treino
    config_scatter_plot(axs[0, 0], y_train, y_train_pred, 'Treino')

    # Histograma dos resíduos - Treino
    sns.histplot(residuos_train, bins=30, alpha=0.7, kde=True, ax=axs[0, 1])
    axs[0, 1].set_title('Histograma dos Resíduos - Treino')
    axs[0, 1].set_xlabel('Resíduos - Treino')
    axs[0, 1].set_ylabel('Frequência')

    # Plot de dispersão - Teste
    config_scatter_plot(axs[1, 0], y_test, y_test_pred, 'Teste')

    # Histograma dos resíduos - Teste
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


    aux.plot_disperssao_hist_residuo(y_train, y_pred_train, y_test, y_pred_test, titulo)

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


def plot_folds_loss(train_losses, titulo='Histórico da Função de Perda - Treino (todos os folds)', ylabel='Loss (Erro Quadrático)', xlabel='Época'):
    """
    Plota todas as curvas de loss dos folds e também a curva média com desvio padrão.

    Args:
        train_losses (list of list or np.array): Lista onde cada elemento é uma lista/array com o histórico de loss de um fold.
        titulo (str): Título do gráfico.
        ylabel (str): Rótulo do eixo y.
        xlabel (str): Rótulo do eixo x.
    """
    plt.figure(figsize=(10,6))
    for i, loss_history in enumerate(train_losses):
        plt.plot(loss_history, alpha=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(titulo)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Média e desvio padrão entre folds
    train_losses_array = np.array(train_losses)
    mean_loss = train_losses_array.mean(axis=0)
    std_loss = train_losses_array.std(axis=0)

    plt.figure(figsize=(10,6))
    plt.plot(mean_loss, color='black', linewidth=2)
    plt.fill_between(range(len(mean_loss)), mean_loss-std_loss, mean_loss+std_loss, color='gray', alpha=0.2, label='±1 Desvio padrão')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Loss médio e desvio padrão entre folds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()