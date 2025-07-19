import numpy as np
def treinar_perceptron_reg(X_treino: np.ndarray, y_treino: np.ndarray, epocas: int = 100, taxa_aprendizagem: float = 0.001) -> (np.ndarray, list):
    """
    Treina um modelo de regressão linear usando um perceptron com SGD.

    Args:
        X_treino (np.ndarray): Matriz de características de treino.
        y_treino (np.ndarray): Vetor de rótulos de treino.
        epocas (int): Número de épocas para o treinamento.
        taxa_aprendizagem (float): Taxa de aprendizado para o SGD.

    Returns:
       W np.ndarray: Vetor de pesos treinados do modelo.
       curva_aprendizagem_mse list: Lista com o erro quadrático médio (MSE) por época.
    """
    # Adiciona a coluna de '1's para o bias
    X_treino_b = np.c_[np.ones(X_treino.shape[0]), X_treino]

    # Número de características
    n_features = X_treino_b.shape[1]

    # Inicialização dos pesos
    W = 0.01 * np.random.randn(1, n_features)

    curva_aprendizagem_mse = []

    # --- ETAPA DE TREINAMENTO ---
    for epoca in range(epocas):
        erro_quadratico_acumulado = 0

        # Embaralhar os dados de treino a cada época para o SGD
        indices = np.random.permutation(X_treino_b.shape[0])
        X_treino_b_shuffled = X_treino_b[indices]
        y_treino_shuffled = y_treino[indices]

        for xi, alvo in zip(X_treino_b_shuffled, y_treino_shuffled):
            # Propagação Direta (Forward Pass)
            ativacao = np.dot(W, xi)
            predicao = ativacao  # Função de ativação LINEAR: y = u

            # Cálculo do erro
            erro = alvo - predicao

            # Atualização dos Pesos (Regra Delta Generalizada para neurônio linear)
            W += taxa_aprendizagem * erro * xi

            erro_quadratico_acumulado += erro**2

        mse_epoca = erro_quadratico_acumulado / X_treino_b.shape[0]
        curva_aprendizagem_mse.append(mse_epoca)
        if (epoca + 1) % 20 == 0:
            print(f"Época {epoca+1}/{epocas}, MSE Treino: {mse_epoca.item():.4f}")

    return W, curva_aprendizagem_mse


def prever_perceptron_reg(X_teste: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Realiza predições usando o modelo de perceptron treinado.

    Args:
        X_teste (np.ndarray): Matriz de características de teste.
        W (np.ndarray): Vetor de pesos do modelo treinado.

    Returns:
        np.ndarray: Vetor com as predições.
    """
    # Adiciona a coluna de '1's para o bias
    X_teste_b = np.c_[np.ones(X_teste.shape[0]), X_teste]

    # Realiza a predição através do produto escalar
    y_predito = np.dot(X_teste_b, W.T).flatten()

    return y_predito






#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

# from trabalho_ic_aplicada.models.aux import plot_disperssao_hist_residuo
# from trabalho_ic_aplicada.models.reg_perceptron import treinar_perceptron_reg, prever_perceptron_reg
#%%
# fetch dataset
real_estate_valuation = fetch_ucirepo(id=477)

features = real_estate_valuation.variables.iloc[1:, 0].values
# data (as pandas dataframes)
X = real_estate_valuation.data.features.to_numpy()

# Removendo a primeira coluna (X1 transaction date) que não é um atributo relevante, segundo analise.
X = X[:, 1:]

y = real_estate_valuation.data.targets.to_numpy().ravel()
#%%

# --- Parâmetros da Simulação ---
n_epocas = 100
n_rodadas = 10
taxa_aprendizagem = 0.001
pct_treino = 0.8

# --- Armazenamento de Métricas ---
metricas_rmse = []
metricas_r2 = []

# --- Início do Treino (Loop de Rodadas - Monte Carlo) ---
for r in range(n_rodadas):
    print(f"--- Rodada {r+1}/{n_rodadas} ---")

    # 1. Divisão dos dados em treino e teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, train_size=pct_treino, shuffle=True
    )

    # 2. Normalização
    # X_treino_norm, X_teste_norm = normalizacao_zscore(X_treino, X_teste)
    scaler_quantile = QuantileTransformer(n_quantiles=X_treino.shape[0], output_distribution='uniform')
    X_treino_norm = scaler_quantile.fit_transform(X_treino)
    X_teste_norm = scaler_quantile.transform(X_teste)

    curva_aprendizagem_mse = []

    W, curva_mse = treinar_perceptron_reg(X_treino_norm, y_treino)

    # --- ETAPA DE TESTE (GENERALIZAÇÃO) ---
    predicoes_teste = prever_perceptron_reg(X_teste_norm, W)

    # Cálculo das métricas de regressão
    rmse = np.sqrt(mean_squared_error(y_teste, predicoes_teste))
    r2 = r2_score(y_teste, predicoes_teste)

    metricas_rmse.append(rmse)
    metricas_r2.append(r2)
    print(f"Resultado da Rodada {r+1}: RMSE = {rmse:.4f}, R² = {r2:.4f}\n")

# --- Estatísticas Finais ---
print("\n--- Resultados Finais das Rodadas de Treino/Teste ---")
print(f"RMSE Médio: {np.mean(metricas_rmse):.4f}")
print(f"RMSE Desvio Padrão: {np.std(metricas_rmse):.4f}")
print(f"R² Médio: {np.mean(metricas_r2):.4f}")
print(f"R² Desvio Padrão: {np.std(metricas_r2):.4f}")

# --- Visualizações da última rodada ---

# 1. Curva de Aprendizagem
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(curva_mse)
plt.title("Curva de Aprendizagem (Última Rodada)")
plt.xlabel("Época")
plt.ylabel("Erro Quadrático Médio (MSE)")
plt.grid(True)

# 2. Gráfico de Dispersão: Predito vs. Real
plt.subplot(1, 2, 2)
plt.scatter(y_teste, predicoes_teste, alpha=0.6)
plt.plot([y_teste.min(), y_teste.max()], [y_teste.min(), y_teste.max()], '--r', linewidth=2)
plt.title("Valores Reais vs. Preditos (Última Rodada)")
plt.xlabel("Valores Reais")
plt.ylabel("Valores Preditos")
plt.grid(True)

plt.tight_layout()
plt.show()