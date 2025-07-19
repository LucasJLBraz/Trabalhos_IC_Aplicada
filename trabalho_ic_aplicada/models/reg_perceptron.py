import numpy as np
def treinar_perceptron_reg(X_treino: np.ndarray, y_treino: np.ndarray, epocas: int = 100, taxa_aprendizagem: float = 0.001, verbose: bool = False) -> (np.ndarray, list):
    """
    Treina um modelo de regressão linear usando um perceptron com SGD.

    Args:
        X_treino (np.ndarray): Matriz de características de treino.
        y_treino (np.ndarray): Vetor de rótulos de treino.
        epocas (int): Número de épocas para o treinamento.
        taxa_aprendizagem (float): Taxa de aprendizado para o SGD.
        verbose (bool): Se True, imprime o progresso do treinamento a cada 20 épocas.

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
        if verbose and ((epoca + 1) % 20 == 0):
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
