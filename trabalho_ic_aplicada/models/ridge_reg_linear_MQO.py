import numpy as np
def treinar_reglin_l2(X_treino: np.ndarray, y_treino: np.array, lambda_reg: float) -> np.ndarray:
    """
    Treina um modelo de regressão linear múltipla com regularização L2.

    Args:
        X_treino (np.array): Matriz (n_amostras, n_features) com as
                                      variáveis de entrada do treino.
        y_treino (np.array): Vetor (n_amostras,) com a variável de saída do treino.
        lambda_reg (float): O hiperparâmetro de regularização (lambda).

    Returns:
        np.array: O vetor de coeficientes do modelo treinado (beta_hat),
                  incluindo o intercepto.
    """

    # Validação inicial das dimensões para garantir a consistência dos dados.
    n_amostras, n_features = X_treino.shape
    if n_amostras != len(y_treino):
        raise ValueError("O número de amostras em X e y deve ser o mesmo.")

    # Construção da Matriz de Desenho X: adiciona a coluna de '1's para o intercepto.
    X = np.c_[np.ones(n_amostras), X_treino]

    # Dimensão total dos parâmetros (features + intercepto)
    k = n_features + 1

    # Construção dos termos para regularização de Tikhonov: (X.T @ X + λI) @ β = X.T @ y
    A = X.T @ X
    I = np.identity(k)
    A_reg = A + lambda_reg * I

    b = X.T @ y_treino

    # Solução do sistema linear A_reg @ β = b para β.
    # Usar np.linalg.solve.
    beta_hat = np.linalg.solve(A_reg, b)

    return beta_hat

def prever(modelo_beta, X_novos_dados_features):
    """
    Realiza predições usando um modelo de regressão linear treinado.

    Args:
        modelo_beta (np.array): O vetor de coeficientes (beta_hat) retornado
                                pela função de treinamento.
        X_novos_dados_features (np.array): Matriz com os novos dados de entrada
                                           para os quais se deseja a predição.

    Returns:
        np.array: Um vetor com os valores preditos.
    """
    # Adiciona a coluna de '1's para o intercepto, garantindo a compatibilidade dimensional.
    X_novos_dados = np.c_[np.ones(X_novos_dados_features.shape[0]), X_novos_dados_features]

    # Realiza a predição através do produto escalar: y_pred = X @ beta_hat
    y_predito = X_novos_dados @ modelo_beta

    return y_predito