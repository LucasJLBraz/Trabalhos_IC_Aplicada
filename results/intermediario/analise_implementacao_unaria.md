# Relatório de Implementação: Detecção de Intrusos com Modelos Unários (Detecção de Anomalias)

**Data:** 28/08/2025
**Autor:** Gemini

## 1. Objetivo e Justificativa

Esta atividade implementa uma abordagem de **detecção de anomalias (classificação unária)** para o problema de controle de acesso. A premissa é que um sistema de segurança robusto deve ser treinado apenas com exemplos de usuários autorizados para aprender um modelo de "normalidade". Qualquer desvio significativo desse padrão é considerado uma anomalia (intruso). Esta metodologia é conceitualmente superior à classificação binária para este problema, pois não requer conhecimento prévio sobre os intrusos, aumentando a capacidade de generalização.

## 2. Metodologia Robusta (Sem Vazamento de Dados)

Para garantir que a avaliação de desempenho seja imparcial e livre de vazamento de dados, foi implementado um protocolo com uma separação estrita entre os conjuntos de dados de **seleção de hiperparâmetros** e de **teste final**.

O fluxo de trabalho geral é:

1.  **Divisão de Dados Global:**
    *   O conjunto de **usuários autorizados** é dividido em Treino (70%), Validação (15%) e Teste (15%).
    *   O conjunto de **intrusos** é dividido em Validação (50%) e Teste (50%).
    *   O conjunto de **Teste** (autorizados + intrusos) é completamente isolado e só será usado na avaliação final.

2.  **Fase 1: Busca de Hiperparâmetros (usando o conjunto de Validação)**
    *   Para cada modelo, uma busca aleatória testa 30 conjuntos de hiperparâmetros.
    *   Para cada conjunto, o modelo é treinado nos dados de **Treino de autorizados**.
    *   O desempenho é medido no conjunto de **Validação**. Um limiar de decisão é otimizado nos scores de validação para maximizar o F1-Score. Este F1-Score é a métrica usada para julgar os hiperparâmetros.
    *   O melhor conjunto de hiperparâmetros é selecionado.

3.  **Fase 2: Avaliação Final (usando o conjunto de Teste)**
    *   Com os melhores hiperparâmetros definidos, o modelo final é avaliado em 50 repetições independentes (com novas divisões de dados em cada uma).
    *   Em cada repetição, o modelo é treinado nos dados de **Treino de autorizados**.
    *   O limiar de decisão é definido de forma agnóstica aos intrusos (ver Truque 4).
    *   As métricas finais (FNR, FPR, etc.) são calculadas no conjunto de **Teste** e armazenadas.

### Pipeline de Pré-processamento

O pipeline `PCA -> Box-Cox -> Z-Score` é mantido. Para o PCA, optou-se por reter **99% da variância explicada**. Para um modelo de reconstrução como o Autoencoder, preservar mais variância fornece um sinal de entrada mais rico, o que pode levar a reconstruções mais detalhadas e, consequentemente, a uma melhor distinção entre o baixo erro de reconstrução de dados normais e o alto erro de dados anômalos.

## 3. Implementação e Modelos

### 3.1. Modelo Principal: Autoencoder com MLP

O Autoencoder é uma rede neural de regressão treinada para reconstruir sua própria entrada. A implementação usa a classe `MLPRegressor`.

-   **Arquitetura:** A rede possui camadas de codificação que comprimem a entrada até um "gargalo" (bottleneck) e camadas de decodificação que a reconstroem. A dimensão do gargalo é um hiperparâmetro a ser otimizado.
-   **Treinamento:** O modelo minimiza o Erro Quadrático Médio (MSE) entre as imagens de entrada e as imagens reconstruídas, usando apenas os dados dos usuários autorizados.
-   **Score de Anomalia:** O erro de reconstrução (MSE). Faces autorizadas devem ter baixo erro; faces de intrusos, alto erro.

### 3.2. Modelos de Baseline (Scikit-learn)

1.  **One-Class SVM:** Aprende uma fronteira de alta dimensão em torno dos dados normais.
2.  **Isolation Forest:** Constrói árvores aleatórias; anomalias, por serem diferentes, são tipicamente isoladas com menos partições.

### 3.3. Espaço de Busca para Otimização

O espaço de busca foi focado nos hiperparâmetros de maior impacto para cada modelo, a fim de manter a viabilidade computacional.

-   **Autoencoder MLP:**
    -   `hidden_middle_factor`: Fator para o tamanho do gargalo (`q_dim * factor`), de `[0.25, 0.5, 0.75]`.
    -   `lr`: Taxa de aprendizado, de `[0.005, 0.01]`.
    -   `l2`: Regularização, de `[1e-5, 1e-4]`.
-   **One-Class SVM:**
    -   `nu`: Fração esperada de anomalias, de `[0.01, 0.05, 0.1, 0.15]`.
    -   `gamma`: Coeficiente do kernel, de `[0.01, 0.1, 1, 10]`.
-   **Isolation Forest:**
    -   `n_estimators`: Número de árvores, de `[50, 100, 200]`.

## 4. Truques de Implementação e Detalhes Técnicos

1.  **Justificativa de Hiperparâmetros do Autoencoder:**
    *   **Otimizador `Nesterov`:** Em resposta ao feedback, o otimizador `ADAM` foi substituído. O Gradiente Acelerado de Nesterov (`nesterov`) foi escolhido por ser um otimizador baseado em momento que frequentemente converge mais rápido e de forma mais estável que o SGD padrão, servindo como uma alternativa robusta e clássica ao ADAM.
    *   **Ativação `tanh`:** A função de ativação `tanh` foi usada nas camadas ocultas porque sua saída é centrada em zero (-1 a 1). Isso ajuda a manter os gradientes normalizados durante o backpropagation, o que pode acelerar a convergência.
    *   **`clip_grad=5.0`:** O recorte de gradiente (`gradient clipping`) é uma técnica defensiva para prevenir o problema de "explosão de gradientes". O valor 5.0 foi escolhido como um limite razoável para permitir atualizações de peso significativas, mas evitar mudanças extremas que poderiam desestabilizar o treinamento.

2.  **API de Score Unificada:** Para simplificar a lógica, todos os modelos foram encapsulados para que um **score de anomalia maior sempre indique maior probabilidade de ser um intruso**. Isso envolveu inverter o sinal do `decision_function` dos modelos `sklearn`, que por padrão retornam valores menores para anomalias.

3.  **Divisão Estratificada Robusta:** A metodologia agora emprega uma divisão de dados em três conjuntos (treino, validação, teste) para garantir que a seleção de hiperparâmetros seja feita de forma independente da avaliação final, eliminando qualquer vazamento de dados.

4.  **Definição de Limiar Agnóstica a Intrusos:** Na avaliação final (Fase 2), o limiar de decisão é definido como o **percentil 95 dos scores de anomalia calculados sobre os dados de treino autorizados**. Isso significa que o limiar é escolhido para aceitar 95% dos dados "normais" de treino. É uma abordagem robusta que não usa nenhuma informação (nem mesmo de validação) sobre os intrusos para definir o ponto de corte final, tornando o teste mais realista.
