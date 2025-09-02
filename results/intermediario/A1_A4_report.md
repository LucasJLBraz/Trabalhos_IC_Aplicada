# Relatório Metodológico Detalhado: Atividades 1-4

**Autor:** Gemini
**Data:** 28/08/2025
**Foco:** Análise aprofundada da metodologia, decisões de implementação e justificativas técnicas para as Atividades 1-4 do TC2, com base no script `src/tc2_faces_A1_A4.py`.

## 1. Visão Geral e Objetivo

As Atividades 1 a 4 constituem a base do projeto, focando na classificação de faces em duas condições principais:

1.  **Sem pré-processamento com PCA (Atividades 1-2):** Avalia o desempenho dos classificadores diretamente sobre os pixels das imagens, testando o impacto da dimensionalidade (escala da imagem) e da normalização.
2.  **Com PCA como Rotação (Atividades 3-4):** Avalia o impacto de decorrelacionar os dados de entrada, aplicando uma transformação PCA que mantém a dimensionalidade original (`q=d`).

O objetivo é entender o comportamento de quatro classificadores (MQ, PL, MLP-1H, MLP-2H) e estabelecer um baseline de desempenho e custo computacional.

## 2. Protocolo Experimental Robusto

Para garantir a validade estatística e a reprodutibilidade dos resultados, um protocolo rigoroso foi implementado no script `tc2_faces_A1_A4.py`.

1.  **Divisão Estratificada (Holdout):** Em cada uma das 50 repetições finais, os dados são divididos em 80% para treino e 20% para teste. A função `train_test_split_stratified` garante que a proporção de imagens por sujeito seja a mesma em ambos os conjuntos, o que é crucial para datasets com poucas amostras por classe como o Yale A.

2.  **Busca de Hiperparâmetros (Random Search):** Para cada modelo (exceto o MQ, que não tem hiperparâmetros de treino), uma busca aleatória (`random search`) com 200 amostras (`N_SAMPLES_RS`) é executada para encontrar a melhor combinação de hiperparâmetros. A normalização dos dados também é tratada como um hiperparâmetro dentro desta busca.

3.  **Validação Interna na Busca (k-fold implícito):** Para estabilizar a seleção, cada conjunto de hiperparâmetros amostrado é avaliado 10 vezes (`K_SELECT_EVAL`) com diferentes sementes. A média de desempenho (acurácia) nessas 10 execuções é usada como o score daquele candidato. Isso evita que uma única divisão de dados "sortuda" ou "azarada" influencie a escolha do melhor modelo.

4.  **Avaliação Final:** O conjunto de hiperparâmetros vencedor da busca é então submetido a 50 repetições independentes (`N_REPEATS_BEST`) para gerar as estatísticas finais (média, desvio padrão, etc.), que são salvas nas tabelas de resultados.

## 3. Pré-processamento e Análise de Escala (Atividade 1)

-   **Leitura e Vetorização:** As imagens do dataset Yale A são lidas e convertidas para tons de cinza. Em seguida, são redimensionadas para três escalas diferentes: 20x20, 30x30 e 40x40. Cada imagem 2D é então "achatada" (vetorizada) para formar um vetor de características unidimensional (de dimensão 400, 900 e 1600, respectivamente).
-   **Análise de Custo-Benefício:** A Atividade 1 tem como objetivo prático medir o tempo de treinamento em cada escala. O gráfico `tempo_escala_A1_A2.png` gerado pelo script visualiza o trade-off entre a dimensionalidade dos dados e o custo computacional. A escolha da escala de 30x30 para as atividades subsequentes no script é uma decisão pragmática que equilibra a retenção de informação facial com a viabilidade de executar centenas de experimentos de forma rápida.

## 4. Modelos de Classificação e Detalhes de Implementação

As implementações residem em `trabalho_ic_aplicada/models/` e foram projetadas para serem modulares.

### 4.1. Classificador de Mínimos Quadrados (MQ)
-   **Arquivo:** `clf_mqo.py`
-   **Metodologia:** É um modelo linear que encontra uma solução analítica (forma fechada) para o problema de classificação, tratando-o como uma regressão em alvos one-hot-encoded. A predição é feita calculando `X @ W` e tomando a classe com o maior score.
-   **Truque de Implementação:** Para evitar problemas com matrizes singulares, a solução utiliza a pseudo-inversa de Moore-Penrose (`np.linalg.pinv`) quando não há regularização L2. Com L2, ele resolve o sistema linear `(X.T @ X + λI) @ W = X.T @ Y`, que é numericamente mais estável.

-   **Metodologia:** Solução analítica via pseudo-inversa ou resolvendo o sistema linear para a forma com regularização L2.
-   **Espaço de Busca:**
    -   `l2` (Regularização L2): `{0.0, 1e-4, 1e-3, 1e-2, 1e-1}`

### 4.2. Perceptron Logístico (PL) / Regressão Softmax
-   **Arquivo:** `clf_pl.py`
-   **Metodologia:** É um classificador linear treinado com Gradiente Descendente. A camada de saída utiliza a função **softmax** para gerar probabilidades de pertencimento a cada classe, e a função de custo é a **entropia cruzada (cross-entropy)**, que é o padrão para problemas de classificação multi-classe.
-   **Otimizadores:** A implementação vai além do gradiente descendente simples (SGD), permitindo o uso de otimizadores mais avançados como `Nesterov` e `Adam` (ver `optim.py`), que aceleram a convergência.
-   **Metodologia:** Classificador linear treinado com gradiente descendente e função de custo de entropia cruzada.
-   **Espaço de Busca:**
    -   `lr` (Taxa de Aprendizado): `{0.005, 0.01, 0.02}`
    -   `epochs` (Épocas de Treino): `{100, 200, 300}`
    -   `l2` (Regularização L2): `{0.0, 1e-4, 1e-3}`
    -   `opt` (Otimizador): `{sgd, momentum, nesterov, rmsprop, adam}`

### 4.3. Perceptron de Múltiplas Camadas (MLP-1H e MLP-2H)
-   **Arquivo:** `clf_mlp.py`
-   **Metodologia:** São redes neurais com uma ou duas camadas ocultas, respectivamente. Assim como o PL, usam uma saída softmax e custo de entropia cruzada. A presença de camadas ocultas com funções de ativação não-lineares (como `tanh`, `ReLU`, `swish`, etc.) permite que o modelo aprenda fronteiras de decisão complexas e não-lineares.

-   **Truque 1: Justificativa do Espaço de Busca:** O espaço de busca de hiperparâmetros (tamanho das camadas, taxa de aprendizado, etc.) foi definido empiricamente para ser amplo, mas razoável. 
    -   **Tamanho das Camadas (`hidden`):** Varia de 4 a 512. Camadas pequenas correm o risco de *underfitting* (não capturar a complexidade dos dados), enquanto camadas muito grandes, para um dataset pequeno como o Yale A (165 imagens), aumentam drasticamente o risco de *overfitting* (memorizar os dados de treino) e o custo computacional. O intervalo escolhido explora esse balanço.
    -   **Taxa de Aprendizado (`lr`):** Valores entre `0.005` e `0.02` são um bom ponto de partida para otimizadores como Adam/Nesterov. Taxas muito altas podem fazer o treino divergir; taxas muito baixas podem torná-lo excessivamente lento ou ficar preso em mínimos locais ruins.
    -   **Metodologia:** Redes neurais com 1 ou 2 camadas ocultas, saída softmax e custo de entropia cruzada.
-   **Espaço de Busca (MLP-1H):**
    -   `hidden` (Neurônios na camada oculta): `{(4,), (8,), (16,), (32,), (64,), (128,), (256,), (512,)}`
    -   `activation` (Função de Ativação): `{tanh, sigmoid, relu, leaky_relu, relu6, swish}`
    -   `lr`, `epochs`, `l2`, `opt`: Idênticos ao PL, com `epochs` em `{150, 200, 300}`.
    -   `clip_grad` (Clipping de Gradiente): `{0.0 (desativado), 2.0, 5.0, 10.0}`
-   **Espaço de Busca (MLP-2H):**
    -   `hidden` (Neurônios nas camadas): 64 combinações de `(h1, h2)` onde `h1` e `h2` são amostrados de `{4, 8, ..., 512}`.
    -   Demais hiperparâmetros idênticos ao MLP-1H.

-   **Truque 2: Justificativa do Clipping de Gradiente (`clip_grad`):**
    -   **O que é?** É uma técnica defensiva que impõe um teto ao valor absoluto dos gradientes durante o backpropagation. Se um gradiente excede esse teto (ex: 5.0), ele é reduzido para o valor do teto.
    -   **Por que usar?** Durante o treino de redes neurais, especialmente com certas combinações de arquitetura, taxa de aprendizado e inicialização de pesos, os gradientes podem "explodir", assumindo valores muito grandes. Isso leva a atualizações de peso enormes que desestabilizam completamente o treinamento, muitas vezes resultando em perdas `NaN` (Not a Number). O clipping garante a estabilidade do treino, permitindo que ele prossiga mesmo sob condições mais agressivas, sendo uma prática essencial para a robustez do processo experimental.

## 5. PCA como Rotação (Atividades 3-4)

-   **Metodologia:** Nesta etapa, o PCA é aplicado, mas o número de componentes principais (`q`) é definido como a dimensionalidade original dos dados (`d`). O resultado não é uma redução de dimensionalidade, mas uma **rotação** do sistema de coordenadas. Os novos eixos (os componentes principais) são ortogonais entre si, o que significa que as características resultantes são **linearmente decorrelacionadas**.
-   **Impacto Esperado e Justificativa:**
    1.  **Melhora de Condicionamento:** A decorrelação dos dados melhora o número de condicionamento da matriz de covariância. Para modelos que dependem de inversão de matrizes (como o MQ) ou da forma da superfície de erro (como os modelos baseados em gradiente), isso pode levar a uma convergência mais rápida e estável.
    2.  **Impacto nos Modelos:** Espera-se que as MLPs se beneficiem mais, pois a decorrelação pode simplificar a superfície de erro que elas precisam navegar. Para modelos lineares como o PL, o efeito pode ser ambíguo: a rotação pode, em alguns casos, tornar a fronteira de decisão linearmente separável mais difícil de encontrar, mesmo que a otimização seja mais estável.
