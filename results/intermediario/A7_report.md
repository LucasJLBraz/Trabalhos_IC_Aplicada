# Relatório Metodológico Detalhado: Atividade 7

**Autor:** Gemini
**Data:** 28/08/2025
**Foco:** Análise aprofundada da metodologia de transformações não-lineares (Box-Cox) aplicadas após a redução de dimensionalidade, com base no script `src/tc2_faces_A7.py`.

## 1. Visão Geral e Objetivo

A Atividade 7 representa a etapa mais refinada do pipeline de pré-processamento. Após termos estabelecido um pipeline que projeta os dados em um subespaço de baixa dimensão e alta informação (`PCA` com `q=79`), esta atividade investiga se podemos melhorar ainda mais o desempenho dos classificadores aplicando uma **transformação não-linear** sobre esses componentes.

O objetivo é testar a hipótese de que, ao tornar a distribuição de cada componente principal mais próxima de uma distribuição Gaussiana (normal), podemos facilitar a tarefa dos classificadores, especialmente os lineares.

O script `tc2_faces_A7.py` implementa e avalia o seguinte pipeline: `PCA (redução) -> Box-Cox -> Z-Score`.

## 2. Metodologia Detalhada

O protocolo experimental é idêntico ao da Atividade 6, mas com a adição da transformação Box-Cox no pipeline de pré-processamento.

### 2.1. Pipeline de Pré-processamento da Atividade 7

Para cada uma das 50 repetições, o fluxo de transformações é o seguinte:

1.  **Divisão Treino/Teste:** Os dados são divididos em 80% para treino e 20% para teste de forma estratificada.
2.  **PCA com Redução:** Um modelo PCA é ajustado (`fit`) nos dados de **treino** e usado para transformar (`transform`) ambos os conjuntos (treino e teste) para o subespaço de `q=79` dimensões.
3.  **Transformação Box-Cox:**
    -   **Ajuste:** A transformação Box-Cox é **ajustada exclusivamente sobre os componentes principais do conjunto de treino**. Para cada um dos `q` componentes, o script encontra o parâmetro `lambda` ótimo que maximiza a verossimilhança logarítmica (`log-likelihood`), aproximando a distribuição daquele componente de uma normal.
    -   **Truque de Implementação (Positividade):** A transformação Box-Cox requer que todos os dados de entrada sejam estritamente positivos. Como os componentes principais são centrados em zero, o script adiciona um `shift` a cada componente (baseado no seu valor mínimo no conjunto de treino) para garantir a positividade antes de aplicar a transformação.
    -   **Aplicação:** Os parâmetros (`lambda` e `shift`) aprendidos no conjunto de treino são então usados para transformar os componentes principais de ambos os conjuntos, treino e teste.
4.  **Normalização Z-Score Final:**
    -   Após a transformação Box-Cox, uma normalização Z-Score final é aplicada. Os parâmetros (média e desvio padrão) são ajustados nos dados de treino já transformados pela Box-Cox e aplicados a ambos os conjuntos.
    -   **Justificativa:** Esta etapa final garante que, após a transformação não-linear, cada característica tenha média 0 e desvio padrão 1, que são condições ideais para os algoritmos de otimização baseados em gradiente.

### 2.2. Busca e Avaliação

-   Após o pré-processamento completo, a busca de hiperparâmetros (`select_best_by_random_search`) e a avaliação final (`eval_best_over_repeats`) procedem como antes.
-   Uma diferença notável é que, como o pipeline agora inclui uma normalização Z-Score fixa no final, a busca de hiperparâmetros **não inclui mais a normalização como uma variável**, focando apenas nos parâmetros intrínsecos de cada modelo (taxa de aprendizado, regularização, etc.).

## 3. Hipóteses e Resultados Esperados

### 3.1. A Hipótese Central

A principal hipótese por trás da Atividade 7 é que **modelos de classificação, especialmente os lineares (MQ e PL), funcionam melhor quando as características de entrada seguem uma distribuição Gaussiana**. As fronteiras de decisão lineares são ótimas para separar classes que são Gaussianas com a mesma matriz de covariância. Ao usar Box-Cox para "normalizar" a distribuição de cada componente principal, estamos tentando aproximar os dados dessas condições ideais.

### 3.2. A Análise Crítica da Hipótese (O "Truque" Conceitual)

Embora a hipótese acima seja válida em geral, há um detalhe conceitual crucial que o experimento irá revelar:

-   **O Teorema do Limite Central em Ação:** Os componentes principais são, por definição, combinações lineares das variáveis originais (os pixels). O Teorema do Limite Central sugere que a soma de um grande número de variáveis aleatórias (mesmo que não-Gaussianas) tende a seguir uma distribuição Gaussiana.
-   **Consequência:** Portanto, é muito provável que os primeiros e mais importantes componentes principais **já sejam naturalmente próximos de uma distribuição Gaussiana**. 

### 3.3. Resultados Esperados

Com base na análise crítica acima, os resultados esperados são:

1.  **Desempenho Prejudicado ou Inalterado:** Espera-se que a transformação Box-Cox **não traga benefícios significativos e possa até piorar o desempenho** de todos os classificadores. 
    -   **Justificativa:** Se os componentes já são aproximadamente normais, aplicar uma segunda transformação de normalização potente e não-linear como a Box-Cox pode ser desnecessário e até prejudicial. Ela pode distorcer a geometria dos dados que já era favorável, potencialmente tornando as classes menos separáveis.

2.  **Aumento do Custo Computacional:** O processo de encontrar o `lambda` ótimo para cada componente em cada repetição adiciona um custo computacional extra ao pipeline de pré-processamento, o que pode ser observado em um leve aumento no tempo total de execução.

Em suma, a Atividade 7 é um excelente exercício de validação de hipóteses. Ela testa uma técnica de pré-processamento padrão, mas em um contexto (após o PCA) onde ela pode não ser apropriada. O resultado esperado é a confirmação de que a aplicação de técnicas de pré-processamento deve ser criteriosa e baseada na análise das propriedades dos dados em cada etapa do pipeline, e não apenas na aplicação cega de "boas práticas".
