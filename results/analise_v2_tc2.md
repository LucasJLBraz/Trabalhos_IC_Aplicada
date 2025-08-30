# Análise Crítica (V2) do Relatório - Trabalho 2

**Avaliador:** Gemini
**Data:** 27/08/2025
**Foco:** Rigor conceitual, validação metodológica e consistência para um padrão de pós-graduação.

## 1. Avaliação Sumária

Esta análise aprofunda a revisão anterior, incorporando a crítica fornecida e aplicando um escrutínio mais rigoroso. O trabalho fundamental continua sendo de alta qualidade, com uma implementação própria robusta e um protocolo experimental que, na superfície, é sólido. 

Contudo, uma análise detalhada revela **fragilidades metodológicas e omissões sistemáticas no relatório** que são críticas a nível de mestrado. A principal falha não está na execução do código, mas na **justificativa das escolhas** e na **comunicação incompleta dos resultados**, o que impede a reprodutibilidade e a validação externa baseada apenas no texto. O trabalho é bom, mas carece do rigor analítico e da transparência esperados em uma dissertação ou artigo científico.

> **Nota sobre Evolução Metodológica:** Conforme as discussões desta análise, o critério de seleção de hiperparâmetros nos scripts foi atualizado de 'acurácia' para 'F1-Score (macro)'. Embora a presente análise se baseie nos resultados originais (gerados com acurácia), esta correção metodológica será fundamental para a geração de novos resultados e a subsequente reescrita do relatório final, tornando-o mais robusto e alinhado às melhores práticas.
> UTILZAMOS 100 BUSCAS AO INVES DE 60 para otimizar os modelos.

---

## 2. Inconsistências Críticas: Relatório vs. Código vs. Resultados
20X20 FOI UTILIZADO AO INVEZ DE 40X40. O MOTIVO FOI CUSTO COMPUTACIONAL.

> 

Uma verificação cruzada entre o PDF, os CSVs em `results/` e o código em `src/` revela uma falha sistemática: **as tabelas do relatório omitem hiperparâmetros essenciais**, tornando a replicação a partir do texto impossível.

**Falha Central: Omissão de Hiperparâmetros de Treino**

As Tabelas 1, 2, 3 e 4 do relatório **sistematicamente omitem** a **taxa de aprendizado (`lr`)** e o **número de épocas (`epochs`)** para os modelos baseados em gradiente (PL, MLP-1H, MLP-2H). Elas apresentam apenas o otimizador, a função de ativação e a regularização L2.

-   **Exemplo (Tabela 1 vs. `tabela1.csv`):**
    -   **Relatório (PL):** Afirma que o vencedor usou `min-max [-1,1]`, `Nesterov` e `λ₂=10⁻⁴`.
    -   **CSV (`tabela1.csv`):** Revela que a configuração completa foi `minmax_pm1`, `nesterov`, `lr=0.02`, `epochs=100`, `l2=0.0001`.
-   **Por que é um erro crasso?** A taxa de aprendizado e as épocas são, indiscutivelmente, dois dos hiperparâmetros mais influentes no treino de redes neurais. Omiti-los da tabela de resultados é como publicar a receita de um bolo omitindo a temperatura e o tempo de forno. Isso invalida a tabela como um registro completo da configuração vencedora e denota uma falha grave de comunicação científica.

**Outras Inconsistências Notáveis (herdadas da crítica anterior):**

-   **Erro Conceitual de Nomenclatura:** A Tabela 3 usa `min-max [1,1]`. Não existe tal normalização; o correto seria `min-max [-1,1]` (ou `minmax_pm1` como no código). Isso sinaliza uma falta de atenção a conceitos básicos.
-   **Afirmações Fatuais Incorretas:** O texto afirma que o ganho de tempo com PCA-rotate foi "para todos" (Seção 4.3), mas o PL ficou mais lento. Afirma que o FPR do PL e MQ na Atividade 8 é "idêntica", mas os valores em `tabela4.csv` são `0.0178` e `0.0182`.
-   **Arredondamento Problemático:** O tempo de treino do MQ na Tabela 2 é arredondado para `0.000`, escondendo a ordem de grandeza real (`3.3e-4 s`), o que é enganoso.

---

## 3. Análise Crítica da Metodologia

Indo além das inconsistências, a própria metodologia, embora boa, apresenta pontos de fragilidade conceitual quando analisada sob uma ótica de pós-graduação.

### 3.1. Métrica de Seleção de Hiperparâmetros

-   **Problema:** O código nos scripts `tc2_faces_A1_A4.py`, `A5_A6.py` e `A7.py` utiliza a **acurácia média (`acc`)** como critério para selecionar o melhor conjunto de hiperparâmetros (`score = np.mean([r["acc"] for r in reps])`).
-   **Por que é uma fragilidade?** A acurácia é uma métrica pobre para problemas multi-classe, especialmente se houver qualquer desbalanceamento (mesmo que leve). Um modelo pode obter alta acurácia simplesmente ao acertar as classes majoritárias. Métricas como **F1-Score (macro)** são mais robustas, pois balanceiam precisão e recall e tratam todas as classes com igual importância. O fato de o F1-Score ser calculado e reportado, mas não usado para a seleção do modelo, é uma oportunidade perdida e uma fraqueza metodológica.
-   **Contraponto Positivo:** Na Atividade 8, a métrica de seleção foi corretamente alterada para `f1_intruso`. Isso mostra que o autor entende a importância de escolher a métrica certa para o problema, mas falhou em aplicar esse raciocínio às atividades anteriores.

### 3.2. Justificativa e Análise do Protocolo de Busca

-   **Problema:** O relatório estabelece um orçamento de 60 amostras para o *random search* em espaços que podem conter até ~700.000 combinações (MLP-2H). 
-   **Por que é uma fragilidade?** Embora o *random search* seja eficiente, a amostragem é extremamente esparsa (inferior a 0.01%). O relatório não **contextualiza nem justifica** essa escolha. Um trabalho de mestrado deveria, no mínimo, reconhecer essa limitação, afirmando que o objetivo é "encontrar regiões de alta performance" em vez de "o conjunto ótimo de hiperparâmetros", e talvez citar literatura que suporta a eficácia de buscas esparsas (e.g., Bergstra & Bengio, 2012).

### 3.3. Análise Qualitativa do PCA

-   **Problema:** O PCA é tratado puramente como uma ferramenta matemática para redução de dimensionalidade e decorrelação. Não há nenhuma análise sobre **o que** os componentes principais representam.
-   **Por que é uma fragilidade?** Em reconhecimento de faces, os componentes principais são as famosas **"eigenfaces"**. Visualizá-las oferece uma visão qualitativa poderosa sobre quais características o modelo está priorizando (iluminação, traços faciais, fundo, etc.). Omitir essa análise é perder uma camada inteira de interpretação do modelo e do problema.

### 3.4. Análise Aprofundada da Falha da Transformação Box-Cox

-   **O Paradoxo:** A teoria, conforme apontado por seu colega, sugere que normalizar os dados com Box-Cox (para torná-los mais gaussianos) e depois padronizar com Z-score deveria beneficiar modelos lineares como o Perceptron Logístico (PL) e Mínimos Quadrados (MQ), que muitas vezes assumem ou se beneficiam de normalidade nos dados. Contudo, os resultados da Atividade 7 (Tabela 4 do relatório) mostram o oposto: uma queda de desempenho acentuada para todos os modelos em comparação com a Atividade 6 (Tabela 3).

-   **A Explicação (O Contexto Importa):** A afirmação do seu colega está correta em um vácuo, mas falha ao não considerar o estado dos dados **após a aplicação do PCA**. Os dados que alimentam a transformação Box-Cox não são os pixels brutos, mas sim os **10 componentes principais**.
    1.  **Propriedade do PCA:** Os componentes principais são combinações lineares das variáveis originais (pixels). Pelo Teorema do Limite Central, combinações lineares de muitas variáveis aleatórias tendem a ter uma distribuição próxima da Gaussiana (normal).
    2.  **A Hipótese Central:** A transformação Box-Cox foi prejudicial porque os dados de entrada (os 10 componentes principais) **já eram aproximadamente normais**. Aplicar uma segunda e poderosa transformação de normalização em dados que já são bem comportados não só é desnecessário, como pode ser **nocivo**. A transformação Box-Cox, por ser não-linear, pode ter introduzido distorções (assimetrias) em distribuições que já eram simétricas, prejudicando a separabilidade linear das classes que modelos como PL e MQ exploram.

-   **Conclusão da Análise:** A falha da Box-Cox neste experimento não invalida a teoria geral. Em vez disso, fornece uma visão mais profunda: a eficácia de uma técnica de pré-processamento depende criticamente do estado dos dados naquele ponto do pipeline. Neste caso, o PCA já havia "condicionado" os dados de tal forma que a Box-Cox se tornou contraproducente. Esta é uma conclusão de alto nível que deveria constar no relatório.

### 3.5. Visualizações Essenciais para Análise de Erro

-   **Problema:** O relatório se baseia quase que exclusivamente em métricas de desempenho agregadas (acurácia, F1-score). Essas métricas, embora úteis, escondem a natureza dos erros. Para um trabalho de mestrado, é crucial analisar *quais* erros o modelo comete.
-   **Por que é uma fragilidade?** Sem uma análise de erro, não é possível responder perguntas como: "O modelo confunde mais o sujeito 5 com o 7? Por quê? Eles são parecidos?". Para a Atividade 8, uma única métrica de FNR/FPR não mostra o comportamento do classificador em diferentes limiares de decisão, o que é fundamental para aplicações de segurança.
-   **Visualizações Necessárias:**
    1.  **Matriz de Confusão:** Para as atividades de classificação multi-classe (A2, A4, A6, A7), uma matriz de confusão para o melhor modelo de cada cenário é essencial. Ela mostra exatamente os pares de sujeitos que são mais confundidos.
    2.  **Curva ROC (Receiver Operating Characteristic):** Para a Atividade 8 (detecção de intruso), a Curva ROC é a ferramenta padrão da indústria e da academia. Ela visualiza o trade-off entre a taxa de verdadeiros positivos (detectar um intruso) e a taxa de falsos positivos (acusar um usuário legítimo de ser um intruso) para todos os limiares de decisão possíveis.

---

## 4. Sugestões para Elevar o Patamar do Trabalho

As seguintes ações, se implementadas, elevariam o trabalho de um ótimo projeto de disciplina para um nível próximo ao de um artigo científico.

### 4.1. Reforçar a Metodologia (Correções no Código)

**Ação 1: Alterar a Métrica de Seleção de Hiperparâmetros.**

-   **O quê:** Substitua a acurácia pelo F1-Score (macro) como o critério de seleção nas Atividades 2, 4, 6 e 7.
-   **Onde:** Nos scripts `src/tc2_faces_A1_A4.py`, `src/tc2_faces_A5_A6.py` e `src/tc2_faces_A7.py`, localize a função `select_best_by_random_search`.
-   **Como:**
    
    ```python
    # Linha a ser alterada em todos os scripts relevantes
    # DE:
    score = np.mean([r["acc"] for r in reps])
    
    # PARA:
    score = np.mean([r["f1_macro"] for r in reps])
    ```
-   **Impacto:** Esta mudança requer a re-execução dos experimentos, mas tornaria a seleção de modelos metodologicamente mais sólida e defensável.

**Ação 2: Adicionar o Método `predict_proba` aos Classificadores.**

-   **O quê:** Implementar um método que retorne as probabilidades da classe, não apenas o `argmax`.
-   **Por quê:** Essencial para a análise da Curva ROC (sugestão 4.3) e para qualquer análise de limiar.
-   **Onde:** Nos arquivos de modelo em `trabalho_ic_aplicada/models/`.
-   **Como:**

    ```python
    # Em trabalho_ic_aplicada/models/clf_pl.py (e similar para MLP)
    # Adicionar este método à classe SoftmaxRegression
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xb = self._add_bias(X)
        return _softmax(Xb @ self.W_)
    
    # Em trabalho_ic_aplicada/models/clf_mqo.py
    # Adicionar este método à classe LeastSquaresClassifier
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xb = self._add_bias(X)
        scores = Xb @ self.W_
        # Normaliza scores para pseudo-probabilidades (não é um softmax real, mas funciona para ROC)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    ```

### 4.2. Análise Qualitativa das Eigenfaces (Novo Notebook)

-   **O quê:** Crie um notebook para visualizar as eigenfaces.
-   **Onde:** `notebooks/TC2/analise_qualitativa_pca.ipynb`.
-   **Como:**

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from trabalho_ic_aplicada.dataset_faces import build_face_dataset
    from trabalho_ic_aplicada.models.pca_np import PCA_np

    # 1. Carregar dados e ajustar PCA
    img_h, img_w = (40, 40)
    X, y, _ = build_face_dataset("./data/raw/Kit_projeto_FACES", size=(img_h, img_w))
    pca = PCA_np()
    pca.fit(X)

    # 2. Plotar as primeiras eigenfaces
    def plot_eigenfaces(pca_model, h, w, n_eigenfaces=10):
        fig, axes = plt.subplots(2, 5, figsize=(12, 5),
                                 subplot_kw={'xticks':[], 'yticks':[]},
                                 gridspec_kw=dict(hspace=0.1, wspace=0.1))
        for i, ax in enumerate(axes.flat):
            if i < n_eigenfaces:
                eigenface = pca_model.Vt_[i, :].reshape(h, w)
                ax.imshow(eigenface, cmap='bone')
                ax.set_title(f"PC {i+1}")
        plt.suptitle("Visualização das Primeiras 10 Eigenfaces")
        plt.savefig("results/eigenfaces_visualization.png")
        plt.show()

    plot_eigenfaces(pca, img_h, img_w)
    ```
-   **Impacto:** Adiciona uma seção inteira de análise qualitativa ao relatório, discutindo se os componentes capturam iluminação, identidade ou outras variações.

### 4.3. Análise da Curva ROC para o Problema do Intruso (Novo Notebook)

-   **O quê:** Analise o trade-off FNR/FPR de forma profissional.
-   **Onde:** `notebooks/TC2/analise_roc_intruso.ipynb`.
-   **Como:**

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc # Uso aceitável para análise post-hoc
    # ... (importar seus modelos e dados como no script A8)

    # Assumindo que você tem: 
    # - um modelo treinado (ex: `model`)
    # - X_test, y_test
    # - intruder_id

    # 1. Obter probabilidades (requer Ação 4.1.2)
    y_probas = model.predict_proba(X_test)
    # Probabilidade da classe ser "intruso"
    proba_intruso = y_probas[:, intruder_id]

    # 2. Binarizar o y_test
    y_true_bin = (y_test == intruder_id).astype(int)

    # 3. Calcular a curva ROC
    fpr, tpr, thresholds = roc_curve(y_true_bin, proba_intruso)
    roc_auc = auc(fpr, tpr)

    # 4. Plotar
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR) / Sensibilidade')
    plt.title(f'Curva ROC para Detecção de Intruso - Modelo {model.__class__.__name__}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("results/roc_curve_intruder.png")
    plt.show()
    ```
-   **Impacto:** Substitui a análise de um único ponto (FNR/FPR da Tabela 6) por uma análise completa do desempenho do classificador em todos os limiares. É o padrão-ouro para este tipo de problema.

---

## 5. Conclusão e Avaliação Final (Revisada)

O trabalho demonstra uma base de implementação muito forte, mas revela lacunas no rigor da comunicação científica e na profundidade da análise metodológica. As omissões sistemáticas nas tabelas de resultados são a falha mais grave, pois comprometem a credibilidade do relatório como um documento autocontido. As escolhas metodológicas, embora boas, poderiam ser mais bem justificadas e alinhadas com as melhores práticas em cenários específicos (e.g., F1-score para seleção).

**Nota Final: 8.0 / 10**

**Justificativa:** A nota reflete um trabalho com excelente potencial e uma implementação robusta (valendo a maior parte da nota), mas que é penalizado por falhas significativas na apresentação e justificativa dos resultados. Para um nível de mestrado, não basta apenas executar os experimentos; é preciso comunicá-los com total transparência e defender cada escolha metodológica. As sugestões acima, se implementadas, abordam diretamente essas lacunas e elevariam o trabalho a um patamar de excelência (nota > 9.0).
