# Análise Crítica (V3) do Relatório e Metodologia - TC2

**Avaliador:** Gemini
**Data:** 28/08/2025
**Foco:** Análise de rigor científico, consistência entre artefatos (relatório, código, resultados) e profundidade analítica para um padrão de pós-graduação.

## 1. Veredito Sumário

O trabalho apresenta uma base de implementação em Python louvável, demonstrando esforço e conhecimento técnico na construção dos modelos a partir de rotinas de baixo nível. Contudo, essa qualidade na implementação é ofuscada por **falhas sistêmicas e graves na comunicação dos resultados, na consistência entre os artefatos do projeto e no rigor metodológico**.

O relatório, em sua forma atual, **não se sustenta como um documento científico autônomo**. Ele contém informações factualmente incorretas, omite hiperparâmetros críticos que inviabilizam a reprodutibilidade, e apresenta análises superficiais que não exploram a profundidade dos resultados obtidos. A discrepância entre a escala de imagem reportada no PDF (40x40) e a utilizada nos scripts para gerar os resultados principais (30x30) é um erro crasso que compromete a validade de grande parte das conclusões apresentadas.

Esta análise detalha essas falhas e oferece um caminho claro para a correção, visando transformar o projeto de um bom exercício de programação em um trabalho de pesquisa com mérito acadêmico.

---

## 2. Falhas Críticas de Consistência e Reprodutibilidade

A análise cruzada do relatório (`.pdf`), dos resultados (`.csv`) e do código-fonte (`.py`) revela uma desconexão fundamental entre o que foi reportado e o que foi executado.

### 2.1. Discrepância na Escala das Imagens: O Erro Fundamental

-   **Onde:** O relatório (Seção 3.1, Figura 1) afirma que a escala de 40x40 foi escolhida como padrão para as atividades subsequentes.
-   **Problema:** Os scripts que geram as tabelas de resultados (`src/tc2_faces_A1_A4.py`, `src/tc2_faces_A5_A6.py`, `src/tc2_faces_A7.py`, `src/tc2_faces_A8.py`) **utilizam a escala 30x30**. A variável `scale_label, X_sel, y_sel` é fixada no índice `[1]` ou `-1` (dependendo do script) da lista de datasets, que corresponde a `(30,30)`.
-   **Impacto:** **Crasso e sistêmico**. Todas as tabelas de resultados (Tabela 1 a 4) e as conclusões sobre desempenho, tempo de execução e dimensionalidade (`d=900`, não 1600) estão baseadas em uma premissa diferente daquela declarada no relatório. Isso invalida a narrativa do documento. O arquivo `pca_q_98.txt` também foi gerado para a escala 30x30, não 40x40.

### 2.2. Omissão Sistemática de Hiperparâmetros Críticos

-   **Onde:** Tabelas 1, 2, 3 e 4 do relatório.
-   **Problema:** As tabelas **omitem sistematicamente os hiperparâmetros mais importantes** para modelos de redes neurais: **taxa de aprendizado (`lr`) e número de épocas (`epochs`)**. Também omitem o parâmetro de `clip_grad` quando aplicável.
-   **Impacto:** Impossibilita a reprodutibilidade a partir do relatório. É uma falha grave de comunicação científica. Por exemplo, na Tabela 2, o MLP-1H vencedor usou `lr=0.005` e `epochs=300` (informação presente apenas no CSV), mas o relatório omite isso. Sem essa informação, os resultados são irreplicáveis e, portanto, cientificamente inválidos.

### 2.3. Inconsistências Factuais e de Nomenclatura

-   **Erro Conceitual (`min-max [1,1]`):** A Tabela 3 do PDF menciona a normalização `min-max [1,1]`. Esta normalização não existe. O correto é `min-max [-1,1]`, que corresponde a `minmax_pm1` no código. Isso denota falta de atenção a conceitos básicos.
-   **Afirmações Contraditórias:**
    -   O texto afirma que a PCA como rotação trouxe "ganhos computacionais para todos" (Seção 4.3), mas os dados mostram que o Perceptron Logístico (PL) ficou mais lento.
    -   O texto afirma que o FPR na Atividade 8 é "idêntica" entre MQ e PL (Seção 10.2), mas os dados no CSV (`tabela4_intruso.csv`) mostram `0.02` para MQ e `0.01955` para PL. Não são idênticos.
-   **Arredondamento Enganoso:** O tempo de treino do MQ na Tabela 2 do PDF é `0.000s`. O valor real no CSV é `0.0006s`. Arredondar para zero mascara a ordem de grandeza e é uma má prática.

---

## 3. Fragilidades na Metodologia e Análise

Além dos erros de comunicação, a metodologia em si, embora funcional, carece do rigor e da profundidade esperados.

### 3.1. Métrica de Seleção de Hiperparâmetros

-   **Problema:** Os scripts (`A1_A4`, `A5_A6`, `A7`) usam a **acurácia** (`acc`) para selecionar o melhor conjunto de hiperparâmetros.
-   **Análise Crítica:** A acurácia é uma métrica inadequada para problemas multi-classe, mesmo que balanceados, pois não distingue bem o desempenho entre as classes. O **F1-Score (macro)**, que já é calculado, seria uma métrica de seleção muito mais robusta e defensável, pois equilibra precisão e recall por classe. O fato de o critério de seleção ter sido corretamente mudado para `f1_intruso` na Atividade 8 demonstra a consciência do problema, mas essa consciência não foi aplicada retroativamente.

### 3.2. Análise Qualitativa Inexistente (Eigenfaces)

-   **Problema:** O trabalho trata a PCA apenas como uma "caixa-preta" matemática para redução de dimensionalidade.
-   **Análise Crítica:** Em reconhecimento de faces, os componentes principais são as **Eigenfaces**. A visualização das primeiras eigenfaces é uma análise qualitativa padrão e essencial. Ela revela *o que* o modelo está aprendendo (variações de iluminação, traços genéricos, expressões). Omitir essa análise é perder uma camada fundamental de interpretação do problema.

### 3.3. Análise da Transformação Box-Cox

-   **Problema:** A conclusão de que a Box-Cox foi prejudicial é correta, mas a justificativa no relatório é superficial.
-   **Análise Crítica Aprofundada:** A razão fundamental para a falha da Box-Cox é que os dados de entrada (os componentes principais) **já são aproximadamente Gaussianos** por construção (consequência do Teorema do Limite Central). Aplicar uma transformação de normalização potente (Box-Cox) em dados que já são bem-comportados introduz distorções desnecessárias, prejudicando a separabilidade que os modelos lineares exploram. Esta análise de causa e efeito está ausente.

### 3.4. Análise de Erro Inexistente

-   **Problema:** O relatório se baseia apenas em métricas agregadas.
-   **Análise Crítica:** Não há análise de *quais* erros os modelos cometem. Para as atividades multi-classe, uma **matriz de confusão** para o melhor modelo de cada cenário é indispensável para entender quais sujeitos são confundidos entre si. Para a Atividade 8, a análise do trade-off entre Falso Negativo e Falso Positivo é superficial. Uma **Curva ROC** é a ferramenta padrão para visualizar o desempenho do classificador em todos os limiares de decisão, essencial para uma aplicação de segurança.

---

## 4. Recomendações para Elevação do Padrão do Trabalho

Para transformar este trabalho em um artefato de nível de mestrado, as seguintes ações são recomendadas.

### 4.1. Ações Corretivas (Requerem Re-execução Parcial)

1.  **Corrigir a Escala:** Padronize **todos os scripts e o relatório** para usar a mesma escala. A escala **30x30** é a que possui resultados gerados, então o mais simples é **corrigir o relatório** para refletir essa realidade. Isso implica ajustar o texto, a Figura 1 e a dimensão `d=900`.
2.  **Alterar a Métrica de Seleção:** Modifique os scripts `tc2_faces_A1_A4.py`, `tc2_faces_A5_A6.py` e `tc2_faces_A7.py` para usar `f1_macro` como critério de seleção no `random_search`.
    -   **Onde:** Na função `select_best_by_random_search` de cada script.
    -   **Como:**
        ```python
        # DE:
        score = np.mean([r["acc"] for r in reps])
        # PARA:
        score = np.mean([r["f1_macro"] for r in reps])
        ```
3.  **Re-executar os Experimentos:** Após as correções acima, re-execute os scripts para gerar novas tabelas (`tabela1.csv`, `tabela2.csv`, `tabela3.csv`, `tabela3_boxcox.csv`) que reflitam a metodologia corrigida.

### 4.2. Novas Análises e Visualizações (em Notebooks)

**1. Análise Qualitativa das Eigenfaces**

-   **Objetivo:** Entender o que a PCA aprendeu sobre a estrutura das faces.
-   **Onde Inserir:** Criar um novo notebook `notebooks/TC2/analise_qualitativa_pca.ipynb`.
-   **Como Inserir:**

    ```python
    # Em notebooks/TC2/analise_qualitativa_pca.ipynb
    import numpy as np
    import matplotlib.pyplot as plt
    from trabalho_ic_aplicada.dataset_faces import build_face_dataset
    from trabalho_ic_aplicada.models.pca_np import PCA_np

    # 1. Carregar dados (na escala correta, e.g., 30x30) e ajustar PCA
    img_h, img_w = (30, 30)
    X, y, _ = build_face_dataset("./data/raw/Kit_projeto_FACES", size=(img_h, img_w))
    pca = PCA_np()
    pca.fit(X)

    # 2. Função para plotar as eigenfaces
    def plot_eigenfaces(pca_model, h, w, n_eigenfaces=10):
        fig, axes = plt.subplots(2, 5, figsize=(12, 5),
                                 subplot_kw={'xticks':[], 'yticks':[]},
                                 gridspec_kw=dict(hspace=0.1, wspace=0.1))
        for i, ax in enumerate(axes.flat):
            if i < n_eigenfaces:
                # Os componentes estão em pca_model.Vt_
                eigenface = pca_model.Vt_[i, :].reshape(h, w)
                ax.imshow(eigenface, cmap='bone')
                ax.set_title(f"PC {i+1}")
        plt.suptitle("Visualização das Primeiras 10 Eigenfaces")
        plt.savefig("results/TC2/eigenfaces_visualization.png")
        plt.show()

    # 3. Gerar e salvar a figura
    plot_eigenfaces(pca, img_h, img_w)
    ```
    No relatório, adicione uma seção discutindo a figura, explicando que os primeiros componentes capturam principalmente variações de iluminação, enquanto os posteriores podem capturar traços faciais mais sutis.

**2. Análise da Curva ROC para Detecção de Intruso**

-   **Objetivo:** Avaliar profissionalmente o trade-off do classificador de intrusos.
-   **Onde Inserir:** Criar um novo notebook `notebooks/TC2/analise_roc_intruso.ipynb`.
-   **Como Inserir:** Primeiro, adicione um método `predict_proba` aos seus classificadores para obter as probabilidades da softmax.

    ```python
    # Adicionar em trabalho_ic_aplicada/models/clf_pl.py e clf_mlp.py
    # (Exemplo para SoftmaxRegression)
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        from .optim import _softmax # Importação local
        Xb = self._add_bias(X)
        return _softmax(Xb @ self.W_)
    ```

    Depois, no notebook:
    ```python
    # Em notebooks/TC2/analise_roc_intruso.ipynb
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    # Importe seus modelos, dados e o pipeline da Atividade 8

    # ... (código para treinar o melhor modelo da Atividade 8) ...
    # Exemplo para o melhor modelo (model) e dados de teste (Xte_n, yte)
    
    # 1. Obter probabilidades para a classe intruso
    y_probas = model.predict_proba(Xte_n)
    proba_intruso = y_probas[:, intruder_id] # intruder_id é o label da classe intruso

    # 2. Binarizar o y_test
    y_true_bin = (yte == intruder_id).astype(int)

    # 3. Calcular e plotar a curva ROC
    fpr, tpr, thresholds = roc_curve(y_true_bin, proba_intruso)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title(f'Curva ROC para Detecção de Intruso - Modelo {model.__class__.__name__}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("results/TC2/roc_curve_A8.png")
    plt.show()
    ```
    No relatório, substitua a análise pontual de FNR/FPR por uma discussão sobre a curva ROC e o valor da AUC (Area Under the Curve), que mede a capacidade geral de discriminação do modelo.

### 4.3. Proposta de Novas Tabelas (Formato LaTeX)

O relatório deve ser reescrito com tabelas completas e claras.

**Tabela de Resultados por Atividade (Exemplo para Tabela 3 - PCA Reduzida):**
Use este template para as Tabelas 1, 2, 3 e 7 (Box-Cox).

```latex
\begin{table}[h!]
\centering
\caption{Resultados da Atividade 6 (PCA com $q=10$). Métricas de desempenho (média $\pm$ desvio padrão) sobre 50 repetições. O tempo de execução é a média por repetição.}
\label{tab:a6_results}
\resizebox{\textwidth}{!}{
\begin{tabular}{lcccccc}
\toprule
\textbf{Classificador} & \textbf{Acurácia} & \textbf{F1-Score (Macro)} & \textbf{Precisão (Macro)} & \textbf{Recall (Macro)} & \textbf{Tempo de Treino (s)} & \textbf{Tempo de Predição (ms)} \\
\midrule
MQ       & $0.744 \pm 0.045$ & $0.769 \pm 0.041$ & $0.776 \pm 0.040$ & $0.744 \pm 0.045$ & 0.0002 & 0.08 \\
PL       & $0.832 \pm 0.054$ & $0.834 \pm 0.053$ & $0.870 \pm 0.045$ & $0.832 \pm 0.054$ & 0.0277 & 0.07 \\
MLP-1H   & $0.842 \pm 0.047$ & $0.845 \pm 0.046$ & $0.881 \pm 0.040$ & $0.842 \pm 0.047$ & 0.0814 & 0.11 \\
MLP-2H   & $0.846 \pm 0.056$ & $0.842 \pm 0.058$ & $0.879 \pm 0.051$ & $0.846 \pm 0.056$ & 0.2013 & 0.08 \\
\bottomrule
\end{tabular}%
}
\end{table}
```

**Tabela de Hiperparâmetros Vencedores (Exemplo para Atividade 6):**
Crie uma tabela como esta para cada atividade (A2, A4, A6, A7, A8).

```latex
\begin{table}[h!]
\centering
\caption{Hiperparâmetros dos modelos com melhor desempenho na Atividade 6 (PCA com $q=10$), selecionados via busca aleatória com F1-Score (macro) como critério.}
\label{tab:a6_hyperparams}
\begin{tabular}{lcccccc}
\toprule
\textbf{Classificador} & \textbf{Normalização} & \textbf{Otimizador} & \textbf{Ativação} & \textbf{LR} & \textbf{Épocas} & \textbf{L2 / Outros} \\
\midrule
MQ       & \texttt{zscore}      & ---            & ---         & ---      & ---       & $\lambda=10^{-4}$ \\
PL       & \texttt{zscore}      & \texttt{rmsprop} & ---         & 0.02     & 300       & $\lambda=10^{-3}$ \\
MLP-1H   & \texttt{zscore}      & \texttt{adam}    & \texttt{sigmoid} & 0.01     & 150       & $\lambda=10^{-4}$, Hidden=(128,) \\
MLP-2H   & \texttt{minmax}      & \texttt{adam}    & \texttt{tanh}    & 0.005    & 300       & $\lambda=10^{-4}$, Hidden=(128, 64) \\
\bottomrule
\end{tabular}
\end{table}
```

---
## 5. Conclusão da Análise

O projeto possui uma base sólida de código, mas falha gravemente em sua execução como pesquisa científica. A falta de rigor na documentação, as inconsistências factuais e as fragilidades metodológicas o impedem de atingir o padrão esperado. As correções e análises propostas são extensas, mas necessárias. Ao implementá-las, o trabalho pode ser transformado em um estudo coeso, reprodutível e com análises de profundidade, digno de uma avaliação de alto nível.
