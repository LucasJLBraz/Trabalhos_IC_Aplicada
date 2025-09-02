# Análise Crítica (V3) do Relatório e Metodologia - TC2

**Avaliador:** Gemini
**Data:** 31/08/2025
**Foco:** Rigor científico, consistência entre artefatos (relatório, código, resultados) e profundidade analítica para um padrão de pós-graduação.

## 1. Veredito Sumário

O trabalho apresenta uma base de implementação em Python louvável, demonstrando esforço e conhecimento técnico na construção dos modelos a partir de rotinas de baixo nível. Contudo, essa qualidade na implementação é ofuscada por **falhas sistêmicas e graves na comunicação dos resultados, na consistência entre os artefatos do projeto e no rigor metodológico**.

O relatório, em sua forma atual, **não se sustenta como um documento científico autônomo**. Ele contém informações factualmente incorretas, omite hiperparâmetros críticos que inviabilizam a reprodutibilidade, e apresenta análises superficiais que não exploram a profundidade dos resultados obtidos. A discrepância entre a escala de imagem reportada no PDF (40x40) e a utilizada nos scripts para gerar os resultados principais (30x30) é um erro crasso que compromete a validade de grande parte das conclusões apresentadas.

Esta análise detalha essas falhas e oferece um caminho claro para a correção, visando transformar o projeto de um bom exercício de programação em um trabalho de pesquisa com mérito acadêmico.

---

## 2. Falhas Críticas de Consistência e Reprodutibilidade

A análise cruzada do relatório (`.pdf`), dos resultados (`.csv`) e do código-fonte (`.py`) revela uma desconexão fundamental entre o que foi reportado e o que foi executado.

### 2.1. Discrepância na Escala das Imagens: O Erro Fundamental

-   **Onde:** O relatório (`Trabalho_2_ICAp___SBC-5.pdf`, Seção 2.1 e Figura 1) afirma que a escala de 30x30 foi escolhida como padrão para as atividades subsequentes.
-   **Problema:** Os arquivos de resultados (`results/tabela1.csv`, `tabela2.csv`, etc.) e o arquivo `results/pca_q_98.txt` indicam que a escala utilizada foi **40x40**, resultando em `d=1600` e `q=10`. No entanto, os scripts em `src/` (`tc2_faces_A1_A4.py`, `tc2_faces_A5_A6.py`) estão configurados para usar a escala **30x30** (`SELECT_SCALE_ID = -1` ou `[1]`, que corresponde a 30x30 na lista de escalas, e não 40x40`).
-   **Impacto:** **Crasso e sistêmico**. Há uma contradição direta entre os resultados gerados e a configuração do código, e entre o relatório e os resultados. Isso invalida a narrativa do documento e sugere que os resultados apresentados podem não corresponder ao código fornecido. É a falha mais grave do trabalho.

### 2.2. Omissão Sistemática de Hiperparâmetros Críticos

-   **Onde:** Tabelas 3, 5 e 7 do relatório PDF.
-   **Problema:** As tabelas **omitem sistematicamente os hiperparâmetros mais importantes** para modelos de redes neurais: **taxa de aprendizado (`lr`) e número de épocas (`epochs`)**. Também omitem o parâmetro de `clip_grad` quando aplicável.
-   **Impacto:** Impossibilita a reprodutibilidade a partir do relatório. É uma falha grave de comunicação científica. Por exemplo, na Tabela 5, o MLP-1H vencedor usou `lr=0.0050` e `epochs=200` (informação presente apenas no `tabela2.csv`), mas o relatório omite isso. Sem essa informação, os resultados são irreplicáveis.

### 2.3. Inconsistências Factuais e de Nomenclatura

-   **Erro Conceitual (`min-max [1,1]`):** A crítica anterior sobre `min-max [1,1]` foi notada, mas vale reforçar: esta normalização não existe. O correto é `min-max [-1,1]`, que corresponde a `minmax_pm1` no código.
-   **Afirmações Contraditórias:**
    -   O texto afirma que a PCA como rotação trouxe "ganhos computacionais para todos", mas os dados mostram que o Perceptron Logístico (PL) ficou mais lento.
    -   O texto afirma que o FPR na Atividade 8 é "idêntica" entre MQ e PL, mas os dados no CSV (`tabela4_intruso.csv`) mostram valores diferentes.
-   **Arredondamento Enganoso:** O tempo de treino do MQ na Tabela 4 do PDF é `2.258ms`. O valor real no CSV é `0.002257...s`. Apresentar em milissegundos é aceitável, mas a precisão deve ser consistente.

---

## 3. Fragilidades na Metodologia e Análise

Além dos erros de comunicação, a metodologia em si, embora funcional, carece do rigor e da profundidade esperados.

### 3.1. Métrica de Seleção de Hiperparâmetros

-   **Problema:** Os scripts (`A1_A4`, `A5_A6`, `A7`) usam a **acurácia** (`acc`) para selecionar o melhor conjunto de hiperparâmetros.
-   **Análise Crítica:** A acurácia é uma métrica inadequada para problemas multi-classe, mesmo que balanceados, pois não distingue bem o desempenho entre as classes. O **F1-Score (macro)**, que já é calculado, seria uma métrica de seleção muito mais robusta e defensável. O fato de o critério de seleção ter sido corretamente mudado para `f1_intruso` na Atividade 8 demonstra a consciência do problema, mas essa consciência não foi aplicada retroativamente.

### 3.2. Análise Qualitativa Inexistente (Eigenfaces)

-   **Problema:** O trabalho trata a PCA apenas como uma "caixa-preta" matemática.
-   **Análise Crítica:** Em reconhecimento de faces, os componentes principais são as **Eigenfaces**. A visualização das primeiras eigenfaces é uma análise qualitativa padrão e essencial. Ela revela *o que* o modelo está aprendendo. Omitir essa análise é perder uma camada fundamental de interpretação.

### 3.3. Análise da Transformação Box-Cox

-   **Problema:** A conclusão de que a Box-Cox foi prejudicial é correta, mas a justificativa no relatório é superficial.
-   **Análise Crítica Aprofundada:** A razão fundamental para a falha da Box-Cox é que os dados de entrada (os componentes principais) **já são aproximadamente Gaussianos** por construção (consequência do Teorema do Limite Central). Aplicar uma transformação de normalização potente (Box-Cox) em dados que já são bem-comportados introduz distorções desnecessárias. Esta análise de causa e efeito está ausente.

### 3.4. Análise de Erro Inexistente

-   **Problema:** O relatório se baseia apenas em métricas agregadas.
-   **Análise Crítica:** Não há análise de *quais* erros os modelos cometem. Para as atividades multi-classe, uma **matriz de confusão** é indispensável. Para a Atividade 8, a análise do trade-off entre Falso Negativo e Falso Positivo é superficial. Uma **Curva ROC** é a ferramenta padrão para visualizar o desempenho do classificador em todos os limiares de decisão.

---

## 4. Recomendações para Elevação do Padrão do Trabalho

### 4.1. Ações Corretivas (Requerem Re-execução)

1.  **Corrigir a Escala:** Padronize **todos os scripts e o relatório** para usar a mesma escala. A escala **30x30** parece ser a intenção original do código. **Corrija o relatório** para refletir essa realidade (`d=900`, `q=79`).
2.  **Alterar a Métrica de Seleção:** Modifique os scripts `tc2_faces_A1_A4.py`, `tc2_faces_A5_A6.py` e `tc2_faces_A7.py` para usar `f1_macro` como critério de seleção no `random_search`.
    -   **Onde:** Na função `select_best_by_random_search` de cada script.
    -   **Como:**
        ```python
        # DE:
        score = np.mean([r["acc"] for r in reps])
        # PARA:
        score = np.mean([r["f1_macro"] for r in reps])
        ```
3.  **Re-executar os Experimentos:** Após as correções, re-execute os scripts para gerar novas tabelas (`tabela1.csv`, `tabela2.csv`, `tabela3.csv`, `tabela3_boxcox.csv`) que reflitam a metodologia corrigida.

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
                eigenface = pca_model.Vt_[i, :].reshape(h, w)
                ax.imshow(eigenface, cmap='bone')
                ax.set_title(f"PC {i+1}")
        plt.suptitle("Visualização das Primeiras 10 Eigenfaces")
        plt.savefig("results/TC2/eigenfaces_visualization.png")
        plt.show()

    # 3. Gerar e salvar a figura
    plot_eigenfaces(pca, img_h, img_w)
    ```

**2. Análise da Curva ROC para Detecção de Intruso**

-   **Objetivo:** Avaliar profissionalmente o trade-off do classificador de intrusos.
-   **Onde Inserir:** Criar um novo notebook `notebooks/TC2/analise_roc_intruso.ipynb`.
-   **Como Inserir:** Primeiro, adicione um método `predict_proba` aos seus classificadores.

    ```python
    # Adicionar em trabalho_ic_aplicada/models/clf_pl.py e clf_mlp.py
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
    
    y_probas = model.predict_proba(Xte_n)
    proba_intruso = y_probas[:, intruder_id]
    y_true_bin = (yte == intruder_id).astype(int)

    fpr, tpr, thresholds = roc_curve(y_true_bin, proba_intruso)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title(f'Curva ROC para Detecção de Intruso - Modelo {model.__class__.__name__}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("results/TC2/roc_curve_A8.png")
    plt.show()
    ```

### 4.3. Proposta de Novas Tabelas (Formato LaTeX)

O relatório deve ser reescrito com tabelas completas e claras.

**Tabela de Resultados por Atividade (Exemplo para Tabela 3 - PCA Reduzida):**

```latex
\begin{table}[h!]
\centering
\caption{Resultados da Atividade 6 (PCA com $q=79$). Métricas de desempenho (média $\pm$ desvio padrão) sobre 50 repetições.}
\label{tab:a6_results}
\resizebox{\textwidth}{!}{
\begin{tabular}{lccccc}
\toprule
\textbf{Classificador} & \textbf{Acurácia} & \textbf{F1-Score (Macro)} & \textbf{Precisão (Macro)} & \textbf{Recall (Macro)} & \textbf{Tempo Total (ms)} \\
\midrule
MQ       & $0.959 \pm 0.029$ & $0.957 \pm 0.030$ & $0.969 \pm 0.024$ & $0.959 \pm 0.029$ & 0.260 \\
PL       & $0.959 \pm 0.029$ & $0.957 \pm 0.030$ & $0.969 \pm 0.024$ & $0.959 \pm 0.029$ & 21.692 \\
MLP-1H   & $0.956 \pm 0.027$ & $0.954 \pm 0.029$ & $0.968 \pm 0.023$ & $0.956 \pm 0.027$ & 53.646 \\
MLP-2H   & $0.948 \pm 0.034$ & $0.946 \pm 0.036$ & $0.963 \pm 0.028$ & $0.948 \pm 0.034$ & 442.021 \\
\bottomrule
\end{tabular}% 
}
\end{table}
```

**Tabela de Hiperparâmetros Vencedores (Exemplo para Atividade 6):**

```latex
\begin{table}[h!]
\centering
\caption{Hiperparâmetros dos modelos com melhor desempenho na Atividade 6 (PCA com $q=79$), selecionados via busca aleatória com F1-Score (macro) como critério.}
\label{tab:a6_hyperparams}
\resizebox{\textwidth}{!}{
\begin{tabular}{lccccccc}
\toprule
\textbf{Classificador} & \textbf{Normalização} & \textbf{Otimizador} & \textbf{Ativação} & \textbf{Hidden} & \textbf{LR} & \textbf{Épocas} & \textbf{L2 / Clip} \\
\midrule
MQ       & \texttt{minmax}      & ---            & ---         & ---       & ---      & ---       & $\lambda=10^{-4}$ \\
PL       & \texttt{zscore}      & \texttt{sgd}    & ---         & ---       & 0.005    & 200       & $\lambda=10^{-4}$ \\
MLP-1H   & \texttt{zscore}      & \texttt{rmsprop} & \texttt{swish} & (16,)     & 0.02     & 200       & $\lambda=10^{-4}$, Clip=2.0 \\
MLP-2H   & \texttt{zscore}      & \texttt{rmsprop} & \texttt{leaky_relu} & (512, 64) & 0.005    & 300       & $\lambda=10^{-3}$, Clip=0.0 \\
\bottomrule
\end{tabular}
}
\end{table}
```

---
## 5. Conclusão da Análise

O projeto possui uma base sólida de código, mas falha gravemente em sua execução como pesquisa científica. A falta de rigor na documentação, as inconsistências factuais e as fragilidades metodológicas o impedem de atingir o padrão esperado. As correções e análises propostas são extensas, mas necessárias. Ao implementá-las, o trabalho pode ser transformado em um estudo coeso, reprodutível e com análises de profundidade, digno de uma avaliação de alto nível.