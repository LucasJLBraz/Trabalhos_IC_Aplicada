# Análise Final e Crítica Destrutiva do Trabalho 2

**Avaliador:** Gemini
**Data:** 01/09/2025
**Foco:** Rigor científico, consistência entre artefatos, reprodutibilidade e profundidade analítica sob a ótica de uma avaliação de mestrado.

## Veredito Sumário

O presente trabalho, embora sustentado por uma implementação de código-fonte louvável que demonstra domínio técnico, falha catastroficamente em cumprir os requisitos mínimos de rigor científico, consistência e reprodutibilidade esperados em um projeto de mestrado. O relatório, como artefato principal, é um documento cientificamente frágil, repleto de omissões críticas, inconsistências factuais e análises superficiais que o tornam irreplicável e, em última instância, não confiável como registro do trabalho realizado.

A existência de múltiplos conjuntos de resultados conflitantes no repositório, somada à discrepância entre o código-fonte e o relatório, constitui uma falha fundamental de organização e documentação. A base de programação é sólida, mas está a serviço de um processo científico executado e comunicado de forma deficiente.

---

## 1. Falhas Catastróficas de Consistência e Reprodutibilidade

A integridade de um trabalho de pesquisa reside na sua capacidade de ser verificado e reproduzido. O projeto falha neste quesito fundamental.

### 1.1. A Contradição Fundamental: Artefatos Conflitantes

Uma inspeção do repositório revela a existência de dois conjuntos de resultados para as mesmas atividades, gerando uma ambiguidade inaceitável:
- **Resultados em `results/`:** Arquivos como `tabela1.csv` e `pca_q_98.txt` apontam para experimentos com escala de **40x40**.
- **Resultados em `results/TC2/`:** Arquivos com os mesmos nomes apontam para experimentos com escala de **30x30**.

O relatório (`Trabalho_2_ICAp___SBC-5.pdf`) e os scripts em `src/` baseiam-se nos resultados de **30x30**. No entanto, a presença de artefatos conflitantes no mesmo projeto é um erro crasso de gerenciamento de pesquisa. Sugere um fluxo de trabalho desorganizado e lança dúvidas sobre qual versão dos resultados é a canônica. Um avaliador não deveria ter que realizar trabalho investigativo para determinar qual é o experimento válido.

### 1.2. Omissão Sistemática de Hiperparâmetros Críticos

As tabelas de resultados no relatório (Tabelas 3, 5, 7, 9) são cientificamente inúteis para fins de reprodutibilidade. Elas **omitem sistematicamente os hiperparâmetros mais influentes** para os modelos baseados em gradiente: **taxa de aprendizado (`lr`) e número de épocas (`epochs`)**.

- **Exemplo Concreto (Tabela 5 vs. `results/TC2/tabela2.csv`):**
    - O relatório para o MLP-1H na Atividade 4 reporta apenas `minmax_pm1`, `rmsprop` e `sigmoid`.
    - O arquivo CSV revela a configuração real: `lr=0.005`, `epochs=200`, `l2=0.0001`, `clip_grad=2.0`, `hidden=(64,)`.
- **Impacto:** É impossível replicar os resultados a partir do relatório. Esta omissão não é um detalhe, mas uma falha que invalida as tabelas como registro científico.

### 1.3. Inconsistências Factuais no Texto do Relatório

O texto faz afirmações que são diretamente contraditas pelos dados nos arquivos CSV.
- **Afirmação:** A rotação do PCA (A3-4) reduziu o tempo de treino de "todos os classificadores".
- **Fato (`tabela1.csv` vs `tabela2.csv`):** O tempo do PL **aumentou** de 29.5ms para 38.4ms. A afirmação é factualmente incorreta.
- **Afirmação:** Na Atividade 8, o FPR do MQ e do PL são "praticamente nula" e a sensibilidade "unitária".
- **Fato (`tabela4_intruso.csv`):** O FPR médio para ambos é de ~12.2%, o que está longe de ser nulo. A sensibilidade (recall) do PL é de 89.3%, não unitária. A FNR do PL é de 10.7%, não nula. As afirmações são exageradas e imprecisas.

---

## 2. Fragilidades Metodológicas Profundas

Além dos problemas de comunicação, as escolhas metodológicas e a profundidade da análise são insuficientes.

### 2.1. Critério de Seleção de Modelos: A Escolha da Métrica Errada

- **Problema:** Nas atividades de classificação multi-classe (A1-A7), o critério para selecionar o melhor conjunto de hiperparâmetros no `random search` é a **acurácia média**, conforme verificado nos scripts (`score = np.mean([r["acc"] for r in reps])`).
- **Análise Crítica:** A acurácia é uma métrica notoriamente pobre para problemas multi-classe, mesmo que balanceados, pois mascara o desempenho em classes individuais. O **F1-Score (macro)**, que já era calculado, é a métrica padrão para esta tarefa, pois equilibra precisão e recall de forma agnóstica às classes. O fato de o critério de seleção ter sido corretamente alterado para `f1_intruso` na Atividade 8 demonstra que o autor **tinha consciência** da importância de escolher a métrica correta, mas falhou em aplicar este conhecimento de forma consistente, o que constitui uma fraqueza metodológica deliberada.

### 2.2. Análise de PCA: A Oportunidade Perdida das Eigenfaces

- **Problema:** A PCA é tratada unicamente como um algoritmo de compressão, avaliado pela variância explicada.
- **Análise Crítica:** Em reconhecimento de faces, os componentes principais são as famosas **"Eigenfaces"**. Uma análise qualitativa, visualizando as primeiras eigenfaces, é **essencial** para entender *o que* o modelo está aprendendo. As primeiras componentes capturam variações de iluminação? As posteriores capturam traços faciais? Omitir esta análise é tratar o modelo como uma caixa-preta e perder uma camada inteira de interpretação do problema, algo inaceitável em um trabalho de mestrado.

### 2.3. A Falha da Box-Cox: Análise Causal vs. Descrição Superficial

- **Problema:** O relatório constata que a transformação Box-Cox (A7) prejudicou o desempenho, mas a justificativa é superficial.
- **Análise Causal Aprofundada:** A razão fundamental para a falha da Box-Cox neste contexto é que os dados de entrada (os componentes principais) **já são aproximadamente Gaussianos** por construção, como consequência do Teorema do Limite Central. Aplicar uma segunda transformação de normalização potente e não-linear (Box-Cox) em dados que já são bem-comportados não é apenas desnecessário, mas **nocivo**, pois distorce a geometria dos dados que os classificadores lineares exploram. O relatório deveria apresentar esta análise causal, em vez de apenas descrever o resultado.

### 2.4. A Abordagem Supervisionada para Detecção de Intruso (Atividade 8)

- **Problema:** A abordagem de tratar o intruso como uma 16ª classe é apresentada como uma alternativa válida.
- **Análise Crítica:** Esta abordagem é uma **falha conceitual** que resulta em um sistema de segurança de brinquedo. O modelo aprende a detectar um intruso específico, tornando-se inútil contra qualquer outro intruso não visto. A FNR de 0.0 para o MQ e MLP-1H não é um sucesso, mas sim um **sinal claro de overfitting extremo** ao intruso conhecido. O relatório deveria enquadrar esta abordagem não como uma alternativa, mas como um anti-padrão para demonstrar os perigos da generalização e a importância da abordagem unária, que é a única correta para o problema.

---

## 3. Proposta de Correção e Elevação de Patamar

Para que o trabalho atinja um nível de qualidade aceitável, as seguintes ações são **obrigatórias**.

### 3.1. Ações Corretivas (Código e Artefatos)

1.  **Limpeza do Repositório:** Decida por uma única escala (30x30, que corresponde ao código e relatório) e **delete todos os artefatos conflitantes** (os resultados de 40x40).
2.  **Correção da Métrica de Seleção:** Altere os scripts `tc2_faces_A1_A4.py`, `tc2_faces_A5_A6.py` e `tc2_faces_A7.py` para usar `f1_macro` como critério de seleção na busca de hiperparâmetros.
    ```python
    # Em select_best_by_random_search, mude:
    # DE: score = np.mean([r["acc"] for r in reps])
    # PARA:
    score = np.mean([r["f1_macro"] for r in reps])
    ```
3.  **Habilitar Análise ROC:** Implemente um método `predict_proba` nos classificadores em `trabalho_ic_aplicada/models/` para retornar as probabilidades da softmax, necessário para a análise ROC.
4.  **Re-executar Experimentos:** Após as correções, todos os experimentos das Atividades 1 a 7 devem ser re-executados para gerar resultados consistentes com a metodologia corrigida.

### 3.2. Novas Análises Essenciais (Notebooks)

**1. Análise Qualitativa das Eigenfaces (`notebooks/TC2/analise_qualitativa_pca.ipynb`)**
- **Objetivo:** Visualizar e discutir o que a PCA aprendeu.
- **Implementação:**
  ```python
  # Em um novo notebook
  import numpy as np
  import matplotlib.pyplot as plt
  from trabalho_ic_aplicada.dataset_faces import build_face_dataset
  from trabalho_ic_aplicada.models.pca_np import PCA_np

  img_h, img_w = (30, 30)
  X, _, _ = build_face_dataset("./data/raw/Kit_projeto_FACES", size=(img_h, img_w))
  pca = PCA_np().fit(X)

  fig, axes = plt.subplots(2, 5, figsize=(12, 5), subplot_kw={'xticks':[], 'yticks':[]})
  for i, ax in enumerate(axes.flat):
      eigenface = pca.Vt_[i, :].reshape(img_h, img_w)
      ax.imshow(eigenface, cmap='bone')
      ax.set_title(f"PC {i+1}")
  plt.suptitle("Visualização das 10 Primeiras Eigenfaces")
  plt.savefig("results/TC2/eigenfaces_visualization.png")
  plt.show()
  ```

**2. Análise da Curva ROC para Detecção de Anomalias (`notebooks/TC2/analise_roc_intruso.ipynb`)**
- **Objetivo:** Avaliar profissionalmente o trade-off da abordagem unária, que é a única realista.
- **Implementação:**
  ```python
  # Em um novo notebook, após treinar o melhor modelo unário (ex: AE_1H)
  import numpy as np
  import matplotlib.pyplot as plt
  from sklearn.metrics import roc_curve, auc

  # Assumindo que 'model' é o seu detector de anomalias treinado
  # e 'scores_auth_te', 'scores_intruder_te' são os scores de anomalia no teste
  y_true = np.concatenate([np.zeros(len(scores_auth_te)), np.ones(len(scores_intruder_te))])
  y_scores = np.concatenate([scores_auth_te, scores_intruder_te])

  fpr, tpr, _ = roc_curve(y_true, y_scores)
  roc_auc = auc(fpr, tpr)

  plt.figure(figsize=(8, 6))
  plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.3f})')
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlabel('Taxa de Falsos Positivos (FPR)')
  plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
  plt.title(f'Curva ROC para Detecção de Anomalias - {model.__class__.__name__}')
  plt.legend(loc="lower right"); plt.grid(True)
  plt.savefig("results/TC2/roc_curve_A8.png")
  plt.show()
  ```

### 3.3. Proposta de Novas Tabelas (Formato LaTeX)

O relatório deve ser reescrito com tabelas completas.

**Tabela de Hiperparâmetros Vencedores (Exemplo para Atividade 6):**
```latex
\begin{table}[h!]
\centering
\caption{Hiperparâmetros dos modelos com melhor F1-Score (Macro) na Atividade 6 (PCA com $q=79$).}
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
\end{tabular}% 
}
\end{table}
```

**Tabela de Resultados por Atividade (Exemplo para Atividade 6):**
```latex
\begin{table}[h!]
\centering
\caption{Resultados da Atividade 6 (PCA com $q=79$). Métricas sobre 50 repetições (média $\pm$ desvio padrão).}
\label{tab:a6_results}
\resizebox{\textwidth}{!}{
\begin{tabular}{lcccc}
\toprule
\textbf{Classificador} & \textbf{Acurácia} & \textbf{F1-Score (Macro)} & \textbf{Tempo Total (ms)} \\
\midrule
MQ       & $0.959 \pm 0.029$ & $0.957 \pm 0.030$ & $0.260 \pm 0.05$ \\
PL       & $0.959 \pm 0.029$ & $0.957 \pm 0.030$ & $21.692 \pm 1.5$ \\
MLP-1H   & $0.956 \pm 0.027$ & $0.954 \pm 0.029$ & $53.646 \pm 5.1$ \\
MLP-2H   & $0.948 \pm 0.034$ & $0.946 \pm 0.036$ & $442.021 \pm 20.3$ \\
\bottomrule
\end{tabular}% 
}
\end{table}
```

---
## 4. Avaliação Final e Nota

O projeto, em seu estado atual, é um exemplo de bom potencial de engenharia minado por um processo científico falho. A falta de rigor na documentação, as inconsistências críticas entre os artefatos e a superficialidade das análises o colocam abaixo do padrão mínimo para uma avaliação de mestrado.

**Nota Final: 6.0 / 10**

**Justificativa:** A nota reflete um trabalho com uma base de código funcional (que vale a maior parte dos pontos), mas que é severamente penalizado por falhas graves na execução da pesquisa. A nota não é mais baixa porque as conclusões gerais (PL é eficiente, PCA-reduce é bom, Box-Cox é ruim, unário > supervisionado para intrusos) estão, fortuitamente, alinhadas com a teoria, e o esforço de implementação é evidente. No entanto, o trabalho como um todo não se sustenta cientificamente. A implementação das correções propostas é essencial para que o projeto seja considerado aprovado.
