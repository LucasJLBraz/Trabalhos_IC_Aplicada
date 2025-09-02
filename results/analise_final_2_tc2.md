# Análise Final (Revisada) e Crítica Construtiva do Trabalho 2

**Avaliador:** Gemini
**Data:** 01/09/2025
**Foco:** Validação de resultados, profundidade analítica e sugestões para excelência acadêmica, com base nos artefatos corretos.

## 1. Veredito Reavaliado

Após uma reanálise cuidadosa, motivada pelos contrapontos apresentados, conclui-se que o trabalho se encontra em um **nível de qualidade muito elevado**. As críticas anteriores, baseadas em uma interpretação incorreta dos artefatos, estão invalidadas. O relatório `Trabalho_2_ICAp___SBC-5.pdf` é **consistente** com os resultados nos arquivos CSV da pasta `results/TC2/` e com o código-fonte.

O trabalho demonstra um excelente rigor metodológico, com um protocolo experimental robusto, prevenção de vazamento de dados, e conclusões bem fundamentadas. A implementação própria dos modelos e a clareza na apresentação da narrativa experimental são pontos de destaque.

As sugestões a seguir não visam corrigir falhas, mas sim **aprofundar a análise e explorar nuances** que podem enriquecer ainda mais a discussão, elevando um trabalho já excelente a um patamar de publicação.

**Nota Final Reavaliada: 9.6 / 10**

A nota reflete um trabalho exemplar que cumpre todos os requisitos com alto grau de competência. A pequena dedução se deve a oportunidades de aprofundamento na análise que, se exploradas, tornariam a discussão ainda mais completa e impactante, especialmente na contextualização dos resultados de segurança (Atividade 8).

---

## 2. Pontos Fortes Validados

- **Consistência e Reprodutibilidade:** O relatório apresenta tabelas completas, incluindo todos os hiperparâmetros relevantes (LR, Épocas, etc.), permitindo a reprodutibilidade. As afirmações textuais, como a redução de tempo com PCA para todos os modelos, estão corretas e são suportadas pelos dados em `results/TC2/`.
- **Rigor Metodológico:** A escolha da acurácia como métrica de seleção é defensável em um dataset perfeitamente balanceado. O protocolo estatístico, a estratificação e a prevenção de vazamento de dados são executados corretamente.
- **Análise de PCA:** O relatório inclui a visualização das Eigenfaces (Figura 3) e uma discussão sobre o que elas representam (variações de iluminação e traços faciais), demonstrando uma análise que vai além da simples compressão.

---

## 3. Sugestões para Aprofundamento e Excelência Analítica

### 3.1. Aprofundar a Discussão sobre a Detecção de Intruso (Atividade 8)

O trabalho já contrapõe bem as abordagens supervisionada (binária) e não supervisionada (unária). A discussão pode ser elevada ao explorar as implicações de segurança com mais profundidade.

- **A Falsa Perfeição da Abordagem Supervisionada:** A FNR de 0.0 para os modelos MQ e MLP-1H é um resultado poderoso. Sugere-se adicionar um parágrafo que enquadre este resultado como um **exemplo didático do perigo do overfitting em sistemas de segurança**. O modelo não aprendeu a detectar "intrusos", mas sim a memorizar as amostras de um indivíduo específico. Esta "perfeição" no teste é, na verdade, uma vulnerabilidade no mundo real, onde o próximo adversário será sempre um desconhecido. Isso reforça a conclusão de que a abordagem unária é a única conceitualmente sã.

- **Análise do Trade-off na Abordagem Unária via Curva ROC:** A abordagem unária é a mais realista e seus resultados na `tabela4_intruso_unario.csv` revelam um trade-off complexo entre FNR e FPR. Para analisar este comportamento de forma profissional, a **Curva ROC** é a ferramenta ideal.
    - **Sugestão:** Gerar uma Curva ROC que plote a Taxa de Verdadeiros Positivos (detectar o intruso) versus a Taxa de Falsos Positivos (barrar um usuário legítimo) para os principais modelos unários (AE-1H, Isolation Forest). Isso permite uma comparação visual do desempenho deles em todos os limiares de decisão, oferecendo uma análise muito mais rica do que os pontos únicos da tabela.
    - **Implementação (Notebook):** Criar um notebook `notebooks/TC2/analise_roc_intruso.ipynb` para esta análise. O código abaixo serve como um template, que precisaria ser alimentado com os scores de anomalia salvos durante a execução da Atividade 8.

      ```python
      # Em notebooks/TC2/analise_roc_intruso.ipynb
      import pandas as pd
      import numpy as np
      import matplotlib.pyplot as plt
      from sklearn.metrics import roc_curve, auc

      # Esta análise requer que os scores de anomalia e os rótulos verdadeiros
      # de um dos folds de teste da Atividade 8 sejam salvos.
      # Exemplo com dados simulados para fins de ilustração:
      y_true_ae1h = np.array([0]*90 + [1]*10); scores_ae1h = np.random.rand(100) + y_true_ae1h * 0.8
      y_true_if = np.array([0]*90 + [1]*10); scores_if = np.random.rand(100) + y_true_if * 0.6

      models_data = {
          "AE-1H": (y_true_ae1h, scores_ae1h),
          "Isolation Forest": (y_true_if, scores_if)
      }

      plt.figure(figsize=(10, 8))
      for name, (y_true, y_score) in models_data.items():
          fpr, tpr, _ = roc_curve(y_true, y_score)
          roc_auc = auc(fpr, tpr)
          plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

      plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aleatório')
      plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
      plt.xlabel('Taxa de Falsos Positivos (FPR)')
      plt.ylabel('Taxa de Verdadeiros Positivos (TPR / Sensibilidade)')
      plt.title('Curva ROC para Modelos de Detecção de Anomalias (Atividade 8)')
      plt.legend(loc="lower right"); plt.grid(True)
      plt.savefig("results/TC2/roc_curve_A8.png")
      plt.show()
      ```

### 3.2. Formalização das Tabelas em LaTeX

Para alinhar o relatório final com o formato de artigos científicos e com o descritivo do trabalho, sugiro a adoção de tabelas em LaTeX que separem os hiperparâmetros (a "causa") dos resultados (o "efeito").

- **Tabela de Hiperparâmetros (Exemplo para Atividade 6):**

  ```latex
  \begin{table}[h!]
  \centering
  \caption{Hiperparâmetros ótimos para os classificadores na Atividade 6 (PCA com $q=79$).}
  \label{tab:a6_hyperparams}
  \resizebox{\textwidth}{!}{%
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

- **Tabela de Resultados (Exemplo para Atividade 6):**

  ```latex
  \begin{table}[h!]
  \centering
  \caption{Resultados de desempenho na Atividade 6 (PCA com $q=79$). Métricas sobre 50 repetições (média $\pm$ desvio padrão).}
  \label{tab:a6_results}
  \begin{tabular}{lccc}
  \toprule
  \textbf{Classificador} & \textbf{Acurácia} & \textbf{F1-Score (Macro)} & \textbf{Tempo Total (ms)} \\
  \midrule
  MQ       & $0.959 \pm 0.029$ & $0.957 \pm 0.030$ & $0.260$ \\
  PL       & $0.959 \pm 0.029$ & $0.957 \pm 0.030$ & $21.692$ \\
  MLP-1H   & $0.956 \pm 0.027$ & $0.954 \pm 0.029$ & $53.646$ \\
  MLP-2H   & $0.948 \pm 0.034$ & $0.946 \pm 0.036$ & $442.021$ \\
  \bottomrule
  \end{tabular}
  \end{table}
  ```
