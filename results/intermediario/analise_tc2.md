
# Análise Crítica do Relatório - Trabalho 2: Reconhecimento de Faces

**Avaliador:** Gemini
**Data:** 27/08/2025

## 1. Avaliação Geral

O trabalho apresentado é de **excelente qualidade**, demonstrando um profundo entendimento dos conceitos de classificação de padrões, pré-processamento e validação de modelos. A estrutura metodológica é robusta e segue as melhores práticas científicas, como a separação rigorosa entre treino e teste, a prevenção de vazamento de dados e o uso de um protocolo estatístico com múltiplas repetições para garantir a confiabilidade dos resultados.

A implementação em Python, realizada sem depender de frameworks de alto nível como o `scikit-learn` para a lógica central dos modelos, é um diferencial notável que evidencia um domínio técnico aprofundado.

O relatório é bem escrito, claro e consegue conectar os resultados obtidos com os objetivos propostos em cada atividade. As análises são, em geral, pertinentes e bem fundamentadas. As poucas críticas levantadas são, na sua maioria, detalhes de refinamento e pequenas inconsistências, que não comprometem a validade e a qualidade do trabalho.

---

## 2. Pontos Fortes

1.  **Rigor Metodológico:** A adesão a um protocolo experimental estrito é o maior ponto forte do trabalho. O uso de:
    *   **Divisão Estratificada:** Garante que a proporção de classes seja mantida entre treino e teste, crucial para datasets com poucos exemplos por classe como o Yale A.
    *   **Busca de Hiperparâmetros (Random Search):** Uma forma eficiente de explorar espaços de busca complexos, como os das MLPs.
    *   **Múltiplas Repetições:** A reavaliação do melhor candidato (10x) e a avaliação final (50x) fornecem estimativas de desempenho (média e desvio-padrão) muito mais confiáveis do que uma única execução.

2.  **Prevenção de Vazamento de Dados (Data Leakage):** O relatório e o código demonstram a preocupação correta em ajustar os transformadores (PCA, Normalizadores, Box-Cox) **exclusivamente** no conjunto de treino de cada repetição. Este é um erro metodológico comum e crítico que foi evitado com sucesso aqui.

3.  **Implementação Própria:** A decisão de implementar os classificadores e otimizadores em `numpy` é louvável. Isso força um entendimento detalhado dos algoritmos, que transparece na qualidade das análises.

4.  **Qualidade da Análise:** As discussões em cada seção são bem fundamentadas. Por exemplo, a hipótese de que a rotação do PCA pode prejudicar modelos lineares e a justificativa para a falha da transformação de Box-Cox são excelentes pontos de análise.

---

## 3. Análise Crítica e Pontos de Melhoria

A seguir, uma análise detalhada de inconsistências e pontos que poderiam ser mais aprofundados.

### 3.1. Inconsistências no Relatório

-   **Figura da Variância Explicada (Atividade 5):** O relatório na Seção 5 menciona `fig_variance_explained.png`, que parece ser um nome de arquivo genérico. O script `src/tc2_faces_A5_A6.py` gera a figura com o nome `results/pca_variance_explained_A3.png`. O relatório deveria referenciar o nome de arquivo correto para garantir a rastreabilidade.

-   **Cálculo do Espaço de Busca (MLP-2H):** Na Seção 2.2, o relatório afirma que o espaço de busca da MLP-2H com normalização tem 349.920 combinações. Uma verificação baseada nos hiperparâmetros listados e no código (`MLP2HSampler` em `tc2_faces_A1_A4.py`) resulta em um número diferente: `6 (h1) * 6 (h2) * 6 (act) * 3 (lr) * 3 (epochs) * 3 (l2) * 5 (opt) * 3 (clip) * 4 (norm) = 699.840`. O valor no relatório está incorreto por um fator de 2. Embora isso não afete os resultados (já que foi feito um *sampling* de 60), é uma pequena imprecisão no texto.

-   **Nomenclatura de Hiperparâmetros:** O relatório utiliza símbolos matemáticos padrão (λ para L2, η para taxa de aprendizado), enquanto o código e as tabelas de resultados usam nomes de variáveis (`l2`, `lr`). Isso é comum, mas para um rigor máximo, seria ideal manter a consistência ou adicionar uma nota de rodapé mapeando os símbolos aos nomes das colunas.

### 3.2. Aprofundamento Metodológico

-   **Justificativa para o Clipping de Gradiente:** A Seção 2.3 oferece uma excelente justificativa qualitativa para o uso do clipping. O trabalho poderia ser enriquecido ao **demonstrar** essa necessidade. Uma sugestão é rodar um dos modelos MLP (ex: MLP-1H com ativação `relu` e `lr` alto) com e sem clipping para algumas sementes aleatórias e plotar as curvas de perda do treino. A expectativa é ver picos ou instabilidades na versão sem clipping.

-   **Análise da Transformação Box-Cox (Atividade 7):** A conclusão de que a Box-Cox é prejudicial está correta e bem justificada. Para tornar a análise ainda mais forte, seria interessante **visualizar** a distorção. Uma sugestão é plotar o histograma de 2 ou 3 dos primeiros componentes principais (após o PCA) e, ao lado, o histograma desses mesmos componentes após a transformação Box-Cox. Isso mostraria visualmente como a transformação, que busca a normalidade, pode distorcer dados que já são aproximadamente gaussianos.

-   **Métricas da Atividade 8 (Intruso):**
    -   A Tabela 6 reporta `FNR` e `Sensibilidade` (TPR). Como `TPR = 1 - FNR`, apresentar ambas as métricas é redundante. Em relatórios, é comum escolher uma delas (geralmente TPR/Sensibilidade ou Recall) para evitar redundância.
    -   A discussão sobre ajustar o limiar de decisão da softmax para calibrar o trade-off FNR/FPR é excelente. Esta é a abordagem correta para um sistema de controle de acesso real. O trabalho se beneficiaria enormemente ao transformar essa discussão em um experimento. Uma **Curva ROC (Receiver Operating Characteristic)** seria a visualização ideal para isso, plotando TPR vs. FPR para vários limiares.

-   **Desvio da Especificação (Atividade 8):** O enunciado pedia 11 imagens de intruso e foram usadas 10. O relatório justifica a mudança por "limitações de disponibilidade". Isso é aceitável e foi bem documentado, o que é a prática correta ao se desviar de um requisito.

---

## 4. Sugestões de Novos Experimentos (Opcional)

Caso haja tempo e interesse em aprofundar a análise, sugiro os seguintes pontos.

### 4.1. Visualizar o Efeito do Clipping de Gradiente

Crie um script de análise ou um notebook para comparar as curvas de perda.

**Código Sugerido (Conceitual):**

```python
# Em um novo script ou notebook
import numpy as np
import matplotlib.pyplot as plt
from trabalho_ic_aplicada.models.clf_mlp import MLPClassifier
from trabalho_ic_aplicada.dataset_faces import build_face_dataset

# Carregar dados (ex: 40x40)
X, y, _ = build_face_dataset("./data/raw/Kit_projeto_FACES", size=(40,40))

# Parâmetros que podem causar instabilidade
params = {
    "hidden": (256,),
    "activation": "relu",
    "lr": 0.02, # LR relativamente alto
    "epochs": 150,
    "opt": "adam",
    "l2": 0
}

plt.figure(figsize=(10, 6))

# Rodar com e sem clipping para algumas sementes
for seed in [42, 101, 2025]:
    # Com clipping (padrão do seu código)
    mlp_clip = MLPClassifier(**params, clip_grad=5.0, seed=seed)
    mlp_clip.fit(X, y)
    plt.plot(mlp_clip.loss_history_, color='b', alpha=0.6, label=f'Com Clip (seed={seed})' if seed==42 else "")

    # Sem clipping
    mlp_no_clip = MLPClassifier(**params, clip_grad=np.inf, seed=seed) # Desativa o clipping
    mlp_no_clip.fit(X, y)
    plt.plot(mlp_no_clip.loss_history_, color='r', linestyle='--', alpha=0.6, label=f'Sem Clip (seed={seed})' if seed==42 else "")

plt.title("Efeito do Clipping de Gradiente na Estabilidade do Treino")
plt.xlabel("Época")
plt.ylabel("Loss (Cross-Entropy)")
plt.legend()
plt.grid(True)
plt.ylim(0, np.nanmin(mlp_clip.loss_history_)*5) # Limita o eixo Y para focar na região de interesse
plt.savefig("results/analise_clipping.png")
plt.show()
```

### 4.2. Visualizar a Distorção da Box-Cox

**Código Sugerido (para um notebook):**

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from trabalho_ic_aplicada.dataset_faces import build_face_dataset
from trabalho_ic_aplicada.models.pca_np import PCA_np
from src.tc2_faces_A7 import fit_boxcox_then_zscore # Importar sua função

# Carregar dados e aplicar PCA com q=10
X, y, _ = build_face_dataset("./data/raw/Kit_projeto_FACES", size=(40,40))
pca = PCA_np(q=10)
X_pca = pca.fit_transform(X)

# Aplicar Box-Cox
X_boxcox, _ = fit_boxcox_then_zscore(X_pca)

# Plotar histogramas
fig, axs = plt.subplots(2, 3, figsize=(15, 8))

for i in range(3):
    # Antes da Box-Cox
    sns.histplot(X_pca[:, i], kde=True, ax=axs[0, i])
    axs[0, i].set_title(f'Componente Principal {i+1} (Original)')

    # Depois da Box-Cox + z-score
    sns.histplot(X_boxcox[:, i], kde=True, ax=axs[1, i])
    axs[1, i].set_title(f'Componente Principal {i+1} (Após Box-Cox)')

plt.tight_layout()
plt.savefig("results/analise_boxcox_dist.png")
plt.show()
```

---

## 5. Avaliação Final

O trabalho é exemplar e demonstra um alto nível de competência técnica e rigor metodológico. As conclusões são bem suportadas pelos dados e as análises são perspicazes.

As críticas apontadas são pequenas e, em sua maioria, relacionadas a detalhes de apresentação no relatório, não comprometendo a qualidade da execução experimental ou da implementação.

**Nota Final: 9.5 / 10**

**Justificativa:** A nota reflete um trabalho quase perfeito. A dedução de 0.5 ponto se deve às pequenas inconsistências factuais no relatório (cálculo do espaço de busca, nome da figura) que poderiam ser sanadas com uma revisão final. A robustez do protocolo experimental, a qualidade da implementação e a profundidade da análise são os grandes destaques e justificam a avaliação elevada.
