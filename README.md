# Trabalhos de Inteligência Computacional Aplicada

Este repositório contém as implementações e experimentos para os trabalhos da disciplina de Inteligência Computacional Aplicada (TIP7077/CCP9011).

## Estrutura do Projeto

- **/trabalho_ic_aplicada**: Biblioteca Python com os módulos principais, incluindo implementações dos modelos, processamento de dados e otimizadores.
- **/src**: Scripts Python executáveis para rodar os experimentos do Trabalho 2.
- **/notebooks**: Jupyter Notebooks utilizados para os experimentos do Trabalho 1.
- **/data**: Diretório para armazenar os datasets.
  - **/data/raw**: Dados brutos, como o dataset de faces.
- **/results**: Saídas geradas pelos scripts, como tabelas de resultados (.csv) e figuras (.png).
- **/references**: Enunciados dos trabalhos e outros materiais de apoio.

## Setup do Ambiente

1.  **Clone o repositório:**
    ```bash
    git clone <url-do-repositorio>
    cd Trabalhos_IC_Aplicada
    ```

2.  **Crie um ambiente virtual (recomendado):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # No Windows: .venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Trabalho 1: Regressão de Valores de Imóveis

### Objetivo

Implementar e avaliar modelos de regressão — Regressão Linear Múltipla (MQ), Perceptron de Regressão (PS) e Redes Neurais Artificiais (MLP com 1 e 2 camadas ocultas) — para prever o valor de imóveis, utilizando o dataset "Real estate valuation".

### Como Executar

Os experimentos estão implementados nos Jupyter Notebooks localizados na pasta `notebooks/TC1/`. O dataset é baixado automaticamente via `ucimlrepo`.

Para reproduzir os resultados, execute as células dos seguintes notebooks em ordem:

1.  **`notebooks/TC1/0.03-ljlb-regressao-multipla-MQO.ipynb`**: Implementa e avalia o modelo de Regressão Linear por Mínimos Quadrados.
2.  **`notebooks/TC1/1.03-ljlb-regressao-perceptron-simples.ipynb`**: Implementa e avalia o Perceptron de Regressão, treinado com a Regra Delta.
3.  **`notebooks/TC1/2.03-ljlb-regressao-MLP_H1.ipynb`**: Implementa e avalia a MLP com uma camada oculta.
4.  **`notebooks/TC1/2.03-ljlb-regressao-MLP_H2.ipynb`**: Implementa e avalia a MLP com duas camadas ocultas.

Cada notebook é autônomo e gera as análises solicitadas no enunciado, incluindo:
- Histograma dos resíduos.
- Gráfico de dispersão do valor real vs. predito.
- Coeficiente de correlação e outras métricas de avaliação.

---

## Trabalho 2: Reconhecimento de Faces

### Objetivo

Desenvolver e avaliar um sistema de reconhecimento de faces utilizando os classificadores: Mínimos Quadrados (MQ), Perceptron Logístico (PL) e MLP (1 e 2 camadas). O trabalho explora a extração de características com Análise de Componentes Principais (PCA) e o impacto de diferentes etapas de pré-processamento.

### Setup de Dados

1.  **Baixe o dataset Yale A.**
2.  Crie o diretório `data/raw/Kit_projeto_FACES/`.
3.  Descompacte o conteúdo do dataset dentro deste diretório. O código está preparado para ler imagens organizadas em subpastas por sujeito (ex: `data/raw/Kit_projeto_FACES/subject01/...`) ou em um formato plano onde os arquivos são nomeados como `subject01.centerlight`.

### Como Executar

Os experimentos são automatizados por meio de scripts na pasta `src/`. Os resultados (tabelas e figuras) são salvos automaticamente no diretório `results/`.

#### Atividades 1 a 4: Seleção de Escala e Comparativo Sem/Com PCA "Rotate"

Execute o script para realizar a busca de hiperparâmetros e avaliar os modelos sem PCA e com PCA no modo "rotate" (sem redução de dimensão).

```bash
python src/tc2_faces_A1_A4.py
```

**Saídas:**
- `results/tempo_escala_A1_A2.png`: Gráfico de tempo de treino por escala de imagem.
- `results/tabela1.csv`: Resultados da avaliação dos modelos sem PCA.
- `results/tabela2.csv`: Resultados da avaliação com PCA "rotate".

#### Atividades 5 e 6: Redução de Dimensionalidade com PCA

Execute o script para determinar a dimensão `q` ideal para o PCA e reavaliar os modelos.

```bash
python src/tc2_faces_A5_A6.py
```

**Saídas:**
- `results/pca_variance_explained_A3.png`: Gráfico da variância explicada acumulada.
- `results/pca_q_98.txt`: Arquivo de texto com a dimensão `q` escolhida para reter ≥98% da variância.
- `results/tabela3.csv`: Resultados da avaliação com PCA e redução de dimensionalidade.

#### Atividade 7: Pipeline com Transformação de Box-Cox

Execute o script para avaliar o pipeline completo, adicionando a transformação de Box-Cox após o PCA.

```bash
python src/tc2_faces_A7.py
```

**Saídas:**
- `results/tabela3_boxcox.csv`: Resultados da avaliação com o pipeline PCA + Box-Cox.
- `results/tabela3_comparativo_A7.csv`: Tabela comparativa dos resultados com e sem Box-Cox.
- `results/comparativo_A7_acc.png`: Gráfico comparativo de acurácia.

#### Atividade 8: Cenário de Controle de Acesso (Intruso)

**Setup Extra:** Para este cenário, adicione as imagens do "intruso" ao dataset. O script `trabalho_ic_aplicada/dataset_faces.py` identifica como intruso qualquer classe (subpasta) cujo nome **não** comece com "subject".

Após adicionar os dados do intruso, execute o script:

```bash
python src/tc2_faces_A8.py
```

**Saídas:**
- `results/tabela4_intruso.csv`: Tabela com as métricas de desempenho para o cenário de controle de acesso, incluindo taxa de falsos negativos (FNR) e falsos positivos (FPR).