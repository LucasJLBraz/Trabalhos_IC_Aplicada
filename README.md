# Trabalhos de Inteligência Computacional Aplicada

> Este repositório contém as implementações, experimentos e análises para os trabalhos da disciplina de Inteligência Computacional Aplicada. O projeto aborda dois problemas principais: regressão de preços de imóveis e reconhecimento de faces.

## 📖 Índice

- [Trabalhos de Inteligência Computacional Aplicada](#trabalhos-de-inteligencia-computacional-aplicada)
  - [📖 Índice](#-índice)
  - [🎯 Visão Geral](#-visão-geral)
  - [📂 Estrutura do Repositório](#-estrutura-do-repositório)
  - [🚀 Como Reproduzir os Resultados](#-como-reproduzir-os-resultados)
    - [1. Configuração do Ambiente](#1-configuração-do-ambiente)
    - [2. Execução dos Experimentos](#2-execução-dos-experimentos)
      - [Trabalho 1: Regressão de Valores de Imóveis](#trabalho-1-regressão-de-valores-de-imóveis)
      - [Trabalho 2: Reconhecimento de Faces](#trabalho-2-reconhecimento-de-faces)
        - [Opção A: Execução Automatizada (Recomendado)](#opção-a-execução-automatizada-recomendado)
        - [Opção B: Execução Manual por Atividade](#opção-b-execução-manual-por-atividade)
  - [📊 Resultados e Relatórios](#-resultados-e-relatórios)

## 🎯 Visão Geral

Este projeto é dividido em duas partes principais (Trabalhos 1 e 2):

1.  **Trabalho 1 (TC1):** Focado em **Regressão**. Foram implementados e avaliados quatro modelos para prever o valor de imóveis:
    *   Regressão Linear Múltipla (via Mínimos Quadrados - MQ)
    *   Perceptron de Regressão (treinado com Regra Delta)
    *   Multi-Layer Perceptron (MLP) com 1 camada oculta
    *   Multi-Layer Perceptron (MLP) com 2 camadas ocultas

2.  **Trabalho 2 (TC2):** Focado em **Classificação e Reconhecimento de Faces**. O objetivo foi desenvolver um sistema de reconhecimento facial, explorando:
    *   Classificadores: Mínimos Quadrados (MQ), Perceptron Logístico (PL) e MLP.
    *   Extração de características com Análise de Componentes Principais (PCA).
    *   Impacto de pré-processamento, como a transformação de Box-Cox.
    *   Análise de um cenário de segurança com detecção de "intrusos".

## 📂 Estrutura do Repositório

O projeto está organizado da seguinte forma para garantir clareza e reprodutibilidade:

```
/
├── data/                 # Datasets (brutos, processados, etc.).
├── notebooks/            # Jupyter Notebooks para análises e experimentos.
├── references/           # Enunciados, artigos e material de apoio.
├── results/              # Saídas dos experimentos (tabelas .csv, figuras .png).
├── src/                  # Scripts Python executáveis para os experimentos.
├── trabalho_ic_aplicada/ # Pacote Python principal do projeto.
│   ├── models/           # Implementações dos modelos e algoritmos.
│   │   ├── clf_mqo.py: Classificador por Mínimos Quadrados (`LeastSquaresClassifier`).
│   │   ├── clf_pl.py: Classificador por Perceptron Logístico (`SoftmaxRegression`).
│   │   ├── clf_mlp.py: Classificador Multi-Layer Perceptron (`MLPClassifier`).
│   │   ├── reg_perceptron.py: Funções para Regressão com Perceptron (estilo procedural).
│   │   ├── ridge_reg_linear_MQO.py: Funções para Regressão Ridge (estilo procedural).
│   │   ├── reg_mlp.py: Funções para Regressão com MLP (estilo procedural).
│   │   ├── reg_mlp_class.py: Classe para Regressão com MLP (`MLPRegressor`).
│   │   ├── pca_np.py: Implementação do PCA com NumPy (`PCA_np`).
│   │   ├── optim.py: Otimizadores (Adam, SGD, etc.) para treino de redes neurais.
│   │   ├── preprocess_np.py: Classes para pré-processamento (normalização Z-Score, MinMax).
│   │   └── aux.py: Funções auxiliares para métricas, plots e validação cruzada.
│   ├── dataset_faces.py  # Módulos de carregamento de dados de faces.
├── requirements.txt      # Dependências do projeto Python.
├── run_all_experiments.sh # Script para executar todos os experimentos do TC2.
└── README.md             # Este arquivo.
```

## 🚀 Como Reproduzir os Resultados

Siga os passos abaixo para configurar o ambiente e executar os experimentos.

### 1. Configuração do Ambiente

1.  **Clone o repositório:**
    ```bash
    # Substitua pela URL do seu repositório
    git clone https://github.com/seu-usuario/Trabalhos_IC_Aplicada.git
    cd Trabalhos_IC_Aplicada
    ```

2.  **Crie e ative um ambiente virtual:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    # No Windows, use: .venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Setup dos Dados (Apenas para o Trabalho 2):**
    *   Baixe o dataset **Yale A**.
    *   Crie o diretório: `data/raw/Kit_projeto_FACES/`.
    *   Descompacte o conteúdo do dataset neste diretório. O código espera que as imagens de cada pessoa estejam em subpastas (ex: `subject01`, `subject02`, etc.).

### 2. Execução dos Experimentos

#### Trabalho 1: Regressão de Valores de Imóveis

Os experimentos do TC1 estão nos Jupyter Notebooks e devem ser executados manualmente. O dataset é baixado automaticamente.

Abra e execute as células dos notebooks na pasta `notebooks/TC1/` na seguinte ordem:

1.  `0.03-ljlb-regressao-multipla-MQO.ipynb` (Modelo MQ)
2.  `1.03-ljlb-regressao-perceptron-simples.ipynb` (Perceptron de Regressão)
3.  `2.03-ljlb-regressao-MLP_H1.ipynb` (MLP com 1 camada oculta)
4.  `2.03-ljlb-regressao-MLP_H2.ipynb` (MLP com 2 camadas ocultas)

#### Trabalho 2: Reconhecimento de Faces

Para o TC2, você pode executar tudo de uma vez ou cada script individualmente.

##### Opção A: Execução Automatizada (Recomendado)

Use o script `run_all_experiments.sh` para executar todas as atividades (A1 a A8) em sequência.

```bash
bash run_all_experiments.sh
```

Alternativamente, você pode usar o `Makefile`:

```bash
make all
```

##### Opção B: Execução Manual por Atividade

Execute os scripts na pasta `src/` na ordem desejada. Cada script corresponde a uma ou mais atividades do enunciado.

-   **Atividades 1-4 (Comparativo com/sem PCA "rotate"):**
    ```bash
    python src/tc2_faces_A1_A4.py
    ```

-   **Atividades 5-6 (PCA com redução de dimensão):**
    ```bash
    python src/tc2_faces_A5_A6.py
    ```

-   **Atividade 7 (Pipeline com Box-Cox):**
    ```bash
    python src/tc2_faces_A7.py
    ```

-   **Atividade 8 (Cenário com Intruso):**
    *   **Setup extra:** Adicione as imagens do "intruso" ao dataset. O script `trabalho_ic_aplicada/dataset_faces.py` identifica como intruso qualquer classe (subpasta) cujo nome **não** comece com "subject".
    *   Execute o script:
    ```bash
    python src/tc2_faces_A8.py
    ```

## 📊 Resultados e Relatórios

-   **Saídas dos Experimentos:** Todas as tabelas (`.csv`) e figuras (`.png`) geradas pelos scripts são salvas no diretório `/results`, organizadas por trabalho.
-   **Relatório Final:** O relatório acadêmico está sendo desenvolvido em LaTeX. Os arquivos-fonte (`.tex`) e o PDF final podem ser encontrados em `/relatorio`
