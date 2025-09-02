# Relatório Metodológico Detalhado: Atividade 8

**Autor:** Gemini
**Data:** 28/08/2025
**Foco:** Análise e comparação profunda entre as abordagens de classificação binária (`tc2_faces_A8.py`) e detecção de anomalias unária (`tc2_faces_A8_unario.py`) para o problema de controle de acesso.

## 1. Visão Geral e Objetivo

A Atividade 8 aborda o problema mais prático e desafiador do projeto: **controle de acesso**, onde o sistema deve distinguir entre usuários autorizados e um "intruso". Este problema pode ser modelado de duas formas fundamentalmente diferentes, cada uma com suas próprias hipóteses, vantagens e desvantagens.

Este relatório analisa e contrasta as duas implementações:

1.  **Abordagem Supervisionada (Binária):** Implementada em `tc2_faces_A8.py`, trata o intruso como uma nova classe a ser aprendida.
2.  **Abordagem Não Supervisionada (Unária):** Implementada em `tc2_faces_A8_unario.py`, trata o intruso como uma anomalia a ser detectada.

## 2. Análise da Abordagem 1: Classificação Supervisionada (Binária)

-   **Script:** `tc2_faces_A8.py`
-   **Filosofia:** "Aprenda a aparência dos usuários autorizados E a aparência do intruso, e depois separe os dois grupos."

### 2.1. Metodologia Detalhada

1.  **Tratamento dos Dados:** O conjunto de dados é formado por 16 classes: 15 classes de sujeitos autorizados e 1 classe de "intruso".
2.  **Pipeline:** O script aplica o pipeline `PCA (redução) -> Box-Cox -> Z-Score` a todos os dados.
3.  **Treinamento:** Os classificadores (MQ, PL, MLP) são treinados para resolver um problema de classificação de 16 classes.
4.  **Avaliação:** Após a predição, as classes são binarizadas: a predição da classe "intruso" é considerada um resultado positivo, e a predição de qualquer uma das 15 classes autorizadas é um resultado negativo. As métricas (FNR, FPR, etc.) são calculadas com base nessa binarização.
5.  **Otimização:** A busca de hiperparâmetros seleciona o modelo que maximiza o **F1-Score da classe intruso**, focando no equilíbrio entre detectar o intruso (recall) e não gerar falsos alarmes (precisão).

### 2.2. Análise Crítica e Resultados Esperados

-   **Ponto Forte:** A metodologia é direta e utiliza as ferramentas de classificação padrão de forma eficaz. Se o objetivo é criar um sistema que é excelente em detectar **aquele intruso específico** que foi usado no treino, esta abordagem terá um desempenho altíssimo.
-   **Falha Conceitual (Vazamento de Informação para o Mundo Real):** A principal fraqueza desta abordagem é que ela **superespecializa (overfits) no conceito de "intruso" que ela viu**. O modelo não aprende a detectar "estranhos" em geral; ele aprende a detectar as características específicas das 11 imagens fornecidas. 
-   **Resultados Esperados:**
    -   **Taxa de Falsos Negativos (FNR) muito baixa, próxima de zero.** Como o modelo foi explicitamente treinado para reconhecer as imagens do intruso, é muito improvável que ele erre ao classificar as imagens de teste do mesmo intruso.
    -   **Generalização Pobre:** Se uma pessoa completamente diferente tentasse acessar o sistema, o desempenho seria imprevisível e provavelmente ruim. O modelo não possui um mecanismo para lidar com anomalias nunca vistas.

## 3. Análise da Abordagem 2: Detecção de Anomalias (Unária)

-   **Script:** `tc2_faces_A8_unario.py`
-   **Filosofia:** "Aprenda apenas a aparência dos usuários autorizados. Qualquer coisa que não se pareça com eles é, por definição, um intruso."

### 3.1. Metodologia Detalhada

1.  **Tratamento dos Dados:** O conjunto de dados de **treino** contém **exclusivamente** imagens dos 15 sujeitos autorizados. As imagens do intruso são usadas apenas nos conjuntos de validação e teste.
2.  **Pipeline:** O pipeline de pré-processamento (`PCA -> Box-Cox -> Z-Score`) é ajustado apenas nos dados de treino autorizados.
3.  **Treinamento:** Os modelos (Autoencoder, One-Class SVM, etc.) aprendem um modelo de "normalidade" a partir dos dados de treino autorizados.
4.  **Score de Anomalia:** Em vez de prever uma classe, os modelos geram um **score de anomalia** contínuo para cada nova imagem.
5.  **Otimização e Avaliação:** Conforme descrito no relatório `A8_unario_report.md`, o processo usa um conjunto de validação para otimizar hiperparâmetros e um limiar de decisão, e um conjunto de teste isolado para a avaliação final, com um limiar agnóstico.

### 3.2. Análise Crítica e Resultados Esperados

-   **Ponto Forte:** Esta abordagem é **conceitualmente robusta e generalizável**. Ela é projetada para o cenário do mundo real, onde a aparência do próximo intruso é desconhecida. O sistema aprende a detectar desvios da norma, não a reconhecer um inimigo específico.
-   **Desafio:** A tarefa é inerentemente mais difícil. O modelo precisa definir uma fronteira precisa em torno do que é "normal" sem nunca ter visto um exemplo do que é "anormal".
-   **Resultados Esperados:**
    -   **Taxa de Falsos Negativos (FNR) maior que zero.** É esperado e realista que o modelo possa cometer erros, classificando um intruso como autorizado se ele for suficientemente parecido com um dos usuários do sistema. Um FNR de 0% seria suspeito e indicaria overfitting.
    -   **Trade-off FNR vs. FPR:** A análise se concentrará no equilíbrio entre FNR (deixar um intruso entrar) e FPR (negar acesso a um usuário legítimo). A otimização do limiar é crucial aqui.
    -   **Comparativo de Modelos:** A avaliação permitirá comparar qual tipo de modelo unário é mais eficaz: os baseados em reconstrução (PCA, Autoencoder) ou os baseados em densidade/fronteira (SVM, Isolation Forest).

## 4. Contraste Final e Conclusão

| Característica | Abordagem Supervisionada (Binária) | Abordagem Não Supervisionada (Unária) |
| :--- | :--- | :--- |
| **Hipótese Central** | O intruso é uma classe conhecida. | O intruso é uma anomalia desconhecida. |
| **Dados de Treino** | Autorizados + Intrusos | Apenas Autorizados |
| **Tarefa do Modelo** | Separar classes (Discriminativa) | Modelar a "normalidade" (Generativa/Fronteira) |
| **Métrica de Sucesso** | FNR artificialmente baixo no teste. | FNR realista, equilíbrio com FPR. |
| **Generalização** | **Baixa.** Falha com intrusos não vistos. | **Alta.** Potencial para detectar qualquer intruso. |
| **Robustez no Mundo Real**| **Fraca.** Sistema quebradiço. | **Forte.** Abordagem padrão para segurança. |

**Conclusão:** A abordagem **supervisionada** (`tc2_faces_A8.py`) cumpre uma tarefa de classificação acadêmica, mas resulta em um sistema de segurança falho. A abordagem **unária** (`tc2_faces_A8_unario.py`) é metodologicamente mais complexa e desafiadora, mas é a **única abordagem correta** para construir um sistema de controle de acesso que tenha alguma validade no mundo real. A comparação dos resultados de ambos os scripts ilustrará de forma prática o conceito de generalização de modelo e os perigos do vazamento de informação conceitual (treinar com dados que não estarão disponíveis em produção).
