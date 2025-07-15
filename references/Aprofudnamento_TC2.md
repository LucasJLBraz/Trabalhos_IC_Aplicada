### Análise Estruturada do Projeto 2

O fio condutor do projeto é avaliar como diferentes técnicas de pré-processamento (normalização, PCA para descorrelação e PCA para redução de dimensionalidade) afetam o desempenho de um conjunto de classificadores lineares e não-lineares.

---

#### **Atividades 1 e 2: O Ponto de Partida (Baseline)**

* **O que é pedido, exatamente?**
    1.  **Atividade 1:** Você deve executar o script `face_preprocessing_column.m` para converter as imagens em vetores de atributos (pixels brutos), **sem aplicar PCA**. A dimensão desses vetores será, por exemplo, $20 \times 20 = 400$.
    2.  **Atividade 2:** Utilizando esses dados brutos, você deve treinar e testar os quatro classificadores (MQ, PL, MLP-1H, MLP-2H). O processo deve ser repetido 50 vezes (`Nr=50`) com divisões aleatórias de treino (80%) e teste (20%)[cite: 11, 13]. Finalmente, você deve preencher a **Tabela 1** com as estatísticas de desempenho (taxa de acerto) [cite: 16-21].
    3.  **Nota Crítica (OBS-2):** Antes de preencher a tabela, você deve realizar uma mini-otimização para cada classificador, testando diferentes métodos de normalização, funções de ativação e variantes do gradiente descendente. A tabela deve conter o resultado da **melhor configuração** encontrada para cada modelo.

* **Por que isso é importante?**
    Esta etapa estabelece a **linha de base (baseline)**. Ela responde à pergunta: "Qual a performance dos classificadores diretamente nos dados de pixels, que são altamente correlacionados e de alta dimensão?". O resultado aqui será o seu principal ponto de comparação para todas as atividades subsequentes.

* **Como "Walk the Extra Mile"?**
    * **Documente a Otimização:** Não se limite a apresentar o resultado final na tabela. Em seu relatório, inclua uma seção breve descrevendo o processo de busca de hiperparâmetros. Por exemplo: "Para a MLP-2H, foram testadas as funções de ativação ReLU e Tanh. A normalização Z-score foi comparada com a normalização Min-Max. A configuração que apresentou a maior média de acurácia no teste foi a de [especificar configuração], sendo este o resultado reportado na Tabela 1". Isso demonstra rigor metodológico.
    * **Analise o Tempo de Execução:** A tabela pede o tempo de execução. Compare-os. Por que o MQ é ordens de magnitude mais rápido que as MLPs? Relacione isso à sua natureza analítica (solução de um sistema de equações) versus a natureza iterativa (gradiente descendente) das redes neurais [cite: 6414-6415, 7723].

---

#### **Atividades 3 e 4: O Efeito da Descorrelação (PCA sem Redução)**

* **O que é pedido, exatamente?**
    Você deve reprocessar os dados, desta vez **ativando a PCA** no script, mas garantindo que **não haja redução de dimensionalidade** ($q = p$, onde $p$ é a dimensão original, e.g., 400)[cite: 28, 29]. Em seguida, repita o treinamento e teste dos classificadores e preencha a **Tabela 2**. Por fim, compare os resultados com a Tabela 1.

* **Por que isso é importante?**
    Esta é uma etapa conceitual chave. Ela isola um dos efeitos da PCA: a **rotação do espaço de atributos para um novo sistema de coordenadas onde os eixos (componentes principais) são descorrelacionados**. A informação não é perdida, apenas re-representada. A questão a ser respondida é: "A simples remoção da correlação entre os atributos melhora o desempenho dos classificadores?".
    * **Fundamentação Teórica:** O material `PCA_transformation_MAR2025.pdf` explica que a transformação PCA resulta em uma matriz de covariância diagonal, indicando que as novas variáveis (componentes) são descorrelacionadas [cite: 6998-7022].

* **Como "Walk the Extra Mile"?**
    * **Verifique a Descorrelação:** Calcule e visualize (como um mapa de calor) a matriz de covariância dos dados antes e depois desta etapa. A matriz original terá valores significativos fora da diagonal (indicando alta correlação entre pixels vizinhos). A matriz após a PCA deve ser diagonal, provando que você compreende o que o algoritmo fez.
    * **Análise Diferenciada:** Analise por que o desempenho de certos modelos pode (ou não) ter melhorado. Modelos lineares como o MQ podem ser particularmente sensíveis à multicolinearidade, então a descorrelação pode ter um impacto positivo. As MLPs, por sua capacidade de aprender transformações não-lineares, talvez sejam menos afetadas. Sua análise deve refletir essa nuance.

---

#### **Atividades 5 e 6: O Efeito da Redução de Dimensionalidade (PCA com Compressão)**

* **O que é pedido, exatamente?**
    1.  **Atividade 5:** Analise o gráfico de variância explicada acumulada (gerado na etapa anterior) e determine o número de componentes principais `q` necessários para reter **pelo menos 98%** da variância total dos dados.
    2.  **Atividade 6:** Reprocesse os dados usando este novo valor de `q < p`, efetivamente comprimindo a informação. Treine e teste os classificadores nos dados de dimensão reduzida e preencha a **Tabela 3**. Compare os resultados com a Tabela 2[cite: 45, 46].

* **Por que isso é importante?**
    Esta é a aplicação mais comum da PCA. O objetivo é testar a hipótese de que a maior parte da informação útil para a classificação reside em um subespaço de menor dimensão.
    * **Benefícios Potenciais:** Reduzir a dimensionalidade pode diminuir drasticamente o tempo de treinamento, mitigar o risco de *overfitting* (a "maldição da dimensionalidade") e melhorar a generalização ao remover componentes que representam principalmente ruído.

* **Como "Walk the Extra Mile"?**
    * **Visualize as Eigenfaces:** Os vetores próprios (as componentes principais) podem ser remodelados para as dimensões originais da imagem, criando as "eigenfaces". Visualize as primeiras e as últimas eigenfaces. As primeiras capturam as variações mais importantes (iluminação, traços gerais), enquanto as últimas parecem-se mais com ruído. Apresentar isso no seu relatório é uma forma visualmente poderosa de explicar o que o PCA está fazendo.
    * **Curva de Desempenho vs. Dimensão:** Em vez de apenas escolher `q` para 98%, crie um gráfico mostrando a taxa de acerto média (e o tempo de treinamento) em função da porcentagem de variância retida (e.g., para 80%, 90%, 95%, 98%, 99%, 100%). Isso mostrará o *trade-off* entre compressão, acurácia e custo computacional.

---

#### **Atividades 7 e 8: Refinamento Adicional e Aplicação Prática**

* **Atividade 7: Transformação Box-Cox:**
    * **O que é pedido?** Adicionar uma transformação Box-Cox aos dados após a PCA (com redução de dimensão) e antes da normalização z-score. O objetivo é avaliar se forçar os componentes principais a terem uma distribuição mais próxima da Gaussiana melhora o desempenho.
    * **Por que isso é importante?** PCA garante descorrelação, não gaussianidade. Esta etapa investiga se a normalização da *distribuição* dos seus novos atributos, e não apenas da sua *escala*, beneficia os classificadores.
    * **Extra Mile:** Pesquise e explique brevemente como o parâmetro $\lambda$ da transformação de Box-Cox é tipicamente otimizado. Plote o histograma de uma ou duas das principais componentes *antes* e *depois* da transformação para demonstrar visualmente seu efeito normalizador.

* **Atividade 8: Controle de Acesso (Detecção de Intruso):**
    * **O que é pedido?** Modificar o problema de classificação de 15 classes para um problema de classificação binária: "usuário autorizado" (qualquer uma das 15 pessoas do Yale A) vs. "intruso" (11 imagens suas). Você deve usar a pipeline completa (PCA + Box-Cox + Normalização) e calcular métricas de desempenho mais adequadas para este cenário, como taxa de falsos negativos/positivos, sensibilidade e precisão.
    * **Por que isso é importante?** Esta é a etapa mais aplicada. Ela simula um problema real de segurança, que é inerentemente desbalanceado. A acurácia, sozinha, é uma métrica pobre aqui. Se 99% das tentativas de acesso são de usuários válidos, um classificador que sempre diz "sim" terá 99% de acurácia, mas será inútil. É por isso que métricas como falsos positivos e negativos são essenciais.
    * **Extra Mile:**
        * **Matriz de Confusão:** Apresente a matriz de confusão para este problema binário. Ela é a fonte para o cálculo de todas as métricas solicitadas e oferece uma visão completa do desempenho.
        * **Curva ROC e AUC:** Para os modelos que fornecem uma saída contínua antes da decisão final (como PL e MLP), plote a **Curva ROC** (Receiver Operating Characteristic) e calcule a **AUC** (Area Under the Curve). Esta é a métrica padrão para avaliar classificadores binários, pois mostra o desempenho em todos os limiares de decisão possíveis.
        * **Análise de Custo do Erro:** Discuta o trade-off entre os tipos de erro. Em um sistema de controle de acesso, qual erro é mais custoso: um falso negativo (deixar um intruso entrar) ou um falso positivo (barrar um usuário autorizado)? A resposta a essa pergunta determinaria qual modelo é "o melhor" para esta aplicação específica, independentemente da acurácia pura.

Este projeto é uma excelente jornada pela pipeline de um problema de reconhecimento de padrões. Siga as atividades sequencialmente, analise criticamente cada resultado e aproveite as oportunidades para aprofundar a análise. Boa sorte!