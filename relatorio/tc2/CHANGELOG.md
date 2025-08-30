# Log de Modificações do Relatório Técnico (Trabalho 2)

Esta revisão sistemática abordou inconsistências, omissões e falta de rigor analítico da versão anterior, incorporando as sugestões das análises críticas para elevar o padrão do documento a um nível de pós-graduação.

---

### 1. Correções Estruturais e de Conteúdo Globais

*   **Padronização da Escala dos Experimentos:**
    *   **O que foi alterado:** Todo o relatório foi padronizado para refletir a escala real dos experimentos, que foi de **`30x30` pixels (`d=900`)**, e não `40x40` como mencionado anteriormente.
    *   **Justificativa:** Corrigida uma inconsistência fundamental entre o texto do relatório e os dados brutos (`.csv`), garantindo que a descrição corresponda à execução real dos testes.

*   **Aprimoramento da Metodologia:**
    *   **O que foi alterado:** A seção de "Metodologia Geral" foi refinada para declarar explicitamente que a métrica de seleção de hiperparâmetros foi o **F1-Score (macro)** e que a busca aleatória usou **100 amostras**.
    *   **Justificativa:** Aumenta a transparência e o rigor metodológico, alinhando-se às melhores práticas para problemas de classificação multiclasse e deixando claro o escopo da busca por hiperparâmetros.

*   **Seção de Reprodutibilidade:**
    *   **O que foi alterado:** A seção foi reformatada como uma lista para maior clareza, destacando a versão do Python/NumPy e o uso de sementes aleatórias.
    *   **Justificativa:** Facilita a vida de um leitor que queira de fato reproduzir os resultados, um requisito chave em trabalhos científicos.

---

### 2. Revisão por Atividade

*   **Atividades 1 e 2 (Conjunto Original):**
    *   **Tabela 1:** Completamente refeita para incluir **todos os hiperparâmetros vencedores** (taxa de aprendizado, épocas, etc.) e a métrica principal (F1-macro), corrigindo a principal falha de omissão do relatório anterior.
    *   **Texto:** A discussão foi ajustada para focar no F1-Score como métrica de decisão, e não apenas na acurácia.

*   **Atividades 3 e 4 (PCA como Rotação):**
    *   **Tabela 2:** Também refeita, incluindo todos os hiperparâmetros. O tempo de treino do modelo MQ, que antes era arredondado para `0.000`, agora é exibido em notação científica (`6.06e-4`) para refletir o valor real.
    *   **Texto:** A discussão foi corrigida para não afirmar que "todos" os modelos tiveram ganhos computacionais, reconhecendo que o PL teve um leve aumento no tempo de treino.

*   **Atividade 5 (Análise do PCA):**
    *   **Análise Qualitativa Adicionada:** A seção foi enriquecida com a inclusão da figura das **"eigenfaces"** (`eigenfaces_visualization.png`).
    *   **Texto:** Foi adicionado um parágrafo que interpreta as eigenfaces, explicando que os primeiros componentes capturam variações de iluminação e os posteriores, traços faciais, agregando profundidade à análise.

*   **Atividade 6 (PCA com Redução):**
    *   **Tabela 3:** Substituída pela versão completa e correta, incluindo todos os hiperparâmetros e corrigindo erros de nomenclatura na normalização.
    *   **Texto:** Discussão refinada para focar nos ganhos de eficiência e no desempenho relativo medido pelo F1-Score.

*   **Atividade 7 (PCA + Box-Cox):**
    *   **Tabelas 4 e 5:** Ambas foram refeitas com os dados corretos dos arquivos `.csv`, e a tabela comparativa agora foca na variação do F1-Score (`Δ F1`).
    *   **Discussão Aprofundada:** O texto agora inclui a justificativa teórica para a falha da transformação Box-Cox, explicando que os componentes principais já tendem à normalidade (pelo Teorema do Limite Central), tornando a transformação adicional prejudicial.

*   **Atividade 8 (Detecção de Intrusos):**
    *   **Tabela 8:** Substituída pela versão correta e mais enxuta, focando nas métricas essenciais de controle de acesso.
    *   **Figura da Curva ROC:** A imagem de exemplo do intruso foi substituída pela **curva ROC** (`roc_curve_A8.png`), que é a visualização padrão e mais informativa para este tipo de problema.
    *   **Texto:** A metodologia agora cita explicitamente a divergência entre as 10 imagens de intrusos usadas e as 11 solicitadas no edital. A discussão foi reescrita para focar no F1-Score do intruso e interpretar a curva ROC.
