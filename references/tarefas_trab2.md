Perfeito! Vou “mastigar” o **Projeto 2 — Reconhecimento de Faces** em um plano de tarefas bem claro, com checklists por etapa. Onde for útil, cito exatamente o que o enunciado pede.

---

# Plano de execução (Kanban mastigado)

## 0) Preparação e organização

* **Crie a estrutura de pastas**:

  ```
  projeto_faces/
    dados/           # Yale A e suas 11 imagens de “intruso” (Ativ. 8)
    scripts/         # face_preprocessing_column.m, compara_todos.m etc.
    resultados/      # tabelas .csv, gráficos e logs
    relatorio/       # texto final com respostas às questões
  ```
* **Confirme o material do kit**: imagens (Yale A), scripts MATLAB/Octave para pré-processamento e PCA.&#x20;

---

## 1) Atividade 1 — Pré-processamento **sem PCA**

**Objetivo:** gerar vetores de atributos das imagens (redimensionadas), ainda sem PCA.

**Passos:**

* Abrir `face_preprocessing_column.m`.
* **Comentar as linhas 56–60** (desliga PCA).&#x20;
* **Escolher as dimensões de redimensionamento** na **linha 37** (ex.: `[20 20] → 400`, `[30 30] → 900`).&#x20;
* **Executar** e salvar os vetores de atributos (treino/teste) em `resultados/`.

Checklist:

* [ ] Linhas 56–60 comentadas
* [ ] Dimensão escolhida e anotada
* [ ] Features vetorizadas salvas

---

## 2) Atividade 2 — Comparar classificadores (sem PCA)

**Objetivo:** rodar 50 repetições com **Ptrain = 80** e preencher **Tabela 1**.

**Passos:**

* Abrir `compara_todos.m`.
* Setar **Ptrain = 80** e **Nr = 50**.&#x20;
* Garantir que os classificadores avaliados incluam **MQ, PL, MLP-1H e MLP-2H**.&#x20;
* **Métricas por rodada:** taxa de acerto; ao final, **média, desvio-padrão, mínimo, máximo, mediana** + **tempo de execução** (use `tic/toc`).&#x20;
* **Testar variações e escolher a “melhor versão” de cada classificador** (normalizações, ativações, variantes de gradiente). Na tabela, registre **só a melhor** de cada um.&#x20;

**O que testar (rápido checklist):**

* [ ] Sem normalização / **z-score** / **\[0, +1]** / **\[−1, +1]**&#x20;
* [ ] Ativações (sigmoidais, ReLU etc.) e variantes de GD (SGD, Momentum, Adam se disponível)&#x20;

**Entregáveis:**

* [ ] **Tabela 1** preenchida (melhor versão de cada classificador)&#x20;
* [ ] **Questão 1 e 2** respondidas (comparar desempenhos; melhor em acerto e em tempo).&#x20;

---

## 3) Atividade 3 — PCA **sem redução** (somente descorrelação)

**Objetivo:** aplicar PCA apenas para **diagonalizar** a covariância (mesma dimensão original).

**Passos:**

* Em `face_preprocessing_column.m`, **descomentar as linhas 56–60** (liga PCA).
* Definir **q = 400** ou **q = 900** (coerente com a dimensão escolhida na Ativ. 1).&#x20;

Notas:

* Aqui **não há redução dimensional**; só **descorrelação** dos atributos (base Z).&#x20;

Checklist:

* [ ] Linhas 56–60 descomentadas
* [ ] q ajustado (= dimensão original)

---

## 4) Atividade 4 — Repetir a Ativ. 2 **com PCA (sem redução)**

**Objetivo:** repetir o protocolo da Ativ. 2 e preencher a **Tabela 2**.

**Passos:**

* Rodar `compara_todos.m` novamente (Ptrain=80; Nr=50) agora **com PCA ativa**.&#x20;
* Preencher **Tabela 2** e **comentar mudanças** vs. Tabela 1 (**Questão 4**).&#x20;

Entregáveis:

* [ ] **Tabela 2** completa
* [ ] **Questão 4** respondida

---

## 5) Atividade 5 — Escolher **q** para ≥ **98%** da variância

**Objetivo:** reduzir a dimensão com PCA mantendo ≥98% da variância.

**Passos:**

* Usar a **curva de variância explicada acumulada** para escolher **q** que preserve ≥98%.&#x20;
* Conferir no vetor **VEq** a componente cuja acumulada supera **98%**.&#x20;
* Executar o script com esse **q** (agora há **redução dimensional** + **descorrelação**).&#x20;
* Registrar o valor de **q** escolhido (**Questão 5**).&#x20;

Entregáveis:

* [ ] **q** documentado e justificado (gráfico + linha de corte)
* [ ] **Questão 5** respondida

---

## 6) Atividade 6 — Avaliar com **PCA reduzida**

**Objetivo:** repetir comparação (Ptrain=80; Nr=50) usando PCA com redução e preencher **Tabela 3**.

**Passos:**

* Treinar e avaliar os 4 classificadores com o novo conjunto (PCA reduzida).&#x20;
* Preencher a **Tabela 3** (média, min, máx, mediana, desvio, tempo).&#x20;
* Discutir efeitos da **redução de dimensionalidade** nos desempenhos (**Questão 6**).&#x20;

Entregáveis:

* [ ] **Tabela 3** completa
* [ ] **Questão 6** respondida

---

## 7) Atividade 7 — **Box-Cox** + **z-score** após PCA (com redução)

**Objetivo:** investigar se Box-Cox (seguido de z-score) melhora os resultados.

**Passos:**

* Aplicar **Box-Cox** nos dados transformados por PCA, depois **z-score**.&#x20;
* **Repetir** a avaliação da Ativ. 6 e comparar com a Tabela 3 (**Questão 7**).&#x20;

Observação técnica: Box-Cox requer valores **positivos**; se necessário, aplicar deslocamento constante >|mínimo| antes, só para a transformação, e registrar isso no relatório.

Entregáveis:

* [ ] Tabela comparativa (com/sem Box-Cox)
* [ ] **Questão 7** respondida

---

## 8) Atividade 8 — **Controle de acesso** com “intruso”

**Objetivo:** pipeline final para acesso/negação (autorizado vs. intruso).

**Passos:**

* Pipeline: **Imagens vetorizadas → PCA → Box-Cox → z-score → Classificador**.&#x20;
* **Adicionar 11 imagens próprias** como **intruso** (não cadastrado).&#x20;
* Rodar **50 rodadas** e calcular **médias + desvios** dos índices:
  **Acurácia**, **taxa de falsos negativos** (*“pessoas às quais acesso foi permitido incorretamente”*), **taxa de falsos positivos** (*“pessoas às quais acesso não foi permitido incorretamente”*), **sensibilidade** e **precisão**.&#x20;

Dica prática:

* Deixe **claro no relatório** que você seguiu **as definições do enunciado** para FN/FP nesta aplicação de controle de acesso (mesmo que a convenção usual varie por quem é a “classe positiva”).

Entregáveis:

* [ ] Tabela de métricas (média ± desvio) por classificador
* [ ] Discussão dos melhores p/ uso em controle de acesso

---

# O que entregar no relatório (sugestão de estrutura)

1. **Descrição do setup** (dimensões escolhidas; q’s; normalizações; Box-Cox).
2. **Tabelas 1, 2, 3** + gráficos de apoio (histogramas de acertos; curva de variância explicada).  &#x20;
3. **Respostas** às **Questões 1, 2, 4, 5, 6, 7, 8** (referenciando as tabelas/figuras).     &#x20;
4. **Discussão crítica**: efeito de normalização/ativação/otimizador; impacto da PCA (sem/ com redução); impacto do Box-Cox; análise de tempo vs. acurácia (trade-off).
5. **Conclusão**: qual configuração é mais robusta e prática para **controle de acesso**.

---

## Boas práticas para ganhar pontos (e economizar tempo depois)

* **Logue tudo**: salve CSV por rodada e uma tabela “top-config” por classificador.
* **Seed reprodutível** para particionamento/treino.
* **Mensure tempo** por rodada/classificador (tic/toc) e reporte na tabela, como o enunciado pede.&#x20;
* **Confusion matrix** no cenário de controle de acesso (entendimento visual de FN/FP definidos pelo enunciado).&#x20;

---

Se quiser, já te entrego um **checklist editável** (Markdown/CSV) e **templates de tabela** para você só colar os números das 50 rodadas. Também posso esboçar um **script-base** (MATLAB/Octave) para consolidar métricas e gerar as Tabelas 1–3 automaticamente.
