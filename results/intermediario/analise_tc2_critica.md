Per seu pedido, segue uma **revisão dura e criteriosa**, com foco em lógica, nomenclatura, conceitos, hipóteses e afirmações. Apontei apenas problemas que podem ser corrigidos **no texto/tabelas/figuras** (ou com microajustes de script para exportação), **sem novos experimentos**. Indico a seção/trecho, explico por que é problemático academicamente e proponho correção objetiva. No fim, sintetizo e atribuo **nota (0–10)**.

> Base: versão **Trabalho\_2\_ICAp\_\_\_SBC-3.pdf** e o enunciado oficial **TC2\_PPGETI\_2025.1**. &#x20;

---

## 1) Resumo e Introdução

**\[P-1] “Promessa de métricas” vs. o que aparece nas tabelas**

* **Onde**: Resumo e Introdução prometem relatar **acurácia, precisão, recall, F1 e tempos** “para cada atividade”. Em várias tabelas centrais (p.ex., Tabelas 1–4) os tempos e a acurácia recebem destaque, mas **precisão/recall/F1 (macro)** aparecem ora ausentes na coluna, ora apenas no texto; a promessa fica **ambígua** para um leitor externo.&#x20;
* **Por que é problema**: coerência entre objetivos e evidências é requisito básico em avaliação de mestrado; prometer algo “em todas as atividades” e, na prática, destacar só parte das métricas abre espaço para contestação.
* **Correção**: ou **padronize os cabeçalhos** para incluir “Precisão (macro) / Recall (macro) / F1 (macro)” em todas as tabelas resumidas, ou acrescente no início de Resultados: *“Para brevidade, as Tabelas reportam acurácia e tempo; as demais métricas macro estão nos CSVs em anexo, e são discutidas nos parágrafos seguintes.”* (sem reexecutar nada).

**Ponto forte**: A introdução estrutura bem as decisões do pipeline (escala, PCA como rotação e redução, Box–Cox) e antecipa o protocolo estatístico.&#x20;

---

## 2) Metodologia Geral

**\[P-2] Ordem PCA → normalização precisa ficar explícita e uniforme**

* **Onde**: Metodologia geral e A6 já dizem corretamente que **PCA é ajustada no treino e aplicada antes da normalização**. Em passagens anteriores (A1–A2) a redação ainda permite leitura de normalização “pré-PCA”.&#x20;
* **Por que é problema**: a ordem altera a geometria e o condicionamento; ambiguidade metodológica é crítica em banca.
* **Correção**: incluir uma frase normativa no início da Metodologia: *“Sempre que há PCA, ela é **fitada no treino** e **aplicada antes** de qualquer normalização subsequente.”*

**\[P-3] Notação matemática inconsistente (“R1600” / “d = 1,600”)**

* **Onde**: A6/A8 usam “**R1600**” e “**d = 1,600**” (vírgula como separador de milhar), o que é estranho para texto técnico.&#x20;
* **Por que é problema**: notação inconsistente quebra o padrão acadêmico; o correto seria $\mathbb{R}^{1600}$ (ou “dimensão 1600”).
* **Correção**: padronizar para “$\mathbb{R}^{1600}$” no corpo do texto ou “dimensão 1600” nas partes não LaTeX.

**Ponto forte**: O desenho estatístico está claro (60 amostras de random search × 10 reavaliações × 50 repetições), oferecendo lastro de 650 treinos por modelo/cenário.&#x20;

---

## 3) Atividades 1–2 (sem PCA)

**\[P-4] Tabela 1 – formatação e alinhamento de colunas**

* **Onde**: Tabela 1 traz em algumas células elementos que **deveriam estar na coluna correta** (p.ex., “(10⁻⁴)” aparece junto da Acurácia do PL).&#x20;
* **Por que é problema**: mistura de hiperparâmetro na coluna de métrica prejudica leitura e sugere erro de *export*.
* **Correção**: revisar a função que monta a tabela (ou editar a tabela no PDF) garantindo: “λℓ2” **só** na coluna de regularização; “–” para campos não aplicáveis.

**Ponto forte**: Discussão coesa: PL melhor compromisso acurácia/tempo; MLPs competitivas porém mais caras em treino.&#x20;

---

## 4) Atividades 3–4 (PCA como rotação)

**\[P-5] Contradição textual sobre tempo após PCA-rotate**

* **Onde**: Seção 4.3 abre com “**ganhos computacionais para todos**”; porém, na 4.2 e na própria tabela, o **PL fica mais lento** (0,029 s → 0,040 s). Há também um trecho **truncado**: “reduziram seus tempos de treino em **mais de 40** … MLP-2H de 0,612 s para 0,529 s”. Falta “%”.&#x20;
* **Por que é problema**: inconsistência interna é erro lógico; cortar a unidade (“%”) deixa a frase sem sentido.
* **Correção**: “A PCA-rotate trouxe **reduções** marcantes em MQ e MLPs; **no PL houve leve **aumento** (0,029 s→0,040 s). As MLPs reduziram \~**14–40 %\*\* do tempo, e a MLP-2H foi de 0,612 s para 0,529 s.” Ajustar a primeira frase de 4.3 para não dizer “para todos”.

**\[P-6] Tabela 2 – zero arredondado esconde ordem de grandeza**

* **Onde**: Tempo do MQ aparece como **0,000 s**; no texto você cita **3,3×10⁻⁴ s**.&#x20;
* **Por que é problema**: arredondar a zero apaga a informação (parece “sem custo”).
* **Correção**: usar notação científica ou 6 casas para tempos sub-ms (ex.: 0,00033 s).

**Ponto forte**: Interpretação correta: rotação decorrela e condiciona; redes costumam se beneficiar mais que lineares.&#x20;

---

## 5) Atividade 5 (escolha de $q$) e 6 (PCA reduzida)

**\[P-7] Figura da variância explicada – vestígio de placeholder**

* **Onde**: Linha com “**fig\_variance\_explained.png**” no corpo do texto sugere *placeholder* (não a figura em si).&#x20;
* **Por que é problema**: relatório final não deve conter rótulos de arquivo; deve exibir a figura.
* **Correção**: inserir a figura gerada (curva com linha de 98% e $q=10$) e remover o nome do arquivo cru.

**\[P-8] Tabela 3 – erro de nomenclatura da normalização**

* **Onde**: “**min–max \[1,1]**” (várias linhas) é **conceitualmente incorreto**; intervalos válidos são **\[0,1]** ou **\[−1,1]**.&#x20;
* **Por que é problema**: é um erro duro de nomenclatura; um avaliador pode ler como “não sabe o que é min–max”.
* **Correção**: corrigir para **\[0,1]** ou **\[−1,1]** conforme o CSV do vencedor; se veio de *mapping* automático, ajuste a função de *stringify* antes de exportar.

**Ponto forte**: Justificativa de $q=10$ (≥98%) é clara e alinha-se ao enunciado. &#x20;

---

## 6) Atividade 7 (PCA + Box–Cox + z)

**\[P-9] Semântica de λ: texto vs. tabela**

* **Onde**: Texto usa **λ\_{BC}** (Box–Cox) corretamente; Tabela 4 traz **“λℓ2”** (regularização) — correto também. Mas a proximidade pode confundir.&#x20;
* **Por que é problema**: leitores podem achar que “λ” na tabela é o da Box–Cox.
* **Correção**: manter “λℓ2” explícito no cabeçalho (como está) e, no parágrafo antes da Tabela 4, incluir uma frase: *“Na Tabela 4, **λℓ2** refere-se à regularização do modelo; **os λ da Box–Cox são por componente e não são tabulados**.”*

**Ponto forte**: A explicação da **queda geral** com Box–Cox é tecnicamente sólida (componentes já \~gaussianos; distorção desnecessária).&#x20;

---

## 7) Conclusões e Reprodutibilidade

**\[P-10] Reprodutibilidade – versions/seeds pouco explícitas no PDF**

* **Onde**: Diz que tudo está no repositório; menciona “Python 3.11; NumPy 1.24; seed=42”, mas sem padronizar como bloco reprodutível.&#x20;
* **Por que é problema**: banca valoriza *artefato textual* auto-suficiente.
* **Correção**: inserir uma linha objetiva: *“Ambiente: Python 3.11; NumPy 1.24; sem scikit-learn nos resultados; semente global 42; sementes por repetição registradas nos scripts.”*

**Ponto forte**: Deixa claro que **nenhum resultado** usa *scikit-learn*; apenas verificação local — bom para escopo/ética experimental.&#x20;

---

## 8) Atividade 8 (intruso)

**\[P-11] Cardinalidade do intruso ≠ enunciado**

* **Onde**: Texto reconhece **10 fotos** (enunciado pede **11**). Falta citar explicitamente o edital para contextualizar a exceção (você já menciona, mas pode ancorar melhor). &#x20;
* **Por que é problema**: parecer de não conformidade.
* **Correção**: frase no início da A8: *“O edital solicita 11 imagens (TC2\_PPGETI\_2025.1); por limitação, usamos 10, mantendo estratificação 8/2 por repetição.”*

**\[P-12] “FPR idêntica” vs. tabela**

* **Onde**: Texto afirma “**FPR idêntica** à do MQ” para o PL; a Tabela 6 mostra **0,0178** (PL) vs **0,0182** (MQ) — **não é idêntica**.&#x20;
* **Por que é problema**: afirmação factual incorreta (ainda que diferença pequena).
* **Correção**: trocar por “**muito semelhante** (≈0,018)”, citando os dois valores.

**\[P-13] FNR média = 0% (MQ) sem nota técnica**

* **Onde**: Tabela 6 indica **FNR=0,0000** para MQ; o texto diz “nenhuma das dez fotos…”.&#x20;
* **Por que é problema**: média exatamente zero em 50 repetições é plausível, mas chama atenção.
* **Correção**: nota de rodapé: *“FNR média 0,0000 (4 casas); desvio-padrão 0,0000; nenhuma repetição errou intruso com MQ.”*

**\[P-14] Parêntese solto e estilo numérico**

* **Onde**: “desvio-padrão 0,023)” — parêntese sobrando. E inconsistências de casas decimais ao longo do texto.&#x20;
* **Por que é problema**: cuidado editorial.
* **Correção**: remover “)”; padronizar: métricas com **4 casas**, tempos com **3 casas**; usar “≈” apenas quando o número **não** vier da tabela.

**\[P-15] Tempos A8: MLP-1H (144 ms) > MLP-2H (125 ms) sem explicação**

* **Onde**: Seção 10.2. É possível (largura/otimizador podem inverter custos), mas é **contraintuitivo**.&#x20;
* **Por que é problema**: o leitor espera 2H ≥ 1H.
* **Correção**: uma frase explicativa: *“No melhor conjunto de hiperparâmetros, a 2H selecionada teve larguras/otimizador que a tornaram marginalmente mais rápida que a 1H.”*

**Ponto forte**: Metodologia A8 é muito bem fechada (positivo = intruso; métricas definidas; seleção por **F1 do intruso**; pipeline conforme enunciado). &#x20;

---

## Síntese (o relatório “faz sentido”? Há erros “crassos”? Está inventando algo?)

* **Sentido global**: **Sim, faz sentido**. A espinha dorsal metodológica (A1–A7 e A8) está **coerente com o enunciado** e com o que se espera tecnicamente do pipeline (PCA→redução→Box–Cox, etc.). As conclusões principais (PL muito eficiente; PCA-reduce com $q=10$ preserva desempenho e derruba custo; Box–Cox não ajuda; A8 com positivo=intruso) **batem** com os resultados e com a teoria. &#x20;
* **Erros crassos**: Não há “invenções” metodológicas ou afirmações absurdas. **Há, sim, erros formais/lógicos** que **precisam** ser sanados: “ganhos para todos” quando o PL ficou mais lento; **normalização “min–max \[1,1]”** (erro conceitual de rótulo); “FPR idêntica” quando difere na 4ª casa; *placeholder* de figura; arredondamento a zero mascarando ordem de grandeza. Isso não invalida o estudo, mas **tira pontos** em rigor.&#x20;
* **Hipóteses/afirmações**: São, em geral, **bem justificadas** (condicionamento com PCA; queda com Box–Cox em PCs \~gaussianos). Faltam apenas **afinamentos de texto** para não over-generalizar (ex.: “para todos”).&#x20;

---

## Nota final (0–10): **8,6**

**Justificativa**: O trabalho é **metodologicamente sólido**, com desenho estatístico robusto e discussão alinhada à teoria. Perde pontos por **inconsistências editoriais e de nomenclatura** (P-5, P-6, P-8, P-12, P-14), um **erro conceitual de rótulo** (“min–max \[1,1]”), e por frases **imprecisas** em momentos críticos. Todos os ajustes sugeridos são **textuais ou de exportação** (ou microfixes de geração de tabela) e **podem ser feitos hoje** sem novos experimentos. Corrigindo esses pontos, o relatório sobe confortavelmente para a faixa **9,3–9,6**.

---

### Checklist rápido de correções (sem mexer em experimentos)

* Corrigir **“min–max \[1,1]” → \[0,1]** ou **\[−1,1]** (Tabela 3).&#x20;
* Ajustar 4.3: **remover “para todos”** e completar “**40%**”.&#x20;
* Tabela 2: mostrar **0,00033 s** (ou notação científica), não **0,000**.&#x20;
* Inserir a **figura da variância explicada** (remover “fig\_variance\_explained.png” cru).&#x20;
* Trocar “**FPR idêntica**” por “**muito semelhante** (≈0,018)”, citando valores.&#x20;
* Padronizar casas decimais: **4** para métricas; **3** para tempos; usar “≈” só fora da tabela.&#x20;
* Esclarecer notação: $\mathbb{R}^{1600}$ ou “dimensão 1600”; uniformizar “macro”.&#x20;
* Acrescentar 1 frase em A8 sobre a **exceção 10≠11** fotos, ancorada no edital.&#x20;

Se quiser, posso te devolver **um bloco de texto pronto** com as reformulações exatas (para colar no PDF) e **uma funçãozinha** que corrige a *string* “min–max \[1,1]” durante a exportação das tabelas.
