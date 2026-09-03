# Checklist de progresso — Calculadora de Ancestralidade Genômica

Acompanhamento seção a seção do enunciado (`Projeto_Calculadora_Ancestralidade_1000Genomes.pdf`).
Tudo consolidado num único notebook: **`notebooks/01_calculadora_ancestralidade.ipynb`**
(local ↔ Colab, já executado de ponta a ponta). Os capítulos abaixo batem 1:1 com os
títulos `## N. ...` dentro do notebook.

| # | Seção do PDF | Status | Capítulo no notebook |
|---|---|:---:|---|
| 1 | Visão geral do projeto | ✅ | Cap. 1 — markdown (contexto, sem código) |
| 2 | Problema (formato do arquivo microarray) | ✅ | Cap. 2 — preview de `genotipo_microarray.csv` |
| 3 | As 26 populações do 1000 Genomes | ✅ | Cap. 3 — confere `metadata_populations.csv` |
| 4 | Interpretação científica (resultados ≠ % literal) | ✅ | Cap. 4 — markdown; reforçado nos Caps. 25 e 29 |
| 5 | Preparação prévia da referência (feita pelo professor) | ✅ | Cap. 5 — markdown (sem código) |
| 6 | Organização dos dados no Google Drive | ✅ | Cap. 6 — markdown (sem código) |
| 7 | Conectando o Colab ao Google Drive (portabilidade) | ✅ | Cap. 7 — célula de configuração local/Colab + checagem de arquivos |
| 8 | Matriz genotípica (8.1 leitura PLINK2, 8.2 mapa harmonização) | ✅ | Cap. 8 — `X_ref_final` (3202×572011) via PLINK2 |
| 9 | Preparação do Genoma Testado (X_individuo) | ✅ | Cap. 9 — `X_individuo` (572011,), 99,80% aproveitável |
| 10 | Frequências alélicas (globais) | ✅ | Cap. 10-11 (célula combinada) |
| 11 | Frequências para as 26 populações | ✅ | Cap. 10-11 (célula combinada) |
| 12 | SNPs informativos (Δp) + subconjuntos 1k/5k/10k/20k/50k | ✅ | Cap. 12 |
| 13 | Linkage Disequilibrium e LD pruning | ✅ | Cap. 13 — 328.389/572.011 SNPs mantidos |
| 14 | PCA — análise exploratória (10 componentes) | ✅ | Cap. 14 |
| 15 | Gráficos de PCA (por população / superpopulação) | ✅ | Cap. 15 |
| 16 | Projetando o indivíduo na PCA + 5 perguntas | ✅ | Cap. 16 — 5 perguntas respondidas em markdown |
| 17 | Distância aos centroides populacionais | ✅ | Cap. 17 — IBS/TSI mais próximos |
| 18 | Score de afinidade relativa (softmax) | ✅ | Cap. 18 |
| 19 | Classificação supervisionada (kNN) | ✅ | Cap. 19 — acurácia direta 80,97% (26 pop) |
| 20 | Classificação hierárquica (26 direto vs. 2 níveis) | ✅ | Cap. 20 — superpop 100%, hierárquico 26 pop 77,07% (< direto) |
| 21 | Probabilidades do classificador (`predict_proba`) | ✅ | Cap. 21 — indivíduo: IBS 93,5% / TSI 6,5% |
| 22 | Modelo de mistura por frequências (pesos w_k, SLSQP) | ✅ | Cap. 22 — convergiu, MSE 0,180 < baseline 0,200 |
| 23 | Limitações do modelo de mistura | ✅ | Cap. 23 — markdown |
| 24 | Resultado agregado por superpopulação (mistura) | ✅ | Cap. 24 — EUR 83,0% / AMR 16,6% / AFR 0,4% |
| 25 | Saída final da calculadora (relatório formatado) | ✅ | Cap. 25 |
| 26 | Visualizações obrigatórias (5 itens) | ✅ 5/5 | Caps. 15 (itens 3-4) + 26 (itens 1, 2, 5) |
| 27 | Validação interna (Top-1/Top-3, matriz de confusão 26×26) | ✅ | Cap. 27 — 641 indivíduos: Top-1 80,97%, Top-3 98,13%, superpop 100% |
| 28 | Comparação entre métodos (tabela) | ✅ | Cap. 28 |
| 29 | Pergunta final do projeto (redação) | ✅ | Cap. 29 |
| 30 | Fluxo conceitual esperado | ✅ | Cap. 30 — diagrama + mapeamento pros capítulos |

**Status geral: 30/30 capítulos concluídos.** Notebook executado com nbclient
(kernel `atv1-ancestralidade`), 0 erros, outputs e gráficos preenchidos.

Scripts individuais que existiram durante o desenvolvimento (`scripts/02` a `15`)
foram absorvidos pelo notebook único e não são mais necessários como referência
separada.
