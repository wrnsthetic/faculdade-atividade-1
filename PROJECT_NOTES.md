# Calculadora de Ancestralidade Genômica — notas do projeto

## Objetivo e interpretação

Implementar a calculadora didática prevista no enunciado da disciplina: comparar
um indivíduo genotipado em microarray Illumina com as 26 populações do 1000
Genomes. Os resultados descrevem **afinidade genética relativa ao painel de
referência**; eles não são percentuais literais de nacionalidade, nem inferência
de ancestralidade "pura".

## Requisitos confirmados no enunciado

- Referência fornecida: 3.202 indivíduos e 575.480 SNPs bialélicos em PLINK 2.
- O mapa reduzido contém 572.011 SNPs únicos para a análise.
- `X_ref_final.shape == (3202, 572011)`.
- `X_individuo.shape == (572011,)`.
- Dosagem em relação ao alelo `ALT`: `0` = `REF/REF`, `1` = `REF/ALT`,
  `2` = `ALT/ALT`, `-9` = chamada ausente ou incompatível.
- A conversão do indivíduo usa exclusivamente `processed/harmonization_map.csv`.
- A PCA será ajustada somente na referência; o indivíduo será projetado depois.

## Dados em uso

- `Data/individuo/genotipo_microarray.csv`: 654.027 registros; entrada individual.
- `Data/processed/1000G_GSA_SNP.pgen`, `.pvar.zst`, `.psam`: referência PLINK 2.
- `Data/processed/harmonization_map.csv`: ligação marcador Illumina ↔ variante.
- `Data/processed/metadata_populations.csv`: rótulos das 26 populações.

`Data/referencia/` e o manifesto Illumina são materiais de origem/rastreabilidade
e não participam da execução principal. A preparação do professor já filtrou o
1000 Genomes para GSA v3, manteve SNPs bialélicos, tratou strand e removeu
incompatibilidades.

## Estado validado

- Metadados: 3.202 amostras, 26 populações e `AFR`, `AMR`, `EAS`, `EUR`, `SAS`.
- Mapa: 572.011 linhas, `INDEX` e `Name` únicos.
- PLINK 2: 3.202 amostras × 575.480 variantes, máximo de dois alelos por variante
  e todos os alelos `REF` conhecidos.
- Matrizes geradas e verificadas: `X_ref_final` `(3202, 572011)`,
  `X_individuo` `(572011,)`, ambas com `dtype=int8`; a referência contém apenas
  dosagens `0/1/2` na verificação amostral (ausências, se presentes, são `-9`).
- A interseção por `Name` é perfeita: 572.011/572.011 marcadores do mapa estão
  no arquivo individual. Aqui `Name` é identificador do microarray, **não** rsID.
- `INDEX` representa a posição na matriz original de 575.480 variantes e possui
  lacunas; ele deve ser ordenado, não renumerado, para construir o conjunto final.

## Harmonização do indivíduo

A harmonização coloca o indivíduo e a referência no mesmo espaço de variantes e
na mesma orientação de dosagem. Para cada marcador do mapa, a chamada de duas
bases do indivíduo é comparada aos campos `REF` e `ALT`, resultando em `0/1/2`
ou `-9`.

O mapa contém `NEEDS_COMPLEMENT`. O enunciado indica que ele sinaliza necessidade
de complementar alelos, porém não especifica em que lado da transformação. O
pipeline compara chamadas diretas com chamadas complementadas apenas quando a
flag é verdadeira e seleciona a estratégia de maior compatibilidade `REF/ALT`.

| Medida | Valor |
|---|---:|
| Chamadas A/C/G/T válidas | 570.874 |
| Compatíveis sem novo complemento | 570.874 |
| Compatíveis complementando onde a flag é verdadeira | 302.919 |
| Orientação escolhida | direta |
| `REF/REF` | 415.188 |
| `REF/ALT` | 102.406 |
| `ALT/ALT` | 53.280 |
| Ausentes/incompatíveis (`-9`) | 1.137 |
| SNPs aproveitáveis | 570.874 / 572.011 (99,801%) |

Aplicar `NEEDS_COMPLEMENT` novamente às chamadas duplicaria uma correção já
refletida no painel preparado. A decisão é auditável e deverá ser mencionada no
relatório como particularidade dos dados fornecidos.

## Artefatos e estratégia da referência

- `src/harmonization.py`: função reutilizável para gerar `X_individuo` e métricas.
- `scripts/02_harmonize_individual.py`: gera vetor e relatório JSON.
- `scripts/03_prepare_reference_export.py`: listas de variantes e alelos ALT.
- `scripts/04_build_x_ref_final.py`: gera a matriz referência amostra-major.
- `notebooks/01_configuracao_e_validacao.ipynb`: checagem inicial portátil.

PLINK 2 extrai os IDs do mapa e exporta formato variante-major (`--export Av`).
`--export-allele` força a contagem do `ALT`, mantendo `0/1/2`. A conversão em
blocos valida a ordem dos IDs e produz matrizes `int8` compatíveis:

```text
X_ref_final: (3202, 572011)
X_individuo: (572011,)
```

Dados e resultados pesados ficam em `Data/` e `outputs/`: não são versionados.

## Portabilidade local ↔ Colab

- Detectar `E:\ATV1\Data` localmente.
- No Colab usar `/content/drive/MyDrive/Projeto_Ancestralidade`, ou
  `ANCESTRY_DATA_DIR`.
- Não usar caminhos fixos fora da célula de configuração.
- O notebook final será `.ipynb`; instalações e montagem do Drive são condicionais.
- Localmente, PLINK 2 está em `tools/plink2/plink2.exe`; no Colab haverá célula
  opcional para baixar o binário Linux.

## Próximas etapas

1. Validar `X_ref_final` e preparar imputação de `-9` por frequência ALT.
2. Calcular frequências ALT globais e por população.
3. Avaliar SNPs informativos por `Δp` e LD pruning.
4. PCA, projeção, centroides, classificação, validação e mistura por frequências.
5. Produzir visualizações e discutir limitações do painel de referência.
