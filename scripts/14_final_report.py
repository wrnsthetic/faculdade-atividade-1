"""Monta o relatório final da calculadora, no formato da seção 25 do enunciado."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "Data")
    parser.add_argument("--intermediate", type=Path, default=ROOT / "outputs" / "intermediate")
    args = parser.parse_args()

    genotype = pd.read_csv(args.base / "individuo" / "genotipo_microarray.csv", low_memory=False)
    with (args.intermediate / "harmonization_metrics.json").open(encoding="utf-8") as f:
        metrics = json.load(f)

    X_individuo = np.load(args.intermediate / "X_individuo.npy")
    ld_pruned = np.load(args.intermediate / "snp_subsets" / "ld_pruned.npy")
    usable_and_pruned = int(np.count_nonzero(X_individuo[ld_pruned] != -9))

    pesos_pop = pd.read_csv(args.intermediate / "mixture_weights_population.csv", index_col=0)["peso"] * 100
    pesos_super = pd.read_csv(args.intermediate / "mixture_weights_superpop.csv", index_col=0)["peso"] * 100

    n_snps_arquivo = len(genotype)
    n_encontrados_1000g = metrics["mapped_snps"]
    n_apos_harmonizacao = metrics["usable_snps"]
    n_apos_qc = metrics["usable_snps"]
    n_apos_ld = usable_and_pruned

    linhas = []
    linhas.append("=" * 38)
    linhas.append("ANÁLISE GENÔMICA POPULACIONAL")
    linhas.append("=" * 38)
    linhas.append("")
    linhas.append(f"SNPs no arquivo:                 {n_snps_arquivo:,}".replace(",", "."))
    linhas.append(f"SNPs encontrados no 1000G:        {n_encontrados_1000g:,}".replace(",", "."))
    linhas.append(f"SNPs após harmonização:           {n_apos_harmonizacao:,}".replace(",", "."))
    linhas.append(f"SNPs após QC:                     {n_apos_qc:,}".replace(",", "."))
    linhas.append(f"SNPs após LD pruning:             {n_apos_ld:,}".replace(",", "."))
    linhas.append("")
    linhas.append("SUPERPOPULAÇÕES (modelo de mistura por frequências)")
    for pop, val in pesos_super.items():
        linhas.append(f"{pop:<6} {val:5.1f}%")
    linhas.append("")
    linhas.append("POPULAÇÕES DE REFERÊNCIA (top 10, modelo de mistura por frequências)")
    for pop, val in pesos_pop.head(10).items():
        linhas.append(f"{pop:<6} {val:5.1f}%")
    linhas.append("...")
    linhas.append("")
    linhas.append(
        "AVISO: estes valores são afinidade genética relativa ao painel de\n"
        "referência (1000 Genomes, 26 populações), não porcentagens literais de\n"
        "nacionalidade ou ancestralidade 'pura'. Populações admixed (ex.: PUR,\n"
        "CLM, MXL, PEL) têm histórias demográficas próprias."
    )

    texto = "\n".join(linhas)
    print(texto)
    (args.intermediate / "final_report.txt").write_text(texto, encoding="utf-8")


if __name__ == "__main__":
    main()
