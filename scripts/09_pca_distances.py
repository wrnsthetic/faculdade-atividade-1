"""Distância do indivíduo aos centroides populacionais e score de afinidade (seções 17-18)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intermediate", type=Path, default=ROOT / "outputs" / "intermediate")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    ref = pd.read_csv(args.intermediate / "pca_reference_coords.csv", index_col=0)
    ind = pd.read_csv(args.intermediate / "pca_individuo_coords.csv", index_col=0)
    pc_cols = [c for c in ref.columns if c.startswith("PC")]
    ind_vec = ind.loc["INDIVIDUO", pc_cols].to_numpy()

    def rank_distances(group_col: str) -> pd.Series:
        centroides = ref.groupby(group_col)[pc_cols].mean()
        d = np.linalg.norm(centroides.to_numpy() - ind_vec, axis=1)
        return pd.Series(d, index=centroides.index).sort_values()

    ranking_pop = rank_distances("Population")
    ranking_super = rank_distances("SuperPop")

    def softmax_score(ranking: pd.Series, temperature: float) -> pd.Series:
        d = ranking.to_numpy()
        scores = np.exp(-d / temperature)
        scores = scores / scores.sum()
        return pd.Series(scores, index=ranking.index).sort_values(ascending=False)

    similaridade_pop = softmax_score(ranking_pop, args.temperature)
    similaridade_super = softmax_score(ranking_super, args.temperature)

    ranking_pop.to_csv(args.intermediate / "pca_distance_ranking_population.csv", header=["distancia"])
    ranking_super.to_csv(args.intermediate / "pca_distance_ranking_superpop.csv", header=["distancia"])
    similaridade_pop.to_csv(args.intermediate / "pca_affinity_score_population.csv", header=["score"])
    similaridade_super.to_csv(args.intermediate / "pca_affinity_score_superpop.csv", header=["score"])

    print("Ranking de distância por população (top 10):")
    print(ranking_pop.head(10))
    print("\nScore de afinidade por população (top 10):")
    print(similaridade_pop.head(10))
    print("\nRanking de distância por superpopulação:")
    print(ranking_super)
    print("\nScore de afinidade por superpopulação:")
    print(similaridade_super)


if __name__ == "__main__":
    main()
