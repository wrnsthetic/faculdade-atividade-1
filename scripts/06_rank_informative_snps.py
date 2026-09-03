"""Ranking de SNPs informativos por Δp e recorte em subconjuntos (seção 12 do enunciado)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUBSET_SIZES = [1_000, 5_000, 10_000, 20_000, 50_000]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intermediate", type=Path, default=ROOT / "outputs" / "intermediate")
    args = parser.parse_args()

    freq = pd.read_csv(args.intermediate / "freq_by_population.csv", index_col=0)
    pop_cols = [c for c in freq.columns if c not in ("INDEX", "freq_alt_global")]

    # A posição da coluna em X_ref_final/X_individuo é a ordem da linha neste
    # arquivo (0-based); a coluna "INDEX" é só metadado do espaço original de
    # 575.480 variantes e tem lacunas, não serve para indexar a matriz.
    position = np.arange(len(freq))
    delta_p = freq[pop_cols].max(axis=1) - freq[pop_cols].min(axis=1)
    ranking = pd.DataFrame(
        {
            "position": position,
            "INDEX": freq["INDEX"].to_numpy(),
            "delta_p": delta_p.to_numpy(),
            "pop_max": freq[pop_cols].idxmax(axis=1).to_numpy(),
            "pop_min": freq[pop_cols].idxmin(axis=1).to_numpy(),
        },
        index=freq.index,
    ).sort_values("delta_p", ascending=False)
    ranking.to_csv(args.intermediate / "delta_p_ranking.csv")

    subset_dir = args.intermediate / "snp_subsets"
    subset_dir.mkdir(parents=True, exist_ok=True)
    ordered_positions = ranking["position"].to_numpy()
    for size in SUBSET_SIZES:
        top_positions = np.sort(ordered_positions[:size])
        np.save(subset_dir / f"top_{size}.npy", top_positions)

    print(f"Ranking completo: {len(ranking)} SNPs")
    print(ranking.head(10)[["delta_p", "pop_max", "pop_min"]])
    print("\nSubconjuntos salvos em", subset_dir)
    for size in SUBSET_SIZES:
        print(f"  top_{size}.npy -> {size} SNPs (menor delta_p incluído: "
              f"{ranking['delta_p'].iloc[size - 1]:.3f})")


if __name__ == "__main__":
    main()
