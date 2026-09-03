"""Converte a lista prune.in (IDs) do PLINK2 em posições de coluna de X_ref_final."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intermediate", type=Path, default=ROOT / "outputs" / "intermediate")
    parser.add_argument("--prune-in", type=Path, default=None)
    args = parser.parse_args()
    prune_in = args.prune_in or args.intermediate / "ld_pruning.prune.in"

    variants = pd.read_csv(args.intermediate / "reference_variant_order.csv")
    kept_ids = pd.read_csv(prune_in, header=None, names=["ID"])["ID"]

    # A posição da coluna em X_ref_final/X_individuo é a ordem da linha neste
    # arquivo (0-based); a coluna "INDEX" tem lacunas (espaço original de
    # 575.480 variantes) e não serve para indexar a matriz.
    kept_mask = variants["ID"].isin(kept_ids).to_numpy()
    positions = np.sort(np.flatnonzero(kept_mask))

    if len(positions) != len(kept_ids):
        raise ValueError(
            f"Esperados {len(kept_ids)} SNPs após pruning; encontrados {len(positions)} no mapa."
        )

    out_path = args.intermediate / "snp_subsets" / "ld_pruned.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, positions)
    print(f"LD-pruned subset: {len(positions)} SNPs -> {out_path}")


if __name__ == "__main__":
    main()
