"""Prepara as listas de variantes para exportar X_ref_final com PLINK 2."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "Data")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "intermediate")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    variants = pd.read_csv(args.base / "processed" / "harmonization_map.csv")
    required = {"INDEX", "ID", "ALT"}
    if missing := required.difference(variants.columns):
        raise ValueError(f"Colunas ausentes no mapa: {sorted(missing)}")

    variants = variants.sort_values("INDEX")
    if args.limit is not None:
        variants = variants.head(args.limit)

    args.output.mkdir(parents=True, exist_ok=True)
    variants[["ID"]].to_csv(args.output / "reference_variant_ids.txt", index=False, header=False)
    variants[["ID", "ALT"]].to_csv(
        args.output / "reference_alt_alleles.txt", index=False, header=False, sep="\t"
    )
    variants[["INDEX", "ID", "REF", "ALT"]].to_csv(
        args.output / "reference_variant_order.csv", index=False
    )
    print(f"Listas geradas para {len(variants)} variantes.")


if __name__ == "__main__":
    main()
