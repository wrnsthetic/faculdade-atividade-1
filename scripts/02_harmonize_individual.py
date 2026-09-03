"""Gera o vetor X_individuo e seu relatório de controle de qualidade."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.harmonization import harmonize_individual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "Data")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "intermediate")
    args = parser.parse_args()

    genotype = pd.read_csv(args.base / "individuo" / "genotipo_microarray.csv", low_memory=False)
    variants = pd.read_csv(args.base / "processed" / "harmonization_map.csv")
    x_individual, metrics = harmonize_individual(genotype, variants)

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "X_individuo.npy", x_individual)
    with (args.output / "harmonization_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics.to_dict(), file, indent=2, ensure_ascii=False)

    print(f"X_individuo: {x_individual.shape}, dtype={x_individual.dtype}")
    for key, value in metrics.to_dict().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
