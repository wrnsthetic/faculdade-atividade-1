"""Calcula a frequência do alelo ALT global e por população (seções 10-11 do enunciado)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def allele_frequency(dosage: np.ndarray) -> np.ndarray:
    """p = soma(dosagens ALT) / (2 * indivíduos válidos), ignorando -9."""
    valid = dosage != -9
    valid_dosage = np.where(valid, dosage, 0).astype(np.float64)
    n_valid = valid.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        freq = valid_dosage.sum(axis=0) / (2 * n_valid)
    freq[n_valid == 0] = np.nan
    return freq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "Data")
    parser.add_argument("--intermediate", type=Path, default=ROOT / "outputs" / "intermediate")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "intermediate")
    args = parser.parse_args()

    X_ref = np.load(args.intermediate / "X_ref_final.npy", mmap_mode="r")
    sample_ids = np.load(args.intermediate / "reference_sample_ids.npy", allow_pickle=True)
    variants = pd.read_csv(args.intermediate / "reference_variant_order.csv")
    metadata = pd.read_csv(args.base / "processed" / "metadata_populations.csv")

    freq_alt = allele_frequency(np.asarray(X_ref))
    assert freq_alt.shape == (X_ref.shape[1],)

    sample_to_pop = metadata.set_index("Sample")["Population"]
    populations = sample_to_pop.reindex(sample_ids)
    if populations.isna().any():
        missing = sample_ids[populations.isna().to_numpy()]
        raise ValueError(f"Amostras sem população conhecida: {missing[:5].tolist()}...")

    freq_by_pop = {}
    for pop, idx in populations.groupby(populations).groups.items():
        positions = populations.index.get_indexer(idx)
        freq_by_pop[pop] = allele_frequency(np.asarray(X_ref[positions, :]))

    freq_df = pd.DataFrame(freq_by_pop, index=variants["ID"])
    freq_df.insert(0, "INDEX", variants["INDEX"].to_numpy())
    freq_df.insert(1, "freq_alt_global", freq_alt)

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "freq_alt_global.npy", freq_alt)
    freq_df.to_csv(args.output / "freq_by_population.csv")

    print(f"freq_alt_global: shape={freq_alt.shape}, nan={np.isnan(freq_alt).sum()}")
    print(f"freq_by_population: {freq_df.shape[0]} SNPs x {len(freq_by_pop)} populações")
    print(freq_df.drop(columns=["INDEX"]).describe().loc[["mean", "min", "max"]])


if __name__ == "__main__":
    main()
