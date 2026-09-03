"""Converte a exportação variante-major do PLINK em X_ref_final (.npy)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traw", type=Path, required=True)
    parser.add_argument("--variant-order", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "intermediate")
    parser.add_argument("--chunksize", type=int, default=1000)
    args = parser.parse_args()

    variants = pd.read_csv(args.variant_order)
    header = pd.read_csv(args.traw, sep="\t", nrows=0)
    metadata_columns = ["CHR", "SNP", "(C)M", "POS", "COUNTED", "ALT"]
    if list(header.columns[:6]) != metadata_columns:
        raise ValueError("Cabeçalho .traw inesperado; não é possível inferir as colunas de amostras.")

    sample_columns = list(header.columns[6:])
    sample_ids = [sample.removeprefix("0_") for sample in sample_columns]
    if len(sample_ids) != 3202:
        raise ValueError(f"Esperadas 3202 amostras; obtidas {len(sample_ids)}.")

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "reference_sample_ids.npy", np.asarray(sample_ids, dtype=str))
    matrix_path = args.output / "X_ref_final.npy"
    matrix = np.lib.format.open_memmap(
        matrix_path,
        mode="w+",
        dtype=np.int8,
        shape=(len(sample_ids), len(variants)),
    )

    start = 0
    for chunk in pd.read_csv(args.traw, sep="\t", chunksize=args.chunksize, na_values=["NA"]):
        stop = start + len(chunk)
        expected_ids = variants["ID"].iloc[start:stop].to_numpy()
        observed_ids = chunk["SNP"].to_numpy()
        if not np.array_equal(observed_ids, expected_ids):
            raise ValueError(f"A ordem das variantes diverge na posição {start}.")

        values = chunk.iloc[:, 6:].fillna(-9).to_numpy(dtype=np.int8)
        if not np.isin(values, [-9, 0, 1, 2]).all():
            raise ValueError(f"Dosagem inválida encontrada entre as variantes {start} e {stop}.")
        matrix[:, start:stop] = values.T
        start = stop
        if start % 50000 == 0 or start == len(variants):
            print(f"Convertidas {start:,}/{len(variants):,} variantes.")

    matrix.flush()
    del matrix
    if start != len(variants):
        raise ValueError(f"Esperadas {len(variants)} variantes; exportação contém {start}.")

    print(f"X_ref_final salvo em {matrix_path} com dimensão ({len(sample_ids)}, {len(variants)}).")


if __name__ == "__main__":
    main()
