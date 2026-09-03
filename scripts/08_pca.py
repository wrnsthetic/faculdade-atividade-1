"""PCA da referência (LD-pruned) e projeção do indivíduo (seções 14-16 do enunciado)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
N_COMPONENTS = 10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "Data")
    parser.add_argument("--intermediate", type=Path, default=ROOT / "outputs" / "intermediate")
    parser.add_argument(
        "--subset",
        type=Path,
        default=None,
        help="Arquivo .npy com posições de coluna a usar (padrão: snp_subsets/ld_pruned.npy)",
    )
    args = parser.parse_args()
    subset_path = args.subset or args.intermediate / "snp_subsets" / "ld_pruned.npy"

    X_ref = np.load(args.intermediate / "X_ref_final.npy", mmap_mode="r")
    X_individuo = np.load(args.intermediate / "X_individuo.npy")
    sample_ids = np.load(args.intermediate / "reference_sample_ids.npy", allow_pickle=True)
    freq_alt = np.load(args.intermediate / "freq_alt_global.npy")
    metadata = pd.read_csv(args.base / "processed" / "metadata_populations.csv").set_index("Sample")
    positions = np.load(subset_path)

    print(f"Usando subconjunto '{subset_path.name}': {len(positions)} SNPs")

    X_ref_subset = np.asarray(X_ref[:, positions], dtype=np.float32)
    ind_subset = X_individuo[positions].astype(np.float32)
    missing = ind_subset == -9
    if missing.any():
        imputed = 2.0 * freq_alt[positions][missing]
        ind_subset[missing] = imputed
        print(f"Imputados {missing.sum()} SNPs ausentes do indivíduo pela freq_alt_global (2p).")

    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    ref_coords = pca.fit_transform(X_ref_subset)
    ind_coords = pca.transform(ind_subset.reshape(1, -1))

    pc_cols = [f"PC{i}" for i in range(1, N_COMPONENTS + 1)]
    pca_df = pd.DataFrame(ref_coords, columns=pc_cols, index=sample_ids).join(metadata)
    ind_df = pd.DataFrame(ind_coords, columns=pc_cols, index=["INDIVIDUO"])

    out_dir = args.intermediate
    pca_df.to_csv(out_dir / "pca_reference_coords.csv")
    ind_df.to_csv(out_dir / "pca_individuo_coords.csv")
    np.save(out_dir / "pca_explained_variance_ratio.npy", pca.explained_variance_ratio_)

    print("\nVariância explicada:")
    for i, v in enumerate(pca.explained_variance_ratio_, start=1):
        print(f"  PC{i}: {v * 100:.2f}%")

    print("\nCoordenadas do indivíduo (PC1-PC5):")
    print(ind_df[pc_cols[:5]])


if __name__ == "__main__":
    main()
