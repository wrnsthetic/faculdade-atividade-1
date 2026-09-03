"""Classificação supervisionada: kNN direto (26 pops) vs. hierárquico (seções 19-21)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
N_COMPONENTS = 10
N_NEIGHBORS = 15


def fit_pca_knn(X_train, y_train):
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights="distance")
    knn.fit(X_train_pca, y_train)
    return pca, knn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "Data")
    parser.add_argument("--intermediate", type=Path, default=ROOT / "outputs" / "intermediate")
    parser.add_argument("--subset", type=Path, default=None)
    args = parser.parse_args()
    subset_path = args.subset or args.intermediate / "snp_subsets" / "ld_pruned.npy"

    X_ref = np.load(args.intermediate / "X_ref_final.npy", mmap_mode="r")
    X_individuo = np.load(args.intermediate / "X_individuo.npy")
    sample_ids = np.load(args.intermediate / "reference_sample_ids.npy", allow_pickle=True)
    freq_alt = np.load(args.intermediate / "freq_alt_global.npy")
    metadata = pd.read_csv(args.base / "processed" / "metadata_populations.csv").set_index("Sample")
    positions = np.load(subset_path)

    X = np.asarray(X_ref[:, positions], dtype=np.float32)
    ind_vec = X_individuo[positions].astype(np.float32)
    missing = ind_vec == -9
    if missing.any():
        ind_vec[missing] = 2.0 * freq_alt[positions][missing]

    meta_aligned = metadata.reindex(sample_ids)
    y_pop = meta_aligned["Population"].to_numpy()
    y_super = meta_aligned["SuperPop"].to_numpy()

    idx = np.arange(len(sample_ids))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.20, stratify=y_pop, random_state=42
    )
    X_train, X_test = X[idx_train], X[idx_test]
    y_pop_train, y_pop_test = y_pop[idx_train], y_pop[idx_test]
    y_super_train, y_super_test = y_super[idx_train], y_super[idx_test]

    results = {}

    # --- 19: classificação direta das 26 populações ---
    pca_direct, knn_direct = fit_pca_knn(X_train, y_pop_train)
    X_test_pca_direct = pca_direct.transform(X_test)
    pred_direct = knn_direct.predict(X_test_pca_direct)
    acc_direct = accuracy_score(y_pop_test, pred_direct)
    report_direct = classification_report(y_pop_test, pred_direct, output_dict=True, zero_division=0)
    cm_direct = confusion_matrix(y_pop_test, pred_direct, labels=knn_direct.classes_)
    pd.DataFrame(cm_direct, index=knn_direct.classes_, columns=knn_direct.classes_).to_csv(
        args.intermediate / "classifier_confusion_matrix_26pop.csv"
    )
    results["direct_26pop_accuracy"] = acc_direct

    # --- 20: classificação hierárquica (superpop -> população dentro da superpop) ---
    pca_super, knn_super = fit_pca_knn(X_train, y_super_train)
    X_test_pca_super = pca_super.transform(X_test)
    pred_super = knn_super.predict(X_test_pca_super)
    acc_super = accuracy_score(y_super_test, pred_super)
    results["superpop_accuracy"] = acc_super

    second_level = {}
    for superpop in np.unique(y_super_train):
        mask_train = y_super_train == superpop
        pca_s, knn_s = fit_pca_knn(X_train[mask_train], y_pop_train[mask_train])
        second_level[superpop] = (pca_s, knn_s)

    pred_hier = np.empty_like(pred_super)
    for i, superpop_pred in enumerate(pred_super):
        pca_s, knn_s = second_level[superpop_pred]
        coords = pca_s.transform(X_test[i : i + 1])
        pred_hier[i] = knn_s.predict(coords)[0]
    acc_hier = accuracy_score(y_pop_test, pred_hier)
    acc_hier_super_correct_only = accuracy_score(
        y_pop_test[pred_super == y_super_test], pred_hier[pred_super == y_super_test]
    )
    results["hierarchical_26pop_accuracy"] = acc_hier
    results["hierarchical_pop_accuracy_given_correct_superpop"] = acc_hier_super_correct_only

    # --- 21: predict_proba para o indivíduo (com ressalva de interpretação) ---
    ind_pca_direct = pca_direct.transform(ind_vec.reshape(1, -1))
    proba_direct = pd.Series(
        knn_direct.predict_proba(ind_pca_direct)[0], index=knn_direct.classes_
    ).sort_values(ascending=False)

    ind_pca_super = pca_super.transform(ind_vec.reshape(1, -1))
    proba_super = pd.Series(
        knn_super.predict_proba(ind_pca_super)[0], index=knn_super.classes_
    ).sort_values(ascending=False)
    ind_superpop_pred = proba_super.idxmax()
    pca_s, knn_s = second_level[ind_superpop_pred]
    ind_pca_s = pca_s.transform(ind_vec.reshape(1, -1))
    proba_hier = pd.Series(
        knn_s.predict_proba(ind_pca_s)[0], index=knn_s.classes_
    ).sort_values(ascending=False)

    proba_direct.to_csv(args.intermediate / "classifier_individuo_proba_26pop.csv", header=["proba"])
    proba_super.to_csv(args.intermediate / "classifier_individuo_proba_superpop.csv", header=["proba"])
    proba_hier.to_csv(
        args.intermediate / f"classifier_individuo_proba_hier_within_{ind_superpop_pred}.csv",
        header=["proba"],
    )

    with (args.intermediate / "classifier_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with (args.intermediate / "classifier_report_26pop.json").open("w", encoding="utf-8") as f:
        json.dump(report_direct, f, indent=2, ensure_ascii=False)

    print(f"Subconjunto de SNPs usado: {subset_path.name} ({X.shape[1]} SNPs)")
    print(f"Treino: {len(idx_train)} amostras | Teste: {len(idx_test)} amostras\n")
    print(f"[19] Acurácia direta (26 populações): {acc_direct:.4f}")
    print(f"[20] Acurácia superpopulação (5 classes): {acc_super:.4f}")
    print(f"[20] Acurácia hierárquica (26 populações via 2 níveis): {acc_hier:.4f}")
    print(f"     (dado que a superpop foi acertada): {acc_hier_super_correct_only:.4f}")
    print("\n[21] predict_proba do indivíduo (direta, 26 pop, top 5):")
    print(proba_direct.head(5))
    print("\n[21] predict_proba do indivíduo (superpopulação):")
    print(proba_super)
    print(f"\n[21] predict_proba do indivíduo (2º nível, dentro de {ind_superpop_pred}, top 5):")
    print(proba_hier.head(5))


if __name__ == "__main__":
    main()
