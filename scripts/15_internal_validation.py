"""Validação interna leave-many-out (seção 27): Top-1, Top-3, acurácia de
superpopulação e matriz de confusão 26x26, usando o mesmo classificador kNN
sobre PCA (mesmo split e random_state do script 11)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
N_COMPONENTS = 10
N_NEIGHBORS = 15


def top_k_accuracy(proba: np.ndarray, classes: np.ndarray, y_true: np.ndarray, k: int) -> float:
    top_k_idx = np.argsort(-proba, axis=1)[:, :k]
    top_k_labels = classes[top_k_idx]
    hits = (top_k_labels == y_true[:, None]).any(axis=1)
    return float(hits.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "Data")
    parser.add_argument("--intermediate", type=Path, default=ROOT / "outputs" / "intermediate")
    parser.add_argument("--figures", type=Path, default=ROOT / "outputs" / "figures")
    parser.add_argument("--subset", type=Path, default=None)
    args = parser.parse_args()
    subset_path = args.subset or args.intermediate / "snp_subsets" / "ld_pruned.npy"
    args.figures.mkdir(parents=True, exist_ok=True)

    X_ref = np.load(args.intermediate / "X_ref_final.npy", mmap_mode="r")
    sample_ids = np.load(args.intermediate / "reference_sample_ids.npy", allow_pickle=True)
    metadata = pd.read_csv(args.base / "processed" / "metadata_populations.csv").set_index("Sample")
    positions = np.load(subset_path)

    X = np.asarray(X_ref[:, positions], dtype=np.float32)
    meta_aligned = metadata.reindex(sample_ids)
    y_pop = meta_aligned["Population"].to_numpy()
    y_super = meta_aligned["SuperPop"].to_numpy()

    idx = np.arange(len(sample_ids))
    idx_train, idx_test = train_test_split(idx, test_size=0.20, stratify=y_pop, random_state=42)
    X_train, X_test = X[idx_train], X[idx_test]
    y_pop_train, y_pop_test = y_pop[idx_train], y_pop[idx_test]
    y_super_train, y_super_test = y_super[idx_train], y_super[idx_test]
    ids_test = sample_ids[idx_test]

    # população (26 classes)
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights="distance")
    knn.fit(X_train_pca, y_pop_train)
    proba = knn.predict_proba(X_test_pca)
    pred = knn.classes_[np.argmax(proba, axis=1)]

    top1 = accuracy_score(y_pop_test, pred)
    top3 = top_k_accuracy(proba, knn.classes_, y_pop_test, k=3)

    # superpopulação (5 classes)
    pca_s = PCA(n_components=N_COMPONENTS, random_state=42)
    Xs_train_pca = pca_s.fit_transform(X_train)
    Xs_test_pca = pca_s.transform(X_test)
    knn_s = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights="distance")
    knn_s.fit(Xs_train_pca, y_super_train)
    pred_super = knn_s.predict(Xs_test_pca)
    acc_super = accuracy_score(y_super_test, pred_super)

    cm = confusion_matrix(y_pop_test, pred, labels=knn.classes_)
    cm_df = pd.DataFrame(cm, index=knn.classes_, columns=knn.classes_)
    cm_df.to_csv(args.intermediate / "internal_validation_confusion_matrix.csv")

    comparativo = pd.DataFrame(
        {
            "sample": ids_test,
            "populacao_real": y_pop_test,
            "populacao_prevista": pred,
            "acertou_top1": y_pop_test == pred,
        }
    )
    comparativo.to_csv(args.intermediate / "internal_validation_predictions.csv", index=False)

    with (args.intermediate / "internal_validation_metrics.json").open("w", encoding="utf-8") as f:
        import json

        json.dump(
            {
                "n_validados": int(len(idx_test)),
                "top1_accuracy_population": top1,
                "top3_accuracy_population": top3,
                "accuracy_superpop": acc_super,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    fig, ax = plt.subplots(figsize=(11, 10), facecolor="#fcfcfb")
    im = ax.imshow(cm_df.to_numpy(), cmap="Blues")
    ax.set_xticks(range(len(cm_df.columns)))
    ax.set_xticklabels(cm_df.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cm_df.index)))
    ax.set_yticklabels(cm_df.index, fontsize=7)
    ax.set_xlabel("Predito", color="#0b0b0b")
    ax.set_ylabel("Real", color="#0b0b0b")
    ax.set_title("Matriz de confusão 26×26 — validação interna", color="#0b0b0b", loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8, label="nº de indivíduos")
    fig.tight_layout()
    fig.savefig(args.figures / "matriz_confusao_26pop.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)

    print(f"Indivíduos validados (held-out): {len(idx_test)}")
    print(f"Top-1 accuracy (população): {top1:.4f}")
    print(f"Top-3 accuracy (população): {top3:.4f}")
    print(f"Accuracy (superpopulação): {acc_super:.4f}")
    print("\nExemplos de comparação (10 primeiros):")
    print(comparativo.head(10).to_string(index=False))
    print("\nMatriz de confusão salva em outputs/figures/matriz_confusao_26pop.png")


if __name__ == "__main__":
    main()
