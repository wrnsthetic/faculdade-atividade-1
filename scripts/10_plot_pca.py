"""Visualizações de PCA (seções 15-16 e itens 3-5 da seção 26 do enunciado)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Paleta categórica validada (dataviz skill): ordem fixa, nunca ciclada por dado.
SUPERPOP_COLORS = {
    "AFR": "#2a78d6",  # blue
    "AMR": "#eb6834",  # orange
    "EAS": "#1baf7a",  # aqua
    "EUR": "#eda100",  # yellow
    "SAS": "#e87ba4",  # magenta
}
POP_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
INDIVIDUAL_COLOR = "#e34948"  # red, reservado só pro indivíduo


def style_axes(ax, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.tick_params(colors=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)


def plot_superpop(pca_df: pd.DataFrame, ind: pd.Series, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=SURFACE)
    for superpop, color in SUPERPOP_COLORS.items():
        subset = pca_df[pca_df["SuperPop"] == superpop]
        ax.scatter(subset["PC1"], subset["PC2"], label=superpop, s=14, alpha=0.65,
                   color=color, linewidths=0, zorder=2)
    ax.scatter([ind["PC1"]], [ind["PC2"]], marker="*", s=420, color=INDIVIDUAL_COLOR,
               edgecolors=INK, linewidths=1.2, label="INDIVÍDUO", zorder=5)
    style_axes(ax, "PC1", "PC2")
    ax.set_title("PCA por superpopulação — indivíduo destacado", color=INK, loc="left")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_all_populations(pca_df: pd.DataFrame, ind: pd.Series, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7.5), facecolor=SURFACE)
    for superpop, color in SUPERPOP_COLORS.items():
        pops = sorted(pca_df.loc[pca_df["SuperPop"] == superpop, "Population"].unique())
        for marker, pop in zip(POP_MARKERS, pops):
            subset = pca_df[pca_df["Population"] == pop]
            ax.scatter(subset["PC1"], subset["PC2"], label=pop, s=14, alpha=0.7,
                       color=color, marker=marker, linewidths=0, zorder=2)
    ax.scatter([ind["PC1"]], [ind["PC2"]], marker="*", s=420, color=INDIVIDUAL_COLOR,
               edgecolors=INK, linewidths=1.2, label="INDIVÍDUO", zorder=5)
    style_axes(ax, "PC1", "PC2")
    ax.set_title("PCA das 26 populações — cor = superpopulação, forma = população",
                 color=INK, loc="left")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_focal(pca_df: pd.DataFrame, ind: pd.Series, nearest_pops: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=SURFACE)
    background = pca_df[~pca_df["Population"].isin(nearest_pops)]
    ax.scatter(background["PC1"], background["PC2"], s=10, alpha=0.15, color=MUTED,
               linewidths=0, zorder=1, label="outras populações")
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    for color, pop in zip(palette, nearest_pops):
        subset = pca_df[pca_df["Population"] == pop]
        ax.scatter(subset["PC1"], subset["PC2"], label=pop, s=22, alpha=0.85,
                   color=color, linewidths=0, zorder=3)
    ax.scatter([ind["PC1"]], [ind["PC2"]], marker="*", s=420, color=INDIVIDUAL_COLOR,
               edgecolors=INK, linewidths=1.2, label="INDIVÍDUO", zorder=5)
    style_axes(ax, "PC1", "PC2")
    ax.set_title(f"PCA focal — 5 populações mais próximas: {', '.join(nearest_pops)}",
                 color=INK, loc="left")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intermediate", type=Path, default=ROOT / "outputs" / "intermediate")
    parser.add_argument("--figures", type=Path, default=ROOT / "outputs" / "figures")
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)

    pca_df = pd.read_csv(args.intermediate / "pca_reference_coords.csv", index_col=0)
    ind = pd.read_csv(args.intermediate / "pca_individuo_coords.csv", index_col=0).loc["INDIVIDUO"]
    ranking_pop = pd.read_csv(
        args.intermediate / "pca_distance_ranking_population.csv", index_col=0
    )["distancia"]
    nearest_5 = ranking_pop.head(5).index.tolist()

    plot_superpop(pca_df, ind, args.figures / "pca_por_superpopulacao.png")
    plot_all_populations(pca_df, ind, args.figures / "pca_26_populacoes.png")
    plot_focal(pca_df, ind, nearest_5, args.figures / "pca_focal_5_populacoes.png")

    print("Figuras salvas em", args.figures)
    for f in sorted(args.figures.glob("*.png")):
        print(" -", f.name)


if __name__ == "__main__":
    main()
