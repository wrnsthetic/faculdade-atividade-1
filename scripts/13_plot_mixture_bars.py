"""Gráficos de barras do modelo de mistura (itens 1-2 da seção 26 do enunciado)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SUPERPOP_COLORS = {
    "AFR": "#2a78d6", "AMR": "#eb6834", "EAS": "#1baf7a", "EUR": "#eda100", "SAS": "#e87ba4",
}
POP_TO_SUPER = {
    "ACB": "AFR", "ASW": "AFR", "ESN": "AFR", "GWD": "AFR", "LWK": "AFR", "MSL": "AFR", "YRI": "AFR",
    "CLM": "AMR", "MXL": "AMR", "PEL": "AMR", "PUR": "AMR",
    "CDX": "EAS", "CHB": "EAS", "CHS": "EAS", "JPT": "EAS", "KHV": "EAS",
    "CEU": "EUR", "FIN": "EUR", "GBR": "EUR", "IBS": "EUR", "TSI": "EUR",
    "BEB": "SAS", "GIH": "SAS", "ITU": "SAS", "PJL": "SAS", "STU": "SAS",
}
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intermediate", type=Path, default=ROOT / "outputs" / "intermediate")
    parser.add_argument("--figures", type=Path, default=ROOT / "outputs" / "figures")
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)

    pesos = pd.read_csv(args.intermediate / "mixture_weights_population.csv", index_col=0)["peso"]
    pesos_super = pd.read_csv(args.intermediate / "mixture_weights_superpop.csv", index_col=0)["peso"]

    resultado = pesos * 100
    colors = [SUPERPOP_COLORS[POP_TO_SUPER[pop]] for pop in resultado.index]
    fig, ax = plt.subplots(figsize=(15, 5), facecolor=SURFACE)
    ax.bar(resultado.index, resultado.to_numpy(), color=colors, zorder=2)
    style_axes(ax)
    ax.set_ylabel("Contribuição relativa (%)", color=INK)
    ax.set_xlabel("População 1000 Genomes", color=INK)
    ax.set_title("Mistura por frequências — 26 populações", color=INK, loc="left")
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.tight_layout()
    fig.savefig(args.figures / "barras_26_populacoes.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    resultado_super = pesos_super * 100
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=SURFACE)
    colors_super = [SUPERPOP_COLORS[s] for s in resultado_super.index]
    ax.bar(resultado_super.index, resultado_super.to_numpy(), color=colors_super, zorder=2)
    style_axes(ax)
    ax.set_ylabel("Contribuição relativa (%)", color=INK)
    ax.set_xlabel("Superpopulação", color=INK)
    ax.set_title("Mistura por frequências — 5 superpopulações", color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(args.figures / "barras_5_superpopulacoes.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    print("Figuras salvas:")
    print(" -", args.figures / "barras_26_populacoes.png")
    print(" -", args.figures / "barras_5_superpopulacoes.png")


if __name__ == "__main__":
    main()
