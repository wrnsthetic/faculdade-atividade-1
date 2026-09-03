"""Modelo simplificado de mistura por frequências (seções 22-24 do enunciado)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]

POP_TO_SUPER = {
    "ACB": "AFR", "ASW": "AFR", "ESN": "AFR", "GWD": "AFR", "LWK": "AFR", "MSL": "AFR", "YRI": "AFR",
    "CLM": "AMR", "MXL": "AMR", "PEL": "AMR", "PUR": "AMR",
    "CDX": "EAS", "CHB": "EAS", "CHS": "EAS", "JPT": "EAS", "KHV": "EAS",
    "CEU": "EUR", "FIN": "EUR", "GBR": "EUR", "IBS": "EUR", "TSI": "EUR",
    "BEB": "SAS", "GIH": "SAS", "ITU": "SAS", "PJL": "SAS", "STU": "SAS",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intermediate", type=Path, default=ROOT / "outputs" / "intermediate")
    args = parser.parse_args()

    freq = pd.read_csv(args.intermediate / "freq_by_population.csv", index_col=0)
    pop_cols = [c for c in freq.columns if c not in ("INDEX", "freq_alt_global")]
    X_individuo = np.load(args.intermediate / "X_individuo.npy")

    assert len(freq) == len(X_individuo), "freq_by_population e X_individuo fora de ordem/tamanho."
    valid = X_individuo != -9
    P = freq.loc[valid, pop_cols].to_numpy(dtype=np.float64)
    g = X_individuo[valid].astype(np.float64)
    populacoes = pop_cols
    K = len(populacoes)
    print(f"SNPs usados no ajuste: {valid.sum()} / {len(X_individuo)} (excluídos {(~valid).sum()} ausentes)")

    def objective(w: np.ndarray) -> float:
        esperado = 2 * P.dot(w)
        return float(np.mean((g - esperado) ** 2))

    w0 = np.ones(K) / K
    bounds = [(0, 1) for _ in range(K)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    resultado = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)

    pesos = pd.Series(resultado.x, index=populacoes).sort_values(ascending=False)
    pesos.to_csv(args.intermediate / "mixture_weights_population.csv", header=["peso"])

    resultado_pop = pesos.to_frame("score")
    resultado_pop["superpop"] = resultado_pop.index.map(POP_TO_SUPER)
    resultado_super = resultado_pop.groupby("superpop")["score"].sum().sort_values(ascending=False)
    resultado_super.to_csv(args.intermediate / "mixture_weights_superpop.csv", header=["peso"])

    mse_final = objective(resultado.x)
    mse_uniforme = objective(w0)

    print(f"\nConvergiu: {resultado.success} ({resultado.message})")
    print(f"MSE final: {mse_final:.4f} | MSE pesos uniformes (baseline): {mse_uniforme:.4f}")
    print("\nPesos por população (top 10):")
    print(pesos.head(10))
    print("\nPesos agregados por superpopulação:")
    print(resultado_super)


if __name__ == "__main__":
    main()
