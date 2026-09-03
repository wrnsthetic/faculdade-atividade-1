"""Conversão do genótipo Illumina para dosagens ALT da referência."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


COMPLEMENT = str.maketrans("ACGT", "TGCA")
REQUIRED_MAP_COLUMNS = {"INDEX", "Name", "REF", "ALT", "NEEDS_COMPLEMENT"}


@dataclass(frozen=True)
class HarmonizationMetrics:
    mapped_snps: int
    valid_base_calls: int
    direct_compatible: int
    map_flag_complement_compatible: int
    selected_orientation: str
    ref_ref: int
    ref_alt: int
    alt_alt: int
    missing_or_incompatible: int

    @property
    def usable_snps(self) -> int:
        return self.ref_ref + self.ref_alt + self.alt_alt

    @property
    def usable_rate(self) -> float:
        return self.usable_snps / self.mapped_snps

    def to_dict(self) -> dict[str, int | float | str]:
        values = asdict(self)
        values["usable_snps"] = self.usable_snps
        values["usable_rate"] = self.usable_rate
        return values


def _dosages(calls: pd.Series, variants: pd.DataFrame) -> np.ndarray:
    """Codifica chamadas em 0/1/2/-9 na ordem do mapa de variantes."""
    first, second = calls.str[0], calls.str[1]
    ref, alt = variants["REF"], variants["ALT"]
    valid = calls.str.fullmatch(r"[ACGT]{2}")

    dosage = np.select(
        [
            first.eq(ref) & second.eq(ref),
            (first.eq(ref) & second.eq(alt)) | (first.eq(alt) & second.eq(ref)),
            first.eq(alt) & second.eq(alt),
        ],
        [0, 1, 2],
        default=-9,
    ).astype(np.int8)
    dosage[~valid.to_numpy()] = -9
    return dosage


def harmonize_individual(
    genotype: pd.DataFrame,
    harmonization_map: pd.DataFrame,
) -> tuple[np.ndarray, HarmonizationMetrics]:
    """Retorna ``X_individuo`` e métricas de QC.

    O mapa já foi produzido após a harmonização prévia da referência descrita
    no enunciado. Para evitar uma correção de fita duplicada, as chamadas são
    avaliadas nas duas orientações. A orientação com maior compatibilidade com
    REF/ALT é selecionada globalmente e registrada nas métricas.
    """
    missing_map = REQUIRED_MAP_COLUMNS.difference(harmonization_map.columns)
    if missing_map:
        raise ValueError(f"Colunas ausentes no mapa: {sorted(missing_map)}")
    if "Name" not in genotype or "35.Genotipo" not in genotype:
        raise ValueError("O genótipo deve conter as colunas 'Name' e '35.Genotipo'.")
    if not harmonization_map["INDEX"].is_unique or not harmonization_map["Name"].is_unique:
        raise ValueError("O mapa de harmonização deve ter INDEX e Name únicos.")
    if not genotype["Name"].is_unique:
        raise ValueError("O arquivo individual tem marcadores Name duplicados.")

    variants = harmonization_map.sort_values("INDEX").reset_index(drop=True).copy()
    if (variants["INDEX"] < 0).any():
        raise ValueError("INDEX não pode conter valores negativos.")

    calls_frame = genotype[["Name", "35.Genotipo"]].copy()
    merged = variants.merge(calls_frame, on="Name", how="left", validate="one_to_one")
    calls = merged["35.Genotipo"].fillna("--").astype(str).str.upper()
    map_flag_calls = calls.where(
        ~merged["NEEDS_COMPLEMENT"], calls.str.translate(COMPLEMENT)
    )

    direct = _dosages(calls, merged)
    map_flag_complement = _dosages(map_flag_calls, merged)
    direct_compatible = int(np.count_nonzero(direct >= 0))
    map_flag_complement_compatible = int(np.count_nonzero(map_flag_complement >= 0))

    if map_flag_complement_compatible > direct_compatible:
        dosage = map_flag_complement
        orientation = "map_flag_complement"
    else:
        dosage = direct
        orientation = "direct"

    metrics = HarmonizationMetrics(
        mapped_snps=len(merged),
        valid_base_calls=int(calls.str.fullmatch(r"[ACGT]{2}").sum()),
        direct_compatible=direct_compatible,
        map_flag_complement_compatible=map_flag_complement_compatible,
        selected_orientation=orientation,
        ref_ref=int(np.count_nonzero(dosage == 0)),
        ref_alt=int(np.count_nonzero(dosage == 1)),
        alt_alt=int(np.count_nonzero(dosage == 2)),
        missing_or_incompatible=int(np.count_nonzero(dosage == -9)),
    )
    return dosage, metrics
