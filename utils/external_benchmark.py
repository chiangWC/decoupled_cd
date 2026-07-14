from __future__ import annotations

from typing import Mapping


def classify_external_win(
    margins: Mapping[str, float],
    *,
    overall_tolerance: float = 0.002,
) -> tuple[bool, bool]:
    required = {"S", "H", "T"}
    if set(margins) != required:
        raise ValueError(f"Expected exactly S/H/T margins, got {sorted(margins)}")
    ordinary = (
        float(margins["S"]) >= -overall_tolerance
        and float(margins["H"]) >= -overall_tolerance
        and float(margins["T"]) > 0.0
    )
    strict = (
        float(margins["S"]) >= 0.0
        and float(margins["H"]) >= 0.0
        and float(margins["T"]) > 0.0
    )
    return ordinary, strict
