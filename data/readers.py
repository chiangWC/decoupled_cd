from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


INTERACTION_REQUIRED_COLUMNS = ("stu_id", "exer_id", "label")
Q_MATRIX_REQUIRED_COLUMNS = ("exer_id", "cpt_seq")


def _validate_columns(df: pd.DataFrame, required_columns: Iterable[str], source: Path) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"{source} is missing required columns: {missing_text}")


def read_interactions(csv_path: str | Path) -> pd.DataFrame:
    """
    Read a student interaction table and validate the minimum schema.
    """
    path = Path(csv_path)
    df = pd.read_csv(path)
    _validate_columns(df, INTERACTION_REQUIRED_COLUMNS, path)
    return df.copy()


def read_q_matrix(csv_path: str | Path) -> pd.DataFrame:
    """
    Read an exercise-to-concept table and validate the minimum schema.
    """
    path = Path(csv_path)
    df = pd.read_csv(path)
    _validate_columns(df, Q_MATRIX_REQUIRED_COLUMNS, path)
    return df.copy()
