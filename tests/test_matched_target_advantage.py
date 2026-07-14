from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analyze_matched_target_advantage import matched_effect


def _frame() -> pd.DataFrame:
    rows = []
    for unit in range(6):
        for group, bucket in (("target", "zero"), ("reference", "full")):
            for label in (0, 1):
                external_prob = float(
                    np.clip(
                        0.5 + (label - 0.5) * 0.1 + (unit - 2.5) * 0.08,
                        0.05,
                        0.95,
                    )
                )
                if group == "target":
                    ours_prob = 0.1 if label == 0 else 0.9
                else:
                    ours_prob = external_prob
                rows.append(
                    {
                        "stu_id": unit,
                        "exer_id": unit,
                        "label": label,
                        "ours_prob": ours_prob,
                        "external_prob": external_prob,
                        "coverage_bucket": bucket,
                        "brier_advantage": (label - external_prob) ** 2
                        - (label - ours_prob) ** 2,
                        "log_loss_advantage": -(
                            label * np.log(external_prob)
                            + (1 - label) * np.log(1 - external_prob)
                        )
                        + (
                            label * np.log(ours_prob)
                            + (1 - label) * np.log(1 - ours_prob)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def test_matched_effect_detects_target_specific_advantage() -> None:
    result = matched_effect(
        _frame(),
        match_name="student",
        target_bucket="zero",
        reference_bucket="full",
        bootstrap=100,
        seed=7,
    )
    assert result["eligible_units"] == 6
    assert result["brier_advantage_did"] > 0.0
    assert result["brier_advantage_did_ci_low"] > 0.0
    assert result["log_loss_advantage_did"] > 0.0
    assert result["auc_margin_did"] > 0.0


def test_matched_effect_drops_units_without_both_groups() -> None:
    frame = _frame()
    incomplete = frame.loc[
        ~((frame["stu_id"] == 0) & (frame["coverage_bucket"] == "zero"))
    ]
    result = matched_effect(
        incomplete,
        match_name="student",
        target_bucket="zero",
        reference_bucket="full",
        bootstrap=20,
        seed=11,
    )
    assert result["eligible_units"] == 5
