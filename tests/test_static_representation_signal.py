from __future__ import annotations

import pandas as pd

from scripts.audit_static_representation_signal import (
    aggregate_gate,
    summarize_overall,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stu_id": ["a", "a", "b", "b", "c", "c"],
            "label": [0, 1, 0, 1, 0, 1],
            "coverage_bucket": ["zero"] * 6,
            "coverage_group": ["low_coverage"] * 6,
            "prob_q_only": [0.2, 0.8, 0.7, 0.3, 0.6, 0.4],
            "prob_full": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
            "prob_shuffle_0": [0.25, 0.75, 0.65, 0.35, 0.55, 0.45],
            "prob_shuffle_1": [0.3, 0.7, 0.6, 0.4, 0.55, 0.45],
            "prob_shuffle_2": [0.35, 0.65, 0.6, 0.4, 0.55, 0.45],
        }
    )


def test_summary_selects_one_stronger_overall_control() -> None:
    summary = summarize_overall(
        _predictions(),
        target_scope="bucket:zero",
    )
    assert summary["stronger_control"] in {
        "q_only",
        "shuffle_0",
        "shuffle_1",
        "shuffle_2",
    }
    assert summary["full_overall_auc_exceeds_every_control"] is True
    assert summary["deltas_full_minus_control"]["overall_brier"] < 0


def test_aggregate_requires_new_opportunity() -> None:
    summaries = {
        "assist09": {
            "deterministic_dataset_pass": True,
            "deltas_full_minus_control": {"overall_auc": 0.02},
        },
        "junyi": {
            "deterministic_dataset_pass": True,
            "deltas_full_minus_control": {"overall_auc": 0.006},
        },
        "nips34": {"deterministic_dataset_pass": False},
        "xes3g5m": {"deterministic_dataset_pass": False},
    }
    gate = aggregate_gate(summaries)
    assert gate["at_least_two_datasets"] is True
    assert gate["new_xes_or_nips_required"] is False
    assert gate["bootstrap_needed"] is False


def test_aggregate_activates_bootstrap_with_xes() -> None:
    summaries = {
        "assist09": {
            "deterministic_dataset_pass": True,
            "deltas_full_minus_control": {"overall_auc": 0.02},
        },
        "junyi": {"deterministic_dataset_pass": False},
        "nips34": {"deterministic_dataset_pass": False},
        "xes3g5m": {
            "deterministic_dataset_pass": True,
            "deltas_full_minus_control": {"overall_auc": 0.006},
        },
    }
    gate = aggregate_gate(summaries)
    assert gate["at_least_two_datasets"] is True
    assert gate["new_xes_or_nips_required"] is True
    assert gate["one_overall_delta_at_least_0_010"] is True
    assert gate["bootstrap_needed"] is True
