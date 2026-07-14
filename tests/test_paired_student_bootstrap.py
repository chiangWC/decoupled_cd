from __future__ import annotations

import pandas as pd

from scripts.paired_student_bootstrap import paired_student_cluster_bootstrap


def test_clustered_bootstrap_preserves_pairing_and_finds_positive_delta() -> None:
    full = pd.DataFrame(
        {
            "stu_id": [0, 0, 1, 1, 2, 2, 3, 3],
            "exer_id": list(range(8)),
            "label": [0, 1, 0, 1, 0, 1, 0, 1],
            "prob": [0.1, 0.9, 0.2, 0.8, 0.15, 0.85, 0.25, 0.75],
            "coverage_bucket": ["zero"] * 8,
            "coverage_group": ["low_coverage"] * 8,
        }
    )
    control = full.copy()
    control["prob"] = [0.45, 0.55, 0.6, 0.4, 0.5, 0.5, 0.55, 0.45]
    result = paired_student_cluster_bootstrap(
        full=full,
        control=control,
        scope="bucket:zero",
        replicates=200,
        seed=2024,
    )
    assert result["delta_auc"] > 0.0
    assert result["ci_low"] > 0.0
    assert result["student_count"] == 4
