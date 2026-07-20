from __future__ import annotations

import pandas as pd

from scripts.analyze_story_mechanisms import (
    coverage_bucket,
    q_group_statistics,
    student_history_features,
)


def test_coverage_bucket_uses_train_history_only() -> None:
    q_map = {
        "10": frozenset({"a"}),
        "11": frozenset({"a", "b"}),
        "12": frozenset({"c"}),
    }
    train = pd.DataFrame(
        {"stu_id": ["0", "1"], "exer_id": ["10", "12"], "label": [1, 0]}
    )
    target = pd.DataFrame(
        {
            "stu_id": ["0", "0", "0"],
            "exer_id": ["10", "11", "12"],
            "label": [1, 1, 0],
        }
    )
    assert coverage_bucket(train, target, q_map).tolist() == [
        "full",
        "partial",
        "zero",
    ]


def test_history_item_ease_excludes_current_student() -> None:
    q_map = {"10": frozenset({"a"})}
    train = pd.DataFrame(
        {
            "stu_id": ["0", "0", "1", "1"],
            "exer_id": ["10", "10", "10", "10"],
            "label": [1, 1, 0, 0],
        }
    )
    result = student_history_features(train, q_map).set_index("stu_id")
    assert result.loc["0", "mean_history_item_ease"] < 0.5
    assert result.loc["1", "mean_history_item_ease"] > 0.5


def test_same_q_groups_require_two_well_observed_items() -> None:
    q_map = {
        "10": frozenset({"a"}),
        "11": frozenset({"a"}),
        "12": frozenset({"b"}),
    }
    rows = []
    for index in range(20):
        rows.append((str(index), "10", 1))
        rows.append((str(index), "11", 0))
        rows.append((str(index), "12", index % 2))
    train = pd.DataFrame(rows, columns=["stu_id", "exer_id", "label"])
    groups, _, items = q_group_statistics(train, q_map)
    assert len(groups) == 1
    assert groups.iloc[0]["eligible_items"] == 2
    assert set(items["exer_id"]) == {"10", "11"}
    assert groups.iloc[0]["item_rate_range"] > 0.5
