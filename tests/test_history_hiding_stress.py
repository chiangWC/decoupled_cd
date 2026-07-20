from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import torch

from scripts.evaluate_history_hiding_stress import (
    fixed_coverage_masks,
    history_mask_hash,
    mask_train_history_interactions,
)


def test_history_masks_are_nested_for_a_fixed_seed() -> None:
    frame = pd.DataFrame(
        {
            "row_id": list(range(40)),
            "stu_id": [index // 10 for index in range(40)],
            "exer_id": list(range(40)),
            "label": [index % 2 for index in range(40)],
        }
    )
    keep_80 = mask_train_history_interactions(frame, hide_ratio=0.2, seed=17)
    keep_60 = mask_train_history_interactions(frame, hide_ratio=0.4, seed=17)
    keep_40 = mask_train_history_interactions(frame, hide_ratio=0.6, seed=17)
    keep_20 = mask_train_history_interactions(frame, hide_ratio=0.8, seed=17)
    assert set(keep_20["row_id"]) <= set(keep_40["row_id"])
    assert set(keep_40["row_id"]) <= set(keep_60["row_id"])
    assert set(keep_60["row_id"]) <= set(keep_80["row_id"])
    repeated = mask_train_history_interactions(frame, hide_ratio=0.4, seed=17)
    assert history_mask_hash(keep_60) == history_mask_hash(repeated)


def test_fixed_coverage_masks_use_original_history() -> None:
    bundle = SimpleNamespace(
        q_matrix_tensor=torch.tensor([[1, 0], [1, 1], [0, 1]]),
        interaction_exercise_ids=torch.tensor([0, 1, 2]),
        interaction_student_ids=torch.tensor([0, 0, 0]),
        student_tkc_mask=torch.tensor([[1, 0]]),
    )
    masks = fixed_coverage_masks(bundle)
    assert masks["full"].tolist() == [True, False, False]
    assert masks["partial"].tolist() == [False, True, False]
    assert masks["zero"].tolist() == [False, False, True]
