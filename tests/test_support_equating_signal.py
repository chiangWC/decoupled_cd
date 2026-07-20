from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "audit_support_equating_signal.py"
SPEC = importlib.util.spec_from_file_location("audit_support_equating_signal", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _check_q_union_uses_all_long_form_rows() -> None:
    with tempfile.TemporaryDirectory() as raw_directory:
        path = Path(raw_directory) / "Q_matrix.csv"
        pd.DataFrame(
            {"exer_id": [1, 1, 2], "cpt_seq": [10, 11, 12]}
        ).to_csv(path, index=False)
        q_matrix, lookup = MODULE._load_q_union(path)
    assert len(q_matrix) == 2
    assert set(lookup["1"]) == {"10", "11"}


def _check_optimizer_split_is_atomic_and_deterministic() -> None:
    rows = []
    for student in ("a", "b", "c"):
        for exercise in range(30):
            rows.append(
                {
                    "stu_id": student,
                    "exer_id": str(exercise),
                    "cpt_seq": str(exercise % 5),
                    "label": exercise % 2,
                }
            )
    frame, _ = MODULE.canonicalize_interactions(pd.DataFrame(rows))
    first_support, first_query, first_audit = MODULE.split_optimizer_support_query(
        frame,
        seed=2024,
        query_fraction=0.25,
        min_support_rows=10,
        min_query_rows=3,
    )
    second_support, second_query, second_audit = MODULE.split_optimizer_support_query(
        frame.sample(frac=1.0, random_state=7),
        seed=2024,
        query_fraction=0.25,
        min_support_rows=10,
        min_query_rows=3,
    )
    first_support_groups = set(
        zip(first_support["stu_id"], first_support["exer_id"], strict=True)
    )
    first_query_groups = set(
        zip(first_query["stu_id"], first_query["exer_id"], strict=True)
    )
    assert not (first_support_groups & first_query_groups)
    assert first_support_groups == set(
        zip(second_support["stu_id"], second_support["exer_id"], strict=True)
    )
    assert first_query_groups == set(
        zip(second_query["stu_id"], second_query["exer_id"], strict=True)
    )
    assert first_audit == second_audit


def _check_item_statistics_leave_the_entire_student_out() -> None:
    frame = pd.DataFrame(
        {
            "stu_id": ["s1", "s2", "s1", "s2"],
            "exer_id": ["i1", "i1", "i2", "i2"],
            "cpt_seq": ["c1", "c1", "c2", "c2"],
            "label": [1, 0, 1, 1],
        }
    )
    store = MODULE.build_item_count_store(frame)
    statistics = MODULE.item_statistics(
        store, leave_student_out="s1", prior_strength=1.0
    )
    i1 = store.item_index["i1"]
    i2 = store.item_index["i2"]
    assert statistics.attempts[i1] == 1
    assert statistics.attempts[i2] == 1
    assert np.isclose(statistics.global_rate, 0.5)
    assert np.isclose(statistics.ease[i1], 0.25)
    assert np.isclose(statistics.ease[i2], 0.75)


def _check_equating_features_only_extend_the_shared_base() -> None:
    support = pd.DataFrame(
        {
            "stu_id": ["s"] * 10,
            "exer_id": [f"i{index}" for index in range(10)],
            "cpt_seq": [f"c{index % 3}" for index in range(10)],
            "label": [0, 1] * 5,
        }
    )
    store = MODULE.build_item_count_store(support)
    statistics = MODULE.item_statistics(
        store, leave_student_out=None, prior_strength=1.0
    )
    q_lookup = {f"i{index}": (f"c{index % 3}",) for index in range(10)}
    profile = MODULE.build_support_profile(
        support,
        statistics=statistics,
        item_index=store.item_index,
        q_lookup=q_lookup,
        num_concepts=3,
        bin_confidence_strength=5.0,
    )
    base, full = MODULE.build_row_features(
        profile,
        target_item="i0",
        target_concepts=("c0",),
        statistics=statistics,
        item_index=store.item_index,
        local_bandwidth=0.1,
        local_confidence_strength=5.0,
    )
    assert len(base) == len(MODULE.BASE_NUMERIC_NAMES)
    assert len(full) == len(MODULE.BASE_NUMERIC_NAMES) + len(
        MODULE.EQUATING_NUMERIC_NAMES
    )
    assert np.allclose(full[: len(base)], base)


def _check_capacity_control_matches_equated_width() -> None:
    base = np.arange(18, dtype=float).reshape(3, 6)
    capacity = MODULE._capacity_numeric(base)
    assert capacity.shape == (
        3,
        len(MODULE.BASE_NUMERIC_NAMES) + len(MODULE.EQUATING_NUMERIC_NAMES),
    )
    assert np.allclose(capacity[:, :6], base)


def _check_gate_requires_two_auc_gains_and_noninferior_brier() -> None:
    assert MODULE.passes_gate([0.003, 0.005], [0.0, 1e-4])[0]
    assert not MODULE.passes_gate([0.0029, 0.01], [0.0, 0.0])[0]
    assert not MODULE.passes_gate([0.003, 0.005], [0.0, 1.01e-4])[0]


class TestSupportEquatingSignal(unittest.TestCase):
    test_q_union_uses_all_long_form_rows = staticmethod(
        _check_q_union_uses_all_long_form_rows
    )
    test_optimizer_split_is_atomic_and_deterministic = staticmethod(
        _check_optimizer_split_is_atomic_and_deterministic
    )
    test_item_statistics_leave_the_entire_student_out = staticmethod(
        _check_item_statistics_leave_the_entire_student_out
    )
    test_equating_features_only_extend_the_shared_base = staticmethod(
        _check_equating_features_only_extend_the_shared_base
    )
    test_capacity_control_matches_equated_width = staticmethod(
        _check_capacity_control_matches_equated_width
    )
    test_gate_requires_two_auc_gains_and_noninferior_brier = staticmethod(
        _check_gate_requires_two_auc_gains_and_noninferior_brier
    )
