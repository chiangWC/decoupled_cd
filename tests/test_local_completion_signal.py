from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "audit_local_completion_signal.py"
SPEC = importlib.util.spec_from_file_location("audit_local_completion_signal", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _check_pseudo_holdout_removes_rows_touching_hidden_concept() -> None:
    frame = pd.DataFrame(
        {
            "stu_id": ["s"] * 10,
            "exer_id": [str(index) for index in range(10)],
            "cpt_seq": ["a,b", "a", "b", "c", "c", "d", "d", "e", "e", "f"],
            "label": [1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    holdout = MODULE.build_pseudo_holdout(
        frame,
        student="s",
        seed=2024,
        hide_fraction=0.20,
        min_history_rows=3,
        min_visible_concepts=2,
        min_hidden_concepts=1,
    )
    assert holdout is not None
    hidden = set(holdout.hidden_concepts)
    assert hidden
    assert all(
        not (hidden & set(MODULE._concepts(value)))
        for value in holdout.visible_rows["cpt_seq"]
    )
    assert all(holdout.target_attempts[concept] > 0 for concept in hidden)


def _check_long_form_q_is_unioned_and_attached_to_interactions() -> None:
    with tempfile.TemporaryDirectory() as raw_directory:
        q_path = Path(raw_directory) / "Q_matrix.csv"
        pd.DataFrame(
            {"exer_id": [1, 1, 2], "cpt_seq": [10, 11, 12]}
        ).to_csv(q_path, index=False)
        q_matrix, q_lookup = MODULE._load_q_union(q_path)
    assert len(q_matrix) == 2
    assert set(q_lookup["1"].split(",")) == {"10", "11"}
    interactions = pd.DataFrame(
        {
            "stu_id": ["s", "s"],
            "exer_id": ["1", "2"],
            "cpt_seq": ["10", "12"],
            "label": [1, 0],
        }
    )
    attached = MODULE._attach_q_union(interactions, q_lookup)
    assert set(attached.loc[0, "cpt_seq"].split(",")) == {"10", "11"}


def _check_student_hash_partition_is_deterministic_and_disjoint() -> None:
    students = [f"student-{index}" for index in range(100)]
    role = lambda student: (
        "eval"
        if MODULE.stable_fraction(2024, "local-completion-student", student) < 0.25
        else "fit"
    )
    first = {student: role(student) for student in students}
    second = {student: role(student) for student in reversed(students)}
    assert first == second
    fit = {student for student, value in first.items() if value == "fit"}
    evaluate = set(students) - fit
    assert fit
    assert evaluate
    assert not (fit & evaluate)



def _check_opportunity_gate_uses_stronger_control_delta_and_brier() -> None:
    assert MODULE.passes_opportunity_gate([0.005, 0.005], [0.0, 1e-4])[0]
    assert MODULE.passes_opportunity_gate([0.010, 0.0], [-0.001, 0.0])[0]
    assert not MODULE.passes_opportunity_gate([0.004, 0.004], [-0.001, -0.001])[0]
    assert not MODULE.passes_opportunity_gate([0.010, 0.005], [0.0, 1.01e-4])[0]


class TestLocalCompletionSignal(unittest.TestCase):
    test_pseudo_holdout_removes_rows_touching_hidden_concept = staticmethod(
        _check_pseudo_holdout_removes_rows_touching_hidden_concept
    )
    test_long_form_q_is_unioned_and_attached_to_interactions = staticmethod(
        _check_long_form_q_is_unioned_and_attached_to_interactions
    )
    test_student_hash_partition_is_deterministic_and_disjoint = staticmethod(
        _check_student_hash_partition_is_deterministic_and_disjoint
    )
    test_opportunity_gate_uses_stronger_control_delta_and_brier = staticmethod(
        _check_opportunity_gate_uses_stronger_control_delta_and_brier
    )
