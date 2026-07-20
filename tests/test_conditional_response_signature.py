from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "audit_conditional_response_signature.py"
SPEC = importlib.util.spec_from_file_location(
    "audit_conditional_response_signature", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _frame(student: str, rows: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_row_id": [f"{student}-{index}" for index in range(rows)],
            "stu_id": [student] * rows,
            "exer_id": [f"i{index}" for index in range(rows)],
            "cpt_seq": [f"c{index % 4}" for index in range(rows)],
            "label": [index % 2 for index in range(rows)],
        }
    )


def _profile(
    student: str,
    theta: float,
    labels: tuple[int, int] = (1, 0),
) -> object:
    support = pd.DataFrame(
        {
            "source_row_id": [f"{student}-support"],
            "stu_id": [student],
            "exer_id": ["support"],
            "cpt_seq": ["support_concept"],
            "label": [int(theta >= 0)],
        }
    )
    query = pd.DataFrame(
        {
            "source_row_id": [f"{student}-i1", f"{student}-i2"],
            "stu_id": [student, student],
            "exer_id": ["i1", "i2"],
            "cpt_seq": ["c", "c"],
            "label": list(labels),
        }
    )
    return MODULE.StudentProfile(
        student=student,
        support=support,
        query=query,
        theta=theta,
        raw_accuracy=float(theta >= 0),
        seen_concepts=frozenset({"support_concept"}),
        fold=0,
    )


class TestConditionalResponseSignature(unittest.TestCase):
    def test_theta_uses_fixed_beta_one_posterior_and_clipping(self) -> None:
        support = pd.DataFrame({"label": [1, 1, 0]})
        theta, raw = MODULE.support_theta(support)
        expected = np.log((3.0 / 5.0) / (2.0 / 5.0))
        self.assertAlmostEqual(theta, expected)
        self.assertAlmostEqual(raw, 2.0 / 3.0)
        all_correct = pd.DataFrame({"label": np.ones(1000, dtype=int)})
        clipped, _ = MODULE.support_theta(all_correct)
        self.assertLessEqual(clipped, 4.0)

    def test_optimizer_split_is_atomic_deterministic_and_five_fold(self) -> None:
        frame = pd.concat([_frame(f"s{index}") for index in range(100)], ignore_index=True)
        q_lookup = {f"i{index}": (f"c{index % 4}",) for index in range(30)}
        first, first_audit = MODULE.optimizer_profiles(frame, q_lookup=q_lookup)
        second, second_audit = MODULE.optimizer_profiles(
            frame.sample(frac=1.0, random_state=11), q_lookup=q_lookup
        )
        first_assignments = {profile.student: profile.fold for profile in first}
        second_assignments = {profile.student: profile.fold for profile in second}
        self.assertEqual(first_assignments, second_assignments)
        self.assertEqual(
            first_audit["fold_assignment_sha256"],
            second_audit["fold_assignment_sha256"],
        )
        self.assertEqual(set(first_audit["fold_counts"]), {"0", "1", "2", "3", "4"})
        for profile in first:
            support_items = set(profile.support["exer_id"].astype(str))
            query_items = set(profile.query["exer_id"].astype(str))
            self.assertFalse(support_items & query_items)

    def test_stable_rank_quintiles_keep_all_bins_when_edges_collapse(self) -> None:
        profiles = [_profile(f"s{index}", 0.0) for index in range(25)]
        binner = MODULE.RankQuintileBinner.fit(profiles, seed=2024)
        assigned = [
            binner.assign(student=profile.student, theta=profile.theta)
            for profile in profiles
        ]
        self.assertEqual(set(assigned), {0, 1, 2, 3, 4})
        self.assertEqual(binner.collapsed_edges, 3)

    def test_leave_item_out_exact_q_curve_and_singleton_zero(self) -> None:
        profiles = [
            _profile(f"s{index}", float(index - 2)) for index in range(5)
        ]
        items = ["i1", "i2", "i3"]
        item_index = {item: index for index, item in enumerate(items)}
        q_lookup = {"i1": ("c",), "i2": ("c",), "i3": ("singleton",)}
        statistics, _ = MODULE.build_reference_statistics(
            profiles,
            items=items,
            item_index=item_index,
            q_lookup=q_lookup,
        )
        self.assertTrue(np.allclose(statistics.signatures[item_index["i3"]], 0.0))
        # In every populated bin, i1=1 and i2=0. The i1 prototype therefore
        # excludes i1 and combines i2 only with the fixed beta-20 global prior.
        global_probability = 0.5
        prototype = (0.0 + 20.0 * global_probability) / 21.0
        item_probability = (1.0 + 20.0 * prototype) / 21.0
        expected = (1.0 / 21.0) * (
            MODULE._logit(item_probability) - MODULE._logit(prototype)
        )
        populated = statistics.signatures[item_index["i1"], :5]
        self.assertTrue(np.allclose(populated, expected))

    def test_derangement_has_no_eligible_fixed_point(self) -> None:
        items = ["a", "b", "c", "single"]
        q_lookup = {
            "a": ("q",),
            "b": ("q",),
            "c": ("q",),
            "single": ("other",),
        }
        mapping, audit = MODULE.make_derangement(items=items, q_lookup=q_lookup)
        self.assertIsNone(mapping["single"])
        for item in ("a", "b", "c"):
            self.assertNotEqual(mapping[item], item)
            self.assertEqual(q_lookup[mapping[item]], q_lookup[item])
        self.assertEqual(audit["eligible_fixed_points"], 0)

    def test_local_signature_has_declared_thirteen_dimensions(self) -> None:
        curve = np.zeros(len(MODULE.SIGNATURE_NAMES), dtype=float)
        curve[:5] = [0.0, 1.0, 3.0, 6.0, 10.0]
        curve[5:10] = [0.1, 0.2, 0.3, 0.4, 0.5]
        signature = MODULE.signature_at_bin(curve, 2)
        self.assertEqual(len(signature), 13)
        self.assertAlmostEqual(signature[10], 3.0)
        self.assertAlmostEqual(signature[11], 0.3)
        self.assertAlmostEqual(signature[12], 2.5)

    def test_sparse_design_and_capacity_match_full_width(self) -> None:
        examples = MODULE.ExampleSet.empty()
        for row in range(2):
            examples.common_numeric.append(np.arange(6, dtype=float) + row)
            examples.q_indices.append((row,))
            examples.item_indices.append(row)
            examples.theta.append(float(row))
            examples.full_signature.append(np.arange(13, dtype=float))
            examples.capacity_signature.append(np.arange(13, dtype=float)[::-1])
            examples.labels.append(row)
            examples.rows.append({"source_row_id": str(row)})
        matrices = {
            variant: MODULE.make_design_matrix(
                examples, variant=variant, num_items=2, num_concepts=2
            )
            for variant in MODULE.VARIANTS
        }
        self.assertTrue(all(sparse.isspmatrix_csr(value) for value in matrices.values()))
        self.assertEqual(matrices["direct"].shape[1], 8)
        self.assertEqual(matrices["strong_2pl"].shape[1], 12)
        self.assertEqual(matrices["full"].shape[1], 25)
        self.assertEqual(matrices["full"].shape, matrices["capacity"].shape)

    def test_preprocessing_scales_dense_blocks_but_not_sparse_indicators(self) -> None:
        fit = MODULE.ExampleSet.empty()
        validation = MODULE.ExampleSet.empty()
        for examples, count in ((fit, 10), (validation, 3)):
            for row in range(count):
                examples.common_numeric.append(
                    np.arange(6, dtype=float) + float(row)
                )
                examples.q_indices.append((row % 2,))
                examples.item_indices.append(row % 2)
                examples.theta.append(float(row) / 10.0)
                examples.full_signature.append(
                    np.arange(13, dtype=float) + float(row)
                )
                examples.capacity_signature.append(
                    np.arange(13, dtype=float)[::-1] - float(row)
                )
                examples.labels.append(row % 2)
                examples.rows.append({"source_row_id": str(row)})
        fit_matrices, validation_matrices, audit = (
            MODULE.make_preprocessed_designs(
                fit, validation, num_items=2, num_concepts=2
            )
        )
        del validation_matrices
        q_offset = len(MODULE.COMMON_NUMERIC_NAMES)
        item_offset = q_offset + 2
        strong = fit_matrices["strong_2pl"]
        for row in range(10):
            self.assertEqual(strong[row, q_offset + row % 2], 1.0)
            self.assertEqual(strong[row, item_offset + row % 2], 1.0)
        self.assertFalse(audit["q_and_item_one_hot_scaled"])
        self.assertFalse(audit["theta_item_interaction_scaled"])
        self.assertTrue(
            np.allclose(
                fit_matrices["direct"][:, :6].toarray(),
                fit_matrices["strong_2pl"][:, :6].toarray(),
            )
        )
        self.assertEqual(
            fit_matrices["full"].shape, fit_matrices["capacity"].shape
        )

    def test_gate_is_conjunctive_and_uses_one_control_per_dataset(self) -> None:
        def summary(name: str, auc: float, brier: float, ci: float, zero_auc: float, zero_brier: float) -> dict:
            return {
                "dataset": name,
                "stronger_control": "capacity",
                "delta_vs_stronger_control": {
                    "overall_auc": auc,
                    "overall_brier": brier,
                    "exact_zero_auc": zero_auc,
                    "exact_zero_brier": zero_brier,
                },
                "bootstrap": {"capacity": {"overall": {"ci_low": ci}}},
            }

        passing = [
            summary("ASSIST17", 0.003, 0.0001, 0.00001, 0.001, 0.0),
            summary("XES3G5M", 0.005, 0.0001, -0.001, 0.005, 0.0002),
        ]
        self.assertTrue(MODULE.evaluate_gate(passing)["passed"])
        failing = [dict(passing[0]), dict(passing[1])]
        failing[1] = summary("XES3G5M", 0.005, 0.0001, -0.001, 0.0049, 0.0002)
        self.assertFalse(MODULE.evaluate_gate(failing)["passed"])


if __name__ == "__main__":
    unittest.main()
