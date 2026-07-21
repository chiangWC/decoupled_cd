from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from data.pool_protocol import derive_q_matrix
from scripts.audit_option_contrast_signal import (
    LoadedProtocol,
    OptionSignatureModel,
    build_pseudo_protocol,
    deterministic_aggregate_gate,
    fast_student_cluster_bootstrap,
    summarize_predictions,
)
from scripts.paired_student_bootstrap import paired_student_cluster_bootstrap


class OptionContrastSignalTest(unittest.TestCase):
    def protocol(self) -> LoadedProtocol:
        rows = []
        for student in range(50):
            for item in range(20):
                label = int((student + item) % 3 != 0)
                correct_option = 0
                selected_option = (
                    correct_option
                    if label
                    else 1 + ((student + 2 * item) % 3)
                )
                rows.append(
                    {
                        "source_row_id": f"{student}-{item}",
                        "stu_id": str(student),
                        "exer_id": str(item),
                        "cpt_seq": str(item),
                        "label": label,
                        "selected_option": selected_option,
                        "correct_option": correct_option,
                        "option_count": 4,
                    }
                )
        frame = pd.DataFrame(rows)
        q_matrix, _ = derive_q_matrix(frame)
        q_lookup = {
            str(row.exer_id): (str(row.exer_id),)
            for row in q_matrix.itertuples(index=False)
        }
        return LoadedProtocol(
            name="toy",
            train=frame,
            q_matrix=q_matrix,
            q_lookup=q_lookup,
            target_scope="bucket:zero",
            input_audit={},
        )

    def signature_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        reference_rows = []
        for student in range(30):
            group = student % 2
            for item in range(6):
                if item == 0:
                    label = 0
                    selected = 1 if group == 0 else 2
                else:
                    label = int((group == 0 and item < 3) or (group == 1 and item >= 3))
                    selected = 0 if label else 1 + ((student + item) % 3)
                reference_rows.append(
                    {
                        "stu_id": str(student),
                        "exer_id": str(item),
                        "cpt_seq": str(item),
                        "label": label,
                        "selected_option": selected,
                        "correct_option": 0,
                        "option_count": 4,
                    }
                )
        held_rows = []
        for student in range(30, 36):
            for item in range(6):
                label = 0 if item == 0 else int((student + item) % 2)
                held_rows.append(
                    {
                        "stu_id": str(student),
                        "exer_id": str(item),
                        "cpt_seq": str(item),
                        "label": label,
                        "selected_option": 1 + ((student + item) % 3) if not label else 0,
                        "correct_option": 0,
                        "option_count": 4,
                    }
                )
        return pd.DataFrame(reference_rows), pd.DataFrame(held_rows)

    def test_pseudo_mask_is_exact_k_label_and_option_blind(self) -> None:
        protocol = self.protocol()
        pseudo = build_pseudo_protocol(protocol)
        self.assertEqual(len(pseudo.support), 50 * 16)
        self.assertEqual(len(pseudo.target_public), 50 * 4)
        self.assertFalse(
            {
                "label",
                "selected_option",
                "correct_option",
                "option_count",
            }
            & set(pseudo.target_public.columns)
        )
        changed = protocol.train.copy()
        changed["label"] = 1 - changed["label"]
        changed["selected_option"] = (
            changed["selected_option"].astype(int) + 1
        ) % 4
        changed_protocol = LoadedProtocol(
            name=protocol.name,
            train=changed,
            q_matrix=protocol.q_matrix,
            q_lookup=protocol.q_lookup,
            target_scope=protocol.target_scope,
            input_audit={},
        )
        changed_pseudo = build_pseudo_protocol(changed_protocol)
        left = pseudo.target_public[
            ["source_row_id", "fold"]
        ].sort_values("source_row_id").reset_index(drop=True)
        right = changed_pseudo.target_public[
            ["source_row_id", "fold"]
        ].sort_values("source_row_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right)

    def test_shuffle_does_not_read_held_real_wrong_option(self) -> None:
        reference, held = self.signature_frames()
        q_lookup = {str(item): (str(item),) for item in range(6)}
        model = OptionSignatureModel(
            dataset="toy",
            fold=0,
            reference_support=reference,
            q_lookup=q_lookup,
        )
        first, first_audit = model.shuffled_options(
            held,
            leave_one_student_out=False,
        )
        changed = held.copy()
        wrong = changed["label"] == 0
        changed.loc[wrong, "selected_option"] = changed.loc[
            wrong, "selected_option"
        ].map({1: 2, 2: 3, 3: 1})
        second, second_audit = model.shuffled_options(
            changed,
            leave_one_student_out=False,
        )
        np.testing.assert_array_equal(first, second)
        self.assertFalse(first_audit["hash_uses_real_option"])
        self.assertEqual(
            first_audit["uniform_legal_fallback"],
            second_audit["uniform_legal_fallback"],
        )

    def test_wrong_option_only_changes_option_residual_path(self) -> None:
        reference, held = self.signature_frames()
        q_lookup = {str(item): (str(item),) for item in range(6)}
        model = OptionSignatureModel(
            dataset="toy",
            fold=0,
            reference_support=reference,
            q_lookup=q_lookup,
        )
        actual = held["selected_option"].to_numpy(dtype=int)
        changed = actual.copy()
        wrong = held["label"].to_numpy(dtype=int) == 0
        changed[wrong] = np.where(changed[wrong] == 1, 2, 1)
        state_actual = model.states(
            held,
            chosen_options=actual,
            leave_one_student_out=False,
        )
        state_changed = model.states(
            held,
            chosen_options=changed,
            leave_one_student_out=False,
        )
        np.testing.assert_allclose(
            state_actual.direct_state,
            state_changed.direct_state,
        )
        self.assertGreater(
            float(
                np.max(
                    np.abs(
                        state_actual.option_residual
                        - state_changed.option_residual
                    )
                )
            ),
            0.0,
        )

    def test_fast_bootstrap_matches_reference_implementation(self) -> None:
        students = np.repeat(np.arange(20).astype(str), 2)
        labels = np.tile([0, 1], 20)
        full = np.linspace(0.05, 0.95, len(labels))
        control = np.roll(full, 3)
        fast = fast_student_cluster_bootstrap(
            labels=labels,
            students=students,
            full_probability=full,
            control_probability=control,
            replicates=200,
            seed=2024,
        )
        full_frame = pd.DataFrame(
            {
                "stu_id": students,
                "exer_id": np.arange(len(labels)).astype(str),
                "label": labels,
                "prob": full,
                "coverage_bucket": "zero",
            }
        )
        control_frame = full_frame.copy()
        control_frame["prob"] = control
        reference = paired_student_cluster_bootstrap(
            full=full_frame,
            control=control_frame,
            scope="bucket:zero",
            replicates=200,
            seed=2024,
        )
        self.assertAlmostEqual(fast["delta_auc"], reference["delta_auc"])
        self.assertAlmostEqual(fast["ci_low"], reference["ci_low"])
        self.assertAlmostEqual(fast["ci_high"], reference["ci_high"])

    def test_control_selection_is_once_by_target_auc(self) -> None:
        rows = pd.DataFrame(
            {
                "stu_id": ["0", "0", "1", "1", "2", "2"],
                "exer_id": [str(value) for value in range(6)],
                "label": [0, 1, 0, 1, 0, 1],
                "coverage_bucket": ["zero"] * 6,
                "coverage_group": ["low_coverage"] * 6,
                "prob_direct": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
                "prob_shuffle": [0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
                "prob_full": [0.05, 0.95, 0.1, 0.9, 0.2, 0.8],
            }
        )
        summary = summarize_predictions(
            rows,
            target_scope="bucket:zero",
        )
        self.assertEqual(summary["stronger_control"], "direct")
        self.assertTrue(summary["control_selected_once_by_target_auc"])

    def test_aggregate_gate_requires_ednet_and_two_datasets(self) -> None:
        summaries = {
            name: {
                "deterministic_dataset_pass": name != "nips34",
                "deltas_full_minus_control": {
                    "target_auc": 0.011 if name == "ednet" else 0.006
                },
            }
            for name in ("nips34", "ednet", "enem")
        }
        gate = deterministic_aggregate_gate(summaries)
        self.assertTrue(gate["bootstrap_needed"])
        summaries["ednet"]["deterministic_dataset_pass"] = False
        gate = deterministic_aggregate_gate(summaries)
        self.assertFalse(gate["bootstrap_needed"])


if __name__ == "__main__":
    unittest.main()
