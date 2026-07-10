from __future__ import annotations

import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from models import decoupled_cdm_v2
from models.decoupled_cdm_v2 import DecoupledCDMV2
from scripts import analyze_prediction_slices
from scripts import train as train_script
from trainers.engine import _train_student_recompute_minibatch_epoch


class V2SupportAdaptiveTests(unittest.TestCase):
    def require_support_api(self) -> None:
        self.assertIn(
            "dual_graph_support_adaptive",
            inspect.signature(DecoupledCDMV2).parameters,
        )
        self.assertTrue(
            hasattr(decoupled_cdm_v2, "_compute_ukc_reachability_support")
        )
        self.assertTrue(
            hasattr(decoupled_cdm_v2, "_legacy_local_graph_density_feature")
        )

    @staticmethod
    def model_kwargs() -> dict[str, int]:
        return {
            "num_students": 2,
            "num_exercises": 3,
            "num_concepts": 3,
            "concept_dim": 4,
        }

    @staticmethod
    def tensors() -> dict[str, torch.Tensor | None]:
        q_matrix = torch.eye(3, dtype=torch.float32)
        concept_graph = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        student_tkc_mask = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
        return {
            "q_matrix": q_matrix,
            "concept_graph": concept_graph,
            "prerequisite_graph": None,
            "similarity_graph": None,
            "student_exercise_mask": torch.tensor(
                [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]
            ),
            "response_matrix": torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]
            ),
            "student_tkc_mask": student_tkc_mask,
            "student_ukc_mask": 1.0 - student_tkc_mask,
            "student_concept_evidence": torch.zeros(2, 3, 6),
            "exercise_evidence": torch.zeros(3, 6),
            "interaction_student_ids": torch.tensor([0, 0, 1, 1]),
            "interaction_exercise_ids": torch.tensor([0, 1, 1, 2]),
            "interaction_labels": torch.tensor([1.0, 0.0, 1.0, 1.0]),
        }

    @staticmethod
    def forward(model: DecoupledCDMV2, tensors, *, use_student_subset=False):
        return model(
            q_matrix=tensors["q_matrix"],
            concept_graph=tensors["concept_graph"],
            prerequisite_graph=tensors["prerequisite_graph"],
            similarity_graph=tensors["similarity_graph"],
            student_exercise_mask=tensors["student_exercise_mask"],
            response_matrix=tensors["response_matrix"],
            student_tkc_mask=tensors["student_tkc_mask"],
            student_ukc_mask=tensors["student_ukc_mask"],
            student_concept_evidence=tensors["student_concept_evidence"],
            exercise_evidence=tensors["exercise_evidence"],
            target_student_ids=tensors["interaction_student_ids"],
            target_exercise_ids=tensors["interaction_exercise_ids"],
            use_student_subset=use_student_subset,
        )

    def test_legacy_normalized_row_sum_density_is_constant_log_two(self) -> None:
        self.require_support_api()
        graph = torch.tensor(
            [
                [0.5, 0.5, 0.0],
                [0.1, 0.7, 0.2],
                [0.0, 0.25, 0.75],
            ]
        )
        tkc = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])

        density = decoupled_cdm_v2._legacy_local_graph_density_feature(
            graph, tkc
        )

        expected = torch.full((2, 1), torch.log(torch.tensor(2.0)))
        self.assertTrue(torch.allclose(density, expected))

    def test_identity_graph_has_zero_support_and_no_ukc_is_one(self) -> None:
        self.require_support_api()
        graph = torch.eye(3)
        original = graph.clone()
        tkc = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        ukc = 1.0 - tkc

        support = decoupled_cdm_v2._compute_ukc_reachability_support(
            graph, tkc, ukc
        )

        self.assertTrue(torch.equal(graph, original))
        self.assertTrue(torch.equal(support, torch.tensor([0.0, 1.0])))

    def test_support_uses_target_source_topology_not_edge_magnitude(self) -> None:
        self.require_support_api()
        tkc = torch.tensor([[1.0, 0.0, 0.0]])
        ukc = 1.0 - tkc
        half_topology = torch.tensor(
            [[1.0, 0.0, 0.0], [0.2, 0.8, 0.0], [0.0, 0.0, 1.0]]
        )
        same_topology_new_weights = torch.tensor(
            [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0]]
        )
        full_topology = torch.tensor(
            [[1.0, 0.0, 0.0], [0.2, 0.8, 0.0], [0.3, 0.0, 0.7]]
        )

        half = decoupled_cdm_v2._compute_ukc_reachability_support(
            half_topology, tkc, ukc
        )
        reweighted = decoupled_cdm_v2._compute_ukc_reachability_support(
            same_topology_new_weights, tkc, ukc
        )
        full = decoupled_cdm_v2._compute_ukc_reachability_support(
            full_topology, tkc, ukc
        )

        self.assertTrue(torch.equal(half, torch.tensor([0.5])))
        self.assertTrue(torch.equal(reweighted, half))
        self.assertTrue(torch.equal(full, torch.tensor([1.0])))

    def test_zero_support_weight_is_exactly_equivalent_to_mo1(self) -> None:
        self.require_support_api()
        torch.manual_seed(17)
        mo1 = DecoupledCDMV2(**self.model_kwargs(), dual_graph=True)
        torch.manual_seed(17)
        support_model = DecoupledCDMV2(
            **self.model_kwargs(), dual_graph_support_adaptive=True
        )

        self.assertEqual(mo1.dg_gate.in_features, 8)
        self.assertEqual(support_model.dg_gate.in_features, 8)
        self.assertIsNone(mo1.support_weight)
        self.assertEqual(float(support_model.support_weight), 0.0)
        self.assertEqual(
            set(support_model.state_dict()) - set(mo1.state_dict()),
            {"support_weight"},
        )
        for name, value in mo1.state_dict().items():
            self.assertTrue(torch.equal(value, support_model.state_dict()[name]), name)

        tensors = self.tensors()
        mo1.eval()
        support_model.eval()
        with torch.no_grad():
            expected = self.forward(mo1, tensors)
            actual = self.forward(support_model, tensors)

        self.assertTrue(torch.equal(expected.cognitive_probs, actual.cognitive_probs))
        self.assertTrue(torch.equal(expected.probs, actual.probs))

    def test_support_subset_indexing_matches_full_forward(self) -> None:
        self.require_support_api()
        model = DecoupledCDMV2(
            **self.model_kwargs(), dual_graph_support_adaptive=True
        )
        with torch.no_grad():
            model.support_weight.fill_(1.25)
        model.eval()
        tensors = self.tensors()
        tensors["interaction_student_ids"] = torch.tensor([1, 1])
        tensors["interaction_exercise_ids"] = torch.tensor([1, 2])

        with torch.no_grad():
            full = self.forward(model, tensors, use_student_subset=False)
            subset = self.forward(model, tensors, use_student_subset=True)

        self.assertTrue(torch.allclose(full.probs, subset.probs, atol=1e-7, rtol=0.0))

    def test_support_checkpoint_round_trips_through_summary_loader(self) -> None:
        self.require_support_api()
        support_model = DecoupledCDMV2(
            **self.model_kwargs(), dual_graph_support_adaptive=True
        )
        with torch.no_grad():
            support_model.support_weight.fill_(0.75)
        legacy_model = DecoupledCDMV2(**self.model_kwargs(), dual_graph=True)
        bundles = {
            "train": SimpleNamespace(
                num_students=2,
                num_exercises=3,
                num_concepts=3,
            )
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            support_path = root / "support.pt"
            legacy_path = root / "legacy.pt"
            torch.save(support_model.state_dict(), support_path)
            torch.save(legacy_model.state_dict(), legacy_path)

            loaded_support = analyze_prediction_slices.load_model(
                summary={
                    "model": "v2",
                    "v2_dual_graph_support_adaptive": True,
                },
                checkpoint_path=str(support_path),
                bundles=bundles,
                concept_dim=4,
                device="cpu",
            )
            loaded_legacy = analyze_prediction_slices.load_model(
                summary={"model": "v2", "v2_dual_graph": True},
                checkpoint_path=str(legacy_path),
                bundles=bundles,
                concept_dim=4,
                device="cpu",
            )

        self.assertTrue(loaded_support.dual_graph_support_adaptive)
        self.assertEqual(float(loaded_support.support_weight), 0.75)
        self.assertFalse(loaded_legacy.dual_graph_support_adaptive)
        self.assertIsNone(loaded_legacy.support_weight)

    def test_support_and_tr2_execute_one_cpu_optimizer_step(self) -> None:
        self.require_support_api()
        model = DecoupledCDMV2(
            **self.model_kwargs(), dual_graph_support_adaptive=True
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        tensors = self.tensors()

        stats = _train_student_recompute_minibatch_epoch(
            model=model,
            tensors=tensors,
            optimizer=optimizer,
            student_batch_size=2,
            consistency_weight=0.5,
        )

        self.assertEqual(stats.optimizer_steps, 1)
        self.assertTrue(torch.isfinite(torch.tensor(stats.mean_loss)))
        self.assertIsNotNone(model.support_weight.grad)
        self.assertTrue(torch.isfinite(model.support_weight.grad))

    @staticmethod
    def parse_and_validate(*arguments: str):
        with mock.patch.object(
            sys, "argv", ["train.py", *arguments]
        ):
            try:
                args = train_script.parse_args()
            except SystemExit as exc:
                raise AssertionError("train CLI rejected a required argument") from exc
        train_script.validate_model_args(args)
        return args

    def test_cli_accepts_support_with_tr2_student_minibatch(self) -> None:
        args = self.parse_and_validate(
            "--model",
            "v2",
            "--v2-dual-graph-support-adaptive",
            "--v2-consistency-weight",
            "0.5",
            "--training-mode",
            "student_recompute_minibatch",
            "--student-batch-size",
            "2",
        )

        self.assertTrue(args.v2_dual_graph_support_adaptive)
        self.assertEqual(args.v2_consistency_weight, 0.5)

    def test_cli_rejects_ambiguous_support_gate_modes(self) -> None:
        for conflicting_flag in ("--v2-dual-graph-adaptive", "--v2-router"):
            with self.subTest(conflicting_flag=conflicting_flag):
                with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                    self.parse_and_validate(
                        "--model",
                        "v2",
                        "--v2-dual-graph-support-adaptive",
                        conflicting_flag,
                    )

    def test_cli_rejects_tr2_outside_student_recompute_minibatch(self) -> None:
        for training_mode in ("full_batch", "recompute_minibatch"):
            arguments = [
                "--model",
                "v2",
                "--v2-dual-graph-support-adaptive",
                "--v2-consistency-weight",
                "0.5",
                "--training-mode",
                training_mode,
            ]
            if training_mode == "recompute_minibatch":
                arguments.extend(["--batch-size", "2"])
            with self.subTest(training_mode=training_mode):
                with self.assertRaisesRegex(
                    ValueError, "student_recompute_minibatch"
                ):
                    self.parse_and_validate(*arguments)


if __name__ == "__main__":
    unittest.main()
