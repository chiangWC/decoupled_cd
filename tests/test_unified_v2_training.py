from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from models import UnifiedArchitectureSpec, UnifiedDecoupledCDM
from scripts import analyze_prediction_slices
from scripts import evaluate_doa
from scripts import train as train_script
from trainers.engine import (
    _train_full_batch_epoch,
    _train_student_recompute_minibatch_epoch,
    train_model,
)


class UnifiedV2TrainingTests(unittest.TestCase):
    @staticmethod
    def parse_and_validate(*arguments: str):
        with mock.patch.object(sys, "argv", ["train.py", *arguments]):
            try:
                args = train_script.parse_args()
            except SystemExit as exc:
                raise AssertionError("train CLI rejected unified_v2") from exc
        train_script.validate_model_args(args)
        return args

    @staticmethod
    def tensors() -> dict[str, torch.Tensor | None]:
        evidence = torch.zeros(2, 3, 6)
        evidence[0, 0, :2] = torch.tensor([4.0, 3.0])
        evidence[0, 1, :2] = torch.tensor([3.0, 1.0])
        evidence[1, 1, :2] = torch.tensor([5.0, 4.0])
        evidence[1, 2, :2] = torch.tensor([3.0, 0.0])
        tkc = (evidence[..., 0] > 0).float()
        return {
            "q_matrix": torch.eye(3),
            "concept_graph": torch.eye(3),
            "prerequisite_graph": None,
            "similarity_graph": None,
            "student_exercise_mask": torch.tensor(
                [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]
            ),
            "response_matrix": torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            ),
            "student_tkc_mask": tkc,
            "student_ukc_mask": 1.0 - tkc,
            "student_concept_evidence": evidence,
            "exercise_evidence": torch.zeros(3, 6),
            "interaction_student_ids": torch.tensor([0, 0, 1, 1]),
            "interaction_exercise_ids": torch.tensor([0, 1, 1, 2]),
            "interaction_labels": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        }

    @staticmethod
    def model(
        *, inference: str = "prior", composer: str = "mask"
    ) -> UnifiedDecoupledCDM:
        return UnifiedDecoupledCDM(
            num_students=2,
            num_exercises=3,
            num_concepts=3,
            dim=4,
            architecture=UnifiedArchitectureSpec(
                inference=inference,
                composer=composer,
            ),
        )

    @staticmethod
    def forward(model: UnifiedDecoupledCDM, tensors):
        return model(
            q_matrix=tensors["q_matrix"],
            concept_graph=tensors["concept_graph"],
            student_exercise_mask=tensors["student_exercise_mask"],
            response_matrix=tensors["response_matrix"],
            student_tkc_mask=tensors["student_tkc_mask"],
            student_ukc_mask=tensors["student_ukc_mask"],
            student_concept_evidence=tensors["student_concept_evidence"],
            target_student_ids=tensors["interaction_student_ids"],
            target_exercise_ids=tensors["interaction_exercise_ids"],
        )

    def test_cli_accepts_unified_and_rejects_invalid_architecture(self) -> None:
        args = self.parse_and_validate("--model", "unified_v2")
        self.assertEqual(args.unified_inference, "prior")
        self.assertEqual(args.unified_composer, "mask")

        with self.assertRaisesRegex(
            ValueError, "coverage composer requires graph inference"
        ):
            self.parse_and_validate(
                "--model",
                "unified_v2",
                "--unified-inference",
                "prior",
                "--unified-composer",
                "coverage",
            )

    def test_unified_requires_positive_mastery_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            self.parse_and_validate(
                "--model",
                "unified_v2",
                "--unified-mastery-loss-weight",
                "0",
            )

    def test_unified_rejects_legacy_v1_and_v2_flags(self) -> None:
        for legacy_flag in (
            "--high-concept-logit-adapter",
            "--v2-ukc-propagation",
            "--v2-dual-graph",
            "--v2-dual-graph-adaptive",
            "--v2-router",
            "--v2-attn-readout",
            "--v2-irt-head",
            "--v2-consistency-adaptive",
            "--v2-curriculum",
        ):
            with self.subTest(legacy_flag=legacy_flag):
                with self.assertRaisesRegex(ValueError, "flags"):
                    self.parse_and_validate(
                        "--model", "unified_v2", legacy_flag
                    )

    def test_unified_mastery_weight_must_be_finite(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            self.parse_and_validate(
                "--model",
                "unified_v2",
                "--unified-mastery-loss-weight",
                "nan",
            )
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            train_model(
                train_bundle=None,
                model=self.model(),
                unified_mastery_bce_weight=float("nan"),
            )

    def test_summary_writes_manifest_fingerprint_and_positive_weight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "summary.json"
            fake_bundle = SimpleNamespace(
                num_students=2,
                num_exercises=3,
                num_concepts=3,
            )
            train_result = SimpleNamespace(
                checkpoint_selection_metric="auc",
                checkpoint_selection_start_epoch=1,
                checkpoint_selection_window=1,
                best_validation_score=float("nan"),
                final_loss=0.5,
                best_val_auc=float("nan"),
                best_epoch=1,
                best_checkpoint_path=None,
                history=[],
            )
            metrics = {
                "auc": 0.5,
                "acc": 0.5,
                "rmse": 0.5,
                "brier": 0.25,
                "ece": 0.0,
                "loss": 0.693,
            }
            argv = [
                "train.py",
                "--model",
                "unified_v2",
                "--interactions",
                "fixture.csv",
                "--q-matrix",
                "q.csv",
                "--epochs",
                "1",
                "--device",
                "cpu",
                "--output",
                str(output_path),
                "--log-dir",
                str(Path(temp_dir) / "logs"),
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    train_script,
                    "prepare_step_data_bundle",
                    return_value=fake_bundle,
                ),
                mock.patch.object(
                    train_script, "train_model", return_value=train_result
                ),
                mock.patch.object(
                    train_script, "evaluate_model", return_value=metrics
                ),
                mock.patch.object(train_script, "append_summary_csv"),
                mock.patch.object(train_script, "save_history_csv"),
            ):
                train_script.main()

            summary = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["model"], "unified_v2")
            self.assertEqual(summary["architecture_manifest"]["modules"], "m1-m4")
            self.assertRegex(
                summary["architecture_fingerprint"], r"^[0-9a-f]{64}$"
            )
            self.assertGreater(summary["unified_mastery_loss_weight"], 0.0)

    def test_direct_mastery_supervision_runs_in_both_training_modes(self) -> None:
        for mode in ("full_batch", "student_recompute_minibatch"):
            with self.subTest(mode=mode):
                torch.manual_seed(7)
                model = self.model()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                tensors = self.tensors()
                if mode == "full_batch":
                    stats = _train_full_batch_epoch(
                        model=model,
                        tensors=tensors,
                        optimizer=optimizer,
                        unified_mastery_bce_weight=0.5,
                    )
                else:
                    stats = _train_student_recompute_minibatch_epoch(
                        model=model,
                        tensors=tensors,
                        optimizer=optimizer,
                        student_batch_size=1,
                        unified_mastery_bce_weight=0.5,
                    )
                self.assertTrue(torch.isfinite(torch.tensor(stats.mean_loss)))
                self.assertGreater(stats.optimizer_steps, 0)
                self.assertIsNotNone(model.decoder.mastery_head.weight.grad)
                output = self.forward(model, tensors)
                self.assertIsNone(output.mastery_aux_logits)

    def test_saved_unified_checkpoint_reloads_with_identical_predictions(self) -> None:
        torch.manual_seed(11)
        model = self.model(inference="graph", composer="coverage")
        tensors = self.tensors()
        before = self.forward(model, tensors)
        manifest = model.architecture.manifest()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "unified.pt"
            torch.save(model.state_dict(), checkpoint)
            summary = {
                "model": "unified_v2",
                "architecture_manifest": manifest,
                "architecture_fingerprint": model.architecture.fingerprint(),
            }
            bundles = {
                "train": SimpleNamespace(
                    num_students=2,
                    num_exercises=3,
                    num_concepts=3,
                )
            }
            loaded = analyze_prediction_slices.load_model(
                summary=summary,
                checkpoint_path=str(checkpoint),
                bundles=bundles,
                concept_dim=4,
                device="cpu",
            )

        after = self.forward(loaded, tensors)
        self.assertEqual(loaded.architecture.manifest(), manifest)
        self.assertTrue(torch.equal(before.probs, after.probs))
        self.assertTrue(torch.equal(before.mastery, after.mastery))

    def test_doa_extracts_unified_mastery(self) -> None:
        tensors = self.tensors()
        bundle = SimpleNamespace(
            q_matrix_tensor=tensors["q_matrix"],
            concept_graph=tensors["concept_graph"],
            prerequisite_graph=None,
            similarity_graph=None,
            student_exercise_mask=tensors["student_exercise_mask"],
            response_matrix_tensor=tensors["response_matrix"],
            student_tkc_mask=tensors["student_tkc_mask"],
            student_ukc_mask=tensors["student_ukc_mask"],
            student_concept_evidence_tensor=tensors[
                "student_concept_evidence"
            ],
            exercise_evidence_tensor=tensors["exercise_evidence"],
            interaction_student_ids=tensors["interaction_student_ids"],
            interaction_exercise_ids=tensors["interaction_exercise_ids"],
            interaction_labels=tensors["interaction_labels"],
        )

        mastery = evaluate_doa.extract_mastery(
            model=self.model(), bundle=bundle, device="cpu"
        )

        self.assertIsNotNone(mastery)
        self.assertEqual(tuple(mastery.shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
