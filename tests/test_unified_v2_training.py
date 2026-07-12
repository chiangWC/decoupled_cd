from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from models import UnifiedArchitectureSpec, UnifiedDecoupledCDM
from models.unified_v2_components import smoothed_evidence_logits
from scripts import analyze_prediction_slices
from scripts import evaluate_doa
from scripts import evaluate_history_hiding_stress
from scripts import train as train_script
from trainers.engine import (
    _train_full_batch_epoch,
    _train_student_recompute_minibatch_epoch,
    observed_mastery_evidence_loss,
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

    @classmethod
    def bundle(cls):
        tensors = cls.tensors()
        return SimpleNamespace(
            allow_target_in_history=True,
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

    @staticmethod
    def model(*, completion: str = "prior") -> UnifiedDecoupledCDM:
        evidence = UnifiedV2TrainingTests.tensors()[
            "student_concept_evidence"
        ]
        return UnifiedDecoupledCDM(
            num_students=2,
            num_exercises=3,
            num_concepts=3,
            dim=4,
            architecture=UnifiedArchitectureSpec(completion=completion),
            initial_mastery_logits=smoothed_evidence_logits(evidence[..., :2]),
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

    def test_cli_accepts_a0_and_rejects_positive_completion_loss(self) -> None:
        args = self.parse_and_validate(
            "--model",
            "unified_v2",
            "--unified-completion",
            "prior",
            "--unified-completion-loss-weight",
            "0",
        )
        self.assertEqual(args.unified_completion, "prior")
        self.assertEqual(args.unified_completion_rank, 32)
        self.assertEqual(args.unified_evidence_loss_weight, 1.0)
        self.assertEqual(args.unified_completion_loss_weight, 0.0)

        with self.assertRaisesRegex(ValueError, "must be zero"):
            self.parse_and_validate(
                "--model",
                "unified_v2",
                "--unified-completion",
                "prior",
                "--unified-completion-loss-weight",
                "0.1",
            )
        lowrank = self.parse_and_validate(
            "--model",
            "unified_v2",
            "--unified-completion",
            "lowrank",
            "--unified-completion-loss-weight",
            "0.1",
        )
        self.assertEqual(lowrank.unified_completion, "lowrank")
        with self.assertRaisesRegex(ValueError, "must be positive"):
            self.parse_and_validate(
                "--model",
                "unified_v2",
                "--unified-completion",
                "lowrank",
                "--unified-completion-loss-weight",
                "0",
            )

    def test_lowrank_model_construction_fails_explicitly_until_task_8(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "Task 8"):
            self.model(completion="lowrank")

    def test_a0_forward_uses_train_evidence_and_one_mastery_path(self) -> None:
        tensors = self.tensors()
        evidence = tensors["student_concept_evidence"]
        model = self.model()
        self.assertEqual(
            set(dict(model.named_children())),
            {"mastery_estimator", "completer", "decoder", "behavior_model"},
        )
        output = model(
            q_matrix=tensors["q_matrix"],
            concept_graph=tensors["concept_graph"],
            student_exercise_mask=tensors["student_exercise_mask"],
            response_matrix=tensors["response_matrix"],
            student_tkc_mask=tensors["student_tkc_mask"],
            student_ukc_mask=tensors["student_ukc_mask"],
            student_concept_evidence=evidence,
            target_student_ids=torch.tensor([0, 1]),
            target_exercise_ids=torch.tensor([0, 1]),
        )

        self.assertEqual(tuple(output.mastery.shape), (2, 3))
        self.assertIs(output.student_state, output.mastery)
        self.assertTrue(
            torch.equal(output.mastery_observed_mask, evidence[..., 0] > 0)
        )
        self.assertTrue(torch.all(output.guess_probs + output.slip_probs < 1.0))
        expected_tkc = (
            output.mastery.unsqueeze(-1)
            * output.mastery_observed_mask.unsqueeze(-1)
        )
        expected_ukc = (
            output.mastery.unsqueeze(-1)
            * (~output.mastery_observed_mask).unsqueeze(-1)
        )
        self.assertTrue(torch.equal(output.tkc_states, expected_tkc))
        self.assertTrue(torch.equal(output.ukc_states, expected_ukc))

    def test_observed_loss_ignores_missing_cell_targets(self) -> None:
        tensors = self.tensors()
        output = self.forward(self.model(), tensors)
        evidence = tensors["student_concept_evidence"]
        changed = evidence.clone()
        missing = evidence[..., 0] == 0
        changed[..., 1][missing] = 999.0

        first = observed_mastery_evidence_loss(output, evidence[..., :2])
        second = observed_mastery_evidence_loss(output, changed[..., :2])

        torch.testing.assert_close(first, second)

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

    def test_unified_rejects_every_explicit_legacy_model_option(self) -> None:
        forbidden_options = {
            "--v2-ukc-propagation": None,
            "--v2-ukc-layers": "1",
            "--v2-ukc-evidence-cap": "20.0",
            "--v2-target-aware-readout": None,
            "--v2-monotonic-readout": None,
            "--v2-bounded-gs": None,
            "--v2-hybrid-readout": None,
            "--v2-lowrank-mastery": None,
            "--v2-lowrank-dim": "64",
            "--v2-mastery-aux-weight": "0.0",
            "--v2-response-graph": None,
            "--v2-rg-layers": "2",
            "--v2-dual-graph": None,
            "--v2-dual-graph-adaptive": None,
            "--v2-dual-graph-support-adaptive": None,
            "--v2-router": None,
            "--v2-attn-readout": None,
            "--v2-irt-head": None,
            "--v2-contrastive-weight": "0.0",
            "--v2-consistency-weight": "0.0",
            "--v2-consistency-adaptive": None,
            "--v2-curriculum": None,
            "--v2-rg-primary": None,
            "--v2-history-dropout-frac": "0.0",
            "--v2-masked-response-weight": "0.0",
            "--v2-masked-response-frac": "0.15",
            "--v2-rg-mastery": None,
            "--v2-target-fusion": None,
            "--v2-gs-max-guess": "0.3",
            "--v2-gs-max-slip": "0.3",
            "--v2-readout-dropout": "0.0",
            "--v2-ukc-consistency-weight": "0.0",
            "--v2-ukc-consistency-drop-frac": "0.2",
            "--dual-cdm-ensemble": None,
            "--dual-cdm-secondary-concept-dim": "80",
            "--dual-cdm-branch-bce-weight": "0.0",
            "--student-gate-prior-alpha": "1.0",
            "--student-gate-prior-beta": "1.0",
            "--student-fusion-mode": "adaptive",
            "--alpha": "1.0",
            "--beta": "1.0",
            "--gs-mode": "conditional",
            "--high-concept-logit-adapter": None,
            "--high-concept-logit-min-count": "3",
            "--pairwise-history-interaction-adapter": None,
            "--pairwise-history-interaction-min-count": "2",
            "--gs-difficulty-adapter": None,
            "--interpretable-readout-expert-adapter": None,
            "--interpretable-readout-expert-count": "3",
            "--student-conditioned-ukc-readout-residual": None,
            "--concept-evidence-readout-residual": None,
            "--concept-evidence-readout-min-count": "2",
            "--concept-evidence-readout-max-count": "0",
            "--concept-evidence-readout-min-seen-ratio": "1.0",
            "--concept-evidence-readout-max-logit": "0.5",
            "--concept-evidence-prior-residual": None,
            "--concept-evidence-prior-min-count": "2",
            "--concept-evidence-prior-max-count": "0",
            "--concept-evidence-prior-min-seen-ratio": "1.0",
            "--concept-evidence-prior-max-logit": "0.5",
            "--concept-evidence-prior-strength": "2.0",
            "--concept-evidence-prior-confidence-cap": "20.0",
            "--concept-evidence-prior-min-confidence": "0.0",
            "--concept-evidence-prior-min-abs-mastery": "0.0",
            "--concept-evidence-prior-positive-scale": "1.0",
            "--concept-evidence-prior-negative-scale": "1.0",
            "--concept-evidence-prior-train-start-epoch": "1",
            "--concept-evidence-prior-train-warmup-epochs": "0",
            "--concept-evidence-prior-apply-mode": "all",
            "--history-evidence-logit-prior-residual": None,
            "--history-evidence-logit-prior-location": "cognitive",
            "--history-evidence-logit-prior-min-count": "1",
            "--history-evidence-logit-prior-max-count": "0",
            "--history-evidence-logit-prior-min-seen-ratio": "0.0",
            "--history-evidence-logit-prior-max-logit": "3.0",
            "--history-evidence-logit-prior-component-cap": "3.0",
            "--history-evidence-logit-prior-weight-student": "1.0",
            "--history-evidence-logit-prior-weight-exercise": "1.0",
            "--history-evidence-logit-prior-weight-target-concept": "0.6",
            "--history-evidence-logit-prior-weight-concept": "0.35",
            "--history-evidence-logit-prior-weight-mastery": "0.0",
            "--history-evidence-logit-prior-prior-weight": "5.0",
            "--history-evidence-logit-prior-mastery-confidence-cap": "20.0",
            "--history-evidence-cognitive-alignment-weight": "0.0",
            "--history-evidence-cognitive-alignment-final-weight": "0.0",
            "--history-evidence-cognitive-alignment-anneal-start-epoch": "1",
            "--history-evidence-cognitive-alignment-anneal-end-epoch": "1",
            "--prerequisite-graph": "prerequisite.csv",
            "--similarity-graph": "similarity.csv",
            "--graph-mode": "single",
        }
        for option, value in forbidden_options.items():
            with self.subTest(option=option):
                arguments = ["--model", "unified_v2", option]
                if value is not None:
                    arguments[-1] = f"{option}={value}"
                with self.assertRaisesRegex(ValueError, "does not accept"):
                    self.parse_and_validate(*arguments)

    def test_unified_preserves_common_data_optimizer_and_training_options(self) -> None:
        args = self.parse_and_validate(
            "--model",
            "unified_v2",
            "--concept-graph",
            "concept.csv",
            "--concept-dim",
            "16",
            "--epochs",
            "4",
            "--learning-rate",
            "0.002",
            "--weight-decay",
            "0.01",
            "--training-mode",
            "student_recompute_minibatch",
            "--student-batch-size",
            "8",
            "--seed",
            "9",
        )
        self.assertEqual(args.concept_graph, "concept.csv")
        self.assertEqual(args.concept_dim, 16)
        self.assertEqual(args.training_mode, "student_recompute_minibatch")

    def test_unified_rejects_abbreviated_legacy_model_options(self) -> None:
        for option, value in (
            ("--student-fusion", "adaptive"),
            ("--student-gate-prior-a", "1.0"),
            ("--graph-m", "single"),
        ):
            with self.subTest(option=option):
                with mock.patch.object(
                    sys,
                    "argv",
                    ["train.py", "--model", "unified_v2", option, value],
                ), mock.patch.object(sys, "stderr", io.StringIO()):
                    with self.assertRaises(SystemExit):
                        train_script.parse_args()

    def test_unified_loss_weights_must_be_finite_and_valid(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            self.parse_and_validate(
                "--model",
                "unified_v2",
                "--unified-evidence-loss-weight",
                "nan",
            )
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            train_model(
                train_bundle=None,
                model=self.model(),
                unified_evidence_loss_weight=float("nan"),
                unified_completion_loss_weight=0.0,
            )

    def test_summary_writes_manifest_fingerprint_and_positive_weight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "summary.json"
            fake_bundle = SimpleNamespace(
                num_students=2,
                num_exercises=3,
                num_concepts=3,
                student_concept_evidence_tensor=self.tensors()[
                    "student_concept_evidence"
                ],
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
            self.assertEqual(
                summary["architecture_manifest"]["modules"],
                "m1-prior-m3-m4",
            )
            self.assertEqual(
                summary["architecture_manifest"]["cognitive_decoder"],
                "neuralcdm-monotonic",
            )
            self.assertEqual(summary["architecture_manifest"]["version"], 3)
            self.assertRegex(
                summary["architecture_fingerprint"], r"^[0-9a-f]{64}$"
            )
            self.assertEqual(summary["unified_completion"], "prior")
            self.assertEqual(summary["unified_completion_rank"], 32)
            self.assertGreater(summary["unified_evidence_loss_weight"], 0.0)
            self.assertEqual(summary["unified_completion_loss_weight"], 0.0)

    def test_observed_evidence_supervision_runs_in_both_training_modes(self) -> None:
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
                        unified_evidence_loss_weight=0.5,
                    )
                else:
                    stats = _train_student_recompute_minibatch_epoch(
                        model=model,
                        tensors=tensors,
                        optimizer=optimizer,
                        student_batch_size=1,
                        unified_evidence_loss_weight=0.5,
                    )
                self.assertTrue(torch.isfinite(torch.tensor(stats.mean_loss)))
                self.assertGreater(stats.optimizer_steps, 0)
                self.assertIsNotNone(model.mastery_estimator.logits.grad)
                output = self.forward(model, tensors)
                self.assertIsNone(output.mastery_aux_logits)

    def test_saved_unified_checkpoint_reloads_with_identical_predictions(self) -> None:
        torch.manual_seed(11)
        model = self.model()
        tensors = self.tensors()
        before = self.forward(model, tensors)
        manifest = model.architecture.manifest()
        model.set_checkpoint_loss_weights(
            evidence_loss_weight=1.0,
            completion_loss_weight=0.0,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "unified.pt"
            torch.save(model.state_dict(), checkpoint)
            summary = {
                "model": "unified_v2",
                "architecture_manifest": manifest,
                "architecture_fingerprint": model.architecture.fingerprint(),
                "unified_completion": "prior",
                "unified_completion_rank": 32,
                "unified_evidence_loss_weight": 1.0,
                "unified_completion_loss_weight": 0.0,
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

    def test_checkpoint_rejects_tampered_same_length_fingerprint(self) -> None:
        model = self.model()
        model.set_checkpoint_loss_weights(
            evidence_loss_weight=1.0,
            completion_loss_weight=0.0,
        )
        state = model.state_dict()
        fingerprint = state["_checkpoint_architecture_fingerprint"].clone()
        fingerprint[0] = (
            ord("0") if int(fingerprint[0]) != ord("0") else ord("1")
        )
        state["_checkpoint_architecture_fingerprint"] = fingerprint
        summary = {
            "model": "unified_v2",
            "architecture_manifest": model.architecture.manifest(),
            "architecture_fingerprint": model.architecture.fingerprint(),
            "unified_completion": "prior",
            "unified_completion_rank": 32,
            "unified_evidence_loss_weight": 1.0,
            "unified_completion_loss_weight": 0.0,
        }
        bundles = {
            "train": SimpleNamespace(
                num_students=2,
                num_exercises=3,
                num_concepts=3,
            )
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "tampered.pt"
            torch.save(state, checkpoint)
            with self.assertRaisesRegex(ValueError, "fingerprint.*mismatch"):
                analyze_prediction_slices.load_model(
                    summary=summary,
                    checkpoint_path=str(checkpoint),
                    bundles=bundles,
                    concept_dim=4,
                    device="cpu",
                )

    def test_checkpoint_rejects_summary_loss_weight_disagreement(self) -> None:
        model = self.model()
        model.set_checkpoint_loss_weights(
            evidence_loss_weight=0.75,
            completion_loss_weight=0.0,
        )
        summary = {
            "model": "unified_v2",
            "architecture_manifest": model.architecture.manifest(),
            "architecture_fingerprint": model.architecture.fingerprint(),
            "unified_completion": "prior",
            "unified_completion_rank": 32,
            "unified_evidence_loss_weight": 0.5,
            "unified_completion_loss_weight": 0.0,
        }
        bundles = {
            "train": SimpleNamespace(
                num_students=2,
                num_exercises=3,
                num_concepts=3,
            )
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "mismatch.pt"
            torch.save(model.state_dict(), checkpoint)
            with self.assertRaisesRegex(
                ValueError,
                "evidence_loss_weight.*mismatch",
            ):
                analyze_prediction_slices.load_model(
                    summary=summary,
                    checkpoint_path=str(checkpoint),
                    bundles=bundles,
                    concept_dim=4,
                    device="cpu",
                )

    def test_trained_checkpoint_saves_unified_configuration(self) -> None:
        model = self.model()
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "trained.pt"
            train_model(
                train_bundle=self.bundle(),
                valid_bundle=self.bundle(),
                model=model,
                epochs=1,
                checkpoint_path=str(checkpoint),
                unified_evidence_loss_weight=0.75,
                unified_completion_loss_weight=0.0,
            )
            state = torch.load(checkpoint, weights_only=True)

        def decode(name: str) -> str:
            return bytes(state[name].tolist()).decode("utf-8")

        self.assertEqual(
            json.loads(decode("_checkpoint_architecture_manifest")),
            model.architecture.manifest(),
        )
        self.assertEqual(
            decode("_checkpoint_architecture_fingerprint"),
            model.architecture.fingerprint(),
        )
        self.assertEqual(decode("_checkpoint_unified_completion"), "prior")
        self.assertEqual(
            int(state["_checkpoint_unified_completion_rank"]), 32
        )
        self.assertEqual(
            float(state["_checkpoint_unified_evidence_loss_weight"]),
            0.75,
        )
        self.assertEqual(
            float(state["_checkpoint_unified_completion_loss_weight"]),
            0.0,
        )

    def test_unified_loader_rejects_tampered_architecture_metadata(self) -> None:
        model = self.model()
        manifest = model.architecture.manifest()
        fingerprint = model.architecture.fingerprint()
        invalid_summaries = {
            "fingerprint": {
                "architecture_manifest": manifest,
                "architecture_fingerprint": "0" * 64,
            },
            "modules": {
                "architecture_manifest": {**manifest, "modules": "m1-m4"},
                "architecture_fingerprint": fingerprint,
            },
            "decoder": {
                "architecture_manifest": {
                    **manifest,
                    "cognitive_decoder": "free",
                },
                "architecture_fingerprint": fingerprint,
            },
            "mastery": {
                "architecture_manifest": {
                    **manifest,
                    "mastery_output": "student",
                },
                "architecture_fingerprint": fingerprint,
            },
            "version": {
                "architecture_manifest": {**manifest, "version": 1},
                "architecture_fingerprint": fingerprint,
            },
            "field_type": {
                "architecture_manifest": {**manifest, "completion": 1},
                "architecture_fingerprint": fingerprint,
            },
            "version_type": {
                "architecture_manifest": {**manifest, "version": True},
                "architecture_fingerprint": fingerprint,
            },
            "modules_type": {
                "architecture_manifest": {**manifest, "modules": 123},
                "architecture_fingerprint": fingerprint,
            },
            "extra_field": {
                "architecture_manifest": {**manifest, "extra": "field"},
                "architecture_fingerprint": fingerprint,
            },
            "fingerprint_type": {
                "architecture_manifest": manifest,
                "architecture_fingerprint": 123,
            },
        }
        bundles = {
            "train": SimpleNamespace(
                num_students=2,
                num_exercises=3,
                num_concepts=3,
            )
        }
        for name, metadata in invalid_summaries.items():
            with self.subTest(name=name):
                summary = {"model": "unified_v2", **metadata}
                with mock.patch.object(
                    analyze_prediction_slices, "UnifiedDecoupledCDM"
                ) as constructor:
                    with self.assertRaisesRegex(
                        ValueError,
                        "architecture_manifest|architecture_fingerprint",
                    ):
                        analyze_prediction_slices.load_model(
                            summary=summary,
                            checkpoint_path="unused.pt",
                            bundles=bundles,
                            concept_dim=4,
                            device="cpu",
                        )
                    constructor.assert_not_called()

    def test_coverage_evaluation_forwards_loaded_unified_model(self) -> None:
        torch.manual_seed(13)
        model = self.model()
        model.set_checkpoint_loss_weights(
            evidence_loss_weight=1.0,
            completion_loss_weight=0.0,
        )
        manifest = model.architecture.manifest()
        summary = {
            "model": "unified_v2",
            "architecture_manifest": manifest,
            "architecture_fingerprint": model.architecture.fingerprint(),
            "unified_completion": "prior",
            "unified_completion_rank": 32,
            "unified_evidence_loss_weight": 1.0,
            "unified_completion_loss_weight": 0.0,
        }
        bundles = {
            "train": SimpleNamespace(
                num_students=2,
                num_exercises=3,
                num_concepts=3,
            )
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "unified.pt"
            torch.save(model.state_dict(), checkpoint)
            loaded = analyze_prediction_slices.load_model(
                summary=summary,
                checkpoint_path=str(checkpoint),
                bundles=bundles,
                concept_dim=4,
                device="cpu",
            )
            labels, probs, loss = evaluate_history_hiding_stress.predict_bundle(
                bundle=self.bundle(),
                model=loaded,
                device="cpu",
            )

        self.assertEqual(tuple(labels.shape), (4,))
        self.assertEqual(tuple(probs.shape), (4,))
        self.assertTrue(torch.isfinite(torch.tensor(loss)))

    def test_doa_extracts_unified_mastery(self) -> None:
        mastery = evaluate_doa.extract_mastery(
            model=self.model(), bundle=self.bundle(), device="cpu"
        )

        self.assertIsNotNone(mastery)
        self.assertEqual(tuple(mastery.shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
