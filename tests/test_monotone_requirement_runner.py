from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import pandas as pd
import torch

from scripts import run_monotone_requirement_screen as runner


def _write_dataset(root: Path) -> None:
    source = root / "assist_17_chold_v2"
    source.mkdir(parents=True)
    students = [f"s{index:03d}" for index in range(80)]
    q_rows = []
    rows = []
    for item_index in range(40):
        item = f"i{item_index:02d}"
        concepts = (
            ("0", "1")
            if item_index % 4 == 0
            else ("1", "2")
            if item_index % 4 == 1
            else ("3",)
            if item_index % 4 == 2
            else ("0", "2", "3")
        )
        for concept in concepts:
            q_rows.append({"exer_id": item, "cpt_seq": concept})
        for student_index, student in enumerate(students):
            rows.append(
                {
                    "stu_id": student,
                    "exer_id": item,
                    "cpt_seq": "999",
                    "label": (student_index + item_index) % 2,
                }
            )
    pd.DataFrame(rows).to_csv(source / "train.csv", index=False)
    pd.DataFrame(q_rows).to_csv(source / "Q_matrix.csv", index=False)


class TestMonotoneRequirementRunner(unittest.TestCase):
    def test_predict_dataset_is_label_blind_and_writes_hashed_barrier(self) -> None:
        with tempfile.TemporaryDirectory() as raw_data, tempfile.TemporaryDirectory() as raw_output:
            data_root = Path(raw_data)
            output_root = Path(raw_output)
            _write_dataset(data_root)
            protocol = runner._build_protocol(data_root, "ASSIST17")
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=runner.PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            with mock.patch.object(
                runner,
                "load_audit_query_labels",
                side_effect=AssertionError("predict must not reveal labels"),
            ):
                barrier = runner.predict_dataset(
                    dataset="ASSIST17",
                    data_root=data_root,
                    output_root=output_root,
                    expected_commit=head,
                    device=torch.device("cpu"),
                    steps=2,
                    require_formal=False,
                )

            self.assertFalse(barrier["audit_labels_loaded"])
            self.assertTrue(barrier["all_predictions_present"])
            self.assertEqual(
                set(barrier["prediction_semantic_sha256"]),
                set(runner.VARIANT_NAMES),
            )
            directory = output_root / "ASSIST17"
            for variant in runner.VARIANT_NAMES:
                prediction = pd.read_csv(directory / f"predictions_{variant}.csv")
                self.assertNotIn("label", prediction)
                manifest = json.loads(
                    (directory / f"manifest_{variant}.json").read_text()
                )
                self.assertFalse(manifest["audit_labels_loaded"])
                self.assertLess(
                    manifest["optimizer_rows_used"],
                    manifest["optimizer_total_rows"],
                )
                self.assertEqual(
                    manifest["optimizer_rows_used"],
                    manifest["optimizer_eligible_rows"],
                )
                self.assertEqual(
                    runner._prediction_semantic_sha256(prediction),
                    manifest["prediction_semantic_sha256"],
                )
            with self.assertRaisesRegex(RuntimeError, "Non-formal"):
                runner._load_and_verify_barrier(
                    output_root=output_root,
                    dataset="ASSIST17",
                    expected_commit=head,
                )
            verified, predictions = runner._load_and_verify_barrier(
                output_root=output_root,
                dataset="ASSIST17",
                expected_commit=head,
                require_formal=False,
                expected_protocol_sha256=protocol.protocol_sha256,
                expected_audit=protocol.audit,
            )
            self.assertEqual(verified["barrier_sha256"], barrier["barrier_sha256"])
            self.assertEqual(set(predictions), set(runner.VARIANT_NAMES))
            self.assertIn(
                "maximum_absolute_mixed_difference",
                barrier["surface"],
            )

            corrupted = predictions["full"].copy()
            corrupted.loc[0, "eligible"] = not bool(corrupted.loc[0, "eligible"])
            with self.assertRaisesRegex(
                RuntimeError,
                "Outcome-free eligible mismatch",
            ):
                runner._assert_outcome_free_metadata(
                    frame=corrupted,
                    expected=protocol.audit.metadata_frame(),
                    dataset="ASSIST17",
                    variant="full",
                )

            prediction_path = directory / "predictions_full.csv"
            tampered = pd.read_csv(prediction_path)
            tampered.loc[0, "prob"] = min(
                0.99,
                float(tampered.loc[0, "prob"]) + 0.05,
            )
            runner._atomic_csv(prediction_path, tampered)
            semantic_sha = runner._prediction_semantic_sha256(tampered)
            manifest_path = directory / "manifest_full.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["prediction_file_sha256"] = runner.sha256_file(
                prediction_path
            )
            manifest["prediction_semantic_sha256"] = semantic_sha
            runner._atomic_json(manifest_path, manifest)
            barrier_path = directory / "prediction_barrier.json"
            tampered_barrier = json.loads(barrier_path.read_text())
            tampered_barrier["prediction_semantic_sha256"]["full"] = semantic_sha
            tampered_barrier["manifest_file_sha256"]["full"] = runner.sha256_file(
                manifest_path
            )
            tampered_barrier.pop("barrier_sha256")
            tampered_barrier["barrier_sha256"] = runner.hashlib.sha256(
                runner._canonical_json(tampered_barrier).encode("utf-8")
            ).hexdigest()
            runner._atomic_json(barrier_path, tampered_barrier)
            with self.assertRaisesRegex(RuntimeError, "Checkpoint replay mismatch"):
                runner._load_and_verify_barrier(
                    output_root=output_root,
                    dataset="ASSIST17",
                    expected_commit=head,
                    require_formal=False,
                    expected_protocol_sha256=protocol.protocol_sha256,
                    expected_audit=protocol.audit,
                )

    def test_frozen_source_registry_rejects_changed_data(self) -> None:
        with tempfile.TemporaryDirectory() as raw_data:
            data_root = Path(raw_data)
            _write_dataset(data_root)
            train_path, q_path = runner._dataset_paths(data_root, "ASSIST17")
            actual = {
                "train_full_sha256": runner.sha256_file(train_path),
                "q_matrix_sha256": runner.sha256_file(q_path),
            }
            with mock.patch.dict(
                runner.FROZEN_SOURCE_SHA256,
                {"ASSIST17": actual},
            ):
                self.assertEqual(
                    runner._assert_frozen_source(
                        dataset="ASSIST17",
                        train_path=train_path,
                        q_path=q_path,
                    ),
                    actual,
                )
                changed = pd.read_csv(train_path)
                changed.loc[0, "label"] = 1 - int(changed.loc[0, "label"])
                changed.to_csv(train_path, index=False)
                with self.assertRaisesRegex(RuntimeError, "Frozen source"):
                    runner._assert_frozen_source(
                        dataset="ASSIST17",
                        train_path=train_path,
                        q_path=q_path,
                    )

    def test_global_barrier_requires_external_hash(self) -> None:
        with tempfile.TemporaryDirectory() as raw_output:
            output_root = Path(raw_output)
            commit = "a" * 40
            payload = {
                "schema_version": 1,
                "datasets": {"ASSIST17": "left", "MOOCRadar": "right"},
                "expected_commit": commit,
                "architecture_fingerprint": runner._architecture_payload(
                    steps=runner.STEPS
                )["fingerprint"],
                "snapshot": {
                    "head": commit,
                    "branch": runner.EXPECTED_BRANCH,
                    "origin_head": commit,
                    "environment": "decoupled_cd",
                    "worktree_clean": True,
                    "formal": True,
                },
                "labels_loaded": False,
                "checkpoint_replay_and_protocol_metadata_verified": True,
            }
            sealed = runner.hashlib.sha256(
                runner._canonical_json(payload).encode("utf-8")
            ).hexdigest()
            payload["global_barrier_sha256"] = sealed
            runner._atomic_json(
                output_root / "global_prediction_barrier.json",
                payload,
            )
            loaded = runner._load_global_barrier(
                output_root=output_root,
                expected_commit=commit,
                expected_global_barrier_sha256=sealed,
            )
            self.assertEqual(loaded["global_barrier_sha256"], sealed)
            with self.assertRaisesRegex(RuntimeError, "Global prediction barrier"):
                runner._load_global_barrier(
                    output_root=output_root,
                    expected_commit=commit,
                    expected_global_barrier_sha256="0" * 64,
                )

    def test_global_barrier_precedes_any_label_load(self) -> None:
        with tempfile.TemporaryDirectory() as raw_output:
            output_root = Path(raw_output)
            (output_root / "ASSIST17").mkdir()
            with mock.patch.object(
                runner, "formal_snapshot", return_value={}
            ), mock.patch.object(
                runner,
                "_load_global_barrier",
                return_value={
                    "datasets": {"ASSIST17": "a", "MOOCRadar": "b"}
                },
            ), mock.patch.object(
                runner,
                "_dataset_paths",
                return_value=(Path("/train"), Path("/q")),
            ), mock.patch.object(
                runner, "_assert_frozen_source", return_value={}
            ), mock.patch.object(
                runner,
                "_build_protocol",
                return_value=mock.Mock(protocol_sha256="p", audit=mock.Mock()),
            ), mock.patch.object(
                runner,
                "_load_and_verify_barrier",
                side_effect=[
                    ({"barrier_sha256": "a"}, {}),
                    FileNotFoundError("missing MOO"),
                ],
            ), mock.patch.object(
                runner, "load_audit_query_labels"
            ) as labels:
                with self.assertRaisesRegex(FileNotFoundError, "missing MOO"):
                    runner.evaluate_all(
                        data_root=Path("/unused"),
                        output_root=output_root,
                        expected_commit="deadbeef",
                        expected_global_barrier_sha256="sealed",
                    )
                labels.assert_not_called()

    def test_surface_noncollapse_uses_only_feasible_fixed_cells(self) -> None:
        model = runner._model(variant="full", num_concepts=4)
        frame, diagnostic = runner._surface_diagnostic(
            model,
            device=torch.device("cpu"),
        )
        self.assertTrue(
            ((frame["b"] <= frame["g"]) & (frame["g"] <= (1 + frame["b"]) / 2)).all()
        )
        self.assertEqual(diagnostic["grid_step"], 0.05)
        self.assertEqual(diagnostic["noncollapse_threshold"], 0.001)
        self.assertGreater(diagnostic["feasible_adjacent_cells"], 0)
        self.assertEqual(
            diagnostic["interaction_scale"],
            "logit(learned_surface)",
        )
        self.assertIn("best_additive_projection", diagnostic)

        clipped = frame.copy()
        clipped.loc[0, "learned_surface"] = 1.0
        with self.assertRaisesRegex(RuntimeError, "logit clamp"):
            runner._surface_summary(clipped)

        capacity = runner._model(variant="capacity", num_concepts=4)
        generator = torch.Generator().manual_seed(2024)
        with torch.no_grad():
            for parameter in capacity.variant_parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator))
        _, capacity_diagnostic = runner._surface_diagnostic(
            capacity, device=torch.device("cpu")
        )
        self.assertLess(
            capacity_diagnostic["maximum_absolute_mixed_difference"],
            1e-4,
        )
        self.assertLess(
            capacity_diagnostic["best_additive_projection"]["rmse"], 2e-5
        )


if __name__ == "__main__":
    unittest.main()
