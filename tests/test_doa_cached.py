from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from scripts import doa_cached
from scripts.plugin_campaign import compute_frozen_config_id, write_evaluation_artifacts


PROTOCOL = {
    "split": "valid",
    "seed": 42,
    "doa_seed": 42,
    "min_responses": 3,
    "max_pairs_per_concept": 100_000,
    "split_seed": 2024,
    "q_matrix_sha256": hashlib.sha256(b"q-matrix").hexdigest(),
}
ASSIGNMENTS_BYTES = b"stu_id,holdout_concepts\n1,100\n2,100\n3,100\n"
ASSIGNMENTS_SHA256 = hashlib.sha256(ASSIGNMENTS_BYTES).hexdigest()


def test_claim_bytes() -> bytes:
    checkpoint_sha256 = "a" * 64
    id_maps_sha256 = "b" * 64
    plugin_config = {"aux_weight": 0.5}
    backbone_config = {"latent_dim": 32}
    record = {
        "frozen_config_id": compute_frozen_config_id(
            checkpoint_sha256=checkpoint_sha256,
            id_maps_sha256=id_maps_sha256,
            plugin_config=plugin_config,
            backbone_config=backbone_config,
            protocol=PROTOCOL,
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "id_maps_sha256": id_maps_sha256,
        "plugin_config": plugin_config,
        "backbone_config": backbone_config,
        "protocol": PROTOCOL,
        "selection_sha256": "c" * 64,
        "dataset": "fixture",
        "holdout_assignments_sha256": ASSIGNMENTS_SHA256,
        "selection_path": "/frozen/selection.json",
        "route_head": "d" * 40,
        "claimed_at_utc": "2026-07-10T12:00:00Z",
        "argv": ["evaluate", "--split", "test"],
    }
    return (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode()


class CachedDoaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.assignments = self.root / "holdout.csv"
        self.assignments.write_text(
            ASSIGNMENTS_BYTES.decode(), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_evaluation(self, name: str = "evaluation") -> Path:
        evaluation = self.root / name
        write_evaluation_artifacts(
            output_dir=evaluation,
            split="test",
            metrics={"auc": 0.75, "acc": 2 / 3, "rmse": 0.4},
            predictions=[0.2, 0.8, 0.7],
            labels=[0.0, 1.0, 1.0],
            test_claim_bytes=test_claim_bytes(),
            interaction_rows=[
                {"stu_id": 1, "exer_id": 10, "cpt_seq": 100, "label": 0},
                {"stu_id": 2, "exer_id": 10, "cpt_seq": 100, "label": 1},
                {"stu_id": 3, "exer_id": 10, "cpt_seq": 100, "label": 1},
            ],
            mastery=np.array([[0.1], [0.8], [0.7]], dtype=np.float32),
            id_maps={
                "stu_ids": ["1", "2", "3"],
                "exer_ids": ["10"],
                "cpt_ids": ["100"],
            },
            metadata={
                "checkpoint_sha256": "a" * 64,
                "source_id_maps_sha256": "b" * 64,
                "plugin_config": {"aux_weight": 0.5},
                "backbone_config": {"latent_dim": 32},
                "protocol": PROTOCOL,
            },
        )
        return evaluation

    def argv(self, evaluation: Path, output: Path | None = None) -> list[str]:
        return [
            "doa_cached.py",
            "--dataset-name",
            "fixture",
            "--evaluation-dir",
            str(evaluation),
            "--model-name",
            "fixture-model",
            "--holdout-assignments",
            str(self.assignments),
            "--output-csv",
            str(output or self.root / "doa.csv"),
        ]

    def test_reads_only_evaluation_cache_and_holdout_assignments(self) -> None:
        evaluation = self.make_evaluation()
        read_paths = []
        real_read_csv = pd.read_csv
        real_read_bytes = doa_cached._read_bytes

        def audited_read_bytes(path):
            read_paths.append(Path(path))
            return real_read_bytes(path)

        doa_result = {
            "doa": 0.5,
            "doa_weighted": 0.6,
            "num_concepts_evaluated": 1,
            "num_pairs": 3,
        }
        with (
            mock.patch.object(sys, "argv", self.argv(evaluation)),
            mock.patch.object(
                doa_cached,
                "_read_bytes",
                side_effect=audited_read_bytes,
            ),
            mock.patch.object(
                doa_cached,
                "compute_doa",
                return_value=doa_result,
            ) as compute_doa,
        ):
            doa_cached.main()

        self.assertCountEqual(
            read_paths,
            [
                self.assignments,
                evaluation / "evaluation_cache_manifest.json",
                evaluation / "evaluation_cache.csv",
                evaluation / "mastery.npy",
                evaluation / "id_maps.json",
            ],
        )
        self.assertFalse(any(path.name == "test.csv" for path in read_paths))
        self.assertEqual(compute_doa.call_count, 2)
        row = real_read_csv(self.root / "doa.csv").iloc[0]
        manifest = evaluation / "evaluation_cache_manifest.json"
        self.assertEqual(row["split"], "test")
        self.assertEqual(row["seed"], 42)
        self.assertEqual(row["doa_seed"], 42)
        self.assertEqual(row["min_responses"], 3)
        self.assertEqual(row["split_seed"], 2024)
        self.assertEqual(row["q_matrix_sha256"], PROTOCOL["q_matrix_sha256"])
        self.assertEqual(row["checkpoint_sha256"], "a" * 64)
        self.assertEqual(row["source_id_maps_sha256"], "b" * 64)
        self.assertEqual(
            row["frozen_config_id"],
            json.loads(test_claim_bytes())["frozen_config_id"],
        )
        self.assertEqual(row["selection_sha256"], "c" * 64)
        self.assertEqual(
            row["test_claim_sha256"],
            hashlib.sha256(test_claim_bytes()).hexdigest(),
        )
        for field, path in {
            "cache_manifest_sha256": manifest,
            "cache_sha256": evaluation / "evaluation_cache.csv",
            "mastery_sha256": evaluation / "mastery.npy",
            "id_maps_sha256": evaluation / "id_maps.json",
            "holdout_assignments_sha256": self.assignments,
        }.items():
            self.assertEqual(
                row[field],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )

    def test_rejects_each_tampered_cached_artifact_before_doa(self) -> None:
        for filename in ("evaluation_cache.csv", "mastery.npy", "id_maps.json"):
            with self.subTest(filename=filename):
                evaluation = self.make_evaluation(filename.replace(".", "-"))
                with (evaluation / filename).open("ab") as handle:
                    handle.write(b"tampered")
                with (
                    mock.patch.object(sys, "argv", self.argv(evaluation)),
                    mock.patch.object(doa_cached, "compute_doa") as compute_doa,
                ):
                    with self.assertRaisesRegex(ValueError, "SHA-256"):
                        doa_cached.main()
                compute_doa.assert_not_called()

    def test_rejects_locked_protocol_drift(self) -> None:
        for field, value in {
            "seed": 7,
            "doa_seed": 2024,
            "min_responses": 1,
            "split_seed": 7,
        }.items():
            with self.subTest(field=field):
                evaluation = self.make_evaluation(f"protocol-{field}")
                manifest_path = evaluation / "evaluation_cache_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["protocol"][field] = value
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with mock.patch.object(sys, "argv", self.argv(evaluation)):
                    with self.assertRaisesRegex(ValueError, field):
                        doa_cached.main()

    def test_rejects_rehashed_cache_with_noncontiguous_row_index(self) -> None:
        evaluation = self.make_evaluation()
        cache_path = evaluation / "evaluation_cache.csv"
        cache = pd.read_csv(cache_path)
        cache.loc[1, "row_index"] = 9
        cache.to_csv(cache_path, index=False)
        manifest_path = evaluation / "evaluation_cache_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["cache_sha256"] = hashlib.sha256(cache_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with mock.patch.object(sys, "argv", self.argv(evaluation)):
            with self.assertRaisesRegex(ValueError, "row_index"):
                doa_cached.main()

    def test_rejects_manifest_without_guarded_claim_binding(self) -> None:
        evaluation = self.make_evaluation()
        manifest_path = evaluation / "evaluation_cache_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for field in (
            "test_claim",
            "test_claim_sha256",
            "frozen_config_id",
            "selection_sha256",
        ):
            manifest.pop(field)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with mock.patch.object(sys, "argv", self.argv(evaluation)):
            with self.assertRaisesRegex(ValueError, "test claim"):
                doa_cached.main()

    def test_rejects_wrong_dataset_or_holdout_assignments(self) -> None:
        evaluation = self.make_evaluation()
        wrong_dataset_argv = self.argv(evaluation)
        wrong_dataset_argv[wrong_dataset_argv.index("fixture")] = "other-dataset"
        with mock.patch.object(sys, "argv", wrong_dataset_argv):
            with self.assertRaisesRegex(ValueError, "dataset"):
                doa_cached.main()

        self.assignments.write_text(
            "stu_id,holdout_concepts\n1,100\n2,100\n",
            encoding="utf-8",
        )
        with mock.patch.object(sys, "argv", self.argv(evaluation)):
            with self.assertRaisesRegex(ValueError, "holdout assignments SHA-256"):
                doa_cached.main()


if __name__ == "__main__":
    unittest.main()
