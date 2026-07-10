from __future__ import annotations

import hashlib
import json
import multiprocessing
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.plugin_campaign import (
    CandidateStore,
    DuplicateTestEvaluationError,
    FrozenConfigMismatchError,
    claim_test_evaluation,
    compute_frozen_config_id,
    prepare_evaluation_resources,
    run_candidate_training,
    select_evaluation_loader,
    write_evaluation_artifacts,
)


PLUGIN_CONFIG = {"aux_weight": 0.1}
PROTOCOL = {
    "split": "valid",
    "seed": 42,
    "doa_seed": 42,
    "min_responses": 3,
    "max_pairs_per_concept": 100_000,
    "split_seed": 2024,
}


def frozen_id(checkpoint_bytes: bytes) -> str:
    return compute_frozen_config_id(
        checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
        plugin_config=PLUGIN_CONFIG,
        protocol=PROTOCOL,
    )


def write_selection(path: Path, checkpoint_bytes: bytes = b"checkpoint") -> str:
    config_id = frozen_id(checkpoint_bytes)
    path.write_text(
        json.dumps({
            "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
            "plugin_config": PLUGIN_CONFIG,
            "protocol": PROTOCOL,
            "frozen_config_id": config_id,
        }),
        encoding="utf-8",
    )
    return config_id


def _claim_worker(
    ledger_dir: str,
    checkpoint_path: str,
    selection_path: str,
    start_event,
    result_queue,
) -> None:
    start_event.wait()
    try:
        claim_test_evaluation(
            ledger_dir=Path(ledger_dir),
            selection_path=Path(selection_path),
            checkpoint_path=Path(checkpoint_path),
            plugin_config=PLUGIN_CONFIG,
            protocol=PROTOCOL,
            route_root=Path(ledger_dir),
            argv=["evaluate", "--split", "test"],
            route_head="a" * 40,
        )
    except DuplicateTestEvaluationError:
        result_queue.put("duplicate")
    else:
        result_queue.put("claimed")


class PluginCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_candidate_snapshot_is_per_epoch_append_only_and_manifested(self) -> None:
        store = CandidateStore(
            output_dir=self.root,
            model_name="orcdf-baseline",
            plugin_config={"aux_weight": 0.0, "decouple": False},
            protocol={
                "split": "valid",
                "seed": 42,
                "doa_seed": 42,
                "min_responses": 3,
                "max_pairs_per_concept": 100_000,
                "split_seed": 2024,
            },
        )

        record = store.save(
            epoch=1,
            validation_metrics={"auc": 0.81, "acc": 0.72, "rmse": 0.44},
            state_dict={"weight": torch.tensor([1.0])},
            mastery=np.array([[0.2, 0.8]], dtype=np.float32),
            id_maps={"stu_ids": ["s1"], "cpt_ids": ["c1", "c2"]},
        )

        candidate_dir = self.root / "candidates" / "epoch-001"
        self.assertTrue((candidate_dir / "checkpoint.pth").is_file())
        self.assertTrue((candidate_dir / "mastery.npy").is_file())
        self.assertTrue((candidate_dir / "id_maps.json").is_file())
        self.assertEqual(record["epoch"], 1)
        self.assertEqual(record["model_name"], "orcdf-baseline-epoch-001")
        self.assertEqual(
            record["checkpoint_path"],
            "candidates/epoch-001/checkpoint.pth",
        )
        self.assertEqual(record["validation"]["auc"], 0.81)
        self.assertEqual(record["plugin_config"]["aux_weight"], 0.0)
        self.assertEqual(record["protocol"]["split"], "valid")
        manifest = self.root / "candidates" / "manifest.jsonl"
        original_checkpoint = (candidate_dir / "checkpoint.pth").read_bytes()

        with self.assertRaises(FileExistsError):
            store.save(
                epoch=1,
                validation_metrics={"auc": 0.99, "acc": 0.99, "rmse": 0.01},
                state_dict={"weight": torch.tensor([9.0])},
                mastery=np.array([[0.9, 0.9]], dtype=np.float32),
                id_maps={"stu_ids": ["changed"], "cpt_ids": ["changed"]},
            )

        self.assertEqual((candidate_dir / "checkpoint.pth").read_bytes(), original_checkpoint)
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        self.assertEqual(rows, [record])

    def test_training_helper_snapshots_every_epoch_without_test_access(self) -> None:
        touched = {"test": False}
        trained: list[int] = []
        snapshotted: list[int] = []
        validation = iter(
            [
                {"auc": 0.70, "acc": 0.60, "rmse": 0.50},
                {"auc": 0.69, "acc": 0.61, "rmse": 0.49},
            ]
        )

        class ForbiddenTestLoader:
            def __iter__(self):
                touched["test"] = True
                raise AssertionError("train mode touched test")

        forbidden = ForbiddenTestLoader()

        result = run_candidate_training(
            epochs=5,
            patience=1,
            train_epoch=lambda epoch: trained.append(epoch),
            evaluate_validation=lambda: next(validation),
            snapshot_candidate=lambda epoch, _metrics: snapshotted.append(epoch),
        )

        self.assertIsNotNone(forbidden)
        self.assertEqual(trained, [1, 2])
        self.assertEqual(snapshotted, [1, 2])
        self.assertFalse(touched["test"])
        self.assertEqual(result["best_epoch"], 1)

    def test_valid_and_test_loader_dispatch(self) -> None:
        loaders = ("train-loader", "valid-loader", "test-loader")

        self.assertEqual(select_evaluation_loader(loaders, "valid"), "valid-loader")
        self.assertEqual(select_evaluation_loader(loaders, "test"), "test-loader")
        with self.assertRaises(ValueError):
            select_evaluation_loader(loaders, "train")

    def test_duplicate_test_claim_is_rejected_and_records_audit_fields(self) -> None:
        checkpoint = self.root / "checkpoint.pth"
        checkpoint.write_bytes(b"checkpoint")
        ledger = self.root / "ledger"
        selection = self.root / "selection.json"
        config_id = write_selection(selection)

        claim_path = claim_test_evaluation(
            ledger_dir=ledger,
            selection_path=selection,
            checkpoint_path=checkpoint,
            plugin_config=PLUGIN_CONFIG,
            protocol=PROTOCOL,
            route_root=self.root,
            argv=["runner.py", "--plugin-mode", "evaluate"],
            route_head="b" * 40,
            claimed_at_utc="2026-07-10T12:00:00Z",
        )

        record = json.loads(claim_path.read_text(encoding="utf-8"))
        self.assertEqual(record["frozen_config_id"], config_id)
        self.assertEqual(record["route_head"], "b" * 40)
        self.assertEqual(record["claimed_at_utc"], "2026-07-10T12:00:00Z")
        self.assertEqual(record["argv"], ["runner.py", "--plugin-mode", "evaluate"])
        self.assertEqual(
            record["checkpoint_sha256"],
            hashlib.sha256(b"checkpoint").hexdigest(),
        )
        with self.assertRaises(DuplicateTestEvaluationError):
            claim_test_evaluation(
                ledger_dir=ledger,
                selection_path=selection,
                checkpoint_path=checkpoint,
                plugin_config=PLUGIN_CONFIG,
                protocol=PROTOCOL,
                route_root=self.root,
                argv=[],
                route_head="b" * 40,
            )

    def test_changing_frozen_id_cannot_create_a_second_test_claim(self) -> None:
        checkpoint = self.root / "checkpoint.pth"
        checkpoint.write_bytes(b"checkpoint")
        ledger = self.root / "ledger"
        selection = self.root / "selection.json"
        config_id = write_selection(selection)
        claim_test_evaluation(
            ledger_dir=ledger,
            selection_path=selection,
            checkpoint_path=checkpoint,
            plugin_config=PLUGIN_CONFIG,
            protocol=PROTOCOL,
            route_root=self.root,
            route_head="d" * 40,
        )

        with self.assertRaises(FrozenConfigMismatchError):
            claim_test_evaluation(
                ledger_dir=ledger,
                selection_path=selection,
                supplied_frozen_config_id="arbitrary-new-id",
                checkpoint_path=checkpoint,
                plugin_config=PLUGIN_CONFIG,
                protocol=PROTOCOL,
                route_root=self.root,
                route_head="d" * 40,
            )

        self.assertEqual([path.name for path in ledger.iterdir()], [f"{config_id}.json"])

    def test_two_processes_racing_for_same_test_claim_have_one_winner(self) -> None:
        checkpoint = self.root / "checkpoint.pth"
        checkpoint.write_bytes(b"checkpoint")
        selection = self.root / "selection.json"
        write_selection(selection)
        context = multiprocessing.get_context("spawn")
        start_event = context.Event()
        result_queue = context.Queue()
        processes = [
            context.Process(
                target=_claim_worker,
                args=(
                    str(self.root / "ledger"),
                    str(checkpoint),
                    str(selection),
                    start_event,
                    result_queue,
                ),
            )
            for _ in range(2)
        ]
        for process in processes:
            process.start()
        start_event.set()
        results = sorted(result_queue.get(timeout=5) for _ in processes)
        for process in processes:
            process.join(timeout=5)
            self.assertEqual(process.exitcode, 0)

        self.assertEqual(results, ["claimed", "duplicate"])

    def test_test_claim_precedes_resource_loader_access(self) -> None:
        checkpoint = self.root / "checkpoint.pth"
        checkpoint.write_bytes(b"checkpoint")
        ledger = self.root / "ledger"
        selection = self.root / "selection.json"
        config_id = write_selection(selection)

        def load_resources():
            self.assertTrue((ledger / f"{config_id}.json").is_file())
            return "resources"

        resources = prepare_evaluation_resources(
            split="test",
            load_resources=load_resources,
            checkpoint_path=checkpoint,
            ledger_dir=ledger,
            selection_path=selection,
            plugin_config=PLUGIN_CONFIG,
            protocol=PROTOCOL,
            route_root=self.root,
            argv=["evaluate"],
            route_head="c" * 40,
        )

        self.assertEqual(resources, "resources")

    def test_evaluation_artifacts_include_metrics_predictions_mastery_and_ids(self) -> None:
        output = self.root / "evaluation"

        write_evaluation_artifacts(
            output_dir=output,
            split="valid",
            metrics={"auc": 0.8, "acc": 0.7, "rmse": 0.4},
            predictions=[0.2, 0.9],
            labels=[0.0, 1.0],
            mastery=np.array([[0.1, 0.9]], dtype=np.float32),
            id_maps={"stu_ids": ["s1"], "cpt_ids": ["c1", "c2"]},
            metadata={"checkpoint_sha256": "d" * 64},
        )

        metrics = json.loads((output / "metrics.json").read_text())
        self.assertEqual(metrics["split"], "valid")
        self.assertEqual(metrics["auc"], 0.8)
        self.assertEqual(metrics["checkpoint_sha256"], "d" * 64)
        self.assertEqual(
            (output / "predictions.csv").read_text().splitlines(),
            ["prob,label", "0.2,0.0", "0.9,1.0"],
        )
        np.testing.assert_array_equal(
            np.load(output / "mastery.npy"),
            np.array([[0.1, 0.9]], dtype=np.float32),
        )
        self.assertEqual(
            json.loads((output / "id_maps.json").read_text())["stu_ids"],
            ["s1"],
        )

    def test_candidate_protocol_rejects_unapproved_reproducibility_values(self) -> None:
        approved = {
            "split": "valid",
            "seed": 42,
            "doa_seed": 42,
            "min_responses": 3,
            "max_pairs_per_concept": 100_000,
            "split_seed": 2024,
        }

        for field, value in {
            "seed": 7,
            "doa_seed": 2024,
            "min_responses": 1,
            "split_seed": 7,
        }.items():
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    CandidateStore(
                        output_dir=self.root / field,
                        model_name="invalid",
                        plugin_config={"aux_weight": 0.0},
                        protocol={**approved, field: value},
                    )


if __name__ == "__main__":
    unittest.main()
