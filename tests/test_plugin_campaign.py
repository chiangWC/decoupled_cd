from __future__ import annotations

import hashlib
import json
import multiprocessing
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from scripts import plugin_campaign
from scripts.plugin_campaign import (
    CandidateStore,
    DuplicateTestEvaluationError,
    EvaluationArtifactSnapshot,
    FrozenConfigMismatchError,
    claim_test_evaluation,
    compute_frozen_config_id,
    prepare_evaluation_resources,
    run_candidate_training,
    select_evaluation_loader,
    write_evaluation_artifacts,
)


PLUGIN_CONFIG = {"aux_weight": 0.1}
BACKBONE_CONFIG = {
    "latent_dim": 32,
    "gcn_layers": 2,
    "keep_prob": 0.9,
    "if_type": "ncd",
    "mode": "all",
    "flip_ratio": 0.1,
    "ssl_temp": 0.5,
    "ssl_weight": 0.01,
    "prednet_len1": 512,
    "prednet_len2": 256,
    "dropout": 0.2,
}
Q_MATRIX_SHA256 = hashlib.sha256(b"q-matrix").hexdigest()
PROTOCOL = {
    "split": "valid",
    "seed": 42,
    "doa_seed": 42,
    "min_responses": 3,
    "max_pairs_per_concept": 100_000,
    "split_seed": 2024,
    "q_matrix_sha256": Q_MATRIX_SHA256,
}
HOLDOUT_ASSIGNMENTS_SHA256 = "9" * 64
ID_MAPS_BYTES = b'{"cpt_ids":["c1"],"exer_ids":["e1"],"stu_ids":["s1"]}\n'


def frozen_id(
    checkpoint_bytes: bytes,
    id_maps_bytes: bytes = ID_MAPS_BYTES,
    protocol: dict | None = None,
) -> str:
    return compute_frozen_config_id(
        checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
        id_maps_sha256=hashlib.sha256(id_maps_bytes).hexdigest(),
        plugin_config=PLUGIN_CONFIG,
        backbone_config=BACKBONE_CONFIG,
        protocol=protocol or PROTOCOL,
    )


def write_selection(
    path: Path,
    checkpoint_bytes: bytes = b"checkpoint",
    *,
    protocol: dict | None = None,
    holdout_sha: str | None = HOLDOUT_ASSIGNMENTS_SHA256,
) -> str:
    protocol = protocol or PROTOCOL
    id_maps_path = path.parent / "id_maps.json"
    if not id_maps_path.exists():
        id_maps_path.write_bytes(ID_MAPS_BYTES)
    id_maps_sha256 = hashlib.sha256(id_maps_path.read_bytes()).hexdigest()
    config_id = frozen_id(
        checkpoint_bytes,
        id_maps_path.read_bytes(),
        protocol,
    )
    selection = {
            "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
            "id_maps_sha256": id_maps_sha256,
            "plugin_config": PLUGIN_CONFIG,
            "backbone_config": BACKBONE_CONFIG,
            "protocol": protocol,
            "frozen_config_id": config_id,
            "dataset": "fixture",
    }
    if holdout_sha is not None:
        selection["holdout_assignments_sha256"] = holdout_sha
    path.write_text(json.dumps(selection), encoding="utf-8")
    return config_id


def test_claim_bytes(
    *,
    checkpoint_sha256: str = "d" * 64,
    id_maps_sha256: str = "e" * 64,
    protocol: dict | None = None,
    holdout_sha: str | None = HOLDOUT_ASSIGNMENTS_SHA256,
) -> bytes:
    protocol = protocol or PROTOCOL
    config_id = compute_frozen_config_id(
        checkpoint_sha256=checkpoint_sha256,
        id_maps_sha256=id_maps_sha256,
        plugin_config=PLUGIN_CONFIG,
        backbone_config=BACKBONE_CONFIG,
        protocol=protocol,
    )
    record = {
        "frozen_config_id": config_id,
        "checkpoint_sha256": checkpoint_sha256,
        "id_maps_sha256": id_maps_sha256,
        "plugin_config": PLUGIN_CONFIG,
        "backbone_config": BACKBONE_CONFIG,
        "protocol": protocol,
        "selection_sha256": "f" * 64,
        "dataset": "fixture",
        "selection_path": "/frozen/selection.json",
        "route_head": "a" * 40,
        "claimed_at_utc": "2026-07-10T12:00:00Z",
        "argv": ["evaluate", "--split", "test"],
    }
    if holdout_sha is not None:
        record["holdout_assignments_sha256"] = holdout_sha
    return (
        json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


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
            backbone_config=BACKBONE_CONFIG,
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

    def test_new_protocol_binds_dataset_split_and_stable_recipe_id(self) -> None:
        self.assertTrue(
            hasattr(plugin_campaign, "compute_recipe_id"),
            "plugin campaign must expose compute_recipe_id",
        )
        compute_recipe_id = plugin_campaign.compute_recipe_id
        protocol_data_kind = plugin_campaign.protocol_data_kind
        plugin = {
            "aux_weight": 0.1,
            "aux_detach_item_difficulty": False,
            "aux_warmup_fraction": 0.0,
        }
        backbone = {"latent_dim": 32, "gcn_layers": 3}
        self.assertEqual(
            compute_recipe_id(plugin, backbone),
            compute_recipe_id(
                dict(reversed(list(plugin.items()))),
                dict(reversed(list(backbone.items()))),
            ),
        )
        standard = {
            **PROTOCOL,
            "data_protocol": "standard",
            "dataset_name": "assist17",
        }
        holdout = {
            **PROTOCOL,
            "data_protocol": "holdout",
            "dataset_name": "assist17",
        }
        common = {
            "checkpoint_sha256": "a" * 64,
            "id_maps_sha256": "b" * 64,
            "plugin_config": plugin,
            "backbone_config": backbone,
        }
        self.assertNotEqual(
            compute_frozen_config_id(protocol=standard, **common),
            compute_frozen_config_id(protocol=holdout, **common),
        )
        self.assertEqual(protocol_data_kind(standard), "standard")
        self.assertEqual(protocol_data_kind(holdout), "holdout")
        self.assertEqual(protocol_data_kind(PROTOCOL), "holdout")
        self.assertEqual(
            frozen_id(b"checkpoint"),
            "48fe750cd028719f87026d0696666f10b2dbf897c15166da4081c33ee79c6776",
        )

        invalid_protocols = (
            ("provided together", {**PROTOCOL, "data_protocol": "standard"}),
            ("provided together", {**PROTOCOL, "dataset_name": "assist17"}),
            ("non-empty string", {
                **PROTOCOL,
                "data_protocol": "standard",
                "dataset_name": "   ",
            }),
            ("standard.*holdout", {
                **PROTOCOL,
                "data_protocol": "cross-validation",
                "dataset_name": "assist17",
            }),
        )
        for expected, protocol in invalid_protocols:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, expected):
                    compute_frozen_config_id(protocol=protocol, **common)

    def test_candidate_snapshot_is_per_epoch_append_only_and_manifested(self) -> None:
        recipe = {
            "aux_weight": 0.0,
            "aux_detach_item_difficulty": False,
            "aux_warmup_fraction": 0.0,
            "decouple": False,
        }
        store = CandidateStore(
            output_dir=self.root,
            model_name="orcdf-baseline",
            plugin_config={"aux_weight": 0.0, "decouple": False},
            recipe_config=recipe,
            backbone_config=BACKBONE_CONFIG,
            protocol={
                "split": "valid",
                "seed": 42,
                "doa_seed": 42,
                "min_responses": 3,
                "max_pairs_per_concept": 100_000,
                "split_seed": 2024,
                "q_matrix_sha256": Q_MATRIX_SHA256,
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
        self.assertEqual(record["backbone_config"], BACKBONE_CONFIG)
        self.assertEqual(record["recipe_config"], recipe)
        self.assertEqual(
            record["recipe_id"],
            plugin_campaign.compute_recipe_id(recipe, BACKBONE_CONFIG),
        )
        self.assertEqual(record["protocol"]["split"], "valid")
        for artifact in ("checkpoint", "mastery", "id_maps"):
            path = candidate_dir / (
                f"{artifact}.pth" if artifact == "checkpoint"
                else f"{artifact}.npy" if artifact == "mastery"
                else "id_maps.json"
            )
            self.assertEqual(
                record[f"{artifact}_sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
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
            backbone_config=BACKBONE_CONFIG,
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
            record["selection_sha256"],
            hashlib.sha256(selection.read_bytes()).hexdigest(),
        )
        self.assertEqual(record["dataset"], "fixture")
        self.assertEqual(
            record["holdout_assignments_sha256"],
            HOLDOUT_ASSIGNMENTS_SHA256,
        )
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
                backbone_config=BACKBONE_CONFIG,
                protocol=PROTOCOL,
                route_root=self.root,
                argv=[],
                route_head="b" * 40,
            )

    def test_standard_claim_omits_holdout_but_holdout_still_requires_it(self) -> None:
        checkpoint = self.root / "checkpoint.pth"
        checkpoint.write_bytes(b"checkpoint")
        standard_protocol = {
            **PROTOCOL,
            "data_protocol": "standard",
            "dataset_name": "fixture",
        }
        holdout_protocol = {
            **PROTOCOL,
            "data_protocol": "holdout",
            "dataset_name": "fixture",
        }
        standard_selection = self.root / "standard-selection.json"
        standard_id = write_selection(
            standard_selection,
            protocol=standard_protocol,
            holdout_sha=None,
        )

        standard_claim = claim_test_evaluation(
            ledger_dir=self.root / "ledger",
            selection_path=standard_selection,
            checkpoint_path=checkpoint,
            plugin_config=PLUGIN_CONFIG,
            backbone_config=BACKBONE_CONFIG,
            protocol=standard_protocol,
            route_root=self.root,
            route_head="1" * 40,
        )

        self.assertEqual(standard_claim.name, f"{standard_id}.json")
        self.assertNotIn(
            "holdout_assignments_sha256",
            json.loads(standard_claim.read_text(encoding="utf-8")),
        )
        bound_standard_selection = self.root / "bound-standard-selection.json"
        write_selection(
            bound_standard_selection,
            protocol=standard_protocol,
        )
        with self.assertRaisesRegex(
            FrozenConfigMismatchError,
            "standard selection",
        ):
            claim_test_evaluation(
                ledger_dir=self.root / "bound-standard-ledger",
                selection_path=bound_standard_selection,
                checkpoint_path=checkpoint,
                plugin_config=PLUGIN_CONFIG,
                backbone_config=BACKBONE_CONFIG,
                protocol=standard_protocol,
                route_root=self.root,
                route_head="5" * 40,
            )
        null_standard_selection = self.root / "null-standard-selection.json"
        write_selection(
            null_standard_selection,
            protocol=standard_protocol,
            holdout_sha=None,
        )
        null_bound_selection = json.loads(
            null_standard_selection.read_text(encoding="utf-8")
        )
        null_bound_selection["holdout_assignments_sha256"] = None
        null_standard_selection.write_text(
            json.dumps(null_bound_selection),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            FrozenConfigMismatchError,
            "standard selection",
        ):
            claim_test_evaluation(
                ledger_dir=self.root / "null-standard-ledger",
                selection_path=null_standard_selection,
                checkpoint_path=checkpoint,
                plugin_config=PLUGIN_CONFIG,
                backbone_config=BACKBONE_CONFIG,
                protocol=standard_protocol,
                route_root=self.root,
                route_head="6" * 40,
            )
        missing_holdout_selection = self.root / "missing-holdout-selection.json"
        write_selection(
            missing_holdout_selection,
            protocol=holdout_protocol,
            holdout_sha=None,
        )
        with self.assertRaises(FrozenConfigMismatchError):
            claim_test_evaluation(
                ledger_dir=self.root / "missing-holdout-ledger",
                selection_path=missing_holdout_selection,
                checkpoint_path=checkpoint,
                plugin_config=PLUGIN_CONFIG,
                backbone_config=BACKBONE_CONFIG,
                protocol=holdout_protocol,
                route_root=self.root,
                route_head="2" * 40,
            )

    def test_standard_and_holdout_use_distinct_frozen_claim_paths(self) -> None:
        checkpoint = self.root / "checkpoint.pth"
        checkpoint.write_bytes(b"checkpoint")
        standard_protocol = {
            **PROTOCOL,
            "data_protocol": "standard",
            "dataset_name": "fixture",
        }
        holdout_protocol = {
            **PROTOCOL,
            "data_protocol": "holdout",
            "dataset_name": "fixture",
        }
        standard_selection = self.root / "standard-selection.json"
        holdout_selection = self.root / "holdout-selection.json"
        standard_id = write_selection(
            standard_selection,
            protocol=standard_protocol,
            holdout_sha=None,
        )
        holdout_id = write_selection(
            holdout_selection,
            protocol=holdout_protocol,
        )

        standard_claim = claim_test_evaluation(
            ledger_dir=self.root / "ledger",
            selection_path=standard_selection,
            checkpoint_path=checkpoint,
            plugin_config=PLUGIN_CONFIG,
            backbone_config=BACKBONE_CONFIG,
            protocol=standard_protocol,
            route_root=self.root,
            route_head="3" * 40,
        )
        holdout_claim = claim_test_evaluation(
            ledger_dir=self.root / "ledger",
            selection_path=holdout_selection,
            checkpoint_path=checkpoint,
            plugin_config=PLUGIN_CONFIG,
            backbone_config=BACKBONE_CONFIG,
            protocol=holdout_protocol,
            route_root=self.root,
            route_head="4" * 40,
        )

        self.assertNotEqual(standard_id, holdout_id)
        self.assertNotEqual(standard_claim, holdout_claim)

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
            backbone_config=BACKBONE_CONFIG,
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
                backbone_config=BACKBONE_CONFIG,
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
            backbone_config=BACKBONE_CONFIG,
            protocol=PROTOCOL,
            route_root=self.root,
            argv=["evaluate"],
            route_head="c" * 40,
        )

        self.assertEqual(resources, "resources")

    def test_prepare_uses_claim_bytes_returned_by_exclusive_writer(self) -> None:
        checkpoint = self.root / "checkpoint.pth"
        checkpoint.write_bytes(b"checkpoint")
        selection = self.root / "selection.json"
        write_selection(selection)
        snapshot = plugin_campaign.TestClaimSnapshot(
            path=self.root / "ledger" / "claim.json",
            payload=b'{"immutable":true}\n',
        )

        with mock.patch.object(
            plugin_campaign,
            "claim_test_evaluation",
            return_value=snapshot,
        ):
            resources, claim_bytes = prepare_evaluation_resources(
                split="test",
                load_resources=lambda: "resources",
                checkpoint_path=checkpoint,
                ledger_dir=self.root / "ledger",
                selection_path=selection,
                plugin_config=PLUGIN_CONFIG,
                backbone_config=BACKBONE_CONFIG,
                protocol=PROTOCOL,
                route_root=self.root,
                route_head="c" * 40,
                return_test_claim_bytes=True,
            )

        self.assertEqual(resources, "resources")
        self.assertEqual(claim_bytes, snapshot.payload)

    def test_test_claim_rejects_tampered_checkpoint_sibling_id_maps(self) -> None:
        checkpoint = self.root / "checkpoint.pth"
        checkpoint.write_bytes(b"checkpoint")
        selection = self.root / "selection.json"
        write_selection(selection)
        (self.root / "id_maps.json").write_text('{"tampered": true}\n', encoding="utf-8")

        with self.assertRaisesRegex(FrozenConfigMismatchError, "id_maps"):
            claim_test_evaluation(
                ledger_dir=self.root / "ledger",
                selection_path=selection,
                checkpoint_path=checkpoint,
                plugin_config=PLUGIN_CONFIG,
                backbone_config=BACKBONE_CONFIG,
                protocol=PROTOCOL,
                route_root=self.root,
                route_head="e" * 40,
            )
        self.assertFalse((self.root / "ledger").exists())

    def test_test_claim_rejects_backbone_cli_drift(self) -> None:
        checkpoint = self.root / "checkpoint.pth"
        checkpoint.write_bytes(b"checkpoint")
        selection = self.root / "selection.json"
        write_selection(selection)

        with self.assertRaisesRegex(FrozenConfigMismatchError, "backbone_config"):
            claim_test_evaluation(
                ledger_dir=self.root / "ledger",
                selection_path=selection,
                checkpoint_path=checkpoint,
                plugin_config=PLUGIN_CONFIG,
                backbone_config={**BACKBONE_CONFIG, "latent_dim": 64},
                protocol=PROTOCOL,
                route_root=self.root,
                route_head="e" * 40,
            )

    def test_test_claim_rejects_snapshot_q_bytes_outside_protocol(self) -> None:
        checkpoint = self.root / "checkpoint.pth"
        checkpoint.write_bytes(b"checkpoint")
        selection = self.root / "selection.json"
        write_selection(selection)
        snapshot = EvaluationArtifactSnapshot(
            checkpoint_bytes=checkpoint.read_bytes(),
            id_maps_bytes=(self.root / "id_maps.json").read_bytes(),
            q_matrix_bytes=b"different-q-matrix",
        )

        with self.assertRaisesRegex(FrozenConfigMismatchError, "Q-matrix"):
            claim_test_evaluation(
                ledger_dir=self.root / "ledger",
                selection_path=selection,
                checkpoint_path=checkpoint,
                plugin_config=PLUGIN_CONFIG,
                backbone_config=BACKBONE_CONFIG,
                protocol=PROTOCOL,
                route_root=self.root,
                route_head="e" * 40,
                artifact_snapshot=snapshot,
            )
        self.assertFalse((self.root / "ledger").exists())

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

    def test_test_evaluation_cache_is_row_aligned_and_manifest_is_written_last(self) -> None:
        output = self.root / "test-evaluation"
        metadata = {
            "checkpoint_sha256": "d" * 64,
            "source_id_maps_sha256": "e" * 64,
            "plugin_config": PLUGIN_CONFIG,
            "backbone_config": BACKBONE_CONFIG,
            "protocol": PROTOCOL,
        }
        manifest_payload = {}
        original_writer = plugin_campaign._write_evaluation_cache_manifest

        def observe_manifest(path, payload):
            for filename in (
                "evaluation_cache.csv",
                "mastery.npy",
                "id_maps.json",
            ):
                artifact = output / filename
                self.assertTrue(artifact.is_file(), filename)
                expected_field = {
                    "evaluation_cache.csv": "cache_sha256",
                    "mastery.npy": "mastery_sha256",
                    "id_maps.json": "id_maps_sha256",
                }[filename]
                self.assertEqual(
                    payload[expected_field],
                    hashlib.sha256(artifact.read_bytes()).hexdigest(),
                )
            manifest_payload.update(payload)
            original_writer(path, payload)

        with mock.patch.object(
            plugin_campaign,
            "_write_evaluation_cache_manifest",
            side_effect=observe_manifest,
        ) as manifest_writer:
            write_evaluation_artifacts(
                output_dir=output,
                split="test",
                metrics={"auc": 0.8, "acc": 0.7, "rmse": 0.4},
                predictions=[0.2, 0.9],
                labels=[0.0, 1.0],
                test_claim_bytes=test_claim_bytes(),
                interaction_rows=[
                    {
                        "stu_id": "student-1",
                        "exer_id": "exercise-1",
                        "cpt_seq": "concept-1,concept-2",
                        "label": 0,
                    },
                    {
                        "stu_id": 2,
                        "exer_id": 20,
                        "cpt_seq": 200,
                        "label": 1,
                    },
                ],
                mastery=np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32),
                id_maps={
                    "stu_ids": ["student-1", "2"],
                    "exer_ids": ["exercise-1", "20"],
                    "cpt_ids": ["concept-1", "concept-2", "200"],
                },
                metadata=metadata,
            )

        manifest_writer.assert_called_once()
        self.assertEqual(
            (output / "evaluation_cache.csv").read_text().splitlines(),
            [
                "row_index,stu_id,exer_id,cpt_seq,label,prob",
                '0,student-1,exercise-1,"concept-1,concept-2",0.0,0.2',
                "1,2,20,200,1.0,0.9",
            ],
        )
        manifest = json.loads(
            (output / "evaluation_cache_manifest.json").read_text()
        )
        self.assertEqual(manifest, manifest_payload)
        self.assertEqual(manifest["format_version"], 1)
        self.assertEqual(manifest["evaluation_split"], "test")
        self.assertEqual(manifest["row_count"], 2)
        self.assertEqual(manifest["protocol"], PROTOCOL)
        self.assertEqual(manifest["checkpoint_sha256"], "d" * 64)
        self.assertEqual(manifest["source_id_maps_sha256"], "e" * 64)
        self.assertEqual(manifest["dataset"], "fixture")
        self.assertEqual(
            manifest["holdout_assignments_sha256"],
            HOLDOUT_ASSIGNMENTS_SHA256,
        )
        self.assertEqual(manifest["selection_sha256"], "f" * 64)
        self.assertEqual(
            manifest["test_claim_sha256"],
            hashlib.sha256(test_claim_bytes()).hexdigest(),
        )
        self.assertEqual(
            manifest["frozen_config_id"],
            json.loads(test_claim_bytes())["frozen_config_id"],
        )

    def test_test_evaluation_cache_requires_guarded_claim_binding(self) -> None:
        output = self.root / "unguarded-test-evaluation"
        with self.assertRaisesRegex(ValueError, "test claim"):
            write_evaluation_artifacts(
                output_dir=output,
                split="test",
                metrics={"auc": 0.8, "acc": 0.7, "rmse": 0.4},
                predictions=[0.2],
                labels=[0.0],
                interaction_rows=[
                    {"stu_id": 1, "exer_id": 10, "cpt_seq": 100, "label": 0}
                ],
                mastery=np.array([[0.1]], dtype=np.float32),
                id_maps={
                    "stu_ids": ["1"],
                    "exer_ids": ["10"],
                    "cpt_ids": ["100"],
                },
                metadata={
                    "checkpoint_sha256": "d" * 64,
                    "source_id_maps_sha256": "e" * 64,
                    "plugin_config": PLUGIN_CONFIG,
                    "backbone_config": BACKBONE_CONFIG,
                    "protocol": PROTOCOL,
                },
            )
        self.assertFalse(output.exists())

    def test_standard_test_cache_omits_holdout_assignment_binding(self) -> None:
        output = self.root / "standard-test-evaluation"
        standard_protocol = {
            **PROTOCOL,
            "data_protocol": "standard",
            "dataset_name": "fixture",
        }
        write_evaluation_artifacts(
            output_dir=output,
            split="test",
            metrics={"auc": 0.8, "acc": 0.7, "rmse": 0.4},
            predictions=[0.2],
            labels=[0.0],
            test_claim_bytes=test_claim_bytes(
                protocol=standard_protocol,
                holdout_sha=None,
            ),
            interaction_rows=[
                {"stu_id": 1, "exer_id": 10, "cpt_seq": 100, "label": 0}
            ],
            mastery=np.array([[0.1]], dtype=np.float32),
            id_maps={
                "stu_ids": ["1"],
                "exer_ids": ["10"],
                "cpt_ids": ["100"],
            },
            metadata={
                "checkpoint_sha256": "d" * 64,
                "source_id_maps_sha256": "e" * 64,
                "plugin_config": PLUGIN_CONFIG,
                "backbone_config": BACKBONE_CONFIG,
                "protocol": standard_protocol,
            },
        )

        manifest = json.loads(
            (output / "evaluation_cache_manifest.json").read_text()
        )
        self.assertNotIn("holdout_assignments_sha256", manifest)
        self.assertNotIn(
            "holdout_assignments_sha256",
            manifest["test_claim"],
        )

    def test_evaluation_cache_rejects_row_or_label_misalignment_before_writing(self) -> None:
        common = {
            "split": "test",
            "metrics": {"auc": 0.8, "acc": 0.7, "rmse": 0.4},
            "predictions": [0.2, 0.9],
            "labels": [0.0, 1.0],
            "mastery": np.array([[0.1]], dtype=np.float32),
            "id_maps": {
                "stu_ids": ["1"],
                "exer_ids": ["10"],
                "cpt_ids": ["100"],
            },
            "metadata": {
                "checkpoint_sha256": "d" * 64,
                "source_id_maps_sha256": "e" * 64,
                "protocol": PROTOCOL,
            },
            "test_claim_bytes": test_claim_bytes(),
        }
        invalid = {
            "row count": [
                {"stu_id": 1, "exer_id": 10, "cpt_seq": 100, "label": 0}
            ],
            "label": [
                {"stu_id": 1, "exer_id": 10, "cpt_seq": 100, "label": 1},
                {"stu_id": 1, "exer_id": 10, "cpt_seq": 100, "label": 1},
            ],
        }

        for expected, interaction_rows in invalid.items():
            with self.subTest(expected=expected):
                output = self.root / expected.replace(" ", "-")
                with self.assertRaisesRegex(ValueError, expected):
                    write_evaluation_artifacts(
                        output_dir=output,
                        interaction_rows=interaction_rows,
                        **common,
                    )
                self.assertFalse(output.exists())

    def test_candidate_protocol_rejects_unapproved_reproducibility_values(self) -> None:
        approved = {
            "split": "valid",
            "seed": 42,
            "doa_seed": 42,
            "min_responses": 3,
            "max_pairs_per_concept": 100_000,
            "split_seed": 2024,
            "q_matrix_sha256": Q_MATRIX_SHA256,
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
                        recipe_config={"aux_weight": 0.0},
                        backbone_config=BACKBONE_CONFIG,
                        protocol={**approved, field: value},
                    )


if __name__ == "__main__":
    unittest.main()
