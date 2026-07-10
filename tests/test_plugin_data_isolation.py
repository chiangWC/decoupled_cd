from __future__ import annotations

import json
import hashlib
import io
import sys
import tempfile
import threading
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from scripts.plugin_campaign import (
    compute_frozen_config_id,
    prepare_evaluation_resources,
    snapshot_evaluation_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORCDF_ROOT = PROJECT_ROOT / "external" / "ORCDF"
SVGCD_ROOT = PROJECT_ROOT / "external" / "SVGCD"
sys.path.insert(0, str(ORCDF_ROOT))
sys.path.insert(0, str(SVGCD_ROOT))

from ORCDF import dataset as orcdf_dataset  # noqa: E402
from ORCDF import plugin as orcdf_plugin  # noqa: E402
from SVGCD import dataset as svgcd_dataset  # noqa: E402


PLUGIN_CONFIG = {"aux_weight": 0.1}
BACKBONE_CONFIG = {"fixture_model": "fixed"}
Q_MATRIX_TEXT = (
    "exer_id,cpt_seq\n10,100\n10,101\n20,200\n30,300\n30,301\n"
)
PROTOCOL = {
    "split": "valid",
    "seed": 42,
    "doa_seed": 42,
    "min_responses": 3,
    "max_pairs_per_concept": 100_000,
    "split_seed": 2024,
    "q_matrix_sha256": hashlib.sha256(Q_MATRIX_TEXT.encode()).hexdigest(),
}


def frozen_id(checkpoint_bytes: bytes, id_maps_bytes: bytes) -> str:
    return compute_frozen_config_id(
        checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
        id_maps_sha256=hashlib.sha256(id_maps_bytes).hexdigest(),
        plugin_config=PLUGIN_CONFIG,
        backbone_config=BACKBONE_CONFIG,
        protocol=PROTOCOL,
    )


class PluginDataIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_dir = self.root / "data"
        self.data_dir.mkdir()
        self.write_split("train.csv", [(1, 10, "100", 1)])
        self.write_split("valid.csv", [(2, 20, "200", 0)])
        self.write_split("test.csv", [(1, 30, "300,301", 1)])
        self.q_matrix = self.data_dir / "Q_matrix.csv"
        self.q_matrix.write_text(Q_MATRIX_TEXT, encoding="utf-8")
        self.schema = self.root / "candidate" / "id_maps.json"
        self.schema.parent.mkdir()
        self.schema.write_text(
            json.dumps({
                "stu_ids": ["1", "2"],
                "exer_ids": ["10", "20", "30"],
                "cpt_ids": ["100", "101", "200", "300", "301"],
            }),
            encoding="utf-8",
        )
        self.checkpoint = self.schema.parent / "checkpoint.pth"
        torch.save({"marker": "original"}, self.checkpoint)
        checkpoint_bytes = self.checkpoint.read_bytes()
        id_maps_sha256 = hashlib.sha256(self.schema.read_bytes()).hexdigest()
        self.config_id = frozen_id(checkpoint_bytes, self.schema.read_bytes())
        self.selection = self.root / "selection.json"
        self.selection.write_text(
            json.dumps({
                "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
                "id_maps_sha256": id_maps_sha256,
                "plugin_config": PLUGIN_CONFIG,
                "backbone_config": BACKBONE_CONFIG,
                "protocol": PROTOCOL,
                "frozen_config_id": self.config_id,
            }),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_split(self, name: str, rows: list[tuple[int, int, str, int]]) -> None:
        lines = ["stu_id,exer_id,cpt_seq,label"]
        lines.extend(f'{stu},{exer},"{concepts}",{label}' for stu, exer, concepts, label in rows)
        (self.data_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def orcdf_args(self):
        return SimpleNamespace(
            data_dir=str(self.data_dir),
            train_file="train.csv",
            valid_file="valid.csv",
            test_file="test.csv",
            mode="all",
            flip_ratio=0.0,
            batch_size=2,
        )

    def svgcd_args(self):
        return SimpleNamespace(
            data_dir=str(self.data_dir),
            train_file="train.csv",
            valid_file="valid.csv",
            test_file="test.csv",
            batch_size=2,
            eval_batch_size=2,
        )

    def processors(self):
        return (
            ("orcdf", orcdf_dataset, orcdf_dataset.CognitiveDataProcessor, self.orcdf_args()),
            ("svgcd", svgcd_dataset, svgcd_dataset.CognitiveDataProcessor, self.svgcd_args()),
        )

    def processor_context(self, name, module, processor_class, spy):
        stack = ExitStack()
        if name == "orcdf":
            stack.enter_context(mock.patch.object(module, "device", torch.device("cpu")))
        stack.enter_context(
            mock.patch.object(processor_class, "_read_csv", autospec=True, side_effect=spy)
        )
        return stack

    def test_train_and_valid_processors_never_read_test(self) -> None:
        for name, module, processor_class, args in self.processors():
            for split_mode in ("train", "valid"):
                with self.subTest(name=name, split_mode=split_mode):
                    calls: list[str] = []
                    original_read = processor_class._read_csv

                    def spy(instance, filename, _original=original_read):
                        calls.append(filename)
                        return _original(instance, filename)

                    with self.processor_context(name, module, processor_class, spy):
                        processor = processor_class(
                            args,
                            mock.Mock(),
                            split_mode=split_mode,
                            q_matrix_path=self.q_matrix,
                        )
                        processor.get_loaders()

                    self.assertEqual(calls, ["train.csv", "valid.csv"])
                    self.assertEqual(processor.stu_ids, ["1", "2"])
                    self.assertEqual(processor.exer_ids, ["10", "20", "30"])
                    self.assertEqual(
                        processor.cpt_ids,
                        ["100", "101", "200", "300", "301"],
                    )
                    self.assertEqual(float(processor.q_matrix[2, 3]), 1.0)
                    self.assertEqual(float(processor.q_matrix[2, 4]), 1.0)

    def test_orcdf_decouple_tensors_use_frozen_string_id_mapping(self) -> None:
        with mock.patch.object(orcdf_dataset, "device", torch.device("cpu")):
            processor = orcdf_dataset.CognitiveDataProcessor(
                self.orcdf_args(),
                mock.Mock(),
                split_mode="train",
                q_matrix_path=self.q_matrix,
            )

        mask = orcdf_plugin.build_tkc_mask(processor, torch.float64)
        graph = orcdf_plugin.build_concept_graph(processor, torch.float64)

        self.assertEqual(tuple(mask.shape), (2, 5))
        self.assertEqual(float(mask.sum()), 2.0)
        self.assertEqual(tuple(graph.shape), (5, 5))
        self.assertGreater(float(graph[0, 1]), 0.0)

    def test_legacy_processor_default_still_loads_all_splits(self) -> None:
        for name, module, processor_class, args in self.processors():
            with self.subTest(name=name):
                with ExitStack() as stack:
                    if name == "orcdf":
                        stack.enter_context(
                            mock.patch.object(module, "device", torch.device("cpu"))
                        )
                    processor = processor_class(args, mock.Mock())

                self.assertEqual(len(processor.test_triplets), 1)

    def test_test_claim_exists_before_processor_reads_test_and_schema_is_reused(self) -> None:
        for index, (name, module, processor_class, args) in enumerate(self.processors()):
            with self.subTest(name=name):
                ledger = self.root / f"ledger-{name}"
                claim_path = ledger / f"{self.config_id}.json"
                calls: list[str] = []
                original_read = processor_class._read_csv

                def spy(instance, filename, _original=original_read):
                    if filename == "test.csv":
                        self.assertTrue(claim_path.is_file())
                    calls.append(filename)
                    return _original(instance, filename)

                def load_processor():
                    with self.processor_context(name, module, processor_class, spy):
                        return processor_class(
                            args,
                            mock.Mock(),
                            split_mode="test",
                            id_maps_path=self.schema,
                            q_matrix_path=self.q_matrix,
                        )

                processor = prepare_evaluation_resources(
                    split="test",
                    load_resources=load_processor,
                    checkpoint_path=self.checkpoint,
                    ledger_dir=ledger,
                    selection_path=self.selection,
                    plugin_config=PLUGIN_CONFIG,
                    backbone_config=BACKBONE_CONFIG,
                    protocol=PROTOCOL,
                    route_root=self.root,
                    argv=["evaluate", "--split", "test"],
                    route_head=(f"{index + 1:x}" * 40)[:40],
                )

                self.assertEqual(calls, ["train.csv", "valid.csv", "test.csv"])
                self.assertEqual(processor.stu_ids, ["1", "2"])
                self.assertEqual(processor.exer_ids, ["10", "20", "30"])
                self.assertEqual(
                    processor.cpt_ids,
                    ["100", "101", "200", "300", "301"],
                )
                self.assertEqual(len(processor.test_triplets), 1)
                self.assertEqual(processor.test_triplets[0][1], 2)

    def test_unknown_test_ids_fail_after_claim_without_expanding_schema(self) -> None:
        self.write_split("test.csv", [(999, 30, "300", 1)])
        for index, (name, module, processor_class, args) in enumerate(self.processors()):
            with self.subTest(name=name):
                ledger = self.root / f"unknown-ledger-{name}"
                original_read = processor_class._read_csv

                def spy(instance, filename, _original=original_read):
                    return _original(instance, filename)

                def load_processor():
                    with self.processor_context(name, module, processor_class, spy):
                        return processor_class(
                            args,
                            mock.Mock(),
                            split_mode="test",
                            id_maps_path=self.schema,
                            q_matrix_path=self.q_matrix,
                        )

                with self.assertRaisesRegex(ValueError, "unknown student ID"):
                    prepare_evaluation_resources(
                        split="test",
                        load_resources=load_processor,
                        checkpoint_path=self.checkpoint,
                        ledger_dir=ledger,
                        selection_path=self.selection,
                        plugin_config=PLUGIN_CONFIG,
                        backbone_config=BACKBONE_CONFIG,
                        protocol=PROTOCOL,
                        route_root=self.root,
                        argv=["evaluate"],
                        route_head=(f"{index + 3:x}" * 40)[:40],
                    )
                self.assertTrue((ledger / f"{self.config_id}.json").is_file())

    def test_test_evaluation_consumes_snapshot_during_concurrent_mutation(self) -> None:
        snapshot = snapshot_evaluation_artifacts(self.checkpoint, self.q_matrix)
        ledger = self.root / "snapshot-ledger"
        mutate = threading.Event()
        mutated = threading.Event()

        def replace_artifacts():
            mutate.wait(timeout=5)
            torch.save({"marker": "mutated"}, self.checkpoint)
            self.schema.write_text('{"tampered": true}\n', encoding="utf-8")
            self.q_matrix.write_text("exer_id,cpt_seq\n999,999\n", encoding="utf-8")
            mutated.set()

        thread = threading.Thread(target=replace_artifacts)
        thread.start()

        def load_resources():
            mutate.set()
            self.assertTrue(mutated.wait(timeout=5))
            with mock.patch.object(orcdf_dataset, "device", torch.device("cpu")):
                processor = orcdf_dataset.CognitiveDataProcessor(
                    self.orcdf_args(),
                    mock.Mock(),
                    split_mode="test",
                    id_maps_payload=json.loads(snapshot.id_maps_bytes),
                    q_matrix_bytes=snapshot.q_matrix_bytes,
                )
            checkpoint = torch.load(
                io.BytesIO(snapshot.checkpoint_bytes),
                map_location="cpu",
            )
            return processor, checkpoint

        processor, checkpoint = prepare_evaluation_resources(
            split="test",
            load_resources=load_resources,
            checkpoint_path=self.checkpoint,
            artifact_snapshot=snapshot,
            ledger_dir=ledger,
            selection_path=self.selection,
            plugin_config=PLUGIN_CONFIG,
            backbone_config=BACKBONE_CONFIG,
            protocol=PROTOCOL,
            route_root=self.root,
            route_head="a" * 40,
        )
        thread.join(timeout=5)

        self.assertEqual(checkpoint["marker"], "original")
        self.assertEqual(processor.exer_ids, ["10", "20", "30"])
        self.assertEqual(processor.cpt_ids, ["100", "101", "200", "300", "301"])
        self.assertEqual(float(processor.q_matrix[2, 3]), 1.0)
        self.assertEqual(float(processor.q_matrix[2, 4]), 1.0)

    def test_test_exercise_or_concept_missing_from_q_fails(self) -> None:
        invalid_rows = (
            ("exercise", (1, 999, "300", 1)),
            ("concept", (1, 30, "999", 1)),
        )
        for kind, row in invalid_rows:
            self.write_split("test.csv", [row])
            for name, module, processor_class, args in self.processors():
                with self.subTest(kind=kind, name=name):
                    with self.processor_context(
                        name,
                        module,
                        processor_class,
                        processor_class._read_csv,
                    ):
                        with self.assertRaisesRegex(ValueError, f"unknown {kind} ID"):
                            processor_class(
                                args,
                                mock.Mock(),
                                split_mode="test",
                                id_maps_path=self.schema,
                                q_matrix_path=self.q_matrix,
                            )


if __name__ == "__main__":
    unittest.main()
