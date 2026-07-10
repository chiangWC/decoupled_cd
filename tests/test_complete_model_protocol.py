from __future__ import annotations

import hashlib
import importlib.util
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "scripts" / "complete_model_protocol.py"
PROTOCOL_SPEC = importlib.util.spec_from_file_location(
    "complete_model_protocol", PROTOCOL_PATH
)
assert PROTOCOL_SPEC is not None and PROTOCOL_SPEC.loader is not None
PROTOCOL_MODULE = importlib.util.module_from_spec(PROTOCOL_SPEC)
sys.modules[PROTOCOL_SPEC.name] = PROTOCOL_MODULE
PROTOCOL_SPEC.loader.exec_module(PROTOCOL_MODULE)


def _claim_worker(frozen_record: dict[str, object], start_event, result_queue) -> None:
    start_event.wait()
    try:
        PROTOCOL_MODULE.claim_test_evaluation(
            frozen_record=frozen_record,
            route_head=str(frozen_record["protocol_route_commit"]),
            argv=["evaluate", "--split", "test"],
        )
    except PROTOCOL_MODULE.DuplicateTestEvaluationError:
        result_queue.put("duplicate")
    else:
        result_queue.put("claimed")


class CompleteModelProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.repo = self.root / "repo"
        self.artifacts = self.root / "artifacts"
        self.data = self.root / "data"
        self.repo.mkdir()
        self.artifacts.mkdir()
        self.data.mkdir()
        subprocess.run(["git", "-C", str(self.repo), "init", "-q"], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "config", "user.name", "Protocol Test"],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(self.repo),
                "config",
                "user.email",
                "protocol@example.invalid",
            ],
            check=True,
        )
        (self.repo / "tracked.txt").write_text("route\n", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(self.repo), "add", "tracked.txt"], check=True
        )
        subprocess.run(
            ["git", "-C", str(self.repo), "commit", "-qm", "fixture"], check=True
        )
        self.route_head = subprocess.run(
            ["git", "-C", str(self.repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        self.train = self.data / "train.csv"
        self.valid = self.data / "valid.csv"
        self.test = self.data / "test.csv"
        self.q_matrix = self.data / "Q_matrix.csv"
        self.train.write_text("stu_id,exer_id,label\ns1,e1,1\n", encoding="utf-8")
        self.valid.write_text("stu_id,exer_id,label\ns1,e2,0\n", encoding="utf-8")
        self.test.write_text("stu_id,exer_id,label\ns1,e3,1\n", encoding="utf-8")
        self.q_matrix.write_text("exer_id,cpt_seq\ne1,c1\ne2,c1\ne3,c2\n", encoding="utf-8")
        self.checkpoint = self.artifacts / "best.pt"
        self.checkpoint.write_bytes(b"checkpoint-v1")
        self.summary = self.artifacts / "summary.json"
        self.summary_payload = {
            # Formal split-mode training does not populate this field; freeze gets
            # the human-readable dataset name independently.
            "dataset": None,
            "model": "v2",
            "seed": 42,
            "concept_dim": 64,
            "training_mode": "full_batch",
            "v2_dual_graph_support_adaptive": False,
            "train_interactions": str(self.train),
            "valid_interactions": str(self.valid),
            # Validation search deliberately uses valid as the test proxy.
            "test_interactions": str(self.valid),
            "q_matrix": str(self.q_matrix),
            "best_checkpoint_path": str(self.checkpoint),
            "best_val_auc": 0.81,
            "valid_metrics": {"auc": 0.81},
            "test_metrics": {"auc": 0.81},
            "history": [{"epoch": 1, "val_auc": 0.81}],
        }
        self._write_summary()
        self.training_status = self.artifacts / "training_status.json"
        self._write_training_status(
            self.training_status,
            summary_path=self.summary,
            checkpoint_path=self.checkpoint,
            data_paths=(self.train, self.valid, self.q_matrix),
        )
        self.frozen = self.artifacts / "frozen.json"
        self.ledger = PROTOCOL_MODULE.stable_ledger_dir(self.repo)
        self.fake_evaluator = self.root / "fake_evaluate_coverage_slice.py"
        self.counter = self.root / "evaluator_calls.jsonl"
        self.fake_evaluator.write_text(
            """
import json
import os
import sys
from pathlib import Path

def value(name):
    return sys.argv[sys.argv.index(name) + 1]

split = value('--split')
claim_path = os.environ.get('COMPLETE_MODEL_TEST_CLAIM')
if split == 'test':
    assert claim_path and Path(claim_path).is_file(), 'test opened before claim'
evaluation_summary = json.loads(Path(value('--summary')).read_text(encoding='utf-8'))
Path(evaluation_summary['best_checkpoint_path']).read_bytes()
for flag in ('--train-interactions', '--valid-interactions', '--test-interactions', '--q-matrix'):
    Path(value(flag)).read_bytes()
with Path(os.environ['FAKE_EVALUATOR_COUNTER']).open('a', encoding='utf-8') as handle:
    handle.write(json.dumps(sys.argv[1:]) + '\\n')
payload = {
    'dataset': value('--dataset-name'),
    'split': split,
    'slices': [
        {'scope': 'overall', 'model': 'v2', 'auc': 0.91, 'count': 10},
        {'scope': 'bucket:zero', 'model': 'v2', 'auc': 0.87, 'count': 4},
    ],
}
Path(value('--output')).write_text(json.dumps(payload), encoding='utf-8')
Path(value('--slice-csv')).write_text('scope,auc\\noverall,0.91\\nbucket:zero,0.87\\n', encoding='utf-8')
Path(value('--summary-csv')).write_text('overall_auc\\n0.91\\n', encoding='utf-8')
""".lstrip(),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_summary(self) -> None:
        self.summary.write_text(
            json.dumps(self.summary_payload, sort_keys=True), encoding="utf-8"
        )

    def _fingerprint(self, path: Path) -> dict[str, object]:
        return {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def _write_training_status(
        self,
        path: Path,
        *,
        summary_path: Path,
        checkpoint_path: Path,
        data_paths: tuple[Path, ...],
    ) -> None:
        payload = {
            "status": "completed",
            "exit_code": 0,
            "parameters": {"seed": 42, "split_seed": 2024},
            "code": {"route_commit": self.route_head, "vendor_commits": {}},
            "datasets": [self._fingerprint(item) for item in data_paths],
            "output_hashes": {
                summary_path.name: {"exists": True, **self._fingerprint(summary_path)},
                checkpoint_path.name: {
                    "exists": True,
                    **self._fingerprint(checkpoint_path),
                },
            },
        }
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    def freeze(self) -> dict[str, object]:
        return PROTOCOL_MODULE.freeze_selection(
            summary_path=self.summary,
            training_status_path=self.training_status,
            dataset_name="fixture",
            output_path=self.frozen,
            repo_root=self.repo,
            route_head=self.route_head,
            frozen_at_utc="2026-07-10T12:00:00Z",
        )

    def evaluate(
        self,
        *,
        split: str,
        output_dir: Path,
        test_interactions: Path | None = None,
    ) -> dict[str, object]:
        with mock.patch.dict(
            os.environ, {"FAKE_EVALUATOR_COUNTER": str(self.counter)}
        ):
            return PROTOCOL_MODULE.evaluate_frozen_selection(
                frozen_path=self.frozen,
                split=split,
                output_dir=output_dir,
                test_interactions=test_interactions,
                evaluator_path=self.fake_evaluator,
                python_executable=sys.executable,
                device="cpu",
                route_head=self.route_head,
            )

    def test_freeze_binds_route_summary_checkpoint_data_and_config_hashes(self) -> None:
        record = self.freeze()

        self.assertEqual(record["training_route_commit"], self.route_head)
        self.assertEqual(record["protocol_route_commit"], self.route_head)
        self.assertEqual(
            record["summary"]["sha256"],
            hashlib.sha256(self.summary.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            record["checkpoint"]["sha256"],
            hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            record["data"]["train_interactions"]["sha256"],
            hashlib.sha256(self.train.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            record["data"]["valid_interactions"]["sha256"],
            hashlib.sha256(self.valid.read_bytes()).hexdigest(),
        )
        self.assertRegex(record["config_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(record["frozen_config_id"], r"^[0-9a-f]{64}$")
        self.assertEqual(record["ledger_dir"], str(self.ledger.resolve()))
        self.assertEqual(
            record["training_status"]["sha256"],
            hashlib.sha256(self.training_status.read_bytes()).hexdigest(),
        )

    def test_freeze_rejects_summary_that_used_real_test_during_validation(self) -> None:
        self.summary_payload["test_interactions"] = str(self.test)
        self._write_summary()

        with self.assertRaisesRegex(
            PROTOCOL_MODULE.FrozenSelectionMismatchError, "valid proxy"
        ):
            self.freeze()

        self.assertFalse(self.frozen.exists())

    def test_freeze_rejects_valid_fields_that_both_point_to_real_test(self) -> None:
        self.summary_payload["valid_interactions"] = str(self.test)
        self.summary_payload["test_interactions"] = str(self.test)
        self._write_summary()

        with self.assertRaisesRegex(
            PROTOCOL_MODULE.FrozenSelectionMismatchError,
            "valid.csv|campaign",
        ):
            self.freeze()

        self.assertFalse(self.frozen.exists())

    def test_valid_evaluation_is_repeatable_and_never_uses_real_test(self) -> None:
        self.freeze()

        first = self.evaluate(split="valid", output_dir=self.artifacts / "valid-1")
        second = self.evaluate(split="valid", output_dir=self.artifacts / "valid-2")

        self.assertEqual(first["scopes"]["overall"]["auc"], 0.91)
        self.assertEqual(second["scopes"]["zero"]["auc"], 0.87)
        calls = [json.loads(line) for line in self.counter.read_text().splitlines()]
        self.assertEqual(len(calls), 2)
        for argv in calls:
            self.assertEqual(argv[argv.index("--split") + 1], "valid")
            proxy = argv[argv.index("--test-interactions") + 1]
            self.assertTrue(proxy.startswith("/proc/self/fd/"))
            self.assertNotIn(str(self.test.resolve()), argv)
        expected_valid_sha = hashlib.sha256(self.valid.read_bytes()).hexdigest()
        self.assertEqual(
            first["input_snapshot_hashes"]["test_interactions"],
            expected_valid_sha,
        )
        self.assertEqual(
            second["input_snapshot_hashes"]["test_interactions"],
            expected_valid_sha,
        )
        self.assertFalse(self.ledger.exists())

    def test_test_claim_precedes_first_test_read_and_duplicate_is_rejected(self) -> None:
        frozen = self.freeze()
        claim_path = self.ledger / f"{frozen['frozen_config_id']}.json"
        original_copyfile = PROTOCOL_MODULE.shutil.copyfile
        observed_test_read = False

        def guarded_copyfile(source: Path, target: Path, *args, **kwargs):
            nonlocal observed_test_read
            if Path(source).resolve() == self.test.resolve():
                self.assertTrue(claim_path.is_file(), "test copied before claim")
                observed_test_read = True
            return original_copyfile(source, target, *args, **kwargs)

        with mock.patch.object(
            PROTOCOL_MODULE.shutil, "copyfile", side_effect=guarded_copyfile
        ):
            result = self.evaluate(
                split="test",
                output_dir=self.artifacts / "test-1",
                test_interactions=self.test,
            )
            with self.assertRaises(PROTOCOL_MODULE.DuplicateTestEvaluationError):
                self.evaluate(
                    split="test",
                    output_dir=self.artifacts / "test-2",
                    test_interactions=self.test,
                )

        self.assertTrue(observed_test_read)
        self.assertEqual(len(self.counter.read_text().splitlines()), 1)
        self.assertEqual(result["split"], "test")
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
        self.assertEqual(claim["status"], "completed")
        self.assertEqual(
            claim["test_interactions_sha256"],
            hashlib.sha256(self.test.read_bytes()).hexdigest(),
        )

    def test_two_test_claims_racing_have_exactly_one_winner(self) -> None:
        frozen = self.freeze()
        context = multiprocessing.get_context("spawn")
        start_event = context.Event()
        result_queue = context.Queue()
        processes = [
            context.Process(
                target=_claim_worker,
                args=(frozen, start_event, result_queue),
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

    def test_tampered_summary_checkpoint_data_or_frozen_config_is_rejected(self) -> None:
        self.freeze()
        originals = {
            self.summary: self.summary.read_bytes(),
            self.checkpoint: self.checkpoint.read_bytes(),
            self.train: self.train.read_bytes(),
            self.training_status: self.training_status.read_bytes(),
            self.frozen: self.frozen.read_bytes(),
        }
        mutations = {
            "summary": (self.summary, b"{}"),
            "checkpoint": (self.checkpoint, b"tampered-checkpoint"),
            "data": (self.train, b"tampered-data"),
            "training_status": (self.training_status, b"{}"),
        }
        for name, (path, payload) in mutations.items():
            with self.subTest(name=name):
                path.write_bytes(payload)
                with self.assertRaises(PROTOCOL_MODULE.FrozenSelectionMismatchError):
                    self.evaluate(split="valid", output_dir=self.artifacts / name)
                path.write_bytes(originals[path])

        frozen_payload = json.loads(self.frozen.read_text(encoding="utf-8"))
        frozen_payload["config_sha256"] = "0" * 64
        self.frozen.write_text(json.dumps(frozen_payload), encoding="utf-8")
        with self.assertRaises(PROTOCOL_MODULE.FrozenSelectionMismatchError):
            self.evaluate(split="valid", output_dir=self.artifacts / "config")
        self.frozen.write_bytes(originals[self.frozen])
        self.assertFalse(self.counter.exists())

    def test_refreeze_and_relocation_cannot_reset_test_claim_identity(self) -> None:
        first = self.freeze()
        PROTOCOL_MODULE.claim_test_evaluation(
            frozen_record=first,
            route_head=self.route_head,
            argv=["evaluate", "--split", "test"],
        )

        relocated = self.root / "relocated"
        relocated_data = relocated / "data"
        relocated_artifacts = relocated / "artifacts"
        relocated_data.mkdir(parents=True)
        relocated_artifacts.mkdir()
        relocated_train = relocated_data / "train.csv"
        relocated_valid = relocated_data / "valid.csv"
        relocated_q = relocated_data / "Q_matrix.csv"
        relocated_checkpoint = relocated_artifacts / "best.pt"
        for source, target in (
            (self.train, relocated_train),
            (self.valid, relocated_valid),
            (self.q_matrix, relocated_q),
            (self.checkpoint, relocated_checkpoint),
        ):
            shutil.copyfile(source, target)
        relocated_summary = relocated_artifacts / "summary.json"
        relocated_payload = {
            **self.summary_payload,
            "train_interactions": str(relocated_train),
            "valid_interactions": str(relocated_valid),
            "test_interactions": str(relocated_valid),
            "q_matrix": str(relocated_q),
            "best_checkpoint_path": str(relocated_checkpoint),
        }
        relocated_summary.write_text(
            json.dumps(relocated_payload, sort_keys=True), encoding="utf-8"
        )
        relocated_status = relocated_artifacts / "training_status.json"
        self._write_training_status(
            relocated_status,
            summary_path=relocated_summary,
            checkpoint_path=relocated_checkpoint,
            data_paths=(relocated_train, relocated_valid, relocated_q),
        )
        second_frozen_path = relocated_artifacts / "frozen.json"
        second = PROTOCOL_MODULE.freeze_selection(
            summary_path=relocated_summary,
            training_status_path=relocated_status,
            dataset_name="fixture",
            output_path=second_frozen_path,
            repo_root=self.repo,
            route_head=self.route_head,
        )

        self.assertEqual(second["frozen_config_id"], first["frozen_config_id"])
        self.assertEqual(second["ledger_dir"], first["ledger_dir"])
        with self.assertRaises(PROTOCOL_MODULE.DuplicateTestEvaluationError):
            PROTOCOL_MODULE.claim_test_evaluation(
                frozen_record=second,
                route_head=self.route_head,
                argv=["evaluate", "--split", "test"],
            )

        (self.repo / "protocol_docs.txt").write_text("no model change\n", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(self.repo), "add", "protocol_docs.txt"], check=True
        )
        subprocess.run(
            ["git", "-C", str(self.repo), "commit", "-qm", "protocol docs"],
            check=True,
        )
        new_protocol_head = subprocess.run(
            ["git", "-C", str(self.repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        third = PROTOCOL_MODULE.freeze_selection(
            summary_path=relocated_summary,
            training_status_path=relocated_status,
            dataset_name="renamed-fixture",
            output_path=relocated_artifacts / "frozen-after-protocol-change.json",
            repo_root=self.repo,
            route_head=new_protocol_head,
        )
        self.assertEqual(third["frozen_config_id"], first["frozen_config_id"])
        with self.assertRaises(PROTOCOL_MODULE.DuplicateTestEvaluationError):
            PROTOCOL_MODULE.claim_test_evaluation(
                frozen_record=third,
                route_head=new_protocol_head,
                argv=["evaluate", "--split", "test"],
            )

    def test_dirty_route_is_rejected_before_evaluation(self) -> None:
        self.freeze()
        dirty_file = self.repo / "untracked.py"
        dirty_file.write_text("dirty\n", encoding="utf-8")
        try:
            with self.assertRaisesRegex(
                PROTOCOL_MODULE.FrozenSelectionMismatchError, "dirty"
            ):
                self.evaluate(
                    split="valid", output_dir=self.artifacts / "dirty-route"
                )
        finally:
            dirty_file.unlink()
        self.assertFalse(self.counter.exists())

    def test_evaluator_reads_verified_snapshots_after_sources_change(self) -> None:
        self.freeze()
        originals = {
            self.summary: self.summary.read_bytes(),
            self.checkpoint: self.checkpoint.read_bytes(),
            self.train: self.train.read_bytes(),
            self.valid: self.valid.read_bytes(),
            self.q_matrix: self.q_matrix.read_bytes(),
            self.test: self.test.read_bytes(),
        }
        original_run = PROTOCOL_MODULE.subprocess.run

        def mutate_before_evaluator(command, *args, **kwargs):
            if len(command) > 1 and Path(command[1]) == self.fake_evaluator.resolve():
                for path in originals:
                    path.write_bytes(b"changed-after-snapshot")
                for path in (self.artifacts / "snapshot-test" / "input_snapshot").iterdir():
                    path.chmod(0o600)
                    path.write_bytes(b"changed-snapshot-path")
            return original_run(command, *args, **kwargs)

        with mock.patch.object(
            PROTOCOL_MODULE.subprocess, "run", side_effect=mutate_before_evaluator
        ):
            result = self.evaluate(
                split="test",
                output_dir=self.artifacts / "snapshot-test",
                test_interactions=self.test,
            )

        argv = result["evaluator_argv"]
        self.assertTrue(
            argv[argv.index("--summary") + 1].startswith("/proc/self/fd/")
        )
        self.assertTrue(
            argv[argv.index("--test-interactions") + 1].startswith("/proc/self/fd/")
        )
        self.assertEqual(
            result["input_snapshot_hashes"]["test_interactions"],
            hashlib.sha256(originals[self.test]).hexdigest(),
        )
        self.assertRegex(
            result["input_snapshot_hashes"]["evaluation_summary"],
            r"^[0-9a-f]{64}$",
        )
        self.assertEqual(result["scopes"]["zero"]["auc"], 0.87)

    def test_committed_code_archive_is_pinned_to_git_object_bytes(self) -> None:
        original = (self.repo / "tracked.txt").read_bytes()

        descriptor, archive = PROTOCOL_MODULE.create_sealed_git_archive(
            self.repo, self.route_head
        )
        try:
            (self.repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
            with zipfile.ZipFile(f"/proc/self/fd/{descriptor}") as handle:
                self.assertEqual(handle.read("tracked.txt"), original)
            self.assertRegex(archive["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(archive["size_bytes"], 0)
        finally:
            os.close(descriptor)

    def test_memfd_falls_back_to_libc_when_python_wrapper_is_absent(self) -> None:
        with mock.patch.object(
            PROTOCOL_MODULE.os, "memfd_create", None, create=True
        ):
            descriptor, record = PROTOCOL_MODULE._sealed_memfd_from_bytes(
                "fallback", b"immutable"
            )
        try:
            self.assertEqual(Path(record["path"]).read_bytes(), b"immutable")
            self.assertEqual(
                record["sha256"], hashlib.sha256(b"immutable").hexdigest()
            )
        finally:
            os.close(descriptor)

    def test_immutable_summary_is_built_from_bound_in_memory_payload(self) -> None:
        snapshots = {
            "checkpoint": self._fingerprint(self.checkpoint),
            "train_interactions": self._fingerprint(self.train),
            "valid_interactions": self._fingerprint(self.valid),
            "q_matrix": self._fingerprint(self.q_matrix),
        }
        with mock.patch.object(
            PROTOCOL_MODULE,
            "_load_json_object",
            side_effect=AssertionError("must not reread a mutable summary path"),
        ):
            descriptors, immutable = PROTOCOL_MODULE._immutable_evaluation_inputs(
                summary_payload=self.summary_payload,
                snapshots=snapshots,
                split="valid",
            )
        try:
            sealed_summary = json.loads(
                Path(immutable["evaluation_summary"]["path"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(sealed_summary["model"], "v2")
            self.assertTrue(
                sealed_summary["best_checkpoint_path"].startswith("/proc/self/fd/")
            )
        finally:
            for descriptor in descriptors:
                os.close(descriptor)

    def test_committed_evaluator_uses_python_safe_path_mode(self) -> None:
        self.assertEqual(
            PROTOCOL_MODULE._committed_evaluator_prefix("python3"),
            ["python3", "-P", "-m", "scripts.evaluate_coverage_slice"],
        )

    def test_relative_summary_paths_are_rewritten_to_snapshot_paths(self) -> None:
        for field, path in (
            ("train_interactions", self.train),
            ("valid_interactions", self.valid),
            ("test_interactions", self.valid),
            ("q_matrix", self.q_matrix),
            ("best_checkpoint_path", self.checkpoint),
        ):
            self.summary_payload[field] = os.path.relpath(path, self.summary.parent)
        self._write_summary()
        self._write_training_status(
            self.training_status,
            summary_path=self.summary,
            checkpoint_path=self.checkpoint,
            data_paths=(self.train, self.valid, self.q_matrix),
        )
        self.freeze()

        result = self.evaluate(
            split="valid", output_dir=self.artifacts / "relative-valid"
        )

        summary_argument = result["evaluator_argv"][
            result["evaluator_argv"].index("--summary") + 1
        ]
        self.assertTrue(summary_argument.startswith("/proc/self/fd/"))
        summary_path = self.artifacts / "relative-valid" / "input_snapshot" / "summary.json"
        snapshot_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertTrue(Path(snapshot_summary["best_checkpoint_path"]).is_absolute())
        self.assertEqual(result["scopes"]["overall"]["auc"], 0.91)

    def test_test_result_records_overall_zero_and_all_output_hashes(self) -> None:
        self.freeze()

        result = self.evaluate(
            split="test",
            output_dir=self.artifacts / "test-result",
            test_interactions=self.test,
        )

        self.assertEqual(result["scopes"]["overall"]["auc"], 0.91)
        self.assertEqual(result["scopes"]["zero"]["auc"], 0.87)
        for name in (
            "coverage.json",
            "coverage_slices.csv",
            "coverage_summary.csv",
            "overall.json",
            "zero.json",
        ):
            path = self.artifacts / "test-result" / name
            self.assertEqual(
                result["artifact_hashes"][name],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        calls = self.counter.read_text().splitlines()
        self.assertEqual(len(calls), 1, "coverage evaluator must run exactly once")


if __name__ == "__main__":
    unittest.main()
