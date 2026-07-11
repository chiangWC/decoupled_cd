from __future__ import annotations

import json
import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.unified_v2_spec import UnifiedArchitectureSpec
from scripts import unified_validation_controller as controller_module
from scripts.unified_dataset_audit import canonical_sha256
from scripts.run_unified_validation import main as validation_main
from scripts.unified_validation_controller import (
    DATASET_DIRECTORIES,
    _advance_active_pair,
    authorize_next,
    consume_split_capability,
    initialize_controller,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_IDS = ("ASSIST09", "ASSIST17", "MOOCRadar", "XES3G5M")


class UnifiedValidationControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.state_dir = self.root / "controller"
        self.cohort_path = self.root / "cohort.json"
        cohort: dict[str, object] = {
            "schema_version": 1,
            "dataset_ids": list(DATASET_IDS),
        }
        cohort["cohort_sha256"] = canonical_sha256(cohort)
        self.cohort = cohort
        self.cohort_path.write_text(json.dumps(cohort), encoding="utf-8")

        self.manifest_path = self.root / "manifest.json"
        self.manifest = UnifiedArchitectureSpec(
            inference="graph",
            composer="mask",
        ).manifest()
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

        self.baseline_path = self.root / "baseline.json"
        self.baseline_rows = [
            {
                "dataset_id": dataset_id,
                "cohort_sha256": cohort["cohort_sha256"],
                "architecture_fingerprint": "b" * 64,
                "standard_overall_auc": 0.70,
                "holdout_overall_auc": 0.69,
                "zero_auc": 0.60,
                "ordinary_doa": 0.61,
                "weighted_doa": 0.62,
            }
            for dataset_id in DATASET_IDS
        ]
        self.baseline_path.write_text(
            json.dumps({"rows": self.baseline_rows}),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def initialize(self) -> dict[str, object]:
        return initialize_controller(
            state_dir=self.state_dir,
            repo_root=PROJECT_ROOT,
            cohort_path=self.cohort_path,
            manifest_path=self.manifest_path,
            architecture="m2",
            baseline_rows_path=self.baseline_path,
        )

    def issue(self) -> tuple[Path, dict[str, object]]:
        token_path = self.root / "capability.json"
        token = authorize_next(
            state_dir=self.state_dir,
            repo_root=PROJECT_ROOT,
            output_path=token_path,
        )
        return token_path, token

    def consume(
        self,
        token_path: Path,
        *,
        split_id: str,
    ) -> dict[str, object]:
        attempt_dir = self.root / split_id / "attempt-001"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        return consume_split_capability(
            state_dir=self.state_dir,
            repo_root=PROJECT_ROOT,
            token_path=token_path,
            dataset_id="ASSIST09",
            split_id=split_id,
            architecture="m2",
            data_root=self.root / "data",
            raw_output=Path("validation-summary.json"),
            attempt_dir=attempt_dir,
        )

    def _fingerprint(self, path: Path) -> dict[str, object]:
        return {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def _prepare_pair_artifacts(
        self,
        token_path: Path,
        token: dict[str, object],
        *,
        dataset_id: str,
        standard_overall_auc: float = 0.71,
        holdout_overall_auc: float = 0.70,
        zero_auc: float = 0.61,
        ordinary_doa: float = 0.62,
        weighted_doa: float = 0.63,
    ) -> dict[str, dict[str, object]]:
        data_root = self.root / "registered-data"
        standard_dir, holdout_dir = DATASET_DIRECTORIES[dataset_id]
        for directory_name, split_id in (
            (standard_dir, "standard"),
            (holdout_dir, "holdout"),
        ):
            directory = data_root / directory_name
            directory.mkdir(parents=True, exist_ok=True)
            for name in ("train.csv", "valid.csv", "Q_matrix.csv"):
                (directory / name).write_text(
                    f"{dataset_id},{split_id},{name}\n", encoding="utf-8"
                )
            if split_id == "holdout":
                (directory / "student_concept_holdout_assignments.csv").write_text(
                    f"{dataset_id},holdout\n", encoding="utf-8"
                )

        records: dict[str, dict[str, object]] = {}
        for split_id in ("standard", "holdout"):
            attempt_dir = self.root / dataset_id / split_id / "attempt-001"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            record = consume_split_capability(
                state_dir=self.state_dir,
                repo_root=PROJECT_ROOT,
                token_path=token_path,
                dataset_id=dataset_id,
                split_id=split_id,
                architecture="m2",
                data_root=data_root,
                raw_output=Path("validation-summary.json"),
                attempt_dir=attempt_dir,
            )
            summary_path = Path(str(record["summary_path"]))
            capability = token["capabilities"][split_id]
            summary = {
                "schema_version": 1,
                "dataset_id": dataset_id,
                "split_id": split_id,
                "architecture": "m2",
                "architecture_manifest": self.manifest,
                "architecture_fingerprint": record["architecture_fingerprint"],
                "cohort_sha256": self.cohort["cohort_sha256"],
                "controller_id": record["controller_id"],
                "controller_route_commit": record["route_commit"],
                "capability_counter": record["counter"],
                "capability_nonce": capability["nonce"],
                "seed": 42,
                "evaluation_input_role": "valid",
                "overall_auc": (
                    standard_overall_auc
                    if split_id == "standard"
                    else holdout_overall_auc
                ),
                "zero_auc": zero_auc,
                "ordinary_doa": ordinary_doa,
                "weighted_doa": weighted_doa,
                "mastery_shape": [2, 3],
                "final_loss": 0.4,
                "peak_gpu_memory_gb": 0.1,
            }
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            status = {
                "status": "completed",
                "exit_code": 0,
                "parameters": {"seed": 42},
                "code": {"route_commit": record["route_commit"]},
                "immutable_inputs": {
                    "architecture_manifest": {
                        **self._fingerprint(self.manifest_path),
                        "architecture_fingerprint": record[
                            "architecture_fingerprint"
                        ],
                    },
                    "cohort": {
                        **self._fingerprint(self.cohort_path),
                        "cohort_sha256": self.cohort["cohort_sha256"],
                    },
                },
                "datasets": [
                    self._fingerprint(Path(path)) for path in record["data_paths"]
                ],
                "output_hashes": {
                    "validation-summary.json": {
                        "exists": True,
                        **self._fingerprint(summary_path),
                    }
                },
            }
            status_path = attempt_dir / "status.json"
            status_path.write_text(json.dumps(status), encoding="utf-8")
            records[split_id] = {
                "consumption": record,
                "summary": summary,
                "summary_path": summary_path,
                "status": status,
                "status_path": status_path,
            }
        return records

    def test_initialization_copies_and_binds_registered_inputs(self) -> None:
        state = self.initialize()

        expected_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.assertEqual(state["route_commit"], expected_head)
        self.assertEqual(state["dataset_ids"], list(DATASET_IDS))
        self.assertEqual(state["cursor"], 0)
        self.assertEqual(state["successes"], 0)
        self.assertEqual(state["issuance_counter"], 0)
        self.assertIsNone(state["active_pair"])
        self.assertEqual(
            state["baseline_sha256"],
            canonical_sha256({"rows": self.baseline_rows}),
        )
        self.assertEqual(
            json.loads((self.state_dir / "cohort.json").read_text()),
            self.cohort,
        )
        self.assertEqual(
            json.loads((self.state_dir / "manifest.json").read_text()),
            self.manifest,
        )
        self.assertEqual(
            json.loads((self.state_dir / "baseline.json").read_text()),
            {"rows": self.baseline_rows},
        )
        for directory in ("issued", "consumed", "proofs"):
            self.assertTrue((self.state_dir / directory).is_dir())

    def test_initialization_is_exclusive(self) -> None:
        self.initialize()

        with self.assertRaises(FileExistsError):
            self.initialize()

    def test_initialization_rejects_baseline_order_or_cohort_mismatch(self) -> None:
        broken = list(reversed(self.baseline_rows))
        broken[0] = dict(broken[0], cohort_sha256="c" * 64)
        self.baseline_path.write_text(
            json.dumps({"rows": broken}),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "baseline.*cohort|dataset order"):
            self.initialize()
        self.assertFalse(self.state_dir.exists())

    def test_controller_init_cli_registers_the_iteration(self) -> None:
        result = validation_main(
            [
                "controller-init",
                "--controller-state-dir",
                str(self.state_dir),
                "--repo-root",
                str(PROJECT_ROOT),
                "--cohort",
                str(self.cohort_path),
                "--architecture",
                "m2",
                "--architecture-manifest",
                str(self.manifest_path),
                "--baseline-rows",
                str(self.baseline_path),
            ]
        )

        self.assertEqual(result, 0)
        self.assertTrue((self.state_dir / "state.json").is_file())

    def test_first_authorization_issues_only_first_dataset_pair(self) -> None:
        self.initialize()

        _, token = self.issue()

        self.assertEqual(token["dataset_id"], "ASSIST09")
        self.assertEqual(token["counter"], 1)
        self.assertEqual(set(token["capabilities"]), {"standard", "holdout"})
        self.assertNotIn("authorization_sha256", token)
        self.assertEqual(len(list((self.state_dir / "issued").glob("*.json"))), 2)
        state = json.loads((self.state_dir / "state.json").read_text())
        self.assertEqual(state["cursor"], 0)
        self.assertEqual(state["active_pair"]["dataset_id"], "ASSIST09")

    def test_existing_token_output_cannot_create_half_issued_pair(self) -> None:
        self.initialize()
        output_path = self.root / "capability.json"
        output_path.write_text("existing\n", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            authorize_next(
                state_dir=self.state_dir,
                repo_root=PROJECT_ROOT,
                output_path=output_path,
            )

        state = json.loads((self.state_dir / "state.json").read_text())
        self.assertIsNone(state["active_pair"])
        self.assertEqual(state["issuance_counter"], 0)
        self.assertFalse(any((self.state_dir / "issued").glob("*.json")))

    def test_final_token_write_failure_rolls_back_issuance_transaction(self) -> None:
        self.initialize()
        output_path = self.root / "capability.json"
        original_atomic_json = controller_module._atomic_json

        def fail_output(path: Path, payload: object) -> None:
            if path.resolve() == output_path.resolve():
                raise OSError("injected final token write failure")
            original_atomic_json(path, payload)

        with patch(
            "scripts.unified_validation_controller._atomic_json",
            side_effect=fail_output,
        ):
            with self.assertRaisesRegex(OSError, "injected final token"):
                authorize_next(
                    state_dir=self.state_dir,
                    repo_root=PROJECT_ROOT,
                    output_path=output_path,
                )

        state = json.loads((self.state_dir / "state.json").read_text())
        self.assertIsNone(state["active_pair"])
        self.assertEqual(state["issuance_counter"], 0)
        self.assertFalse(any((self.state_dir / "issued").glob("*.json")))
        self.assertFalse(output_path.exists())

    def test_forged_recomputed_hash_token_has_no_registry_authority(self) -> None:
        self.initialize()
        _, token = self.issue()
        forged = json.loads(json.dumps(token))
        forged["capabilities"]["standard"]["nonce"] = "f" * 64
        forged["authorization_sha256"] = canonical_sha256(forged)
        forged_path = self.root / "forged.json"
        forged_path.write_text(json.dumps(forged), encoding="utf-8")

        output_parent = self.root / "must-not-exist"
        with self.assertRaisesRegex(ValueError, "registry|capability"):
            consume_split_capability(
                state_dir=self.state_dir,
                repo_root=PROJECT_ROOT,
                token_path=forged_path,
                dataset_id="ASSIST09",
                split_id="standard",
                architecture="m2",
                data_root=self.root / "data",
                raw_output=output_parent / "summary.json",
                attempt_dir=self.root / "attempt-forged",
            )
        self.assertFalse(output_parent.exists())

    def test_each_split_capability_is_consumed_exactly_once(self) -> None:
        self.initialize()
        token_path, token = self.issue()

        standard = self.consume(token_path, split_id="standard")
        self.assertEqual(standard["nonce"], token["capabilities"]["standard"]["nonce"])
        with self.assertRaisesRegex(ValueError, "consumed|registry|issued"):
            self.consume(token_path, split_id="standard")
        holdout = self.consume(token_path, split_id="holdout")
        self.assertEqual(holdout["nonce"], token["capabilities"]["holdout"]["nonce"])
        self.assertFalse(any((self.state_dir / "issued").glob("*.json")))
        self.assertEqual(len(list((self.state_dir / "consumed").glob("*.json"))), 2)

    def test_route_commit_mismatch_rejects_issuance(self) -> None:
        route_repo = self.root / "route"
        route_repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=route_repo, check=True)
        subprocess.run(
            ["git", "config", "user.email", "fixture@example.com"],
            cwd=route_repo,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Fixture"], cwd=route_repo, check=True
        )
        marker = route_repo / "marker.txt"
        marker.write_text("one\n", encoding="utf-8")
        subprocess.run(["git", "add", "marker.txt"], cwd=route_repo, check=True)
        subprocess.run(["git", "commit", "-qm", "one"], cwd=route_repo, check=True)
        initialize_controller(
            state_dir=self.state_dir,
            repo_root=route_repo,
            cohort_path=self.cohort_path,
            manifest_path=self.manifest_path,
            architecture="m2",
            baseline_rows_path=self.baseline_path,
        )
        marker.write_text("two\n", encoding="utf-8")
        subprocess.run(["git", "add", "marker.txt"], cwd=route_repo, check=True)
        subprocess.run(["git", "commit", "-qm", "two"], cwd=route_repo, check=True)

        with self.assertRaisesRegex(ValueError, "route HEAD mismatch"):
            authorize_next(
                state_dir=self.state_dir,
                repo_root=route_repo,
                output_path=self.root / "token.json",
            )
        self.assertFalse((self.root / "token.json").exists())

    def test_run_split_consumes_capability_before_output_or_data_work(self) -> None:
        self.initialize()
        token_path, token = self.issue()
        attempt_dir = self.root / "outer" / "attempt-001"
        attempt_dir.mkdir(parents=True)
        output_path = attempt_dir / "validation-summary.json"

        with patch.dict(
            "os.environ",
            {"CAMPAIGN_ATTEMPT_DIR": str(attempt_dir)},
            clear=False,
        ):
            with self.assertRaises(FileNotFoundError):
                validation_main(
                    [
                        "run-split",
                        "--dataset-id",
                        "ASSIST09",
                        "--split-id",
                        "standard",
                        "--architecture",
                        "m2",
                        "--data-root",
                        str(self.root / "missing-data"),
                        "--controller-state-dir",
                        str(self.state_dir),
                        "--repo-root",
                        str(PROJECT_ROOT),
                        "--capability",
                        str(token_path),
                        "--device",
                        "cpu",
                        "--output",
                        "validation-summary.json",
                    ]
                )

        nonce = token["capabilities"]["standard"]["nonce"]
        consumed = list((self.state_dir / "consumed").glob(f"*{nonce}.json"))
        self.assertEqual(len(consumed), 1)
        self.assertFalse(output_path.exists())

    def test_verified_pair_advances_only_to_next_dataset_and_stales_old_token(self) -> None:
        self.initialize()
        first_path, first = self.issue()
        self._prepare_pair_artifacts(first_path, first, dataset_id="ASSIST09")

        second_path = self.root / "capability-2.json"
        second = authorize_next(
            state_dir=self.state_dir,
            repo_root=PROJECT_ROOT,
            output_path=second_path,
        )

        self.assertEqual(second["dataset_id"], "ASSIST17")
        self.assertEqual(second["counter"], 2)
        state = json.loads((self.state_dir / "state.json").read_text())
        self.assertEqual(state["cursor"], 1)
        self.assertEqual(state["successes"], 1)
        self.assertEqual(len(list((self.state_dir / "proofs").glob("*.json"))), 1)
        with self.assertRaisesRegex(ValueError, "stale|registry|issued"):
            consume_split_capability(
                state_dir=self.state_dir,
                repo_root=PROJECT_ROOT,
                token_path=first_path,
                dataset_id="ASSIST09",
                split_id="standard",
                architecture="m2",
                data_root=self.root / "registered-data",
                raw_output=Path("validation-summary.json"),
                attempt_dir=self.root / "stale-attempt",
            )

    def test_summary_output_hash_mismatch_cannot_advance_progress(self) -> None:
        self.initialize()
        token_path, token = self.issue()
        records = self._prepare_pair_artifacts(
            token_path, token, dataset_id="ASSIST09"
        )
        Path(records["standard"]["summary_path"]).write_text(
            json.dumps({"forged": True}), encoding="utf-8"
        )

        next_path = self.root / "must-not-issue.json"
        with self.assertRaisesRegex(ValueError, "output hash|summary.*hash"):
            authorize_next(
                state_dir=self.state_dir,
                repo_root=PROJECT_ROOT,
                output_path=next_path,
            )
        self.assertFalse(next_path.exists())
        self.assertFalse(any((self.state_dir / "proofs").glob("*.json")))

    def test_status_attempt_path_mismatch_cannot_advance_progress(self) -> None:
        self.initialize()
        token_path, token = self.issue()
        records = self._prepare_pair_artifacts(
            token_path, token, dataset_id="ASSIST09"
        )
        status = records["standard"]["status"]
        status["output_hashes"]["validation-summary.json"]["path"] = str(
            self.root / "other-attempt" / "validation-summary.json"
        )
        Path(records["standard"]["status_path"]).write_text(
            json.dumps(status), encoding="utf-8"
        )

        with self.assertRaisesRegex(ValueError, "registered summary path"):
            authorize_next(
                state_dir=self.state_dir,
                repo_root=PROJECT_ROOT,
                output_path=self.root / "must-not-issue.json",
            )

    def test_outer_status_route_commit_mismatch_cannot_advance(self) -> None:
        self.initialize()
        token_path, token = self.issue()
        records = self._prepare_pair_artifacts(
            token_path, token, dataset_id="ASSIST09"
        )
        status = records["standard"]["status"]
        status["code"]["route_commit"] = "f" * 40
        Path(records["standard"]["status_path"]).write_text(
            json.dumps(status), encoding="utf-8"
        )

        with self.assertRaisesRegex(ValueError, "status route commit mismatch"):
            authorize_next(
                state_dir=self.state_dir,
                repo_root=PROJECT_ROOT,
                output_path=self.root / "must-not-issue.json",
            )

    def test_outer_manifest_fingerprint_mismatch_cannot_advance(self) -> None:
        self.initialize()
        token_path, token = self.issue()
        records = self._prepare_pair_artifacts(
            token_path, token, dataset_id="ASSIST09"
        )
        status = records["standard"]["status"]
        status["immutable_inputs"]["architecture_manifest"][
            "architecture_fingerprint"
        ] = "f" * 64
        Path(records["standard"]["status_path"]).write_text(
            json.dumps(status), encoding="utf-8"
        )

        with self.assertRaisesRegex(ValueError, "architecture fingerprint mismatch"):
            authorize_next(
                state_dir=self.state_dir,
                repo_root=PROJECT_ROOT,
                output_path=self.root / "must-not-issue.json",
            )

    def test_outer_cohort_hash_mismatch_cannot_advance(self) -> None:
        self.initialize()
        token_path, token = self.issue()
        records = self._prepare_pair_artifacts(
            token_path, token, dataset_id="ASSIST09"
        )
        status = records["standard"]["status"]
        status["immutable_inputs"]["cohort"]["cohort_sha256"] = "f" * 64
        Path(records["standard"]["status_path"]).write_text(
            json.dumps(status), encoding="utf-8"
        )

        with self.assertRaisesRegex(ValueError, "cohort SHA-256 mismatch"):
            authorize_next(
                state_dir=self.state_dir,
                repo_root=PROJECT_ROOT,
                output_path=self.root / "must-not-issue.json",
            )

    def test_dataset_hash_mismatch_cannot_advance_progress(self) -> None:
        self.initialize()
        token_path, token = self.issue()
        records = self._prepare_pair_artifacts(
            token_path, token, dataset_id="ASSIST09"
        )
        data_path = Path(records["standard"]["consumption"]["data_paths"][0])
        data_path.write_text("tampered\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "dataset.*hash"):
            authorize_next(
                state_dir=self.state_dir,
                repo_root=PROJECT_ROOT,
                output_path=self.root / "must-not-issue.json",
            )

    def test_bound_summary_still_rejects_test_path_after_hash_refresh(self) -> None:
        self.initialize()
        token_path, token = self.issue()
        records = self._prepare_pair_artifacts(
            token_path, token, dataset_id="ASSIST09"
        )
        standard = records["standard"]
        summary = standard["summary"]
        summary["validation_metrics_path"] = "/private/test.csv"
        summary_path = Path(standard["summary_path"])
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        status = standard["status"]
        status["output_hashes"]["validation-summary.json"] = {
            "exists": True,
            **self._fingerprint(summary_path),
        }
        Path(standard["status_path"]).write_text(json.dumps(status), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "test metric/path is forbidden"):
            authorize_next(
                state_dir=self.state_dir,
                repo_root=PROJECT_ROOT,
                output_path=self.root / "must-not-issue.json",
            )

    def test_advance_rechecks_controller_owned_baseline_hash_at_use(self) -> None:
        self.initialize()
        token_path, token = self.issue()
        self._prepare_pair_artifacts(token_path, token, dataset_id="ASSIST09")
        state = json.loads((self.state_dir / "state.json").read_text())
        baseline_path = self.state_dir / "baseline.json"
        baseline = json.loads(baseline_path.read_text())
        baseline["rows"][0]["zero_auc"] = 0.10
        baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "baseline hash"):
            _advance_active_pair(state_dir=self.state_dir, state=state)
        self.assertFalse(any((self.state_dir / "proofs").glob("*.json")))

    def test_two_joint_gate_failures_make_three_successes_unreachable(self) -> None:
        self.initialize()
        first_path, first = self.issue()
        self._prepare_pair_artifacts(
            first_path,
            first,
            dataset_id="ASSIST09",
            standard_overall_auc=0.69,
        )
        second_path = self.root / "capability-2.json"
        second = authorize_next(
            state_dir=self.state_dir,
            repo_root=PROJECT_ROOT,
            output_path=second_path,
        )
        self.assertEqual(second["dataset_id"], "ASSIST17")
        self._prepare_pair_artifacts(
            second_path,
            second,
            dataset_id="ASSIST17",
            weighted_doa=0.61,
        )

        third_path = self.root / "must-not-issue.json"
        with self.assertRaisesRegex(RuntimeError, "primary cohort is unreachable"):
            authorize_next(
                state_dir=self.state_dir,
                repo_root=PROJECT_ROOT,
                output_path=third_path,
            )
        self.assertFalse(third_path.exists())
        state = json.loads((self.state_dir / "state.json").read_text())
        self.assertEqual(state["cursor"], 2)
        self.assertEqual(state["successes"], 0)
        self.assertTrue(state["blocked"])

    def test_final_global_gate_requires_one_zero_delta_at_least_point_001(self) -> None:
        self.initialize()
        token_path, token = self.issue()
        for index, dataset_id in enumerate(DATASET_IDS):
            self._prepare_pair_artifacts(
                token_path,
                token,
                dataset_id=dataset_id,
                zero_auc=0.6005,
            )
            result_path = self.root / f"controller-output-{index}.json"
            result = authorize_next(
                state_dir=self.state_dir,
                repo_root=PROJECT_ROOT,
                output_path=result_path,
            )
            if index < len(DATASET_IDS) - 1:
                token_path = result_path
                token = result

        self.assertTrue(result["complete"])
        self.assertFalse(result["global_pass"])
        self.assertNotIn("capabilities", result)

    def test_authorize_cli_rejects_caller_decision_json(self) -> None:
        self.initialize()
        decision_path = self.root / "fake-decision.json"
        decision_path.write_text(
            json.dumps({"successes": 4, "remaining": 0}), encoding="utf-8"
        )

        with self.assertRaises(SystemExit):
            validation_main(
                [
                    "authorize",
                    "--controller-state-dir",
                    str(self.state_dir),
                    "--repo-root",
                    str(PROJECT_ROOT),
                    "--decision",
                    str(decision_path),
                    "--output",
                    str(self.root / "token.json"),
                ]
            )
        state = json.loads((self.state_dir / "state.json").read_text())
        self.assertEqual(state["cursor"], 0)
        self.assertIsNone(state["active_pair"])


if __name__ == "__main__":
    unittest.main()
