from __future__ import annotations

import json
import hashlib
import subprocess
import sys
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
        self.repo_root = self.root / "route-repo"
        (self.repo_root / "scripts").mkdir(parents=True)
        self.route_runner = self.repo_root / "scripts" / "run_remote_campaign.py"
        self.route_runner.write_text("# tracked route runner\n", encoding="utf-8")
        (self.repo_root / ".gitignore").write_text(
            "ignored-runtime/\n", encoding="utf-8"
        )
        subprocess.run(["git", "init", "-q"], cwd=self.repo_root, check=True)
        subprocess.run(
            ["git", "config", "user.email", "fixture@example.com"],
            cwd=self.repo_root,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Fixture"],
            cwd=self.repo_root,
            check=True,
        )
        subprocess.run(["git", "add", "."], cwd=self.repo_root, check=True)
        subprocess.run(
            ["git", "commit", "-qm", "route fixture"],
            cwd=self.repo_root,
            check=True,
        )
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
        self.data_root = self.root / "registered-data"
        for standard_dir, holdout_dir in DATASET_DIRECTORIES.values():
            for split_id, directory_name in (
                ("standard", standard_dir),
                ("holdout", holdout_dir),
            ):
                directory = self.data_root / directory_name
                directory.mkdir(parents=True, exist_ok=True)
                for name in ("train.csv", "valid.csv", "Q_matrix.csv"):
                    (directory / name).write_text(
                        f"{directory_name},{name}\n", encoding="utf-8"
                    )
                if split_id == "holdout":
                    (directory / "student_concept_holdout_assignments.csv").write_text(
                        f"{directory_name},holdout\n", encoding="utf-8"
                    )
        self.artifact_root = self.root / "registered-artifacts"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def initialize(self) -> dict[str, object]:
        return initialize_controller(
            state_dir=self.state_dir,
            repo_root=self.repo_root,
            cohort_path=self.cohort_path,
            manifest_path=self.manifest_path,
            architecture="m2",
            baseline_rows_path=self.baseline_path,
            data_root=self.data_root,
            artifact_root=self.artifact_root,
        )

    def issue(self) -> tuple[Path, dict[str, object]]:
        token_path = self.root / "capability.json"
        token = authorize_next(
            state_dir=self.state_dir,
            repo_root=self.repo_root,
            output_path=token_path,
        )
        return token_path, token

    def consume(
        self,
        token_path: Path,
        *,
        split_id: str,
    ) -> dict[str, object]:
        state = json.loads((self.state_dir / "state.json").read_text())
        self._set_unit_launch(state, dataset_id="ASSIST09")
        attempt_dir = self.artifact_root / "ASSIST09" / split_id / "attempt-001"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        return consume_split_capability(
            state_dir=self.state_dir,
            repo_root=self.repo_root,
            token_path=token_path,
            dataset_id="ASSIST09",
            split_id=split_id,
            architecture="m2",
            data_root=Path(state["data_root"]),
            raw_output=Path("validation-summary.json"),
            attempt_dir=attempt_dir,
        )

    def _set_unit_launch(
        self, state: dict[str, object], *, dataset_id: str
    ) -> dict[str, object]:
        state["launch"] = {
            "phase": "running",
            "launch_id": f"unit-{dataset_id}",
            "counter": state["active_pair"]["counter"],
            "dataset_id": dataset_id,
            "before_attempts": {"standard": [], "holdout": []},
            "commands": {
                "standard": ["outer", "--", "unit-standard"],
                "holdout": ["outer", "--", "unit-holdout"],
            },
        }
        (self.state_dir / "state.json").write_text(json.dumps(state))
        return state

    def _advance_unit_launch(self) -> dict[str, object]:
        state = json.loads((self.state_dir / "state.json").read_text())
        result = _advance_active_pair(state_dir=self.state_dir, state=state)
        result["launch"] = None
        (self.state_dir / "state.json").write_text(json.dumps(result))
        return result

    def _fingerprint(self, path: Path) -> dict[str, object]:
        return {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def _write_controlled_outer_runner(self) -> Path:
        path = self.root / "controlled_outer_runner.py"
        path.write_text(
            r'''
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, __REPO_ROOT__)
from scripts.unified_validation_controller import consume_split_capability

args = sys.argv[1:]
def value(flag):
    return args[args.index(flag) + 1]
separator = args.index("--")
child = args[separator + 1:]
def child_value(flag):
    return child[child.index(flag) + 1]
dataset_paths = [Path(args[index + 1]).resolve() for index, item in enumerate(args) if item == "--dataset-file"]
artifact_root = Path(value("--artifact-root")).resolve()
artifact_root.mkdir(parents=True, exist_ok=True)
attempts = sorted(path for path in artifact_root.glob("attempt-*") if path.is_dir())
attempt_dir = artifact_root / f"attempt-{len(attempts) + 1:03d}"
attempt_dir.mkdir()
if os.environ.get("CONTROLLED_OUTER_EXTRA_ATTEMPT") == child_value("--split-id"):
    (artifact_root / "attempt-999").mkdir()
(attempt_dir / "status.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
split_id = child_value("--split-id")
if os.environ.get("CONTROLLED_OUTER_FAIL_SPLIT") == split_id:
    raise SystemExit(7)
binding = consume_split_capability(
    state_dir=Path(child_value("--controller-state-dir")),
    repo_root=Path(child_value("--repo-root")),
    token_path=Path(child_value("--capability")),
    dataset_id=child_value("--dataset-id"),
    split_id=split_id,
    architecture=child_value("--architecture"),
    data_root=Path(child_value("--data-root")),
    raw_output=Path(child_value("--output")),
    attempt_dir=attempt_dir,
)
summary_path = Path(binding["summary_path"])
summary = {
    "schema_version": 1,
    "dataset_id": binding["dataset_id"],
    "split_id": split_id,
    "architecture": binding["architecture"],
    "architecture_manifest": binding["architecture_manifest"],
    "architecture_fingerprint": binding["architecture_fingerprint"],
    "cohort_sha256": binding["cohort_sha256"],
    "controller_id": binding["controller_id"],
    "controller_route_commit": binding["route_commit"],
    "capability_counter": binding["counter"],
    "capability_nonce": binding["nonce"],
    "seed": 42,
    "evaluation_input_role": "valid",
    "overall_auc": 0.71 if split_id == "standard" else 0.70,
    "zero_auc": 0.61,
    "ordinary_doa": 0.62,
    "weighted_doa": 0.63,
    "mastery_shape": [2, 3],
    "final_loss": 0.4,
    "peak_gpu_memory_gb": 0.1,
    "numerical_recipe": {"mastery_loss_weight": 0.1},
}
summary_path.write_text(json.dumps(summary), encoding="utf-8")
def fingerprint(file_path):
    data = file_path.read_bytes()
    return {"path": str(file_path.resolve()), "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
manifest_path = Path(value("--architecture-manifest")).resolve()
cohort_path = Path(value("--cohort")).resolve()
manifest = json.loads(manifest_path.read_text())
cohort = json.loads(cohort_path.read_text())
route = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(value("--repo-root")), check=True, capture_output=True, text=True).stdout.strip()
status = {
    "status": "completed",
    "exit_code": 0,
    "invocation": {"command": child},
    "parameters": {"seed": 42},
    "code": {"route_commit": route},
    "immutable_inputs": {
        "architecture_manifest": {**fingerprint(manifest_path), "architecture_fingerprint": binding["architecture_fingerprint"]},
        "cohort": {**fingerprint(cohort_path), "cohort_sha256": cohort["cohort_sha256"]},
    },
    "datasets": [fingerprint(item) for item in dataset_paths],
    "output_hashes": {"validation-summary.json": {"exists": True, **fingerprint(summary_path)}},
}
(attempt_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
print(attempt_dir)
'''.replace("__REPO_ROOT__", repr(str(PROJECT_ROOT))).lstrip(),
            encoding="utf-8",
        )
        return path

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
        state = json.loads((self.state_dir / "state.json").read_text())
        data_root = Path(state["data_root"])
        self._set_unit_launch(state, dataset_id=dataset_id)

        records: dict[str, dict[str, object]] = {}
        for split_id in ("standard", "holdout"):
            attempt_dir = self.artifact_root / dataset_id / split_id / "attempt-001"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            record = consume_split_capability(
                state_dir=self.state_dir,
                repo_root=self.repo_root,
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
                "numerical_recipe": {"mastery_loss_weight": 0.1},
            }
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            status = {
                "status": "completed",
                "exit_code": 0,
                "invocation": {"command": [f"unit-{split_id}"]},
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
        state = json.loads((self.state_dir / "state.json").read_text())
        state["launch"]["attempts"] = {
            split_id: str(records[split_id]["consumption"]["attempt_dir"])
            for split_id in ("standard", "holdout")
        }
        (self.state_dir / "state.json").write_text(json.dumps(state))
        return records

    def test_initialization_copies_and_binds_registered_inputs(self) -> None:
        state = self.initialize()

        expected_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo_root,
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

    def test_initialization_rejects_test_reference_in_baseline(self) -> None:
        rows = json.loads(json.dumps(self.baseline_rows))
        rows[0]["validation_path"] = "/private/test.csv"
        self.baseline_path.write_text(json.dumps({"rows": rows}), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "test metric/path is forbidden"):
            self.initialize()
        self.assertFalse(self.state_dir.exists())

    def test_initialization_registers_validation_data_and_artifact_roots(self) -> None:
        data_root = self.root / "all-validation-data"
        expected_paths: list[Path] = []
        for standard_dir, holdout_dir in DATASET_DIRECTORIES.values():
            for split_id, directory_name in (
                ("standard", standard_dir),
                ("holdout", holdout_dir),
            ):
                directory = data_root / directory_name
                directory.mkdir(parents=True)
                for name in ("train.csv", "valid.csv", "Q_matrix.csv"):
                    path = directory / name
                    path.write_text(f"{directory_name},{name}\n", encoding="utf-8")
                    expected_paths.append(path.resolve())
                if split_id == "holdout":
                    path = directory / "student_concept_holdout_assignments.csv"
                    path.write_text(f"{directory_name},holdout\n", encoding="utf-8")
                    expected_paths.append(path.resolve())
        artifact_root = self.root / "registered-artifacts"

        state = initialize_controller(
            state_dir=self.state_dir,
            repo_root=self.repo_root,
            cohort_path=self.cohort_path,
            manifest_path=self.manifest_path,
            architecture="m2",
            baseline_rows_path=self.baseline_path,
            data_root=data_root,
            artifact_root=artifact_root,
        )

        records = state["validation_data"]
        registered_paths = {
            Path(record["path"]) for split in records.values() for record in split
        }
        copied_root = self.state_dir / "data"
        self.assertTrue(all(path.is_relative_to(copied_root) for path in registered_paths))
        original_hashes = {
            path.relative_to(data_root): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in expected_paths
        }
        for path in expected_paths:
            path.write_text("source changed after registration\n", encoding="utf-8")
        copied_hashes = {
            path.relative_to(copied_root): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in registered_paths
        }
        self.assertEqual(copied_hashes, original_hashes)
        self.assertEqual(state["artifact_root"], str(artifact_root.resolve()))

    def test_controller_owned_validation_copy_tamper_is_rejected(self) -> None:
        self.initialize()
        state = json.loads((self.state_dir / "state.json").read_text())
        copied_path = Path(next(iter(state["validation_data"].values()))[0]["path"])
        copied_path.write_text("tampered controller copy\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "validation data.*hash"):
            authorize_next(
                state_dir=self.state_dir,
                repo_root=self.repo_root,
                output_path=self.root / "must-not-issue.json",
            )

    def test_controller_init_cli_registers_the_iteration(self) -> None:
        result = validation_main(
            [
                "controller-init",
                "--controller-state-dir",
                str(self.state_dir),
                "--repo-root",
                str(self.repo_root),
                "--cohort",
                str(self.cohort_path),
                "--architecture",
                "m2",
                "--architecture-manifest",
                str(self.manifest_path),
                "--baseline-rows",
                str(self.baseline_path),
                "--data-root",
                str(self.data_root),
                "--artifact-root",
                str(self.artifact_root),
            ]
        )

        self.assertEqual(result, 0)
        self.assertTrue((self.state_dir / "state.json").is_file())

    def test_controller_init_rejects_modified_tracked_route_at_same_head(self) -> None:
        self.route_runner.write_text("# locally modified runner\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "dirty"):
            self.initialize()
        self.assertFalse(self.state_dir.exists())

    def test_controller_init_rejects_staged_route_at_same_head(self) -> None:
        self.route_runner.write_text("# staged runner\n", encoding="utf-8")
        subprocess.run(
            ["git", "add", "scripts/run_remote_campaign.py"],
            cwd=self.repo_root,
            check=True,
        )

        with self.assertRaisesRegex(ValueError, "dirty"):
            self.initialize()
        self.assertFalse(self.state_dir.exists())

    def test_controller_init_rejects_untracked_nonignored_route_file(self) -> None:
        (self.repo_root / "untracked.py").write_text(
            "# untracked route code\n", encoding="utf-8"
        )

        with self.assertRaisesRegex(ValueError, "dirty"):
            self.initialize()
        self.assertFalse(self.state_dir.exists())

    def test_dirty_route_blocks_authorize_and_run_pair_before_launch(self) -> None:
        self.initialize()
        self.route_runner.write_text("# dirty before authorize\n", encoding="utf-8")
        blocked_output = self.root / "dirty-token.json"
        with self.assertRaisesRegex(ValueError, "dirty"):
            authorize_next(
                state_dir=self.state_dir,
                repo_root=self.repo_root,
                output_path=blocked_output,
            )
        self.assertFalse(blocked_output.exists())

        subprocess.run(
            ["git", "restore", "scripts/run_remote_campaign.py"],
            cwd=self.repo_root,
            check=True,
        )
        self.issue()
        self.route_runner.write_text("# dirty before run-pair\n", encoding="utf-8")
        with patch("scripts.unified_validation_controller._outer_command") as command:
            with self.assertRaisesRegex(ValueError, "dirty"):
                controller_module.run_registered_pair(
                    state_dir=self.state_dir,
                    repo_root=self.repo_root,
                )
        command.assert_not_called()

    def test_ignored_files_and_external_controller_artifacts_are_allowed(self) -> None:
        ignored = self.repo_root / "ignored-runtime" / "status.json"
        ignored.parent.mkdir()
        ignored.write_text("{}\n", encoding="utf-8")

        state = self.initialize()
        token_path, token = self.issue()

        self.assertEqual(state["route_commit"], token["route_commit"])
        self.assertTrue(token_path.is_file())
        self.assertFalse(self.state_dir.is_relative_to(self.repo_root))
        self.assertFalse(self.artifact_root.is_relative_to(self.repo_root))

    def test_route_verification_never_executes_git_from_inherited_path(self) -> None:
        fake_bin = self.root / "fake-bin"
        fake_bin.mkdir()
        marker = self.root / "fake-git-executed"
        fake_git = fake_bin / "git"
        fake_git.write_text(
            "#!/bin/sh\n"
            f"touch {marker}\n"
            'exec /usr/bin/git "$@"\n',
            encoding="utf-8",
        )
        fake_git.chmod(0o755)

        with patch.dict(
            "os.environ",
            {"PATH": f"{fake_bin}:/usr/bin:/bin"},
            clear=False,
        ):
            self.initialize()

        self.assertFalse(marker.exists())

    def test_git_repository_environment_cannot_redirect_route_verification(self) -> None:
        self.initialize()
        alternate = self.root / "alternate-clean-route"
        subprocess.run(
            ["/usr/bin/git", "clone", "-q", str(self.repo_root), str(alternate)],
            check=True,
        )
        empty_config = self.root / "empty-git-config"
        empty_config.write_text("", encoding="utf-8")
        self.route_runner.write_text("# dirty registered route\n", encoding="utf-8")
        poisoned = {
            "GIT_DIR": str(alternate / ".git"),
            "GIT_COMMON_DIR": str(alternate / ".git"),
            "GIT_WORK_TREE": str(alternate),
            "GIT_INDEX_FILE": str(alternate / ".git" / "index"),
            "GIT_OBJECT_DIRECTORY": str(alternate / ".git" / "objects"),
            "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(
                alternate / ".git" / "objects"
            ),
            "GIT_CONFIG_GLOBAL": str(empty_config),
            "GIT_CONFIG_SYSTEM": str(empty_config),
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "core.worktree",
            "GIT_CONFIG_VALUE_0": str(alternate),
        }
        blocked_output = self.root / "redirected-token.json"

        with patch.dict("os.environ", poisoned, clear=False):
            with self.assertRaisesRegex(ValueError, "dirty"):
                authorize_next(
                    state_dir=self.state_dir,
                    repo_root=self.repo_root,
                    output_path=blocked_output,
                )

        self.assertFalse(blocked_output.exists())

    def test_local_core_worktree_cannot_redirect_route_verification(self) -> None:
        self.initialize()
        alternate = self.root / "local-config-clean-route"
        subprocess.run(
            ["/usr/bin/git", "clone", "-q", str(self.repo_root), str(alternate)],
            check=True,
        )
        subprocess.run(
            ["/usr/bin/git", "config", "core.worktree", str(alternate)],
            cwd=self.repo_root,
            check=True,
        )
        self.route_runner.write_text("# dirty real work tree\n", encoding="utf-8")
        blocked_output = self.root / "local-config-token.json"

        with self.assertRaisesRegex(ValueError, "dirty"):
            authorize_next(
                state_dir=self.state_dir,
                repo_root=self.repo_root,
                output_path=blocked_output,
            )

        self.assertFalse(blocked_output.exists())

    def test_local_fsmonitor_hook_is_disabled_for_route_verification(self) -> None:
        marker = self.root / "fsmonitor-executed"
        monitor = self.root / "fsmonitor-hook"
        monitor.write_text(
            "#!/bin/sh\n"
            f"touch {marker}\n"
            "exit 0\n",
            encoding="utf-8",
        )
        monitor.chmod(0o755)
        subprocess.run(
            ["/usr/bin/git", "config", "core.fsmonitor", str(monitor)],
            cwd=self.repo_root,
            check=True,
        )

        controller_module._route_head(self.repo_root)

        self.assertFalse(marker.exists())

    def test_clean_linked_worktree_route_is_supported(self) -> None:
        linked = self.root / "linked-route"
        subprocess.run(
            [
                "/usr/bin/git",
                "worktree",
                "add",
                "-q",
                "-b",
                "linked-route-fixture",
                str(linked),
            ],
            cwd=self.repo_root,
            check=True,
        )
        expected = subprocess.run(
            ["/usr/bin/git", "rev-parse", "HEAD"],
            cwd=linked,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        self.assertEqual(controller_module._route_head(linked), expected)

    def test_assume_unchanged_cannot_hide_modified_tracked_route(self) -> None:
        subprocess.run(
            [
                "/usr/bin/git",
                "update-index",
                "--assume-unchanged",
                "scripts/run_remote_campaign.py",
            ],
            cwd=self.repo_root,
            check=True,
        )
        self.route_runner.write_text(
            "# hidden assume-unchanged modification\n", encoding="utf-8"
        )

        with self.assertRaisesRegex(ValueError, "index|dirty"):
            self.initialize()
        self.assertFalse(self.state_dir.exists())

    def test_skip_worktree_cannot_hide_modified_tracked_route(self) -> None:
        subprocess.run(
            [
                "/usr/bin/git",
                "update-index",
                "--skip-worktree",
                "scripts/run_remote_campaign.py",
            ],
            cwd=self.repo_root,
            check=True,
        )
        self.route_runner.write_text(
            "# hidden skip-worktree modification\n", encoding="utf-8"
        )

        with self.assertRaisesRegex(ValueError, "index|dirty"):
            self.initialize()
        self.assertFalse(self.state_dir.exists())

    def test_clean_route_supports_tracked_filename_with_newline(self) -> None:
        unusual = self.repo_root / "scripts" / "line\nbreak.py"
        unusual.write_text("# unusual tracked path\n", encoding="utf-8")
        subprocess.run(
            ["/usr/bin/git", "add", "--", str(unusual.relative_to(self.repo_root))],
            cwd=self.repo_root,
            check=True,
        )
        subprocess.run(
            ["/usr/bin/git", "commit", "-qm", "add unusual path"],
            cwd=self.repo_root,
            check=True,
        )

        expected = subprocess.run(
            ["/usr/bin/git", "rev-parse", "HEAD"],
            cwd=self.repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.assertEqual(controller_module._route_head(self.repo_root), expected)

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
                repo_root=self.repo_root,
                output_path=output_path,
            )

        state = json.loads((self.state_dir / "state.json").read_text())
        self.assertIsNone(state["active_pair"])
        self.assertEqual(state["issuance_counter"], 0)
        self.assertFalse(any((self.state_dir / "issued").glob("*.json")))

    def test_forged_output_reservation_cannot_select_dataset_or_counter(self) -> None:
        state = self.initialize()
        original_state = json.loads(json.dumps(state))
        output_path = self.root / "forged-reservation.json"
        common = {
            "schema_version": 1,
            "controller_id": state["controller_id"],
            "route_commit": state["route_commit"],
            "counter": 999,
            "dataset_id": "XES3G5M",
        }
        token = {
            **common,
            "capabilities": {
                split_id: {
                    **common,
                    "split_id": split_id,
                    "nonce": split_id[0] * 64,
                }
                for split_id in ("standard", "holdout")
            },
        }
        output_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "controller_reservation": state["controller_id"],
                    "token": token,
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaises(FileExistsError):
            authorize_next(
                state_dir=self.state_dir,
                repo_root=self.repo_root,
                output_path=output_path,
            )

        self.assertEqual(
            json.loads((self.state_dir / "state.json").read_text()), original_state
        )
        self.assertFalse(any((self.state_dir / "issued").rglob("*.json")))

    def test_forged_output_reservation_cannot_traverse_registry_paths(self) -> None:
        state = self.initialize()
        original_state = json.loads(json.dumps(state))
        output_path = self.root / "traversal-reservation.json"
        common = {
            "schema_version": 1,
            "controller_id": state["controller_id"],
            "route_commit": state["route_commit"],
            "counter": 1,
            "dataset_id": "../XES3G5M",
        }
        token = {
            **common,
            "capabilities": {
                "standard": {
                    **common,
                    "split_id": "standard",
                    "nonce": "../../escape",
                },
                "holdout": {
                    **common,
                    "split_id": "holdout",
                    "nonce": "a" * 64,
                },
            },
        }
        output_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "controller_reservation": state["controller_id"],
                    "token": token,
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaises(FileExistsError):
            authorize_next(
                state_dir=self.state_dir,
                repo_root=self.repo_root,
                output_path=output_path,
            )

        self.assertEqual(
            json.loads((self.state_dir / "state.json").read_text()), original_state
        )
        self.assertFalse(any((self.state_dir / "issued").rglob("*.json")))
        self.assertFalse((self.state_dir / "escape.json").exists())

    def test_pending_token_validator_binds_exact_controller_state(self) -> None:
        state = self.initialize()
        common = {
            "schema_version": 1,
            "controller_id": state["controller_id"],
            "route_commit": state["route_commit"],
            "counter": 1,
            "dataset_id": "ASSIST09",
        }
        valid = {
            **common,
            "capabilities": {
                split_id: {
                    **common,
                    "split_id": split_id,
                    "nonce": character * 64,
                }
                for split_id, character in (("standard", "a"), ("holdout", "b"))
            },
        }
        controller_module._validate_pending_token(state, valid)

        invalid_tokens = []
        for field, value in (
            ("controller_id", "forged"),
            ("route_commit", "f" * 40),
            ("counter", 999),
            ("dataset_id", "XES3G5M"),
        ):
            forged = json.loads(json.dumps(valid))
            forged[field] = value
            invalid_tokens.append(forged)
        forged = json.loads(json.dumps(valid))
        forged["capabilities"]["standard"]["counter"] = 999
        invalid_tokens.append(forged)
        forged = json.loads(json.dumps(valid))
        forged["capabilities"]["extra"] = forged["capabilities"]["standard"]
        invalid_tokens.append(forged)
        forged = json.loads(json.dumps(valid))
        forged["capabilities"]["standard"]["nonce"] = "A" * 64
        invalid_tokens.append(forged)
        forged = json.loads(json.dumps(valid))
        forged["capabilities"]["standard"]["nonce"] = "../escape"
        invalid_tokens.append(forged)

        for index, forged in enumerate(invalid_tokens):
            with self.subTest(case=index), self.assertRaises(ValueError):
                controller_module._validate_pending_token(state, forged)

    def test_consume_uses_active_registry_filename_not_caller_fields(self) -> None:
        self.initialize()
        token_path, _ = self.issue()
        state = json.loads((self.state_dir / "state.json").read_text())
        self._set_unit_launch(state, dataset_id="ASSIST09")
        attempt_dir = self.artifact_root / "ASSIST09" / "standard" / "attempt-001"
        attempt_dir.mkdir(parents=True)

        with patch(
            "scripts.unified_validation_controller._capability_path",
            side_effect=AssertionError("consume reconstructed an untrusted path"),
        ):
            consumed = consume_split_capability(
                state_dir=self.state_dir,
                repo_root=self.repo_root,
                token_path=token_path,
                dataset_id="ASSIST09",
                split_id="standard",
                architecture="m2",
                data_root=Path(state["data_root"]),
                raw_output=Path("validation-summary.json"),
                attempt_dir=attempt_dir,
            )
        self.assertEqual(consumed["split_id"], "standard")

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
                    repo_root=self.repo_root,
                    output_path=output_path,
                )

        state = json.loads((self.state_dir / "state.json").read_text())
        self.assertIsNone(state["active_pair"])
        self.assertEqual(state["issuance_counter"], 0)
        self.assertFalse(any((self.state_dir / "issued").glob("*.json")))
        self.assertFalse(output_path.exists())

    def test_process_death_before_token_publish_is_idempotently_recovered(self) -> None:
        self.initialize()
        output_path = self.root / "crash-capability.json"
        code = f"""
import os
from pathlib import Path
from scripts import unified_validation_controller as controller

original = controller._atomic_json
output = Path({str(output_path)!r}).resolve()
def crash(path, payload):
    if path.resolve() == output:
        os._exit(91)
    original(path, payload)
controller._atomic_json = crash
controller.authorize_next(
    state_dir=Path({str(self.state_dir)!r}),
    repo_root=Path({str(self.repo_root)!r}),
    output_path=output,
)
"""
        crashed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            check=False,
            env=controller_module._sanitized_subprocess_env(),
        )
        self.assertEqual(crashed.returncode, 91)

        recovered = authorize_next(
            state_dir=self.state_dir,
            repo_root=self.repo_root,
            output_path=output_path,
        )

        self.assertEqual(json.loads(output_path.read_text()), recovered)
        self.assertEqual(recovered["dataset_id"], "ASSIST09")
        state = json.loads((self.state_dir / "state.json").read_text())
        self.assertTrue(state["active_pair"]["token_published"])
        self.assertIsNone(state["pending_issuance"])

    def test_process_death_after_placeholder_before_pending_generates_fresh_token(self) -> None:
        self.initialize()
        output_path = self.root / "placeholder-crash-capability.json"
        code = f"""
import os
from pathlib import Path
from scripts import unified_validation_controller as controller

original = controller._exclusive_bytes
output = Path({str(output_path)!r}).resolve()
def crash(path, data, **kwargs):
    original(path, data, **kwargs)
    if path.resolve() == output:
        os._exit(92)
controller._exclusive_bytes = crash
controller.authorize_next(
    state_dir=Path({str(self.state_dir)!r}),
    repo_root=Path({str(self.repo_root)!r}),
    output_path=output,
)
"""
        crashed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            check=False,
            env=controller_module._sanitized_subprocess_env(),
        )
        self.assertEqual(crashed.returncode, 92)
        state_after_crash = json.loads((self.state_dir / "state.json").read_text())
        self.assertIsNone(state_after_crash["pending_issuance"])
        self.assertIsNone(state_after_crash["active_pair"])
        self.assertEqual(output_path.stat().st_size, 0)
        self.assertFalse(any((self.state_dir / "issued").glob("*.json")))

        recovered = authorize_next(
            state_dir=self.state_dir,
            repo_root=self.repo_root,
            output_path=output_path,
        )
        self.assertEqual(recovered["counter"], 1)
        self.assertEqual(recovered["dataset_id"], "ASSIST09")
        self.assertEqual(set(recovered["capabilities"]), {"standard", "holdout"})
        self.assertEqual(json.loads(output_path.read_text()), recovered)

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
                repo_root=self.repo_root,
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

    def test_manual_run_split_cannot_consume_without_controller_launch(self) -> None:
        self.initialize()
        token_path, _ = self.issue()
        attempt_dir = self.artifact_root / "ASSIST09" / "standard" / "attempt-001"
        attempt_dir.mkdir(parents=True)
        state = json.loads((self.state_dir / "state.json").read_text())

        with self.assertRaisesRegex(ValueError, "controller launch"):
            consume_split_capability(
                state_dir=self.state_dir,
                repo_root=self.repo_root,
                token_path=token_path,
                dataset_id="ASSIST09",
                split_id="standard",
                architecture="m2",
                data_root=Path(state["data_root"]),
                raw_output=Path("validation-summary.json"),
                attempt_dir=attempt_dir,
            )
        self.assertEqual(len(list((self.state_dir / "issued").glob("*.json"))), 2)

    def test_run_pair_launches_real_outer_processes_and_advances(self) -> None:
        self.initialize()
        self.issue()
        controlled_runner = self._write_controlled_outer_runner()

        with patch.object(
            controller_module,
            "OUTER_RUNNER_OVERRIDE",
            controlled_runner,
            create=True,
        ):
            result = controller_module.run_registered_pair(
                state_dir=self.state_dir,
                repo_root=self.repo_root,
            )

        self.assertEqual(result["dataset_id"], "ASSIST09")
        self.assertTrue(result["joint_success"])
        state = json.loads((self.state_dir / "state.json").read_text())
        self.assertEqual(state["cursor"], 1)
        self.assertEqual(state["successes"], 1)
        self.assertIsNone(state["launch"])
        for split_id in ("standard", "holdout"):
            attempts = list(
                (self.artifact_root / "ASSIST09" / split_id).glob("attempt-*")
            )
            self.assertEqual(len(attempts), 1)
            self.assertEqual(
                json.loads((attempts[0] / "status.json").read_text())["status"],
                "completed",
            )

    def test_authorize_cannot_advance_a_completed_caller_authored_pair(self) -> None:
        self.initialize()
        token_path, token = self.issue()
        self._prepare_pair_artifacts(token_path, token, dataset_id="ASSIST09")
        state = json.loads((self.state_dir / "state.json").read_text())
        state["launch"] = None
        (self.state_dir / "state.json").write_text(json.dumps(state))

        with self.assertRaisesRegex(RuntimeError, "requires controller run-pair"):
            authorize_next(
                state_dir=self.state_dir,
                repo_root=self.repo_root,
                output_path=self.root / "forbidden-next-token.json",
            )
        self.assertFalse((self.root / "forbidden-next-token.json").exists())

    def test_progress_rejects_consumption_from_a_different_launch_attempt(self) -> None:
        self.initialize()
        token_path, token = self.issue()
        self._prepare_pair_artifacts(token_path, token, dataset_id="ASSIST09")
        state = json.loads((self.state_dir / "state.json").read_text())
        state["launch"]["attempts"]["standard"] = str(
            self.artifact_root / "ASSIST09" / "standard" / "attempt-999"
        )
        (self.state_dir / "state.json").write_text(json.dumps(state))

        with self.assertRaisesRegex(ValueError, "launch attempts"):
            self._advance_unit_launch()

    def test_nonzero_outer_exit_permanently_blocks_controller(self) -> None:
        self.initialize()
        self.issue()
        controlled_runner = self._write_controlled_outer_runner()

        with patch.object(controller_module, "OUTER_RUNNER_OVERRIDE", controlled_runner), patch.dict(
            "os.environ", {"CONTROLLED_OUTER_FAIL_SPLIT": "standard"}, clear=False
        ), self.assertRaisesRegex(RuntimeError, "outer campaign failed"):
            controller_module.run_registered_pair(
                state_dir=self.state_dir, repo_root=self.repo_root
            )

        state = json.loads((self.state_dir / "state.json").read_text())
        self.assertTrue(state["blocked"])
        self.assertEqual(state["launch"]["phase"], "failed")
        with self.assertRaisesRegex(RuntimeError, "blocked/complete"):
            controller_module.run_registered_pair(
                state_dir=self.state_dir, repo_root=self.repo_root
            )

    def test_multiple_new_attempts_permanently_block_controller(self) -> None:
        self.initialize()
        self.issue()
        controlled_runner = self._write_controlled_outer_runner()

        with patch.object(controller_module, "OUTER_RUNNER_OVERRIDE", controlled_runner), patch.dict(
            "os.environ", {"CONTROLLED_OUTER_EXTRA_ATTEMPT": "standard"}, clear=False
        ), self.assertRaisesRegex(RuntimeError, "created 2 attempts"):
            controller_module.run_registered_pair(
                state_dir=self.state_dir, repo_root=self.repo_root
            )

        state = json.loads((self.state_dir / "state.json").read_text())
        self.assertTrue(state["blocked"])
        self.assertEqual(state["launch"]["phase"], "failed")

    def test_interrupted_running_launch_journal_permanently_blocks_controller(self) -> None:
        self.initialize()
        self.issue()
        state = json.loads((self.state_dir / "state.json").read_text())
        state["launch"] = {
            "phase": "running",
            "launch_id": "abandoned-launch",
            "owner_pid": 999999,
            "counter": 1,
            "dataset_id": "ASSIST09",
            "before_attempts": {"standard": [], "holdout": []},
            "commands": {"standard": [], "holdout": []},
        }
        (self.state_dir / "state.json").write_text(json.dumps(state))

        with self.assertRaisesRegex(RuntimeError, "prior controller launch was interrupted"):
            controller_module.run_registered_pair(
                state_dir=self.state_dir, repo_root=self.repo_root
            )
        state = json.loads((self.state_dir / "state.json").read_text())
        self.assertTrue(state["blocked"])
        self.assertEqual(state["launch"]["phase"], "interrupted")
        with self.assertRaisesRegex(RuntimeError, "blocked/complete"):
            controller_module.run_registered_pair(
                state_dir=self.state_dir, repo_root=self.repo_root
            )

    def test_consumed_publish_failure_leaves_issued_capability_retryable(self) -> None:
        self.initialize()
        token_path, token = self.issue()
        state = json.loads((self.state_dir / "state.json").read_text())
        self._set_unit_launch(state, dataset_id="ASSIST09")
        attempt_dir = self.artifact_root / "ASSIST09" / "standard" / "attempt-001"
        attempt_dir.mkdir(parents=True)
        original_atomic_json = controller_module._atomic_json

        def fail_consumed(path: Path, payload: object) -> None:
            if path.parent == self.state_dir / "consumed":
                raise OSError("injected consumed publish failure")
            original_atomic_json(path, payload)

        kwargs = {
            "state_dir": self.state_dir,
            "repo_root": self.repo_root,
            "token_path": token_path,
            "dataset_id": "ASSIST09",
            "split_id": "standard",
            "architecture": "m2",
            "data_root": Path(state["data_root"]),
            "raw_output": Path("validation-summary.json"),
            "attempt_dir": attempt_dir,
        }
        with patch(
            "scripts.unified_validation_controller._atomic_json",
            side_effect=fail_consumed,
        ):
            with self.assertRaisesRegex(OSError, "consumed publish"):
                consume_split_capability(**kwargs)

        self.assertEqual(len(list((self.state_dir / "issued").glob("*.json"))), 2)
        self.assertFalse(any((self.state_dir / "consumed").glob("*.json")))
        recovered = consume_split_capability(**kwargs)
        self.assertEqual(
            recovered["nonce"], token["capabilities"]["standard"]["nonce"]
        )

    def test_cleanup_failure_leaves_complete_consumed_record_authoritative(self) -> None:
        self.initialize()
        token_path, token = self.issue()
        state = json.loads((self.state_dir / "state.json").read_text())
        self._set_unit_launch(state, dataset_id="ASSIST09")
        attempt_dir = self.artifact_root / "ASSIST09" / "standard" / "attempt-001"
        attempt_dir.mkdir(parents=True)
        issued_dir = self.state_dir / "issued"
        original_unlink = controller_module._unlink_fsync

        def fail_issued_cleanup(path: Path) -> None:
            if path.parent == issued_dir:
                raise OSError("injected issued cleanup failure")
            original_unlink(path)

        kwargs = {
            "state_dir": self.state_dir,
            "repo_root": self.repo_root,
            "token_path": token_path,
            "dataset_id": "ASSIST09",
            "split_id": "standard",
            "architecture": "m2",
            "data_root": Path(state["data_root"]),
            "raw_output": Path("validation-summary.json"),
            "attempt_dir": attempt_dir,
        }
        with patch(
            "scripts.unified_validation_controller._unlink_fsync",
            side_effect=fail_issued_cleanup,
        ):
            with self.assertRaisesRegex(OSError, "issued cleanup"):
                consume_split_capability(**kwargs)

        consumed_paths = list((self.state_dir / "consumed").glob("*.json"))
        self.assertEqual(len(consumed_paths), 1)
        consumed = json.loads(consumed_paths[0].read_text())
        self.assertEqual(consumed["attempt_dir"], str(attempt_dir.resolve()))
        self.assertEqual(len(consumed["data_paths"]), 3)
        with self.assertRaisesRegex(ValueError, "consumed|registry"):
            consume_split_capability(**kwargs)

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
            data_root=self.data_root,
            artifact_root=self.artifact_root,
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
        state = json.loads((self.state_dir / "state.json").read_text())
        self._set_unit_launch(state, dataset_id="ASSIST09")
        attempt_dir = self.artifact_root / "ASSIST09" / "standard" / "attempt-001"
        attempt_dir.mkdir(parents=True)
        output_path = attempt_dir / "validation-summary.json"

        with patch.dict(
            "os.environ",
            {"CAMPAIGN_ATTEMPT_DIR": str(attempt_dir)},
            clear=False,
        ):
            with patch(
                "scripts.run_unified_validation._split_paths",
                side_effect=FileNotFoundError("injected data read failure"),
            ), self.assertRaises(FileNotFoundError):
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
                        str(state["data_root"]),
                        "--controller-state-dir",
                        str(self.state_dir),
                        "--repo-root",
                        str(self.repo_root),
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
        self._advance_unit_launch()

        second_path = self.root / "capability-2.json"
        second = authorize_next(
            state_dir=self.state_dir,
            repo_root=self.repo_root,
            output_path=second_path,
        )

        self.assertEqual(second["dataset_id"], "ASSIST17")
        self.assertEqual(second["counter"], 2)
        state = json.loads((self.state_dir / "state.json").read_text())
        self.assertEqual(state["cursor"], 1)
        self.assertEqual(state["successes"], 1)
        self.assertEqual(len(list((self.state_dir / "proofs").glob("*.json"))), 1)
        with self.assertRaisesRegex(ValueError, "stale|registry|issued|active controller"):
            consume_split_capability(
                state_dir=self.state_dir,
                repo_root=self.repo_root,
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

        with self.assertRaisesRegex(ValueError, "output hash|summary.*hash"):
            self._advance_unit_launch()
        next_path = self.root / "must-not-issue.json"
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
            self._advance_unit_launch()

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
            self._advance_unit_launch()

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
            self._advance_unit_launch()

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
            self._advance_unit_launch()

    def test_dataset_hash_mismatch_cannot_advance_progress(self) -> None:
        self.initialize()
        token_path, token = self.issue()
        records = self._prepare_pair_artifacts(
            token_path, token, dataset_id="ASSIST09"
        )
        data_path = Path(records["standard"]["consumption"]["data_paths"][0])
        data_path.write_text("tampered\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "dataset.*hash|validation data hash"):
            self._advance_unit_launch()

    def test_dataset_manifest_rejects_extra_non_object_record(self) -> None:
        self.initialize()
        token_path, token = self.issue()
        records = self._prepare_pair_artifacts(
            token_path, token, dataset_id="ASSIST09"
        )
        status = records["standard"]["status"]
        status["datasets"].append("not-a-fingerprint-record")
        Path(records["standard"]["status_path"]).write_text(
            json.dumps(status), encoding="utf-8"
        )

        with self.assertRaisesRegex(ValueError, "dataset manifest.*object"):
            self._advance_unit_launch()

    def test_summary_hash_and_metrics_come_from_one_snapshot(self) -> None:
        self.initialize()
        token_path, token = self.issue()
        records = self._prepare_pair_artifacts(
            token_path, token, dataset_id="ASSIST09"
        )
        summary_path = Path(records["holdout"]["summary_path"])
        original_sha256 = controller_module._sha256_file
        replaced = False

        def replace_after_hash(path: Path) -> str:
            nonlocal replaced
            digest = original_sha256(path)
            if path.resolve() == summary_path.resolve() and not replaced:
                forged = dict(records["holdout"]["summary"])
                forged["zero_auc"] = 0.99
                summary_path.write_text(json.dumps(forged), encoding="utf-8")
                replaced = True
            return digest

        with patch(
            "scripts.unified_validation_controller._sha256_file",
            side_effect=replace_after_hash,
        ):
            self._advance_unit_launch()

        proof = json.loads(next((self.state_dir / "proofs").glob("*.json")).read_text())
        self.assertEqual(
            proof["split_proofs"]["holdout"]["metrics"]["zero_auc"],
            0.61,
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
            self._advance_unit_launch()

    def test_summary_requires_finite_mastery_training_evidence(self) -> None:
        self.initialize()
        token_path, token = self.issue()
        records = self._prepare_pair_artifacts(
            token_path, token, dataset_id="ASSIST09"
        )
        standard = records["standard"]
        summary_path = Path(standard["summary_path"])
        status_path = Path(standard["status_path"])
        cases = (
            ("mastery_shape", [], "mastery_shape"),
            ("mastery_shape", [2], "mastery_shape"),
            ("final_loss", -0.1, "final_loss"),
            ("final_loss", float("inf"), "final_loss"),
            ("mastery_loss_weight", 0.0, "mastery_loss_weight"),
            ("mastery_loss_weight", -0.1, "mastery_loss_weight"),
            ("mastery_loss_weight", float("inf"), "mastery_loss_weight"),
        )
        for field, value, expected in cases:
            with self.subTest(field=field):
                summary = dict(standard["summary"])
                summary["numerical_recipe"] = dict(summary["numerical_recipe"])
                if field == "mastery_loss_weight":
                    summary["numerical_recipe"][field] = value
                else:
                    summary[field] = value
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                status = dict(standard["status"])
                status["output_hashes"] = {
                    "validation-summary.json": {
                        "exists": True,
                        **self._fingerprint(summary_path),
                    }
                }
                status_path.write_text(json.dumps(status), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, expected):
                    self._advance_unit_launch()

    def test_summary_allows_unbounded_finite_loss_and_mastery_weight(self) -> None:
        self.initialize()
        token_path, token = self.issue()
        records = self._prepare_pair_artifacts(
            token_path, token, dataset_id="ASSIST09"
        )
        for split_id in ("standard", "holdout"):
            record = records[split_id]
            summary = dict(record["summary"])
            summary["final_loss"] = 1.08
            summary["numerical_recipe"] = {"mastery_loss_weight": 2.0}
            summary_path = Path(record["summary_path"])
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            status = dict(record["status"])
            status["output_hashes"] = {
                "validation-summary.json": {
                    "exists": True,
                    **self._fingerprint(summary_path),
                }
            }
            Path(record["status_path"]).write_text(
                json.dumps(status), encoding="utf-8"
            )

        result = self._advance_unit_launch()
        self.assertEqual(result["cursor"], 1)

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
        self._advance_unit_launch()
        second_path = self.root / "capability-2.json"
        second = authorize_next(
            state_dir=self.state_dir,
            repo_root=self.repo_root,
            output_path=second_path,
        )
        self.assertEqual(second["dataset_id"], "ASSIST17")
        self._prepare_pair_artifacts(
            second_path,
            second,
            dataset_id="ASSIST17",
            weighted_doa=0.61,
        )
        self._advance_unit_launch()

        third_path = self.root / "must-not-issue.json"
        with self.assertRaisesRegex(RuntimeError, "primary cohort is unreachable"):
            authorize_next(
                state_dir=self.state_dir,
                repo_root=self.repo_root,
                output_path=third_path,
            )
        self.assertFalse(third_path.exists())
        state = json.loads((self.state_dir / "state.json").read_text())
        self.assertEqual(state["cursor"], 2)
        self.assertEqual(state["successes"], 0)
        self.assertTrue(state["blocked"])

    def test_joint_gate_uses_nonregression_and_strict_improvement_boundaries(self) -> None:
        passing = {
            "standard_overall_auc": 0.0,
            "holdout_overall_auc": 0.0,
            "weighted_doa": 0.0,
            "zero_auc": 0.001,
            "ordinary_doa": 0.001,
        }
        self.assertTrue(controller_module._joint_gate_success(passing))
        for field in (
            "standard_overall_auc",
            "holdout_overall_auc",
            "weighted_doa",
        ):
            with self.subTest(field=field, boundary="regression"):
                deltas = dict(passing)
                deltas[field] = -1e-12
                self.assertFalse(controller_module._joint_gate_success(deltas))
        for field in ("zero_auc", "ordinary_doa"):
            with self.subTest(field=field, boundary="equality"):
                deltas = dict(passing)
                deltas[field] = 0.0
                self.assertFalse(controller_module._joint_gate_success(deltas))

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
            result = self._advance_unit_launch()
            if index < len(DATASET_IDS) - 1:
                result_path = self.root / f"controller-output-{index}.json"
                result = authorize_next(
                    state_dir=self.state_dir,
                    repo_root=self.repo_root,
                    output_path=result_path,
                )
                token_path = result_path
                token = result

        self.assertTrue(result["complete"])
        self.assertFalse(result["global_pass"])
        self.assertNotIn("capabilities", result)

    def test_final_global_gate_passes_with_three_successes_and_threshold(self) -> None:
        self.initialize()
        token_path, token = self.issue()
        for index, dataset_id in enumerate(DATASET_IDS):
            self._prepare_pair_artifacts(
                token_path,
                token,
                dataset_id=dataset_id,
                zero_auc=0.602 if index == 0 else 0.601,
                standard_overall_auc=0.69 if index == 3 else 0.71,
            )
            result = self._advance_unit_launch()
            if index < len(DATASET_IDS) - 1:
                token_path = self.root / f"positive-token-{index}.json"
                token = authorize_next(
                    state_dir=self.state_dir,
                    repo_root=self.repo_root,
                    output_path=token_path,
                )

        self.assertTrue(result["complete"])
        self.assertTrue(result["global_pass"])
        self.assertEqual(result["successes"], 3)
        self.assertTrue(result["zero_delta_threshold_seen"])

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
                    str(self.repo_root),
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
