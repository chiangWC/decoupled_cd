from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_remote_campaign.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("run_remote_campaign", RUNNER)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER_MODULE = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER_MODULE)


class RemoteCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.repo = self.root / "repo"
        self.artifact_root = self.root / "artifacts"
        self.repo.mkdir()
        self.dataset = self.repo / "dataset.csv"
        self.dataset.write_text("stu_id,label\n1,1\n", encoding="utf-8")
        self.architecture_manifest = self.root / "architecture-manifest.json"
        self.architecture_payload = {
            "inference": "prior",
            "composer": "mask",
            "decoder": "monotonic",
            "mastery_output": "student-concept",
            "version": 1,
            "modules": "m1-m4",
        }
        self.architecture_manifest.write_text(
            json.dumps(self.architecture_payload, sort_keys=True),
            encoding="utf-8",
        )
        canonical_manifest = json.dumps(
            self.architecture_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        self.architecture_fingerprint = hashlib.sha256(
            canonical_manifest.encode("utf-8")
        ).hexdigest()
        self.cohort = self.root / "cohort.json"
        cohort_payload = {"dataset_ids": ["ASSIST09", "ASSIST17", "MOOCRadar"]}
        canonical_cohort = json.dumps(
            cohort_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        self.cohort_sha256 = hashlib.sha256(
            canonical_cohort.encode("utf-8")
        ).hexdigest()
        cohort_payload["cohort_sha256"] = self.cohort_sha256
        self.cohort.write_text(
            json.dumps(cohort_payload, sort_keys=True),
            encoding="utf-8",
        )
        self.run_git("init", "-q")
        self.run_git("config", "user.name", "Campaign Test")
        self.run_git("config", "user.email", "campaign@example.invalid")
        self.run_git("add", "dataset.csv")
        self.run_git("commit", "-qm", "fixture")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def run_git(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(self.repo), *args],
            check=True,
            capture_output=True,
            text=True,
        )

    def run_runner(
        self,
        command: list[str],
        *,
        dry_run: bool = False,
        output_files: tuple[str, ...] = (),
        vendor_commits: tuple[str, ...] = (),
        architecture_manifest: Path | None = None,
        cohort: Path | None = None,
        summary_output: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        argv = [
            sys.executable,
            str(RUNNER),
            "--artifact-root",
            str(self.artifact_root),
            "--repo-root",
            str(self.repo),
            "--cwd",
            str(self.repo),
            "--dataset-file",
            str(self.dataset),
            "--capture-env",
            "CAMPAIGN_TEST_ENV",
        ]
        for path in output_files:
            argv.extend(["--output-file", path])
        for vendor_commit in vendor_commits:
            argv.extend(["--vendor-commit", vendor_commit])
        if architecture_manifest is not None:
            argv.extend(["--architecture-manifest", str(architecture_manifest)])
        if cohort is not None:
            argv.extend(["--cohort", str(cohort)])
        if summary_output is not None:
            argv.extend(["--summary-output", summary_output])
        if dry_run:
            argv.append("--dry-run")
        argv.extend(["--", *command])
        env = os.environ.copy()
        env["CAMPAIGN_TEST_ENV"] = "captured"
        return subprocess.run(argv, capture_output=True, text=True, env=env, check=False)

    def load_status(self, attempt: str) -> dict[str, object]:
        return json.loads(
            (self.artifact_root / attempt / "status.json").read_text(encoding="utf-8")
        )

    def test_attempt_numbers_increase_without_overwriting_prior_attempt(self) -> None:
        first = self.run_runner([sys.executable, "-c", "pass"], dry_run=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        first_status = (self.artifact_root / "attempt-001" / "status.json").read_bytes()

        second = self.run_runner([sys.executable, "-c", "pass"], dry_run=True)

        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertTrue((self.artifact_root / "attempt-002" / "status.json").is_file())
        self.assertEqual(
            (self.artifact_root / "attempt-001" / "status.json").read_bytes(),
            first_status,
        )

    def test_terminal_status_is_atomic_and_captures_reproducibility_metadata(self) -> None:
        command = [
            sys.executable,
            "-c",
            (
                "import os; from pathlib import Path; "
                "Path(os.environ['CAMPAIGN_ATTEMPT_DIR'], 'result.txt').write_text('ok\\n')"
            ),
        ]

        vendor_head = self.run_git("rev-parse", "HEAD").stdout.strip()
        completed = self.run_runner(
            command,
            output_files=("result.txt",),
            vendor_commits=(f"orcdf={vendor_head}", f"svgcd={vendor_head}"),
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        attempt_dir = self.artifact_root / "attempt-001"
        status = self.load_status("attempt-001")
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["exit_code"], 0)
        self.assertIsNotNone(status["started_at_utc"])
        self.assertIsNotNone(status["ended_at_utc"])
        self.assertEqual(status["parameters"], {
            "seed": 42,
            "doa_seed": 42,
            "min_responses": 3,
            "split_seed": 2024,
        })
        self.assertEqual(status["invocation"]["command"], command)
        self.assertEqual(status["code"]["route_commit"], status["git"]["head"])
        self.assertEqual(
            status["code"]["vendor_commits"],
            {"orcdf": vendor_head, "svgcd": vendor_head},
        )
        self.assertEqual(status["invocation"]["cwd"], str(self.repo.resolve()))
        self.assertEqual(
            status["invocation"]["environment"]["CAMPAIGN_TEST_ENV"],
            "captured",
        )
        self.assertTrue(status["git"]["clean"])
        self.assertEqual(len(status["git"]["head"]), 40)
        self.assertEqual(
            status["datasets"][0]["sha256"],
            hashlib.sha256(self.dataset.read_bytes()).hexdigest(),
        )
        self.assertIn("python", status["runtime"])
        self.assertIn("torch", status["runtime"])
        self.assertIn("cuda", status["runtime"])
        self.assertIn("gpu", status["runtime"])
        self.assertIn("gpu_peak_memory", status["runtime"])
        self.assertIn("available", status["runtime"]["gpu_peak_memory"])
        self.assertIn("devices", status["runtime"]["gpu_peak_memory"])
        self.assertIn("command.log", status["output_hashes"])
        self.assertIn("result.txt", status["output_hashes"])
        self.assertFalse(any(attempt_dir.glob(".status.json.*")))

    def test_dry_run_records_attempt_without_executing_command(self) -> None:
        marker = self.root / "must-not-exist"
        command = [
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
        ]

        completed = self.run_runner(command, dry_run=True)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertFalse(marker.exists())
        status = self.load_status("attempt-001")
        self.assertEqual(status["status"], "dry_run")
        self.assertEqual(status["code"]["vendor_commits"], {})
        self.assertIsNone(status["exit_code"])
        self.assertIsNotNone(status["ended_at_utc"])

    def test_architecture_manifest_and_cohort_are_bound_to_attempt(self) -> None:
        command = [
            sys.executable,
            "-c",
            (
                "import json, os; from pathlib import Path; "
                "Path(os.environ['CAMPAIGN_ATTEMPT_DIR'], 'canonical.json').write_text("
                f"json.dumps({{'architecture_fingerprint': {self.architecture_fingerprint!r}}}))"
            ),
        ]

        completed = self.run_runner(
            command,
            output_files=("canonical.json",),
            architecture_manifest=self.architecture_manifest,
            cohort=self.cohort,
            summary_output="canonical.json",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        status = self.load_status("attempt-001")
        immutable = status["immutable_inputs"]
        self.assertEqual(
            immutable["architecture_manifest"]["sha256"],
            hashlib.sha256(self.architecture_manifest.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            immutable["architecture_manifest"]["architecture_fingerprint"],
            self.architecture_fingerprint,
        )
        self.assertEqual(
            immutable["cohort"]["sha256"],
            hashlib.sha256(self.cohort.read_bytes()).hexdigest(),
        )
        self.assertEqual(immutable["cohort"]["cohort_sha256"], self.cohort_sha256)
        self.assertEqual(
            status["invocation"]["environment"]["CAMPAIGN_ARCHITECTURE_FINGERPRINT"],
            self.architecture_fingerprint,
        )
        self.assertEqual(
            status["invocation"]["environment"]["CAMPAIGN_COHORT_SHA256"],
            self.cohort_sha256,
        )

    def test_bound_inputs_must_be_provided_together_before_attempt(self) -> None:
        completed = self.run_runner(
            [sys.executable, "-c", "pass"],
            dry_run=True,
            architecture_manifest=self.architecture_manifest,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("together", completed.stderr.lower())
        self.assertFalse(self.artifact_root.exists())

    def test_bound_campaign_requires_canonical_summary_before_attempt(self) -> None:
        completed = self.run_runner(
            [sys.executable, "-c", "pass"],
            dry_run=True,
            architecture_manifest=self.architecture_manifest,
            cohort=self.cohort,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("summary-output", completed.stderr.lower())
        self.assertFalse(self.artifact_root.exists())

    def test_canonical_summary_must_be_a_declared_output_before_attempt(self) -> None:
        completed = self.run_runner(
            [sys.executable, "-c", "pass"],
            dry_run=True,
            output_files=("different.json",),
            architecture_manifest=self.architecture_manifest,
            cohort=self.cohort,
            summary_output="canonical.json",
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("declared", completed.stderr.lower())
        self.assertFalse(self.artifact_root.exists())

    def test_missing_canonical_summary_is_a_terminal_failed_attempt(self) -> None:
        completed = self.run_runner(
            [sys.executable, "-c", "pass"],
            output_files=("canonical.json",),
            architecture_manifest=self.architecture_manifest,
            cohort=self.cohort,
            summary_output="canonical.json",
        )

        self.assertNotEqual(completed.returncode, 0)
        status = self.load_status("attempt-001")
        self.assertEqual(status["status"], "failed")
        self.assertIn("canonical.json", status["error"])
        self.assertIn("does not exist", status["error"])

    def test_non_json_canonical_summary_is_a_terminal_failed_attempt(self) -> None:
        command = [
            sys.executable,
            "-c",
            (
                "import os; from pathlib import Path; "
                "Path(os.environ['CAMPAIGN_ATTEMPT_DIR'], "
                "'canonical.artifact').write_text('not-json')"
            ),
        ]
        completed = self.run_runner(
            command,
            output_files=("canonical.artifact",),
            architecture_manifest=self.architecture_manifest,
            cohort=self.cohort,
            summary_output="canonical.artifact",
        )

        self.assertNotEqual(completed.returncode, 0)
        status = self.load_status("attempt-001")
        self.assertEqual(status["status"], "failed")
        self.assertIn("json", status["error"].lower())

    def test_summary_fingerprint_mismatch_is_terminal_and_attempt_is_immutable(self) -> None:
        command = [
            sys.executable,
            "-c",
            (
                "import json, os; from pathlib import Path; "
                "Path(os.environ['CAMPAIGN_ATTEMPT_DIR'], 'result.json').write_text("
                f"json.dumps({{'architecture_fingerprint': {'f' * 64!r}}}))"
            ),
        ]
        first = self.run_runner(
            command,
            output_files=("result.json",),
            architecture_manifest=self.architecture_manifest,
            cohort=self.cohort,
            summary_output="result.json",
        )

        self.assertNotEqual(first.returncode, 0)
        first_status = self.load_status("attempt-001")
        self.assertEqual(first_status["status"], "failed")
        self.assertIn("fingerprint", first_status["error"].lower())
        original_bytes = (
            self.artifact_root / "attempt-001" / "status.json"
        ).read_bytes()

        second = self.run_runner(
            [sys.executable, "-c", "pass"],
            dry_run=True,
            output_files=("result.json",),
            architecture_manifest=self.architecture_manifest,
            cohort=self.cohort,
            summary_output="result.json",
        )
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertTrue((self.artifact_root / "attempt-002" / "status.json").is_file())
        self.assertEqual(
            (self.artifact_root / "attempt-001" / "status.json").read_bytes(),
            original_bytes,
        )

    def test_duplicate_vendor_names_are_rejected_before_attempt_creation(self) -> None:
        vendor_head = self.run_git("rev-parse", "HEAD").stdout.strip()
        completed = self.run_runner(
            [sys.executable, "-c", "pass"],
            dry_run=True,
            vendor_commits=(f"orcdf={vendor_head}", f"orcdf={vendor_head}"),
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("duplicate", completed.stderr.lower())
        self.assertFalse(self.artifact_root.exists())

    def test_malformed_vendor_commit_is_rejected_before_attempt_creation(self) -> None:
        completed = self.run_runner(
            [sys.executable, "-c", "pass"],
            dry_run=True,
            vendor_commits=("orcdf=not-a-commit",),
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("vendor", completed.stderr.lower())
        self.assertFalse(self.artifact_root.exists())

    def test_vendor_commit_must_exist_in_route_history(self) -> None:
        completed = self.run_runner(
            [sys.executable, "-c", "pass"],
            dry_run=True,
            vendor_commits=("orcdf=" + "a" * 40,),
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("vendor", completed.stderr.lower())
        self.assertFalse(self.artifact_root.exists())

    def test_cwd_must_belong_to_verified_route_checkout(self) -> None:
        other_repo = self.root / "other"
        other_repo.mkdir()
        subprocess.run(["git", "-C", str(other_repo), "init", "-q"], check=True)
        subprocess.run(
            ["git", "-C", str(other_repo), "config", "user.name", "Other"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(other_repo), "config", "user.email", "other@example.invalid"],
            check=True,
        )
        (other_repo / "tracked.txt").write_text("other\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(other_repo), "add", "tracked.txt"], check=True)
        subprocess.run(["git", "-C", str(other_repo), "commit", "-qm", "other"], check=True)
        argv = [
            sys.executable,
            str(RUNNER),
            "--artifact-root",
            str(self.artifact_root),
            "--repo-root",
            str(self.repo),
            "--cwd",
            str(other_repo),
            "--dataset-file",
            str(self.dataset),
            "--dry-run",
            "--",
            sys.executable,
            "-c",
            "pass",
        ]

        completed = subprocess.run(argv, capture_output=True, text=True, check=False)

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("checkout", completed.stderr.lower())
        self.assertFalse(self.artifact_root.exists())

    def test_concurrent_runners_allocate_distinct_attempts(self) -> None:
        command = [sys.executable, "-c", "pass"]
        argv = [
            sys.executable,
            str(RUNNER),
            "--artifact-root",
            str(self.artifact_root),
            "--repo-root",
            str(self.repo),
            "--cwd",
            str(self.repo),
            "--dataset-file",
            str(self.dataset),
            "--dry-run",
            "--",
            *command,
        ]
        child_env = os.environ.copy()
        child_env["MKL_SERVICE_FORCE_INTEL"] = "1"
        processes = [
            subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=child_env,
            )
            for _ in range(2)
        ]
        results = [
            process.communicate(timeout=30) + (process.returncode,)
            for process in processes
        ]

        self.assertTrue(all(code == 0 for _, _, code in results), results)
        attempts = sorted(path.name for path in self.artifact_root.glob("attempt-*"))
        self.assertEqual(attempts, ["attempt-001", "attempt-002"])
        self.assertTrue(
            all((self.artifact_root / name / "status.json").is_file() for name in attempts)
        )

    def test_gpu_sampler_sums_only_child_process_tree_by_uuid(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="100, GPU-a, 128\n101, GPU-a, 256\n999, GPU-b, 512\n",
            stderr="",
        )
        with (
            mock.patch.object(RUNNER_MODULE, "descendant_pids", return_value={100, 101}),
            mock.patch.object(RUNNER_MODULE.subprocess, "run", return_value=completed),
        ):
            sample = RUNNER_MODULE.sample_gpu_process_memory(100)

        self.assertTrue(sample["available"])
        self.assertEqual(sample["devices"], {"GPU-a": 384})

    def test_gpu_sampler_degrades_when_nvidia_smi_is_unavailable(self) -> None:
        with mock.patch.object(
            RUNNER_MODULE.subprocess,
            "run",
            side_effect=FileNotFoundError("nvidia-smi"),
        ):
            sample = RUNNER_MODULE.sample_gpu_process_memory(100)

        self.assertFalse(sample["available"])
        self.assertEqual(sample["devices"], {})
        self.assertIn("FileNotFoundError", sample["error"])

    def test_gpu_peak_record_keeps_maximum_across_samples(self) -> None:
        record = RUNNER_MODULE.empty_gpu_peak_record()
        RUNNER_MODULE.update_gpu_peak_record(
            record, {"available": True, "devices": {"GPU-a": 128}}
        )
        RUNNER_MODULE.update_gpu_peak_record(
            record, {"available": True, "devices": {"GPU-a": 512}}
        )
        RUNNER_MODULE.update_gpu_peak_record(
            record, {"available": True, "devices": {"GPU-a": 256}}
        )

        self.assertEqual(record["successful_samples"], 3)
        self.assertEqual(record["devices"]["GPU-a"]["peak_used_memory_mib"], 512)

    def test_sampler_exception_is_recorded_without_abandoning_child(self) -> None:
        with (
            open(os.devnull, "w", encoding="utf-8") as sink,
            mock.patch.object(
                RUNNER_MODULE,
                "sample_gpu_process_memory",
                side_effect=RuntimeError("sampler exploded"),
            ),
        ):
            exit_code, peak = RUNNER_MODULE.run_with_gpu_sampling(
                [sys.executable, "-c", "raise SystemExit(7)"],
                cwd=self.repo,
                env=os.environ.copy(),
                stdout=sink,
            )

        self.assertEqual(exit_code, 7)
        self.assertIn("sampler exploded", peak["last_error"])

    def test_keyboard_interrupt_terminates_and_reaps_child_process(self) -> None:
        pid_path = self.root / "child.pid"

        def interrupt_after_child_starts(_root_pid: int) -> dict[str, object]:
            deadline = time.monotonic() + 5
            while not pid_path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            raise KeyboardInterrupt

        command = [
            sys.executable,
            "-c",
            (
                "import os, time; from pathlib import Path; "
                f"Path({str(pid_path)!r}).write_text(str(os.getpid())); "
                "time.sleep(30)"
            ),
        ]
        with (
            open(os.devnull, "w", encoding="utf-8") as sink,
            mock.patch.object(
                RUNNER_MODULE,
                "sample_gpu_process_memory",
                side_effect=interrupt_after_child_starts,
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            RUNNER_MODULE.run_with_gpu_sampling(
                command,
                cwd=self.repo,
                env=os.environ.copy(),
                stdout=sink,
            )

        child_pid = int(pid_path.read_text(encoding="utf-8"))
        with self.assertRaises(ProcessLookupError):
            os.kill(child_pid, 0)

    def test_cleanup_kills_descendant_that_ignores_sigterm_after_leader_exits(self) -> None:
        grandchild_pid_path = self.root / "grandchild.pid"
        grandchild_code = (
            "import os, signal, time; from pathlib import Path; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"Path({str(grandchild_pid_path)!r}).write_text(str(os.getpid())); "
            "time.sleep(30)"
        )
        parent_code = (
            "import subprocess, sys, time; "
            f"subprocess.Popen([sys.executable, '-c', {grandchild_code!r}]); "
            "time.sleep(30)"
        )

        def interrupt_after_grandchild_starts(_root_pid: int) -> dict[str, object]:
            deadline = time.monotonic() + 5
            while not grandchild_pid_path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            raise KeyboardInterrupt

        with (
            open(os.devnull, "w", encoding="utf-8") as sink,
            mock.patch.object(
                RUNNER_MODULE,
                "sample_gpu_process_memory",
                side_effect=interrupt_after_grandchild_starts,
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            RUNNER_MODULE.run_with_gpu_sampling(
                [sys.executable, "-c", parent_code],
                cwd=self.repo,
                env=os.environ.copy(),
                stdout=sink,
            )

        grandchild_pid = int(grandchild_pid_path.read_text(encoding="utf-8"))

        def process_is_active() -> bool:
            stat_path = Path(f"/proc/{grandchild_pid}/stat")
            try:
                state = stat_path.read_text(encoding="utf-8").split()[2]
            except (FileNotFoundError, IndexError):
                return False
            return state != "Z"

        try:
            deadline = time.monotonic() + 2
            while process_is_active() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(process_is_active())
        finally:
            if process_is_active():
                os.kill(grandchild_pid, signal.SIGKILL)

    def test_nonzero_child_exit_is_written_as_terminal_failure(self) -> None:
        completed = self.run_runner([sys.executable, "-c", "raise SystemExit(7)"])

        self.assertEqual(completed.returncode, 7, completed.stderr)
        status = self.load_status("attempt-001")
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["exit_code"], 7)
        self.assertIsNotNone(status["ended_at_utc"])

    def test_output_hash_error_is_recorded_in_terminal_status(self) -> None:
        argv = [
            "--artifact-root",
            str(self.artifact_root),
            "--repo-root",
            str(self.repo),
            "--cwd",
            str(self.repo),
            "--dataset-file",
            str(self.dataset),
            "--output-file",
            "result.txt",
            "--",
            sys.executable,
            "-c",
            "pass",
        ]
        args = RUNNER_MODULE.parse_args(argv)
        with (
            mock.patch.object(RUNNER_MODULE, "collect_runtime_metadata", return_value={}),
            mock.patch.object(
                RUNNER_MODULE,
                "sample_gpu_process_memory",
                return_value={"available": False, "devices": {}, "error": "cpu"},
            ),
            mock.patch.object(
                RUNNER_MODULE,
                "fingerprint_output",
                side_effect=OSError("output disappeared"),
            ),
        ):
            exit_code = RUNNER_MODULE.execute(args, ["runner", *argv])

        self.assertEqual(exit_code, 0)
        status = self.load_status("attempt-001")
        self.assertEqual(status["status"], "completed")
        self.assertIsNotNone(status["ended_at_utc"])
        self.assertIn("output disappeared", status["output_hashes"]["result.txt"]["error"])

    def test_dirty_git_tree_is_refused_before_attempt_creation(self) -> None:
        self.dataset.write_text("stu_id,label\n1,0\n", encoding="utf-8")

        completed = self.run_runner([sys.executable, "-c", "pass"], dry_run=True)

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("dirty", completed.stderr.lower())
        self.assertFalse(self.artifact_root.exists())


if __name__ == "__main__":
    unittest.main()
