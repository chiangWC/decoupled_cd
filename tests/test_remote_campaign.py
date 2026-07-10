from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_remote_campaign.py"


class RemoteCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.repo = self.root / "repo"
        self.artifact_root = self.root / "artifacts"
        self.repo.mkdir()
        self.dataset = self.repo / "dataset.csv"
        self.dataset.write_text("stu_id,label\n1,1\n", encoding="utf-8")
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

        completed = self.run_runner(command, output_files=("result.txt",))

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
        self.assertIsNone(status["exit_code"])
        self.assertIsNotNone(status["ended_at_utc"])

    def test_dirty_git_tree_is_refused_before_attempt_creation(self) -> None:
        self.dataset.write_text("stu_id,label\n1,0\n", encoding="utf-8")

        completed = self.run_runner([sys.executable, "-c", "pass"], dry_run=True)

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("dirty", completed.stderr.lower())
        self.assertFalse(self.artifact_root.exists())


if __name__ == "__main__":
    unittest.main()
