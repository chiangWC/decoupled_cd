from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from models.unified_v2_spec import UnifiedArchitectureSpec
from scripts import stable_graph_validation_controller as controller
from scripts import run_unified_validation as outer_runner
from scripts.unified_dataset_audit import canonical_sha256


class StableGraphValidationControllerTests(unittest.TestCase):
    def dependencies(self, root: Path, runner, route: str = "a" * 40):
        return controller.CampaignDependencies.for_test(
            campaign_root=root,
            route_root=Path.cwd(),
            runner=runner,
            route_commit=route,
        )

    @staticmethod
    def validation_runner(offsets=None):
        offsets = offsets or {"a0v4": 0.0, "a2": 0.002}

        def runner(argv, attempt_dir):
            architecture = argv[argv.index("--architecture") + 1]
            dataset = argv[argv.index("--dataset") + 1]
            result = controller.fake_validation_result(architecture, dataset)
            for field in controller.VALIDATION_METRICS:
                result[field] = 0.5 + offsets[architecture]
            return result

        return runner

    def prepare_relative_gate(self, root: Path, runner=None):
        deps = self.dependencies(root, runner or self.validation_runner())
        for architecture in ("a0v4", "a2"):
            for dataset in controller.FROZEN_RECIPES:
                controller.execute(
                    ["run-validation", "--architecture", architecture, "--dataset", dataset],
                    dependencies=deps,
                )
            controller.execute(["replay", "--architecture", architecture], dependencies=deps)
            if architecture == "a0v4":
                controller.execute(["freeze-a0v4"], dependencies=deps)
        controller.execute(["relative-gate"], dependencies=deps)
        return deps

    @staticmethod
    def write_audit(path: Path):
        strongest = {}
        for dataset in controller.FROZEN_RECIPES:
            def row(split, metric):
                return {
                    "dataset_id": dataset,
                    "split": split,
                    "metric": metric,
                    "value": 0.4,
                    "model": "trusted-comparator",
                    "seed": 42,
                    "evaluation_input_role": "valid",
                }
            strongest[dataset] = {
                "standard": {"overall_auc": row("standard", "overall_auc")},
                "holdout": {
                    "overall_auc": row("holdout", "overall_auc"),
                    "zero_auc": row("holdout", "zero_auc"),
                },
            }
        audit = {"schema_version": 1, "strongest_comparators": strongest}
        audit["audit_sha256"] = canonical_sha256(audit)
        path.write_text(json.dumps(audit), encoding="utf-8")
        return audit

    def test_identity_recipes_and_fingerprints(self) -> None:
        self.assertEqual(controller.CAMPAIGN_ID, "unified-ergc-20260712")
        self.assertEqual(controller.FROZEN_RECIPES, {"MOOCRadar": 2, "ASSIST17": 1, "XES3G5M": 0})
        self.assertEqual(controller.architecture_fingerprint("a0v4"), UnifiedArchitectureSpec(completion="prior").fingerprint())
        self.assertEqual(controller.architecture_fingerprint("a2"), UnifiedArchitectureSpec(completion="evidence-relational-graph").fingerprint())
        self.assertFalse(hasattr(controller, "issue_attempt"))

    def test_cli_has_plan_shape_and_no_caller_proofs_or_deltas(self) -> None:
        parser = controller.build_parser()
        commands = next(action.choices for action in parser._actions if getattr(action, "choices", None))
        self.assertEqual(set(commands), {"smoke", "run-validation", "replay", "freeze-a0v4", "relative-gate", "external-gate", "run-test-once", "status"})
        for argv in (
            ["replay", "--architecture", "a2", "--proof", "/tmp/forged"],
            ["freeze-a0v4", "--proof", "/tmp/forged"],
            ["relative-gate", "--deltas", "/tmp/forged"],
            ["external-gate", "--deltas", "/tmp/forged"],
        ):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                parser.parse_args(argv)

    def test_internal_dependencies_invoke_runner_record_argv_and_smoke_once(self) -> None:
        calls = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / controller.CAMPAIGN_ID
            root.mkdir()

            def runner(argv, attempt_dir):
                calls.append((list(argv), attempt_dir))
                return {
                    "architecture_fingerprint": controller.architecture_fingerprint("a2"),
                    "gpu_uuid": "GPU-fake",
                    "peak_gpu_memory_gb": 0.25,
                    "mastery_shape": [3, 4],
                    "final_loss": 0.5,
                    "parameter_count": 12,
                }

            deps = self.dependencies(root, runner)
            controller.execute(["smoke", "--architecture", "a2"], dependencies=deps)
            self.assertEqual(len(calls), 1)
            argv, attempt_dir = calls[0]
            self.assertIn("scripts/run_unified_validation.py", " ".join(argv))
            self.assertIn("42", argv)
            proof = json.loads((attempt_dir / "proof.json").read_text())
            self.assertEqual(proof["runner_argv"], argv)
            self.assertEqual(proof["runner_result"]["gpu_uuid"], "GPU-fake")
            with self.assertRaisesRegex(ValueError, "smoke.*already"):
                controller.execute(["smoke", "--architecture", "a2"], dependencies=deps)

    def test_existing_outer_runner_accepts_separate_stable_commands(self) -> None:
        smoke = outer_runner.parse_args([
            "smoke", "--architecture", "a2", "--devices", "gpu",
            "--seed", "42", "--epochs", "1", "--output", "/tmp/smoke.json",
        ])
        self.assertEqual(smoke.architecture, "a2")
        validation = outer_runner.parse_args([
            "stable-validation", "--architecture", "a0v4", "--dataset", "ASSIST17",
            "--recipe-index", "1", "--seed", "42", "--split-seed", "2024",
            "--cohort-sha256", controller.FROZEN_COHORT_SHA256,
            "--architecture-fingerprint", controller.architecture_fingerprint("a0v4"),
            "--output", "/tmp/validation.json",
        ])
        self.assertEqual(validation.dataset, "ASSIST17")

    def test_concurrent_issuance_is_contiguous_and_unique(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / controller.CAMPAIGN_ID
            root.mkdir()
            deps = self.dependencies(root, self.validation_runner())
            errors = []

            def run(dataset):
                try:
                    controller.execute(["run-validation", "--architecture", "a2", "--dataset", dataset], dependencies=deps)
                except BaseException as error:
                    errors.append(error)

            threads = [threading.Thread(target=run, args=(dataset,)) for dataset in controller.FROZEN_RECIPES]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])
            ledger = json.loads((root / "issuance-ledger.json").read_text())
            self.assertEqual([entry["counter"] for entry in ledger["entries"]], [1, 2, 3])
            self.assertEqual(len({entry["nonce"] for entry in ledger["entries"]}), 3)

    def test_replay_rejects_gap_duplicate_nonce_stale_route_and_artifact_swap(self) -> None:
        mutations = ("counter", "nonce", "route", "artifact")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / controller.CAMPAIGN_ID
                root.mkdir()
                deps = self.dependencies(root, self.validation_runner())
                for dataset in controller.FROZEN_RECIPES:
                    controller.execute(["run-validation", "--architecture", "a2", "--dataset", dataset], dependencies=deps)
                ledger_path = root / "issuance-ledger.json"
                ledger = json.loads(ledger_path.read_text())
                if mutation == "counter":
                    ledger["entries"][1]["counter"] = 3
                    ledger_path.write_text(json.dumps(ledger))
                elif mutation == "nonce":
                    ledger["entries"][1]["nonce"] = ledger["entries"][0]["nonce"]
                    ledger_path.write_text(json.dumps(ledger))
                elif mutation == "route":
                    deps = self.dependencies(root, self.validation_runner(), "b" * 40)
                else:
                    (root / ledger["entries"][0]["attempt_dir"] / "runner-result.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "gap|duplicate|stale|mismatch"):
                    controller.execute(["replay", "--architecture", "a2"], dependencies=deps)

    def test_symlink_root_and_outside_artifact_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            real = base / "real"
            real.mkdir()
            link = base / controller.CAMPAIGN_ID
            link.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                self.dependencies(link, self.validation_runner())

    def test_status_rejects_unknown_or_invalid_decision_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / controller.CAMPAIGN_ID
            root.mkdir()
            deps = self.dependencies(root, self.validation_runner())
            controller.execute(
                ["run-validation", "--architecture", "a2", "--dataset", "ASSIST17"],
                dependencies=deps,
            )
            (root / "decisions" / "forged.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "unregistered decision"):
                controller.execute(["status"], dependencies=deps)

    def test_exact_float_typing_rejects_bool_int_and_nan(self) -> None:
        for value in (True, 1, float("nan")):
            metrics = {field: 0.5 for field in controller.VALIDATION_METRICS}
            metrics["zero_auc"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "exact finite float"):
                controller.validate_metrics(metrics)

    def test_relative_gate_is_derived_and_failed_gate_has_no_nonce(self) -> None:
        deltas = {
            "ASSIST17": {"standard": 0.0, "holdout": 0.0, "zero": 0.001},
            "MOOCRadar": {"standard": 0.0001, "holdout": 0.0001, "zero": 0.0001},
            "XES3G5M": {"standard": 0.0, "holdout": 0.0, "zero": -0.0001},
        }
        self.assertTrue(controller.relative_gate(deltas)["passed"])
        deltas["XES3G5M"]["holdout"] = -1e-12
        self.assertFalse(controller.relative_gate(deltas)["passed"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / controller.CAMPAIGN_ID
            root.mkdir()
            deps = self.prepare_relative_gate(root, self.validation_runner({"a0v4": 0.0, "a2": -0.001}))
            proof = json.loads((root / "decisions" / "relative-gate.json").read_text())
            self.assertFalse(proof["passed"])
            self.assertIsNone(proof["nonce"])

    def test_resealed_replay_and_relative_proofs_have_no_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / controller.CAMPAIGN_ID
            root.mkdir()
            deps = self.prepare_relative_gate(root)
            replay_path = root / "decisions" / "replay-a2.json"
            replay = json.loads(replay_path.read_text())
            replay["rows"]["ASSIST17"]["zero_auc"] = 0.99
            replay.pop("proof_sha256")
            replay["proof_sha256"] = canonical_sha256(replay)
            replay_path.write_text(json.dumps(replay))
            with self.assertRaisesRegex(ValueError, "forged|resealed"):
                controller.execute(["relative-gate"], dependencies=deps)

    def test_external_derives_strongest_rows_and_failed_test_consumes_nonce(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / controller.CAMPAIGN_ID
            root.mkdir()
            calls = []

            def runner(argv, attempt_dir):
                calls.append(list(argv))
                if "stable-test" in argv:
                    raise RuntimeError("forced runner failure")
                return self.validation_runner()(argv, attempt_dir)

            deps = self.prepare_relative_gate(root, runner)
            audit_path = Path(temporary) / "audit.json"
            audit = self.write_audit(audit_path)
            with mock.patch.object(controller, "FROZEN_COMPARATOR_AUDIT_SHA256", audit["audit_sha256"]):
                controller.execute(["external-gate", "--comparator-audit", str(audit_path)], dependencies=deps)
                external = json.loads((root / "decisions" / "external-gate.json").read_text())
                self.assertTrue(external["passed"])
                self.assertIn("row_sha256", external["comparator_rows"]["ASSIST17"]["standard"])
                with self.assertRaisesRegex(RuntimeError, "forced runner failure"):
                    controller.execute(["run-test-once", "--architecture", "a2"], dependencies=deps)
                consumed = json.loads((root / "decisions" / "test-once.json").read_text())
                self.assertEqual(consumed["state"], "failed")
                with self.assertRaisesRegex(ValueError, "already.*consumed"):
                    controller.execute(["run-test-once", "--architecture", "a2"], dependencies=deps)
            self.assertEqual(sum("stable-test" in argv for argv in calls), 1)

    def test_comparator_audit_requires_exact_canonical_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.json"
            path.write_text(json.dumps({"audit_sha256": "0" * 64, "strongest_comparators": {}}))
            with self.assertRaisesRegex(ValueError, "canonical SHA-256 mismatch"):
                controller.load_verified_comparator_audit(path)


if __name__ == "__main__":
    unittest.main()
