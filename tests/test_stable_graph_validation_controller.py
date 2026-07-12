from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest import mock

from models.unified_v2_spec import UnifiedArchitectureSpec
from scripts import stable_graph_validation_controller as controller
from scripts import run_unified_validation as outer_runner
from scripts.unified_dataset_audit import canonical_sha256


class StableGraphValidationControllerTests(unittest.TestCase):
    def dependencies(
        self,
        root: Path,
        runner,
        route: str = "a" * 40,
        source_policy: str = "aggregate-only",
    ):
        return controller.CampaignDependencies.for_test(
            campaign_root=root,
            route_root=Path.cwd(),
            runner=runner,
            route_commit=route,
            source_policy=source_policy,
        )

    def test_test_dependencies_cannot_alias_or_construct_production_authority(self) -> None:
        with self.assertRaisesRegex(ValueError, "production campaign root"):
            controller.CampaignDependencies.for_test(
                campaign_root=controller.DEFAULT_CAMPAIGN_ROOT,
                route_root=Path.cwd(),
                runner=self.validation_runner(),
                route_commit="a" * 40,
                source_policy="aggregate-only",
            )
        with self.assertRaisesRegex(ValueError, "internal authority"):
            controller.CampaignDependencies(
                controller.DEFAULT_CAMPAIGN_ROOT,
                Path.cwd(),
                self.validation_runner(),
                "a" * 40,
                True,
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

    def prepare_relative_gate(
        self, root: Path, runner=None, source_policy: str = "aggregate-only"
    ):
        deps = self.dependencies(
            root, runner or self.validation_runner(), source_policy=source_policy
        )
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
        self.assertEqual(controller.CAMPAIGN_ID, "unified-ergc-r4-20260712")
        self.assertEqual(controller.FROZEN_RECIPES, {"MOOCRadar": 2, "ASSIST17": 1, "XES3G5M": 0})
        self.assertEqual(controller.architecture_fingerprint("a0v4"), UnifiedArchitectureSpec(completion="prior").fingerprint())
        self.assertEqual(controller.architecture_fingerprint("a2"), UnifiedArchitectureSpec(completion="evidence-relational-graph").fingerprint())
        self.assertFalse(hasattr(controller, "issue_attempt"))

    def test_cli_has_plan_shape_and_no_caller_proofs_or_deltas(self) -> None:
        parser = controller.build_parser()
        commands = next(action.choices for action in parser._actions if getattr(action, "choices", None))
        self.assertEqual(set(commands), {"preflight", "smoke", "run-validation", "replay", "freeze-a0v4", "relative-gate", "external-gate", "run-test-once", "status"})
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

            deps = self.dependencies(
                root, runner, source_policy="validation-fixed-sources"
            )
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

    def test_preflight_source_failure_does_not_issue_durable_attempt(self) -> None:
        failures = (
            FileNotFoundError("frozen source missing"),
            ValueError("frozen source SHA-256 mismatch"),
        )
        for failure in failures:
            with self.subTest(failure=str(failure)), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / controller.CAMPAIGN_ID
                root.mkdir()
                deps = self.dependencies(root, self.validation_runner())

                with mock.patch.object(
                    controller,
                    "_preflight_attempt",
                    side_effect=failure,
                ):
                    with self.assertRaisesRegex(type(failure), str(failure)):
                        controller.execute(
                            [
                                "run-validation",
                                "--architecture",
                                "a0v4",
                                "--dataset",
                                "ASSIST17",
                            ],
                            dependencies=deps,
                        )

                self.assertFalse((root / "issuance-ledger.json").exists())
                self.assertEqual(list((root / "attempts").iterdir()), [])

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
        test_once = outer_runner.parse_args([
            "stable-test", "--architecture", "a2", "--seed", "42",
            "--split-seed", "2024", "--output", "/tmp/test.json",
        ])
        self.assertEqual(test_once.architecture, "a2")

    def test_production_preflight_resolves_all_frozen_validation_sources(self) -> None:
        self.assertTrue(
            hasattr(outer_runner, "preflight_stable_validation"),
            "stable runner must expose immutable validation preflight",
        )
        for dataset in controller.FROZEN_RECIPES:
            with self.subTest(dataset=dataset):
                records = outer_runner.preflight_stable_validation(
                    architecture="a0v4",
                    dataset_id=dataset,
                    recipe_index=controller.FROZEN_RECIPES[dataset],
                )
                self.assertEqual(
                    {record["split_id"] for record in records},
                    {"standard", "holdout"},
                )
                self.assertTrue(
                    all(record["files"] for record in records)
                )

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

    def test_forged_ledger_artifact_and_resealed_attempt_is_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / controller.CAMPAIGN_ID
            root.mkdir()
            deps = self.dependencies(root, self.validation_runner())
            for dataset in controller.FROZEN_RECIPES:
                controller.execute(
                    ["run-validation", "--architecture", "a2", "--dataset", dataset],
                    dependencies=deps,
                )
            ledger_path = root / "issuance-ledger.json"
            ledger = json.loads(ledger_path.read_text())
            attempt = root / ledger["entries"][0]["attempt_dir"]
            result_path = attempt / "runner-result.json"
            result = json.loads(result_path.read_text())
            result["forged_metric"] = 0.9
            result_path.write_text(json.dumps(result))
            _, artifact = deps.store.read_regular(
                "attempts/attempt-001/runner-result.json", label="attack artifact"
            )
            proof_path = attempt / "proof.json"
            proof = json.loads(proof_path.read_text())
            proof["runner_result"] = result
            proof["artifacts"] = [artifact]
            proof.pop("proof_sha256")
            proof["proof_sha256"] = canonical_sha256(proof)
            proof_path.write_text(json.dumps(proof))
            _, proof_record = deps.store.read_regular(
                "attempts/attempt-001/proof.json", label="attack proof"
            )
            ledger["entries"][0]["proof_file_sha256"] = proof_record["sha256"]
            ledger_path.write_text(json.dumps(ledger))
            with self.assertRaisesRegex(ValueError, "canonical|recomputed|immutable|field set"):
                controller.execute(["replay", "--architecture", "a2"], dependencies=deps)

    def test_metric_source_artifacts_are_bound_and_recomputed_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / controller.CAMPAIGN_ID
            root.mkdir()

            def runner(argv, attempt_dir):
                architecture = argv[argv.index("--architecture") + 1]
                dataset = argv[argv.index("--dataset") + 1]
                result = controller.fake_validation_result(architecture, dataset)
                for split in ("standard", "holdout"):
                    work = attempt_dir / "stable-validation-work" / split
                    work.mkdir(parents=True)
                    overall = 0.5
                    zero = 0.9 if split == "holdout" else 0.5
                    (work / "coverage-valid.json").write_text(json.dumps({
                        "slices": [
                            {"scope": "overall", "auc": overall},
                            {"scope": "bucket:zero", "auc": zero},
                        ]
                    }))
                    (work / "doa-valid.json").write_text(json.dumps({
                        "rows": [{"doa": 0.5, "doa_weighted": 0.5}]
                    }))
                    (work / "train-summary.json").write_text(json.dumps({
                        "architecture_manifest": controller.architecture_spec(architecture).manifest(),
                        "architecture_fingerprint": controller.architecture_fingerprint(architecture),
                    }))
                return result

            deps = self.dependencies(
                root, runner, source_policy="validation-fixed-sources"
            )
            with self.assertRaisesRegex(ValueError, "metric source|recomputed"):
                controller.execute(
                    ["run-validation", "--architecture", "a2", "--dataset", "ASSIST17"],
                    dependencies=deps,
                )

    def test_real_stable_commands_isolate_and_discover_auxiliary_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / controller.CAMPAIGN_ID
            attempt = root / "attempts" / "attempt-001"
            attempt.mkdir(parents=True)
            data_paths = [Path(temporary) / name for name in ("train.csv", "valid.csv", "Q.csv")]
            for path in data_paths:
                path.write_text("placeholder")

            @contextmanager
            def allocation():
                yield 0, [], None

            commands: list[list[str]] = []

            def run_checked(command, *, env):
                command = list(command)
                commands.append(command)
                output = Path(command[command.index("--output") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                if command[1].endswith("train.py"):
                    output.write_text(json.dumps({
                        "architecture_manifest": controller.architecture_spec("a2").manifest(),
                        "architecture_fingerprint": controller.architecture_fingerprint("a2"),
                        "max_cuda_memory_allocated_gb": 0.25,
                    }))
                    output.with_name(f"{output.stem}_best.pt").write_bytes(b"checkpoint")
                    output.with_name(f"{output.stem}_history.csv").write_text("loss\n")
                    logs = output.parent / "logs"
                    logs.mkdir()
                    (logs / "train.log").write_text("finished\n")
                elif command[1].endswith("evaluate_coverage_slice.py"):
                    output.write_text(json.dumps({
                        "slices": [
                            {"scope": "overall", "auc": 0.5},
                            {"scope": "bucket:zero", "auc": 0.5},
                        ]
                    }))
                    slice_csv = (
                        Path(command[command.index("--slice-csv") + 1])
                        if "--slice-csv" in command
                        else output.with_name(f"{output.stem}_slices.csv")
                    )
                    summary_csv = (
                        Path(command[command.index("--summary-csv") + 1])
                        if "--summary-csv" in command
                        else output.with_name(f"{output.stem}_summary.csv")
                    )
                    slice_csv.write_text("scope,auc\n")
                    summary_csv.write_text("overall_auc\n")
                else:
                    output.write_text(json.dumps({
                        "rows": [{"doa": 0.5, "doa_weighted": 0.5}]
                    }))
                    summary_csv = Path(command[command.index("--summary-csv") + 1])
                    summary_csv.write_text("doa,doa_weighted\n")

            args = mock.Mock(
                output=attempt / "runner-result.json",
                dataset="ASSIST17",
                recipe_index=controller.FROZEN_RECIPES["ASSIST17"],
                architecture="a2",
                architecture_fingerprint=controller.architecture_fingerprint("a2"),
                data_root=Path(temporary),
                cohort_sha256=controller.FROZEN_COHORT_SHA256,
            )
            with (
                mock.patch.object(outer_runner, "locked_gpu", allocation),
                mock.patch.object(
                    outer_runner,
                    "_split_paths",
                    return_value=(*data_paths, None),
                ),
                mock.patch.object(outer_runner, "_run_checked", run_checked),
                mock.patch.object(outer_runner, "_gpu_uuid", return_value="GPU-fake"),
            ):
                outer_runner._run_stable_validation(args)

            for split in ("standard", "holdout"):
                authoritative = attempt / "stable-validation-work" / split
                self.assertEqual(
                    {path.name for path in authoritative.iterdir()},
                    {"train-summary.json", "coverage-valid.json", "doa-valid.json"},
                )
                auxiliary = attempt / "stable-validation-work" / "aux" / split
                self.assertEqual(
                    {path.name for path in auxiliary.iterdir()},
                    {
                        "train-summary_best.pt",
                        "train-summary_history.csv",
                        "coverage-valid_slices.csv",
                        "coverage-valid_summary.csv",
                        "doa-valid_summary.csv",
                        "logs",
                    },
                )
            self.assertEqual(
                {
                    path.name
                    for path in (
                        attempt / "stable-validation-work" / "aux"
                    ).iterdir()
                },
                {"standard", "holdout", "artifact-manifest.json"},
            )
            self.assertTrue(any("--slice-csv" in command for command in commands))
            deps = self.dependencies(
                root,
                self.validation_runner(),
                source_policy="production-layout-fixture",
            )
            sources = controller._discover_source_artifacts(
                deps, "attempts/attempt-001", "validation"
            )
            auxiliary = controller._discover_aux_artifacts(
                deps, "attempts/attempt-001", "validation"
            )
            self.assertEqual(len(sources), 6)
            self.assertEqual(len(auxiliary), 1)
            aux_root = attempt / "stable-validation-work" / "aux"
            manifest_path = aux_root / "artifact-manifest.json"
            original_manifest = manifest_path.read_bytes()
            manifest = json.loads(original_manifest)
            manifest["artifacts"] = manifest["artifacts"][1:]
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "omission|mismatch"):
                controller._discover_aux_artifacts(
                    deps, "attempts/attempt-001", "validation"
                )
            manifest_path.write_bytes(original_manifest)
            bound_path = aux_root / manifest["artifacts"][0]["relative_path"]
            original_bound = bound_path.read_bytes()
            bound_path.write_bytes(original_bound + b"tamper")
            with self.assertRaisesRegex(ValueError, "omission|mismatch"):
                controller._discover_aux_artifacts(
                    deps, "attempts/attempt-001", "validation"
                )
            bound_path.write_bytes(original_bound)
            unexpected = (
                attempt
                / "stable-validation-work"
                / "standard"
                / "unexpected.json"
            )
            unexpected.write_text("{}")
            with self.assertRaisesRegex(ValueError, "missing or extra"):
                controller._discover_source_artifacts(
                    deps, "attempts/attempt-001", "validation"
                )
            unexpected.unlink()

            test_attempt = root / "attempts" / "test-once"
            test_attempt.mkdir()
            for dataset in controller.FROZEN_RECIPES:
                standard_dir = (
                    Path(temporary)
                    / outer_runner.DATASET_DIRECTORIES[dataset][0]
                )
                standard_dir.mkdir(parents=True, exist_ok=True)
                for name in ("train.csv", "valid.csv", "test.csv", "Q_matrix.csv"):
                    (standard_dir / name).write_text("placeholder")
            args = mock.Mock(
                output=test_attempt / "runner-result.json",
                architecture="a2",
                data_root=Path(temporary),
            )
            with (
                mock.patch.object(outer_runner, "locked_gpu", allocation),
                mock.patch.object(outer_runner, "_run_checked", run_checked),
                mock.patch.object(outer_runner, "_gpu_uuid", return_value="GPU-fake"),
            ):
                outer_runner._run_stable_test(args)
            for dataset in controller.FROZEN_RECIPES:
                authoritative = test_attempt / "stable-test-work" / dataset
                self.assertEqual(
                    {path.name for path in authoritative.iterdir()},
                    {"train-summary.json", "coverage-test.json", "doa-test.json"},
                )
                auxiliary = (
                    test_attempt / "stable-test-work" / "aux" / dataset
                )
                self.assertEqual(
                    {path.name for path in auxiliary.iterdir()},
                    {
                        "train-summary_best.pt",
                        "train-summary_history.csv",
                        "coverage-test_slices.csv",
                        "coverage-test_summary.csv",
                        "doa-test_summary.csv",
                        "logs",
                    },
                )
            self.assertEqual(
                {
                    path.name
                    for path in (test_attempt / "stable-test-work" / "aux").iterdir()
                },
                {*controller.FROZEN_RECIPES, "artifact-manifest.json"},
            )
            sources = controller._discover_source_artifacts(
                deps, "attempts/test-once", "test"
            )
            auxiliary = controller._discover_aux_artifacts(
                deps, "attempts/test-once", "test"
            )
            self.assertEqual(len(sources), 9)
            self.assertEqual(len(auxiliary), 1)

    def test_test_sources_reject_duplicate_required_coverage_scope(self) -> None:
        for duplicate_scope in ("overall", "bucket:zero"):
            with (
                self.subTest(scope=duplicate_scope),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary) / controller.CAMPAIGN_ID
                root.mkdir()
                deps = self.dependencies(
                    root,
                    self.validation_runner(),
                    source_policy="test-fixed-sources",
                )
                result = controller.fake_test_result()
                for dataset in controller.FROZEN_RECIPES:
                    work = (
                        root
                        / "attempts"
                        / "test-once"
                        / "stable-test-work"
                        / dataset
                    )
                    work.mkdir(parents=True)
                    slices = [
                        {"scope": "overall", "auc": 0.5},
                        {"scope": "bucket:zero", "auc": 0.5},
                    ]
                    if dataset == "ASSIST17":
                        slices.append({"scope": duplicate_scope, "auc": 0.5})
                    (work / "coverage-test.json").write_text(
                        json.dumps({"slices": slices})
                    )
                    (work / "doa-test.json").write_text(json.dumps({
                        "rows": [{"doa": 0.5, "doa_weighted": 0.5}]
                    }))
                    (work / "train-summary.json").write_text(json.dumps({
                        "architecture_manifest": controller.architecture_spec("a2").manifest(),
                        "architecture_fingerprint": controller.architecture_fingerprint("a2"),
                    }))
                artifacts = controller._discover_source_artifacts(
                    deps, "attempts/test-once", "test"
                )
                with self.assertRaisesRegex(ValueError, "coverage source"):
                    controller._validate_test_sources(deps, result, artifacts)

    def test_valid_forged_aggregate_cannot_omit_fixed_production_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / controller.CAMPAIGN_ID
            root.mkdir()

            def runner(argv, attempt_dir):
                architecture = argv[argv.index("--architecture") + 1]
                dataset = argv[argv.index("--dataset") + 1]
                result = controller.fake_validation_result(architecture, dataset)
                for split in ("standard", "holdout"):
                    work = attempt_dir / "stable-validation-work" / split
                    work.mkdir(parents=True)
                    (work / "coverage-valid.json").write_text(json.dumps({
                        "slices": [
                            {"scope": "overall", "auc": 0.5},
                            {"scope": "bucket:zero", "auc": 0.5},
                        ]
                    }))
                    (work / "doa-valid.json").write_text(json.dumps({
                        "rows": [{"doa": 0.5, "doa_weighted": 0.5}]
                    }))
                    (work / "train-summary.json").write_text(json.dumps({
                        "architecture_manifest": controller.architecture_spec(architecture).manifest(),
                        "architecture_fingerprint": controller.architecture_fingerprint(architecture),
                    }))
                return result

            deps = self.dependencies(
                root, runner, source_policy="validation-fixed-sources"
            )
            for dataset in controller.FROZEN_RECIPES:
                controller.execute(
                    ["run-validation", "--architecture", "a2", "--dataset", dataset],
                    dependencies=deps,
                )
            ledger_path = root / "issuance-ledger.json"
            ledger = json.loads(ledger_path.read_text())
            attempt = root / ledger["entries"][0]["attempt_dir"]
            extra = (
                attempt
                / "stable-validation-work"
                / "standard"
                / "unexpected.json"
            )
            extra.write_text("{}")
            with self.assertRaisesRegex(ValueError, "missing or extra"):
                controller.execute(["replay", "--architecture", "a2"], dependencies=deps)
            extra.unlink()
            for split in ("standard", "holdout"):
                work = attempt / "stable-validation-work" / split
                for path in work.iterdir():
                    path.unlink()
                work.rmdir()
            (attempt / "stable-validation-work").rmdir()
            result_path = attempt / "runner-result.json"
            forged = json.loads(result_path.read_text())
            forged["zero_auc"] = 0.6
            result_path.write_text(json.dumps(forged))
            _, aggregate = deps.store.read_regular(
                f"{ledger['entries'][0]['attempt_dir']}/runner-result.json",
                label="forged aggregate",
            )
            proof_path = attempt / "proof.json"
            proof = json.loads(proof_path.read_text())
            proof["runner_result"] = forged
            proof["artifacts"] = [aggregate]
            proof.pop("proof_sha256")
            proof["proof_sha256"] = canonical_sha256(proof)
            proof_path.write_text(json.dumps(proof))
            _, proof_record = deps.store.read_regular(
                f"{ledger['entries'][0]['attempt_dir']}/proof.json",
                label="forged proof",
            )
            ledger["entries"][0]["proof_file_sha256"] = proof_record["sha256"]
            ledger_path.write_text(json.dumps(ledger))
            with self.assertRaisesRegex(ValueError, "required.*source|missing.*source"):
                controller.execute(["replay", "--architecture", "a2"], dependencies=deps)

    def test_docs_only_route_progresses_but_code_change_stales_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            route = base / "route"
            route.mkdir()
            for directory in controller.IMPLEMENTATION_DIRECTORIES:
                (route / directory).mkdir()
            (route / "scripts" / "stable.py").write_text("VALUE = 1\n")
            (route / "docs").mkdir()
            subprocess.run(["git", "init", "-q"], cwd=route, check=True)
            subprocess.run(["git", "config", "user.email", "fixture@example.com"], cwd=route, check=True)
            subprocess.run(["git", "config", "user.name", "Fixture"], cwd=route, check=True)
            subprocess.run(["git", "add", "."], cwd=route, check=True)
            subprocess.run(["git", "commit", "-qm", "code"], cwd=route, check=True)
            root = base / controller.CAMPAIGN_ID
            root.mkdir()
            deps = controller.CampaignDependencies.for_test(
                campaign_root=root,
                route_root=route,
                runner=self.validation_runner(),
                source_policy="aggregate-only",
            )
            controller.execute(
                ["run-validation", "--architecture", "a0v4", "--dataset", "ASSIST17"],
                dependencies=deps,
            )
            (route / "docs" / "report.md").write_text("report\n")
            subprocess.run(["git", "add", "docs/report.md"], cwd=route, check=True)
            subprocess.run(["git", "commit", "-qm", "docs"], cwd=route, check=True)
            for dataset in controller.FROZEN_RECIPES:
                controller.execute(
                    ["run-validation", "--architecture", "a2", "--dataset", dataset],
                    dependencies=deps,
                )
            controller.execute(["replay", "--architecture", "a2"], dependencies=deps)
            (route / "scripts" / "stable.py").write_text("VALUE = 2\n")
            subprocess.run(["git", "add", "scripts/stable.py"], cwd=route, check=True)
            subprocess.run(["git", "commit", "-qm", "code drift"], cwd=route, check=True)
            with self.assertRaisesRegex(ValueError, "implementation.*stale|code tree"):
                controller.execute(["status"], dependencies=deps)

    def test_strict_test_schema_and_resealed_success_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / controller.CAMPAIGN_ID
            root.mkdir()

            def runner(argv, attempt_dir):
                if "stable-test" in argv:
                    result = controller.fake_test_result()
                    result["extra"] = True
                    return result
                return self.validation_runner()(argv, attempt_dir)

            deps = self.prepare_relative_gate(root, runner)
            audit_path = Path(temporary) / "audit.json"
            audit = self.write_audit(audit_path)
            with mock.patch.object(controller, "FROZEN_COMPARATOR_AUDIT_SHA256", audit["audit_sha256"]):
                controller.execute(["external-gate", "--comparator-audit", str(audit_path)], dependencies=deps)
                with self.assertRaisesRegex(RuntimeError, "field set|schema"):
                    controller.execute(["run-test-once", "--architecture", "a2"], dependencies=deps)
                path = root / "decisions" / "test-once.json"
                record = json.loads(path.read_text())
                record["state"] = "succeeded"
                record.pop("proof_sha256")
                record["proof_sha256"] = canonical_sha256(record)
                path.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, "test.*artifact|recomputed|succeeded"):
                    controller.execute(["status"], dependencies=deps)

    def test_successful_test_binds_strict_result_artifact_and_status_recomputes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / controller.CAMPAIGN_ID
            root.mkdir()

            def runner(argv, attempt_dir):
                if "stable-test" in argv:
                    result = controller.fake_test_result()
                    for dataset in controller.FROZEN_RECIPES:
                        work = attempt_dir / "stable-test-work" / dataset
                        work.mkdir(parents=True)
                        (work / "coverage-test.json").write_text(json.dumps({
                            "slices": [
                                {"scope": "overall", "auc": 0.5},
                                {"scope": "bucket:zero", "auc": 0.5},
                            ]
                        }))
                        (work / "doa-test.json").write_text(json.dumps({
                            "rows": [{"doa": 0.5, "doa_weighted": 0.5}]
                        }))
                        (work / "train-summary.json").write_text(json.dumps({
                            "architecture_manifest": controller.architecture_spec("a2").manifest(),
                            "architecture_fingerprint": controller.architecture_fingerprint("a2"),
                        }))
                    return result
                return self.validation_runner()(argv, attempt_dir)

            deps = self.prepare_relative_gate(
                root, runner, source_policy="test-fixed-sources"
            )
            audit_path = Path(temporary) / "audit.json"
            audit = self.write_audit(audit_path)
            with mock.patch.object(controller, "FROZEN_COMPARATOR_AUDIT_SHA256", audit["audit_sha256"]):
                controller.execute(["external-gate", "--comparator-audit", str(audit_path)], dependencies=deps)
                controller.execute(["run-test-once", "--architecture", "a2"], dependencies=deps)
                controller.execute(["status"], dependencies=deps)
                path = root / "decisions" / "test-once.json"
                record = json.loads(path.read_text())
                self.assertEqual(record["state"], "succeeded")
                self.assertEqual(record["cohort_sha256"], controller.FROZEN_COHORT_SHA256)
                self.assertEqual(record["recipes"], controller.FROZEN_RECIPES)
                self.assertEqual(
                    record["artifacts"][0]["path"],
                    "attempts/test-once/runner-result.json",
                )
                self.assertEqual(
                    set(record["runner_result"]["rows"]),
                    set(controller.FROZEN_RECIPES),
                )
                record["artifacts"] = [record["artifacts"][0]]
                record.pop("proof_sha256")
                record["proof_sha256"] = canonical_sha256(record)
                path.write_text(json.dumps(record))
                with self.assertRaisesRegex(
                    ValueError, "fixed discovered artifacts"
                ):
                    controller.execute(["status"], dependencies=deps)

    def test_test_result_rejects_missing_extra_and_non_float_metrics(self) -> None:
        cases = {}
        missing = controller.fake_test_result()
        missing.pop("recipes")
        cases["missing"] = missing
        extra = controller.fake_test_result()
        extra["extra"] = None
        cases["extra"] = extra
        for label, value in (("bool", True), ("int", 1), ("nan", float("nan"))):
            payload = controller.fake_test_result()
            payload["rows"]["ASSIST17"]["zero_auc"] = value
            cases[label] = payload
        for label, payload in cases.items():
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "field set|finite float|schema"):
                controller._validate_runner_result("test", "a2", None, payload)

    def test_every_post_consumption_failure_finalizes_failed(self) -> None:
        for stage in ("mkdir", "argv", "runner", "artifact", "finalize"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / controller.CAMPAIGN_ID
                root.mkdir()

                def hook(current):
                    if current == stage:
                        raise RuntimeError(f"forced {stage}")

                deps = self.prepare_relative_gate(root)

                def trusted_runner(argv, attempt_dir):
                    if "stable-test" in argv:
                        return controller.fake_test_result()
                    return self.validation_runner()(argv, attempt_dir)

                deps = controller.CampaignDependencies.for_test(
                    campaign_root=root,
                    route_root=Path.cwd(),
                    runner=trusted_runner,
                    route_commit="a" * 40,
                    failure_hook=hook,
                    source_policy="aggregate-only",
                )
                audit_path = Path(temporary) / "audit.json"
                audit = self.write_audit(audit_path)
                with mock.patch.object(controller, "FROZEN_COMPARATOR_AUDIT_SHA256", audit["audit_sha256"]):
                    controller.execute(["external-gate", "--comparator-audit", str(audit_path)], dependencies=deps)
                    with self.assertRaisesRegex(RuntimeError, f"forced {stage}"):
                        controller.execute(["run-test-once", "--architecture", "a2"], dependencies=deps)
                record = json.loads((root / "decisions" / "test-once.json").read_text())
                self.assertEqual(record["state"], "failed")

    def test_parent_swap_race_is_rejected_by_retained_root_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / controller.CAMPAIGN_ID
            root.mkdir()
            swapped = base / "swapped"

            def race(stage):
                if stage == "before-campaign-open":
                    root.rename(swapped)
                    root.mkdir()

            deps = controller.CampaignDependencies.for_test(
                campaign_root=root,
                route_root=Path.cwd(),
                runner=self.validation_runner(),
                route_commit="a" * 40,
                race_hook=race,
                source_policy="aggregate-only",
            )
            with self.assertRaisesRegex(ValueError, "root identity|parent swap"):
                controller.execute(
                    ["run-validation", "--architecture", "a2", "--dataset", "ASSIST17"],
                    dependencies=deps,
                )

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
