from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.unified_v2_spec import UnifiedArchitectureSpec
from scripts.unified_cohort import load_verified_cohort
from scripts.unified_validation_controller import (
    DATASET_DIRECTORIES,
    authorize_next,
    consume_split_capability,
    initialize_controller,
    run_registered_pair,
    LegacyArchitectureIdentity,
)


ELIGIBLE_DATASET_IDS = (
    "ASSIST09",
    "ASSIST17",
    "MOOCRadar",
    "XES3G5M",
)
ASSET_READY_WITHOUT_EXACT_ZERO = ("NIPS34",)
ARCHITECTURES = {
    "a0": ("prior", 0.0),
    "a1": ("lowrank", 1.0),
}
STABLE_ARCHITECTURES = {
    "a0v4": ("prior", 0.0),
    "a2": ("evidence-relational-graph", 1.0),
}
RUNNER_ARCHITECTURES = {**ARCHITECTURES, **STABLE_ARCHITECTURES}
STABLE_FROZEN_RECIPES = {"MOOCRadar": 2, "ASSIST17": 1, "XES3G5M": 0}


@dataclass(frozen=True)
class NumericalRecipe:
    training_mode: str
    concept_dim: int
    epochs: int
    learning_rate: float
    weight_decay: float
    patience: int
    student_batch_size: int | None = None
    mastery_loss_weight: float = 0.1


RECIPES = {
    "ASSIST09": (
        NumericalRecipe("full_batch", 64, 300, 1e-3, 0.0, 5, None),
    ),
    "ASSIST17": (
        NumericalRecipe("student_recompute_minibatch", 64, 40, 1e-3, 0.0, 5, 64),
        NumericalRecipe("student_recompute_minibatch", 64, 80, 2e-3, 0.0, 5, 128),
    ),
    "NIPS34": (
        NumericalRecipe("full_batch", 64, 300, 1e-3, 0.0, 5, None),
    ),
    "MOOCRadar": (
        NumericalRecipe("student_recompute_minibatch", 64, 30, 1e-3, 0.0, 5, 64),
        NumericalRecipe("student_recompute_minibatch", 128, 30, 1e-3, 0.0, 5, 64),
        NumericalRecipe("student_recompute_minibatch", 256, 30, 1e-3, 0.0, 5, 64),
    ),
    "XES3G5M": (
        NumericalRecipe("student_recompute_minibatch", 64, 30, 1e-3, 0.0, 5, 64),
        NumericalRecipe("full_batch", 64, 3000, 1e-3, 0.0, 50, None),
    ),
}


@dataclass(frozen=True)
class GpuSnapshot:
    index: int
    memory_used_mib: int
    memory_total_mib: int
    utilization_percent: int

    @property
    def memory_fraction(self) -> float:
        return self.memory_used_mib / max(self.memory_total_mib, 1)

    @property
    def idle(self) -> bool:
        return self.utilization_percent == 0 and self.memory_fraction <= 0.01


def architecture_spec(architecture: str) -> LegacyArchitectureIdentity | UnifiedArchitectureSpec:
    if architecture in STABLE_ARCHITECTURES:
        return UnifiedArchitectureSpec(completion=STABLE_ARCHITECTURES[architecture][0])
    try:
        completion, _ = ARCHITECTURES[architecture]
    except KeyError as error:
        raise ValueError(f"unknown unified architecture: {architecture}") from error
    return LegacyArchitectureIdentity(completion=completion)


def architecture_fingerprint(architecture: str) -> str:
    return architecture_spec(architecture).fingerprint()


def _validate_legacy_manifest(manifest: object, fingerprint: object) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("architecture_manifest must be a JSON object")
    if manifest.get("version") == 4:
        UnifiedArchitectureSpec.from_manifest(
            manifest, architecture_fingerprint=fingerprint
        )
        return
    architecture = str(manifest.get("completion"))
    if architecture not in {"prior", "lowrank"}:
        raise ValueError("invalid architecture_manifest: version 3 is required")
    expected = LegacyArchitectureIdentity(architecture)
    if manifest != expected.manifest() or fingerprint != expected.fingerprint():
        raise ValueError("architecture manifest/fingerprint mismatch")


def _unified_training_flags(
    spec: LegacyArchitectureIdentity | UnifiedArchitectureSpec,
    *,
    evidence_loss_weight: float,
    completion_rank: int = 32,
) -> list[str]:
    completion_loss_weight = evidence_loss_weight * {
        "prior": 0.0,
        "lowrank": 1.0,
        "evidence-relational-graph": 1.0,
    }[spec.completion]
    return [
        "--unified-completion",
        spec.completion,
        "--unified-completion-rank",
        str(completion_rank),
        "--unified-evidence-loss-weight",
        str(evidence_loss_weight),
        "--unified-completion-loss-weight",
        str(completion_loss_weight),
    ]


def parse_gpu_inventory(output: str) -> list[GpuSnapshot]:
    snapshots: list[GpuSnapshot] = []
    for line_number, line in enumerate(output.splitlines(), start=1):
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise ValueError(f"invalid nvidia-smi row {line_number}: {line!r}")
        try:
            snapshots.append(GpuSnapshot(*(int(field) for field in fields)))
        except ValueError as error:
            raise ValueError(
                f"invalid nvidia-smi integer at row {line_number}: {line!r}"
            ) from error
    if not snapshots:
        raise RuntimeError("nvidia-smi returned no GPUs")
    return snapshots


def select_gpu_index(snapshots: Sequence[GpuSnapshot]) -> int:
    eligible = [
        snapshot
        for snapshot in snapshots
        if snapshot.memory_used_mib == 0
        or snapshot.memory_used_mib * 2 < snapshot.memory_total_mib
    ]
    if not eligible:
        raise RuntimeError("no GPU is idle or under half memory")
    eligible.sort(key=lambda item: (item.memory_used_mib, item.index))
    return eligible[0].index


def query_gpu_inventory() -> tuple[str, list[GpuSnapshot]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout, parse_gpu_inventory(completed.stdout)


def _gpu_uuid(index: int) -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == str(index) and fields[1]:
            return fields[1]
    raise RuntimeError(f"nvidia-smi returned no UUID for physical GPU {index}")


@contextmanager
def locked_gpu() -> Iterator[tuple[int, list[GpuSnapshot], Path]]:
    _, snapshots = query_gpu_inventory()
    eligible = sorted(
        (
            snapshot
            for snapshot in snapshots
            if snapshot.memory_used_mib == 0
            or snapshot.memory_used_mib * 2 < snapshot.memory_total_mib
        ),
        key=lambda item: (item.memory_used_mib, item.index),
    )
    if not eligible:
        raise RuntimeError("no GPU is idle or under half memory")
    saw_ineligible_after_lock = False
    for candidate in eligible:
        gpu_index = candidate.index
        lock_path = Path(f"/tmp/unified-mastery-gpu-{gpu_index}.lock")
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            try:
                fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            _, refreshed = query_gpu_inventory()
            selected = next(
                (item for item in refreshed if item.index == gpu_index),
                None,
            )
            if selected is None or not (
                selected.memory_used_mib == 0
                or selected.memory_used_mib * 2 < selected.memory_total_mib
            ):
                saw_ineligible_after_lock = True
                fcntl.flock(lock_handle, fcntl.LOCK_UN)
                continue
            try:
                yield gpu_index, refreshed, lock_path
                return
            finally:
                fcntl.flock(lock_handle, fcntl.LOCK_UN)
    if saw_ineligible_after_lock and len(eligible) == 1:
        raise RuntimeError(
            f"selected GPU {eligible[0].index} became ineligible while acquiring lock"
        )
    raise RuntimeError("no unlocked eligible GPU")


def _split_paths(
    *,
    dataset_id: str,
    split_id: str,
    data_root: Path,
) -> tuple[Path, Path, Path, Path | None]:
    if dataset_id not in ELIGIBLE_DATASET_IDS:
        raise ValueError(f"dataset is not exact-zero eligible: {dataset_id}")
    if split_id not in {"standard", "holdout"}:
        raise ValueError(f"unknown validation split: {split_id}")
    standard_dir, holdout_dir = DATASET_DIRECTORIES[dataset_id]
    directory = data_root / (standard_dir if split_id == "standard" else holdout_dir)
    train_path = directory / "train.csv"
    valid_path = directory / "valid.csv"
    q_matrix_path = directory / "Q_matrix.csv"
    assignments = (
        directory / "student_concept_holdout_assignments.csv"
        if split_id == "holdout"
        else None
    )
    return train_path, valid_path, q_matrix_path, assignments


def build_train_command(
    *,
    dataset_id: str,
    split_id: str,
    architecture: str,
    data_root: Path,
    output: Path,
    device: str,
    recipe: NumericalRecipe | None = None,
) -> list[str]:
    train_path, valid_path, q_matrix_path, _ = _split_paths(
        dataset_id=dataset_id,
        split_id=split_id,
        data_root=data_root,
    )
    selected_recipe = recipe or RECIPES[dataset_id][0]
    spec = architecture_spec(architecture)
    command = [
        sys.executable,
        "scripts/train.py",
        "--model",
        "unified_v2",
        *_unified_training_flags(
            spec,
            evidence_loss_weight=selected_recipe.mastery_loss_weight,
        ),
        "--train-interactions",
        str(train_path),
        "--valid-interactions",
        str(valid_path),
        "--test-interactions",
        str(valid_path),
        "--q-matrix",
        str(q_matrix_path),
        "--concept-dim",
        str(selected_recipe.concept_dim),
        "--epochs",
        str(selected_recipe.epochs),
        "--learning-rate",
        str(selected_recipe.learning_rate),
        "--weight-decay",
        str(selected_recipe.weight_decay),
        "--early-stop-patience",
        str(selected_recipe.patience),
        "--training-mode",
        selected_recipe.training_mode,
        "--seed",
        "42",
        "--device",
        device,
        "--log-dir",
        str(output.parent / "logs"),
        "--output",
        str(output),
    ]
    if selected_recipe.student_batch_size is not None:
        command.extend(
            ["--student-batch-size", str(selected_recipe.student_batch_size)]
        )
    return command


def _finite_float(value: object, *, field: str) -> float:
    if type(value) not in {float, int} or not math.isfinite(float(value)):
        raise ValueError(f"{field} must be finite")
    return float(value)


def validate_smoke_summary(
    summary: Mapping[str, object],
    *,
    expected_fingerprint: str,
    require_gpu_peak: bool,
) -> None:
    if summary.get("architecture_fingerprint") != expected_fingerprint:
        raise ValueError("smoke architecture fingerprint mismatch")
    _validate_legacy_manifest(summary.get("architecture_manifest"), expected_fingerprint)
    shape = summary.get("mastery_shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(value) is not int or value <= 0 for value in shape)
    ):
        raise ValueError("smoke mastery must be nonempty")
    _finite_float(summary.get("final_loss"), field="final_loss")
    parameter_count = summary.get("parameter_count")
    if type(parameter_count) is not int or parameter_count < 0:
        raise ValueError("parameter_count must be a nonnegative integer")
    peak = summary.get("peak_gpu_memory_gb")
    if require_gpu_peak and _finite_float(peak, field="peak_gpu_memory_gb") <= 0.0:
        raise ValueError("GPU smoke must record positive peak memory")


def _validate_split_summary(
    summary: Mapping[str, object],
    *,
    dataset_id: str,
    split_id: str,
    fingerprint: str,
    cohort_sha256: str,
) -> None:
    if summary.get("dataset_id") != dataset_id:
        raise ValueError("validation dataset identity mismatch")
    if summary.get("split_id") != split_id:
        raise ValueError("validation split identity mismatch")
    if summary.get("architecture_fingerprint") != fingerprint:
        raise ValueError("validation architecture fingerprint mismatch")
    _validate_legacy_manifest(summary.get("architecture_manifest"), fingerprint)
    if summary.get("cohort_sha256") != cohort_sha256:
        raise ValueError("validation summary cohort SHA-256 mismatch")
    if summary.get("seed") != 42:
        raise ValueError("unified validation requires seed 42")
    validate_smoke_summary(
        summary,
        expected_fingerprint=fingerprint,
        require_gpu_peak=False,
    )
    for field in ("overall_auc", "zero_auc", "ordinary_doa", "weighted_doa"):
        value = _finite_float(summary.get(field), field=field)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field} must be in [0, 1]")


def assemble_candidate_rows(
    split_summaries: Sequence[Mapping[str, object]],
    *,
    cohort_dataset_ids: Sequence[str],
    cohort_sha256: str,
) -> list[dict[str, object]]:
    indexed: dict[tuple[str, str], Mapping[str, object]] = {}
    for summary in split_summaries:
        key = (str(summary.get("dataset_id")), str(summary.get("split_id")))
        if key in indexed:
            raise ValueError(f"duplicate validation split summary: {key}")
        indexed[key] = summary

    rows: list[dict[str, object]] = []
    fingerprints: set[str] = set()
    for dataset_id in cohort_dataset_ids:
        try:
            standard = indexed[(dataset_id, "standard")]
            holdout = indexed[(dataset_id, "holdout")]
        except KeyError as error:
            raise ValueError(
                f"missing validation split for frozen dataset: {dataset_id}"
            ) from error
        fingerprint = str(standard.get("architecture_fingerprint"))
        _validate_split_summary(
            standard,
            dataset_id=dataset_id,
            split_id="standard",
            fingerprint=fingerprint,
            cohort_sha256=cohort_sha256,
        )
        _validate_split_summary(
            holdout,
            dataset_id=dataset_id,
            split_id="holdout",
            fingerprint=fingerprint,
            cohort_sha256=cohort_sha256,
        )
        fingerprints.add(fingerprint)
        if (
            standard.get("recipe_index") != holdout.get("recipe_index")
            or standard.get("numerical_recipe") != holdout.get("numerical_recipe")
        ):
            raise ValueError("validation split summaries use different recipes")
        if standard.get("parameter_count") != holdout.get("parameter_count"):
            raise ValueError("validation split summaries use different parameter counts")
        rows.append(
            {
                "dataset_id": dataset_id,
                "cohort_sha256": cohort_sha256,
                "architecture_fingerprint": fingerprint,
                "standard_overall_auc": float(standard["overall_auc"]),
                "holdout_overall_auc": float(holdout["overall_auc"]),
                "zero_auc": float(holdout["zero_auc"]),
                "ordinary_doa": float(standard["ordinary_doa"]),
                "weighted_doa": float(standard["weighted_doa"]),
                "recipe_index": standard.get("recipe_index", 0),
                "numerical_recipe": standard.get("numerical_recipe"),
                "parameter_count": standard["parameter_count"],
            }
        )
    unexpected = set(indexed) - {
        (dataset_id, split_id)
        for dataset_id in cohort_dataset_ids
        for split_id in ("standard", "holdout")
    }
    if unexpected:
        raise ValueError(f"validation summaries include non-cohort rows: {unexpected}")
    if len(fingerprints) != 1:
        raise ValueError("candidate has mixed architecture fingerprints")
    return rows


def can_reach_primary_cohort(*, successes: int, remaining: int) -> bool:
    return successes + remaining >= 3


def _authorize(args: argparse.Namespace) -> None:
    authorize_next(
        state_dir=args.controller_state_dir,
        repo_root=args.repo_root,
        output_path=_output_path(args.output),
    )


def _run_pair(args: argparse.Namespace) -> None:
    run_registered_pair(
        state_dir=args.controller_state_dir,
        repo_root=args.repo_root,
    )


def _run_checked(command: Sequence[str], *, env: Mapping[str, str]) -> None:
    print(json.dumps({"command": list(command)}, ensure_ascii=False), flush=True)
    subprocess.run(list(command), check=True, env=dict(env))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")


def _output_path(raw_path: Path) -> Path:
    if raw_path.is_absolute():
        return raw_path
    attempt_dir = os.environ.get("CAMPAIGN_ATTEMPT_DIR")
    return (Path(attempt_dir) if attempt_dir else Path.cwd()) / raw_path


def _gpu_payload(snapshots: Sequence[GpuSnapshot]) -> list[dict[str, int]]:
    return [
        {
            "index": item.index,
            "memory_used_mib": item.memory_used_mib,
            "memory_total_mib": item.memory_total_mib,
            "utilization_percent": item.utilization_percent,
        }
        for item in snapshots
    ]


def _extract_split_metrics(
    *,
    coverage_path: Path,
    doa_path: Path,
) -> tuple[float, float, float, float]:
    coverage = _load_json(coverage_path)
    slices = coverage.get("slices")
    if not isinstance(slices, list):
        raise ValueError("coverage output has no slices")
    by_scope = {
        row.get("scope"): row
        for row in slices
        if isinstance(row, dict)
    }
    try:
        overall_auc = _finite_float(by_scope["overall"]["auc"], field="overall_auc")
        zero_auc = _finite_float(by_scope["bucket:zero"]["auc"], field="zero_auc")
    except KeyError as error:
        raise ValueError("coverage output lacks overall or exact-zero AUC") from error

    doa = _load_json(doa_path)
    rows = doa.get("rows")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError("DOA output must contain exactly one model row")
    ordinary_doa = _finite_float(rows[0].get("doa"), field="ordinary_doa")
    weighted_doa = _finite_float(rows[0].get("doa_weighted"), field="weighted_doa")
    return overall_auc, zero_auc, ordinary_doa, weighted_doa


def build_evaluation_commands(
    *,
    dataset_id: str,
    split_id: str,
    architecture: str,
    data_root: Path,
    train_summary_path: Path,
    coverage_path: Path,
    doa_path: Path,
    device: str,
) -> tuple[list[str], list[str]]:
    train_path, valid_path, q_matrix_path, assignments = _split_paths(
        dataset_id=dataset_id,
        split_id=split_id,
        data_root=data_root,
    )
    common_evaluation = [
        "--dataset-name",
        dataset_id,
        "--summary",
        str(train_summary_path),
        "--model-name",
        architecture,
        "--split",
        "valid",
        "--train-interactions",
        str(train_path),
        "--valid-interactions",
        str(valid_path),
        "--test-interactions",
        str(valid_path),
        "--q-matrix",
        str(q_matrix_path),
        "--device",
        device,
    ]
    coverage_command = [
        sys.executable,
        "scripts/evaluate_coverage_slice.py",
        *common_evaluation,
        "--output",
        str(coverage_path),
    ]
    doa_command = [
        sys.executable,
        "scripts/evaluate_doa.py",
        *common_evaluation,
        "--min-responses",
        "3",
        "--doa-seed",
        "42",
        "--output",
        str(doa_path),
    ]
    if assignments is not None:
        doa_command.extend(["--holdout-assignments", str(assignments)])
    return coverage_command, doa_command


def _run_split(args: argparse.Namespace) -> None:
    binding = consume_split_capability(
        state_dir=args.controller_state_dir,
        repo_root=args.repo_root,
        token_path=args.capability,
        dataset_id=args.dataset_id,
        split_id=args.split_id,
        architecture=args.architecture,
        data_root=args.data_root,
        raw_output=args.output,
        attempt_dir=_required_attempt_dir(),
    )
    cohort_hash = str(binding["cohort_sha256"])
    if args.architecture_fingerprint != binding["architecture_fingerprint"]:
        raise ValueError("command architecture fingerprint mismatch")
    if args.cohort_sha256 != cohort_hash:
        raise ValueError("command cohort SHA-256 mismatch")
    if args.recipe_index != binding["recipe_index"]:
        raise ValueError("command numerical recipe index mismatch")
    try:
        selected_recipe = RECIPES[args.dataset_id][args.recipe_index]
    except IndexError as error:
        raise ValueError("command numerical recipe index is out of range") from error
    frozen_recipe = binding.get("numerical_recipe")
    if isinstance(frozen_recipe, Mapping) and selected_recipe.__dict__ != frozen_recipe:
        raise ValueError("A1 numerical recipe differs from frozen A0 recipe")
    output_path = _output_path(args.output).resolve()
    work_dir = output_path.parent / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    train_summary_path = work_dir / "train-summary.json"
    coverage_path = work_dir / "coverage-valid.json"
    doa_path = work_dir / "doa-valid.json"
    train_path, valid_path, q_matrix_path, assignments = _split_paths(
        dataset_id=args.dataset_id,
        split_id=args.split_id,
        data_root=args.data_root,
    )
    required_paths = [train_path, valid_path, q_matrix_path]
    if assignments is not None:
        required_paths.append(assignments)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    base_env = dict(os.environ)
    if args.device == "cpu":
        allocation: Iterator[tuple[int | None, list[GpuSnapshot], Path | None]]

        @contextmanager
        def cpu_allocation() -> Iterator[
            tuple[int | None, list[GpuSnapshot], Path | None]
        ]:
            yield None, [], None

        allocation = cpu_allocation()
    else:
        allocation = locked_gpu()

    with allocation as (gpu_index, gpu_snapshots, lock_path):
        device = "cpu" if gpu_index is None else "cuda:0"
        child_env = dict(base_env)
        if gpu_index is not None:
            child_env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        command = build_train_command(
            dataset_id=args.dataset_id,
            split_id=args.split_id,
            architecture=args.architecture,
            data_root=args.data_root,
            output=train_summary_path,
            device=device,
            recipe=selected_recipe,
        )
        _run_checked(command, env=child_env)

        coverage_command, doa_command = build_evaluation_commands(
            dataset_id=args.dataset_id,
            split_id=args.split_id,
            architecture=args.architecture,
            data_root=args.data_root,
            train_summary_path=train_summary_path,
            coverage_path=coverage_path,
            doa_path=doa_path,
            device=device,
        )
        _run_checked(coverage_command, env=child_env)
        _run_checked(doa_command, env=child_env)

        train_summary = _load_json(train_summary_path)
        fingerprint = architecture_fingerprint(args.architecture)
        _validate_legacy_manifest(
            train_summary.get("architecture_manifest"),
            train_summary.get("architecture_fingerprint"),
        )
        if train_summary.get("architecture_fingerprint") != fingerprint:
            raise ValueError("training summary architecture fingerprint mismatch")
        final_loss = _finite_float(train_summary.get("final_loss"), field="final_loss")
        parameter_count = train_summary.get("parameter_count")
        if type(parameter_count) is not int or parameter_count < 0:
            raise ValueError("training summary parameter_count is invalid")
        peak = train_summary.get("max_cuda_memory_allocated_gb")
        if gpu_index is not None:
            peak = _finite_float(peak, field="max_cuda_memory_allocated_gb")
            if peak <= 0.0:
                raise ValueError("GPU validation must record positive peak memory")

        overall_auc, zero_auc, ordinary_doa, weighted_doa = _extract_split_metrics(
            coverage_path=coverage_path,
            doa_path=doa_path,
        )
        summary = {
            "schema_version": 3,
            "dataset_id": args.dataset_id,
            "split_id": args.split_id,
            "architecture": args.architecture,
            "architecture_manifest": architecture_spec(args.architecture).manifest(),
            "architecture_fingerprint": fingerprint,
            "cohort_sha256": cohort_hash,
            "controller_id": binding["controller_id"],
            "controller_route_commit": binding["route_commit"],
            "capability_counter": binding["counter"],
            "capability_nonce": binding["nonce"],
            "seed": 42,
            "split_seed": binding["split_seed"],
            "evaluation_input_role": "valid",
            "overall_auc": overall_auc,
            "zero_auc": zero_auc,
            "ordinary_doa": ordinary_doa,
            "weighted_doa": weighted_doa,
            "mastery_shape": [
                int(train_summary["num_students"]),
                int(train_summary["num_concepts"]),
            ],
            "final_loss": final_loss,
            "parameter_count": parameter_count,
            "peak_gpu_memory_gb": peak,
            "gpu_selection": {
                "physical_index": gpu_index,
                "lock_path": None if lock_path is None else str(lock_path),
                "inventory_after_lock": _gpu_payload(gpu_snapshots),
            },
            "recipe_index": args.recipe_index,
            "numerical_recipe": selected_recipe.__dict__,
        }
        _validate_split_summary(
            summary,
            dataset_id=args.dataset_id,
            split_id=args.split_id,
            fingerprint=fingerprint,
            cohort_sha256=cohort_hash,
        )
        _write_json(output_path, summary)


def _write_synthetic_fixture(root: Path) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    train_path = root / "train.csv"
    valid_path = root / "valid.csv"
    q_path = root / "Q_matrix.csv"
    rows = [
        (0, 0, "0", 1),
        (0, 1, "1", 0),
        (1, 0, "0", 0),
        (1, 2, "2", 1),
        (2, 1, "1", 1),
        (2, 2, "2", 0),
    ]
    valid_rows = [
        (0, 3, "0", 0),
        (0, 4, "1", 1),
        (1, 3, "0", 1),
        (1, 5, "2", 0),
        (2, 4, "1", 0),
        (2, 5, "2", 1),
    ]
    for path, selected in ((train_path, rows), (valid_path, valid_rows)):
        with path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("stu_id", "exer_id", "cpt_seq", "label"))
            writer.writerows(selected)
    with q_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("exer_id", "cpt_seq"))
        writer.writerows(
            ((0, "0"), (1, "1"), (2, "2"), (3, "0"), (4, "1"), (5, "2"))
        )
    return train_path, valid_path, q_path


def _run_smoke(args: argparse.Namespace) -> None:
    raw_output = (
        args.output
        if args.output is not None
        else args.output_root / "smoke.json"
    )
    output_path = _output_path(raw_output).resolve()
    root = output_path.parent / "smoke-work"
    train_path, valid_path, q_path = _write_synthetic_fixture(root / "data")
    devices = ("cpu", "gpu") if args.devices == "both" else (args.devices,)
    records: list[dict[str, object]] = []
    for device_kind in devices:
        for architecture in (args.architecture,):
            summary_path = root / f"{device_kind}-{architecture}.json"
            recipe = NumericalRecipe("full_batch", 4, 1, 1e-3, 0.0, 1)
            if device_kind == "cpu":
                gpu_index = None
                snapshots: list[GpuSnapshot] = []

                @contextmanager
                def allocation() -> Iterator[tuple[None, list[GpuSnapshot], None]]:
                    yield None, [], None

                slot = allocation()
            else:
                slot = locked_gpu()
            with slot as (gpu_index, snapshots, lock_path):
                device = "cpu" if gpu_index is None else "cuda:0"
                env = dict(os.environ)
                if gpu_index is not None:
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
                spec = architecture_spec(architecture)
                command = [
                    sys.executable,
                    "scripts/train.py",
                    "--model",
                    "unified_v2",
                    *_unified_training_flags(
                        spec,
                        evidence_loss_weight=0.1,
                    ),
                    "--train-interactions",
                    str(train_path),
                    "--valid-interactions",
                    str(valid_path),
                    "--test-interactions",
                    str(valid_path),
                    "--q-matrix",
                    str(q_path),
                    "--concept-dim",
                    "4",
                    "--epochs",
                    str(args.epochs),
                    "--early-stop-patience",
                    "1",
                    "--seed",
                    str(args.seed),
                    "--device",
                    device,
                    "--log-dir",
                    str(root / "logs"),
                    "--output",
                    str(summary_path),
                ]
                _run_checked(command, env=env)
                raw = _load_json(summary_path)
                record = {
                    "architecture": architecture,
                    "device_kind": device_kind,
                    "architecture_manifest": raw["architecture_manifest"],
                    "architecture_fingerprint": raw["architecture_fingerprint"],
                    "mastery_shape": [raw["num_students"], raw["num_concepts"]],
                    "final_loss": raw["final_loss"],
                    "parameter_count": raw["parameter_count"],
                    "peak_gpu_memory_gb": raw["max_cuda_memory_allocated_gb"],
                    "physical_gpu_index": gpu_index,
                    "physical_gpu_uuid": (
                        None if gpu_index is None else _gpu_uuid(gpu_index)
                    ),
                    "gpu_lock_path": None if lock_path is None else str(lock_path),
                    "gpu_inventory_after_lock": _gpu_payload(snapshots),
                }
                validate_smoke_summary(
                    record,
                    expected_fingerprint=architecture_fingerprint(architecture),
                    require_gpu_peak=device_kind == "gpu",
                )
                records.append(record)
    _write_json(
        output_path,
        {
            "schema_version": 3,
            "seed": args.seed,
            "synthetic": True,
            "records": records,
        },
    )


def _assemble(args: argparse.Namespace) -> None:
    cohort = load_verified_cohort(args.cohort)
    dataset_ids = cohort.get("dataset_ids")
    cohort_hash = cohort.get("cohort_sha256")
    if not isinstance(dataset_ids, list) or not isinstance(cohort_hash, str):
        raise ValueError("invalid frozen cohort")
    summaries = [_load_json(path) for path in args.split_summary]
    rows = assemble_candidate_rows(
        summaries,
        cohort_dataset_ids=dataset_ids,
        cohort_sha256=cohort_hash,
    )
    _write_json(_output_path(args.output), {"rows": rows})


def _manifest(args: argparse.Namespace) -> None:
    _write_json(_output_path(args.output), architecture_spec(args.architecture).manifest())


def _required_attempt_dir() -> Path:
    attempt_dir = os.environ.get("CAMPAIGN_ATTEMPT_DIR")
    if not attempt_dir:
        raise ValueError("registered run-split requires CAMPAIGN_ATTEMPT_DIR")
    return Path(attempt_dir)


def _run_stable_validation(args: argparse.Namespace) -> None:
    """Run both validation protocols under the stable controller's issuance."""
    output_path = _output_path(args.output).resolve()
    work_root = output_path.parent / "stable-validation-work"
    work_root.mkdir(parents=True, exist_ok=True)
    recipe = RECIPES[args.dataset][args.recipe_index]
    expected_fingerprint = architecture_fingerprint(args.architecture)
    if expected_fingerprint != args.architecture_fingerprint:
        raise ValueError("stable command architecture fingerprint mismatch")
    split_metrics: dict[str, tuple[float, float, float, float]] = {}
    peaks: list[float] = []
    gpu_uuid: str | None = None
    with locked_gpu() as (gpu_index, snapshots, lock_path):
        child_env = dict(os.environ)
        child_env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        for split_id in ("standard", "holdout"):
            split_root = work_root / split_id
            split_root.mkdir(parents=True, exist_ok=True)
            train_summary_path = split_root / "train-summary.json"
            coverage_path = split_root / "coverage-valid.json"
            doa_path = split_root / "doa-valid.json"
            train_path, valid_path, q_matrix_path, assignments = _split_paths(
                dataset_id=args.dataset,
                split_id=split_id,
                data_root=args.data_root,
            )
            required = [train_path, valid_path, q_matrix_path]
            if assignments is not None:
                required.append(assignments)
            for path in required:
                if not path.is_file():
                    raise FileNotFoundError(path)
            command = build_train_command(
                dataset_id=args.dataset,
                split_id=split_id,
                architecture=args.architecture,
                data_root=args.data_root,
                output=train_summary_path,
                device="cuda:0",
                recipe=recipe,
            )
            _run_checked(command, env=child_env)
            coverage_command, doa_command = build_evaluation_commands(
                dataset_id=args.dataset,
                split_id=split_id,
                architecture=args.architecture,
                data_root=args.data_root,
                train_summary_path=train_summary_path,
                coverage_path=coverage_path,
                doa_path=doa_path,
                device="cuda:0",
            )
            _run_checked(coverage_command, env=child_env)
            _run_checked(doa_command, env=child_env)
            train_summary = _load_json(train_summary_path)
            _validate_legacy_manifest(
                train_summary.get("architecture_manifest"),
                train_summary.get("architecture_fingerprint"),
            )
            if train_summary.get("architecture_fingerprint") != expected_fingerprint:
                raise ValueError("stable training fingerprint mismatch")
            peak = _finite_float(
                train_summary.get("max_cuda_memory_allocated_gb"),
                field="max_cuda_memory_allocated_gb",
            )
            if peak <= 0.0:
                raise ValueError("stable GPU validation must record positive peak memory")
            peaks.append(peak)
            split_metrics[split_id] = _extract_split_metrics(
                coverage_path=coverage_path,
                doa_path=doa_path,
            )
        gpu_uuid = _gpu_uuid(gpu_index)
    standard = split_metrics["standard"]
    holdout = split_metrics["holdout"]
    _write_json(output_path, {
        "schema_version": 4,
        "architecture": args.architecture,
        "architecture_manifest": architecture_spec(args.architecture).manifest(),
        "architecture_fingerprint": expected_fingerprint,
        "dataset_id": args.dataset,
        "recipe_index": args.recipe_index,
        "seed": 42,
        "split_seed": 2024,
        "cohort_sha256": args.cohort_sha256,
        "gpu_uuid": gpu_uuid,
        "peak_gpu_memory_gb": max(peaks),
        "standard_overall_auc": standard[0],
        "holdout_overall_auc": holdout[0],
        "zero_auc": holdout[1],
        "ordinary_doa": standard[2],
        "weighted_doa": standard[3],
    })


def _run_stable_test(args: argparse.Namespace) -> None:
    """Evaluate the exact frozen A2 cohort on test, after controller consumption."""
    if args.architecture != "a2":
        raise ValueError("stable test runner accepts only a2")
    output_path = _output_path(args.output).resolve()
    work_root = output_path.parent / "stable-test-work"
    work_root.mkdir(parents=True, exist_ok=True)
    fingerprint = architecture_fingerprint("a2")
    rows: dict[str, dict[str, float]] = {}
    peaks: list[float] = []
    with locked_gpu() as (gpu_index, snapshots, lock_path):
        child_env = dict(os.environ)
        child_env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        for dataset, recipe_index in STABLE_FROZEN_RECIPES.items():
            dataset_root = work_root / dataset
            dataset_root.mkdir(parents=True, exist_ok=True)
            train_summary_path = dataset_root / "train-summary.json"
            coverage_path = dataset_root / "coverage-test.json"
            doa_path = dataset_root / "doa-test.json"
            standard_dir = args.data_root / DATASET_DIRECTORIES[dataset][0]
            train_path = standard_dir / "train.csv"
            valid_path = standard_dir / "valid.csv"
            test_path = standard_dir / "test.csv"
            q_matrix_path = standard_dir / "Q_matrix.csv"
            for path in (train_path, valid_path, test_path, q_matrix_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            recipe = RECIPES[dataset][recipe_index]
            train_command = build_train_command(
                dataset_id=dataset,
                split_id="standard",
                architecture="a2",
                data_root=args.data_root,
                output=train_summary_path,
                device="cuda:0",
                recipe=recipe,
            )
            _run_checked(train_command, env=child_env)
            common = [
                "--dataset-name", dataset,
                "--summary", str(train_summary_path),
                "--model-name", "a2",
                "--split", "test",
                "--train-interactions", str(train_path),
                "--valid-interactions", str(valid_path),
                "--test-interactions", str(test_path),
                "--q-matrix", str(q_matrix_path),
                "--device", "cuda:0",
            ]
            _run_checked([
                sys.executable, "scripts/evaluate_coverage_slice.py", *common,
                "--output", str(coverage_path),
            ], env=child_env)
            _run_checked([
                sys.executable, "scripts/evaluate_doa.py", *common,
                "--min-responses", "3", "--doa-seed", "42",
                "--output", str(doa_path),
            ], env=child_env)
            train_summary = _load_json(train_summary_path)
            _validate_legacy_manifest(
                train_summary.get("architecture_manifest"),
                train_summary.get("architecture_fingerprint"),
            )
            if train_summary.get("architecture_fingerprint") != fingerprint:
                raise ValueError("stable test fingerprint mismatch")
            peak = _finite_float(
                train_summary.get("max_cuda_memory_allocated_gb"),
                field="max_cuda_memory_allocated_gb",
            )
            if peak <= 0.0:
                raise ValueError("stable GPU test must record positive peak memory")
            peaks.append(peak)
            overall, zero, ordinary, weighted = _extract_split_metrics(
                coverage_path=coverage_path,
                doa_path=doa_path,
            )
            rows[dataset] = {
                "overall_auc": overall,
                "zero_auc": zero,
                "ordinary_doa": ordinary,
                "weighted_doa": weighted,
            }
        gpu_uuid = _gpu_uuid(gpu_index)
    _write_json(output_path, {
        "schema_version": 4,
        "architecture": "a2",
        "architecture_manifest": architecture_spec("a2").manifest(),
        "architecture_fingerprint": fingerprint,
        "seed": 42,
        "split_seed": 2024,
        "gpu_uuid": gpu_uuid,
        "peak_gpu_memory_gb": max(peaks),
        "rows": rows,
    })


def _controller_init(args: argparse.Namespace) -> None:
    initialize_controller(
        state_dir=args.controller_state_dir,
        repo_root=args.repo_root,
        cohort_path=args.cohort,
        manifest_path=args.architecture_manifest,
        architecture=args.architecture,
        baseline_rows_path=args.baseline_rows,
        data_root=args.data_root,
        artifact_root=args.artifact_root,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run validation-only Unified V3 jobs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    controller_init = subparsers.add_parser("controller-init")
    controller_init.add_argument("--controller-state-dir", type=Path, required=True)
    controller_init.add_argument("--repo-root", type=Path, required=True)
    controller_init.add_argument("--cohort", type=Path, required=True)
    controller_init.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    controller_init.add_argument("--architecture-manifest", type=Path, required=True)
    controller_init.add_argument("--baseline-rows", type=Path, required=True)
    controller_init.add_argument("--data-root", type=Path, required=True)
    controller_init.add_argument("--artifact-root", type=Path, required=True)
    controller_init.set_defaults(handler=_controller_init)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.set_defaults(handler=_manifest)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--architecture", choices=RUNNER_ARCHITECTURES, required=True)
    smoke.add_argument("--devices", choices=("cpu", "gpu", "both"), default="both")
    smoke.add_argument("--seed", type=int, choices=(42,), default=42)
    smoke.add_argument("--epochs", type=int, choices=(1,), default=1)
    smoke_output = smoke.add_mutually_exclusive_group(required=True)
    smoke_output.add_argument("--output", type=Path)
    smoke_output.add_argument("--output-root", type=Path)
    smoke.set_defaults(handler=_run_smoke)

    stable_validation = subparsers.add_parser("stable-validation")
    stable_validation.add_argument("--architecture", choices=STABLE_ARCHITECTURES, required=True)
    stable_validation.add_argument("--dataset", choices=RECIPES, required=True)
    stable_validation.add_argument("--recipe-index", type=int, required=True)
    stable_validation.add_argument("--seed", type=int, choices=(42,), default=42)
    stable_validation.add_argument("--split-seed", type=int, choices=(2024,), default=2024)
    stable_validation.add_argument("--cohort-sha256", required=True)
    stable_validation.add_argument("--architecture-fingerprint", required=True)
    stable_validation.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    stable_validation.add_argument("--output", type=Path, required=True)
    stable_validation.set_defaults(handler=_run_stable_validation)

    stable_test = subparsers.add_parser("stable-test")
    stable_test.add_argument("--architecture", choices=("a2",), required=True)
    stable_test.add_argument("--seed", type=int, choices=(42,), default=42)
    stable_test.add_argument("--split-seed", type=int, choices=(2024,), default=2024)
    stable_test.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    stable_test.add_argument("--output", type=Path, required=True)
    stable_test.set_defaults(handler=_run_stable_test)

    authorize = subparsers.add_parser("authorize")
    authorize.add_argument("--controller-state-dir", type=Path, required=True)
    authorize.add_argument("--repo-root", type=Path, required=True)
    authorize.add_argument("--output", type=Path, required=True)
    authorize.set_defaults(handler=_authorize)

    run_pair = subparsers.add_parser("run-pair")
    run_pair.add_argument("--controller-state-dir", type=Path, required=True)
    run_pair.add_argument("--repo-root", type=Path, required=True)
    run_pair.set_defaults(handler=_run_pair)

    run_split = subparsers.add_parser("run-split")
    run_split.add_argument("--dataset-id", choices=ELIGIBLE_DATASET_IDS, required=True)
    run_split.add_argument("--split-id", choices=("standard", "holdout"), required=True)
    run_split.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    run_split.add_argument("--architecture-fingerprint", required=True)
    run_split.add_argument("--cohort-sha256", required=True)
    run_split.add_argument("--recipe-index", type=int, required=True)
    run_split.add_argument("--data-root", type=Path, required=True)
    run_split.add_argument("--controller-state-dir", type=Path, required=True)
    run_split.add_argument("--repo-root", type=Path, required=True)
    run_split.add_argument("--capability", type=Path, required=True)
    run_split.add_argument("--device", choices=("auto", "cpu"), default="auto")
    run_split.add_argument("--output", type=Path, required=True)
    run_split.set_defaults(handler=_run_split)

    assemble = subparsers.add_parser("assemble")
    assemble.add_argument("--cohort", type=Path, required=True)
    assemble.add_argument("--split-summary", type=Path, action="append", required=True)
    assemble.add_argument("--output", type=Path, required=True)
    assemble.set_defaults(handler=_assemble)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
