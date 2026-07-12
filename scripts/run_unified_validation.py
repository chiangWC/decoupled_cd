from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import stat
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
A2_SMOKE_DIAGNOSTIC_FIELDS = {
    "schema_version",
    "masked_edge_count",
    "observed_hard_assembly_max_abs_error",
    "reconstruction_loss",
    "gradient_parameter_count",
    "gradient_present_count",
    "gradient_finite_count",
    "gradient_nonzero_parameter_count",
    "aggregate_gradient_norm",
}
RUNNER_ARCHITECTURES = {**ARCHITECTURES, **STABLE_ARCHITECTURES}
STABLE_FROZEN_RECIPES = {"MOOCRadar": 2, "ASSIST17": 1, "XES3G5M": 0}
FROZEN_VALIDATION_DATA_ROOT = Path(
    "/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/"
    "unified-mastery-20260712/controllers/a0-controller/data"
)
FROZEN_VALIDATION_FILES = {
    "assist_17/train.csv": ("c42ddeff9605896395b6e413b0583a0a71ac31f0895faecd7457cfac483ad899", 3801552),
    "assist_17/valid.csv": ("3065f9edbd33609a5e00e80503dc55e696d721497e4350805fb8897a667e8780", 411215),
    "assist_17/Q_matrix.csv": ("23a59ec57c3b454d2d3fece3760aa91ec65a297e357766265466591b5bb0f9e3", 28938),
    "assist_17_chold_v2/train.csv": ("e3c281f01d2fedaafa289c6d950c6ae64c48e2d2d3b2f6a69bd28e88a3ef236b", 3662694),
    "assist_17_chold_v2/valid.csv": ("07b74bcfcd2c28693469d9eedddbe8b8d0fcc89d4659b079d75198886267fe70", 531157),
    "assist_17_chold_v2/Q_matrix.csv": ("23a59ec57c3b454d2d3fece3760aa91ec65a297e357766265466591b5bb0f9e3", 28938),
    "assist_17_chold_v2/student_concept_holdout_assignments.csv": ("ed88754b25dd797fc768801bc5f18b54c9d9ac4abc2eb4f9f73f31d0469435c6", 80536),
    "moocradar/train.csv": ("43826e307ea3307bb9f39c88dcec84dcf07e79ac87a67729f5221918d0619998", 5939748),
    "moocradar/valid.csv": ("1a180c21af134a1a88b88038d2b3e9a986f92d1e1ba51cede3f535bc02dceae9", 744558),
    "moocradar/Q_matrix.csv": ("b2526274cf733d0170028037b727c682eb81366804dc0f72703699fb90aaf8af", 12893),
    "moocradar_chold_v2/train.csv": ("a60037cffe8378f63e38a0b96dc79bc0ec8396abc2af620a513a11f1d5882f89", 5142346),
    "moocradar_chold_v2/valid.csv": ("53739593a1b0ca3ee33da267409acd2d26d35b7663b780a58b96b65bb915e60b", 762379),
    "moocradar_chold_v2/Q_matrix.csv": ("b2526274cf733d0170028037b727c682eb81366804dc0f72703699fb90aaf8af", 12893),
    "moocradar_chold_v2/student_concept_holdout_assignments.csv": ("15db288a2a00b76b49fe296a6cefd215fbc3958a4f38278c63bb6ad162f5ead3", 189041),
    "xes3g5m/train.csv": ("27d0f8715176f83040048498f345ca1ca3af9a1dbaef9ea3083d1821f6621740", 2320422),
    "xes3g5m/valid.csv": ("59b1a462dd1853a20941164ca48661920d918972ff1a5eabde54d6e1701ceb2f", 290071),
    "xes3g5m/Q_matrix.csv": ("13965c21cc2728281df235877805fcbf137bf851a4f9e232de6ec14620546df7", 12910),
    "xes3g5m_chold_v2/train.csv": ("6016884a69ba395650313668a6d3d511f5778160bfa94335e7ca2e6c3d7f0c1f", 2023758),
    "xes3g5m_chold_v2/valid.csv": ("104b637649d3a8c1d82d4ddd5171b2cbbd7e61754e8c480c0405fb2f3514357f", 292243),
    "xes3g5m_chold_v2/Q_matrix.csv": ("13965c21cc2728281df235877805fcbf137bf851a4f9e232de6ec14620546df7", 12910),
    "xes3g5m_chold_v2/student_concept_holdout_assignments.csv": ("b641a9662de94866427c7aacbec6093d06096c767b9154e8d69912a03e3dbac8", 120269),
}


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
    architecture: str,
    *,
    evidence_loss_weight: float,
) -> list[str]:
    spec = architecture_spec(architecture)
    if architecture in STABLE_ARCHITECTURES:
        completion_loss_weight = STABLE_ARCHITECTURES[architecture][1]
    else:
        completion_loss_weight = (
            evidence_loss_weight * ARCHITECTURES[architecture][1]
        )
    return [
        "--unified-completion",
        spec.completion,
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


def _validated_frozen_file(path: Path, *, data_root: Path) -> dict[str, object]:
    try:
        relative = path.relative_to(data_root).as_posix()
        expected_sha256, expected_size = FROZEN_VALIDATION_FILES[relative]
    except (KeyError, ValueError) as error:
        raise ValueError(f"unregistered frozen validation source: {path}") from error
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise FileNotFoundError(f"cannot read frozen validation source: {path}") from error
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise ValueError(f"frozen validation source size mismatch: {path}")
        digest = hashlib.sha256()
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    if (before.st_dev, before.st_ino, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ):
        raise ValueError(f"frozen validation source changed during hashing: {path}")
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"frozen validation source SHA-256 mismatch: {path}")
    return {
        "path": str(path),
        "relative_path": relative,
        "size_bytes": expected_size,
        "sha256": expected_sha256,
    }


def _validate_generated_argv(command: Sequence[str]) -> None:
    script = Path(command[1]).name
    argv = list(command[2:])
    if script == "train.py":
        from scripts import train as train_script

        parsed = train_script.parse_args(argv)
        train_script.validate_model_args(parsed)
    elif script == "evaluate_coverage_slice.py":
        from scripts import evaluate_coverage_slice

        evaluate_coverage_slice.parse_args(argv)
    elif script == "evaluate_doa.py":
        from scripts import evaluate_doa

        evaluate_doa.parse_args(argv)
    else:
        raise ValueError(f"unregistered stable command parser: {script}")


def _validate_stable_graph_training_argv(
    command: Sequence[str],
    *,
    architecture: str,
    expected_hidden_dim: int | None,
    require_smoke_diagnostics: bool,
) -> None:
    hidden_flag = "--unified-graph-hidden-dim"
    diagnostic_flag = "--unified-a2-smoke-diagnostics"
    hidden_count = list(command).count(hidden_flag)
    diagnostic_count = list(command).count(diagnostic_flag)
    if architecture == "a2":
        if type(expected_hidden_dim) is not int or expected_hidden_dim <= 0:
            raise ValueError("A2 preflight hidden dimension is invalid")
        if hidden_count != 1:
            raise ValueError("A2 preflight must bind graph hidden dimension once")
        index = list(command).index(hidden_flag)
        if index + 1 >= len(command) or command[index + 1] != str(expected_hidden_dim):
            raise ValueError("A2 graph hidden dimension differs from frozen recipe")
        expected_diagnostic_count = 1 if require_smoke_diagnostics else 0
        if diagnostic_count != expected_diagnostic_count:
            raise ValueError("A2 smoke diagnostic flag binding mismatch")
    elif hidden_count != 0 or diagnostic_count != 0:
        raise ValueError("non-A2 command must not carry graph-only flags")


def preflight_stable_smoke(*, architecture: str) -> dict[str, object]:
    root = Path("/preflight/stable-smoke")
    command = [
        sys.executable,
        "scripts/train.py",
        "--model",
        "unified_v2",
        *_unified_training_flags(architecture, evidence_loss_weight=0.1),
        "--train-interactions",
        str(root / "train.csv"),
        "--valid-interactions",
        str(root / "valid.csv"),
        "--test-interactions",
        str(root / "valid.csv"),
        "--q-matrix",
        str(root / "Q_matrix.csv"),
        "--concept-dim",
        "4",
        "--epochs",
        "1",
        "--early-stop-patience",
        "1",
        "--seed",
        "42",
        "--device",
        "cuda:0",
        "--log-dir",
        str(root / "logs"),
        "--output",
        str(root / "summary.json"),
    ]
    if architecture == "a2":
        command.extend(
            [
                "--unified-graph-hidden-dim",
                "4",
                "--unified-a2-smoke-diagnostics",
            ]
        )
    _validate_stable_graph_training_argv(
        command,
        architecture=architecture,
        expected_hidden_dim=4 if architecture == "a2" else None,
        require_smoke_diagnostics=architecture == "a2",
    )
    _validate_generated_argv(command)
    return {"architecture": architecture, "command": command}


def preflight_stable_validation(
    *, architecture: str, dataset_id: str, recipe_index: int
) -> list[dict[str, object]]:
    if dataset_id not in STABLE_FROZEN_RECIPES:
        raise ValueError(f"dataset is outside stable frozen cohort: {dataset_id}")
    if recipe_index != STABLE_FROZEN_RECIPES[dataset_id]:
        raise ValueError("stable preflight recipe differs from frozen recipe")
    recipe = RECIPES[dataset_id][recipe_index]
    records: list[dict[str, object]] = []
    for split_id in ("standard", "holdout"):
        train_path, valid_path, q_matrix_path, assignments = _split_paths(
            dataset_id=dataset_id,
            split_id=split_id,
            data_root=FROZEN_VALIDATION_DATA_ROOT,
        )
        paths = [train_path, valid_path, q_matrix_path]
        if assignments is not None:
            paths.append(assignments)
        files = [
            _validated_frozen_file(path, data_root=FROZEN_VALIDATION_DATA_ROOT)
            for path in paths
        ]
        output_root = Path("/preflight/stable-validation") / dataset_id / split_id
        train_command = build_train_command(
            dataset_id=dataset_id,
            split_id=split_id,
            architecture=architecture,
            data_root=FROZEN_VALIDATION_DATA_ROOT,
            output=output_root / "train-summary.json",
            device="cuda:0",
            recipe=recipe,
        )
        coverage_command, doa_command = build_evaluation_commands(
            dataset_id=dataset_id,
            split_id=split_id,
            architecture=architecture,
            data_root=FROZEN_VALIDATION_DATA_ROOT,
            train_summary_path=output_root / "train-summary.json",
            coverage_path=output_root / "coverage-valid.json",
            doa_path=output_root / "doa-valid.json",
            device="cuda:0",
        )
        for command in (train_command, coverage_command, doa_command):
            _validate_generated_argv(command)
        _validate_stable_graph_training_argv(
            train_command,
            architecture=architecture,
            expected_hidden_dim=(
                recipe.concept_dim if architecture == "a2" else None
            ),
            require_smoke_diagnostics=False,
        )
        records.append(
            {
                "dataset_id": dataset_id,
                "split_id": split_id,
                "recipe_index": recipe_index,
                "files": files,
                "commands": {
                    "train": train_command,
                    "coverage": coverage_command,
                    "doa": doa_command,
                },
            }
        )
    return records


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
    command = [
        sys.executable,
        "scripts/train.py",
        "--model",
        "unified_v2",
        *_unified_training_flags(
            architecture,
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
    if architecture == "a2":
        command.extend(
            [
                "--unified-graph-hidden-dim",
                str(selected_recipe.concept_dim),
            ]
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
    require_a2_diagnostics: bool = False,
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
    diagnostics = summary.get("a2_smoke_diagnostics")
    if not require_a2_diagnostics:
        if diagnostics is not None:
            raise ValueError("non-A2 smoke must not claim A2 diagnostics")
        return
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != A2_SMOKE_DIAGNOSTIC_FIELDS:
        raise ValueError("A2 smoke diagnostic schema mismatch")
    if diagnostics.get("schema_version") != 1:
        raise ValueError("A2 smoke diagnostic version mismatch")
    masked_edge_count = diagnostics.get("masked_edge_count")
    if type(masked_edge_count) is not int or masked_edge_count <= 0:
        raise ValueError("A2 smoke masked edge count must be positive")
    if diagnostics.get("observed_hard_assembly_max_abs_error") != 0.0:
        raise ValueError("A2 smoke observed hard assembly must be exact")
    reconstruction_loss = _finite_float(
        diagnostics.get("reconstruction_loss"),
        field="reconstruction_loss",
    )
    if reconstruction_loss <= 0.0:
        raise ValueError("A2 smoke reconstruction loss must be positive")
    parameter_count = diagnostics.get("gradient_parameter_count")
    if type(parameter_count) is not int or parameter_count <= 0:
        raise ValueError("A2 smoke gradient parameter count must be positive")
    for field in ("gradient_present_count", "gradient_finite_count"):
        if diagnostics.get(field) != parameter_count:
            raise ValueError(f"A2 smoke {field} must cover every parameter")
    nonzero_count = diagnostics.get("gradient_nonzero_parameter_count")
    if (
        type(nonzero_count) is not int
        or nonzero_count <= 0
        or nonzero_count > parameter_count
    ):
        raise ValueError("A2 smoke must have a nonzero graph gradient")
    aggregate_norm = _finite_float(
        diagnostics.get("aggregate_gradient_norm"),
        field="aggregate_gradient_norm",
    )
    if aggregate_norm <= 0.0:
        raise ValueError("A2 smoke aggregate gradient norm must be positive")


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
    coverage_slice_csv: Path | None = None,
    coverage_summary_csv: Path | None = None,
    doa_summary_csv: Path | None = None,
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
    if coverage_slice_csv is not None:
        coverage_command.extend(["--slice-csv", str(coverage_slice_csv)])
    if coverage_summary_csv is not None:
        coverage_command.extend(["--summary-csv", str(coverage_summary_csv)])
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
    if doa_summary_csv is not None:
        doa_command.extend(["--summary-csv", str(doa_summary_csv)])
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
    if args.architecture == "a2" and not args.a2_smoke_diagnostics:
        raise ValueError("A2 smoke requires --a2-smoke-diagnostics")
    if args.architecture != "a2" and args.a2_smoke_diagnostics:
        raise ValueError("A2 smoke diagnostics are graph-only")
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
                command = [
                    sys.executable,
                    "scripts/train.py",
                    "--model",
                    "unified_v2",
                    *_unified_training_flags(
                        architecture,
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
                if architecture == "a2":
                    command.extend(
                        [
                            "--unified-graph-hidden-dim",
                            "4",
                            "--unified-a2-smoke-diagnostics",
                        ]
                    )
                _validate_stable_graph_training_argv(
                    command,
                    architecture=architecture,
                    expected_hidden_dim=(4 if architecture == "a2" else None),
                    require_smoke_diagnostics=architecture == "a2",
                )
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
                    "a2_smoke_diagnostics": raw.get(
                        "a2_smoke_diagnostics"
                    ),
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
                    require_a2_diagnostics=architecture == "a2",
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


def _publish_authoritative_json(source: Path, destination: Path) -> None:
    """Atomically move a completed generated JSON into its fixed source slot."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    os.replace(source, destination)


def _write_aux_artifact_manifest(aux_root: Path, *, kind: str) -> Path:
    records: list[dict[str, object]] = []
    for path in sorted(aux_root.rglob("*"), key=lambda item: item.as_posix()):
        status = path.lstat()
        if stat.S_ISLNK(status.st_mode):
            raise ValueError(f"auxiliary artifact must not be a symlink: {path}")
        if stat.S_ISDIR(status.st_mode):
            continue
        if not stat.S_ISREG(status.st_mode):
            raise ValueError(f"auxiliary artifact must be regular: {path}")
        relative = path.relative_to(aux_root).as_posix()
        if relative == "artifact-manifest.json":
            raise ValueError("auxiliary artifact manifest already exists")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(f"auxiliary artifact changed while hashing: {path}")
        records.append(
            {
                "relative_path": relative,
                "size_bytes": before.st_size,
                "sha256": digest.hexdigest(),
                "device": before.st_dev,
                "inode": before.st_ino,
            }
        )
    manifest = aux_root / "artifact-manifest.json"
    _write_json(
        manifest,
        {"schema_version": 1, "kind": kind, "artifacts": records},
    )
    return manifest


def rehearse_stable_validation_layout(attempt_dir: Path) -> dict[str, object]:
    data_root = attempt_dir / "rehearsal-data"
    for directory_name in DATASET_DIRECTORIES["ASSIST17"]:
        _write_synthetic_fixture(data_root / directory_name)
    assignments = (
        data_root
        / DATASET_DIRECTORIES["ASSIST17"][1]
        / "student_concept_holdout_assignments.csv"
    )
    with assignments.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("stu_id", "num_rows", "num_concepts", "mode", "holdout_concepts", "holdout_rows")
        )
        for student_id in range(3):
            writer.writerow((student_id, 4, 3, "random", "", 0))
    work_root = attempt_dir / "stable-validation-work"
    recipe = NumericalRecipe("full_batch", 4, 1, 1e-3, 0.0, 1)
    for split_id in ("standard", "holdout"):
        authoritative = work_root / split_id
        authoritative.mkdir(parents=True)
        aux_root = work_root / "aux" / split_id
        aux_root.mkdir(parents=True)
        train_summary = aux_root / "train-summary.json"
        coverage = aux_root / "coverage-valid.json"
        doa = aux_root / "doa-valid.json"
        train_command = build_train_command(
            dataset_id="ASSIST17",
            split_id=split_id,
            architecture="a0v4",
            data_root=data_root,
            output=train_summary,
            device="cpu",
            recipe=recipe,
        )
        _run_checked(train_command, env=dict(os.environ))
        coverage_command, doa_command = build_evaluation_commands(
            dataset_id="ASSIST17",
            split_id=split_id,
            architecture="a0v4",
            data_root=data_root,
            train_summary_path=train_summary,
            coverage_path=coverage,
            doa_path=doa,
            device="cpu",
        )
        _run_checked(coverage_command, env=dict(os.environ))
        _run_checked(doa_command, env=dict(os.environ))
        for source, name in (
            (train_summary, "train-summary.json"),
            (coverage, "coverage-valid.json"),
            (doa, "doa-valid.json"),
        ):
            _publish_authoritative_json(source, authoritative / name)
    manifest = _write_aux_artifact_manifest(work_root / "aux", kind="validation")
    return {
        "attempt_dir": str(attempt_dir),
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }


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
            aux_root = work_root / "aux" / split_id
            aux_root.mkdir(parents=True, exist_ok=True)
            train_summary_path = aux_root / "train-summary.json"
            coverage_path = aux_root / "coverage-valid.json"
            doa_path = aux_root / "doa-valid.json"
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
                coverage_slice_csv=aux_root / "coverage-valid_slices.csv",
                coverage_summary_csv=aux_root / "coverage-valid_summary.csv",
                doa_summary_csv=aux_root / "doa-valid_summary.csv",
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
            for source, name in (
                (train_summary_path, "train-summary.json"),
                (coverage_path, "coverage-valid.json"),
                (doa_path, "doa-valid.json"),
            ):
                _publish_authoritative_json(source, split_root / name)
        gpu_uuid = _gpu_uuid(gpu_index)
    _write_aux_artifact_manifest(work_root / "aux", kind="validation")
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
            aux_root = work_root / "aux" / dataset
            aux_root.mkdir(parents=True, exist_ok=True)
            train_summary_path = aux_root / "train-summary.json"
            coverage_path = aux_root / "coverage-test.json"
            doa_path = aux_root / "doa-test.json"
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
                "--slice-csv", str(aux_root / "coverage-test_slices.csv"),
                "--summary-csv", str(aux_root / "coverage-test_summary.csv"),
            ], env=child_env)
            _run_checked([
                sys.executable, "scripts/evaluate_doa.py", *common,
                "--min-responses", "3", "--doa-seed", "42",
                "--output", str(doa_path),
                "--summary-csv", str(aux_root / "doa-test_summary.csv"),
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
                "recipe_index": recipe_index,
                "overall_auc": overall,
                "zero_auc": zero,
                "ordinary_doa": ordinary,
                "weighted_doa": weighted,
            }
            for source, name in (
                (train_summary_path, "train-summary.json"),
                (coverage_path, "coverage-test.json"),
                (doa_path, "doa-test.json"),
            ):
                _publish_authoritative_json(source, dataset_root / name)
        gpu_uuid = _gpu_uuid(gpu_index)
    _write_aux_artifact_manifest(work_root / "aux", kind="test")
    _write_json(output_path, {
        "schema_version": 4,
        "architecture": "a2",
        "architecture_manifest": architecture_spec("a2").manifest(),
        "architecture_fingerprint": fingerprint,
        "seed": 42,
        "split_seed": 2024,
        "cohort_sha256": (
            "6342dc8a5f73a4e03a1645780597b625c"
            "1480ba7a6513668b6766089cdd5b8a5"
        ),
        "recipes": dict(STABLE_FROZEN_RECIPES),
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
    smoke.add_argument("--a2-smoke-diagnostics", action="store_true")
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
    stable_validation.add_argument(
        "--data-root", type=Path, default=FROZEN_VALIDATION_DATA_ROOT
    )
    stable_validation.add_argument("--output", type=Path, required=True)
    stable_validation.set_defaults(handler=_run_stable_validation)

    stable_test = subparsers.add_parser("stable-test")
    stable_test.add_argument("--architecture", choices=("a2",), required=True)
    stable_test.add_argument("--seed", type=int, choices=(42,), default=42)
    stable_test.add_argument("--split-seed", type=int, choices=(2024,), default=2024)
    stable_test.add_argument(
        "--data-root", type=Path, default=FROZEN_VALIDATION_DATA_ROOT
    )
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
