from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.monotone_requirement_protocol import (
    RequirementFeatureSet,
    TrainOnlyRequirementProtocol,
    build_train_only_requirement_protocol,
    collate_requirement_batch,
    load_audit_query_labels,
)
from data.pool_protocol import sha256_file
from models.monotone_requirement_surface import MonotoneRequirementSurface
from utils.monotone_requirement_evaluation import (
    VARIANT_NAMES,
    aggregate_stage1_gate,
    align_prediction_manifests,
    evaluate_requirement_surface,
)


MODEL_SEED = 42
STEPS = 2_000
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 0.0
PREDICTION_BATCH_SIZE = 4_096
EXPECTED_BRANCH = "codex/student-local-inductive"
DATASET_DIRS = {
    "ASSIST17": "assist_17_chold_v2",
    "MOOCRadar": "moocradar_chold_v2",
}
FROZEN_SOURCE_SHA256 = {
    "ASSIST17": {
        "train_full_sha256": "e3c281f01d2fedaafa289c6d950c6ae64c48e2d2d3b2f6a69bd28e88a3ef236b",
        "q_matrix_sha256": "23a59ec57c3b454d2d3fece3760aa91ec65a297e357766265466591b5bb0f9e3",
    },
    "MOOCRadar": {
        "train_full_sha256": "a60037cffe8378f63e38a0b96dc79bc0ec8396abc2af620a513a11f1d5882f89",
        "q_matrix_sha256": "b2526274cf733d0170028037b727c682eb81366804dc0f72703699fb90aaf8af",
    },
}
MODEL_MODES = {
    "full": "full",
    "pooled": "pooled_direct",
    "capacity": "capacity_additive",
    "standard_ncd": "standard_ncd",
}
SURFACE_STEP = 0.05
SURFACE_NONCOLLAPSE_THRESHOLD = 0.001


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _hash_values(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def _hash_tensor(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(_hash_tensor(value).encode("ascii"))
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _seed_everything() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(MODEL_SEED)
    torch.manual_seed(MODEL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MODEL_SEED)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _run_git(arguments: Sequence[str]) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def formal_snapshot(expected_commit: str) -> dict[str, Any]:
    head = _run_git(["rev-parse", "HEAD"])
    if head != expected_commit:
        raise RuntimeError(f"HEAD {head} differs from expected {expected_commit}.")
    branch = _run_git(["branch", "--show-current"])
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"Formal screen requires branch {EXPECTED_BRANCH}.")
    if _run_git(["status", "--porcelain", "--untracked-files=all"]):
        raise RuntimeError("Formal screen requires a clean worktree.")
    environment = os.environ.get("CONDA_DEFAULT_ENV")
    if environment != "decoupled_cd":
        raise RuntimeError("Formal screen requires the decoupled_cd Conda environment.")

    remote_sha = ""
    errors = []
    for delay in (0.0, 0.5, 1.0):
        if delay:
            time.sleep(delay)
        result = subprocess.run(
            ["git", "ls-remote", "origin", f"refs/heads/{EXPECTED_BRANCH}"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            remote_sha = result.stdout.split()[0]
            break
        errors.append(result.stderr.strip() or f"exit={result.returncode}")
    if remote_sha != head:
        raise RuntimeError(
            f"Origin does not contain formal HEAD {head}; got {remote_sha!r}. "
            f"Attempts: {errors}"
        )
    return {
        "head": head,
        "branch": branch,
        "origin_head": remote_sha,
        "environment": environment,
        "worktree_clean": True,
        "formal": True,
    }


def _dataset_paths(data_root: Path, dataset: str) -> tuple[Path, Path]:
    if dataset not in DATASET_DIRS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    source = (data_root / DATASET_DIRS[dataset]).resolve()
    train_path = source / "train.csv"
    q_path = source / "Q_matrix.csv"
    if not train_path.is_file() or not q_path.is_file():
        raise FileNotFoundError(f"Missing train/Q files under {source}.")
    return train_path, q_path

def _assert_frozen_source(
    *,
    dataset: str,
    train_path: Path,
    q_path: Path,
) -> dict[str, str]:
    actual = {
        "train_full_sha256": sha256_file(train_path),
        "q_matrix_sha256": sha256_file(q_path),
    }
    expected = FROZEN_SOURCE_SHA256[dataset]
    if actual != expected:
        raise RuntimeError(
            f"Frozen source SHA-256 mismatch for {dataset}: "
            f"expected {expected}, got {actual}."
        )
    return actual


def _build_protocol(data_root: Path, dataset: str) -> TrainOnlyRequirementProtocol:
    train_path, q_path = _dataset_paths(data_root, dataset)
    return build_train_only_requirement_protocol(
        dataset=dataset,
        train_path=train_path,
        q_matrix_path=q_path,
    )


def eligible_indices(features: RequirementFeatureSet) -> np.ndarray:
    indices = np.fromiter(
        (index for index, record in enumerate(features.records) if record.eligible),
        dtype=np.int64,
    )
    if not len(indices):
        raise RuntimeError("No eligible optimizer rows.")
    labels = np.asarray([features.records[int(index)].label for index in indices])
    if np.unique(labels).size != 2:
        raise RuntimeError("Eligible optimizer rows require both response classes.")
    return indices


def _common_unary_sha256(model: MonotoneRequirementSurface) -> str:
    return _state_dict_sha256(
        {
            name: value
            for name, value in model.state_dict().items()
            if name.startswith("unary_lattice.")
        }
    )


@torch.no_grad()
def _initialization_audit(num_concepts: int) -> dict[str, Any]:
    full = _model(variant="full", num_concepts=num_concepts)
    capacity = _model(variant="capacity", num_concepts=num_concepts)
    readiness = torch.ones(4, num_concepts)
    q_mask = torch.zeros_like(readiness, dtype=torch.bool)
    pairs = ((0.1, 0.8), (0.2, 0.4), (0.35, 0.9), (0.6, 0.7))
    for row, values in enumerate(pairs):
        readiness[row, :2] = torch.tensor(values)
        q_mask[row, :2] = True
    offsets = torch.tensor([-0.4, 0.0, 0.3, 0.8])
    difference = float(
        (full(readiness, q_mask, offsets) - capacity(readiness, q_mask, offsets))
        .abs()
        .max()
        .item()
    )
    unary_hashes = {
        "full": _common_unary_sha256(full),
        "capacity": _common_unary_sha256(capacity),
    }
    if difference >= 1e-6 or len(set(unary_hashes.values())) != 1:
        raise RuntimeError("Full/Capacity initialization parity failed.")
    if (
        full.active_variant_parameter_count != 16
        or capacity.active_variant_parameter_count != 16
    ):
        raise RuntimeError("Full/Capacity active parameter parity failed.")
    return {
        "fixture_max_absolute_prediction_difference": difference,
        "maximum_allowed_difference": 1e-6,
        "active_parameter_count": 16,
        "common_unary_sha256": next(iter(unary_hashes.values())),
    }


def _architecture_payload(*, steps: int) -> dict[str, Any]:
    payload = {
        "family": "bottleneck_aware_monotone_requirement_surface_screen_v1",
        "model_modes": MODEL_MODES,
        "model_seed": MODEL_SEED,
        "optimizer": "Adam",
        "steps": steps,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "epsilon": 1e-3,
        "surface_step": SURFACE_STEP,
        "surface_noncollapse_threshold": SURFACE_NONCOLLAPSE_THRESHOLD,
        "source_sha256": {
            "protocol": sha256_file(
                PROJECT_ROOT / "data/monotone_requirement_protocol.py"
            ),
            "model": sha256_file(
                PROJECT_ROOT / "models/monotone_requirement_surface.py"
            ),
            "evaluation": sha256_file(
                PROJECT_ROOT / "utils/monotone_requirement_evaluation.py"
            ),
            "runner": sha256_file(Path(__file__)),
        },
    }
    payload["fingerprint"] = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload




def _model(
    *,
    variant: str,
    num_concepts: int,
) -> MonotoneRequirementSurface:
    if variant not in MODEL_MODES:
        raise ValueError(f"Unknown manifest variant: {variant}")
    _seed_everything()
    return MonotoneRequirementSurface(
        num_concepts=num_concepts,
        mode=MODEL_MODES[variant],
        epsilon=1e-3,
        seed=MODEL_SEED,
    )


def _active_named_parameters(
    model: MonotoneRequirementSurface,
) -> list[tuple[str, torch.nn.Parameter]]:
    parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("variant.")
    ]
    if not parameters:
        raise RuntimeError("Variant has no active parameters.")
    return parameters


def _active_parameters(model: MonotoneRequirementSurface) -> list[torch.nn.Parameter]:
    return [parameter for _, parameter in _active_named_parameters(model)]


def _profile_supported_forward_flops(
    *,
    variant: str,
    num_concepts: int,
    batch: Any,
) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    row_count = min(32, int(batch.readiness.size(0)))
    cpu_model = _model(variant=variant, num_concepts=num_concepts)
    readiness = batch.readiness[:row_count].detach().cpu()
    q_mask = batch.q_mask[:row_count].detach().cpu()
    item_offset = batch.item_offset[:row_count].detach().cpu()
    cpu_model.eval()
    with profile(
        activities=[ProfilerActivity.CPU],
        with_flops=True,
        record_shapes=False,
    ) as profiler:
        with torch.no_grad():
            cpu_model(readiness, q_mask, item_offset)
    operator_flops = {
        event.key: int(event.flops)
        for event in profiler.key_averages()
        if event.flops is not None and int(event.flops) > 0
    }
    total = sum(operator_flops.values())
    if total <= 0:
        raise RuntimeError(f"FLOPs profiler found no supported ops for {variant}.")
    return {
        "method": "torch.profiler_cpu_supported_operators",
        "sample_rows": row_count,
        "sample_active_q_coordinates": int(q_mask.sum().item()),
        "supported_forward_flops": total,
        "supported_forward_flops_per_row": total / row_count,
        "operator_flops": operator_flops,
        "coverage_note": (
            "Counts only operators for which torch.profiler reports FLOPs; "
            "transcendentals, indexing, comparisons, and some elementwise ops "
            "are not included."
        ),
    }


def _gradient_audit(
    model: MonotoneRequirementSurface,
    *,
    batch: Any,
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    probabilities = model(batch.readiness, batch.q_mask, batch.item_offset)
    assert batch.labels is not None
    loss = F.binary_cross_entropy(probabilities, batch.labels)
    loss.backward()
    total = 0
    nonzero = 0
    minimum = float("inf")
    maximum = 0.0
    per_parameter = {}
    for name, parameter in _active_named_parameters(model):
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError("Missing or non-finite active-parameter gradient.")
        gradient = parameter.grad.detach().abs()
        current_nonzero = int(torch.count_nonzero(gradient))
        if current_nonzero == 0:
            raise RuntimeError(f"Active parameter tensor has zero gradient: {name}")
        total += gradient.numel()
        nonzero += current_nonzero
        minimum = min(minimum, float(gradient.min().item()))
        maximum = max(maximum, float(gradient.max().item()))
        per_parameter[name] = {
            "elements": gradient.numel(),
            "nonzero_elements": current_nonzero,
            "maximum_absolute_gradient": float(gradient.max().item()),
        }
    model.zero_grad(set_to_none=True)
    return {
        "loss": float(loss.item()),
        "active_parameter_elements": total,
        "nonzero_gradient_elements": nonzero,
        "nonzero_gradient_fraction": nonzero / total,
        "minimum_absolute_gradient": minimum,
        "maximum_absolute_gradient": maximum,
        "every_parameter_tensor_has_nonzero_gradient": True,
        "per_parameter": per_parameter,
    }


def _train_variant(
    model: MonotoneRequirementSurface,
    *,
    batch: Any,
    steps: int = STEPS,
) -> dict[str, Any]:
    if steps < 1:
        raise ValueError("steps must be positive.")
    optimizer = torch.optim.Adam(
        _active_parameters(model),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    milestones = {1, 10, 100, 500, 1_000, steps}
    losses: dict[str, float] = {}
    model.train()
    assert batch.labels is not None
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        probabilities = model(batch.readiness, batch.q_mask, batch.item_offset)
        loss = F.binary_cross_entropy(probabilities, batch.labels)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite training loss at step {step}.")
        loss.backward()
        optimizer.step()
        if step in milestones:
            losses[str(step)] = float(loss.item())
    second_half_improvement = None
    if steps >= 1_000:
        denominator = max(abs(losses["1000"]), 1e-12)
        second_half_improvement = (
            losses["1000"] - losses[str(steps)]
        ) / denominator
    return {
        "steps": steps,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "full_batch": True,
        "losses": losses,
        "final_loss": losses[str(steps)],
        "relative_loss_improvement_from_step_1000": second_half_improvement,
    }


@torch.no_grad()
def _predict(
    model: MonotoneRequirementSurface,
    *,
    features: RequirementFeatureSet,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    probabilities = []
    for start in range(0, len(features.records), PREDICTION_BATCH_SIZE):
        indices = np.arange(
            start,
            min(start + PREDICTION_BATCH_SIZE, len(features.records)),
            dtype=np.int64,
        )
        batch = collate_requirement_batch(features, indices, device=device)
        if batch.labels is not None:
            raise RuntimeError("Audit prediction batch unexpectedly contains labels.")
        values = model(batch.readiness, batch.q_mask, batch.item_offset)
        probabilities.append(values.detach().cpu().numpy())
    result = features.metadata_frame()
    result["prob"] = np.concatenate(probabilities).astype(np.float64)
    if len(result) != len(features.records) or not np.isfinite(result["prob"]).all():
        raise RuntimeError("Invalid audit prediction vector.")
    return result


def _surface_summary(frame: pd.DataFrame) -> dict[str, Any]:
    required = {"b", "g", "prob", "learned_surface", "learned_logit", "prob_epsilon_zero"}
    if required - set(frame.columns):
        raise ValueError("Surface frame is incomplete.")
    learned_values = frame["learned_surface"].to_numpy(dtype=float)
    clipped_points = int(
        np.count_nonzero((learned_values <= 1e-6) | (learned_values >= 1.0 - 1e-6))
    )
    if clipped_points:
        raise RuntimeError("Learned surface hit the logit clamp on the fixed grid.")
    lookup = {
        (round(float(row.b), 10), round(float(row.g), 10)): float(row.learned_logit)
        for row in frame.itertuples(index=False)
    }
    grid = np.arange(0.05, 1.0, SURFACE_STEP)
    interactions = []
    for b0 in grid[:-1]:
        b1 = b0 + SURFACE_STEP
        for g0 in grid[:-1]:
            g1 = g0 + SURFACE_STEP
            keys = [
                (round(float(b0), 10), round(float(g0), 10)),
                (round(float(b0), 10), round(float(g1), 10)),
                (round(float(b1), 10), round(float(g0), 10)),
                (round(float(b1), 10), round(float(g1), 10)),
            ]
            if all(key in lookup for key in keys):
                value = (
                    lookup[keys[0]] - lookup[keys[1]]
                    - lookup[keys[2]] + lookup[keys[3]]
                )
                interactions.append(
                    {"b0": keys[0][0], "b1": keys[2][0],
                     "g0": keys[0][1], "g1": keys[1][1],
                     "interaction": value}
                )
    if not interactions:
        raise RuntimeError("No feasible surface cells.")
    maximum = max(abs(value["interaction"]) for value in interactions)
    b_values = frame["b"].to_numpy(dtype=float)
    g_values = frame["g"].to_numpy(dtype=float)
    logits = frame["learned_logit"].to_numpy(dtype=float)
    b_levels = sorted(set(b_values))
    g_levels = sorted(set(g_values))
    design = np.ones(
        (len(frame), 1 + len(b_levels) - 1 + len(g_levels) - 1),
        dtype=float,
    )
    column = 1
    for level in b_levels[1:]:
        design[:, column] = np.isclose(b_values, level)
        column += 1
    for level in g_levels[1:]:
        design[:, column] = np.isclose(g_values, level)
        column += 1
    coefficients, *_ = np.linalg.lstsq(design, logits, rcond=None)
    additive_residual = logits - design @ coefficients
    fixed_g_gaps = [
        float(group["learned_logit"].max() - group["learned_logit"].min())
        for _, group in frame.groupby("g", sort=True)
        if group["b"].nunique() > 1
    ]
    epsilon_difference = np.abs(
        frame["prob"].to_numpy(dtype=float)
        - frame["prob_epsilon_zero"].to_numpy(dtype=float)
    )
    return {
        "grid_step": SURFACE_STEP, "feasible_points": len(frame),
        "feasible_adjacent_cells": len(interactions),
        "maximum_absolute_mixed_difference": maximum,
        "interaction_scale": "logit(learned_surface)",
        "learned_surface_clipped_point_count": clipped_points,
        "best_additive_projection": {
            "rmse": float(np.sqrt(np.mean(np.square(additive_residual)))),
            "maximum_absolute_residual": float(np.max(np.abs(additive_residual))),
        },
        "fixed_g_bottleneck_gap": {
            "identified_level_count": len(fixed_g_gaps),
            "mean_logit_gap": float(np.mean(fixed_g_gaps)),
            "maximum_logit_gap": float(np.max(fixed_g_gaps)),
        },
        "epsilon_zero_max_absolute_probability_difference": float(
            np.max(epsilon_difference)
        ),
        "noncollapse_threshold": SURFACE_NONCOLLAPSE_THRESHOLD,
        "noncollapsed": maximum >= SURFACE_NONCOLLAPSE_THRESHOLD,
        "interactions": interactions,
    }


@torch.no_grad()
def _surface_diagnostic(
    model: MonotoneRequirementSurface,
    *,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    grid = np.arange(0.05, 1.0, SURFACE_STEP)
    points = []
    for b in grid:
        for g in grid:
            if b <= g + 1e-12 and g <= (1.0 + b) / 2.0 + 1e-12:
                points.append((round(float(b), 10), round(float(g), 10)))
    readiness = torch.ones(
        (len(points), model.num_concepts),
        dtype=torch.float32,
        device=device,
    )
    q_mask = torch.zeros_like(readiness, dtype=torch.bool)
    for row, (b, g) in enumerate(points):
        readiness[row, 0] = b
        readiness[row, 1] = 2.0 * g - b
        q_mask[row, :2] = True
    offsets = torch.zeros(len(points), dtype=torch.float32, device=device)
    probabilities = model(readiness, q_mask, offsets).detach().cpu().numpy()
    learned_surface = model.learned_surface(readiness, q_mask)
    requirement_surface = model.requirement_surface(readiness, q_mask, offsets)
    learned_logit = torch.logit(
        learned_surface.clamp(1e-6, 1.0 - 1e-6)
    )
    epsilon_zero_probability = learned_surface.clamp(1e-6, 1.0 - 1e-6)
    frame = pd.DataFrame(points, columns=["b", "g"])
    frame["prob"] = probabilities
    frame["requirement_surface"] = requirement_surface.detach().cpu().numpy()
    frame["learned_surface"] = learned_surface.detach().cpu().numpy()
    frame["learned_logit"] = learned_logit.detach().cpu().numpy()
    frame["prob_epsilon_zero"] = epsilon_zero_probability.detach().cpu().numpy()
    diagnostic = _surface_summary(frame)
    return frame, diagnostic


def _prediction_semantic_sha256(frame: pd.DataFrame) -> str:
    return _hash_values(
        f"{row.row_id}:{float(row.prob):.9g}"
        for row in frame[["row_id", "prob"]].itertuples(index=False)
    )


def predict_dataset(
    *,
    dataset: str,
    data_root: Path,
    output_root: Path,
    expected_commit: str,
    device: torch.device,
    steps: int = STEPS,
    require_formal: bool = True,
) -> dict[str, Any]:
    if require_formal and steps != STEPS:
        raise ValueError("Formal prediction fixes 2,000 steps.")
    snapshot = (
        formal_snapshot(expected_commit)
        if require_formal
        else {"head": _run_git(["rev-parse", "HEAD"]), "formal": False}
    )
    output_dir = output_root / dataset
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path, q_path = _dataset_paths(data_root, dataset)
    if require_formal:
        _assert_frozen_source(dataset=dataset, train_path=train_path, q_path=q_path)
    protocol = _build_protocol(data_root, dataset)
    train_indices = eligible_indices(protocol.optimizer)
    initialization_audit = _initialization_audit(len(protocol.concept_index))
    architecture = _architecture_payload(steps=steps)
    train_batch = collate_requirement_batch(
        protocol.optimizer,
        train_indices,
        device=device,
    )
    if train_batch.labels is None:
        raise RuntimeError("Eligible optimizer batch has no labels.")
    input_hashes = {
        "readiness": _hash_tensor(train_batch.readiness),
        "q_mask": _hash_tensor(train_batch.q_mask),
        "item_offset": _hash_tensor(train_batch.item_offset),
        "labels": _hash_tensor(train_batch.labels),
        "row_ids": _hash_values(train_batch.row_ids),
    }
    manifests = {}
    surface_payload: dict[str, Any] | None = None
    for variant in VARIANT_NAMES:
        model = _model(
            variant=variant,
            num_concepts=len(protocol.concept_index),
        ).to(device)
        initial_sha = _state_dict_sha256(model.state_dict())
        if (
            _common_unary_sha256(model)
            != initialization_audit["common_unary_sha256"]
        ):
            raise RuntimeError("Variant does not share the frozen unary initialization.")
        flops_audit = _profile_supported_forward_flops(
            variant=variant,
            num_concepts=len(protocol.concept_index),
            batch=train_batch,
        )
        initial_gradient = _gradient_audit(model, batch=train_batch)
        training = _train_variant(model, batch=train_batch, steps=steps)
        final_gradient = _gradient_audit(model, batch=train_batch)
        final_sha = _state_dict_sha256(model.state_dict())
        prediction = _predict(model, features=protocol.audit, device=device)
        prediction_path = output_dir / f"predictions_{variant}.csv"
        _atomic_csv(prediction_path, prediction)
        semantic_sha = _prediction_semantic_sha256(prediction)
        checkpoint_path = output_dir / f"checkpoint_{variant}.pt"
        _atomic_checkpoint(
            checkpoint_path,
            {
                "variant": variant,
                "model_mode": MODEL_MODES[variant],
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
                "protocol_sha256": protocol.protocol_sha256,
                "expected_commit": expected_commit,
            },
        )
        manifest = {
            "schema_version": 1,
            "dataset": dataset,
            "variant": variant,
            "model_mode": MODEL_MODES[variant],
            "model_seed": MODEL_SEED,
            "expected_commit": expected_commit,
            "protocol_sha256": protocol.protocol_sha256,
            "optimizer_total_rows": len(protocol.optimizer.records),
            "optimizer_eligible_rows": len(train_indices),
            "optimizer_rows_used": len(train_indices),
            "audit_total_rows": len(protocol.audit.records),
            "audit_eligible_rows": int(
                sum(record.eligible for record in protocol.audit.records)
            ),
            "audit_labels_loaded": False,
            "input_hashes": input_hashes,
            "initial_state_sha256": initial_sha,
            "final_state_sha256": final_sha,
            "active_parameter_count": model.active_variant_parameter_count,
            "total_parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "common_unary_sha256": _common_unary_sha256(model),
            "flops_audit": flops_audit,
            "initial_gradient_audit": initial_gradient,
            "final_gradient_audit": final_gradient,
            "training": training,
            "prediction_file": prediction_path.name,
            "prediction_file_sha256": sha256_file(prediction_path),
            "architecture": architecture,
            "prediction_semantic_sha256": semantic_sha,
            "prediction_row_order_sha256": protocol.audit.row_order_sha256,
            "checkpoint_file": checkpoint_path.name,
            "checkpoint_file_sha256": sha256_file(checkpoint_path),
            "snapshot": snapshot,
        }
        _atomic_json(output_dir / f"manifest_{variant}.json", manifest)
        manifests[variant] = manifest
        if variant == "full":
            surface_frame, surface_diagnostic = _surface_diagnostic(
                model,
                device=device,
            )
            surface_path = output_dir / "full_surface.csv"
            _atomic_csv(surface_path, surface_frame)
            surface_diagnostic = _surface_summary(pd.read_csv(surface_path))
            surface_payload = {
                **surface_diagnostic,
                "surface_file": surface_path.name,
                "surface_file_sha256": sha256_file(surface_path),
            }
            _atomic_json(output_dir / "full_surface.json", surface_payload)
        model.cpu()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if surface_payload is None:
        raise RuntimeError("Full surface diagnostic was not produced.")
    semantic_hashes = {
        name: manifests[name]["prediction_semantic_sha256"]
        for name in VARIANT_NAMES
    }
    barrier = {
        "schema_version": 1,
        "dataset": dataset,
        "expected_variants": list(VARIANT_NAMES),
        "all_predictions_present": True,
        "audit_labels_loaded": False,
        "protocol_sha256": protocol.protocol_sha256,
        "prediction_row_order_sha256": protocol.audit.row_order_sha256,
        "prediction_semantic_sha256": semantic_hashes,
        "architecture": architecture,
        "input_hashes": input_hashes,
        "source": protocol.audit_summary,
        "surface": surface_payload,
        "initialization_audit": initialization_audit,
        "snapshot": snapshot,
        "manifest_file_sha256": {
            name: sha256_file(output_dir / f"manifest_{name}.json")
            for name in VARIANT_NAMES
        },
    }
    barrier["barrier_sha256"] = hashlib.sha256(
        _canonical_json(barrier).encode("utf-8")
    ).hexdigest()
    _atomic_json(output_dir / "prediction_barrier.json", barrier)
    return barrier


def _assert_outcome_free_metadata(
    *,
    frame: pd.DataFrame,
    expected: pd.DataFrame,
    dataset: str,
    variant: str,
) -> None:
    columns = (
        "row_id",
        "student_id",
        "item_id",
        "q_pair_id",
        "q_count",
        "eligible",
    )
    missing = set(columns) - set(frame.columns)
    if missing or len(frame) != len(expected):
        raise RuntimeError(
            f"Outcome-free metadata shape failed for {dataset}/{variant}."
        )
    if frame["row_id"].astype(str).duplicated().any():
        raise RuntimeError(f"Duplicate prediction row IDs for {dataset}/{variant}.")
    for column in columns[:4]:
        if not np.array_equal(
            frame[column].astype(str).to_numpy(),
            expected[column].astype(str).to_numpy(),
        ):
            raise RuntimeError(
                f"Outcome-free {column} mismatch for {dataset}/{variant}."
            )
    observed_q_count = pd.to_numeric(
        frame["q_count"], errors="raise"
    ).to_numpy(dtype=float)
    expected_q_count = expected["q_count"].to_numpy(dtype=np.int64)
    if (
        not np.isfinite(observed_q_count).all()
        or not np.equal(observed_q_count, np.floor(observed_q_count)).all()
        or not np.array_equal(observed_q_count.astype(np.int64), expected_q_count)
    ):
        raise RuntimeError(f"Outcome-free q_count mismatch for {dataset}/{variant}.")
    observed_eligible_text = frame["eligible"].astype(str).str.lower()
    if not observed_eligible_text.isin({"true", "false", "1", "0"}).all():
        raise RuntimeError(f"Invalid eligible values for {dataset}/{variant}.")
    observed_eligible = observed_eligible_text.isin({"true", "1"}).to_numpy()
    if not np.array_equal(
        observed_eligible,
        expected["eligible"].to_numpy(dtype=bool),
    ):
        raise RuntimeError(f"Outcome-free eligible mismatch for {dataset}/{variant}.")


def _load_and_verify_barrier(
    *,
    output_root: Path,
    dataset: str,
    expected_commit: str,
    require_formal: bool = True,
    expected_protocol_sha256: str | None = None,
    expected_audit: RequirementFeatureSet | None = None,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    directory = output_root / dataset
    barrier_path = directory / "prediction_barrier.json"
    if not barrier_path.is_file():
        raise FileNotFoundError(f"Missing prediction barrier for {dataset}.")
    barrier = json.loads(barrier_path.read_text(encoding="utf-8"))
    expected_hash = barrier.pop("barrier_sha256", None)
    actual_hash = hashlib.sha256(
        _canonical_json(barrier).encode("utf-8")
    ).hexdigest()
    barrier["barrier_sha256"] = expected_hash
    if expected_hash != actual_hash:
        raise RuntimeError(f"Prediction barrier hash mismatch for {dataset}.")
    if (
        barrier.get("dataset") != dataset
        or barrier.get("expected_variants") != list(VARIANT_NAMES)
        or not barrier.get("all_predictions_present")
        or barrier.get("audit_labels_loaded")
    ):
        raise RuntimeError(f"Invalid pre-reveal barrier for {dataset}.")
    if barrier["snapshot"].get("head") != expected_commit:
        raise RuntimeError(f"Prediction commit mismatch for {dataset}.")

    if require_formal:
        snapshot = barrier["snapshot"]
        expected_snapshot = {
            "head": expected_commit,
            "branch": EXPECTED_BRANCH,
            "origin_head": expected_commit,
            "environment": "decoupled_cd",
            "worktree_clean": True,
            "formal": True,
        }
        if snapshot != expected_snapshot:
            raise RuntimeError(f"Non-formal prediction artifact for {dataset}.")
        if barrier.get("architecture") != _architecture_payload(steps=STEPS):
            raise RuntimeError(f"Architecture fingerprint mismatch for {dataset}.")
        frozen = FROZEN_SOURCE_SHA256[dataset]
        locked_source = {
            key: barrier.get("source", {}).get(key)
            for key in frozen
        }
        if locked_source != frozen:
            raise RuntimeError(f"Frozen source registry mismatch for {dataset}.")
    elif barrier.get("snapshot", {}).get("formal") is not False:
        raise RuntimeError(f"Expected a non-formal test artifact for {dataset}.")
    predictions = {}
    if require_formal and (
        expected_protocol_sha256 is None or expected_audit is None
    ):
        raise RuntimeError("Formal seal requires the rebuilt outcome-free protocol.")
    if (
        expected_protocol_sha256 is not None
        and barrier.get("protocol_sha256") != expected_protocol_sha256
    ):
        raise RuntimeError(f"Rebuilt protocol mismatch for {dataset}.")
    expected_metadata = (
        expected_audit.metadata_frame() if expected_audit is not None else None
    )
    for variant in VARIANT_NAMES:
        manifest_path = directory / f"manifest_{variant}.json"
        prediction_path = directory / f"predictions_{variant}.csv"
        if not manifest_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(f"Incomplete {dataset}/{variant} prediction.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recipe = manifest.get("training", {})
        if (
            sha256_file(manifest_path)
            != barrier["manifest_file_sha256"][variant]
            or manifest.get("dataset") != dataset
            or manifest.get("variant") != variant
            or manifest.get("model_mode") != MODEL_MODES[variant]
            or manifest.get("model_seed") != MODEL_SEED
            or manifest.get("expected_commit") != expected_commit
            or manifest.get("audit_labels_loaded")
            or manifest.get("protocol_sha256") != barrier["protocol_sha256"]
            or manifest.get("snapshot") != barrier["snapshot"]
            or manifest.get("architecture") != barrier["architecture"]
            or manifest.get("input_hashes") != barrier["input_hashes"]
            or manifest.get("prediction_file_sha256") != sha256_file(prediction_path)
            or manifest.get("prediction_semantic_sha256")
            != barrier["prediction_semantic_sha256"][variant]
            or recipe.get("steps") != barrier["architecture"]["steps"]
            or recipe.get("optimizer") != "Adam"
            or recipe.get("learning_rate") != LEARNING_RATE
            or recipe.get("weight_decay") != WEIGHT_DECAY
            or recipe.get("full_batch") is not True
            or manifest.get("flops_audit", {}).get("supported_forward_flops", 0)
            <= 0
            or manifest.get("initial_gradient_audit", {}).get(
                "every_parameter_tensor_has_nonzero_gradient"
            ) is not True
            or manifest.get("final_gradient_audit", {}).get(
                "every_parameter_tensor_has_nonzero_gradient"
            ) is not True
        ):
            raise RuntimeError(f"Manifest verification failed for {dataset}/{variant}.")
        checkpoint_path = directory / manifest["checkpoint_file"]
        if (
            not checkpoint_path.is_file()
            or sha256_file(checkpoint_path) != manifest["checkpoint_file_sha256"]
        ):
            raise RuntimeError(f"Checkpoint hash failed for {dataset}/{variant}.")
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        state_dict = checkpoint.get("state_dict", {})
        checkpoint_unary = {
            name: value
            for name, value in state_dict.items()
            if name.startswith("unary_lattice.")
        }
        if (
            checkpoint.get("variant") != variant
            or checkpoint.get("model_mode") != MODEL_MODES[variant]
            or checkpoint.get("protocol_sha256") != barrier["protocol_sha256"]
            or checkpoint.get("expected_commit") != expected_commit
            or _state_dict_sha256(state_dict) != manifest["final_state_sha256"]
            or _state_dict_sha256(checkpoint_unary)
            != barrier["initialization_audit"]["common_unary_sha256"]
        ):
            raise RuntimeError(f"Checkpoint verification failed for {dataset}/{variant}.")
        frame = pd.read_csv(prediction_path)
        if _prediction_semantic_sha256(frame) != manifest["prediction_semantic_sha256"]:
            raise RuntimeError(f"Semantic prediction hash failed for {dataset}/{variant}.")
        if expected_metadata is not None:
            _assert_outcome_free_metadata(
                frame=frame,
                expected=expected_metadata,
                dataset=dataset,
                variant=variant,
            )
        if expected_audit is not None:
            replay_model = _model(
                variant=variant,
                num_concepts=len(expected_audit.contexts[0].readiness),
            )
            replay_model.load_state_dict(state_dict, strict=True)
            replayed = _predict(
                replay_model,
                features=expected_audit,
                device=torch.device("cpu"),
            )
            _assert_outcome_free_metadata(
                frame=replayed,
                expected=expected_metadata,
                dataset=dataset,
                variant=f"{variant}:checkpoint_replay",
            )
            maximum_difference = float(
                np.max(np.abs(replayed["prob"].to_numpy() - frame["prob"].to_numpy()))
            )
            if maximum_difference > 2e-6:
                raise RuntimeError(
                    f"Checkpoint replay mismatch for {dataset}/{variant}: "
                    f"{maximum_difference}."
                )
        predictions[variant] = frame
    surface_path = directory / barrier["surface"]["surface_file"]
    if (
        not surface_path.is_file()
        or sha256_file(surface_path) != barrier["surface"]["surface_file_sha256"]
    ):
        raise RuntimeError(f"Full surface artifact failed for {dataset}.")
    recomputed_surface = _surface_summary(pd.read_csv(surface_path))
    if (
        recomputed_surface["noncollapsed"] != barrier["surface"]["noncollapsed"]
        or abs(recomputed_surface["maximum_absolute_mixed_difference"]
               - barrier["surface"]["maximum_absolute_mixed_difference"]) > 1e-9
    ):
        raise RuntimeError(f"Full surface decision mismatch for {dataset}.")

    return barrier, predictions

def seal_predictions(
    *,
    data_root: Path,
    output_root: Path,
    expected_commit: str,
) -> dict[str, Any]:
    snapshot = formal_snapshot(expected_commit)
    locked: dict[str, dict[str, Any]] = {}
    for dataset in DATASET_DIRS:
        train_path, q_path = _dataset_paths(data_root, dataset)
        _assert_frozen_source(
            dataset=dataset,
            train_path=train_path,
            q_path=q_path,
        )
        protocol = _build_protocol(data_root, dataset)
        barrier, _ = _load_and_verify_barrier(
            output_root=output_root,
            dataset=dataset,
            expected_commit=expected_commit,
            require_formal=True,
            expected_protocol_sha256=protocol.protocol_sha256,
            expected_audit=protocol.audit,
        )
        locked[dataset] = barrier
    payload = {
        "schema_version": 1,
        "datasets": {
            dataset: locked[dataset]["barrier_sha256"]
            for dataset in DATASET_DIRS
        },
        "expected_commit": expected_commit,
        "architecture_fingerprint": next(iter(locked.values()))[
            "architecture"
        ]["fingerprint"],
        "snapshot": snapshot,
        "checkpoint_replay_and_protocol_metadata_verified": True,
        "labels_loaded": False,
    }
    payload["global_barrier_sha256"] = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    path = output_root / "global_prediction_barrier.json"
    if path.exists():
        raise FileExistsError(f"Global prediction barrier already exists: {path}")
    _atomic_json(path, payload)
    return payload


def _load_global_barrier(
    *,
    output_root: Path,
    expected_commit: str,
    expected_global_barrier_sha256: str,
) -> dict[str, Any]:
    path = output_root / "global_prediction_barrier.json"
    if not path.is_file():
        raise FileNotFoundError("Seal predictions before revealing labels.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("global_barrier_sha256", None)
    actual = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    payload["global_barrier_sha256"] = recorded
    if recorded != actual or recorded != expected_global_barrier_sha256:
        raise RuntimeError("Global prediction barrier SHA-256 mismatch.")
    if (
        payload.get("schema_version") != 1
        or payload.get("expected_commit") != expected_commit
        or payload.get("labels_loaded") is not False
        or payload.get("checkpoint_replay_and_protocol_metadata_verified")
        is not True
        or set(payload.get("datasets", {})) != set(DATASET_DIRS)
        or payload.get("snapshot") != {
            "head": expected_commit,
            "branch": EXPECTED_BRANCH,
            "origin_head": expected_commit,
            "environment": "decoupled_cd",
            "worktree_clean": True,
            "formal": True,
        }
        or payload.get("architecture_fingerprint")
        != _architecture_payload(steps=STEPS)["fingerprint"]
    ):
        raise RuntimeError("Invalid global pre-reveal barrier.")
    return payload



def _empirical_grid(
    *,
    protocol: TrainOnlyRequirementProtocol,
    aligned: pd.DataFrame,
) -> pd.DataFrame:
    indices = np.arange(len(protocol.audit.records), dtype=np.int64)
    batch = collate_requirement_batch(protocol.audit, indices, device="cpu")
    state = pd.DataFrame(
        {
            "row_id": list(batch.row_ids),
            "b": batch.b.numpy(),
            "g": batch.g.numpy(),
        }
    )
    joined = aligned.merge(state, on="row_id", how="inner", validate="one_to_one")
    joined = joined[joined["eligible"] & (joined["q_count"] == 2)].copy()
    edges = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    joined["b_bin"] = pd.cut(
        joined["b"],
        bins=edges,
        labels=False,
        include_lowest=True,
    )
    joined["g_bin"] = pd.cut(
        joined["g"],
        bins=edges,
        labels=False,
        include_lowest=True,
    )
    columns = ["label", *(f"prob_{name}" for name in VARIANT_NAMES)]
    grouped = joined.groupby(["b_bin", "g_bin"], observed=True)
    summary = grouped[columns].mean().reset_index()
    summary.insert(2, "row_count", grouped.size().to_numpy())
    summary["display_eligible"] = summary["row_count"] >= 30
    return summary


def evaluate_all(
    *,
    data_root: Path,
    output_root: Path,
    expected_commit: str,
    expected_global_barrier_sha256: str,
) -> dict[str, Any]:
    formal_snapshot(expected_commit)
    global_barrier = _load_global_barrier(
        output_root=output_root,
        expected_commit=expected_commit,
        expected_global_barrier_sha256=expected_global_barrier_sha256,
    )
    locked = {}
    protocols = {}
    for dataset in DATASET_DIRS:
        train_path, q_path = _dataset_paths(data_root, dataset)
        _assert_frozen_source(
            dataset=dataset,
            train_path=train_path,
            q_path=q_path,
        )
        protocol = _build_protocol(data_root, dataset)
        protocols[dataset] = protocol
        locked[dataset] = _load_and_verify_barrier(
            output_root=output_root,
            dataset=dataset,
            expected_commit=expected_commit,
            require_formal=True,
            expected_protocol_sha256=protocol.protocol_sha256,
            expected_audit=protocol.audit,
        )
        if (
            locked[dataset][0]["barrier_sha256"]
            != global_barrier["datasets"][dataset]
        ):
            raise RuntimeError(f"Dataset barrier changed after seal for {dataset}.")
    global_barrier_sha = expected_global_barrier_sha256

    summaries = {}
    noncollapse = {}
    for dataset, (barrier, predictions) in locked.items():
        protocol = protocols[dataset]
        if protocol.protocol_sha256 != barrier["protocol_sha256"]:
            raise RuntimeError(f"Protocol changed before reveal for {dataset}.")
        current_source = {
            key: protocol.audit_summary[key]
            for key in FROZEN_SOURCE_SHA256[dataset]
        }
        sealed_source = {
            key: barrier["source"][key]
            for key in FROZEN_SOURCE_SHA256[dataset]
        }
        if current_source != sealed_source:
            raise RuntimeError(f"Source changed after prediction for {dataset}.")
        labels = load_audit_query_labels(
            protocol,
            expected_train_sha256=barrier["source"]["train_full_sha256"],
        )
        aligned = align_prediction_manifests(
            predictions=predictions,
            labels=labels,
        )
        summary = evaluate_requirement_surface(
            aligned,
            dataset=dataset,
            expected_row_order_sha256=barrier["prediction_row_order_sha256"],
        )
        summary["global_prediction_barrier_sha256"] = global_barrier_sha
        summary["prediction_barrier_sha256"] = barrier["barrier_sha256"]
        summary["surface"] = barrier["surface"]
        summaries[dataset] = summary
        noncollapse[dataset] = bool(barrier["surface"]["noncollapsed"])
        _atomic_json(output_root / dataset / "evaluation.json", summary)
        _atomic_csv(output_root / dataset / "aligned_revealed.csv", aligned)
        _atomic_csv(
            output_root / dataset / "empirical_grid.csv",
            _empirical_grid(protocol=protocol, aligned=aligned),
        )

    decision = aggregate_stage1_gate(
        summaries,
        surface_noncollapse=noncollapse,
    )
    decision["global_prediction_barrier_sha256"] = global_barrier_sha
    decision["expected_commit"] = expected_commit
    decision["test_or_validation_opened"] = False
    decision["model_seed"] = MODEL_SEED
    _atomic_json(output_root / "stage1_decision.json", decision)
    return decision


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the preregistered train-only monotone requirement screen."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    predict = subparsers.add_parser("predict-dataset")
    predict.add_argument("--dataset", choices=tuple(DATASET_DIRS), required=True)
    predict.add_argument("--data-root", type=Path, required=True)
    predict.add_argument("--output-root", type=Path, required=True)
    predict.add_argument("--expected-commit", required=True)
    predict.add_argument("--device", default="cpu")
    evaluate = subparsers.add_parser("evaluate")
    seal = subparsers.add_parser("seal")
    seal.add_argument("--data-root", type=Path, required=True)
    seal.add_argument("--output-root", type=Path, required=True)
    seal.add_argument("--expected-commit", required=True)
    evaluate.add_argument("--data-root", type=Path, required=True)
    evaluate.add_argument("--output-root", type=Path, required=True)
    evaluate.add_argument("--expected-commit", required=True)
    evaluate.add_argument(
        "--expected-global-barrier-sha256",
        required=True,
    )

    return parser

def main() -> None:
    args = _parser().parse_args()
    if args.command == "predict-dataset":
        barrier = predict_dataset(
            dataset=args.dataset,
            data_root=args.data_root,
            output_root=args.output_root,
            expected_commit=args.expected_commit,
            device=torch.device(args.device),
        )
        print(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "barrier_sha256": barrier["barrier_sha256"],
                    "labels_loaded": False,
                },
                ensure_ascii=False,
            )
        )
    elif args.command == "seal":
        seal = seal_predictions(
            data_root=args.data_root,
            output_root=args.output_root,
            expected_commit=args.expected_commit,
        )
        print(
            json.dumps(
                seal,
                ensure_ascii=False,
            )
        )
    else:
        decision = evaluate_all(
            data_root=args.data_root,
            output_root=args.output_root,
            expected_commit=args.expected_commit,
            expected_global_barrier_sha256=args.expected_global_barrier_sha256,
        )
        print(json.dumps(decision, ensure_ascii=False))


if __name__ == "__main__":
    main()
