from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.pool_protocol import sha256_file
from data.response_concept_credit_features import (
    CreditFeatureBuildResult,
    CreditFeatureSet,
    CreditStudentBatch,
    TARGET_SCOPE_BY_DATASET,
    build_credit_feature_sets,
    collate_credit_student_batch,
)
from models.response_concept_credit_probe import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_ROUTING_ITERATIONS,
    DEFAULT_STATE_DIM,
    VARIANTS,
    ResponseConceptCreditProbe,
)
from scripts.audit_target_local_pairing_protocol import (
    MODEL_SEED,
    SPLIT_SEED,
    build_optimizer_profiles,
    finalize_protocol_for_retained_optimizer,
    join_validation_predictions_with_labels,
    load_validation_labels_for_evaluation,
    load_validation_only_protocol,
    validation_row_order_sha256,
)
from utils.response_credit_evaluation import (
    BOOTSTRAP_REPLICATES,
    MIN_VALID_BOOTSTRAP_REPLICATES,
    PROBABILITY_COLUMNS,
    compute_stage1_gate,
    compute_stage2_gate,
    evaluate_credit_predictions,
)


EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_NAMESPACE = "response-credit-student-epoch-batches"
EXPECTED_DATASETS = tuple(TARGET_SCOPE_BY_DATASET)
EXPECTED_SPLITS = ("holdout", "standard")
LIVE_ORIGIN_ATTEMPTS = 3
LIVE_ORIGIN_BACKOFF_SECONDS = (0.5, 1.0)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_values(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def _hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _prediction_semantic_sha256(
    frame: pd.DataFrame,
    *,
    variant: str,
) -> str:
    digest = hashlib.sha256()
    probability = PROBABILITY_COLUMNS[variant]
    for row in frame.itertuples(index=False):
        digest.update(str(row.source_row_id).encode("utf-8"))
        digest.update(np.asarray([getattr(row, probability)], dtype="<f4").tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, state: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(
        {name: value.detach().cpu() for name, value in state.items()},
        temporary,
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _require_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _git_snapshot(
    *,
    formal: bool,
    expected_commit: str | None,
) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    remote_output = subprocess.run(
        ["git", "branch", "-r", "--contains", head],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    origin_refs = sorted(
        line.strip()
        for line in remote_output.splitlines()
        if line.strip().startswith("origin/")
    )
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    live_origin_head = None
    if formal:
        if expected_commit is None:
            raise RuntimeError("Formal prediction requires --expected-commit.")
        resolved = subprocess.run(
            ["git", "rev-parse", expected_commit],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if resolved != head:
            raise RuntimeError(
                f"HEAD {head} does not match expected commit {resolved}."
            )
        if status.strip():
            raise RuntimeError("Formal prediction requires a clean worktree.")
        if not origin_refs:
            raise RuntimeError(
                "Formal prediction requires HEAD on an origin tracking ref."
            )
        if not branch:
            raise RuntimeError("Formal prediction requires a named git branch.")
        live_origin_head = _live_origin_branch_head(branch)
        if live_origin_head != head:
            raise RuntimeError(
                "Formal prediction requires the current branch HEAD to equal "
                f"the live origin branch: local={head}, origin={live_origin_head}."
            )
    return {
        "head": head,
        "branch": branch,
        "worktree_clean": not bool(status.strip()),
        "origin_remote_tracking_refs_containing_head": origin_refs,
        "live_origin_branch_head": live_origin_head,
        "formal_enforced": formal,
    }


def _live_origin_branch_head(branch: str) -> str:
    """Return the exact commit currently advertised by origin for ``branch``."""
    if not branch or branch.startswith("-"):
        raise ValueError("A valid branch name is required for the origin check.")
    reference = f"refs/heads/{branch}"
    command = ["git", "ls-remote", "--heads", "origin", reference]
    last_returncode: int | None = None
    last_stderr = ""
    for attempt in range(LIVE_ORIGIN_ATTEMPTS):
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        last_returncode = int(result.returncode)
        last_stderr = result.stderr.strip()
        if result.returncode == 0:
            lines = [
                line.split() for line in result.stdout.splitlines() if line.strip()
            ]
            matches = [
                parts[0]
                for parts in lines
                if len(parts) == 2 and parts[1] == reference
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "Expected exactly one live origin ref for "
                    f"{reference}; found {len(matches)}. stderr={last_stderr!r}"
                )
            return matches[0]
        if attempt < LIVE_ORIGIN_ATTEMPTS - 1:
            time.sleep(LIVE_ORIGIN_BACKOFF_SECONDS[attempt])
    raise RuntimeError(
        "Live origin query failed after "
        f"{LIVE_ORIGIN_ATTEMPTS} attempts for {reference}: "
        f"returncode={last_returncode}, stderr={last_stderr!r}"
    )


def _seed_everything() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(MODEL_SEED)
    torch.manual_seed(MODEL_SEED)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MODEL_SEED)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _runtime_environment(device: torch.device) -> dict[str, Any]:
    """Capture the execution environment needed to reproduce a formal job."""
    resolved_index: int | None = None
    device_name: str | None = None
    device_capability: list[int] | None = None
    device_total_memory: int | None = None
    if device.type == "cuda":
        resolved_index = (
            torch.cuda.current_device() if device.index is None else int(device.index)
        )
        properties = torch.cuda.get_device_properties(resolved_index)
        device_name = properties.name
        device_capability = list(torch.cuda.get_device_capability(resolved_index))
        device_total_memory = int(properties.total_memory)
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": (
            None
            if not hasattr(torch.backends, "cudnn")
            else torch.backends.cudnn.version()
        ),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "cuda_available": torch.cuda.is_available(),
        "requested_device": str(device),
        "resolved_cuda_index": resolved_index,
        "cuda_device_name": device_name,
        "cuda_device_capability": device_capability,
        "cuda_device_total_memory_bytes": device_total_memory,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
    }


@dataclass(frozen=True)
class StudentBatchPlan:
    permutations: np.ndarray
    batch_size: int
    namespace: str
    sha256: str

    def epoch_batches(self, epoch: int) -> tuple[np.ndarray, ...]:
        values = self.permutations[epoch]
        return tuple(
            values[start : start + self.batch_size]
            for start in range(0, len(values), self.batch_size)
        )


def build_student_batch_plan(
    student_count: int,
    *,
    dataset: str,
    split_kind: str,
    epochs: int,
    batch_size: int = BATCH_SIZE,
) -> StudentBatchPlan:
    if student_count < 1 or epochs < 1 or batch_size != BATCH_SIZE:
        raise ValueError("Invalid student batch-plan dimensions.")
    seed_bytes = hashlib.sha256(
        f"{MODEL_SEED}\x1f{BATCH_NAMESPACE}\x1f{dataset}\x1f{split_kind}".encode(
            "utf-8"
        )
    ).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(seed_bytes, "little"))
    permutations = np.stack(
        [rng.permutation(student_count) for _ in range(epochs)]
    ).astype(np.int64)
    sha256 = _hash_values(
        (
            BATCH_NAMESPACE,
            MODEL_SEED,
            dataset,
            split_kind,
            epochs,
            batch_size,
            student_count,
            _hash_array(permutations),
        )
    )
    return StudentBatchPlan(permutations, batch_size, BATCH_NAMESPACE, sha256)


def _model_kwargs(features: CreditFeatureSet) -> dict[str, int]:
    return {
        "num_items": features.num_items,
        "num_concepts": features.num_concepts,
        "max_q_cardinality": features.max_q_cardinality,
        "embedding_dim": DEFAULT_EMBEDDING_DIM,
        "state_dim": DEFAULT_STATE_DIM,
        "hidden_dim": DEFAULT_HIDDEN_DIM,
        "routing_iterations": DEFAULT_ROUTING_ITERATIONS,
    }


def initialize_model(
    features: CreditFeatureSet,
    *,
    variant: str,
) -> tuple[ResponseConceptCreditProbe, dict[str, Any]]:
    if variant not in VARIANTS:
        raise ValueError(f"Unexpected variant: {variant}.")
    _seed_everything()
    model = ResponseConceptCreditProbe(**_model_kwargs(features))
    state_hash = _state_dict_sha256(model.state_dict())
    parameter_schema = tuple(
        (name, tuple(parameter.shape))
        for name, parameter in model.named_parameters()
    )
    topology = {
        "family": "response_concept_credit_activation_probe_v1",
        "embedding_dim": DEFAULT_EMBEDDING_DIM,
        "state_dim": DEFAULT_STATE_DIM,
        "hidden_dim": DEFAULT_HIDDEN_DIM,
        "routing_iterations": DEFAULT_ROUTING_ITERATIONS,
        "model_source_sha256": sha256_file(
            PROJECT_ROOT / "models/response_concept_credit_probe.py"
        ),
        "feature_source_sha256": sha256_file(
            PROJECT_ROOT / "data/response_concept_credit_features.py"
        ),
        "runner_source_sha256": sha256_file(
            PROJECT_ROOT / "scripts/run_response_credit_activation.py"
        ),
        "student_id_embedding": False,
        "diagnosis_student_input": "framework_state_only",
        "loss": "response_bce_only",
    }
    topology_sha256 = hashlib.sha256(
        _canonical_json(topology).encode("utf-8")
    ).hexdigest()
    instance_dimensions = {
        key: value
        for key, value in _model_kwargs(features).items()
        if key in {"num_items", "num_concepts", "max_q_cardinality"}
    }
    instance_sha256 = hashlib.sha256(
        _canonical_json(
            {
                "topology_sha256": topology_sha256,
                "instance_dimensions": instance_dimensions,
            }
        ).encode("utf-8")
    ).hexdigest()
    architecture = {
        **topology,
        "topology_sha256": topology_sha256,
        "instance_dimensions": instance_dimensions,
        "instance_sha256": instance_sha256,
    }
    return model, {
        "variant": variant,
        "initialization_sha256": state_hash,
        "common_initialization_sha256": state_hash,
        "total_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "parameter_schema_sha256": _hash_values(parameter_schema),
        "architecture": architecture,
        "variant_architecture_fingerprint": model.architecture_fingerprint(
            variant
        ),
    }


def _forward(
    model: ResponseConceptCreditProbe,
    batch: CreditStudentBatch,
    *,
    variant: str,
) -> Any:
    return model(
        variant=variant,
        support_item_ids=batch.support_item_ids,
        support_q_indices=batch.support_q_indices,
        support_q_mask=batch.support_q_mask,
        support_responses=batch.support_responses,
        support_item_ease=batch.support_item_ease,
        support_item_confidence=batch.support_item_confidence,
        support_group_attempt_confidence=(
            batch.support_group_attempt_confidence
        ),
        support_mask=batch.support_mask,
        query_context_indices=batch.query_context_indices,
        target_q_indices=batch.target_q_indices,
        target_q_mask=batch.target_q_mask,
    )


def gradient_audit(
    model: ResponseConceptCreditProbe,
    *,
    variant: str,
    features: CreditFeatureSet,
    device: torch.device,
) -> dict[str, Any]:
    indices = np.arange(min(8, len(features.contexts)), dtype=np.int64)
    batch = collate_credit_student_batch(features, indices, device=device)
    if batch.labels is None:
        raise RuntimeError("Gradient audit requires optimizer labels.")
    model.to(device)
    model.train()
    model.zero_grad(set_to_none=True)
    output = _forward(model, batch, variant=variant)
    loss = F.binary_cross_entropy_with_logits(output.logits, batch.labels)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Gradient-audit loss is non-finite.")
    loss.backward()
    finite = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            finite[name] = None
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"Non-finite gradient: {name}.")
        finite[name] = float(parameter.grad.norm().item())
    if finite.get("diagnosis_head.2.weight") in {None, 0.0}:
        raise RuntimeError("Diagnosis path has no finite learning signal.")
    if variant in {"full", "capacity"} and finite.get(
        "routing_output.weight"
    ) in {None, 0.0}:
        raise RuntimeError("Learned routing output has no finite learning signal.")
    model.zero_grad(set_to_none=True)
    model.cpu()
    return {
        "loss": float(loss.item()),
        "finite_gradient_parameters": sum(value is not None for value in finite.values()),
        "none_gradient_parameters": [
            name for name, value in finite.items() if value is None
        ],
        "zero_gradient_parameters": [
            name for name, value in finite.items() if value == 0.0
        ],
    }


def train_variant(
    model: ResponseConceptCreditProbe,
    *,
    variant: str,
    features: CreditFeatureSet,
    plan: StudentBatchPlan,
    device: torch.device,
    epochs: int,
) -> dict[str, Any]:
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    epoch_losses = []
    epoch_query_rows = []
    for epoch in range(epochs):
        total_loss = 0.0
        total_rows = 0
        for context_indices in plan.epoch_batches(epoch):
            batch = collate_credit_student_batch(
                features,
                context_indices,
                device=device,
            )
            if batch.labels is None:
                raise RuntimeError("Optimizer feature set is missing labels.")
            optimizer.zero_grad(set_to_none=True)
            output = _forward(model, batch, variant=variant)
            per_row = F.binary_cross_entropy_with_logits(
                output.logits,
                batch.labels,
                reduction="none",
            )
            loss = per_row.sum() / len(per_row)
            loss.backward()
            optimizer.step()
            total_loss += float(per_row.detach().sum().item())
            total_rows += len(per_row)
        epoch_losses.append(total_loss / total_rows)
        epoch_query_rows.append(total_rows)
    model.cpu()
    return {
        "epochs": epochs,
        "loss_reduction": "mean_over_query_rows_in_each_student_batch",
        "reported_epoch_loss": "sum_query_bce_divided_by_epoch_query_rows",
        "epoch_losses": epoch_losses,
        "epoch_query_rows": epoch_query_rows,
        "final_epoch_loss": epoch_losses[-1],
        "final_state_sha256": _state_dict_sha256(model.state_dict()),
    }


@torch.no_grad()
def predict_variant(
    model: ResponseConceptCreditProbe,
    *,
    variant: str,
    features: CreditFeatureSet,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    model.to(device)
    model.eval()
    probabilities: list[np.ndarray] = []
    source_row_ids: list[str] = []
    students: list[str] = []
    exercises: list[str] = []
    in_c: list[np.ndarray] = []
    in_c_strict: list[np.ndarray] = []
    in_t: list[np.ndarray] = []
    target_coverage: list[np.ndarray] = []
    student_entropy: list[np.ndarray] = []
    student_mass_error: list[np.ndarray] = []
    student_observed: list[np.ndarray] = []
    for start in range(0, len(features.contexts), BATCH_SIZE):
        context_indices = np.arange(
            start,
            min(start + BATCH_SIZE, len(features.contexts)),
        )
        batch = collate_credit_student_batch(
            features,
            context_indices,
            device=device,
        )
        output = _forward(model, batch, variant=variant)
        probabilities.append(output.probs.detach().cpu().numpy())
        source_row_ids.extend(batch.source_row_ids)
        students.extend(batch.students)
        exercises.extend(batch.exercises)
        in_c.append(batch.in_c)
        in_c_strict.append(batch.in_c_strict)
        in_t.append(batch.in_t)
        target_coverage.append(batch.target_coverage)

        valid = batch.support_mask
        entropy = output.routing_entropy
        mass_error = (
            output.routed_mass
            - batch.support_q_mask.sum(dim=-1).to(output.routed_mass.dtype)
        ).abs()
        valid_count = valid.sum(dim=1).clamp_min(1)
        student_entropy.append(
            (
                (entropy * valid).sum(dim=1)
                / valid_count.to(entropy.dtype)
            ).detach().cpu().numpy()
        )
        student_mass_error.append(
            (
                (mass_error * valid).sum(dim=1)
                / valid_count.to(mass_error.dtype)
            ).detach().cpu().numpy()
        )
        student_observed.append(
            output.module_diagnostics["observed_concept_fraction"]
            .detach()
            .cpu()
            .numpy()
        )
    model.cpu()
    probs = np.concatenate(probabilities).astype(np.float32)
    if source_row_ids != [record.source_row_id for record in features.records]:
        raise RuntimeError("Student-batch prediction order differs from feature order.")
    if not np.isfinite(probs).all() or len(probs) != len(features.records):
        raise RuntimeError("Prediction vector is invalid.")
    frame = pd.DataFrame(
        {
            "source_row_id": source_row_ids,
            "stu_id": students,
            "exer_id": exercises,
            "target_coverage": np.concatenate(target_coverage),
            "in_c": np.concatenate(in_c),
            "in_c_strict": np.concatenate(in_c_strict),
            "in_t": np.concatenate(in_t),
            PROBABILITY_COLUMNS[variant]: probs,
        }
    )
    return frame, {
        "mean_routing_entropy": float(np.concatenate(student_entropy).mean()),
        "maximum_mean_routed_mass_error": float(
            np.concatenate(student_mass_error).max()
        ),
        "mean_observed_concept_fraction": float(
            np.concatenate(student_observed).mean()
        ),
        "students": len(features.contexts),
        "query_rows": len(features.records),
    }


def build_features(
    *,
    dataset: str,
    source_dir: Path,
) -> tuple[Any, CreditFeatureBuildResult, dict[str, Any]]:
    protocol = load_validation_only_protocol(dataset, source_dir)
    profiles, optimizer_audit = build_optimizer_profiles(
        protocol.optimizer_train,
        q_lookup=protocol.q_lookup,
    )
    protocol = finalize_protocol_for_retained_optimizer(protocol, profiles)
    features = build_credit_feature_sets(
        protocol,
        optimizer_profiles=profiles,
    )
    return protocol, features, optimizer_audit


def run_audit_phase(
    *,
    dataset: str,
    source_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    _require_empty_output(output_dir)
    protocol, features, optimizer_audit = build_features(
        dataset=dataset,
        source_dir=source_dir,
    )
    payload = {
        "schema_version": 1,
        "phase": "response_credit_audit",
        "dataset": dataset,
        "source": protocol.audit,
        "optimizer": optimizer_audit,
        "features": features.audit,
        "leakage_audit": {
            "validation_labels_loaded": False,
            "test_files_opened": False,
            "opened_source_files": ["train.csv", "valid.csv", "Q_matrix.csv"],
        },
    }
    _atomic_json(output_dir / "audit.json", payload)
    return payload


def load_standard_stage1_barrier(
    *,
    formal: bool,
    split_kind: str,
    stage1_json: Path | None,
    git_head: str,
) -> dict[str, Any] | None:
    if not formal or split_kind != "standard":
        return None
    if stage1_json is None:
        raise RuntimeError(
            "Formal standard prediction requires a passed --stage1-json."
        )
    return validate_stage1_decision_artifact(
        stage1_json,
        expected_commit=git_head,
    )


def run_predict_phase(
    *,
    dataset: str,
    split_kind: str,
    variant: str,
    source_dir: Path,
    output_dir: Path,
    device_name: str,
    epochs: int,
    nonformal: bool,
    expected_commit: str | None,
    stage1_json: Path | None,
) -> dict[str, Any]:
    if dataset not in EXPECTED_DATASETS or split_kind not in EXPECTED_SPLITS:
        raise ValueError("Unexpected dataset or split kind.")
    if variant not in VARIANTS:
        raise ValueError("Unexpected variant.")
    if epochs != EPOCHS and not nonformal:
        raise RuntimeError("Non-20-epoch prediction requires --nonformal.")
    formal = not nonformal
    if formal and epochs != EPOCHS:
        raise RuntimeError("Formal prediction must use exactly 20 epochs.")
    git = _git_snapshot(formal=formal, expected_commit=expected_commit)
    stage1_decision = load_standard_stage1_barrier(
        formal=formal,
        split_kind=split_kind,
        stage1_json=stage1_json,
        git_head=git["head"],
    )
    _require_empty_output(output_dir)
    protocol, features, optimizer_audit = build_features(
        dataset=dataset,
        source_dir=source_dir,
    )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    plan = build_student_batch_plan(
        len(features.optimizer.contexts),
        dataset=dataset,
        split_kind=split_kind,
        epochs=epochs,
    )
    plan_path = output_dir / "student_batch_plan.npz"
    _atomic_npz(
        plan_path,
        permutations=plan.permutations,
        batch_size=np.asarray([plan.batch_size], dtype=np.int64),
    )
    model, model_audit = initialize_model(features.optimizer, variant=variant)
    environment = _runtime_environment(device)
    if stage1_decision is not None and (
        stage1_decision.get("architecture_topology_sha256")
        != model_audit["architecture"]["topology_sha256"]
    ):
        raise RuntimeError("Stage 1 architecture differs from the standard model.")
    gradients = gradient_audit(
        model,
        variant=variant,
        features=features.optimizer,
        device=device,
    )
    model_audit["direct_unused_routing_parameters"] = (
        gradients["none_gradient_parameters"]
        if variant == "direct"
        else []
    )
    training = train_variant(
        model,
        variant=variant,
        features=features.optimizer,
        plan=plan,
        device=device,
        epochs=epochs,
    )
    checkpoint_path = output_dir / f"{variant}_epoch_{epochs}.pt"
    _atomic_checkpoint(checkpoint_path, model.state_dict())
    predictions, diagnostics = predict_variant(
        model,
        variant=variant,
        features=features.validation,
        device=device,
    )
    if "label" in predictions:
        raise RuntimeError("Prediction artifact must be label free.")
    order_hash = validation_row_order_sha256(predictions)
    if order_hash != features.validation.row_order_sha256:
        raise RuntimeError("Prediction row order differs from validation features.")
    prediction_path = output_dir / f"validation_predictions_{variant}_unlabeled.csv"
    _atomic_csv(prediction_path, predictions)
    prediction_semantic_hash = _prediction_semantic_sha256(
        predictions,
        variant=variant,
    )
    manifest = {
        "schema_version": 1,
        "phase": "response_credit_predict",
        "formal": formal,
        "nonformal_reason": (
            None if formal else f"explicit_nonformal_epochs_{epochs}"
        ),
        "dataset": dataset,
        "split_kind": split_kind,
        "stage1_barrier": (
            None
            if stage1_decision is None
            else {
                "path": str(stage1_json.resolve()),
                "sha256": sha256_file(stage1_json),
                "stage1_passed": True,
                "git_commit": stage1_decision["git_commit"],
            }
        ),
        "variant": variant,
        "git": git,
        "environment": environment,
        "source": protocol.audit,
        "optimizer_protocol": optimizer_audit,
        "features": features.audit,
        "optimization": {
            "model_seed": MODEL_SEED,
            "split_seed": SPLIT_SEED,
            "epochs": epochs,
            "student_batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "AdamW",
            "scheduler": None,
            "early_stopping": False,
            "checkpoint_selection": f"epoch_{epochs}_only",
            "loss": "query_row_mean_response_bce",
            "batch_namespace": plan.namespace,
            "batch_plan_sha256": plan.sha256,
            "batch_plan_file_sha256": sha256_file(plan_path),
        },
        "model": model_audit,
        "gradient_audit": gradients,
        "training": training,
        "routing_diagnostics": diagnostics,
        "artifacts": {
            "predictions": {
                "path": prediction_path.name,
                "sha256": sha256_file(prediction_path),
                "semantic_sha256": prediction_semantic_hash,
                "row_order_sha256": order_hash,
                "rows": len(predictions),
            },
            "checkpoint": {
                "path": checkpoint_path.name,
                "sha256": sha256_file(checkpoint_path),
                "state_sha256": training["final_state_sha256"],
            },
            "student_batch_plan": {
                "path": plan_path.name,
                "sha256": sha256_file(plan_path),
            },
        },
        "leakage_audit": {
            "validation_labels_loaded": False,
            "prediction_artifact_contains_label": False,
            "test_files_opened": False,
            "opened_source_files": ["train.csv", "valid.csv", "Q_matrix.csv"],
        },
    }
    _atomic_json(output_dir / "prediction_manifest.json", manifest)
    return manifest


def _inspect_prediction_csv(
    directory: Path,
    *,
    variant: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Validate one unlabeled CSV without loading any outcome or source file."""
    relative_path = Path(str(spec.get("path", "")))
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise RuntimeError(f"Prediction path escapes its artifact directory: {directory}.")
    prediction_path = (directory / relative_path).resolve()
    directory_resolved = directory.resolve()
    if prediction_path.parent != directory_resolved or not prediction_path.is_file():
        raise RuntimeError(f"Prediction artifact is missing for {variant}: {prediction_path}.")
    actual_sha256 = sha256_file(prediction_path)
    if actual_sha256 != spec.get("sha256"):
        raise RuntimeError(f"Prediction SHA mismatch for {variant}.")

    with prediction_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise RuntimeError(f"Prediction CSV is empty for {variant}.") from error
        if not header or len(header) != len(set(header)):
            raise RuntimeError(f"Prediction CSV header is invalid for {variant}.")
        if any(column.strip().casefold() == "label" for column in header):
            raise RuntimeError(f"Prediction CSV contains a label column for {variant}.")
        required = {
            "source_row_id",
            "stu_id",
            "exer_id",
            "target_coverage",
            "in_c",
            "in_c_strict",
            "in_t",
            PROBABILITY_COLUMNS[variant],
        }
        if set(header) != required:
            raise RuntimeError(f"Prediction CSV lacks required columns for {variant}.")
        rows = 0
        for row in reader:
            if len(row) != len(header):
                raise RuntimeError(f"Prediction CSV has a malformed row for {variant}.")
            rows += 1
    if rows != int(spec.get("rows", -1)):
        raise RuntimeError(f"Prediction row count mismatch for {variant}.")
    return {
        "path": str(prediction_path),
        "sha256": actual_sha256,
        "rows": rows,
        "header": header,
        "row_order_sha256": spec.get("row_order_sha256"),
        "semantic_sha256": spec.get("semantic_sha256"),
    }


def run_prediction_barrier_phase(
    *,
    prediction_dirs: Sequence[Path],
    split_kind: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Prove that all six formal, label-free predictions exist before evaluation."""
    if split_kind not in EXPECTED_SPLITS:
        raise ValueError(f"Unexpected split kind: {split_kind}.")
    expected_pairs = {
        (dataset, variant)
        for dataset in EXPECTED_DATASETS
        for variant in VARIANTS
    }
    if len(prediction_dirs) != len(expected_pairs):
        raise RuntimeError(
            f"Prediction barrier requires exactly {len(expected_pairs)} directories."
        )

    inspected: dict[tuple[str, str], dict[str, Any]] = {}
    commits: set[str] = set()
    topologies: set[str] = set()
    for raw_directory in prediction_dirs:
        directory = Path(raw_directory).resolve()
        manifest_path = directory / "prediction_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"Prediction manifest is missing: {manifest_path}.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dataset = str(manifest.get("dataset", ""))
        variant = str(manifest.get("variant", ""))
        pair = (dataset, variant)
        if pair not in expected_pairs or pair in inspected:
            raise RuntimeError(f"Unexpected or duplicate prediction pair: {pair}.")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("phase") != "response_credit_predict"
            or manifest.get("formal") is not True
            or manifest.get("split_kind") != split_kind
        ):
            raise RuntimeError(f"Prediction manifest is not formal {split_kind}: {pair}.")

        git = manifest.get("git", {})
        commit = git.get("head")
        if (
            not isinstance(commit, str)
            or not commit
            or git.get("worktree_clean") is not True
            or git.get("formal_enforced") is not True
            or git.get("live_origin_branch_head") != commit
        ):
            raise RuntimeError(f"Prediction git provenance is invalid: {pair}.")
        architecture = manifest.get("model", {}).get("architecture", {})
        topology = architecture.get("topology_sha256")
        if not isinstance(topology, str) or not topology:
            raise RuntimeError(f"Prediction topology is missing: {pair}.")
        leakage = manifest.get("leakage_audit", {})
        if (
            leakage.get("validation_labels_loaded") is not False
            or leakage.get("prediction_artifact_contains_label") is not False
            or leakage.get("test_files_opened") is not False
        ):
            raise RuntimeError(f"Prediction leakage audit failed: {pair}.")
        if manifest.get("optimization", {}).get("epochs") != EPOCHS:
            raise RuntimeError(f"Formal prediction epochs are invalid: {pair}.")

        artifact = _inspect_prediction_csv(
            directory,
            variant=variant,
            spec=manifest.get("artifacts", {}).get("predictions", {}),
        )
        inspected[pair] = {
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "prediction": artifact,
        }
        commits.add(commit)
        topologies.add(topology)

    if set(inspected) != expected_pairs:
        raise RuntimeError("Prediction barrier does not contain the preregistered matrix.")
    if len(commits) != 1 or len(topologies) != 1:
        raise RuntimeError("Predictions do not share one commit and topology.")
    for dataset in EXPECTED_DATASETS:
        dataset_rows = {
            inspected[(dataset, variant)]["prediction"]["rows"]
            for variant in VARIANTS
        }
        dataset_orders = {
            inspected[(dataset, variant)]["prediction"]["row_order_sha256"]
            for variant in VARIANTS
        }
        if len(dataset_rows) != 1 or len(dataset_orders) != 1 or None in dataset_orders:
            raise RuntimeError(f"Prediction rows/order differ across variants for {dataset}.")

    _require_empty_output(output_dir)
    payload = {
        "schema_version": 1,
        "phase": "response_credit_prediction_barrier",
        "formal": True,
        "split_kind": split_kind,
        "git_commit": next(iter(commits)),
        "architecture_topology_sha256": next(iter(topologies)),
        "expected_matrix": {
            "datasets": list(EXPECTED_DATASETS),
            "variants": list(VARIANTS),
            "prediction_count": len(expected_pairs),
        },
        "predictions": {
            dataset: {
                variant: inspected[(dataset, variant)]
                for variant in VARIANTS
            }
            for dataset in EXPECTED_DATASETS
        },
        "checks": {
            "all_six_predictions_present": True,
            "all_formal": True,
            "same_commit": True,
            "same_topology": True,
            "split_matches": True,
            "artifact_hashes_and_rows_match": True,
            "csv_headers_are_label_free": True,
            "prediction_leakage_audits_pass": True,
        },
        "leakage_audit": {
            "validation_labels_loaded": False,
            "source_files_opened": False,
            "test_files_opened": False,
        },
    }
    _atomic_json(output_dir / "prediction_barrier.json", payload)
    return payload


def _load_variant_prediction(
    directory: Path,
    *,
    expected_variant: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest_path = directory / "prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("phase") != "response_credit_predict"
        or manifest.get("variant") != expected_variant
    ):
        raise RuntimeError(f"Unexpected prediction manifest for {expected_variant}.")
    spec = manifest["artifacts"]["predictions"]
    path = directory / spec["path"]
    if sha256_file(path) != spec["sha256"]:
        raise RuntimeError(f"Prediction SHA mismatch for {expected_variant}.")
    frame = pd.read_csv(path)
    probability = PROBABILITY_COLUMNS[expected_variant]
    required = {
        "source_row_id",
        "stu_id",
        "exer_id",
        "target_coverage",
        "in_c",
        "in_c_strict",
        "in_t",
        probability,
    }
    if required - set(frame.columns) or "label" in frame:
        raise RuntimeError(f"Invalid unlabeled artifact for {expected_variant}.")
    if validation_row_order_sha256(frame) != spec["row_order_sha256"]:
        raise RuntimeError(f"Prediction order mismatch for {expected_variant}.")
    semantic_hash = _prediction_semantic_sha256(
        frame,
        variant=expected_variant,
    )
    if semantic_hash != spec["semantic_sha256"]:
        raise RuntimeError(f"Prediction semantic hash mismatch for {expected_variant}.")
    return manifest, frame


def _comparison_signature(manifest: dict[str, Any]) -> str:
    payload = {
        "formal": manifest["formal"],
        "dataset": manifest["dataset"],
        "split_kind": manifest["split_kind"],
        "git": manifest["git"]["head"],
        "source_hashes": manifest["source"]["source_files"],
        "optimizer_protocol": manifest["optimizer_protocol"],
        "features": manifest["features"],
        "optimization": manifest["optimization"],
        "architecture": manifest["model"]["architecture"],
        "initialization": manifest["model"]["common_initialization_sha256"],
        "parameter_count": manifest["model"]["total_parameter_count"],
        "parameter_schema": manifest["model"]["parameter_schema_sha256"],
        "prediction_order": manifest["artifacts"]["predictions"][
            "row_order_sha256"
        ],
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def run_evaluate_phase(
    *,
    full_dir: Path,
    direct_dir: Path,
    capacity_dir: Path,
    source_dir: Path,
    output_dir: Path,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    _require_empty_output(output_dir)
    directories = {
        "full": full_dir,
        "direct": direct_dir,
        "capacity": capacity_dir,
    }
    loaded = {
        variant: _load_variant_prediction(
            directory,
            expected_variant=variant,
        )
        for variant, directory in directories.items()
    }
    manifests = {variant: value[0] for variant, value in loaded.items()}
    signatures = {
        _comparison_signature(manifest) for manifest in manifests.values()
    }
    if len(signatures) != 1:
        raise RuntimeError(
            "Three variants do not share commit/config/data/init/batch/row hashes."
        )
    formal_values = {manifest["formal"] for manifest in manifests.values()}
    if len(formal_values) != 1:
        raise RuntimeError("Variants mix formal and nonformal predictions.")
    formal = next(iter(formal_values))
    if formal and (
        bootstrap_replicates != BOOTSTRAP_REPLICATES
        or any(
            manifest["optimization"]["epochs"] != EPOCHS
            for manifest in manifests.values()
        )
    ):
        raise RuntimeError("Formal evaluation requires 20 epochs and 2000 bootstrap replicates.")

    base = loaded["full"][1].copy()
    identity_columns = [
        "source_row_id",
        "stu_id",
        "exer_id",
        "target_coverage",
        "in_c",
        "in_c_strict",
        "in_t",
    ]
    for variant in ("direct", "capacity"):
        other = loaded[variant][1]
        if not base.loc[:, identity_columns].equals(other.loc[:, identity_columns]):
            raise RuntimeError(f"Prediction identities/slices differ for {variant}.")
        base[PROBABILITY_COLUMNS[variant]] = other[
            PROBABILITY_COLUMNS[variant]
        ].to_numpy()
    full_manifest = manifests["full"]
    expected_source = Path(full_manifest["source"]["source_directory"]).resolve()
    if source_dir.resolve() != expected_source:
        raise RuntimeError("Evaluation source directory differs from prediction source.")
    labels = load_validation_labels_for_evaluation(
        source_dir / "valid.csv",
        feature_rows=base,
        expected_valid_sha256=full_manifest["source"]["source_files"][
            "valid.csv"
        ]["sha256"],
    )
    aligned = join_validation_predictions_with_labels(
        base,
        labels,
        expected_prediction_order_sha256=full_manifest["artifacts"][
            "predictions"
        ]["row_order_sha256"],
    )
    summary, bootstrap = evaluate_credit_predictions(
        aligned,
        dataset=full_manifest["dataset"],
        split_kind=full_manifest["split_kind"],
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=SPLIT_SEED,
    )
    aligned_path = output_dir / "aligned_predictions_with_labels.csv"
    bootstrap_path = output_dir / "joint_c_student_bootstrap.npz"
    _atomic_csv(aligned_path, aligned)
    _atomic_npz(
        bootstrap_path,
        joint_min_delta=bootstrap.joint_min_delta,
        valid=bootstrap.valid,
        invalid_indices=bootstrap.invalid_indices,
        sampled_student_hashes=np.asarray(
            bootstrap.sampled_student_hashes,
            dtype="U64",
        ),
    )
    payload = {
        "schema_version": 1,
        "phase": "response_credit_evaluate",
        "formal": formal,
        "dataset": full_manifest["dataset"],
        "split_kind": full_manifest["split_kind"],
        "git_commit": full_manifest["git"]["head"],
        "architecture_topology_sha256": full_manifest["model"]["architecture"][
            "topology_sha256"
        ],
        "comparison_signature": next(iter(signatures)),
        "evaluation": summary,
        "prediction_manifests": {
            variant: {
                "path": str((directory / "prediction_manifest.json").resolve()),
                "sha256": sha256_file(directory / "prediction_manifest.json"),
            }
            for variant, directory in directories.items()
        },
        "artifacts": {
            "aligned_predictions": {
                "path": aligned_path.name,
                "sha256": sha256_file(aligned_path),
            },
            "joint_c_bootstrap": {
                "path": bootstrap_path.name,
                "sha256": sha256_file(bootstrap_path),
                "sampling_manifest_sha256": (
                    bootstrap.sampling_manifest_sha256
                ),
                "invalid_indices": bootstrap.invalid_indices.tolist(),
                "sampled_student_hashes": list(
                    bootstrap.sampled_student_hashes
                ),
            },
        },
        "leakage_audit": {
            "all_three_prediction_manifests_present_before_labels": True,
            "row_id_join_one_to_one": True,
            "test_files_opened": False,
        },
    }
    _atomic_json(output_dir / "evaluation.json", payload)
    return payload


def _evaluation_input_fingerprints(
    paths: Sequence[Path],
    payloads: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(paths) != len(payloads):
        raise ValueError("Evaluation paths and payloads must have equal length.")
    return [
        {
            "dataset": payload["dataset"],
            "split_kind": payload["split_kind"],
            "formal": payload["formal"],
            "git_commit": payload["git_commit"],
            "architecture_topology_sha256": payload[
                "architecture_topology_sha256"
            ],
            "comparison_signature": payload["comparison_signature"],
            "evaluation_json_path": str(path.resolve()),
            "evaluation_json_sha256": sha256_file(path),
        }
        for path, payload in zip(paths, payloads, strict=True)
    ]


def validate_stage1_decision_artifact(
    stage1_json: Path,
    *,
    expected_commit: str,
    expected_topology: str | None = None,
) -> dict[str, Any]:
    """Validate Stage 1 from its two immutable formal evaluation artifacts."""
    stage1_json = Path(stage1_json)
    decision = json.loads(stage1_json.read_text(encoding="utf-8"))
    if (
        decision.get("schema_version") != 1
        or decision.get("phase") != "response_credit_aggregate"
        or decision.get("aggregate_stage") != "stage1"
        or decision.get("gate") != "response_credit_stage1_holdout"
    ):
        raise RuntimeError("Stage 1 artifact is not an official aggregate schema.")
    if decision.get("stage1_passed") is not True:
        raise RuntimeError("Formal standard prediction requires passed Stage 1.")
    if decision.get("git_commit") != expected_commit:
        raise RuntimeError("Stage 1 decision commit differs from current HEAD.")
    topology = decision.get("architecture_topology_sha256")
    if not isinstance(topology, str) or not topology:
        raise RuntimeError("Stage 1 decision lacks an architecture topology hash.")
    if expected_topology is not None and topology != expected_topology:
        raise RuntimeError("Stage 1 topology differs from current evaluations.")

    datasets = decision.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(EXPECTED_DATASETS):
        raise RuntimeError("Stage 1 must contain both preregistered datasets.")
    checks = decision.get("checks")
    if (
        not isinstance(checks, dict)
        or not checks
        or any(value is not True for value in checks.values())
    ):
        raise RuntimeError("Stage 1 artifact does not contain all passing checks.")

    recomputed = compute_stage1_gate(list(datasets.values()))
    for field in ("stage1_passed", "checks", "control_rule", "datasets"):
        if _canonical_json(decision.get(field)) != _canonical_json(
            recomputed.get(field)
        ):
            raise RuntimeError(
                f"Stage 1 aggregate field {field!r} differs from recomputation."
            )

    fingerprints = decision.get("evaluation_inputs")
    if not isinstance(fingerprints, list) or len(fingerprints) != len(
        EXPECTED_DATASETS
    ):
        raise RuntimeError("Stage 1 requires two evaluation fingerprints.")
    fingerprints_by_dataset = {
        str(value.get("dataset")): value for value in fingerprints
    }
    if set(fingerprints_by_dataset) != set(EXPECTED_DATASETS):
        raise RuntimeError("Stage 1 evaluation fingerprints have wrong datasets.")

    for dataset in EXPECTED_DATASETS:
        fingerprint = fingerprints_by_dataset[dataset]
        if (
            fingerprint.get("formal") is not True
            or fingerprint.get("split_kind") != "holdout"
            or fingerprint.get("git_commit") != expected_commit
            or fingerprint.get("architecture_topology_sha256") != topology
            or not fingerprint.get("comparison_signature")
        ):
            raise RuntimeError(
                f"Stage 1 fingerprint is invalid for {dataset}."
            )
        evaluation_path = Path(str(fingerprint.get("evaluation_json_path", "")))
        if not evaluation_path.is_file() or sha256_file(evaluation_path) != fingerprint.get(
            "evaluation_json_sha256"
        ):
            raise RuntimeError(
                f"Stage 1 evaluation artifact hash mismatch for {dataset}."
            )
        payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != 1
            or payload.get("phase") != "response_credit_evaluate"
            or payload.get("formal") is not True
            or payload.get("dataset") != dataset
            or payload.get("split_kind") != "holdout"
            or payload.get("git_commit") != expected_commit
            or payload.get("architecture_topology_sha256") != topology
            or payload.get("comparison_signature")
            != fingerprint.get("comparison_signature")
            or set(payload.get("prediction_manifests", {})) != set(VARIANTS)
        ):
            raise RuntimeError(
                f"Stage 1 formal holdout evaluation is invalid for {dataset}."
            )
        evaluation = payload.get("evaluation")
        if _canonical_json(evaluation) != _canonical_json(datasets[dataset]):
            raise RuntimeError(
                f"Stage 1 embedded evaluation differs for {dataset}."
            )
        bootstrap = evaluation.get("joint_c_bootstrap", {})
        if (
            bootstrap.get("requested_replicates") != BOOTSTRAP_REPLICATES
            or int(bootstrap.get("valid_replicates", -1))
            < MIN_VALID_BOOTSTRAP_REPLICATES
        ):
            raise RuntimeError(
                f"Stage 1 bootstrap provenance is invalid for {dataset}."
            )
    return decision


def run_aggregate_phase(
    *,
    stage: str,
    evaluation_jsons: Sequence[Path],
    output_dir: Path,
    stage1_json: Path | None,
) -> dict[str, Any]:
    _require_empty_output(output_dir)
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in evaluation_jsons
    ]
    if any(value.get("phase") != "response_credit_evaluate" for value in payloads):
        raise RuntimeError("Aggregate inputs must be credit evaluation artifacts.")
    if any(not value.get("formal") for value in payloads):
        raise RuntimeError("Activation gates reject nonformal evaluations.")
    commits = {value["git_commit"] for value in payloads}
    architectures = {value["architecture_topology_sha256"] for value in payloads}
    if len(commits) != 1 or len(architectures) != 1:
        raise RuntimeError("Gate inputs do not share one commit and architecture.")
    evaluations = [value["evaluation"] for value in payloads]
    evaluation_inputs = _evaluation_input_fingerprints(
        evaluation_jsons,
        payloads,
    )
    if stage == "stage1":
        if stage1_json is not None:
            raise ValueError("Stage 1 must not receive --stage1-json.")
        decision = compute_stage1_gate(evaluations)
    else:
        if stage1_json is None:
            raise ValueError("Stage 2 requires --stage1-json.")
        stage1 = validate_stage1_decision_artifact(
            stage1_json,
            expected_commit=next(iter(commits)),
            expected_topology=next(iter(architectures)),
        )
        if set(stage1["datasets"]) != {
            str(value["dataset"]) for value in evaluations
        }:
            raise RuntimeError(
                "Stage 1 and current standard evaluations have different datasets."
            )
        decision = compute_stage2_gate(evaluations, stage1=stage1)
    decision.update(
        {
            "phase": "response_credit_aggregate",
            "aggregate_stage": stage,
            "git_commit": next(iter(commits)),
            "architecture_topology_sha256": next(iter(architectures)),
            "evaluation_inputs": evaluation_inputs,
            "stage1_artifact": (
                None
                if stage1_json is None
                else {
                    "path": str(stage1_json.resolve()),
                    "sha256": sha256_file(stage1_json),
                }
            ),
            "multi_seed_used": False,
            "bootstrap_is_model_seed": False,
        }
    )
    _atomic_json(output_dir / "activation_decision.json", decision)
    return decision


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True, choices=EXPECTED_DATASETS)
    parser.add_argument("--source-dir", required=True, type=Path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only response-to-concept credit routing audit. "
            "No command accepts a test path."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    _add_source_arguments(audit)
    audit.add_argument("--output-dir", required=True, type=Path)

    predict = subparsers.add_parser("predict")
    _add_source_arguments(predict)
    predict.add_argument("--split-kind", required=True, choices=EXPECTED_SPLITS)
    predict.add_argument("--variant", required=True, choices=VARIANTS)
    predict.add_argument("--output-dir", required=True, type=Path)
    predict.add_argument("--device", required=True)
    predict.add_argument("--epochs", type=int, default=EPOCHS)
    predict.add_argument("--nonformal", action="store_true")
    predict.add_argument("--expected-commit")
    predict.add_argument("--stage1-json", type=Path)

    barrier = subparsers.add_parser("prediction-barrier")
    barrier.add_argument(
        "--prediction-dir",
        required=True,
        action="append",
        type=Path,
    )
    barrier.add_argument("--split-kind", required=True, choices=EXPECTED_SPLITS)
    barrier.add_argument("--output-dir", required=True, type=Path)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--full-dir", required=True, type=Path)
    evaluate.add_argument("--direct-dir", required=True, type=Path)
    evaluate.add_argument("--capacity-dir", required=True, type=Path)
    evaluate.add_argument("--source-dir", required=True, type=Path)
    evaluate.add_argument("--output-dir", required=True, type=Path)
    evaluate.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=BOOTSTRAP_REPLICATES,
    )

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--stage", required=True, choices=("stage1", "stage2"))
    aggregate.add_argument(
        "--evaluation-json",
        required=True,
        action="append",
        type=Path,
    )
    aggregate.add_argument("--stage1-json", type=Path)
    aggregate.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "audit":
        result = run_audit_phase(
            dataset=args.dataset,
            source_dir=args.source_dir,
            output_dir=args.output_dir,
        )
        printable = {
            "dataset": result["dataset"],
            "validation_rows": result["features"]["validation_query_rows"],
            "C_rows": result["features"]["validation_c_rows"],
        }
    elif args.command == "predict":
        result = run_predict_phase(
            dataset=args.dataset,
            split_kind=args.split_kind,
            variant=args.variant,
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            device_name=args.device,
            epochs=args.epochs,
            nonformal=args.nonformal,
            expected_commit=args.expected_commit,
            stage1_json=args.stage1_json,
        )
        printable = {
            "dataset": result["dataset"],
            "split_kind": result["split_kind"],
            "variant": result["variant"],
            "formal": result["formal"],
            "prediction_rows": result["artifacts"]["predictions"]["rows"],
        }
    elif args.command == "prediction-barrier":
        result = run_prediction_barrier_phase(
            prediction_dirs=args.prediction_dir,
            split_kind=args.split_kind,
            output_dir=args.output_dir,
        )
        printable = {
            "phase": result["phase"],
            "split_kind": result["split_kind"],
            "prediction_count": result["expected_matrix"]["prediction_count"],
            "git_commit": result["git_commit"],
            "checks": result["checks"],
        }
    elif args.command == "evaluate":
        result = run_evaluate_phase(
            full_dir=args.full_dir,
            direct_dir=args.direct_dir,
            capacity_dir=args.capacity_dir,
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            bootstrap_replicates=args.bootstrap_replicates,
        )
        printable = {
            "dataset": result["dataset"],
            "split_kind": result["split_kind"],
            "formal": result["formal"],
            "C_delta": result["evaluation"]["control_envelope_deltas"]["C"][
                "auc_vs_control_envelope"
            ],
        }
    else:
        result = run_aggregate_phase(
            stage=args.stage,
            evaluation_jsons=args.evaluation_json,
            output_dir=args.output_dir,
            stage1_json=args.stage1_json,
        )
        printable = {
            "gate": result["gate"],
            "passed": result.get("stage1_passed", result.get("activated")),
        }
    print(json.dumps(printable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
