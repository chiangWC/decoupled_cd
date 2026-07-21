from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.pool_protocol import sha256_file
from data.target_local_pairing_features import (
    ITEM_NUMERIC_NAMES,
    RESPONSE_FEATURE_NAMES,
    SUPPORT_STATISTIC_NAMES,
    FeatureBuildResult,
    PairingBatch,
    PairingFeatureSet,
    build_feature_sets,
    collate_pairing_batch,
    replace_validation_donors,
)
from models.target_local_pairing_probe import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MLP_HIDDEN_DIM,
    DEFAULT_STATE_DIM,
    StrongOPMSProbe,
    TargetLocalPairingProbe,
)
from scripts.audit_target_local_pairing_protocol import (
    INNER_SUPPORT_FRACTION,
    MIN_CHANGED_ROW_FRACTION,
    MIN_MOVABLE_STUDENTS,
    MODEL_SEED,
    NUM_FOLDS,
    OPTIMIZER_FRACTION,
    PERMUTATION_REPLICATES,
    SPLIT_SEED,
    build_donor_mapping,
    build_optimizer_profiles,
    finalize_protocol_for_retained_optimizer,
    join_validation_predictions_with_labels,
    load_validation_labels_for_evaluation,
    load_validation_only_protocol,
    validation_row_order_sha256,
)


EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BOOTSTRAP_REPLICATES = 2_000
BATCH_NAMESPACE = "target-local-pairing-epoch-batches"
EXPECTED_DATASETS = ("ASSIST17", "MOOCRadar", "XES3G5M", "Junyi")
PAIR_VARIANTS = ("real_pair", "late_fusion", "perm_pair")
ALL_VARIANTS = ("real_pair", "strong_opms", "perm_pair", "late_fusion")
PROBABILITY_COLUMNS = {
    "real_pair": "prob_real_pair",
    "strong_opms": "prob_strong_opms",
    "perm_pair": "prob_perm_pair",
    "late_fusion": "prob_late_fusion",
}


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


def _parameter_schema(model: torch.nn.Module) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(
        (name, tuple(parameter.shape))
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def _active_parameter_count(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _common_state_sha256(model: torch.nn.Module) -> str:
    prefixes = ("item_encoder.", "outcome_state.")
    return _state_dict_sha256(
        {
            name: tensor
            for name, tensor in model.state_dict().items()
            if name.startswith(prefixes)
        }
    )


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


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, state: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save({name: value.detach().cpu() for name, value in state.items()}, temporary)
    os.replace(temporary, path)


def _require_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _git_snapshot(expected_commit: str) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != expected_commit:
        raise RuntimeError(f"HEAD {head} does not match expected commit {expected_commit}.")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("Formal prediction requires a clean worktree.")
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
    if not origin_refs:
        raise RuntimeError(
            "HEAD is not contained in any fetched origin remote-tracking branch."
        )
    return {
        "head": head,
        "worktree_clean": True,
        "origin_remote_tracking_refs_containing_head": origin_refs,
    }


def _load_locked_protocol(
    path: Path,
    *,
    dataset: str,
    expected_sha256: str,
) -> tuple[dict[str, Any], str]:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Protocol SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "datasets" in payload:
        matches = [entry for entry in payload["datasets"] if entry["dataset"] == dataset]
        if len(matches) != 1:
            raise RuntimeError("Locked protocol must contain exactly one requested dataset.")
        payload = matches[0]
    if payload.get("dataset") != dataset:
        raise RuntimeError("Locked protocol dataset does not match --dataset.")
    if payload.get("audit") != "target_local_pairing_protocol":
        raise RuntimeError("Unexpected locked protocol audit type.")
    return payload, actual_sha256


def _assert_locked_protocol(
    *,
    locked: dict[str, Any],
    protocol: Any,
    optimizer_audit: dict[str, Any],
) -> None:
    if Path(locked["source"]["source_directory"]).resolve() != protocol.source_dir:
        raise RuntimeError("Protocol source directory differs from the locked audit.")
    if _canonical_json(locked["source"]) != _canonical_json(protocol.audit):
        raise RuntimeError("Rebuilt source protocol differs from the complete lock.")
    if _canonical_json(locked["optimizer"]) != _canonical_json(optimizer_audit):
        raise RuntimeError("Rebuilt optimizer protocol differs from the complete lock.")
    expected_fixed_protocol = {
        "model_seed": MODEL_SEED,
        "split_seed": SPLIT_SEED,
        "optimizer_fraction": OPTIMIZER_FRACTION,
        "inner_support_fraction": INNER_SUPPORT_FRACTION,
        "optimizer_folds": NUM_FOLDS,
        "permutation_replicates": PERMUTATION_REPLICATES,
        "minimum_changed_row_fraction": MIN_CHANGED_ROW_FRACTION,
        "minimum_movable_students": MIN_MOVABLE_STUDENTS,
    }
    if locked["fixed_protocol"] != expected_fixed_protocol:
        raise RuntimeError("Locked protocol fixed constants are unexpected.")


@dataclass(frozen=True)
class BatchPlan:
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


def build_batch_plan(
    row_count: int,
    *,
    dataset: str,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    seed: int = MODEL_SEED,
) -> BatchPlan:
    if row_count < 1 or epochs != EPOCHS or batch_size != BATCH_SIZE or seed != MODEL_SEED:
        raise ValueError("The formal batch plan uses non-empty data and frozen optimization constants.")
    seed_bytes = hashlib.sha256(
        f"{seed}\x1f{BATCH_NAMESPACE}\x1f{dataset}".encode("utf-8")
    ).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(seed_bytes, "little"))
    permutations = np.stack([rng.permutation(row_count) for _ in range(epochs)]).astype(
        np.int64
    )
    sha256 = _hash_values(
        (
            BATCH_NAMESPACE,
            dataset,
            seed,
            epochs,
            batch_size,
            _hash_array(permutations),
        )
    )
    return BatchPlan(permutations, batch_size, BATCH_NAMESPACE, sha256)


def _seed_everything() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(MODEL_SEED)
    torch.manual_seed(MODEL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MODEL_SEED)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _model_kwargs(features: PairingFeatureSet) -> dict[str, int]:
    return {
        "num_items": features.num_items,
        "num_concepts": features.num_concepts,
        "item_numeric_dim": len(ITEM_NUMERIC_NAMES),
        "embedding_dim": DEFAULT_EMBEDDING_DIM,
        "state_dim": DEFAULT_STATE_DIM,
        "mlp_hidden_dim": DEFAULT_MLP_HIDDEN_DIM,
    }


def initialize_models(features: PairingFeatureSet) -> tuple[dict[str, torch.nn.Module], dict[str, Any]]:
    """Create exact-init pair controls and a common-init Strong-OPMS control."""
    kwargs = _model_kwargs(features)
    _seed_everything()
    template = TargetLocalPairingProbe(**kwargs, placement="real_pair")
    pair_state = {name: value.detach().clone() for name, value in template.state_dict().items()}
    models: dict[str, torch.nn.Module] = {}
    for placement in PAIR_VARIANTS:
        model = TargetLocalPairingProbe(**kwargs, placement=placement)
        model.load_state_dict(pair_state, strict=True)
        models[placement] = model

    torch.manual_seed(MODEL_SEED)
    strong = StrongOPMSProbe(**kwargs)
    strong_state = strong.state_dict()
    for name, value in pair_state.items():
        if name.startswith(("item_encoder.", "outcome_state.")):
            strong_state[name] = value.detach().clone()
    strong.load_state_dict(strong_state, strict=True)
    models["strong_opms"] = strong

    pair_schemas = {_parameter_schema(models[name]) for name in PAIR_VARIANTS}
    pair_hashes = {_state_dict_sha256(models[name].state_dict()) for name in PAIR_VARIANTS}
    pair_counts = {_active_parameter_count(models[name]) for name in PAIR_VARIANTS}
    if len(pair_schemas) != 1 or len(pair_hashes) != 1 or len(pair_counts) != 1:
        raise RuntimeError("Pair variants do not have exact parameter/init parity.")
    common_hashes = {_common_state_sha256(model) for model in models.values()}
    if len(common_hashes) != 1:
        raise RuntimeError("Strong and paired controls do not share common initialization.")

    architecture = {
        "family": "target_local_pairing_activation_probe_v1",
        "embedding_dim": DEFAULT_EMBEDDING_DIM,
        "state_dim": DEFAULT_STATE_DIM,
        "mlp_hidden_dim": DEFAULT_MLP_HIDDEN_DIM,
        "item_numeric_names": ITEM_NUMERIC_NAMES,
        "support_statistic_names": SUPPORT_STATISTIC_NAMES,
        "response_feature_names": RESPONSE_FEATURE_NAMES,
        "model_source_sha256": sha256_file(PROJECT_ROOT / "models/target_local_pairing_probe.py"),
        "feature_source_sha256": sha256_file(
            PROJECT_ROOT / "data/target_local_pairing_features.py"
        ),
    }
    architecture["family_sha256"] = hashlib.sha256(
        _canonical_json(architecture).encode("utf-8")
    ).hexdigest()
    audit = {
        "architecture": architecture,
        "pair_initialization_sha256": next(iter(pair_hashes)),
        "common_initialization_sha256": next(iter(common_hashes)),
        "pair_active_parameter_count": next(iter(pair_counts)),
        "strong_active_parameter_count": _active_parameter_count(strong),
        "pair_parameter_difference_percent": 0.0,
        "control_roles": {
            "late_fusion": "exact_raw_input_and_capacity_matched_placement_control",
            "perm_pair": "exact_capacity_independently_trained_donor_target_control",
            "strong_opms": (
                "irreversible_outcome_partition_control_not_raw_input_matched; "
                "continuous_response_residual_and_confidence_do_not_enter_logits"
            ),
        },
        "dataset_parameter_schemas": {
            name: _parameter_schema(model) for name, model in models.items()
        },
    }
    return models, audit


def _common_batch_kwargs(batch: PairingBatch) -> dict[str, torch.Tensor]:
    return {
        "support_item_ids": batch.support_item_ids,
        "support_q_multi_hot": batch.support_q_multi_hot,
        "support_item_numeric": batch.support_item_numeric,
        "support_responses": batch.support_responses,
        "support_item_ease": batch.support_item_ease,
        "support_group_attempt_confidence": batch.support_group_attempt_confidence,
        "support_statistics": batch.support_statistics,
        "support_mask": batch.support_mask,
        "target_item_ids": batch.target_item_ids,
        "target_q_multi_hot": batch.target_q_multi_hot,
        "target_item_numeric": batch.target_item_numeric,
    }


def _forward_variant(
    model: torch.nn.Module,
    batch: PairingBatch,
    *,
    variant: str,
) -> Any:
    kwargs = _common_batch_kwargs(batch)
    if variant == "perm_pair":
        kwargs.update(
            {
                "donor_target_item_ids": batch.donor_target_item_ids,
                "donor_target_q_multi_hot": batch.donor_target_q_multi_hot,
                "donor_target_item_numeric": batch.donor_target_item_numeric,
            }
        )
    return model(**kwargs)


def gradient_audit(
    models: dict[str, torch.nn.Module],
    *,
    features: PairingFeatureSet,
    device: torch.device,
) -> dict[str, Any]:
    indices = np.arange(min(8, len(features.records)), dtype=np.int64)
    batch = collate_pairing_batch(features, indices, device=device)
    if batch.labels is None:
        raise RuntimeError("Gradient audit requires optimizer labels.")
    output: dict[str, Any] = {}
    for variant in ALL_VARIANTS:
        model = models[variant].to(device)
        model.train()
        model.zero_grad(set_to_none=True)
        result = _forward_variant(model, batch, variant=variant)
        loss = F.binary_cross_entropy_with_logits(result.logits, batch.labels)
        loss.backward()
        norms = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError(f"Missing/non-finite gradient for {variant}:{name}.")
            norm = float(parameter.grad.norm().item())
            if norm <= 0.0:
                raise RuntimeError(f"Zero gradient for {variant}:{name}.")
            norms[name] = norm
        output[variant] = {
            "loss": float(loss.item()),
            "parameters_checked": len(norms),
            "minimum_gradient_norm": min(norms.values()),
            "maximum_gradient_norm": max(norms.values()),
        }
        model.zero_grad(set_to_none=True)
        model.cpu()
    return output


def train_variant(
    model: torch.nn.Module,
    *,
    variant: str,
    features: PairingFeatureSet,
    batch_plan: BatchPlan,
    device: torch.device,
) -> dict[str, Any]:
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    epoch_losses = []
    for epoch in range(EPOCHS):
        total_loss = 0.0
        total_rows = 0
        for indices in batch_plan.epoch_batches(epoch):
            batch = collate_pairing_batch(features, indices, device=device)
            if batch.labels is None:
                raise RuntimeError("Optimizer feature set does not contain labels.")
            optimizer.zero_grad(set_to_none=True)
            result = _forward_variant(model, batch, variant=variant)
            loss = F.binary_cross_entropy_with_logits(result.logits, batch.labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(indices)
            total_rows += len(indices)
        epoch_losses.append(total_loss / total_rows)
    model.cpu()
    return {
        "epochs": EPOCHS,
        "epoch_losses": epoch_losses,
        "final_epoch_loss": epoch_losses[-1],
        "final_state_sha256": _state_dict_sha256(model.state_dict()),
    }


@torch.no_grad()
def predict_variant(
    model: torch.nn.Module,
    *,
    variant: str,
    features: PairingFeatureSet,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    model.to(device)
    model.eval()
    probabilities = []
    correct_mass = []
    incorrect_mass = []
    for start in range(0, len(features.records), BATCH_SIZE):
        indices = np.arange(start, min(start + BATCH_SIZE, len(features.records)))
        batch = collate_pairing_batch(features, indices, device=device)
        result = _forward_variant(model, batch, variant=variant)
        probabilities.append(result.probs.detach().cpu().numpy())
        correct_mass.append(result.correct_mass.detach().cpu().numpy())
        incorrect_mass.append(result.incorrect_mass.detach().cpu().numpy())
    model.cpu()
    probs = np.concatenate(probabilities).astype(np.float32)
    if len(probs) != len(features.records) or not np.isfinite(probs).all():
        raise RuntimeError(f"Invalid prediction vector for {variant}.")
    return probs, {
        "mean_correct_mass": float(np.concatenate(correct_mass).mean()),
        "mean_incorrect_mass": float(np.concatenate(incorrect_mass).mean()),
    }


def build_locked_feature_result(
    *,
    dataset: str,
    source_dir: Path,
    locked: dict[str, Any],
) -> tuple[Any, FeatureBuildResult, dict[str, Any]]:
    protocol = load_validation_only_protocol(dataset, source_dir)
    profiles, optimizer_audit = build_optimizer_profiles(
        protocol.optimizer_train,
        q_lookup=protocol.q_lookup,
    )
    protocol = finalize_protocol_for_retained_optimizer(protocol, profiles)
    _assert_locked_protocol(
        locked=locked,
        protocol=protocol,
        optimizer_audit=optimizer_audit,
    )
    expected_mapping = locked["hashes"]["validation_mapping_replicate_zero"]
    result = build_feature_sets(
        protocol,
        optimizer_profiles=profiles,
        expected_validation_mapping_sha256=expected_mapping,
    )
    locked_folds = {int(entry["fold"]): entry for entry in locked["folds"]}
    rebuilt_folds = {int(entry["fold"]): entry for entry in result.audit["folds"]}
    if set(locked_folds) != set(range(NUM_FOLDS)) or set(rebuilt_folds) != set(
        range(NUM_FOLDS)
    ):
        raise RuntimeError("Locked/rebuilt optimizer OOF fold sets are incomplete.")
    for fold in range(NUM_FOLDS):
        locked_fold = locked_folds[fold]
        rebuilt_fold = rebuilt_folds[fold]
        expected_fold = {
            "fold": fold,
            "reference_students": locked_fold["reference_students"],
            "held_students": locked_fold["held_students"],
            "reference_students_sha256": locked_fold[
                "reference_students_sha256"
            ],
            "reference_rows_sha256": locked_fold["reference_rows_sha256"],
            "ease_tercile_edges": locked_fold["ease_tercile_edges"],
            "donor_mapping_sha256": locked_fold["donor"]["mapping_sha256"],
            "zero_count_item_rows_map_to_unk": True,
        }
        if _canonical_json(rebuilt_fold) != _canonical_json(expected_fold):
            raise RuntimeError(f"Rebuilt optimizer OOF fold {fold} differs from lock.")
    if (
        result.audit["optimizer_mapping_replicate_zero_sha256"]
        != locked["hashes"]["optimizer_mapping_replicate_zero"]
    ):
        raise RuntimeError("Aggregate optimizer donor replicate-0 hash differs from lock.")
    validation_reference = locked["validation_reference"]
    if (
        result.validation_statistics.reference_students_hash
        != validation_reference["reference_students_sha256"]
        or result.validation_statistics.reference_rows_hash
        != validation_reference["reference_rows_sha256"]
        or _canonical_json(result.validation_donor_rep0.audit["ease_tercile_edges"])
        != _canonical_json(validation_reference["ease_tercile_edges"])
    ):
        raise RuntimeError("Rebuilt validation reference statistics differ from lock.")
    expected_rows = set(protocol.validation_query["source_row_id"].astype(str))
    feature_rows = {record.source_row_id for record in result.validation.records}
    if feature_rows != expected_rows or len(feature_rows) != len(result.validation.records):
        raise RuntimeError("Validation features do not exactly cover the locked query rows.")
    if result.optimizer.label_sha256 is None or result.validation.label_sha256 is not None:
        raise RuntimeError("Optimizer/validation label separation invariant failed.")
    if len(locked["validation_permutation_audits"]) != PERMUTATION_REPLICATES:
        raise RuntimeError("Locked protocol does not contain exactly 200 donor audits.")
    return protocol, result, optimizer_audit


def _prediction_semantic_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.itertuples(index=False):
        digest.update(str(row.source_row_id).encode("utf-8"))
        digest.update(str(row.stu_id).encode("utf-8"))
        digest.update(str(row.exer_id).encode("utf-8"))
        for column in PROBABILITY_COLUMNS.values():
            # Predictions are generated/stored as float32. Canonicalizing the
            # CSV reload to float32 makes the semantic hash independent of the
            # textual decimal rendering while preserving every model bit.
            digest.update(np.asarray([getattr(row, column)], dtype="<f4").tobytes())
    return digest.hexdigest()


def run_predict_phase(
    *,
    dataset: str,
    source_dir: Path,
    protocol_json: Path,
    expected_protocol_sha256: str,
    expected_commit: str,
    output_dir: Path,
    device_name: str,
) -> dict[str, Any]:
    if dataset not in EXPECTED_DATASETS:
        raise ValueError(f"Dataset must be one of {EXPECTED_DATASETS}.")
    git = _git_snapshot(expected_commit)
    locked, protocol_sha256 = _load_locked_protocol(
        protocol_json,
        dataset=dataset,
        expected_sha256=expected_protocol_sha256,
    )
    protocol, features, optimizer_audit = build_locked_feature_result(
        dataset=dataset,
        source_dir=source_dir,
        locked=locked,
    )
    _require_empty_output(output_dir)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable.")

    batch_plan = build_batch_plan(len(features.optimizer.records), dataset=dataset)
    batch_path = output_dir / "epoch_batch_plan.npz"
    _atomic_npz(
        batch_path,
        permutations=batch_plan.permutations,
        batch_size=np.asarray([batch_plan.batch_size], dtype=np.int64),
    )
    models, model_audit = initialize_models(features.optimizer)
    gradients = gradient_audit(models, features=features.optimizer, device=device)

    training: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    prediction_diagnostics: dict[str, Any] = {}
    checkpoint_files: dict[str, Any] = {}
    for variant in ALL_VARIANTS:
        training[variant] = train_variant(
            models[variant],
            variant=variant,
            features=features.optimizer,
            batch_plan=batch_plan,
            device=device,
        )
        checkpoint_path = output_dir / f"{variant}_final_epoch.pt"
        _atomic_checkpoint(checkpoint_path, models[variant].state_dict())
        checkpoint_files[variant] = {
            "path": checkpoint_path.name,
            "sha256": sha256_file(checkpoint_path),
            "state_sha256": training[variant]["final_state_sha256"],
        }
        predictions[variant], prediction_diagnostics[variant] = predict_variant(
            models[variant],
            variant=variant,
            features=features.validation,
            device=device,
        )

    records = features.validation.records
    prediction_frame = pd.DataFrame(
        {
            "source_row_id": [record.source_row_id for record in records],
            "stu_id": [record.student for record in records],
            "exer_id": [record.exercise for record in records],
            "coverage_bucket": [record.coverage_bucket for record in records],
            "q_cardinality_bucket": [record.q_cardinality_bucket for record in records],
            "optimizer_item_frequency": [
                features.validation.optimizer_item_frequency[record.exercise]
                for record in records
            ],
            **{
                PROBABILITY_COLUMNS[variant]: predictions[variant]
                for variant in ALL_VARIANTS
            },
        }
    )
    if "label" in prediction_frame:
        raise RuntimeError("Predict phase artifact must be outcome-free.")
    prediction_order_sha256 = validation_row_order_sha256(prediction_frame)
    if prediction_order_sha256 != features.validation.row_order_sha256:
        raise RuntimeError("Prediction order differs from the feature order.")
    prediction_path = output_dir / "validation_predictions_unlabeled.csv"
    _atomic_csv(prediction_path, prediction_frame)

    real_as_permuted = TargetLocalPairingProbe(
        **_model_kwargs(features.validation), placement="perm_pair"
    )
    real_as_permuted.load_state_dict(models["real_pair"].state_dict(), strict=True)
    sensitivity_probabilities = np.empty(
        (PERMUTATION_REPLICATES, len(records)), dtype=np.float32
    )
    mapping_hashes = []
    sensitivity_audits = []
    sensitivity_input_hashes = []
    for replicate, expected_audit in enumerate(locked["validation_permutation_audits"]):
        if replicate == 0:
            mapping = features.validation_donor_rep0
        else:
            mapping = build_donor_mapping(
                features.validation_profiles,
                statistics=features.validation_statistics,
                item_index=features.validation.q_item_index,
                q_lookup=protocol.q_lookup,
                split="validation_query",
                fold=None,
                replicate=replicate,
            )
        mapping_hash = mapping.audit["mapping_sha256"]
        if mapping_hash != expected_audit["mapping_sha256"]:
            raise RuntimeError(f"Donor mapping replicate {replicate} differs from lock.")
        replaced = replace_validation_donors(features.validation, mapping)
        sensitivity_probabilities[replicate], _ = predict_variant(
            real_as_permuted,
            variant="perm_pair",
            features=replaced,
            device=device,
        )
        mapping_hashes.append(mapping_hash)
        sensitivity_input_hashes.append(replaced.input_sha256)
        sensitivity_audits.append(
            {
                key: value
                for key, value in mapping.audit.items()
                if key != "cells"
            }
        )

    sensitivity_path = output_dir / "real_weights_donor_sensitivity.npz"
    sensitivity_prediction_hashes = [
        _hash_array(row) for row in sensitivity_probabilities
    ]
    _atomic_npz(
        sensitivity_path,
        probabilities=sensitivity_probabilities,
        source_row_ids=np.asarray(prediction_frame["source_row_id"], dtype="U64"),
        mapping_sha256=np.asarray(mapping_hashes, dtype="U64"),
    )
    manifest = {
        "schema_version": 1,
        "phase": "target_local_pairing_predict",
        "dataset": dataset,
        "git": git,
        "protocol": {
            "path": str(protocol_json.resolve()),
            "sha256": protocol_sha256,
            "source_directory": str(protocol.source_dir),
            "valid_full_sha256": locked["source"]["source_files"]["valid.csv"][
                "sha256"
            ],
            "validation_donor_rep0": locked["donor_replica_zero"],
            "optimizer_audit": optimizer_audit,
        },
        "features": features.audit,
        "optimization": {
            "seed": MODEL_SEED,
            "split_seed": SPLIT_SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "loss": "binary_cross_entropy_with_logits",
            "scheduler": None,
            "dropout": 0.0,
            "early_stopping": False,
            "checkpoint_selection": "final_epoch_only",
            "batch_namespace": batch_plan.namespace,
            "batch_plan_sha256": batch_plan.sha256,
            "batch_plan_file": {
                "path": batch_path.name,
                "sha256": sha256_file(batch_path),
            },
        },
        "models": model_audit,
        "gradient_audit": gradients,
        "training": training,
        "prediction_diagnostics": prediction_diagnostics,
        "artifacts": {
            "predictions": {
                "path": prediction_path.name,
                "sha256": sha256_file(prediction_path),
                "semantic_sha256": _prediction_semantic_sha256(prediction_frame),
                "row_order_sha256": prediction_order_sha256,
                "rows": len(prediction_frame),
            },
            "checkpoints": checkpoint_files,
            "donor_sensitivity": {
                "path": sensitivity_path.name,
                "sha256": sha256_file(sensitivity_path),
                "semantic_probability_sha256": _hash_array(sensitivity_probabilities),
                "prediction_sha256": sensitivity_prediction_hashes,
                "prediction_hash_collection_sha256": _hash_values(
                    sensitivity_prediction_hashes
                ),
                "shape": list(sensitivity_probabilities.shape),
                "mapping_sha256": mapping_hashes,
                "mapping_collection_sha256": _hash_values(mapping_hashes),
                "unique_mapping_hashes": len(set(mapping_hashes)),
                "input_sha256": sensitivity_input_hashes,
                "audits": sensitivity_audits,
                "interpretation": "fixed-model donor sensitivity; diagnostic only",
                "gating": False,
            },
        },
        "leakage_audit": {
            "validation_labels_loaded": False,
            "test_files_opened": False,
            "legacy_manifest_opened": False,
            "prediction_artifact_contains_label": False,
        },
    }
    manifest_path = output_dir / "prediction_manifest.json"
    _atomic_json(manifest_path, manifest)
    return manifest


def _load_prediction_artifacts(
    prediction_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray, list[str]]:
    manifest_path = prediction_dir / "prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("phase") != "target_local_pairing_predict":
        raise RuntimeError("Unexpected prediction manifest phase.")
    prediction_spec = manifest["artifacts"]["predictions"]
    prediction_path = prediction_dir / prediction_spec["path"]
    if sha256_file(prediction_path) != prediction_spec["sha256"]:
        raise RuntimeError("Prediction artifact SHA-256 mismatch.")
    predictions = pd.read_csv(prediction_path)
    required = {
        "source_row_id",
        "stu_id",
        "exer_id",
        "coverage_bucket",
        "optimizer_item_frequency",
        *PROBABILITY_COLUMNS.values(),
    }
    missing = required - set(predictions.columns)
    if missing or "label" in predictions:
        raise RuntimeError(f"Invalid unlabeled prediction schema; missing={sorted(missing)}.")
    if predictions["source_row_id"].astype(str).duplicated().any():
        raise RuntimeError("Prediction artifact contains duplicate source_row_id values.")
    row_order_sha256 = validation_row_order_sha256(predictions)
    if row_order_sha256 != prediction_spec["row_order_sha256"]:
        raise RuntimeError("Prediction row-order SHA-256 mismatch.")
    if _prediction_semantic_sha256(predictions) != prediction_spec["semantic_sha256"]:
        raise RuntimeError("Prediction semantic SHA-256 mismatch.")

    sensitivity_spec = manifest["artifacts"]["donor_sensitivity"]
    sensitivity_path = prediction_dir / sensitivity_spec["path"]
    if sha256_file(sensitivity_path) != sensitivity_spec["sha256"]:
        raise RuntimeError("Donor sensitivity artifact SHA-256 mismatch.")
    with np.load(sensitivity_path, allow_pickle=False) as archive:
        donor_probabilities = np.asarray(archive["probabilities"], dtype=np.float32)
        source_row_ids = archive["source_row_ids"].astype(str).tolist()
        mapping_hashes = archive["mapping_sha256"].astype(str).tolist()
    if donor_probabilities.shape != (PERMUTATION_REPLICATES, len(predictions)):
        raise RuntimeError("Donor sensitivity probability shape is unexpected.")
    if source_row_ids != predictions["source_row_id"].astype(str).tolist():
        raise RuntimeError("Donor sensitivity rows are not aligned to predictions.")
    if mapping_hashes != sensitivity_spec["mapping_sha256"]:
        raise RuntimeError("Donor sensitivity mapping hashes differ from manifest.")
    if _hash_array(donor_probabilities) != sensitivity_spec["semantic_probability_sha256"]:
        raise RuntimeError("Donor sensitivity semantic hash mismatch.")
    prediction_hashes = [_hash_array(row) for row in donor_probabilities]
    if prediction_hashes != sensitivity_spec["prediction_sha256"]:
        raise RuntimeError("Per-replicate donor prediction hashes differ from manifest.")
    return manifest, predictions, donor_probabilities, mapping_hashes


def run_evaluate_phase(
    *,
    prediction_dir: Path,
    source_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    from utils.target_local_pairing_evaluation import (
        evaluate_target_local_pairing,
        summarize_donor_sensitivity,
    )

    manifest, predictions, donor_probabilities, mapping_hashes = (
        _load_prediction_artifacts(prediction_dir)
    )
    _git_snapshot(manifest["git"]["head"])
    if source_dir.resolve() != Path(manifest["protocol"]["source_directory"]).resolve():
        raise RuntimeError("Evaluation source directory differs from prediction source.")
    _require_empty_output(output_dir)
    labels = load_validation_labels_for_evaluation(
        source_dir / "valid.csv",
        feature_rows=predictions,
        expected_valid_sha256=manifest["protocol"]["valid_full_sha256"],
    )
    aligned = join_validation_predictions_with_labels(
        predictions,
        labels,
        expected_prediction_order_sha256=manifest["artifacts"]["predictions"][
            "row_order_sha256"
        ],
    )
    if validation_row_order_sha256(aligned) != manifest["artifacts"]["predictions"][
        "row_order_sha256"
    ]:
        raise RuntimeError("Aligned evaluation rows differ from prediction rows.")

    evaluation = evaluate_target_local_pairing(
        aligned,
        expected_real_order_sha256=manifest["artifacts"]["predictions"][
            "row_order_sha256"
        ],
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        seed=SPLIT_SEED,
    )
    donor_sensitivity = summarize_donor_sensitivity(
        labels=aligned["label"].to_numpy(dtype=np.int64),
        real_probabilities=aligned["prob_real_pair"].to_numpy(
            dtype=np.float64
        ),
        donor_probabilities=donor_probabilities,
        mapping_hashes=mapping_hashes,
        source_row_ids=aligned["source_row_id"].astype(str).tolist(),
        expected_row_order_sha256=manifest["artifacts"]["predictions"][
            "row_order_sha256"
        ],
    )
    aligned_path = output_dir / "aligned_predictions_with_labels.csv"
    loo_path = output_dir / "leave_one_target_item_out.csv"
    _atomic_csv(aligned_path, aligned)
    _atomic_csv(loo_path, evaluation.leave_one_item_out)
    payload = {
        "schema_version": 1,
        "phase": "target_local_pairing_evaluate",
        "dataset": manifest["dataset"],
        "git": manifest["git"],
        "architecture_family_sha256": manifest["models"]["architecture"][
            "family_sha256"
        ],
        "feature_definition": {
            "item_numeric_names": manifest["features"]["item_numeric_names"],
            "support_statistic_names": manifest["features"][
                "support_statistic_names"
            ],
            "response_feature_names": manifest["features"]["response_feature_names"],
        },
        "protocol": {
            "sha256": manifest["protocol"]["sha256"],
            "identified": manifest["protocol"]["validation_donor_rep0"]["identified"],
            "changed_row_fraction": manifest["protocol"]["validation_donor_rep0"][
                "changed_row_fraction"
            ],
            "changed_students": manifest["protocol"]["validation_donor_rep0"][
                "changed_students"
            ],
        },
        "evaluation": evaluation.summary,
        "donor_sensitivity": donor_sensitivity,
        "artifacts": {
            "prediction_manifest": {
                "path": os.path.relpath(
                    prediction_dir / "prediction_manifest.json",
                    start=output_dir,
                ),
                "path_base": "evaluation_directory",
                "sha256": sha256_file(prediction_dir / "prediction_manifest.json"),
            },
            "aligned_predictions": {
                "path": aligned_path.name,
                "sha256": sha256_file(aligned_path),
            },
            "leave_one_target_item_out": {
                "path": loo_path.name,
                "sha256": sha256_file(loo_path),
            },
        },
        "interpretation": {
            "validation_driven": True,
            "donor_sensitivity_is_randomization_test": False,
            "conditional_p_value_reported": False,
            "coverage_slices_gating": False,
            "item_robustness_gating": False,
            "test_accessed": False,
        },
    }
    _atomic_json(output_dir / "evaluation.json", payload)
    return payload


def compute_activation_gate(evaluations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, dict[str, Any]] = {}
    for payload in evaluations:
        dataset = str(payload["dataset"])
        if dataset in by_dataset:
            raise RuntimeError(f"Duplicate evaluation for {dataset}.")
        by_dataset[dataset] = payload
    if set(by_dataset) != set(EXPECTED_DATASETS):
        raise RuntimeError(
            f"Aggregate requires exactly {EXPECTED_DATASETS}; got {tuple(sorted(by_dataset))}."
        )
    architecture_hashes = {
        payload["architecture_family_sha256"] for payload in by_dataset.values()
    }
    commits = {payload["git"]["head"] for payload in by_dataset.values()}
    feature_definitions = {
        _canonical_json(payload["feature_definition"])
        for payload in by_dataset.values()
    }
    if len(architecture_hashes) != 1 or len(commits) != 1 or len(feature_definitions) != 1:
        raise RuntimeError("Datasets do not share one frozen architecture/code/feature definition.")

    identified = {
        dataset: payload
        for dataset, payload in by_dataset.items()
        if bool(payload["protocol"]["identified"])
    }
    effects = {
        dataset: float(payload["evaluation"]["dataset_effect"])
        for dataset, payload in identified.items()
    }
    counted = [dataset for dataset, effect in effects.items() if effect >= 0.002]
    at_least_two = len(counted) >= 2
    at_least_one_large = any(effect >= 0.003 for effect in effects.values())
    joint_intervals = [
        payload["evaluation"].get("joint_min_delta_ci")
        for payload in identified.values()
    ]
    at_least_one_joint_ci = any(
        interval is not None
        and len(interval) == 2
        and float(interval[0]) > 0.0
        for interval in joint_intervals
    )
    no_auc_regression = all(effect >= -0.001 for effect in effects.values())
    counted_brier_guard = all(
        float(by_dataset[dataset]["evaluation"]["max_brier_regression"]) <= 0.0002
        for dataset in counted
    )
    checks = {
        "at_least_two_identified_datasets_effect_ge_0.002": at_least_two,
        "at_least_one_identified_dataset_effect_ge_0.003": at_least_one_large,
        "at_least_one_joint_student_bootstrap_ci_low_gt_0": at_least_one_joint_ci,
        "no_identified_dataset_delta_below_minus_0.001": no_auc_regression,
        "counted_datasets_max_brier_regression_le_0.0002": counted_brier_guard,
    }
    return {
        "schema_version": 1,
        "gate": "target_local_pairing_activation",
        "activated": all(checks.values()),
        "decision_if_pass": "activate CD-specific target-local state module design",
        "decision_if_fail": "close current history/pairing branch",
        "checks": checks,
        "identified_datasets": sorted(identified),
        "counted_effect_datasets": sorted(counted),
        "dataset_effects": effects,
        "architecture_family_sha256": next(iter(architecture_hashes)),
        "git_commit": next(iter(commits)),
        "multi_seed_used": False,
        "bootstrap_is_multi_seed": False,
        "conditional_permutation_p_value_used": False,
        "donor_sensitivity_gating": False,
        "coverage_slices_gating": False,
        "item_robustness_gating": False,
    }


def run_aggregate_phase(
    *,
    evaluation_jsons: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    evaluations = [
        json.loads(path.read_text(encoding="utf-8")) for path in evaluation_jsons
    ]
    for payload in evaluations:
        if payload.get("phase") != "target_local_pairing_evaluate":
            raise RuntimeError("Unexpected evaluation phase in aggregate input.")
    _require_empty_output(output_dir)
    gate = compute_activation_gate(evaluations)
    payload = {
        **gate,
        "datasets": {
            result["dataset"]: {
                "protocol": result["protocol"],
                "metrics": result["evaluation"]["metrics"],
                "deltas_vs_control": result["evaluation"]["deltas_vs_control"],
                "dataset_effect": result["evaluation"]["dataset_effect"],
                "joint_min_delta_ci": result["evaluation"]["joint_min_delta_ci"],
                "max_brier_regression": result["evaluation"]["max_brier_regression"],
                "coverage_descriptive_slices": result["evaluation"].get(
                    "coverage_descriptive_slices", {}
                ),
                "donor_sensitivity": result["donor_sensitivity"],
            }
            for result in evaluations
        },
    }
    _atomic_json(output_dir / "activation_decision.json", payload)
    return payload


def _add_predict_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True, choices=EXPECTED_DATASETS)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--protocol-json", required=True, type=Path)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the validation-only target-local pairing activation audit. "
            "There is deliberately no test-path argument."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    predict = subparsers.add_parser("predict")
    _add_predict_arguments(predict)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--prediction-dir", required=True, type=Path)
    evaluate.add_argument("--source-dir", required=True, type=Path)
    evaluate.add_argument("--output-dir", required=True, type=Path)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument(
        "--evaluation-json", required=True, action="append", type=Path
    )
    aggregate.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "predict":
        result = run_predict_phase(
            dataset=args.dataset,
            source_dir=args.source_dir,
            protocol_json=args.protocol_json,
            expected_protocol_sha256=args.expected_protocol_sha256,
            expected_commit=args.expected_commit,
            output_dir=args.output_dir,
            device_name=args.device,
        )
        printable = {
            "dataset": result["dataset"],
            "prediction_rows": result["artifacts"]["predictions"]["rows"],
            "donor_replicates": result["artifacts"]["donor_sensitivity"]["shape"][0],
        }
    elif args.command == "evaluate":
        result = run_evaluate_phase(
            prediction_dir=args.prediction_dir,
            source_dir=args.source_dir,
            output_dir=args.output_dir,
        )
        printable = {
            "dataset": result["dataset"],
            "dataset_effect": result["evaluation"]["dataset_effect"],
        }
    else:
        result = run_aggregate_phase(
            evaluation_jsons=args.evaluation_json,
            output_dir=args.output_dir,
        )
        printable = {
            "activated": result["activated"],
            "checks": result["checks"],
        }
    print(json.dumps(printable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
