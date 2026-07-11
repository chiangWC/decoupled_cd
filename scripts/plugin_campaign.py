from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


_CONFIG_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DATA_PROTOCOLS = {"standard", "holdout"}
_PROTOCOL_FIELDS = (
    "split",
    "seed",
    "doa_seed",
    "min_responses",
    "max_pairs_per_concept",
    "split_seed",
    "q_matrix_sha256",
)
_APPROVED_PROTOCOL = {
    "split": "valid",
    "seed": 42,
    "doa_seed": 42,
    "min_responses": 3,
    "split_seed": 2024,
}
_INITIALIZATION_CONFIG_FIELDS = {
    "init_mode",
    "baseline_checkpoint_sha256",
    "base_lr",
    "lr_multiplier",
}


class DuplicateTestEvaluationError(RuntimeError):
    """Raised when a frozen configuration already owns a test claim."""


class FrozenConfigMismatchError(RuntimeError):
    """Raised when a caller-provided frozen ID does not match its inputs."""


@dataclass(frozen=True)
class EvaluationArtifactSnapshot:
    """In-memory bytes consumed after a frozen test claim is created."""

    checkpoint_bytes: bytes
    id_maps_bytes: bytes
    q_matrix_bytes: bytes


@dataclass(frozen=True)
class TestClaimSnapshot:
    """Path and the exact bytes written through the exclusive claim FD."""

    path: Path
    payload: bytes


def snapshot_evaluation_artifacts(
    checkpoint_path: Path,
    q_matrix_path: Path,
) -> EvaluationArtifactSnapshot:
    checkpoint_path = Path(checkpoint_path)
    return EvaluationArtifactSnapshot(
        checkpoint_bytes=checkpoint_path.read_bytes(),
        id_maps_bytes=checkpoint_path.with_name("id_maps.json").read_bytes(),
        q_matrix_bytes=Path(q_matrix_path).read_bytes(),
    )


def validate_training_initialization(
    *,
    checkpoint_path: Path,
    snapshot: EvaluationArtifactSnapshot,
    q_matrix_bytes: bytes,
    processor_id_maps: Mapping[str, Any],
    data_protocol: str | None,
    dataset_name: str | None,
    seed: int,
) -> None:
    """Reject a baseline checkpoint that does not match training resources."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"training initialization checkpoint does not exist: {checkpoint_path}"
        )
    if checkpoint_path.read_bytes() != snapshot.checkpoint_bytes:
        raise ValueError("training initialization checkpoint bytes changed")

    id_maps_path = checkpoint_path.with_name("id_maps.json")
    if not id_maps_path.is_file():
        raise FileNotFoundError(
            f"training initialization sibling id_maps.json does not exist: {id_maps_path}"
        )
    if id_maps_path.read_bytes() != snapshot.id_maps_bytes:
        raise ValueError("training initialization ID maps bytes changed")
    if bytes(q_matrix_bytes) != snapshot.q_matrix_bytes:
        raise ValueError("training initialization Q-matrix does not match checkpoint")

    manifest_path = checkpoint_path.parent.parent / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"training initialization manifest does not exist: {manifest_path}"
        )
    try:
        records = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot read training initialization manifest: {manifest_path}"
        ) from exc
    artifact_root = manifest_path.parent.parent
    matches = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("training initialization manifest record must be an object")
        raw_path = record.get("checkpoint_path")
        if isinstance(raw_path, str) and (
            artifact_root / raw_path
        ).resolve() == checkpoint_path.resolve():
            matches.append(record)
    if len(matches) != 1:
        raise ValueError(
            "training initialization manifest must contain exactly one checkpoint record"
        )
    record = matches[0]
    if record.get("checkpoint_sha256") != sha256_bytes(snapshot.checkpoint_bytes):
        raise ValueError("training initialization manifest checkpoint SHA mismatch")
    if record.get("id_maps_sha256") != sha256_bytes(snapshot.id_maps_bytes):
        raise ValueError("training initialization manifest ID maps SHA mismatch")
    plugin = record.get("plugin_config")
    if not isinstance(plugin, Mapping) or plugin.get("aux_weight") != 0.0:
        raise ValueError("training initialization manifest aux_weight must be 0")
    protocol = record.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("training initialization manifest protocol must be an object")
    if seed != 42 or protocol.get("seed") != 42:
        raise ValueError("training initialization seed must be 42")
    if bool(data_protocol) != bool(dataset_name):
        raise ValueError("training initialization data protocol and dataset must pair")
    if protocol.get("data_protocol") != data_protocol:
        raise ValueError("training initialization manifest data protocol mismatch")
    if protocol.get("dataset_name") != dataset_name:
        raise ValueError("training initialization manifest dataset mismatch")
    if protocol.get("q_matrix_sha256") != sha256_bytes(q_matrix_bytes):
        raise ValueError("training initialization manifest Q-matrix SHA mismatch")

    try:
        frozen_id_maps = json.loads(snapshot.id_maps_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("training initialization ID schema is invalid JSON") from exc
    required_fields = {"stu_ids", "exer_ids", "cpt_ids"}
    for label, id_maps in (
        ("frozen", frozen_id_maps),
        ("processor", processor_id_maps),
    ):
        if not isinstance(id_maps, Mapping) or set(id_maps) != required_fields:
            raise ValueError(
                f"training initialization {label} ID schema must contain exactly "
                "stu_ids, exer_ids, and cpt_ids"
            )
        for field in sorted(required_fields):
            values = id_maps[field]
            if (
                not isinstance(values, list)
                or any(not isinstance(value, str) for value in values)
                or len(values) != len(set(values))
            ):
                raise ValueError(
                    f"training initialization {label} ID schema field {field} "
                    "must be a unique string list"
                )
    if frozen_id_maps != dict(processor_id_maps):
        raise ValueError("training initialization ID maps do not match processor")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _validation_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for name in ("auc", "acc", "rmse"):
        try:
            value = float(metrics[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"validation metric {name!r} is required") from exc
        if not math.isfinite(value):
            raise ValueError(f"validation metric {name!r} must be finite")
        normalized[name] = value
    return normalized


def _validated_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    missing = [name for name in _PROTOCOL_FIELDS if name not in protocol]
    if missing:
        raise ValueError(f"campaign protocol is missing: {', '.join(missing)}")
    normalized = _json_copy(dict(protocol))
    has_kind = "data_protocol" in normalized
    has_dataset = "dataset_name" in normalized
    if has_kind != has_dataset:
        raise ValueError(
            "data_protocol and dataset_name must be provided together"
        )
    if has_kind:
        if normalized["data_protocol"] not in _DATA_PROTOCOLS:
            raise ValueError("data_protocol must be 'standard' or 'holdout'")
        if (
            not isinstance(normalized["dataset_name"], str)
            or not normalized["dataset_name"].strip()
        ):
            raise ValueError("dataset_name must be a non-empty string")
    for field, expected in _APPROVED_PROTOCOL.items():
        if normalized[field] != expected:
            raise ValueError(
                f"candidate protocol {field} must be {expected!r}, "
                f"got {normalized[field]!r}"
            )
    _validated_sha256(normalized["q_matrix_sha256"], "q_matrix_sha256")
    return normalized


def _validated_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def compute_recipe_id(
    recipe_config: Mapping[str, Any],
    backbone_config: Mapping[str, Any],
) -> str:
    payload = {
        "recipe_config": _json_copy(dict(recipe_config)),
        "backbone_config": _json_copy(dict(backbone_config)),
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def compute_frozen_config_id(
    *,
    checkpoint_sha256: str,
    id_maps_sha256: str,
    plugin_config: Mapping[str, Any],
    backbone_config: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> str:
    payload = {
        "checkpoint_sha256": _validated_sha256(
            checkpoint_sha256, "checkpoint_sha256"
        ),
        "id_maps_sha256": _validated_sha256(id_maps_sha256, "id_maps_sha256"),
        "plugin_config": _json_copy(dict(plugin_config)),
        "backbone_config": _json_copy(dict(backbone_config)),
        "protocol": _validated_protocol(protocol),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def protocol_data_kind(protocol: Mapping[str, Any]) -> str:
    # Historical selections predate data_protocol and are all holdout runs.
    return str(protocol.get("data_protocol", "holdout"))


class CandidateStore:
    """Append-only per-epoch checkpoint and validation artifact store."""

    def __init__(
        self,
        *,
        output_dir: Path,
        model_name: str,
        plugin_config: Mapping[str, Any],
        recipe_config: Mapping[str, Any],
        backbone_config: Mapping[str, Any],
        protocol: Mapping[str, Any],
    ) -> None:
        self.output_dir = Path(output_dir)
        self.model_name = str(model_name)
        self.plugin_config = _json_copy(dict(plugin_config))
        self.recipe_config = _json_copy(dict(recipe_config))
        self.backbone_config = _json_copy(dict(backbone_config))
        self.protocol = _validated_protocol(protocol)
        if not self.model_name:
            raise ValueError("model_name must not be empty")

    @property
    def candidates_dir(self) -> Path:
        return self.output_dir / "candidates"

    @property
    def manifest_path(self) -> Path:
        return self.candidates_dir / "manifest.jsonl"

    def save(
        self,
        *,
        epoch: int,
        validation_metrics: Mapping[str, Any],
        state_dict: Mapping[str, Any],
        mastery: np.ndarray,
        id_maps: Mapping[str, Any],
    ) -> dict[str, Any]:
        if epoch < 1:
            raise ValueError("epoch must be at least 1")
        validation = _validation_metrics(validation_metrics)
        epoch_name = f"epoch-{epoch:03d}"
        relative_dir = Path("candidates") / epoch_name
        candidate_dir = self.output_dir / relative_dir
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        candidate_dir.mkdir(exist_ok=False)

        checkpoint_path = candidate_dir / "checkpoint.pth"
        mastery_path = candidate_dir / "mastery.npy"
        id_maps_path = candidate_dir / "id_maps.json"
        torch.save(state_dict, checkpoint_path)
        np.save(mastery_path, np.asarray(mastery))
        with id_maps_path.open("x", encoding="utf-8") as handle:
            json.dump(_json_copy(dict(id_maps)), handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")

        record = {
            "epoch": int(epoch),
            "validation": validation,
            "model_name": f"{self.model_name}-{epoch_name}",
            "checkpoint_path": (relative_dir / "checkpoint.pth").as_posix(),
            "mastery_path": (relative_dir / "mastery.npy").as_posix(),
            "id_maps_path": (relative_dir / "id_maps.json").as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "mastery_sha256": sha256_file(mastery_path),
            "id_maps_sha256": sha256_file(id_maps_path),
            "plugin_config": self.plugin_config,
            "backbone_config": self.backbone_config,
            "recipe_config": self.recipe_config,
            "recipe_id": compute_recipe_id(
                self.recipe_config, self.backbone_config
            ),
            "protocol": self.protocol,
        }
        payload = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode()
        descriptor = os.open(
            self.manifest_path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        try:
            with os.fdopen(descriptor, "ab", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(descriptor)
        return _json_copy(record)


def run_candidate_training(
    *,
    epochs: int,
    patience: int,
    train_epoch: Callable[[int], Any],
    evaluate_validation: Callable[[], Mapping[str, Any]],
    snapshot_candidate: Callable[[int, Mapping[str, Any]], Any],
) -> dict[str, Any]:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if patience < 1:
        raise ValueError("patience must be at least 1")
    best_auc = -math.inf
    best_epoch: int | None = None
    stale_epochs = 0
    epochs_completed = 0
    for epoch in range(1, epochs + 1):
        train_epoch(epoch)
        validation = _validation_metrics(evaluate_validation())
        snapshot_candidate(epoch, validation)
        epochs_completed = epoch
        if validation["auc"] > best_auc:
            best_auc = validation["auc"]
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    return {
        "best_auc": best_auc,
        "best_epoch": best_epoch,
        "epochs_completed": epochs_completed,
    }


def select_evaluation_loader(loaders: Sequence[Any], split: str) -> Any:
    if split == "valid":
        return loaders[1]
    if split == "test":
        return loaders[2]
    raise ValueError("evaluation split must be 'valid' or 'test'")


def _route_head(route_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(route_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _evaluation_plugin_config(
    selection_config: Any,
    cli_config: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(selection_config, Mapping):
        raise FrozenConfigMismatchError("selection plugin_config must be an object")
    normalized_selection = _json_copy(dict(selection_config))
    normalized_cli = _json_copy(dict(cli_config))
    if _INITIALIZATION_CONFIG_FIELDS.intersection(normalized_cli):
        raise FrozenConfigMismatchError(
            "evaluation CLI plugin_config must not contain training initialization metadata"
        )
    present = _INITIALIZATION_CONFIG_FIELDS.intersection(normalized_selection)
    if present and present != _INITIALIZATION_CONFIG_FIELDS:
        raise FrozenConfigMismatchError(
            "selection plugin_config has incomplete training initialization metadata"
        )
    prediction_config = {
        key: value
        for key, value in normalized_selection.items()
        if key not in _INITIALIZATION_CONFIG_FIELDS
    }
    if prediction_config != normalized_cli:
        raise FrozenConfigMismatchError(
            "selection plugin_config does not match evaluation CLI"
        )
    if not present:
        return normalized_selection

    init_mode = normalized_selection["init_mode"]
    if init_mode not in {"random", "baseline-finetune"}:
        raise FrozenConfigMismatchError(
            "selection plugin_config init_mode is invalid"
        )
    checkpoint_sha256 = normalized_selection["baseline_checkpoint_sha256"]
    if init_mode == "baseline-finetune":
        try:
            _validated_sha256(
                checkpoint_sha256,
                "selection plugin_config baseline_checkpoint_sha256",
            )
        except ValueError as exc:
            raise FrozenConfigMismatchError(str(exc)) from exc
    elif checkpoint_sha256 is not None:
        raise FrozenConfigMismatchError(
            "random initialization cannot bind a baseline checkpoint"
        )
    try:
        base_lr = float(normalized_selection["base_lr"])
        multiplier = float(normalized_selection["lr_multiplier"])
    except (TypeError, ValueError) as exc:
        raise FrozenConfigMismatchError(
            "selection plugin_config learning-rate provenance is invalid"
        ) from exc
    if not math.isfinite(base_lr) or base_lr <= 0.0:
        raise FrozenConfigMismatchError(
            "selection plugin_config base_lr must be positive and finite"
        )
    if multiplier not in {0.25, 1.0}:
        raise FrozenConfigMismatchError(
            "selection plugin_config lr_multiplier must be 0.25 or 1.0"
        )
    return normalized_selection


def _validated_selection(
    *,
    selection_path: Path,
    checkpoint_path: Path,
    plugin_config: Mapping[str, Any],
    backbone_config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    supplied_frozen_config_id: str | None,
    artifact_snapshot: EvaluationArtifactSnapshot | None = None,
) -> tuple[
    str,
    str,
    str,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    str,
    str,
    str | None,
]:
    try:
        selection_bytes = Path(selection_path).read_bytes()
        selection = json.loads(selection_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrozenConfigMismatchError(
            f"cannot read frozen selection: {selection_path}"
        ) from exc
    if not isinstance(selection, dict):
        raise FrozenConfigMismatchError("frozen selection must be a JSON object")
    checkpoint_path = Path(checkpoint_path)
    if artifact_snapshot is None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint_sha256 = sha256_file(checkpoint_path)
        id_maps_path = checkpoint_path.with_name("id_maps.json")
        if not id_maps_path.is_file():
            raise FrozenConfigMismatchError(
                f"checkpoint sibling id_maps.json does not exist: {id_maps_path}"
            )
        id_maps_sha256 = sha256_file(id_maps_path)
    else:
        checkpoint_sha256 = sha256_bytes(artifact_snapshot.checkpoint_bytes)
        id_maps_sha256 = sha256_bytes(artifact_snapshot.id_maps_bytes)
    normalized_config = _evaluation_plugin_config(
        selection.get("plugin_config"),
        plugin_config,
    )
    normalized_backbone = _json_copy(dict(backbone_config))
    try:
        normalized_protocol = _validated_protocol(protocol)
    except ValueError as exc:
        raise FrozenConfigMismatchError(str(exc)) from exc
    if (
        artifact_snapshot is not None
        and sha256_bytes(artifact_snapshot.q_matrix_bytes)
        != normalized_protocol["q_matrix_sha256"]
    ):
        raise FrozenConfigMismatchError(
            "evaluation Q-matrix bytes do not match evaluation protocol"
        )
    if selection.get("checkpoint_sha256") != checkpoint_sha256:
        raise FrozenConfigMismatchError(
            "selection checkpoint SHA-256 does not match actual checkpoint"
        )
    if selection.get("id_maps_sha256") != id_maps_sha256:
        raise FrozenConfigMismatchError(
            "selection id_maps SHA-256 does not match checkpoint sibling id_maps.json"
        )
    if selection.get("backbone_config") != normalized_backbone:
        raise FrozenConfigMismatchError(
            "selection backbone_config does not match evaluation CLI"
        )
    if selection.get("protocol") != normalized_protocol:
        raise FrozenConfigMismatchError(
            "selection protocol does not match evaluation protocol"
        )
    config_id = compute_frozen_config_id(
        checkpoint_sha256=checkpoint_sha256,
        id_maps_sha256=id_maps_sha256,
        plugin_config=normalized_config,
        backbone_config=normalized_backbone,
        protocol=normalized_protocol,
    )
    if selection.get("frozen_config_id") != config_id:
        raise FrozenConfigMismatchError("selection frozen_config_id is not canonical")
    if supplied_frozen_config_id is not None and supplied_frozen_config_id != config_id:
        raise FrozenConfigMismatchError(
            "supplied frozen_config_id does not match frozen selection"
        )
    dataset = selection.get("dataset")
    if not isinstance(dataset, str) or not dataset.strip():
        raise FrozenConfigMismatchError("selection dataset must be a non-empty string")
    kind = protocol_data_kind(normalized_protocol)
    raw_holdout_sha = selection.get("holdout_assignments_sha256")
    if kind == "holdout":
        try:
            holdout_assignments_sha256 = _validated_sha256(
                raw_holdout_sha,
                "selection holdout_assignments_sha256",
            )
        except ValueError as exc:
            raise FrozenConfigMismatchError(str(exc)) from exc
    elif "holdout_assignments_sha256" in selection:
        raise FrozenConfigMismatchError(
            "standard selection must not bind holdout assignments"
        )
    else:
        holdout_assignments_sha256 = None
    return (
        config_id,
        checkpoint_sha256,
        id_maps_sha256,
        normalized_config,
        normalized_backbone,
        normalized_protocol,
        sha256_bytes(selection_bytes),
        dataset,
        holdout_assignments_sha256,
    )


def claim_test_evaluation(
    *,
    ledger_dir: Path,
    selection_path: Path,
    checkpoint_path: Path,
    plugin_config: Mapping[str, Any],
    backbone_config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    route_root: Path,
    supplied_frozen_config_id: str | None = None,
    argv: Sequence[str] | None = None,
    route_head: str | None = None,
    claimed_at_utc: str | None = None,
    artifact_snapshot: EvaluationArtifactSnapshot | None = None,
    return_snapshot: bool = False,
) -> Path | TestClaimSnapshot:
    (
        frozen_config_id,
        checkpoint_sha256,
        id_maps_sha256,
        normalized_config,
        normalized_backbone,
        normalized_protocol,
        selection_sha256,
        dataset,
        holdout_assignments_sha256,
    ) = _validated_selection(
        selection_path=selection_path,
        checkpoint_path=checkpoint_path,
        plugin_config=plugin_config,
        backbone_config=backbone_config,
        protocol=protocol,
        supplied_frozen_config_id=supplied_frozen_config_id,
        artifact_snapshot=artifact_snapshot,
    )
    if not _CONFIG_ID_PATTERN.fullmatch(frozen_config_id):
        raise FrozenConfigMismatchError("selection frozen_config_id is unsafe")
    resolved_head = route_head or _route_head(Path(route_root))
    if not re.fullmatch(r"[0-9a-f]{40}", resolved_head):
        raise ValueError("route_head must be a 40-character lowercase Git SHA")
    claimed_at = claimed_at_utc or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    record = {
        "frozen_config_id": frozen_config_id,
        "checkpoint_sha256": checkpoint_sha256,
        "id_maps_sha256": id_maps_sha256,
        "plugin_config": normalized_config,
        "backbone_config": normalized_backbone,
        "protocol": normalized_protocol,
        "selection_sha256": selection_sha256,
        "dataset": dataset,
        "selection_path": str(Path(selection_path).resolve()),
        "route_head": resolved_head,
        "claimed_at_utc": claimed_at,
        "argv": list(argv if argv is not None else sys.argv),
    }
    if holdout_assignments_sha256 is not None:
        record["holdout_assignments_sha256"] = holdout_assignments_sha256
    claim_payload = canonical_json_bytes(record)
    ledger_dir = Path(ledger_dir)
    ledger_dir.mkdir(parents=True, exist_ok=True)
    claim_path = ledger_dir / f"{frozen_config_id}.json"
    try:
        descriptor = os.open(
            claim_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise DuplicateTestEvaluationError(
            f"test evaluation already claimed for {frozen_config_id}"
        ) from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(claim_payload)
        handle.flush()
        os.fsync(handle.fileno())
    if return_snapshot:
        return TestClaimSnapshot(path=claim_path, payload=claim_payload)
    return claim_path


def prepare_evaluation_resources(
    *,
    split: str,
    load_resources: Callable[[], Any],
    checkpoint_path: Path,
    plugin_config: Mapping[str, Any],
    backbone_config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    ledger_dir: Path | None = None,
    selection_path: Path | None = None,
    supplied_frozen_config_id: str | None = None,
    route_root: Path | None = None,
    argv: Sequence[str] | None = None,
    route_head: str | None = None,
    artifact_snapshot: EvaluationArtifactSnapshot | None = None,
    return_test_claim_bytes: bool = False,
    return_plugin_config: bool = False,
) -> Any:
    if split not in {"valid", "test"}:
        raise ValueError("evaluation split must be 'valid' or 'test'")
    test_claim_bytes = None
    evaluation_plugin_config = _json_copy(dict(plugin_config))
    if split == "test":
        if ledger_dir is None or selection_path is None or route_root is None:
            raise ValueError(
                "test evaluation requires ledger_dir, selection_path, and route_root"
            )
        claim_snapshot = claim_test_evaluation(
            ledger_dir=ledger_dir,
            selection_path=selection_path,
            checkpoint_path=checkpoint_path,
            plugin_config=plugin_config,
            backbone_config=backbone_config,
            protocol=protocol,
            route_root=route_root,
            supplied_frozen_config_id=supplied_frozen_config_id,
            argv=argv,
            route_head=route_head,
            artifact_snapshot=artifact_snapshot,
            return_snapshot=True,
        )
        if not isinstance(claim_snapshot, TestClaimSnapshot):
            raise RuntimeError("claim writer did not return an immutable snapshot")
        test_claim_bytes = claim_snapshot.payload
        if return_plugin_config:
            evaluation_plugin_config = json.loads(test_claim_bytes)["plugin_config"]
    elif selection_path is not None:
        validated = _validated_selection(
            selection_path=selection_path,
            checkpoint_path=checkpoint_path,
            plugin_config=plugin_config,
            backbone_config=backbone_config,
            protocol=protocol,
            supplied_frozen_config_id=supplied_frozen_config_id,
            artifact_snapshot=artifact_snapshot,
        )
        evaluation_plugin_config = validated[3]
    resources = load_resources()
    if return_test_claim_bytes and return_plugin_config:
        return resources, test_claim_bytes, evaluation_plugin_config
    if return_test_claim_bytes:
        return resources, test_claim_bytes
    if return_plugin_config:
        return resources, evaluation_plugin_config
    return resources


def _artifact_token(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        if float(value).is_integer():
            return str(int(value))
    return str(value)


def _normalized_evaluation_cache_rows(
    *,
    interaction_rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[float],
    labels: Sequence[float],
) -> list[tuple[int, str, str, str, float, float]]:
    rows = list(interaction_rows)
    if len(rows) != len(predictions):
        raise ValueError(
            "evaluation cache row count must match predictions and labels"
        )
    normalized = []
    required = ("stu_id", "exer_id", "cpt_seq", "label")
    for index, (row, probability, evaluated_label) in enumerate(
        zip(rows, predictions, labels, strict=True)
    ):
        if not isinstance(row, Mapping):
            raise ValueError(f"evaluation cache row {index} must be a mapping")
        missing = [field for field in required if field not in row]
        if missing:
            raise ValueError(
                f"evaluation cache row {index} is missing: {', '.join(missing)}"
            )
        try:
            raw_label = float(row["label"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"evaluation cache row {index} label must be numeric"
            ) from exc
        if not math.isfinite(raw_label) or raw_label != evaluated_label:
            raise ValueError(
                f"evaluation cache label mismatch at row {index}: "
                f"raw={raw_label!r}, evaluated={evaluated_label!r}"
            )
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"evaluation cache probability at row {index} must be in [0, 1]"
            )
        cpt_seq = row["cpt_seq"]
        if cpt_seq is None or (
            isinstance(cpt_seq, (float, np.floating))
            and not math.isfinite(float(cpt_seq))
        ):
            normalized_cpt_seq = ""
        else:
            normalized_cpt_seq = _artifact_token(cpt_seq)
        normalized.append(
            (
                index,
                _artifact_token(row["stu_id"]),
                _artifact_token(row["exer_id"]),
                normalized_cpt_seq,
                raw_label,
                probability,
            )
        )
    return normalized


def _write_evaluation_cache_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor = os.open(
        Path(path),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(_json_copy(dict(payload)), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_evaluation_artifacts(
    *,
    output_dir: Path,
    split: str,
    metrics: Mapping[str, Any],
    predictions: Sequence[Any],
    labels: Sequence[Any],
    test_claim_bytes: bytes | None = None,
    interaction_rows: Sequence[Mapping[str, Any]] | None = None,
    mastery: np.ndarray,
    id_maps: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> None:
    if split not in {"valid", "test"}:
        raise ValueError("evaluation split must be 'valid' or 'test'")
    normalized_metrics = _validation_metrics(metrics)
    predictions = [float(value) for value in predictions]
    labels = [float(value) for value in labels]
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have equal length")
    cache_rows = None
    cache_metadata = None
    test_claim = None
    if interaction_rows is not None:
        cache_rows = _normalized_evaluation_cache_rows(
            interaction_rows=interaction_rows,
            predictions=predictions,
            labels=labels,
        )
        cache_metadata = _json_copy(dict(metadata or {}))
        try:
            cache_metadata["checkpoint_sha256"] = _validated_sha256(
                cache_metadata["checkpoint_sha256"], "checkpoint_sha256"
            )
            cache_metadata["source_id_maps_sha256"] = _validated_sha256(
                cache_metadata["source_id_maps_sha256"],
                "source_id_maps_sha256",
            )
            cache_metadata["protocol"] = _validated_protocol(
                cache_metadata["protocol"]
            )
        except KeyError as exc:
            raise ValueError(
                f"evaluation cache metadata is missing {exc.args[0]}"
            ) from exc
        if split == "test":
            if test_claim_bytes is None:
                raise ValueError("test cache requires guarded test claim bytes")
            try:
                test_claim = json.loads(test_claim_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("test claim bytes must contain JSON") from exc
            if not isinstance(test_claim, dict):
                raise ValueError("test claim must be a JSON object")
            if test_claim_bytes != canonical_json_bytes(test_claim):
                raise ValueError("test claim bytes are not canonical")
            for field, expected in (
                ("checkpoint_sha256", cache_metadata["checkpoint_sha256"]),
                ("id_maps_sha256", cache_metadata["source_id_maps_sha256"]),
                ("plugin_config", cache_metadata.get("plugin_config")),
                ("backbone_config", cache_metadata.get("backbone_config")),
                ("protocol", cache_metadata["protocol"]),
            ):
                if test_claim.get(field) != expected:
                    raise ValueError(
                        f"test claim {field} does not match evaluation metadata"
                    )
            expected_frozen_id = compute_frozen_config_id(
                checkpoint_sha256=test_claim["checkpoint_sha256"],
                id_maps_sha256=test_claim["id_maps_sha256"],
                plugin_config=test_claim["plugin_config"],
                backbone_config=test_claim["backbone_config"],
                protocol=test_claim["protocol"],
            )
            if test_claim.get("frozen_config_id") != expected_frozen_id:
                raise ValueError("test claim frozen_config_id is not canonical")
            if not isinstance(test_claim.get("dataset"), str) or not test_claim[
                "dataset"
            ].strip():
                raise ValueError("test claim dataset must be a non-empty string")
            _validated_sha256(
                test_claim.get("selection_sha256"),
                "test claim selection_sha256",
            )
            kind = protocol_data_kind(cache_metadata["protocol"])
            if kind == "holdout":
                _validated_sha256(
                    test_claim.get("holdout_assignments_sha256"),
                    "test claim holdout_assignments_sha256",
                )
            elif "holdout_assignments_sha256" in test_claim:
                raise ValueError(
                    "standard test claim must not bind holdout assignments"
                )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_record = {
        **normalized_metrics,
        "split": split,
        **_json_copy(dict(metadata or {})),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_record, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with (output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["prob", "label"])
        writer.writerows(zip(predictions, labels, strict=True))
    cache_path = output_dir / "evaluation_cache.csv"
    if cache_rows is not None:
        with cache_path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["row_index", "stu_id", "exer_id", "cpt_seq", "label", "prob"]
            )
            writer.writerows(cache_rows)
    np.save(output_dir / "mastery.npy", np.asarray(mastery))
    id_maps_path = output_dir / "id_maps.json"
    with id_maps_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_copy(dict(id_maps)), handle, sort_keys=True)
        handle.write("\n")
    if cache_rows is not None:
        mastery_path = output_dir / "mastery.npy"
        manifest = {
            "format_version": 1,
            "evaluation_split": split,
            "row_count": len(cache_rows),
            "cache_file": cache_path.name,
            "cache_sha256": sha256_file(cache_path),
            "mastery_file": mastery_path.name,
            "mastery_sha256": sha256_file(mastery_path),
            "id_maps_file": id_maps_path.name,
            "id_maps_sha256": sha256_file(id_maps_path),
            "checkpoint_sha256": cache_metadata["checkpoint_sha256"],
            "source_id_maps_sha256": cache_metadata[
                "source_id_maps_sha256"
            ],
            "protocol": cache_metadata["protocol"],
        }
        for field in ("plugin_config", "backbone_config"):
            if field in cache_metadata:
                manifest[field] = cache_metadata[field]
        if test_claim is not None:
            manifest.update(
                {
                    "frozen_config_id": test_claim["frozen_config_id"],
                    "selection_sha256": test_claim["selection_sha256"],
                    "dataset": test_claim["dataset"],
                    "test_claim": test_claim,
                    "test_claim_sha256": sha256_bytes(test_claim_bytes),
                }
            )
            if protocol_data_kind(cache_metadata["protocol"]) == "holdout":
                manifest["holdout_assignments_sha256"] = test_claim[
                    "holdout_assignments_sha256"
                ]
        _write_evaluation_cache_manifest(
            output_dir / "evaluation_cache_manifest.json",
            manifest,
        )
