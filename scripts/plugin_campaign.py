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


class CandidateStore:
    """Append-only per-epoch checkpoint and validation artifact store."""

    def __init__(
        self,
        *,
        output_dir: Path,
        model_name: str,
        plugin_config: Mapping[str, Any],
        backbone_config: Mapping[str, Any],
        protocol: Mapping[str, Any],
    ) -> None:
        self.output_dir = Path(output_dir)
        self.model_name = str(model_name)
        self.plugin_config = _json_copy(dict(plugin_config))
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


def _validated_selection(
    *,
    selection_path: Path,
    checkpoint_path: Path,
    plugin_config: Mapping[str, Any],
    backbone_config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    supplied_frozen_config_id: str | None,
    artifact_snapshot: EvaluationArtifactSnapshot | None = None,
) -> tuple[str, str, str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
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
    normalized_config = _json_copy(dict(plugin_config))
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
    if selection.get("plugin_config") != normalized_config:
        raise FrozenConfigMismatchError(
            "selection plugin_config does not match evaluation CLI"
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
    return (
        config_id,
        checkpoint_sha256,
        id_maps_sha256,
        normalized_config,
        normalized_backbone,
        normalized_protocol,
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
) -> Path:
    (
        frozen_config_id,
        checkpoint_sha256,
        id_maps_sha256,
        normalized_config,
        normalized_backbone,
        normalized_protocol,
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
        "selection_path": str(Path(selection_path).resolve()),
        "route_head": resolved_head,
        "claimed_at_utc": claimed_at,
        "argv": list(argv if argv is not None else sys.argv),
    }
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
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(record, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
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
) -> Any:
    if split not in {"valid", "test"}:
        raise ValueError("evaluation split must be 'valid' or 'test'")
    if split == "test":
        if ledger_dir is None or selection_path is None or route_root is None:
            raise ValueError(
                "test evaluation requires ledger_dir, selection_path, and route_root"
            )
        claim_test_evaluation(
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
        )
    return load_resources()


def write_evaluation_artifacts(
    *,
    output_dir: Path,
    split: str,
    metrics: Mapping[str, Any],
    predictions: Sequence[Any],
    labels: Sequence[Any],
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
    np.save(output_dir / "mastery.npy", np.asarray(mastery))
    with (output_dir / "id_maps.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_copy(dict(id_maps)), handle, sort_keys=True)
        handle.write("\n")
