from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = 1
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
DATA_FIELDS = (
    "train_interactions",
    "valid_interactions",
    "q_matrix",
    "concept_graph",
    "prerequisite_graph",
    "similarity_graph",
)
NON_CONFIG_FIELDS = {
    *DATA_FIELDS,
    "test_interactions",
    "best_checkpoint_path",
    "num_students",
    "num_exercises",
    "num_concepts",
    "best_validation_score",
    "final_loss",
    "best_val_auc",
    "best_epoch",
    "max_cuda_memory_allocated_gb",
    "valid_metrics",
    "test_metrics",
    "history",
    "log_path",
}


class CompleteModelProtocolError(RuntimeError):
    """Base error for complete-model freeze and evaluation protocol failures."""


class FrozenSelectionMismatchError(CompleteModelProtocolError):
    """Raised when a frozen selection no longer matches its bound inputs."""


class DuplicateTestEvaluationError(CompleteModelProtocolError):
    """Raised when a frozen selection already owns a test-evaluation claim."""


class EvaluationExecutionError(CompleteModelProtocolError):
    """Raised when the coverage evaluator fails or emits incomplete output."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def fingerprint_file(path: Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FrozenSelectionMismatchError(f"required file is missing: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _run_git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise FrozenSelectionMismatchError(f"Git preflight failed: {detail}")
    return completed.stdout.strip()


def _verified_route_head(repo_root: Path, supplied: str | None = None) -> str:
    actual = _run_git(repo_root, "rev-parse", "HEAD")
    if not COMMIT_PATTERN.fullmatch(actual):
        raise FrozenSelectionMismatchError("route HEAD is not a full lowercase Git SHA")
    if supplied is not None and supplied != actual:
        raise FrozenSelectionMismatchError(
            f"supplied route HEAD {supplied} does not match checkout {actual}"
        )
    return actual


def _verify_clean_route(repo_root: Path) -> None:
    status = _run_git(repo_root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise FrozenSelectionMismatchError(
            f"route worktree is dirty and cannot run the protocol: {status}"
        )


def _git_common_dir(repo_root: Path) -> Path:
    value = Path(_run_git(repo_root, "rev-parse", "--git-common-dir"))
    if not value.is_absolute():
        value = repo_root / value
    return value.resolve()


def stable_ledger_dir(repo_root: Path) -> Path:
    """Return the one repository-scoped ledger, outside the tracked worktree."""

    return _git_common_dir(Path(repo_root).resolve()) / "codex-test-ledger" / "complete-v1"


def _verify_ancestor(repo_root: Path, ancestor: str, descendant: str) -> None:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", ancestor, descendant],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise FrozenSelectionMismatchError(
            f"training route commit {ancestor} is not an ancestor of {descendant}"
        )


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenSelectionMismatchError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise FrozenSelectionMismatchError(f"{label} must be a JSON object")
    return value


def _resolve_summary_path(
    value: Any, *, summary_path: Path, repo_root: Path, field: str
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise FrozenSelectionMismatchError(f"summary field {field!r} is required")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = {
        (summary_path.parent / path).resolve(),
        (repo_root / path).resolve(),
    }
    existing = {candidate for candidate in candidates if candidate.exists()}
    if len(existing) == 1:
        return existing.pop()
    if len(existing) > 1:
        raise FrozenSelectionMismatchError(
            f"summary field {field!r} has ambiguous relative path {value!r}"
        )
    return (summary_path.parent / path).resolve()


def _summary_inputs(
    summary: Mapping[str, Any], *, summary_path: Path, repo_root: Path
) -> tuple[Path, dict[str, Path], dict[str, Any]]:
    valid_path = _resolve_summary_path(
        summary.get("valid_interactions"),
        summary_path=summary_path,
        repo_root=repo_root,
        field="valid_interactions",
    )
    validation_proxy_path = _resolve_summary_path(
        summary.get("test_interactions"),
        summary_path=summary_path,
        repo_root=repo_root,
        field="test_interactions",
    )
    if validation_proxy_path != valid_path:
        raise FrozenSelectionMismatchError(
            "validation-only summary must use valid_interactions as its test valid proxy"
        )

    checkpoint_path = _resolve_summary_path(
        summary.get("best_checkpoint_path"),
        summary_path=summary_path,
        repo_root=repo_root,
        field="best_checkpoint_path",
    )
    data_paths: dict[str, Path] = {}
    for field in DATA_FIELDS:
        value = summary.get(field)
        if value in (None, ""):
            if field in {"train_interactions", "valid_interactions", "q_matrix"}:
                raise FrozenSelectionMismatchError(
                    f"validation-only summary requires explicit {field}"
                )
            continue
        data_paths[field] = _resolve_summary_path(
            value, summary_path=summary_path, repo_root=repo_root, field=field
        )

    config = {
        str(key): value
        for key, value in summary.items()
        if str(key) not in NON_CONFIG_FIELDS
    }
    # This also rejects NaN and other non-canonical configuration values.
    canonical_sha256(config)
    return checkpoint_path, data_paths, config


def _verify_canonical_split_layout(data_paths: Mapping[str, Path]) -> None:
    required_names = {
        "train_interactions": "train.csv",
        "valid_interactions": "valid.csv",
        "q_matrix": "Q_matrix.csv",
    }
    roots = set()
    for field, expected_name in required_names.items():
        path = data_paths[field]
        if path.name != expected_name:
            raise FrozenSelectionMismatchError(
                f"{field} must be the canonical {expected_name}, got {path.name}"
            )
        roots.add(path.parent.resolve())
    if len(roots) != 1:
        raise FrozenSelectionMismatchError(
            "train.csv, valid.csv, and Q_matrix.csv must share one dataset directory"
        )


def _same_content_fingerprint(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return (
        actual.get("sha256") == expected.get("sha256")
        and actual.get("size_bytes") == expected.get("size_bytes")
    )


def _find_manifest_record(records: Any, path: Path, label: str) -> Mapping[str, Any]:
    if isinstance(records, dict):
        values = list(records.values())
    elif isinstance(records, list):
        values = records
    else:
        values = []
    matches = [
        record
        for record in values
        if isinstance(record, dict)
        and record.get("path")
        and Path(str(record["path"])).resolve() == path.resolve()
    ]
    if len(matches) != 1:
        raise FrozenSelectionMismatchError(
            f"training campaign manifest must contain exactly one {label}: {path}"
        )
    return matches[0]


def _verify_training_status(
    *,
    training_status_path: Path,
    summary_path: Path,
    checkpoint_path: Path,
    data_paths: Mapping[str, Path],
    repo_root: Path,
    protocol_route_commit: str,
) -> tuple[dict[str, Any], str]:
    status = _load_json_object(training_status_path, "training campaign status")
    if status.get("status") != "completed" or status.get("exit_code") != 0:
        raise FrozenSelectionMismatchError("training campaign did not complete successfully")
    if status.get("parameters", {}).get("seed") != 42:
        raise FrozenSelectionMismatchError("training campaign seed must be 42")
    training_route_commit = status.get("code", {}).get("route_commit")
    if not isinstance(training_route_commit, str) or not COMMIT_PATTERN.fullmatch(
        training_route_commit
    ):
        raise FrozenSelectionMismatchError("training campaign has no valid route commit")
    _verify_ancestor(repo_root, training_route_commit, protocol_route_commit)

    for path, label in (
        (summary_path, "summary output"),
        (checkpoint_path, "checkpoint output"),
    ):
        manifest = _find_manifest_record(status.get("output_hashes"), path, label)
        if not _same_content_fingerprint(fingerprint_file(path), manifest):
            raise FrozenSelectionMismatchError(
                f"{label} does not match training campaign output hash"
            )
    for field, path in data_paths.items():
        manifest = _find_manifest_record(status.get("datasets"), path, field)
        if not _same_content_fingerprint(fingerprint_file(path), manifest):
            raise FrozenSelectionMismatchError(
                f"{field} does not match training campaign dataset hash"
            )
    return status, training_route_commit


def _claim_identity_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    """Build a path-independent identity for one semantic final configuration."""

    try:
        data = {
            field: {
                "sha256": fingerprint["sha256"],
                "size_bytes": fingerprint["size_bytes"],
            }
            for field, fingerprint in record["data"].items()
        }
        return {
            "protocol_version": record["protocol_version"],
            "training_route_commit": record["training_route_commit"],
            "checkpoint_sha256": record["checkpoint"]["sha256"],
            "checkpoint_size_bytes": record["checkpoint"]["size_bytes"],
            "data": data,
            "config_sha256": record["config_sha256"],
        }
    except (KeyError, AttributeError) as exc:
        raise FrozenSelectionMismatchError(
            "frozen selection is missing semantic identity fields"
        ) from exc


def _exclusive_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite immutable file: {path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def freeze_selection(
    *,
    summary_path: Path,
    training_status_path: Path,
    dataset_name: str,
    output_path: Path,
    repo_root: Path,
    route_head: str | None = None,
    frozen_at_utc: str | None = None,
) -> dict[str, Any]:
    summary_path = Path(summary_path).expanduser().resolve()
    training_status_path = Path(training_status_path).expanduser().resolve()
    repo_root = Path(repo_root).expanduser().resolve()
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise FrozenSelectionMismatchError("dataset_name must not be empty")
    _verify_clean_route(repo_root)
    summary = _load_json_object(summary_path, "training summary")
    checkpoint_path, data_paths, config = _summary_inputs(
        summary, summary_path=summary_path, repo_root=repo_root
    )
    _verify_canonical_split_layout(data_paths)
    resolved_head = _verified_route_head(repo_root, route_head)
    _, training_route_commit = _verify_training_status(
        training_status_path=training_status_path,
        summary_path=summary_path,
        checkpoint_path=checkpoint_path,
        data_paths=data_paths,
        repo_root=repo_root,
        protocol_route_commit=resolved_head,
    )
    ledger_dir = stable_ledger_dir(repo_root)
    record: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "route_root": str(repo_root),
        "training_route_commit": training_route_commit,
        "protocol_route_commit": resolved_head,
        "dataset": dataset_name.strip(),
        "training_status": fingerprint_file(training_status_path),
        "summary": fingerprint_file(summary_path),
        "checkpoint": fingerprint_file(checkpoint_path),
        "data": {
            field: fingerprint_file(path) for field, path in data_paths.items()
        },
        "config_fields": sorted(config),
        "config_sha256": canonical_sha256(config),
        "ledger_dir": str(ledger_dir),
        "validation_proxy_verified": True,
        "frozen_at_utc": frozen_at_utc or utc_now(),
    }
    record["frozen_config_id"] = canonical_sha256(_claim_identity_payload(record))
    _exclusive_write_json(Path(output_path).expanduser().resolve(), record)
    return json.loads(json.dumps(record))


def load_and_verify_frozen_selection(
    frozen_path: Path, *, route_head: str | None = None
) -> dict[str, Any]:
    frozen_path = Path(frozen_path).expanduser().resolve()
    record = _load_json_object(frozen_path, "frozen selection")
    if record.get("protocol_version") != PROTOCOL_VERSION:
        raise FrozenSelectionMismatchError("unsupported frozen protocol version")
    frozen_config_id = record.get("frozen_config_id")
    if not isinstance(frozen_config_id, str) or not SHA256_PATTERN.fullmatch(
        frozen_config_id
    ):
        raise FrozenSelectionMismatchError("invalid frozen_config_id")
    if canonical_sha256(_claim_identity_payload(record)) != frozen_config_id:
        raise FrozenSelectionMismatchError("frozen_config_id is not canonical")

    repo_root = Path(str(record["route_root"])).resolve()
    _verify_clean_route(repo_root)
    current_head = _verified_route_head(repo_root, route_head)
    if current_head != record.get("protocol_route_commit"):
        raise FrozenSelectionMismatchError(
            "current route commit does not match frozen protocol route commit"
        )
    if Path(str(record.get("ledger_dir"))).resolve() != stable_ledger_dir(repo_root):
        raise FrozenSelectionMismatchError("frozen ledger is not repository-stable")

    training_status_record = record.get("training_status")
    summary_record = record.get("summary")
    checkpoint_record = record.get("checkpoint")
    data_record = record.get("data")
    if not all(
        isinstance(value, dict)
        for value in (
            training_status_record,
            summary_record,
            checkpoint_record,
            data_record,
        )
    ):
        raise FrozenSelectionMismatchError("frozen artifact records must be objects")
    training_status_path = Path(str(training_status_record["path"]))
    if fingerprint_file(training_status_path) != training_status_record:
        raise FrozenSelectionMismatchError(
            "training campaign status hash changed after freeze"
        )
    summary_path = Path(str(summary_record["path"]))
    if fingerprint_file(summary_path) != summary_record:
        raise FrozenSelectionMismatchError("training summary hash changed after freeze")
    summary = _load_json_object(summary_path, "training summary")
    checkpoint_path, data_paths, config = _summary_inputs(
        summary, summary_path=summary_path, repo_root=repo_root
    )
    _verify_canonical_split_layout(data_paths)
    if fingerprint_file(checkpoint_path) != checkpoint_record:
        raise FrozenSelectionMismatchError("checkpoint hash changed after freeze")
    if sorted(config) != record.get("config_fields"):
        raise FrozenSelectionMismatchError("training config fields changed after freeze")
    if canonical_sha256(config) != record.get("config_sha256"):
        raise FrozenSelectionMismatchError("training config hash changed after freeze")
    fresh_data = {field: fingerprint_file(path) for field, path in data_paths.items()}
    if fresh_data != data_record:
        raise FrozenSelectionMismatchError("training/validation data hash changed after freeze")
    _, training_route_commit = _verify_training_status(
        training_status_path=training_status_path,
        summary_path=summary_path,
        checkpoint_path=checkpoint_path,
        data_paths=data_paths,
        repo_root=repo_root,
        protocol_route_commit=current_head,
    )
    if training_route_commit != record.get("training_route_commit"):
        raise FrozenSelectionMismatchError("training route commit changed after freeze")
    return record


def claim_test_evaluation(
    *,
    frozen_record: Mapping[str, Any],
    route_head: str,
    argv: Sequence[str] | None = None,
    claimed_at_utc: str | None = None,
) -> Path:
    frozen_id = frozen_record.get("frozen_config_id")
    if not isinstance(frozen_id, str) or not SHA256_PATTERN.fullmatch(frozen_id):
        raise FrozenSelectionMismatchError("cannot claim an invalid frozen_config_id")
    if route_head != frozen_record.get("protocol_route_commit"):
        raise FrozenSelectionMismatchError("test claim route commit does not match freeze")
    ledger_dir = Path(str(frozen_record.get("ledger_dir"))).resolve()
    expected_ledger = stable_ledger_dir(Path(str(frozen_record["route_root"])))
    if ledger_dir != expected_ledger:
        raise FrozenSelectionMismatchError("test ledger is not repository-stable")
    ledger_dir.mkdir(parents=True, exist_ok=True)
    claim_path = ledger_dir / f"{frozen_id}.json"
    claim = {
        "protocol_version": PROTOCOL_VERSION,
        "frozen_config_id": frozen_id,
        "route_commit": route_head,
        "status": "claimed",
        "claimed_at_utc": claimed_at_utc or utc_now(),
        "argv": list(argv if argv is not None else sys.argv),
    }
    try:
        _exclusive_write_json(claim_path, claim)
    except FileExistsError as exc:
        raise DuplicateTestEvaluationError(
            f"test evaluation already claimed for {frozen_id}"
        ) from exc
    return claim_path


def _finish_claim(claim_path: Path, **updates: Any) -> None:
    claim = _load_json_object(claim_path, "test claim")
    claim.update(updates)
    _atomic_write_json(claim_path, claim)


def _snapshot_bound_file(
    source_record: Mapping[str, Any], destination: Path
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FrozenSelectionMismatchError(
            f"refusing to overwrite input snapshot: {destination}"
        )
    shutil.copyfile(Path(str(source_record["path"])), destination)
    snapshot = fingerprint_file(destination)
    if not _same_content_fingerprint(snapshot, source_record):
        destination.unlink(missing_ok=True)
        raise FrozenSelectionMismatchError(
            f"source changed while snapshotting: {source_record['path']}"
        )
    destination.chmod(0o400)
    return snapshot


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        view = view[written:]


def _create_memfd(name: str) -> int:
    flags = getattr(os, "MFD_ALLOW_SEALING", 0x0002) | getattr(
        os, "MFD_CLOEXEC", 0x0001
    )
    python_wrapper = getattr(os, "memfd_create", None)
    if callable(python_wrapper):
        return int(python_wrapper(name, flags))
    libc = ctypes.CDLL(None, use_errno=True)
    function = libc.memfd_create
    function.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    descriptor = int(function(name.encode("utf-8"), flags))
    if descriptor < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return descriptor


def _seal_memfd(descriptor: int) -> None:
    seals = (
        getattr(fcntl, "F_SEAL_SEAL", 0x0001)
        | getattr(fcntl, "F_SEAL_SHRINK", 0x0002)
        | getattr(fcntl, "F_SEAL_GROW", 0x0004)
        | getattr(fcntl, "F_SEAL_WRITE", 0x0008)
    )
    fcntl.fcntl(descriptor, getattr(fcntl, "F_ADD_SEALS", 1033), seals)
    os.lseek(descriptor, 0, os.SEEK_SET)


def _sealed_memfd_from_bytes(name: str, payload: bytes) -> tuple[int, dict[str, Any]]:
    descriptor = _create_memfd(name)
    try:
        _write_all(descriptor, payload)
        _seal_memfd(descriptor)
    except Exception:
        os.close(descriptor)
        raise
    return descriptor, {
        "path": f"/proc/self/fd/{descriptor}",
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def _sealed_memfd_from_file(
    name: str, path: Path, expected: Mapping[str, Any]
) -> tuple[int, dict[str, Any]]:
    descriptor = _create_memfd(name)
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                _write_all(descriptor, chunk)
                digest.update(chunk)
                size_bytes += len(chunk)
        record = {
            "path": f"/proc/self/fd/{descriptor}",
            "size_bytes": size_bytes,
            "sha256": digest.hexdigest(),
        }
        if not _same_content_fingerprint(record, expected):
            raise FrozenSelectionMismatchError(
                f"snapshot changed before sealing: {path}"
            )
        _seal_memfd(descriptor)
        return descriptor, record
    except Exception:
        os.close(descriptor)
        raise


def create_sealed_git_archive(
    repo_root: Path, commit: str
) -> tuple[int, dict[str, Any]]:
    """Materialize committed code in a sealed zip-importable memory file."""

    completed = subprocess.run(
        ["git", "-C", str(repo_root), "archive", "--format=zip", commit],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise FrozenSelectionMismatchError(
            f"cannot archive protocol route commit {commit}: "
            f"{completed.stderr.decode(errors='replace').strip()}"
        )
    return _sealed_memfd_from_bytes("decoupled-cd-code.zip", completed.stdout)


def _committed_evaluator_prefix(python_executable: str) -> list[str]:
    return [
        python_executable,
        "-P",
        "-m",
        "scripts.evaluate_coverage_slice",
    ]


def _prepare_common_snapshot(
    frozen: Mapping[str, Any], output_dir: Path
) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    snapshot_dir = output_dir / "input_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    snapshots: dict[str, dict[str, Any]] = {}
    snapshots["training_status"] = _snapshot_bound_file(
        frozen["training_status"], snapshot_dir / "training_status.json"
    )
    original_summary = snapshot_dir / "training_summary.json"
    snapshots["training_summary"] = _snapshot_bound_file(
        frozen["summary"], original_summary
    )
    summary_payload = _load_json_object(original_summary, "snapshotted summary")
    snapshots["checkpoint"] = _snapshot_bound_file(
        frozen["checkpoint"], snapshot_dir / "checkpoint.pt"
    )
    for field, record in frozen["data"].items():
        suffix = Path(str(record["path"])).suffix or ".bin"
        snapshots[field] = _snapshot_bound_file(
            record, snapshot_dir / f"{field}{suffix}"
        )
    return snapshot_dir, summary_payload, snapshots


def _write_evaluation_summary(
    *,
    snapshot_dir: Path,
    summary_payload: Mapping[str, Any],
    snapshots: Mapping[str, Mapping[str, Any]],
    test_snapshot_path: Path,
) -> Path:
    rewritten = dict(summary_payload)
    rewritten["best_checkpoint_path"] = str(snapshots["checkpoint"]["path"])
    for field in DATA_FIELDS:
        if field in snapshots:
            rewritten[field] = str(snapshots[field]["path"])
    rewritten["test_interactions"] = str(test_snapshot_path)
    output = snapshot_dir / "summary.json"
    _atomic_write_json(output, rewritten)
    output.chmod(0o400)
    return output


def _immutable_evaluation_inputs(
    *,
    summary_payload: Mapping[str, Any],
    snapshots: Mapping[str, Mapping[str, Any]],
    split: str,
) -> tuple[list[int], dict[str, dict[str, Any]]]:
    descriptors: list[int] = []
    immutable: dict[str, dict[str, Any]] = {}
    try:
        for field in ("checkpoint", *DATA_FIELDS):
            if field not in snapshots:
                continue
            descriptor, record = _sealed_memfd_from_file(
                field,
                Path(str(snapshots[field]["path"])),
                snapshots[field],
            )
            descriptors.append(descriptor)
            immutable[field] = record
        if split == "test":
            descriptor, record = _sealed_memfd_from_file(
                "test_interactions",
                Path(str(snapshots["test_interactions"]["path"])),
                snapshots["test_interactions"],
            )
            descriptors.append(descriptor)
            immutable["test_interactions"] = record
        else:
            immutable["test_interactions"] = immutable["valid_interactions"]

        summary = dict(summary_payload)
        summary["best_checkpoint_path"] = immutable["checkpoint"]["path"]
        for field in DATA_FIELDS:
            if field in immutable:
                summary[field] = immutable[field]["path"]
        summary["test_interactions"] = immutable["test_interactions"]["path"]
        summary_bytes = (
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        descriptor, record = _sealed_memfd_from_bytes(
            "evaluation-summary.json", summary_bytes
        )
        descriptors.append(descriptor)
        immutable["evaluation_summary"] = record
        return descriptors, immutable
    except Exception:
        for descriptor in descriptors:
            os.close(descriptor)
        raise


def _resolve_test_source(
    value: Path, *, frozen: Mapping[str, Any]
) -> Path:
    path = Path(value).expanduser().resolve()
    valid_path = Path(str(frozen["data"]["valid_interactions"]["path"])).resolve()
    if path.name != "test.csv" or path.parent != valid_path.parent or path == valid_path:
        raise FrozenSelectionMismatchError(
            "test evaluation requires canonical test.csv beside frozen valid.csv"
        )
    return path


def _scope_rows(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("slices")
    if not isinstance(rows, list):
        raise EvaluationExecutionError("coverage output has no slices list")
    matches: dict[str, dict[str, Any]] = {}
    for output_name, scope_name in (("overall", "overall"), ("zero", "bucket:zero")):
        candidates = [
            row
            for row in rows
            if isinstance(row, dict) and row.get("scope") == scope_name
        ]
        if len(candidates) != 1:
            raise EvaluationExecutionError(
                f"coverage output requires exactly one {scope_name!r} row"
            )
        matches[output_name] = candidates[0]
    return matches


def evaluate_frozen_selection(
    *,
    frozen_path: Path,
    split: str,
    output_dir: Path,
    test_interactions: Path | None = None,
    evaluator_path: Path | None = None,
    python_executable: str | None = None,
    device: str = "auto",
    gpus: str | None = None,
    route_head: str | None = None,
) -> dict[str, Any]:
    if split not in {"valid", "test"}:
        raise ValueError("split must be 'valid' or 'test'")
    if split == "valid" and test_interactions is not None:
        raise ValueError("valid evaluation does not accept a real test path")
    if split == "test" and test_interactions is None:
        raise ValueError("test evaluation requires --test-interactions")

    frozen = load_and_verify_frozen_selection(frozen_path, route_head=route_head)
    resolved_head = str(frozen["protocol_route_commit"])
    claim_path: Path | None = None
    test_fingerprint: dict[str, Any] | None = None
    open_descriptors: list[int] = []
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir, summary_payload, snapshots = _prepare_common_snapshot(
        frozen, output_dir
    )
    test_source = (
        _resolve_test_source(test_interactions, frozen=frozen)
        if split == "test" and test_interactions is not None
        else None
    )
    try:
        if split == "test":
            assert test_source is not None
            # No content read or file-open of test.csv may precede this claim.
            claim_path = claim_test_evaluation(
                frozen_record=frozen,
                route_head=resolved_head,
                argv=sys.argv,
            )
            test_snapshot_path = snapshot_dir / "test_interactions.csv"
            shutil.copyfile(test_source, test_snapshot_path)
            snapshot_fingerprint = fingerprint_file(test_snapshot_path)
            test_snapshot_path.chmod(0o400)
            test_fingerprint = {
                **snapshot_fingerprint,
                "source_path": str(test_source),
            }
            snapshots["test_interactions"] = snapshot_fingerprint
        else:
            test_snapshot_path = Path(str(snapshots["valid_interactions"]["path"]))

        evaluation_summary = _write_evaluation_summary(
            snapshot_dir=snapshot_dir,
            summary_payload=summary_payload,
            snapshots=snapshots,
            test_snapshot_path=test_snapshot_path,
        )
        input_descriptors, immutable_inputs = _immutable_evaluation_inputs(
            summary_payload=summary_payload,
            snapshots=snapshots,
            split=split,
        )
        open_descriptors.extend(input_descriptors)
        coverage_path = output_dir / "coverage.json"
        slices_path = output_dir / "coverage_slices.csv"
        summary_csv_path = output_dir / "coverage_summary.csv"
        stdout_path = output_dir / "evaluator.stdout.log"
        stderr_path = output_dir / "evaluator.stderr.log"
        python_command = python_executable or sys.executable
        if evaluator_path is not None:
            command = [python_command, str(Path(evaluator_path).expanduser().resolve())]
        else:
            _verify_clean_route(Path(str(frozen["route_root"])))
            if _verified_route_head(Path(str(frozen["route_root"]))) != resolved_head:
                raise FrozenSelectionMismatchError(
                    "protocol route commit changed before code archival"
                )
            code_descriptor, code_archive = create_sealed_git_archive(
                Path(str(frozen["route_root"])), resolved_head
            )
            open_descriptors.append(code_descriptor)
            immutable_inputs["code_archive"] = code_archive
            command = _committed_evaluator_prefix(python_command)
        command.extend([
            "--dataset-name",
            str(frozen["dataset"]),
            "--summary",
            immutable_inputs["evaluation_summary"]["path"],
            "--split",
            split,
            "--train-interactions",
            immutable_inputs["train_interactions"]["path"],
            "--valid-interactions",
            immutable_inputs["valid_interactions"]["path"],
            "--test-interactions",
            immutable_inputs["test_interactions"]["path"],
            "--q-matrix",
            immutable_inputs["q_matrix"]["path"],
            "--device",
            device,
            "--output",
            str(coverage_path),
            "--slice-csv",
            str(slices_path),
            "--summary-csv",
            str(summary_csv_path),
        ])
        if "concept_graph" in immutable_inputs:
            command.extend(
                ["--concept-graph", immutable_inputs["concept_graph"]["path"]]
            )
        if gpus is not None:
            command.extend(["--gpus", gpus])
        environment = os.environ.copy()
        if claim_path is not None:
            environment["COMPLETE_MODEL_TEST_CLAIM"] = str(claim_path)
        if "code_archive" in immutable_inputs:
            environment["PYTHONPATH"] = immutable_inputs["code_archive"]["path"]
            environment["PYTHONSAFEPATH"] = "1"
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=environment,
            cwd=output_dir,
            pass_fds=tuple(open_descriptors),
            check=False,
        )
        for descriptor in open_descriptors:
            os.close(descriptor)
        open_descriptors.clear()
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise EvaluationExecutionError(
                f"coverage evaluator exited with {completed.returncode}: "
                f"{completed.stderr.strip()}"
            )

        coverage = _load_json_object(coverage_path, "coverage output")
        scopes = _scope_rows(coverage)
        overall_path = output_dir / "overall.json"
        zero_path = output_dir / "zero.json"
        _atomic_write_json(overall_path, scopes["overall"])
        _atomic_write_json(zero_path, scopes["zero"])
        artifact_paths = (
            coverage_path,
            slices_path,
            summary_csv_path,
            overall_path,
            zero_path,
            stdout_path,
            stderr_path,
        )
        artifact_hashes = {path.name: sha256_file(path) for path in artifact_paths}
        result: dict[str, Any] = {
            "protocol_version": PROTOCOL_VERSION,
            "frozen_config_id": frozen["frozen_config_id"],
            "route_commit": resolved_head,
            "split": split,
            "evaluator_argv": command,
            "scopes": scopes,
            "artifact_hashes": artifact_hashes,
            "input_snapshot_hashes": {
                name: record["sha256"] for name, record in immutable_inputs.items()
            },
            "completed_at_utc": utc_now(),
        }
        if test_fingerprint is not None:
            result["test_interactions"] = test_fingerprint
        result_path = output_dir / "evaluation_record.json"
        _atomic_write_json(result_path, result)
        if claim_path is not None:
            assert test_fingerprint is not None
            _finish_claim(
                claim_path,
                status="completed",
                completed_at_utc=result["completed_at_utc"],
                test_interactions_path=test_fingerprint["source_path"],
                test_interactions_sha256=test_fingerprint["sha256"],
                scopes=scopes,
                artifact_hashes=artifact_hashes,
                evaluation_record_path=str(result_path),
                evaluation_record_sha256=sha256_file(result_path),
            )
        return result
    except Exception as exc:
        if claim_path is not None:
            _finish_claim(
                claim_path,
                status="failed",
                failed_at_utc=utc_now(),
                error=f"{type(exc).__name__}: {exc}",
            )
        raise
    finally:
        for descriptor in open_descriptors:
            os.close(descriptor)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze complete-model validation choices and enforce test-once."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--summary", type=Path, required=True)
    freeze_parser.add_argument("--training-status", type=Path, required=True)
    freeze_parser.add_argument("--dataset-name", required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    freeze_parser.add_argument("--repo-root", type=Path, default=Path.cwd())

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--frozen", type=Path, required=True)
    evaluate_parser.add_argument("--split", choices=["valid", "test"], required=True)
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument("--test-interactions", type=Path, default=None)
    evaluate_parser.add_argument("--device", default="auto")
    evaluate_parser.add_argument("--gpus", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        if args.command == "freeze":
            payload = freeze_selection(
                summary_path=args.summary,
                training_status_path=args.training_status,
                dataset_name=args.dataset_name,
                output_path=args.output,
                repo_root=args.repo_root,
            )
        else:
            payload = evaluate_frozen_selection(
                frozen_path=args.frozen,
                split=args.split,
                output_dir=args.output_dir,
                test_interactions=args.test_interactions,
                device=args.device,
                gpus=args.gpus,
            )
    except (CompleteModelProtocolError, FileExistsError, ValueError) as exc:
        print(f"protocol error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
