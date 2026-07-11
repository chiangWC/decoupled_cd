from __future__ import annotations

import json
import hashlib
import math
import os
import fcntl
from contextlib import contextmanager
from pathlib import Path
import re
import secrets
import subprocess
from typing import Any, Mapping, Sequence

from models.unified_v2_spec import UnifiedArchitectureSpec
from scripts.unified_cohort import load_verified_cohort
from scripts.unified_dataset_audit import canonical_sha256


COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
ARCHITECTURES = {
    "b0": ("prior", "mask"),
    "m2": ("graph", "mask"),
    "m2-m3": ("graph", "coverage"),
}
BASELINE_METRICS = (
    "standard_overall_auc",
    "holdout_overall_auc",
    "zero_auc",
    "ordinary_doa",
    "weighted_doa",
)
DATASET_DIRECTORIES = {
    "ASSIST09": ("assist_09", "assist_09_chold_v2"),
    "ASSIST17": ("assist_17", "assist_17_chold_v2"),
    "MOOCRadar": ("moocradar", "moocradar_chold_v2"),
    "XES3G5M": ("xes3g5m", "xes3g5m_chold_v2"),
}


def _architecture_spec(architecture: str) -> UnifiedArchitectureSpec:
    try:
        inference, composer = ARCHITECTURES[architecture]
    except KeyError as error:
        raise ValueError(f"unknown unified architecture: {architecture}") from error
    return UnifiedArchitectureSpec(inference=inference, composer=composer)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _route_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    head = completed.stdout.strip()
    if completed.returncode != 0 or not COMMIT_PATTERN.fullmatch(head):
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"cannot resolve exact route HEAD: {detail}")
    return head


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_bytes(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _exclusive_json(path: Path, payload: object) -> None:
    _exclusive_bytes(path, _json_bytes(payload))


def _unlink_fsync(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _fsync_directory(path.parent)


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.parent / f".{path.name}.{secrets.token_hex(16)}.tmp"
    _exclusive_bytes(temporary, _json_bytes(payload))
    os.replace(temporary, path)
    _fsync_directory(path.parent)


@contextmanager
def _controller_lock(state_dir: Path):
    lock_path = state_dir / "controller.lock"
    with lock_path.open("r+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _load_state(state_dir: Path, repo_root: Path) -> dict[str, Any]:
    state = _load_json(state_dir / "state.json", label="controller state")
    registered_head = state.get("route_commit")
    actual_head = _route_head(repo_root.resolve())
    if registered_head != actual_head:
        raise ValueError(
            f"route HEAD mismatch: registered {registered_head}, actual {actual_head}"
        )
    cohort = load_verified_cohort(state_dir / "cohort.json")
    if (
        cohort.get("cohort_sha256") != state.get("cohort_sha256")
        or cohort.get("dataset_ids") != state.get("dataset_ids")
    ):
        raise ValueError("registered cohort does not match controller state")
    manifest = _load_json(
        state_dir / "manifest.json", label="registered architecture manifest"
    )
    if (
        canonical_sha256(manifest) != state.get("manifest_sha256")
        or _architecture_spec(str(state.get("architecture"))).manifest() != manifest
        or _architecture_spec(str(state.get("architecture"))).fingerprint()
        != state.get("architecture_fingerprint")
    ):
        raise ValueError("registered manifest does not match controller state")
    baseline = _load_json(state_dir / "baseline.json", label="registered baseline")
    if canonical_sha256(baseline) != state.get("baseline_sha256"):
        raise ValueError("registered baseline hash does not match controller state")
    return state


def _capability_path(state_dir: Path, capability: Mapping[str, object]) -> Path:
    return state_dir / "issued" / (
        f"{int(capability['counter']):06d}-{capability['dataset_id']}-"
        f"{capability['split_id']}-{capability['nonce']}.json"
    )


def _expected_data_paths(
    *,
    dataset_id: str,
    split_id: str,
    data_root: Path,
) -> list[str]:
    try:
        standard_dir, holdout_dir = DATASET_DIRECTORIES[dataset_id]
    except KeyError as error:
        raise ValueError(f"unknown controller dataset: {dataset_id}") from error
    if split_id not in {"standard", "holdout"}:
        raise ValueError(f"unknown controller split: {split_id}")
    directory = data_root.resolve() / (
        standard_dir if split_id == "standard" else holdout_dir
    )
    paths = [
        directory / "train.csv",
        directory / "valid.csv",
        directory / "Q_matrix.csv",
    ]
    if split_id == "holdout":
        paths.append(directory / "student_concept_holdout_assignments.csv")
    return [str(path) for path in paths]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file_record(
    record: object,
    path: Path,
    *,
    label: str,
) -> dict[str, object]:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} has no immutable hash record")
    expected_path = path.resolve()
    if Path(str(record.get("path"))).resolve() != expected_path:
        raise ValueError(f"{label} does not match registered summary path")
    if not expected_path.is_file():
        raise ValueError(f"{label} is not a regular file: {expected_path}")
    size = expected_path.stat().st_size
    digest = _sha256_file(expected_path)
    if record.get("size_bytes") != size or record.get("sha256") != digest:
        raise ValueError(f"{label} output hash mismatch")
    return {
        "path": str(expected_path),
        "size_bytes": size,
        "sha256": digest,
    }


def _find_path_record(records: object, path: Path, *, label: str) -> Mapping[str, Any]:
    values = records.values() if isinstance(records, Mapping) else records
    if not isinstance(values, (list, tuple)):
        values = list(values) if values is not None else []
    matches = [
        record
        for record in values
        if isinstance(record, Mapping)
        and Path(str(record.get("path"))).resolve() == path.resolve()
    ]
    if len(matches) != 1:
        raise ValueError(f"{label} must have exactly one registered path record")
    return matches[0]


def _finite_metric(summary: Mapping[str, Any], field: str) -> float:
    value = summary.get(field)
    if (
        type(value) is not float
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"validation summary metric is invalid: {field}")
    return value


def _has_test_token(value: str) -> bool:
    expanded = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    expanded = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", expanded)
    return any(
        token.lower() == "test" for token in re.findall(r"[A-Za-z0-9]+", expanded)
    )


def _contains_test_reference(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            _has_test_token(str(key))
            or _contains_test_reference(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_test_reference(item) for item in value)
    return isinstance(value, str) and _has_test_token(value)


def _validate_baseline_rows(
    payload: Mapping[str, Any],
    *,
    dataset_ids: Sequence[str],
    cohort_sha256: str,
) -> list[dict[str, object]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("baseline rows must be a JSON list")
    actual_ids = [row.get("dataset_id") for row in rows if isinstance(row, dict)]
    if actual_ids != list(dataset_ids) or len(actual_ids) != len(rows):
        raise ValueError("baseline dataset order must match the frozen cohort")
    fingerprints: set[str] = set()
    normalized: list[dict[str, object]] = []
    for row in rows:
        assert isinstance(row, dict)
        dataset_id = str(row["dataset_id"])
        if row.get("cohort_sha256") != cohort_sha256:
            raise ValueError(f"baseline cohort mismatch: {dataset_id}")
        fingerprint = row.get("architecture_fingerprint")
        if not isinstance(fingerprint, str) or not re.fullmatch(
            r"[0-9a-f]{64}", fingerprint
        ):
            raise ValueError(f"baseline fingerprint is invalid: {dataset_id}")
        fingerprints.add(fingerprint)
        for field in BASELINE_METRICS:
            value = row.get(field)
            if (
                type(value) is not float
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"baseline {dataset_id}.{field} is invalid")
        normalized.append(dict(row))
    if len(fingerprints) != 1:
        raise ValueError("baseline rows have mixed architecture fingerprints")
    return normalized


def initialize_controller(
    *,
    state_dir: Path,
    repo_root: Path,
    cohort_path: Path,
    manifest_path: Path,
    architecture: str,
    baseline_rows_path: Path,
) -> dict[str, object]:
    state_dir = state_dir.resolve()
    repo_root = repo_root.resolve()
    cohort = load_verified_cohort(cohort_path)
    dataset_ids = cohort.get("dataset_ids")
    cohort_hash = cohort.get("cohort_sha256")
    if not isinstance(dataset_ids, list) or not all(
        isinstance(dataset_id, str) for dataset_id in dataset_ids
    ):
        raise ValueError("frozen cohort dataset order is invalid")
    if not isinstance(cohort_hash, str):
        raise ValueError("frozen cohort SHA-256 is invalid")

    manifest = _load_json(manifest_path, label="architecture manifest")
    spec = _architecture_spec(architecture)
    if manifest != spec.manifest():
        raise ValueError("architecture manifest does not match architecture")

    baseline_payload = _load_json(baseline_rows_path, label="baseline rows")
    baseline_rows = _validate_baseline_rows(
        baseline_payload,
        dataset_ids=dataset_ids,
        cohort_sha256=cohort_hash,
    )
    registered_baseline = {"rows": baseline_rows}
    route_commit = _route_head(repo_root)
    state: dict[str, object] = {
        "schema_version": 1,
        "controller_id": secrets.token_hex(32),
        "route_commit": route_commit,
        "cohort_sha256": cohort_hash,
        "manifest_sha256": canonical_sha256(manifest),
        "architecture": architecture,
        "architecture_fingerprint": spec.fingerprint(),
        "dataset_ids": dataset_ids,
        "baseline_sha256": canonical_sha256(registered_baseline),
        "cursor": 0,
        "successes": 0,
        "zero_delta_threshold_seen": False,
        "issuance_counter": 0,
        "active_pair": None,
        "complete": False,
        "blocked": False,
        "global_pass": None,
    }

    os.mkdir(state_dir, 0o700)
    try:
        for directory in ("issued", "consumed", "proofs"):
            os.mkdir(state_dir / directory, 0o700)
            _fsync_directory(state_dir)
        _exclusive_bytes(state_dir / "controller.lock", b"")
        _exclusive_json(state_dir / "cohort.json", cohort)
        _exclusive_json(state_dir / "manifest.json", manifest)
        _exclusive_json(state_dir / "baseline.json", registered_baseline)
        _exclusive_json(state_dir / "state.json", state)
        _fsync_directory(state_dir)
        _fsync_directory(state_dir.parent)
    except BaseException:
        # Initialization is exclusive; retain no half-created controller.
        for path in sorted(state_dir.rglob("*"), reverse=True):
            if path.is_dir():
                path.rmdir()
            else:
                path.unlink()
        state_dir.rmdir()
        raise
    return state


def _load_consumed_pair(
    state_dir: Path,
    active_pair: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    filenames = active_pair.get("capability_files")
    if not isinstance(filenames, Mapping):
        raise ValueError("active pair has no registered capability files")
    records: dict[str, dict[str, Any]] = {}
    for split_id in ("standard", "holdout"):
        filename = filenames.get(split_id)
        if not isinstance(filename, str):
            raise ValueError(f"active pair has no {split_id} capability record")
        consumed_path = state_dir / "consumed" / filename
        if not consumed_path.is_file():
            if (state_dir / "issued" / filename).exists():
                raise ValueError(f"{split_id} capability has not been consumed")
            raise ValueError(f"{split_id} capability registry record is missing")
        record = _load_json(consumed_path, label=f"consumed {split_id} capability")
        if (
            record.get("counter") != active_pair.get("counter")
            or record.get("dataset_id") != active_pair.get("dataset_id")
            or record.get("split_id") != split_id
        ):
            raise ValueError(f"consumed {split_id} capability identity mismatch")
        records[split_id] = record
    return records


def _verify_registered_input(
    record: object,
    *,
    registered_path: Path,
    registered_payload: Mapping[str, Any],
    label: str,
) -> dict[str, object]:
    if not isinstance(record, Mapping):
        raise ValueError(f"outer status has no {label} immutable input")
    path = Path(str(record.get("path"))).resolve()
    fingerprint = _verify_file_record(record, path, label=label)
    payload = _load_json(path, label=label)
    if payload != registered_payload:
        raise ValueError(f"outer status {label} content mismatch")
    # Also ensure the controller-owned copy still has the same semantics.
    if _load_json(registered_path, label=f"registered {label}") != payload:
        raise ValueError(f"controller-owned {label} content mismatch")
    return fingerprint


def _verify_split_proof(
    *,
    state_dir: Path,
    state: Mapping[str, Any],
    consumption: Mapping[str, Any],
) -> dict[str, Any]:
    split_id = str(consumption.get("split_id"))
    dataset_id = str(consumption.get("dataset_id"))
    attempt_dir = Path(str(consumption.get("attempt_dir"))).resolve()
    summary_path = Path(str(consumption.get("summary_path"))).resolve()
    if summary_path.parent != attempt_dir:
        raise ValueError("consumed capability summary is outside its attempt dir")
    status_path = attempt_dir / "status.json"
    status = _load_json(status_path, label=f"{split_id} outer status")
    if status.get("status") != "completed" or status.get("exit_code") != 0:
        raise ValueError(f"{split_id} outer attempt did not complete")
    if status.get("parameters", {}).get("seed") != 42:
        raise ValueError(f"{split_id} outer attempt seed must be 42")
    if status.get("code", {}).get("route_commit") != state.get("route_commit"):
        raise ValueError(f"{split_id} outer status route commit mismatch")

    registered_manifest = _load_json(
        state_dir / "manifest.json", label="registered architecture manifest"
    )
    registered_cohort = load_verified_cohort(state_dir / "cohort.json")
    immutable_inputs = status.get("immutable_inputs")
    if not isinstance(immutable_inputs, Mapping):
        raise ValueError(f"{split_id} outer status has no immutable inputs")
    manifest_record = immutable_inputs.get("architecture_manifest")
    manifest_fingerprint = _verify_registered_input(
        manifest_record,
        registered_path=state_dir / "manifest.json",
        registered_payload=registered_manifest,
        label="architecture manifest",
    )
    if not isinstance(manifest_record, Mapping) or manifest_record.get(
        "architecture_fingerprint"
    ) != state.get("architecture_fingerprint"):
        raise ValueError("outer status architecture fingerprint mismatch")
    cohort_record = immutable_inputs.get("cohort")
    cohort_fingerprint = _verify_registered_input(
        cohort_record,
        registered_path=state_dir / "cohort.json",
        registered_payload=registered_cohort,
        label="cohort",
    )
    if not isinstance(cohort_record, Mapping) or cohort_record.get(
        "cohort_sha256"
    ) != state.get("cohort_sha256"):
        raise ValueError("outer status cohort SHA-256 mismatch")

    expected_data_paths = consumption.get("data_paths")
    if not isinstance(expected_data_paths, list):
        raise ValueError("consumed capability has no registered dataset paths")
    dataset_records = status.get("datasets")
    if not isinstance(dataset_records, list):
        raise ValueError("outer status has no dataset hash manifest")
    if {str(Path(path).resolve()) for path in expected_data_paths} != {
        str(Path(str(record.get("path"))).resolve())
        for record in dataset_records
        if isinstance(record, Mapping)
    }:
        raise ValueError("outer status dataset paths do not match consumed capability")
    verified_datasets = [
        _verify_file_record(
            _find_path_record(dataset_records, Path(path), label="dataset"),
            Path(path),
            label="dataset hash",
        )
        for path in expected_data_paths
    ]

    output_record = _find_path_record(
        status.get("output_hashes"), summary_path, label="registered summary path"
    )
    summary_fingerprint = _verify_file_record(
        output_record,
        summary_path,
        label="validation summary",
    )
    summary = _load_json(summary_path, label=f"{split_id} validation summary")
    if _contains_test_reference(summary):
        raise ValueError("test metric/path is forbidden in validation summary")
    if (
        summary.get("dataset_id") != dataset_id
        or summary.get("split_id") != split_id
        or summary.get("architecture") != state.get("architecture")
        or summary.get("architecture_manifest") != registered_manifest
        or summary.get("architecture_fingerprint")
        != state.get("architecture_fingerprint")
        or summary.get("cohort_sha256") != state.get("cohort_sha256")
        or summary.get("controller_id") != state.get("controller_id")
        or summary.get("controller_route_commit") != state.get("route_commit")
        or summary.get("capability_counter") != consumption.get("counter")
        or summary.get("capability_nonce") != consumption.get("nonce")
        or summary.get("seed") != 42
        or summary.get("evaluation_input_role") != "valid"
    ):
        raise ValueError(f"{split_id} validation summary binding mismatch")
    metrics = {
        field: _finite_metric(summary, field)
        for field in ("overall_auc", "zero_auc", "ordinary_doa", "weighted_doa")
    }
    return {
        "split_id": split_id,
        "capability": {
            field: consumption[field]
            for field in (
                "controller_id",
                "route_commit",
                "counter",
                "dataset_id",
                "split_id",
                "nonce",
            )
        },
        "attempt_dir": str(attempt_dir),
        "status": {
            "path": str(status_path),
            "size_bytes": status_path.stat().st_size,
            "sha256": _sha256_file(status_path),
        },
        "summary": summary_fingerprint,
        "manifest": manifest_fingerprint,
        "cohort": cohort_fingerprint,
        "datasets": verified_datasets,
        "metrics": metrics,
    }


def _advance_active_pair(
    *,
    state_dir: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    active_pair = state.get("active_pair")
    if not isinstance(active_pair, Mapping):
        return state
    consumed = _load_consumed_pair(state_dir, active_pair)
    split_proofs = {
        split_id: _verify_split_proof(
            state_dir=state_dir,
            state=state,
            consumption=consumed[split_id],
        )
        for split_id in ("standard", "holdout")
    }
    baseline_payload = _load_json(
        state_dir / "baseline.json", label="registered baseline"
    )
    if canonical_sha256(baseline_payload) != state.get("baseline_sha256"):
        raise ValueError("registered baseline hash changed before progress use")
    baseline_rows = baseline_payload.get("rows")
    cursor = state.get("cursor")
    if not isinstance(baseline_rows, list) or type(cursor) is not int:
        raise ValueError("registered baseline/controller cursor is invalid")
    baseline = baseline_rows[cursor]
    if not isinstance(baseline, Mapping):
        raise ValueError("registered baseline row is invalid")
    standard = split_proofs["standard"]["metrics"]
    holdout = split_proofs["holdout"]["metrics"]
    deltas = {
        "standard_overall_auc": standard["overall_auc"]
        - baseline["standard_overall_auc"],
        "holdout_overall_auc": holdout["overall_auc"]
        - baseline["holdout_overall_auc"],
        "weighted_doa": standard["weighted_doa"] - baseline["weighted_doa"],
        "zero_auc": holdout["zero_auc"] - baseline["zero_auc"],
        "ordinary_doa": standard["ordinary_doa"] - baseline["ordinary_doa"],
    }
    joint_success = (
        deltas["standard_overall_auc"] >= 0.0
        and deltas["holdout_overall_auc"] >= 0.0
        and deltas["weighted_doa"] >= 0.0
        and deltas["zero_auc"] > 0.0
        and deltas["ordinary_doa"] > 0.0
    )
    dataset_id = str(active_pair.get("dataset_id"))
    proof: dict[str, Any] = {
        "schema_version": 1,
        "controller_id": state["controller_id"],
        "route_commit": state["route_commit"],
        "counter": active_pair["counter"],
        "dataset_id": dataset_id,
        "baseline_sha256": state["baseline_sha256"],
        "split_proofs": split_proofs,
        "deltas": deltas,
        "joint_success": joint_success,
        "zero_delta_at_least_0.001": deltas["zero_auc"] >= 0.001,
    }
    proof_path = state_dir / "proofs" / (
        f"{int(active_pair['counter']):06d}-{dataset_id}.json"
    )
    if proof_path.exists():
        if _load_json(proof_path, label="registered progress proof") != proof:
            raise ValueError("existing progress proof does not match outer artifacts")
    else:
        _exclusive_json(proof_path, proof)
    state["cursor"] = cursor + 1
    if joint_success:
        state["successes"] = int(state.get("successes", 0)) + 1
    if deltas["zero_auc"] >= 0.001:
        state["zero_delta_threshold_seen"] = True
    state["active_pair"] = None
    dataset_ids = state.get("dataset_ids")
    assert isinstance(dataset_ids, list)
    remaining = len(dataset_ids) - int(state["cursor"])
    if int(state["cursor"]) == len(dataset_ids):
        state["complete"] = True
        state["global_pass"] = (
            int(state["successes"]) >= 3
            and bool(state["zero_delta_threshold_seen"])
        )
    elif int(state["successes"]) + remaining < 3:
        state["blocked"] = True
    _atomic_json(state_dir / "state.json", state)
    return state


def authorize_next(
    *,
    state_dir: Path,
    repo_root: Path,
    output_path: Path,
) -> dict[str, object]:
    state_dir = state_dir.resolve()
    output_path = output_path.resolve()
    with _controller_lock(state_dir):
        state = _load_state(state_dir, repo_root)
        if state.get("blocked"):
            raise RuntimeError("primary cohort is unreachable under registered stop rule")
        if state.get("complete"):
            raise ValueError("controller iteration is already complete")
        _exclusive_bytes(output_path, b"")
        original_state = json.loads(json.dumps(state))
        original_issued = {path.name for path in (state_dir / "issued").glob("*.json")}
        original_proofs = {path.name for path in (state_dir / "proofs").glob("*.json")}
        preserve_progress = False
        committed = False
        try:
            if state.get("active_pair") is not None:
                state = _advance_active_pair(state_dir=state_dir, state=state)
            if state.get("blocked"):
                preserve_progress = True
                raise RuntimeError(
                    "primary cohort is unreachable under registered stop rule"
                )
            if state.get("complete"):
                completion: dict[str, object] = {
                    "schema_version": 1,
                    "controller_id": state["controller_id"],
                    "route_commit": state["route_commit"],
                    "complete": True,
                    "global_pass": state["global_pass"],
                    "successes": state["successes"],
                    "dataset_count": len(state["dataset_ids"]),
                    "zero_delta_threshold_seen": state[
                        "zero_delta_threshold_seen"
                    ],
                }
                _atomic_json(output_path, completion)
                committed = True
                return completion
            dataset_ids = state.get("dataset_ids")
            cursor = state.get("cursor")
            if not isinstance(dataset_ids, list) or type(cursor) is not int:
                raise ValueError("controller progress state is invalid")
            if not 0 <= cursor < len(dataset_ids):
                raise ValueError("controller cursor is outside the frozen cohort")
            dataset_id = dataset_ids[cursor]
            if not isinstance(dataset_id, str):
                raise ValueError("controller dataset identity is invalid")
            counter = int(state.get("issuance_counter", 0)) + 1
            common: dict[str, object] = {
                "schema_version": 1,
                "controller_id": state["controller_id"],
                "route_commit": state["route_commit"],
                "counter": counter,
                "dataset_id": dataset_id,
            }
            capabilities: dict[str, dict[str, object]] = {}
            for split_id in ("standard", "holdout"):
                capability = {
                    **common,
                    "split_id": split_id,
                    "nonce": secrets.token_hex(32),
                }
                _exclusive_json(
                    _capability_path(state_dir, capability), capability
                )
                capabilities[split_id] = capability
            token: dict[str, object] = {
                **common,
                "capabilities": capabilities,
            }
            state["issuance_counter"] = counter
            state["active_pair"] = {
                "counter": counter,
                "dataset_id": dataset_id,
                "capability_files": {
                    split_id: _capability_path(state_dir, capability).name
                    for split_id, capability in capabilities.items()
                },
            }
            _atomic_json(state_dir / "state.json", state)
            _atomic_json(output_path, token)
            committed = True
            return token
        except BaseException:
            if not preserve_progress:
                for directory, original_names in (
                    (state_dir / "issued", original_issued),
                    (state_dir / "proofs", original_proofs),
                ):
                    for path in directory.glob("*.json"):
                        if path.name not in original_names:
                            _unlink_fsync(path)
                _atomic_json(state_dir / "state.json", original_state)
            raise
        finally:
            if not committed:
                _unlink_fsync(output_path)


def consume_split_capability(
    *,
    state_dir: Path,
    repo_root: Path,
    token_path: Path,
    dataset_id: str,
    split_id: str,
    architecture: str,
    data_root: Path,
    raw_output: Path,
    attempt_dir: Path,
) -> dict[str, object]:
    state_dir = state_dir.resolve()
    with _controller_lock(state_dir):
        state = _load_state(state_dir, repo_root)
        token = _load_json(token_path.resolve(), label="capability token")
        capabilities = token.get("capabilities")
        if not isinstance(capabilities, Mapping):
            raise ValueError("capability token has no split registry references")
        capability = capabilities.get(split_id)
        if not isinstance(capability, Mapping):
            raise ValueError(f"capability token has no {split_id} capability")
        capability = dict(capability)
        expected_top = {
            field: capability.get(field)
            for field in (
                "controller_id",
                "route_commit",
                "counter",
                "dataset_id",
            )
        }
        if any(token.get(field) != value for field, value in expected_top.items()):
            raise ValueError("capability token envelope does not match capability")
        active_pair = state.get("active_pair")
        if not isinstance(active_pair, Mapping):
            raise ValueError("controller has no active issued pair")
        if (
            capability.get("controller_id") != state.get("controller_id")
            or capability.get("route_commit") != state.get("route_commit")
            or capability.get("counter") != active_pair.get("counter")
            or capability.get("dataset_id") != active_pair.get("dataset_id")
            or capability.get("dataset_id") != dataset_id
            or capability.get("split_id") != split_id
            or state.get("architecture") != architecture
        ):
            raise ValueError("capability is stale or outside the active registry pair")
        issued_path = _capability_path(state_dir, capability)
        if not issued_path.is_file():
            raise ValueError("capability is not present in the issued registry")
        registered = _load_json(issued_path, label="issued capability registry")
        if registered != capability:
            raise ValueError("capability does not exactly match the issued registry")

        attempt_dir = attempt_dir.resolve()
        output_path = (
            raw_output.resolve()
            if raw_output.is_absolute()
            else attempt_dir / raw_output
        )
        consumed_path = state_dir / "consumed" / issued_path.name
        os.replace(issued_path, consumed_path)
        _fsync_directory(state_dir / "issued")
        _fsync_directory(state_dir / "consumed")
        consumption: dict[str, object] = {
            **capability,
            "architecture": state["architecture"],
            "architecture_fingerprint": state["architecture_fingerprint"],
            "architecture_manifest": _load_json(
                state_dir / "manifest.json",
                label="registered architecture manifest",
            ),
            "cohort_sha256": state["cohort_sha256"],
            "attempt_dir": str(attempt_dir),
            "summary_path": str(output_path.resolve()),
            "data_paths": _expected_data_paths(
                dataset_id=dataset_id,
                split_id=split_id,
                data_root=data_root,
            ),
        }
        _atomic_json(consumed_path, consumption)
        return consumption
