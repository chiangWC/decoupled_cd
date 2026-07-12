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
import stat
import subprocess
import sys
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.unified_v2_spec import UnifiedArchitectureSpec
from scripts.unified_cohort import load_verified_cohort
from scripts.unified_dataset_audit import canonical_sha256


COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
NONCE_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ARCHITECTURES = {
    "a0": ("prior", 0.0),
    "a1": ("lowrank", 1.0),
}
CAMPAIGN_ID = "unified-mastery-20260712"
CONTROLLER_SCHEMA_VERSION = 3
RECIPE_COUNTS = {
    "ASSIST09": 1,
    "ASSIST17": 2,
    "NIPS34": 1,
    "MOOCRadar": 3,
    "XES3G5M": 2,
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
OUTER_RUNNER_OVERRIDE: Path | None = None
TRUSTED_GIT = Path("/usr/bin/git")


def _sanitized_subprocess_env() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("MKL_THREADING_LAYER", None)
    environment.pop("MKL_SERVICE_FORCE_INTEL", None)
    return environment


def _strict_git_env() -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "LANG": "C",
        "HOME": "/nonexistent",
        "XDG_CONFIG_HOME": "/nonexistent",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
    }


def _read_git_metadata(path: Path, *, label: str) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > 4096:
            raise ValueError(f"invalid {label}: {path}")
        data = handle.read(4097)
        after = os.fstat(handle.fileno())
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if len(data) > 4096 or before_identity != after_identity:
        raise ValueError(f"unstable {label}: {path}")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"invalid UTF-8 in {label}: {path}") from error


def _resolve_git_dir(repo_root: Path) -> Path:
    marker = repo_root / ".git"
    if marker.is_symlink():
        raise ValueError("route .git marker must not be a symlink")
    if marker.is_dir():
        return marker.resolve(strict=True)
    if not marker.is_file():
        raise ValueError(f"route has no .git metadata: {repo_root}")
    content = _read_git_metadata(marker, label="linked-worktree .git file")
    lines = content.splitlines()
    if len(lines) != 1 or not lines[0].startswith("gitdir: "):
        raise ValueError("linked-worktree .git file is malformed")
    raw_git_dir = lines[0][len("gitdir: ") :]
    if not raw_git_dir or "\x00" in raw_git_dir:
        raise ValueError("linked-worktree gitdir path is invalid")
    candidate = Path(raw_git_dir)
    git_dir = (
        candidate if candidate.is_absolute() else marker.parent / candidate
    ).resolve(strict=True)
    if not git_dir.is_dir():
        raise ValueError("linked-worktree gitdir is not a directory")

    backpointer_text = _read_git_metadata(
        git_dir / "gitdir", label="linked-worktree gitdir backpointer"
    ).strip()
    if not backpointer_text or "\x00" in backpointer_text:
        raise ValueError("linked-worktree gitdir backpointer is invalid")
    backpointer = Path(backpointer_text)
    if not backpointer.is_absolute():
        backpointer = git_dir / backpointer
    if backpointer.resolve(strict=True) != marker.resolve(strict=True):
        raise ValueError("linked-worktree gitdir backpointer mismatch")

    common_text = _read_git_metadata(
        git_dir / "commondir", label="linked-worktree common-dir pointer"
    ).strip()
    if not common_text or "\x00" in common_text:
        raise ValueError("linked-worktree common-dir pointer is invalid")
    common = Path(common_text)
    if not common.is_absolute():
        common = git_dir / common
    if not common.resolve(strict=True).is_dir():
        raise ValueError("linked-worktree common-dir is not a directory")
    return git_dir


def _trusted_git_command(
    repo_root: Path,
    git_dir: Path,
    *arguments: str,
) -> list[str]:
    if not TRUSTED_GIT.is_file() or not os.access(TRUSTED_GIT, os.X_OK):
        raise ValueError(f"trusted Git executable is unavailable: {TRUSTED_GIT}")
    return [
        str(TRUSTED_GIT),
        "--no-optional-locks",
        f"--git-dir={git_dir}",
        f"--work-tree={repo_root}",
        "-c",
        f"core.worktree={repo_root}",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.hooksPath=/dev/null",
        *arguments,
    ]


def _verify_normal_git_index(
    repo_root: Path,
    git_dir: Path,
    environment: Mapping[str, str],
) -> None:
    for flag, label in (
        ("-v", "assume-unchanged/skip-worktree"),
        ("-f", "fsmonitor-valid"),
    ):
        completed = subprocess.run(
            _trusted_git_command(
                repo_root,
                git_dir,
                "ls-files",
                flag,
                "-z",
            ),
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=False,
            env=dict(environment),
        )
        if completed.returncode != 0:
            detail_bytes = completed.stderr or completed.stdout
            detail = detail_bytes.decode("utf-8", errors="replace").strip()
            raise ValueError(f"cannot verify route index flags: {detail}")
        records = completed.stdout.split(b"\x00")
        if not records or records[-1] != b"":
            raise ValueError("route index listing is not NUL terminated")
        for record in records[:-1]:
            if len(record) < 3 or record[:2] != b"H ":
                raise ValueError(
                    f"route index has abnormal {label} state"
                )


@dataclass(frozen=True)
class LegacyArchitectureIdentity:
    """Immutable v3 identity retained after the production spec moved to v4."""

    completion: str

    def manifest(self) -> dict[str, str | int]:
        if self.completion not in {"prior", "lowrank"}:
            raise ValueError("legacy completion must be prior or lowrank")
        return {
            "mastery_estimator": "evidence-parameter",
            "completion": self.completion,
            "cognitive_decoder": "neuralcdm-monotonic",
            "behavior_model": "conditional-simplex",
            "mastery_output": "student-concept",
            "version": 3,
            "modules": {
                "prior": "m1-prior-m3-m4",
                "lowrank": "m1-lowrank-m3-m4",
            }[self.completion],
        }

    def fingerprint(self) -> str:
        return canonical_sha256(self.manifest())


def legacy_architecture_manifest(architecture: str) -> dict[str, str | int]:
    return _architecture_spec(architecture).manifest()


def _architecture_spec(architecture: str) -> LegacyArchitectureIdentity:
    try:
        completion, _ = ARCHITECTURES[architecture]
    except KeyError as error:
        raise ValueError(f"unknown unified architecture: {architecture}") from error
    return LegacyArchitectureIdentity(completion=completion)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _validated_cohort_expectations(
    value: object,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "dataset_ids",
        "a0_fingerprints",
        "comparator_audit_sha256",
        "dataset_audit_sha256",
    }:
        raise ValueError("registered primary cohort expectations are invalid")
    dataset_ids = value.get("dataset_ids")
    fingerprints = value.get("a0_fingerprints")
    comparator_hash = value.get("comparator_audit_sha256")
    dataset_hash = value.get("dataset_audit_sha256")
    if (
        not isinstance(dataset_ids, list)
        or len(dataset_ids) != 3
        or len(set(dataset_ids)) != 3
        or not all(isinstance(dataset_id, str) and dataset_id for dataset_id in dataset_ids)
        or not isinstance(fingerprints, Mapping)
        or set(fingerprints) != set(dataset_ids)
        or any(
            not isinstance(fingerprint, str)
            or NONCE_PATTERN.fullmatch(fingerprint) is None
            for fingerprint in fingerprints.values()
        )
        or not isinstance(comparator_hash, str)
        or NONCE_PATTERN.fullmatch(comparator_hash) is None
        or not isinstance(dataset_hash, str)
        or NONCE_PATTERN.fullmatch(dataset_hash) is None
    ):
        raise ValueError("registered primary cohort expectations are invalid")
    return {
        "dataset_ids": list(dataset_ids),
        "a0_fingerprints": dict(fingerprints),
        "comparator_audit_sha256": comparator_hash,
        "dataset_audit_sha256": dataset_hash,
    }


def _load_registered_cohort(
    path: Path, *, expectations: object = None
) -> dict[str, Any]:
    payload = _load_json(path, label="registered cohort")
    if payload.get("schema_version") != 2:
        if expectations is not None:
            raise ValueError("schema version 1 cannot bind primary cohort expectations")
        return load_verified_cohort(path)
    trusted = _validated_cohort_expectations(expectations)
    return load_verified_cohort(
        path,
        expected_dataset_ids=trusted["dataset_ids"],
        expected_a0_fingerprints=trusted["a0_fingerprints"],
        expected_comparator_audit_sha256=str(
            trusted["comparator_audit_sha256"]
        ),
        expected_dataset_audit_sha256=str(trusted["dataset_audit_sha256"]),
    )


def _route_head(repo_root: Path) -> str:
    repo_root = repo_root.resolve(strict=True)
    if not repo_root.is_dir():
        raise ValueError(f"route root is not a directory: {repo_root}")
    git_dir = _resolve_git_dir(repo_root)
    environment = _strict_git_env()
    top_level = subprocess.run(
        _trusted_git_command(
            repo_root, git_dir, "rev-parse", "--show-toplevel"
        ),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    top_level_text = top_level.stdout.strip()
    if top_level.returncode != 0 or not top_level_text:
        detail = top_level.stderr.strip() or top_level_text
        raise ValueError(f"cannot resolve route top-level: {detail}")
    try:
        resolved_top_level = Path(top_level_text).resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        detail = top_level.stderr.strip() or top_level.stdout.strip()
        raise ValueError(f"cannot resolve route top-level: {detail}") from error
    if resolved_top_level != repo_root:
        raise ValueError(
            f"route top-level mismatch: expected {repo_root}, "
            f"actual {resolved_top_level}"
        )
    _verify_normal_git_index(repo_root, git_dir, environment)
    completed = subprocess.run(
        _trusted_git_command(repo_root, git_dir, "rev-parse", "HEAD"),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    head = completed.stdout.strip()
    if completed.returncode != 0 or not COMMIT_PATTERN.fullmatch(head):
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"cannot resolve exact route HEAD: {detail}")
    status = subprocess.run(
        _trusted_git_command(
            repo_root,
            git_dir,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if status.returncode != 0:
        detail = status.stderr.strip() or status.stdout.strip()
        raise ValueError(f"cannot verify clean route: {detail}")
    if status.stdout:
        raise ValueError("route is dirty; controller requires exact committed bytes")
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
    if (
        state.get("schema_version") != CONTROLLER_SCHEMA_VERSION
        or state.get("campaign_id") != CAMPAIGN_ID
        or state.get("split_seed") != 2024
    ):
        raise ValueError(
            "controller state schema/campaign/split seed is stale; initialize fresh state"
        )
    registered_head = state.get("route_commit")
    actual_head = _route_head(repo_root.resolve())
    if registered_head != actual_head:
        raise ValueError(
            f"route HEAD mismatch: registered {registered_head}, actual {actual_head}"
        )
    cohort = _load_registered_cohort(
        state_dir / "cohort.json",
        expectations=state.get("cohort_expectations"),
    )
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
    validation_data = state.get("validation_data")
    if (
        not isinstance(validation_data, Mapping)
        or canonical_sha256(validation_data) != state.get("validation_data_sha256")
    ):
        raise ValueError("registered validation data manifest is invalid")
    for records in validation_data.values():
        if not isinstance(records, list) or not all(
            isinstance(record, Mapping) for record in records
        ):
            raise ValueError("registered validation data records are invalid")
        for record in records:
            snapshot = _snapshot_file(Path(str(record.get("path"))))
            try:
                _verify_snapshot_record(record, snapshot, label="validation data")
            except ValueError as error:
                raise ValueError("registered validation data hash mismatch") from error
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


def _stat_identity(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _snapshot_file(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    digest = hashlib.sha256()
    size = 0
    with resolved.open("rb") as handle:
        before = os.fstat(handle.fileno())
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(handle.fileno())
    if _stat_identity(before) != _stat_identity(after) or size != before.st_size:
        raise ValueError(f"file changed while snapshotting: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": size,
        "sha256": digest.hexdigest(),
    }


def _snapshot_json(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, object]]:
    resolved = path.resolve()
    with resolved.open("rb") as handle:
        before = os.fstat(handle.fileno())
        data = handle.read()
        after = os.fstat(handle.fileno())
    if _stat_identity(before) != _stat_identity(after) or len(data) != before.st_size:
        raise ValueError(f"{label} changed while snapshotting: {resolved}")
    try:
        payload = json.loads(data)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot parse {label}: {resolved}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload, {
        "path": str(resolved),
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _copy_snapshot_file(source: Path, destination: Path) -> dict[str, object]:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        destination,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o600,
    )
    digest = hashlib.sha256()
    size = 0
    try:
        with source.open("rb") as source_handle, os.fdopen(
            descriptor, "wb"
        ) as destination_handle:
            before = os.fstat(source_handle.fileno())
            for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
                destination_handle.write(chunk)
            after = os.fstat(source_handle.fileno())
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        if _stat_identity(before) != _stat_identity(after) or size != before.st_size:
            raise ValueError(f"validation source changed while copying: {source}")
    except BaseException:
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        raise
    _fsync_directory(destination.parent)
    return {
        "path": str(destination.resolve()),
        "size_bytes": size,
        "sha256": digest.hexdigest(),
    }


def _verify_snapshot_record(
    record: object,
    snapshot: Mapping[str, object],
    *,
    label: str,
) -> None:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} has no immutable hash record")
    if Path(str(record.get("path"))).resolve() != Path(str(snapshot["path"])):
        raise ValueError(f"{label} does not match registered summary path")
    if (
        record.get("size_bytes") != snapshot["size_bytes"]
        or record.get("sha256") != snapshot["sha256"]
    ):
        raise ValueError(f"{label} output hash mismatch")


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


def _finite_nonnegative(summary: Mapping[str, Any], field: str) -> float:
    value = summary.get(field)
    if type(value) not in {int, float} or not math.isfinite(value) or value < 0.0:
        raise ValueError(f"validation summary value is invalid: {field}")
    return float(value)


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
    if _contains_test_reference(payload):
        raise ValueError("test metric/path is forbidden in registered baseline")
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
        parameter_count = row.get("parameter_count")
        if type(parameter_count) is not int or parameter_count < 0:
            raise ValueError(
                f"baseline {dataset_id}.parameter_count is invalid"
            )
        normalized.append(dict(row))
    if len(fingerprints) != 1:
        raise ValueError("baseline rows have mixed architecture fingerprints")
    return normalized


def _validate_exploration_guards(
    payload: Mapping[str, Any],
    *,
    dataset_ids: Sequence[str],
    cohort_sha256: str,
) -> list[dict[str, object]]:
    if _contains_test_reference(payload):
        raise ValueError("test metric/path is forbidden in exploration guards")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("exploration guards must be a JSON row list")
    actual_ids = [row.get("dataset_id") for row in rows if isinstance(row, dict)]
    if actual_ids != list(dataset_ids) or len(actual_ids) != len(rows):
        raise ValueError("exploration guard order must match the provisional registry")
    required_sources = {
        "standard_overall_auc", "holdout_overall_auc", "zero_auc"
    }
    normalized: list[dict[str, object]] = []
    for row in rows:
        assert isinstance(row, dict)
        dataset_id = str(row["dataset_id"])
        if row.get("cohort_sha256") != cohort_sha256:
            raise ValueError(f"exploration registry mismatch: {dataset_id}")
        for field in required_sources:
            value = row.get(field)
            if (
                type(value) is not float
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"exploration guard {dataset_id}.{field} is invalid")
        sources = row.get("comparator_sources")
        if (
            not isinstance(sources, Mapping)
            or set(sources) != required_sources
            or not all(isinstance(value, str) and value for value in sources.values())
        ):
            raise ValueError(f"exploration comparator sources are invalid: {dataset_id}")
        normalized.append(dict(row))
    return normalized


def initialize_controller(
    *,
    state_dir: Path,
    repo_root: Path,
    cohort_path: Path,
    manifest_path: Path,
    architecture: str,
    baseline_rows_path: Path,
    data_root: Path,
    artifact_root: Path,
    exploration: bool = False,
    cohort_expectations: Mapping[str, object] | None = None,
) -> dict[str, object]:
    state_dir = state_dir.resolve()
    repo_root = repo_root.resolve()
    normalized_cohort_expectations = (
        _validated_cohort_expectations(cohort_expectations)
        if cohort_expectations is not None
        else None
    )
    cohort = _load_registered_cohort(
        cohort_path, expectations=normalized_cohort_expectations
    )
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
    validator = (
        _validate_exploration_guards if exploration else _validate_baseline_rows
    )
    baseline_rows = validator(
        baseline_payload, dataset_ids=dataset_ids, cohort_sha256=cohort_hash
    )
    registered_baseline = {"rows": baseline_rows}
    if normalized_cohort_expectations is not None:
        expected_fingerprints = normalized_cohort_expectations["a0_fingerprints"]
        assert isinstance(expected_fingerprints, Mapping)
        if any(
            row.get("architecture_fingerprint")
            != expected_fingerprints.get(str(row.get("dataset_id")))
            for row in baseline_rows
        ):
            raise ValueError("baseline A0 fingerprints mismatch primary cohort expectations")
    recipe_indices: dict[str, int] = {}
    registered_recipes: dict[str, Mapping[str, object] | None] = {}
    for row in baseline_rows:
        dataset_id = str(row["dataset_id"])
        recipe_index = row.get("recipe_index", 0)
        if (
            type(recipe_index) is not int
            or not 0 <= recipe_index < RECIPE_COUNTS[dataset_id]
        ):
            raise ValueError(f"baseline recipe index is invalid: {dataset_id}")
        recipe = row.get("numerical_recipe")
        if architecture == "a1" and not isinstance(recipe, Mapping):
            raise ValueError(f"A1 requires frozen A0 recipe: {dataset_id}")
        recipe_indices[dataset_id] = recipe_index
        registered_recipes[dataset_id] = (
            dict(recipe)
            if architecture == "a1" and isinstance(recipe, Mapping)
            else None
        )
    data_root = data_root.resolve()
    artifact_root = artifact_root.resolve()
    if _has_test_token(str(data_root)) or _has_test_token(str(artifact_root)):
        raise ValueError("test paths are forbidden in registered controller roots")
    validation_sources: dict[str, list[Path]] = {}
    for dataset_id in dataset_ids:
        for split_id in ("standard", "holdout"):
            key = f"{dataset_id}:{split_id}"
            paths = _expected_data_paths(
                dataset_id=dataset_id,
                split_id=split_id,
                data_root=data_root,
            )
            if any(_has_test_token(path) for path in paths):
                raise ValueError("test path is forbidden in validation data")
            validation_sources[key] = [Path(path).resolve() for path in paths]
    route_commit = _route_head(repo_root)

    os.mkdir(state_dir, 0o700)
    try:
        for directory in ("issued", "consumed", "proofs", "data"):
            os.mkdir(state_dir / directory, 0o700)
            _fsync_directory(state_dir)
        copied_data_root = state_dir / "data"
        validation_data: dict[str, list[dict[str, object]]] = {}
        for key, sources in validation_sources.items():
            validation_data[key] = [
                _copy_snapshot_file(
                    source,
                    copied_data_root / source.relative_to(data_root),
                )
                for source in sources
            ]
        state: dict[str, object] = {
            "schema_version": CONTROLLER_SCHEMA_VERSION,
            "campaign_id": CAMPAIGN_ID,
            "mode": "a0_exploration" if exploration else "candidate_gate",
            "controller_id": secrets.token_hex(32),
            "route_commit": route_commit,
            "cohort_sha256": cohort_hash,
            "cohort_expectations": normalized_cohort_expectations,
            "manifest_sha256": canonical_sha256(manifest),
            "architecture": architecture,
            "architecture_fingerprint": spec.fingerprint(),
            "dataset_ids": dataset_ids,
            "baseline_sha256": canonical_sha256(registered_baseline),
            "source_data_root": str(data_root),
            "data_root": str(copied_data_root),
            "artifact_root": str(artifact_root),
            "validation_data": validation_data,
            "validation_data_sha256": canonical_sha256(validation_data),
            "cursor": 0,
            "successes": 0,
            "overall_failures": 0,
            "recipe_indices": recipe_indices,
            "registered_recipes": registered_recipes,
            "selected_recipes": {},
            "exploration_overall_passed": {},
            "zero_delta_threshold_seen": False,
            "split_seed": 2024,
            "issuance_counter": 0,
            "proof_sha256_by_counter": {},
            "active_pair": None,
            "pending_issuance": None,
            "launch": None,
            "complete": False,
            "blocked": False,
            "global_pass": None,
        }
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
    payload, fingerprint = _snapshot_json(path, label=label)
    _verify_snapshot_record(record, fingerprint, label=label)
    if payload != registered_payload:
        raise ValueError(f"outer status {label} content mismatch")
    # Also ensure the controller-owned copy still has the same semantics.
    registered_copy, _ = _snapshot_json(
        registered_path, label=f"registered {label}"
    )
    if registered_copy != payload:
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
    status, status_fingerprint = _snapshot_json(
        status_path, label=f"{split_id} outer status"
    )
    if status.get("status") != "completed" or status.get("exit_code") != 0:
        raise ValueError(f"{split_id} outer attempt did not complete")
    if status.get("parameters", {}).get("seed") != 42:
        raise ValueError(f"{split_id} outer attempt seed must be 42")
    if (
        status.get("parameters", {}).get("split_seed") != 2024
        or status.get("parameters", {}).get("split_seed")
        != consumption.get("split_seed")
    ):
        raise ValueError(f"{split_id} outer attempt split seed must be 2024")
    if status.get("code", {}).get("route_commit") != state.get("route_commit"):
        raise ValueError(f"{split_id} outer status route commit mismatch")
    launch = state.get("launch")
    commands = launch.get("commands") if isinstance(launch, Mapping) else None
    outer_command = commands.get(split_id) if isinstance(commands, Mapping) else None
    invocation = status.get("invocation")
    if not isinstance(outer_command, list) or "--" not in outer_command:
        raise ValueError(f"{split_id} controller launch command is invalid")
    expected_child_command = outer_command[outer_command.index("--") + 1 :]
    if (
        not isinstance(invocation, Mapping)
        or invocation.get("command") != expected_child_command
    ):
        raise ValueError(f"{split_id} outer status command mismatch")

    registered_manifest = _load_json(
        state_dir / "manifest.json", label="registered architecture manifest"
    )
    registered_cohort = _load_registered_cohort(
        state_dir / "cohort.json",
        expectations=state.get("cohort_expectations"),
    )
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
    if not all(isinstance(record, Mapping) for record in dataset_records):
        raise ValueError("outer status dataset manifest entries must be objects")
    actual_dataset_paths = [
        str(Path(str(record.get("path"))).resolve()) for record in dataset_records
    ]
    if (
        len(actual_dataset_paths) != len(expected_data_paths)
        or len(set(actual_dataset_paths)) != len(actual_dataset_paths)
        or set(actual_dataset_paths)
        != {str(Path(path).resolve()) for path in expected_data_paths}
    ):
        raise ValueError("outer status dataset paths do not match consumed capability")
    verified_datasets = []
    for path in expected_data_paths:
        snapshot = _snapshot_file(Path(path))
        _verify_snapshot_record(
            _find_path_record(dataset_records, Path(path), label="dataset"),
            snapshot,
            label="dataset hash",
        )
        verified_datasets.append(snapshot)

    output_record = _find_path_record(
        status.get("output_hashes"), summary_path, label="registered summary path"
    )
    summary, summary_fingerprint = _snapshot_json(
        summary_path, label=f"{split_id} validation summary"
    )
    _verify_snapshot_record(
        output_record, summary_fingerprint, label="validation summary"
    )
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
        or summary.get("split_seed") != consumption.get("split_seed")
        or summary.get("evaluation_input_role") != "valid"
    ):
        raise ValueError(f"{split_id} validation summary binding mismatch")
    mastery_shape = summary.get("mastery_shape")
    if (
        not isinstance(mastery_shape, list)
        or len(mastery_shape) != 2
        or any(type(value) is not int or value <= 0 for value in mastery_shape)
    ):
        raise ValueError(f"{split_id} mastery_shape must contain two positive integers")
    _finite_nonnegative(summary, "final_loss")
    numerical_recipe = summary.get("numerical_recipe")
    if not isinstance(numerical_recipe, Mapping):
        raise ValueError(f"{split_id} numerical_recipe is missing")
    mastery_loss_weight = _finite_nonnegative(
        numerical_recipe, "mastery_loss_weight"
    )
    if mastery_loss_weight <= 0.0:
        raise ValueError(f"{split_id} mastery_loss_weight must be positive")
    parameter_count = summary.get("parameter_count")
    if type(parameter_count) is not int or parameter_count < 0:
        raise ValueError(f"{split_id} parameter_count must be a nonnegative integer")
    if summary.get("recipe_index") != consumption.get("recipe_index"):
        raise ValueError(f"{split_id} numerical recipe index mismatch")
    expected_recipe = consumption.get("numerical_recipe")
    if isinstance(expected_recipe, Mapping) and numerical_recipe != expected_recipe:
        raise ValueError(f"{split_id} numerical recipe changed from frozen A0")
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
                "split_seed",
            )
        },
        "attempt_dir": str(attempt_dir),
        "status": status_fingerprint,
        "summary": summary_fingerprint,
        "manifest": manifest_fingerprint,
        "cohort": cohort_fingerprint,
        "datasets": verified_datasets,
        "metrics": metrics,
        "recipe_index": summary["recipe_index"],
        "numerical_recipe": dict(numerical_recipe),
        "parameter_count": parameter_count,
    }


def _validated_replay_invocation(
    *, state_dir: Path, state: Mapping[str, Any], consumption: Mapping[str, Any]
) -> list[str]:
    status, _ = _snapshot_json(
        Path(str(consumption["attempt_dir"])) / "status.json",
        label=f"{consumption['split_id']} replay outer status",
    )
    invocation = status.get("invocation")
    command = invocation.get("command") if isinstance(invocation, Mapping) else None
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise ValueError("replay outer status command is invalid")
    if len(command) < 3 or Path(command[1]).name != "run_unified_validation.py":
        raise ValueError("replay outer status command is not the registered runner")
    repo_root = Path(command[1]).resolve().parents[1]
    capability = command[command.index("--capability") + 1] if "--capability" in command else ""
    expected = [
        command[0],
        str(repo_root / "scripts" / "run_unified_validation.py"),
        "run-split",
        "--dataset-id", str(consumption["dataset_id"]),
        "--split-id", str(consumption["split_id"]),
        "--architecture", str(state["architecture"]),
        "--architecture-fingerprint", str(state["architecture_fingerprint"]),
        "--cohort-sha256", str(state["cohort_sha256"]),
        "--recipe-index", str(consumption["recipe_index"]),
        "--data-root", str(Path(str(state["data_root"])).resolve()),
        "--controller-state-dir", str(state_dir.resolve()),
        "--repo-root", str(repo_root),
        "--capability", capability,
        "--output", "validation-summary.json",
    ]
    if command != expected:
        raise ValueError("replay outer status command binding mismatch")
    return command


def _verified_replay_state(
    state_dir: Path,
    *,
    expected_mode: str,
    cohort_expectations: Mapping[str, object] | None = None,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    state = _load_json(state_dir / "state.json", label="controller replay state")
    state_expectations = state.get("cohort_expectations")
    if state_expectations is not None and cohort_expectations is not None:
        if _validated_cohort_expectations(state_expectations) != _validated_cohort_expectations(
            cohort_expectations
        ):
            raise ValueError("controller primary cohort expectations mismatch")
    effective_expectations = (
        cohort_expectations if cohort_expectations is not None else state_expectations
    )
    if (
        state.get("schema_version") != CONTROLLER_SCHEMA_VERSION
        or state.get("campaign_id") != CAMPAIGN_ID
        or state.get("mode") != expected_mode
        or state.get("complete") is not True
        or state.get("active_pair") is not None
        or state.get("launch") is not None
        or state.get("split_seed") != 2024
    ):
        raise ValueError(f"controller replay state is not a completed {expected_mode}")
    if effective_expectations is not None:
        state["cohort_expectations"] = _validated_cohort_expectations(
            effective_expectations
        )
    cohort = _load_registered_cohort(
        state_dir / "cohort.json", expectations=effective_expectations
    )
    manifest = _load_json(state_dir / "manifest.json", label="replay manifest")
    baseline = _load_json(state_dir / "baseline.json", label="replay baseline")
    if (
        cohort.get("cohort_sha256") != state.get("cohort_sha256")
        or cohort.get("dataset_ids") != state.get("dataset_ids")
        or canonical_sha256(manifest) != state.get("manifest_sha256")
        or _architecture_spec(str(state.get("architecture"))).manifest() != manifest
        or _architecture_spec(str(state.get("architecture"))).fingerprint()
        != state.get("architecture_fingerprint")
        or canonical_sha256(baseline) != state.get("baseline_sha256")
    ):
        raise ValueError("controller replay registered state artifact mismatch")
    validation_data = state.get("validation_data")
    if (
        not isinstance(validation_data, Mapping)
        or canonical_sha256(validation_data) != state.get("validation_data_sha256")
    ):
        raise ValueError("controller replay validation registry mismatch")
    for records in validation_data.values():
        if not isinstance(records, list):
            raise ValueError("controller replay validation records are invalid")
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("controller replay validation record is invalid")
            _verify_snapshot_record(
                record,
                _snapshot_file(Path(str(record.get("path")))),
                label="replay validation data",
            )
    baseline_rows = baseline.get("rows")
    if not isinstance(baseline_rows, list):
        raise ValueError("controller replay baseline rows are invalid")
    return state, {
        str(row["dataset_id"]): row
        for row in baseline_rows
        if isinstance(row, Mapping)
    }


def _registered_replay_pairs(
    state_dir: Path, state: Mapping[str, Any]
) -> dict[int, dict[str, dict[str, Any]]]:
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for path in sorted((state_dir / "consumed").glob("*.json")):
        consumption = _load_json(path, label="replay consumed capability")
        if not isinstance(consumption, dict):
            raise ValueError("replay consumed capability is not an object")
        counter = consumption.get("counter")
        split_id = consumption.get("split_id")
        dataset_id = consumption.get("dataset_id")
        state_fields = (
            "controller_id",
            "route_commit",
            "architecture",
            "architecture_fingerprint",
            "cohort_sha256",
            "split_seed",
        )
        if (
            type(counter) is not int
            or split_id not in {"standard", "holdout"}
            or dataset_id not in state.get("dataset_ids", [])
            or path.name != _capability_path(state_dir, consumption).name
            or any(consumption.get(field) != state.get(field) for field in state_fields)
            or consumption.get("data_paths") != _expected_data_paths(
                dataset_id=str(dataset_id),
                split_id=str(split_id),
                data_root=Path(str(state["data_root"])),
            )
        ):
            raise ValueError("replay consumed capability registration mismatch")
        grouped.setdefault(counter, {})[str(split_id)] = consumption
    issuance_counter = state.get("issuance_counter")
    if (
        type(issuance_counter) is not int
        or sorted(grouped) != list(range(1, issuance_counter + 1))
    ):
        raise ValueError("replay issuance registry is missing or non-contiguous")
    return grouped


def _registered_replay_proof_anchor(
    state: Mapping[str, Any], *, issuance_counter: int
) -> tuple[Mapping[str, str], list[Mapping[str, Any]] | None]:
    expected_keys = {str(counter) for counter in range(1, issuance_counter + 1)}
    if "proof_sha256_by_counter" in state:
        registered = state.get("proof_sha256_by_counter")
        if (
            not isinstance(registered, Mapping)
            or set(registered) != expected_keys
            or any(
                not isinstance(value, str)
                or NONCE_PATTERN.fullmatch(value) is None
                for value in registered.values()
            )
        ):
            raise ValueError("registered raw proof counter registry is incomplete")
        return registered, None

    legacy = state.get("finalized_proof_registry")
    legacy_sha256 = state.get("finalized_proof_registry_sha256")
    if not isinstance(legacy, list) or not isinstance(legacy_sha256, str):
        raise ValueError("completed legacy replay has no trusted proof registry anchor")
    if canonical_sha256(legacy) != legacy_sha256:
        raise ValueError("finalized legacy proof registry anchor mismatch")
    counters = [entry.get("counter") for entry in legacy if isinstance(entry, Mapping)]
    if (
        len(counters) != len(legacy)
        or counters != list(range(1, issuance_counter + 1))
        or any(
            set(entry) != {"counter", "dataset_id", "proof_sha256"}
            or not isinstance(entry.get("dataset_id"), str)
            or not isinstance(entry.get("proof_sha256"), str)
            or NONCE_PATTERN.fullmatch(str(entry["proof_sha256"])) is None
            for entry in legacy
            if isinstance(entry, Mapping)
        )
    ):
        raise ValueError("finalized legacy proof registry counter registry is invalid")
    return {
        str(entry["counter"]): str(entry["proof_sha256"])
        for entry in legacy
    }, legacy


def _replay_registered_proofs(
    state_dir: Path,
    *,
    expected_mode: str,
    cohort_expectations: Mapping[str, object] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Rebuild proofs from controller registrations and outer artifacts."""
    from scripts.run_unified_validation import RECIPES

    state_dir = state_dir.resolve()
    state, baseline_by_dataset = _verified_replay_state(
        state_dir,
        expected_mode=expected_mode,
        cohort_expectations=cohort_expectations,
    )
    grouped = _registered_replay_pairs(state_dir, state)
    issuance_counter = state.get("issuance_counter")
    assert isinstance(issuance_counter, int)
    registered_hashes, legacy_anchor = _registered_replay_proof_anchor(
        state, issuance_counter=issuance_counter
    )
    if expected_mode == "candidate_gate":
        dataset_ids = state.get("dataset_ids")
        if (
            not isinstance(dataset_ids, list)
            or issuance_counter != len(dataset_ids)
        ):
            raise ValueError("candidate replay issuance count differs from frozen cohort")
        if legacy_anchor is not None:
            raise ValueError("candidate replay requires controller-registered proof SHA anchor")
    replayed: list[dict[str, Any]] = []
    raw_registry: dict[str, str] = {}
    for counter in range(1, issuance_counter + 1):
        pair = grouped[counter]
        if set(pair) != {"standard", "holdout"}:
            raise ValueError("replay issuance split pair is incomplete")
        dataset_id = str(pair["standard"]["dataset_id"])
        if pair["holdout"]["dataset_id"] != dataset_id:
            raise ValueError("replay issuance dataset pair mismatch")
        if (
            expected_mode == "candidate_gate"
            and dataset_id != state["dataset_ids"][counter - 1]
        ):
            raise ValueError("candidate replay dataset order differs from frozen cohort")
        commands = {
            split_id: _validated_replay_invocation(
                state_dir=state_dir, state=state, consumption=pair[split_id]
            )
            for split_id in ("standard", "holdout")
        }
        replay_state = dict(state)
        replay_state["launch"] = {"commands": {key: ["outer", "--", *value] for key, value in commands.items()}}
        split_proofs = {
            split_id: _verify_split_proof(
                state_dir=state_dir,
                state=replay_state,
                consumption=pair[split_id],
            )
            for split_id in ("standard", "holdout")
        }
        if (
            split_proofs["standard"]["recipe_index"] != split_proofs["holdout"]["recipe_index"]
            or split_proofs["standard"]["numerical_recipe"] != split_proofs["holdout"]["numerical_recipe"]
            or split_proofs["standard"]["parameter_count"] != split_proofs["holdout"]["parameter_count"]
        ):
            raise ValueError("replayed split recipe or parameter binding mismatch")
        recipe_index = split_proofs["standard"]["recipe_index"]
        if (
            type(recipe_index) is not int
            or recipe_index < 0
            or recipe_index >= len(RECIPES[dataset_id])
            or split_proofs["standard"]["numerical_recipe"]
            != RECIPES[dataset_id][recipe_index].__dict__
        ):
            raise ValueError("replayed numerical recipe does not match registry")
        guard = baseline_by_dataset.get(dataset_id)
        if not isinstance(guard, Mapping):
            raise ValueError("replay external guard is missing")
        standard_metrics = split_proofs["standard"]["metrics"]
        holdout_metrics = split_proofs["holdout"]["metrics"]
        deltas = {
            "standard_overall_auc": standard_metrics["overall_auc"] - guard["standard_overall_auc"],
            "holdout_overall_auc": holdout_metrics["overall_auc"] - guard["holdout_overall_auc"],
            "zero_auc": holdout_metrics["zero_auc"] - guard["zero_auc"],
        }
        if expected_mode == "candidate_gate":
            deltas.update({
                "weighted_doa": standard_metrics["weighted_doa"]
                - guard["weighted_doa"],
                "ordinary_doa": standard_metrics["ordinary_doa"]
                - guard["ordinary_doa"],
            })
        proof_path = state_dir / "proofs" / f"{counter:06d}-{dataset_id}.json"
        raw_proof = _load_json(proof_path, label="replay raw proof")
        if not isinstance(raw_proof, Mapping):
            raise ValueError("replay raw proof is not an object")
        expected_fields = {
            "schema_version", "split_seed", "controller_id", "route_commit", "counter",
            "dataset_id", "baseline_sha256", "split_proofs", "deltas", "joint_success",
            "zero_delta_at_least_0.001", "launch",
        }
        if (
            set(raw_proof) != expected_fields
            or any(raw_proof.get(field) != state.get(field) for field in ("controller_id", "route_commit", "baseline_sha256", "split_seed"))
            or raw_proof.get("counter") != counter
            or raw_proof.get("dataset_id") != dataset_id
            or raw_proof.get("split_proofs") != split_proofs
            or raw_proof.get("deltas") != deltas
            or raw_proof.get("joint_success") != _joint_gate_success(deltas)
            or raw_proof.get("zero_delta_at_least_0.001") != (deltas["zero_auc"] >= 0.001)
        ):
            raise ValueError("raw proof does not match controller-owned replay")
        raw_sha = hashlib.sha256(proof_path.read_bytes()).hexdigest()
        if registered_hashes.get(str(counter)) != raw_sha:
            raise ValueError("registered raw proof SHA-256 mismatch")
        raw_registry[str(counter)] = raw_sha
        replayed.append({
            "counter": counter,
            "dataset_id": dataset_id,
            "split_proofs": split_proofs,
            "deltas": deltas,
            "proof_sha256": raw_sha,
        })
    if legacy_anchor is not None:
        replayed_registry = [
            {
                "counter": entry["counter"],
                "dataset_id": entry["dataset_id"],
                "proof_sha256": entry["proof_sha256"],
            }
            for entry in replayed
        ]
        if replayed_registry != legacy_anchor:
            raise ValueError("finalized legacy proof registry does not match replay")
        state["proof_sha256_by_counter"] = raw_registry
        _atomic_json(state_dir / "state.json", state)
    return state, replayed


def replay_registered_exploration_proofs(
    state_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return _replay_registered_proofs(
        state_dir, expected_mode="a0_exploration"
    )


def replay_registered_candidate_proofs(
    state_dir: Path,
    *,
    cohort_expectations: Mapping[str, object] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return _replay_registered_proofs(
        state_dir,
        expected_mode="candidate_gate",
        cohort_expectations=cohort_expectations,
    )


def _joint_gate_success(deltas: Mapping[str, float]) -> bool:
    return (
        deltas["standard_overall_auc"] >= 0.0
        and deltas["holdout_overall_auc"] >= 0.0
        and deltas["zero_auc"] > 0.0
    )


def _next_a0_recipe_index(
    *,
    dataset_id: str,
    recipe_index: int,
    overall_guard_passed: bool,
    eligible_candidates: int,
) -> int | None:
    if recipe_index + 1 >= RECIPE_COUNTS[dataset_id]:
        return None
    if dataset_id == "XES3G5M" and eligible_candidates >= 3:
        return None
    if overall_guard_passed and eligible_candidates >= 3:
        return None
    return recipe_index + 1


def _advance_active_pair(
    *,
    state_dir: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    active_pair = state.get("active_pair")
    if not isinstance(active_pair, Mapping):
        return state
    consumed = _load_consumed_pair(state_dir, active_pair)
    launch = state.get("launch")
    attempts = launch.get("attempts") if isinstance(launch, Mapping) else None
    if (
        not isinstance(launch, Mapping)
        or launch.get("phase") != "running"
        or launch.get("counter") != active_pair.get("counter")
        or launch.get("dataset_id") != active_pair.get("dataset_id")
        or not isinstance(attempts, Mapping)
        or any(
            str(Path(str(attempts.get(split_id))).resolve())
            != str(Path(str(consumed[split_id].get("attempt_dir"))).resolve())
            for split_id in ("standard", "holdout")
        )
    ):
        raise ValueError("controller launch attempts do not bind consumed pair")
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
        "zero_auc": holdout["zero_auc"] - baseline["zero_auc"],
    }
    exploration = state.get("mode") == "a0_exploration"
    if not exploration:
        deltas.update({
            "weighted_doa": standard["weighted_doa"] - baseline["weighted_doa"],
            "ordinary_doa": standard["ordinary_doa"] - baseline["ordinary_doa"],
        })
    joint_success = _joint_gate_success(deltas)
    overall_success = (
        deltas["standard_overall_auc"] >= 0.0
        and deltas["holdout_overall_auc"] >= 0.0
    )
    dataset_id = str(active_pair.get("dataset_id"))
    if (
        split_proofs["standard"]["recipe_index"]
        != split_proofs["holdout"]["recipe_index"]
        or split_proofs["standard"]["numerical_recipe"]
        != split_proofs["holdout"]["numerical_recipe"]
    ):
        raise ValueError("standard/holdout numerical recipes differ")
    if (
        split_proofs["standard"]["parameter_count"]
        != split_proofs["holdout"]["parameter_count"]
    ):
        raise ValueError("standard/holdout parameter counts differ")
    proof: dict[str, Any] = {
        "schema_version": CONTROLLER_SCHEMA_VERSION,
        "split_seed": state["split_seed"],
        "controller_id": state["controller_id"],
        "route_commit": state["route_commit"],
        "counter": active_pair["counter"],
        "dataset_id": dataset_id,
        "baseline_sha256": state["baseline_sha256"],
        "split_proofs": split_proofs,
        "deltas": deltas,
        "joint_success": joint_success,
        "zero_delta_at_least_0.001": deltas["zero_auc"] >= 0.001,
        "launch": dict(launch),
    }
    proof_path = state_dir / "proofs" / (
        f"{int(active_pair['counter']):06d}-{dataset_id}.json"
    )
    if proof_path.exists():
        if _load_json(proof_path, label="registered progress proof") != proof:
            raise ValueError("existing progress proof does not match outer artifacts")
    else:
        _exclusive_json(proof_path, proof)
    proof_hashes = state.get("proof_sha256_by_counter")
    if not isinstance(proof_hashes, dict):
        raise ValueError("controller raw proof SHA registry is invalid")
    proof_hashes[str(active_pair["counter"])] = hashlib.sha256(
        proof_path.read_bytes()
    ).hexdigest()
    state["cursor"] = cursor + 1
    recipe_indices = state.get("recipe_indices")
    if not isinstance(recipe_indices, dict):
        raise ValueError("controller recipe registry is invalid")
    recipe_index = recipe_indices.get(dataset_id)
    if type(recipe_index) is not int:
        raise ValueError("controller recipe index is invalid")
    overall_passed = state.get("exploration_overall_passed")
    if exploration:
        if not isinstance(overall_passed, dict):
            raise ValueError("controller exploration pass registry is invalid")
        if overall_success:
            overall_passed[dataset_id] = True
        eligible_candidates = len(overall_passed)
    else:
        eligible_candidates = int(state.get("successes", 0)) + int(joint_success)
    next_recipe_index = _next_a0_recipe_index(
        dataset_id=dataset_id,
        recipe_index=recipe_index,
        overall_guard_passed=overall_success,
        eligible_candidates=eligible_candidates,
    )
    if state.get("architecture") == "a0" and next_recipe_index is not None:
        state["cursor"] = cursor
        recipe_indices[dataset_id] = next_recipe_index
        state["active_pair"] = None
        _atomic_json(state_dir / "state.json", state)
        return state
    if exploration:
        state["successes"] = len(overall_passed)
        selected_recipes = state.get("selected_recipes")
        if not isinstance(selected_recipes, dict):
            raise ValueError("controller selected recipe registry is invalid")
        selected_recipes[dataset_id] = {
            "recipe_index": recipe_index,
            "numerical_recipe": split_proofs["standard"]["numerical_recipe"],
        }
        state["active_pair"] = None
        dataset_ids = state.get("dataset_ids")
        assert isinstance(dataset_ids, list)
        if int(state["cursor"]) == len(dataset_ids):
            state["complete"] = True
            state["global_pass"] = None
        _atomic_json(state_dir / "state.json", state)
        return state
    if joint_success:
        state["successes"] = int(state.get("successes", 0)) + 1
    if overall_success:
        selected_recipes = state.get("selected_recipes")
        if not isinstance(selected_recipes, dict):
            raise ValueError("controller selected recipe registry is invalid")
        selected_recipes[dataset_id] = {
            "recipe_index": recipe_index,
            "numerical_recipe": split_proofs["standard"]["numerical_recipe"],
        }
    if not overall_success:
        state["overall_failures"] = int(state.get("overall_failures", 0)) + 1
    if deltas["zero_auc"] >= 0.001:
        state["zero_delta_threshold_seen"] = True
    state["active_pair"] = None
    dataset_ids = state.get("dataset_ids")
    assert isinstance(dataset_ids, list)
    required_improvements = math.ceil(2 * len(dataset_ids) / 3)
    if int(state["cursor"]) == len(dataset_ids):
        state["complete"] = True
        state["global_pass"] = (
            int(state.get("overall_failures", 0)) == 0
            and int(state["successes"]) >= required_improvements
            and bool(state["zero_delta_threshold_seen"])
        )
    _atomic_json(state_dir / "state.json", state)
    return state


def _ensure_issued_capabilities(
    state_dir: Path,
    token: Mapping[str, Any],
) -> dict[str, str]:
    capabilities = token.get("capabilities")
    if not isinstance(capabilities, Mapping):
        raise ValueError("pending issuance has no capability payloads")
    filenames: dict[str, str] = {}
    for split_id in ("standard", "holdout"):
        capability = capabilities.get(split_id)
        if not isinstance(capability, Mapping):
            raise ValueError(f"pending issuance has no {split_id} capability")
        path = _capability_path(state_dir, capability)
        if path.exists():
            if _load_json(path, label="pending issued capability") != capability:
                raise ValueError("pending issued capability registry mismatch")
        else:
            _exclusive_json(path, capability)
        filenames[split_id] = path.name
    return filenames


def _validate_pending_token(
    state: Mapping[str, Any], token: Mapping[str, Any]
) -> None:
    dataset_ids = state.get("dataset_ids")
    cursor = state.get("cursor")
    if (
        not isinstance(dataset_ids, list)
        or type(cursor) is not int
        or not 0 <= cursor < len(dataset_ids)
    ):
        raise ValueError("controller progress cannot validate pending issuance")
    common = {
        "schema_version": CONTROLLER_SCHEMA_VERSION,
        "split_seed": state.get("split_seed"),
        "controller_id": state.get("controller_id"),
        "route_commit": state.get("route_commit"),
        "counter": int(state.get("issuance_counter", 0)) + 1,
        "dataset_id": dataset_ids[cursor],
    }
    if set(token) != {*common, "capabilities"} or any(
        token.get(field) != value for field, value in common.items()
    ):
        raise ValueError("pending issuance token envelope is invalid")
    capabilities = token.get("capabilities")
    if not isinstance(capabilities, Mapping) or set(capabilities) != {
        "standard",
        "holdout",
    }:
        raise ValueError("pending issuance split set is invalid")
    nonces: set[str] = set()
    for split_id in ("standard", "holdout"):
        capability = capabilities.get(split_id)
        expected_fields = {*common, "split_id", "nonce"}
        if (
            not isinstance(capability, Mapping)
            or set(capability) != expected_fields
            or any(capability.get(field) != value for field, value in common.items())
            or capability.get("split_id") != split_id
            or not isinstance(capability.get("nonce"), str)
            or NONCE_PATTERN.fullmatch(str(capability["nonce"])) is None
        ):
            raise ValueError(f"pending {split_id} capability is invalid")
        nonces.add(str(capability["nonce"]))
    if len(nonces) != 2:
        raise ValueError("pending issuance capability nonces must be distinct")


def _finish_pending_issuance(
    *,
    state_dir: Path,
    state: dict[str, Any],
    output_path: Path,
) -> dict[str, object]:
    pending = state.get("pending_issuance")
    if not isinstance(pending, Mapping):
        raise ValueError("controller has no pending issuance")
    registered_output = Path(str(pending.get("output_path"))).resolve()
    token = pending.get("token")
    if registered_output != output_path or not isinstance(token, Mapping):
        raise ValueError("pending issuance output does not match registry")
    _validate_pending_token(state, token)
    filenames = _ensure_issued_capabilities(state_dir, token)
    state["active_pair"] = {
        "counter": token["counter"],
        "dataset_id": token["dataset_id"],
        "capability_files": filenames,
        "token": token,
        "output_path": str(output_path),
        "token_published": False,
    }
    state["pending_issuance"] = None
    state["issuance_counter"] = token["counter"]
    _atomic_json(state_dir / "state.json", state)
    _atomic_json(output_path, token)
    state["active_pair"]["token_published"] = True
    _atomic_json(state_dir / "state.json", state)
    return dict(token)


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
        launch = state.get("launch")
        if isinstance(launch, Mapping):
            state["blocked"] = True
            state["launch"] = {**launch, "phase": "interrupted"}
            _atomic_json(state_dir / "state.json", state)
            raise RuntimeError("prior controller launch was interrupted")
        if isinstance(state.get("pending_issuance"), Mapping):
            return _finish_pending_issuance(
                state_dir=state_dir,
                state=state,
                output_path=output_path,
            )
        active_pair = state.get("active_pair")
        if isinstance(active_pair, Mapping) and not active_pair.get(
            "token_published", True
        ):
            registered_output = Path(str(active_pair.get("output_path"))).resolve()
            token = active_pair.get("token")
            if registered_output != output_path or not isinstance(token, Mapping):
                raise ValueError("pending token recovery output does not match registry")
            _atomic_json(output_path, token)
            state["active_pair"]["token_published"] = True
            _atomic_json(state_dir / "state.json", state)
            return dict(token)
        if isinstance(active_pair, Mapping):
            raise RuntimeError("active pair requires controller run-pair")
        if output_path.exists() and state.get("active_pair") is None:
            if output_path.stat().st_size == 0:
                _unlink_fsync(output_path)
            else:
                raise FileExistsError(output_path)
        _exclusive_bytes(output_path, b"")
        original_state = json.loads(json.dumps(state))
        original_issued = {path.name for path in (state_dir / "issued").glob("*.json")}
        original_proofs = {path.name for path in (state_dir / "proofs").glob("*.json")}
        preserve_progress = False
        committed = False
        try:
            if state.get("blocked"):
                preserve_progress = True
                raise RuntimeError(
                    "primary cohort is unreachable under registered stop rule"
                )
            if state.get("complete"):
                completion: dict[str, object] = {
                    "schema_version": CONTROLLER_SCHEMA_VERSION,
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
                "schema_version": CONTROLLER_SCHEMA_VERSION,
                "split_seed": state["split_seed"],
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
                capabilities[split_id] = capability
            token: dict[str, object] = {
                **common,
                "capabilities": capabilities,
            }
            state["pending_issuance"] = {
                "output_path": str(output_path),
                "token": token,
            }
            _atomic_json(state_dir / "state.json", state)
            token = _finish_pending_issuance(
                state_dir=state_dir,
                state=state,
                output_path=output_path,
            )
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


def _outer_command(
    *,
    state_dir: Path,
    repo_root: Path,
    state: Mapping[str, Any],
    dataset_id: str,
    split_id: str,
) -> list[str]:
    runner = OUTER_RUNNER_OVERRIDE or repo_root / "scripts" / "run_remote_campaign.py"
    artifact_root = (
        Path(str(state["artifact_root"]))
        / str(state["architecture"])
        / dataset_id
        / split_id
    )
    records = state["validation_data"][f"{dataset_id}:{split_id}"]
    command = [
        sys.executable,
        str(runner),
        "--artifact-root",
        str(artifact_root),
        "--repo-root",
        str(repo_root),
        "--cwd",
        str(repo_root),
    ]
    for record in records:
        command.extend(["--dataset-file", str(record["path"])])
    command.extend(
        [
            "--output-file",
            "validation-summary.json",
            "--summary-output",
            "validation-summary.json",
            "--architecture-manifest",
            str(state_dir / "manifest.json"),
            "--cohort",
            str(state_dir / "cohort.json"),
            "--seed",
            "42",
            "--split-seed",
            "2024",
            "--doa-seed",
            "42",
            "--min-responses",
            "3",
            "--",
            sys.executable,
            str(repo_root / "scripts" / "run_unified_validation.py"),
            "run-split",
            "--dataset-id",
            dataset_id,
            "--split-id",
            split_id,
            "--architecture",
            str(state["architecture"]),
            "--architecture-fingerprint",
            str(state["architecture_fingerprint"]),
            "--cohort-sha256",
            str(state["cohort_sha256"]),
            "--recipe-index",
            str(state["recipe_indices"][dataset_id]),
            "--data-root",
            str(state["data_root"]),
            "--controller-state-dir",
            str(state_dir),
            "--repo-root",
            str(repo_root),
            "--capability",
            str(state["active_pair"]["output_path"]),
            "--output",
            "validation-summary.json",
        ]
    )
    return command


def _attempt_names(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and re.fullmatch(r"attempt-[0-9]+", path.name)
    )


def _block_launch(
    state_dir: Path,
    repo_root: Path,
    *,
    launch_id: str,
    error: str,
) -> None:
    with _controller_lock(state_dir):
        state = _load_state(state_dir, repo_root)
        launch = state.get("launch")
        if isinstance(launch, Mapping) and launch.get("launch_id") == launch_id:
            state["blocked"] = True
            state["launch"] = {**launch, "phase": "failed", "error": error}
            _atomic_json(state_dir / "state.json", state)


def _terminate_started_processes(
    processes: Mapping[str, subprocess.Popen[str]],
) -> None:
    for process in processes.values():
        if process.poll() is None:
            process.terminate()
    for process in processes.values():
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def run_registered_pair(
    *,
    state_dir: Path,
    repo_root: Path,
    parallel: bool = True,
) -> dict[str, Any]:
    state_dir = state_dir.resolve()
    repo_root = repo_root.resolve()
    with _controller_lock(state_dir):
        state = _load_state(state_dir, repo_root)
        if state.get("blocked") or state.get("complete"):
            raise RuntimeError("controller cannot launch a blocked/complete iteration")
        if state.get("launch") is not None:
            state["blocked"] = True
            state["launch"] = {
                **state["launch"],
                "phase": "interrupted",
            }
            _atomic_json(state_dir / "state.json", state)
            raise RuntimeError("prior controller launch was interrupted")
        active = state.get("active_pair")
        if not isinstance(active, Mapping) or not active.get("token_published"):
            raise ValueError("run-pair requires a published active capability pair")
        dataset_id = str(active["dataset_id"])
        counter = int(active["counter"])
        before_attempts = {
            split_id: _attempt_names(
                Path(str(state["artifact_root"]))
                / str(state["architecture"])
                / dataset_id
                / split_id
            )
            for split_id in ("standard", "holdout")
        }
        commands = {
            split_id: _outer_command(
                state_dir=state_dir,
                repo_root=repo_root,
                state=state,
                dataset_id=dataset_id,
                split_id=split_id,
            )
            for split_id in ("standard", "holdout")
        }
        launch_id = secrets.token_hex(32)
        state["launch"] = {
            "phase": "running",
            "launch_id": launch_id,
            "owner_pid": os.getpid(),
            "counter": counter,
            "dataset_id": dataset_id,
            "before_attempts": before_attempts,
            "commands": commands,
        }
        _atomic_json(state_dir / "state.json", state)

    new_attempts: dict[str, str] = {}
    try:
        processes: dict[str, subprocess.Popen[str]] = {}
        results: dict[str, tuple[str, str]] = {}
        failed_split: str | None = None
        try:
            if parallel:
                for split_id in ("standard", "holdout"):
                    processes[split_id] = subprocess.Popen(
                        commands[split_id],
                        cwd=repo_root,
                        env=_sanitized_subprocess_env(),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                with ThreadPoolExecutor(max_workers=len(processes)) as executor:
                    futures = {
                        executor.submit(process.communicate): split_id
                        for split_id, process in processes.items()
                    }
                    for future in as_completed(futures):
                        split_id = futures[future]
                        results[split_id] = future.result()
                        if (
                            processes[split_id].returncode != 0
                            and failed_split is None
                        ):
                            failed_split = split_id
                            _terminate_started_processes(processes)
            else:
                for split_id in ("standard", "holdout"):
                    process = subprocess.Popen(
                        commands[split_id],
                        cwd=repo_root,
                        env=_sanitized_subprocess_env(),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    processes[split_id] = process
                    results[split_id] = process.communicate()
                    if process.returncode != 0:
                        failed_split = split_id
                        break
        except BaseException:
            _terminate_started_processes(processes)
            raise
        if failed_split is not None:
            split_id = failed_split
            _, stderr = results[split_id]
            raise RuntimeError(
                f"{split_id} outer campaign failed with "
                f"{processes[split_id].returncode}: {stderr.strip()}"
            )
        for split_id in ("standard", "holdout"):
            completed = processes[split_id]
            stdout, stderr = results[split_id]
            split_root = (
                Path(str(state["artifact_root"]))
                / str(state["architecture"])
                / dataset_id
                / split_id
            )
            after = set(_attempt_names(split_root))
            created = sorted(after - set(before_attempts[split_id]))
            if len(created) != 1:
                raise RuntimeError(
                    f"{split_id} outer campaign created {len(created)} attempts"
                )
            attempt_path = (split_root / created[0]).resolve()
            if stdout.strip() and Path(stdout.strip().splitlines()[-1]).resolve() != attempt_path:
                raise RuntimeError(f"{split_id} outer campaign reported wrong attempt")
            new_attempts[split_id] = str(attempt_path)
    except BaseException as error:
        _block_launch(
            state_dir,
            repo_root,
            launch_id=launch_id,
            error=f"{type(error).__name__}: {error}",
        )
        raise

    try:
        with _controller_lock(state_dir):
            state = _load_state(state_dir, repo_root)
            launch = state.get("launch")
            if not isinstance(launch, Mapping) or launch.get("launch_id") != launch_id:
                raise RuntimeError("controller launch journal changed before proof freeze")
            state["launch"] = {**launch, "attempts": new_attempts}
            state = _advance_active_pair(state_dir=state_dir, state=state)
            proof_path = state_dir / "proofs" / f"{counter:06d}-{dataset_id}.json"
            proof = _load_json(proof_path, label="controller launch proof")
            state["launch"] = None
            _atomic_json(state_dir / "state.json", state)
            return proof
    except BaseException as error:
        _block_launch(
            state_dir,
            repo_root,
            launch_id=launch_id,
            error=f"{type(error).__name__}: {error}",
        )
        raise


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
        active_pair = state.get("active_pair")
        if not isinstance(active_pair, Mapping):
            raise ValueError("controller has no active issued pair")
        authoritative_token = active_pair.get("token")
        if not isinstance(authoritative_token, Mapping) or token != authoritative_token:
            raise ValueError("capability token does not match active controller token")
        capabilities = token.get("capabilities")
        if not isinstance(capabilities, Mapping) or set(capabilities) != {
            "standard",
            "holdout",
        }:
            raise ValueError("capability token has no split registry references")
        capability = capabilities.get(split_id)
        if not isinstance(capability, Mapping):
            raise ValueError(f"capability token has no {split_id} capability")
        capability = dict(capability)
        expected_top = {
            "schema_version": CONTROLLER_SCHEMA_VERSION,
            "split_seed": state.get("split_seed"),
            "controller_id": state.get("controller_id"),
            "route_commit": state.get("route_commit"),
            "counter": active_pair.get("counter"),
            "dataset_id": active_pair.get("dataset_id"),
        }
        if (
            set(token) != {*expected_top, "capabilities"}
            or any(token.get(field) != value for field, value in expected_top.items())
            or set(capability) != {*expected_top, "split_id", "nonce"}
            or any(
                capability.get(field) != value
                for field, value in expected_top.items()
            )
            or capability.get("split_id") != split_id
            or not isinstance(capability.get("nonce"), str)
            or NONCE_PATTERN.fullmatch(str(capability["nonce"])) is None
        ):
            raise ValueError("capability token envelope does not match capability")
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
        launch = state.get("launch")
        if (
            not isinstance(launch, Mapping)
            or launch.get("phase") != "running"
            or launch.get("counter") != capability.get("counter")
            or launch.get("dataset_id") != dataset_id
        ):
            raise ValueError("capability consumption requires active controller launch")
        if data_root.resolve() != Path(str(state.get("data_root"))).resolve():
            raise ValueError("controller launch data root mismatch")
        attempt_dir = attempt_dir.resolve()
        expected_attempt_root = (
            Path(str(state.get("artifact_root"))).resolve()
            / architecture
            / dataset_id
            / split_id
        )
        before_attempts = launch.get("before_attempts")
        if (
            attempt_dir.parent != expected_attempt_root
            or not re.fullmatch(r"attempt-[0-9]+", attempt_dir.name)
            or not isinstance(before_attempts, Mapping)
            or attempt_dir.name in before_attempts.get(split_id, [])
        ):
            raise ValueError("attempt is not the unique new controller launch attempt")
        capability_files = active_pair.get("capability_files")
        issued_name = (
            capability_files.get(split_id)
            if isinstance(capability_files, Mapping)
            else None
        )
        if (
            not isinstance(issued_name, str)
            or Path(issued_name).name != issued_name
            or "/" in issued_name
            or "\\" in issued_name
        ):
            raise ValueError("active capability registry filename is invalid")
        issued_path = state_dir / "issued" / issued_name
        consumed_path = state_dir / "consumed" / issued_path.name
        if consumed_path.exists():
            raise ValueError("capability is already present in the consumed registry")
        if not issued_path.is_file():
            raise ValueError("capability is not present in the issued registry")
        registered = _load_json(issued_path, label="issued capability registry")
        if registered != capability:
            raise ValueError("capability does not exactly match the issued registry")

        output_path = (
            raw_output.resolve()
            if raw_output.is_absolute()
            else attempt_dir / raw_output
        )
        consumption: dict[str, object] = {
            **capability,
            "architecture": state["architecture"],
            "architecture_fingerprint": state["architecture_fingerprint"],
            "architecture_manifest": _load_json(
                state_dir / "manifest.json",
                label="registered architecture manifest",
            ),
            "cohort_sha256": state["cohort_sha256"],
            "split_seed": state["split_seed"],
            "recipe_index": state["recipe_indices"][dataset_id],
            "numerical_recipe": state["registered_recipes"].get(dataset_id),
            "attempt_dir": str(attempt_dir),
            "summary_path": str(output_path.resolve()),
            "data_paths": _expected_data_paths(
                dataset_id=dataset_id,
                split_id=split_id,
                data_root=data_root,
            ),
        }
        _atomic_json(consumed_path, consumption)
        _unlink_fsync(issued_path)
        return consumption


if __name__ == "__main__":
    from scripts.unified_a0_exploration import main as exploration_main

    raise SystemExit(exploration_main())
