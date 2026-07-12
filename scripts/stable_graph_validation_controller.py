from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.unified_v2_spec import UnifiedArchitectureSpec
from scripts.unified_campaign import stable_graph_relative_gate
from scripts.unified_dataset_audit import canonical_sha256
from scripts.unified_validation_controller import _route_head


CAMPAIGN_ID = "unified-ergc-20260712"
FROZEN_COHORT_SHA256 = (
    "6342dc8a5f73a4e03a1645780597b625c"
    "1480ba7a6513668b6766089cdd5b8a5"
)
FROZEN_COMPARATOR_AUDIT_SHA256 = (
    "071b5df25d8641fdc249a9a561759610"
    "25b3deeba36a4a471640563ef5d0b181"
)
FROZEN_RECIPES = {"MOOCRadar": 2, "ASSIST17": 1, "XES3G5M": 0}
ARCHITECTURES = {
    "a0v4": ("prior", 0.0),
    "a2": ("evidence-relational-graph", 1.0),
}
VALIDATION_METRICS = (
    "standard_overall_auc",
    "holdout_overall_auc",
    "zero_auc",
    "ordinary_doa",
    "weighted_doa",
)
TEST_METRICS = ("overall_auc", "zero_auc", "ordinary_doa", "weighted_doa")
VALIDATION_RESULT_FIELDS = {
    "schema_version", "architecture", "architecture_manifest",
    "architecture_fingerprint", "cohort_sha256", "dataset_id", "recipe_index", "seed",
    "split_seed", "gpu_uuid", "peak_gpu_memory_gb", *VALIDATION_METRICS,
}
SMOKE_RESULT_FIELDS = {
    "architecture_fingerprint", "gpu_uuid", "peak_gpu_memory_gb",
    "mastery_shape", "final_loss", "parameter_count",
}
TEST_RESULT_FIELDS = {
    "schema_version", "architecture", "architecture_manifest",
    "architecture_fingerprint", "seed", "split_seed", "cohort_sha256",
    "recipes", "gpu_uuid", "peak_gpu_memory_gb", "rows",
}
TEST_ROW_FIELDS = {"recipe_index", *TEST_METRICS}
DEFAULT_CAMPAIGN_ROOT = Path(
    "/home/xph/jwc/research/local_data/decoupled_cd_codex_routes"
) / CAMPAIGN_ID
DEFAULT_COMPARATOR_AUDIT = Path(
    "/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/"
    "unified-mastery-20260712/audit/comparator-audit.json"
)
RUNNER = PROJECT_ROOT / "scripts" / "run_unified_validation.py"
Runner = Callable[[Sequence[str], Path], Mapping[str, object]]
IMPLEMENTATION_DIRECTORIES = (
    "models", "scripts", "trainers", "data", "configs", "tests"
)
_PRODUCTION_AUTHORITY = object()
_TEST_AUTHORITY = object()
_TEST_SENTINEL_BYTES = secrets.token_bytes(32)
_TEST_SENTINEL_NAME = ".stable-graph-test-root"
LEDGER_FIELDS = {"schema_version", "campaign_id", "implementation_code_sha256", "entries"}
ENTRY_PENDING_FIELDS = {
    "counter", "nonce", "kind", "split_ids", "architecture", "dataset_id", "attempt_dir",
    "route_commit", "implementation_code_sha256", "status",
}
ENTRY_COMPLETE_FIELDS = ENTRY_PENDING_FIELDS | {"proof_file_sha256"}
ATTEMPT_PROOF_FIELDS = {
    "schema_version", "campaign_id", "cohort_sha256", "kind", "split_ids", "architecture",
    "architecture_manifest", "architecture_fingerprint", "dataset_id",
    "recipe_index", "seed", "split_seed", "route_commit",
    "implementation_code_sha256", "counter", "nonce", "runner_argv",
    "runner_result", "artifacts", "proof_sha256",
}
ARTIFACT_FIELDS = {"path", "size_bytes", "sha256", "device", "inode"}
VALIDATION_SOURCE_SUFFIXES = tuple(
    f"stable-validation-work/{split}/{name}"
    for split in ("standard", "holdout")
    for name in ("train-summary.json", "coverage-valid.json", "doa-valid.json")
)
TEST_SOURCE_SUFFIXES = tuple(
    f"stable-test-work/{dataset}/{name}"
    for dataset in sorted(FROZEN_RECIPES)
    for name in ("train-summary.json", "coverage-test.json", "doa-test.json")
)
TEST_SOURCE_POLICIES = {
    "aggregate-only", "validation-fixed-sources", "test-fixed-sources",
    "all-fixed-sources", "production-layout-fixture"
}
PRODUCTION_SOURCE_POLICY = "production-fixed-sources"


def _aux_names(kind: str) -> tuple[str, ...]:
    suffix = "valid" if kind == "validation" else "test"
    return (
        "train-summary_best.pt",
        "train-summary_history.csv",
        f"coverage-{suffix}_slices.csv",
        f"coverage-{suffix}_summary.csv",
        f"doa-{suffix}_summary.csv",
    )


class CampaignStore:
    """Campaign filesystem rooted at one retained no-follow directory descriptor."""

    def __init__(self, root: Path, race_hook: Callable[[str], None] | None = None):
        self.root = root.absolute()
        self.race_hook = race_hook
        try:
            descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except OSError as error:
            raise ValueError("campaign root must be a no-follow directory, not a symlink") from error
        status = os.fstat(descriptor)
        if not stat.S_ISDIR(status.st_mode):
            os.close(descriptor)
            raise ValueError("campaign root is not a directory")
        self.root_fd = descriptor
        self.identity = (status.st_dev, status.st_ino)

    def __del__(self) -> None:
        descriptor = getattr(self, "root_fd", -1)
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
            self.root_fd = -1

    def assert_identity(self, stage: str = "before-campaign-open") -> None:
        if self.race_hook is not None:
            self.race_hook(stage)
        try:
            current = self.root.lstat()
        except FileNotFoundError as error:
            raise ValueError("campaign root identity changed by parent swap") from error
        if stat.S_ISLNK(current.st_mode) or not stat.S_ISDIR(current.st_mode):
            raise ValueError("campaign root identity changed by parent swap")
        if (current.st_dev, current.st_ino) != self.identity:
            raise ValueError("campaign root identity changed by parent swap")

    @staticmethod
    def _parts(relative: str | Path) -> tuple[str, ...]:
        path = Path(relative)
        if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("campaign relative path is invalid")
        return path.parts

    def _open_dir(self, parts: Sequence[str], *, create: bool = False) -> int:
        self.assert_identity()
        descriptor = os.dup(self.root_fd)
        try:
            for part in parts:
                if create:
                    try:
                        os.mkdir(part, 0o700, dir_fd=descriptor)
                    except FileExistsError:
                        pass
                child = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
                status = os.fstat(child)
                if not stat.S_ISDIR(status.st_mode):
                    os.close(child)
                    raise ValueError("campaign parent is not a no-follow directory")
                os.close(descriptor)
                descriptor = child
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def mkdir(self, relative: str | Path) -> None:
        descriptor = self._open_dir(self._parts(relative), create=True)
        os.close(descriptor)

    def mkdir_exclusive(self, relative: str | Path) -> None:
        parent, name = self._parent(relative, create=True)
        try:
            os.mkdir(name, 0o700, dir_fd=parent)
            child = os.open(
                name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent,
            )
            os.close(child)
            os.fsync(parent)
            self.assert_identity("after-exclusive-directory-create")
        finally:
            os.close(parent)

    def _parent(self, relative: str | Path, *, create: bool = False) -> tuple[int, str]:
        parts = self._parts(relative)
        return self._open_dir(parts[:-1], create=create), parts[-1]

    def exists(self, relative: str | Path) -> bool:
        try:
            parent, name = self._parent(relative)
        except FileNotFoundError:
            return False
        try:
            try:
                status = os.stat(name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                return False
            if stat.S_ISLNK(status.st_mode):
                raise ValueError("campaign file must not be a symlink")
            return True
        finally:
            os.close(parent)

    def exclusive_bytes(self, relative: str | Path, data: bytes) -> None:
        parent, name = self._parent(relative, create=True)
        try:
            descriptor = os.open(
                name, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                0o600, dir_fd=parent,
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.fsync(parent)
            self.assert_identity("after-campaign-write")
        finally:
            os.close(parent)

    def exclusive_json(self, relative: str | Path, payload: object) -> None:
        self.exclusive_bytes(relative, _json_bytes(payload))

    def atomic_json(self, relative: str | Path, payload: object) -> None:
        parent, name = self._parent(relative, create=True)
        temporary = f".{name}.{secrets.token_hex(16)}.tmp"
        try:
            descriptor = os.open(
                temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                0o600, dir_fd=parent,
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(_json_bytes(payload))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
            os.fsync(parent)
            self.assert_identity("after-campaign-replace")
        finally:
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
            os.close(parent)

    def read_regular(self, relative: str | Path, *, label: str) -> tuple[bytes, dict[str, object]]:
        parent, name = self._parent(relative)
        try:
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
            with os.fdopen(descriptor, "rb") as handle:
                before = os.fstat(handle.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise ValueError(f"{label} is not a regular no-follow file")
                digest = hashlib.sha256()
                chunks: list[bytes] = []
                while chunk := handle.read(1024 * 1024):
                    chunks.append(chunk)
                    digest.update(chunk)
                after = os.fstat(handle.fileno())
            identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            if identity_before != identity_after:
                raise ValueError(f"{label} identity changed while hashing")
            self.assert_identity("after-campaign-read")
            return b"".join(chunks), {
                "path": str(Path(relative)), "size_bytes": before.st_size,
                "sha256": digest.hexdigest(), "device": before.st_dev,
                "inode": before.st_ino,
            }
        finally:
            os.close(parent)

    def read_json(self, relative: str | Path, *, label: str) -> dict[str, Any]:
        data, _ = self.read_regular(relative, label=label)
        try:
            payload = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid {label} JSON") from error
        if type(payload) is not dict:
            raise ValueError(f"{label} must be a JSON object")
        return payload

    def listdir(self, relative: str | Path) -> list[str]:
        descriptor = self._open_dir(self._parts(relative))
        try:
            return sorted(os.listdir(descriptor))
        finally:
            os.close(descriptor)


def architecture_spec(architecture: str) -> UnifiedArchitectureSpec:
    try:
        completion, _ = ARCHITECTURES[architecture]
    except KeyError as error:
        raise ValueError(f"unknown stable graph architecture: {architecture}") from error
    return UnifiedArchitectureSpec(completion=completion)


def architecture_fingerprint(architecture: str) -> str:
    return architecture_spec(architecture).fingerprint()


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


def _seal(payload: Mapping[str, object]) -> dict[str, Any]:
    result = dict(payload)
    if "proof_sha256" in result:
        raise ValueError("payload already contains proof_sha256")
    result["proof_sha256"] = canonical_sha256(result)
    return result


def _verify_seal(payload: Mapping[str, object], label: str) -> None:
    digest = payload.get("proof_sha256")
    unsigned = dict(payload)
    unsigned.pop("proof_sha256", None)
    if type(digest) is not str or len(digest) != 64 or canonical_sha256(unsigned) != digest:
        raise ValueError(f"{label} proof SHA-256 mismatch")


def validate_metrics(value: Mapping[str, object]) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(VALIDATION_METRICS):
        raise ValueError("validation metrics field set mismatch")
    result: dict[str, float] = {}
    for field in VALIDATION_METRICS:
        metric = value[field]
        if type(metric) is not float or not math.isfinite(metric):
            raise ValueError(f"{field} must be an exact finite float")
        if not 0.0 <= metric <= 1.0:
            raise ValueError(f"{field} must be in [0, 1]")
        result[field] = metric
    return result


def fake_validation_result(architecture: str, dataset: str) -> dict[str, object]:
    """Deterministic test fixture; production never calls this helper."""
    return {
        "schema_version": 4,
        "architecture": architecture,
        "architecture_manifest": architecture_spec(architecture).manifest(),
        "architecture_fingerprint": architecture_fingerprint(architecture),
        "cohort_sha256": FROZEN_COHORT_SHA256,
        "dataset_id": dataset,
        "recipe_index": FROZEN_RECIPES[dataset],
        "seed": 42,
        "split_seed": 2024,
        "gpu_uuid": "GPU-fake",
        "peak_gpu_memory_gb": 0.25,
        **{field: 0.5 for field in VALIDATION_METRICS},
    }


def fake_test_result() -> dict[str, object]:
    return {
        "schema_version": 4,
        "architecture": "a2",
        "architecture_manifest": architecture_spec("a2").manifest(),
        "architecture_fingerprint": architecture_fingerprint("a2"),
        "seed": 42,
        "split_seed": 2024,
        "cohort_sha256": FROZEN_COHORT_SHA256,
        "recipes": dict(FROZEN_RECIPES),
        "gpu_uuid": "GPU-fake",
        "peak_gpu_memory_gb": 0.25,
        "rows": {
            dataset: {
                "recipe_index": recipe,
                **{metric: 0.5 for metric in TEST_METRICS},
            }
            for dataset, recipe in FROZEN_RECIPES.items()
        },
    }


def _subprocess_runner(argv: Sequence[str], attempt_dir: Path) -> Mapping[str, object]:
    completed = subprocess.run(
        list(argv),
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "CAMPAIGN_ATTEMPT_DIR": str(attempt_dir)},
    )
    if completed.returncode != 0:
        raise RuntimeError(f"trusted runner failed with exit {completed.returncode}: {completed.stderr}")
    output = Path(argv[argv.index("--output") + 1])
    with output.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    output.unlink()
    if type(payload) is not dict:
        raise ValueError("trusted runner result must be an object")
    records = payload.get("records")
    if isinstance(records, list) and len(records) == 1 and isinstance(records[0], dict):
        record = records[0]
        payload = {
            "architecture_fingerprint": record.get("architecture_fingerprint"),
            "gpu_uuid": record.get("physical_gpu_uuid"),
            "peak_gpu_memory_gb": record.get("peak_gpu_memory_gb"),
            "mastery_shape": record.get("mastery_shape"),
            "final_loss": record.get("final_loss"),
            "parameter_count": record.get("parameter_count"),
        }
    return payload


class CampaignDependencies:
    def __init__(
        self,
        campaign_root: Path,
        route_root: Path,
        runner: Runner,
        route_commit: str | None = None,
        test_only: bool = False,
        *,
        _authority: object | None = None,
        failure_hook: Callable[[str], None] | None = None,
        race_hook: Callable[[str], None] | None = None,
        source_policy: str | None = None,
    ) -> None:
        if _authority not in {_PRODUCTION_AUTHORITY, _TEST_AUTHORITY}:
            raise ValueError("CampaignDependencies require internal authority")
        root = campaign_root.absolute()
        if root.name != CAMPAIGN_ID:
            raise ValueError("exact campaign root name is required")
        resolved_root = root.resolve(strict=True)
        production_resolved = DEFAULT_CAMPAIGN_ROOT.resolve(strict=False)
        if _authority is _TEST_AUTHORITY and resolved_root == production_resolved:
            raise ValueError("test dependencies cannot use the production campaign root")
        if _authority is _PRODUCTION_AUTHORITY and resolved_root != production_resolved:
            raise ValueError("exact production campaign root is required")
        if _authority is _PRODUCTION_AUTHORITY and (
            runner is not _subprocess_runner or route_root.absolute() != PROJECT_ROOT
            or route_commit is not None or test_only
            or source_policy != PRODUCTION_SOURCE_POLICY
        ):
            raise ValueError("production dependencies cannot inject runner or route")
        self.campaign_root = root
        self.route_root = route_root.absolute()
        self.runner = runner
        self.route_commit = route_commit
        self.test_only = _authority is _TEST_AUTHORITY
        self.failure_hook = failure_hook
        if self.test_only and source_policy not in TEST_SOURCE_POLICIES:
            raise ValueError("test dependencies require an explicit exact fixture source policy")
        self.source_policy = source_policy
        self.store = CampaignStore(root)
        if self.test_only:
            sentinel = _TEST_SENTINEL_NAME
            if not self.store.exists(sentinel):
                self.store.exclusive_bytes(sentinel, _TEST_SENTINEL_BYTES)
            data, _ = self.store.read_regular(sentinel, label="test sentinel")
            if data != _TEST_SENTINEL_BYTES:
                raise ValueError("isolated test root sentinel mismatch")
        self.store.race_hook = race_hook

    @classmethod
    def for_test(
        cls,
        *,
        campaign_root: Path,
        route_root: Path,
        runner: Runner,
        route_commit: str | None = None,
        failure_hook: Callable[[str], None] | None = None,
        race_hook: Callable[[str], None] | None = None,
        source_policy: str,
    ) -> CampaignDependencies:
        if route_commit is not None and (type(route_commit) is not str or len(route_commit) != 40):
            raise ValueError("test route commit must be 40-hex")
        return cls(
            campaign_root, route_root, runner, route_commit, True,
            _authority=_TEST_AUTHORITY,
            failure_hook=failure_hook,
            race_hook=race_hook,
            source_policy=source_policy,
        )

    def current_route(self) -> str:
        return self.route_commit if self.route_commit is not None else _route_head(self.route_root)

    def implementation_hash(self, commit: str | None = None) -> str:
        if self.route_commit is not None:
            return hashlib.sha256(b"isolated-test-implementation").hexdigest()
        selected = commit or self.current_route()
        command = [
            "/usr/bin/git", "-C", str(self.route_root), "ls-tree", "-r",
            "--full-tree", selected, "--", *IMPLEMENTATION_DIRECTORIES,
        ]
        completed = subprocess.run(command, check=True, capture_output=True)
        return hashlib.sha256(completed.stdout).hexdigest()

    def route_is_ancestor(self, ancestor: str, current: str) -> bool:
        if self.route_commit is not None:
            return ancestor == current
        completed = subprocess.run(
            ["/usr/bin/git", "-C", str(self.route_root), "merge-base", "--is-ancestor", ancestor, current],
            check=False,
            capture_output=True,
        )
        return completed.returncode == 0

    def route_followups_are_reports_only(self, ancestor: str, current: str) -> bool:
        if self.route_commit is not None:
            return ancestor == current
        completed = subprocess.run(
            [
                "/usr/bin/git", "-C", str(self.route_root), "diff",
                "--name-only", "--no-renames", f"{ancestor}..{current}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        paths = [line for line in completed.stdout.splitlines() if line]
        return all(path.startswith("docs/") for path in paths)

    def fail(self, stage: str) -> None:
        if self.failure_hook is not None:
            self.failure_hook(stage)


def _production_dependencies() -> CampaignDependencies:
    return CampaignDependencies(
        DEFAULT_CAMPAIGN_ROOT, PROJECT_ROOT, _subprocess_runner,
        _authority=_PRODUCTION_AUTHORITY,
        source_policy=PRODUCTION_SOURCE_POLICY,
    )


def _layout(dependencies: CampaignDependencies) -> None:
    store = dependencies.store
    store.mkdir("attempts")
    store.mkdir("decisions")
    if not store.exists("issuance.lock"):
        try:
            store.exclusive_bytes("issuance.lock", b"")
        except FileExistsError:
            pass


def _load_ledger(dependencies: CampaignDependencies) -> dict[str, Any]:
    store = dependencies.store
    if not store.exists("issuance-ledger.json"):
        return {
            "schema_version": 3,
            "campaign_id": CAMPAIGN_ID,
            "implementation_code_sha256": dependencies.implementation_hash(),
            "entries": [],
        }
    ledger = store.read_json("issuance-ledger.json", label="issuance ledger")
    entries = ledger.get("entries")
    if set(ledger) != LEDGER_FIELDS or ledger.get("schema_version") != 3 or ledger.get("campaign_id") != CAMPAIGN_ID or type(entries) is not list:
        raise ValueError("issuance ledger identity is invalid")
    registered_code = ledger.get("implementation_code_sha256")
    if type(registered_code) is not str or len(registered_code) != 64:
        raise ValueError("issuance ledger implementation code hash is invalid")
    if registered_code != dependencies.implementation_hash():
        raise ValueError("campaign implementation code tree is stale")
    for row in entries:
        if type(row) is not dict:
            raise ValueError("issuance ledger entry must be an object")
        expected = ENTRY_COMPLETE_FIELDS if row.get("status") == "complete" else ENTRY_PENDING_FIELDS
        if set(row) != expected:
            raise ValueError("issuance ledger entry field set mismatch")
    counters = [row.get("counter") for row in entries if type(row) is dict]
    nonces = [row.get("nonce") for row in entries if type(row) is dict]
    if counters != list(range(1, len(entries) + 1)):
        raise ValueError("issuance ledger has a counter gap or duplicate")
    if len(nonces) != len(set(nonces)) or any(type(nonce) is not str or len(nonce) != 64 for nonce in nonces):
        raise ValueError("issuance ledger has a duplicate or invalid nonce")
    return ledger


def _with_lock(dependencies: CampaignDependencies):
    class Lock:
        def __enter__(self):
            dependencies.store.assert_identity()
            descriptor = os.open(
                "issuance.lock", os.O_RDWR | os.O_NOFOLLOW,
                dir_fd=dependencies.store.root_fd,
            )
            self.handle = os.fdopen(descriptor, "r+b")
            fcntl.flock(self.handle, fcntl.LOCK_EX)
            return self

        def __exit__(self, exc_type, exc, traceback):
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
    return Lock()


def _runner_argv(kind: str, architecture: str, attempt_dir: Path, dataset: str | None) -> list[str]:
    output = attempt_dir / "runner-result.json"
    command = [sys.executable, str(RUNNER)]
    if kind == "smoke":
        command += ["smoke", "--architecture", architecture, "--devices", "gpu", "--seed", "42", "--epochs", "1"]
    elif kind == "validation":
        assert dataset is not None
        command += [
            "stable-validation", "--architecture", architecture, "--dataset", dataset,
            "--recipe-index", str(FROZEN_RECIPES[dataset]), "--seed", "42", "--split-seed", "2024",
            "--cohort-sha256", FROZEN_COHORT_SHA256,
            "--architecture-fingerprint", architecture_fingerprint(architecture),
        ]
    elif kind == "test":
        command += ["stable-test", "--architecture", "a2", "--seed", "42", "--split-seed", "2024"]
    else:
        raise ValueError("unknown runner kind")
    command += ["--output", str(output)]
    return command


def _validate_runner_result(kind: str, architecture: str, dataset: str | None, result: Mapping[str, object]) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise ValueError("runner result must be an object")
    normalized = dict(result)
    expected_fields = {
        "validation": VALIDATION_RESULT_FIELDS,
        "smoke": SMOKE_RESULT_FIELDS,
        "test": TEST_RESULT_FIELDS,
    }.get(kind)
    if expected_fields is None or set(normalized) != expected_fields:
        raise ValueError(f"{kind} runner result field set mismatch")
    if normalized.get("architecture_fingerprint") != architecture_fingerprint(architecture):
        raise ValueError("runner architecture fingerprint mismatch")
    peak = normalized.get("peak_gpu_memory_gb")
    if type(peak) is not float or not math.isfinite(peak) or peak <= 0.0:
        raise ValueError("runner peak GPU memory must be a positive finite float")
    if type(normalized.get("gpu_uuid")) is not str or not normalized["gpu_uuid"]:
        raise ValueError("runner GPU UUID is invalid")
    if kind == "validation":
        if (
            normalized.get("schema_version") != 4
            or normalized.get("architecture") != architecture
            or normalized.get("architecture_manifest") != architecture_spec(architecture).manifest()
            or normalized.get("cohort_sha256") != FROZEN_COHORT_SHA256
        ):
            raise ValueError("validation runner v4 identity mismatch")
        if normalized.get("dataset_id") != dataset or normalized.get("recipe_index") != FROZEN_RECIPES.get(str(dataset)):
            raise ValueError("runner frozen dataset/recipe mismatch")
        if normalized.get("seed") != 42 or normalized.get("split_seed") != 2024:
            raise ValueError("runner seed binding mismatch")
        metrics = {field: normalized.get(field) for field in VALIDATION_METRICS}
        normalized.update(validate_metrics(metrics))
    elif kind == "smoke":
        shape = normalized.get("mastery_shape")
        if type(shape) is not list or len(shape) != 2 or any(type(item) is not int or item <= 0 for item in shape):
            raise ValueError("smoke mastery shape is invalid")
        if type(normalized.get("final_loss")) is not float or not math.isfinite(normalized["final_loss"]):
            raise ValueError("smoke final loss must be a finite float")
        if type(normalized.get("parameter_count")) is not int or normalized["parameter_count"] < 0:
            raise ValueError("smoke parameter count is invalid")
    elif kind == "test":
        if (
            normalized.get("schema_version") != 4
            or normalized.get("architecture") != "a2"
            or normalized.get("architecture_manifest") != architecture_spec("a2").manifest()
            or normalized.get("seed") != 42
            or normalized.get("split_seed") != 2024
            or normalized.get("cohort_sha256") != FROZEN_COHORT_SHA256
            or normalized.get("recipes") != FROZEN_RECIPES
        ):
            raise ValueError("test runner identity/schema mismatch")
        rows = normalized.get("rows")
        if type(rows) is not dict or set(rows) != set(FROZEN_RECIPES):
            raise ValueError("test runner rows must contain the exact frozen cohort")
        for dataset_id, recipe in FROZEN_RECIPES.items():
            row = rows[dataset_id]
            if type(row) is not dict or set(row) != TEST_ROW_FIELDS or row.get("recipe_index") != recipe:
                raise ValueError("test runner row field set/recipe mismatch")
            for metric in TEST_METRICS:
                value = row[metric]
                if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError("test metric must be an exact finite float in [0, 1]")
    return normalized


def _validate_route_chain(
    dependencies: CampaignDependencies,
    *,
    attempt_commit: object,
    attempt_code_hash: object,
    registered_code_hash: object,
) -> None:
    if type(attempt_commit) is not str or len(attempt_commit) != 40:
        raise ValueError("attempt route commit is invalid")
    current = dependencies.current_route()
    if not dependencies.route_is_ancestor(attempt_commit, current):
        raise ValueError("stale attempt route is not an ancestor of current HEAD")
    current_hash = dependencies.implementation_hash(current)
    attempt_hash = dependencies.implementation_hash(attempt_commit)
    if (
        type(attempt_code_hash) is not str
        or attempt_code_hash != registered_code_hash
        or attempt_hash != registered_code_hash
        or current_hash != registered_code_hash
    ):
        raise ValueError("campaign implementation code tree is stale")
    if not dependencies.route_followups_are_reports_only(attempt_commit, current):
        raise ValueError("attempt route has non-report follow-up commits")


def _canonical_attempt_proof(
    dependencies: CampaignDependencies,
    ledger: Mapping[str, object],
    entry: Mapping[str, object],
    result: Mapping[str, object],
    artifacts: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    if set(entry) != ENTRY_COMPLETE_FIELDS:
        raise ValueError("immutable ledger entry field set mismatch")
    counter = entry.get("counter")
    if type(counter) is not int or counter < 1:
        raise ValueError("immutable attempt counter is invalid")
    attempt_dir = f"attempts/attempt-{counter:03d}"
    if entry.get("attempt_dir") != attempt_dir:
        raise ValueError("immutable attempt directory mismatch")
    kind = entry.get("kind")
    split_ids = entry.get("split_ids")
    architecture = entry.get("architecture")
    dataset = entry.get("dataset_id")
    if kind not in {"smoke", "validation"} or architecture not in ARCHITECTURES:
        raise ValueError("immutable attempt kind/architecture mismatch")
    if kind == "validation":
        if split_ids != ["standard", "holdout"]:
            raise ValueError("validation attempt split identity mismatch")
        if dataset not in FROZEN_RECIPES:
            raise ValueError("immutable attempt dataset is outside frozen cohort")
    elif dataset is not None or split_ids != ["smoke"]:
        raise ValueError("smoke attempt split/dataset identity mismatch")
    nonce = entry.get("nonce")
    if type(nonce) is not str or len(nonce) != 64:
        raise ValueError("immutable attempt nonce is invalid")
    _validate_route_chain(
        dependencies,
        attempt_commit=entry.get("route_commit"),
        attempt_code_hash=entry.get("implementation_code_sha256"),
        registered_code_hash=ledger.get("implementation_code_sha256"),
    )
    normalized = _validate_runner_result(str(kind), str(architecture), dataset if isinstance(dataset, str) else None, result)
    if not artifacts or set(artifacts[0]) != ARTIFACT_FIELDS or artifacts[0].get("path") != f"{attempt_dir}/runner-result.json":
        raise ValueError("canonical attempt artifact field/path mismatch")
    source_count = len(_required_source_suffixes(dependencies, str(kind)))
    _validate_metric_sources(
        dependencies,
        attempt_dir=attempt_dir,
        kind=str(kind),
        result=normalized,
        artifacts=artifacts[1:1 + source_count],
    )
    _validate_aux_artifacts(
        dependencies,
        attempt_dir=attempt_dir,
        kind=str(kind),
        artifacts=artifacts[1 + source_count:],
    )
    argv = _runner_argv(
        str(kind), str(architecture),
        dependencies.campaign_root / attempt_dir,
        dataset if isinstance(dataset, str) else None,
    )
    return _seal({
        "schema_version": 3,
        "campaign_id": CAMPAIGN_ID,
        "cohort_sha256": FROZEN_COHORT_SHA256,
        "kind": kind,
        "split_ids": split_ids,
        "architecture": architecture,
        "architecture_manifest": architecture_spec(str(architecture)).manifest(),
        "architecture_fingerprint": architecture_fingerprint(str(architecture)),
        "dataset_id": dataset,
        "recipe_index": None if dataset is None else FROZEN_RECIPES[str(dataset)],
        "seed": 42,
        "split_seed": 2024,
        "route_commit": entry["route_commit"],
        "implementation_code_sha256": entry["implementation_code_sha256"],
        "counter": counter,
        "nonce": nonce,
        "runner_argv": argv,
        "runner_result": normalized,
        "artifacts": [dict(artifact) for artifact in artifacts],
    })


def _required_source_suffixes(
    dependencies: CampaignDependencies, kind: str
) -> tuple[str, ...]:
    policy = dependencies.source_policy
    if kind == "validation" and policy in {
        PRODUCTION_SOURCE_POLICY, "validation-fixed-sources", "all-fixed-sources",
        "production-layout-fixture",
    }:
        return VALIDATION_SOURCE_SUFFIXES
    if kind == "test" and policy in {
        PRODUCTION_SOURCE_POLICY, "test-fixed-sources", "all-fixed-sources",
        "production-layout-fixture",
    }:
        return TEST_SOURCE_SUFFIXES
    return ()


def _discover_source_artifacts(
    dependencies: CampaignDependencies, attempt_dir: str, kind: str
) -> list[dict[str, object]]:
    required = _required_source_suffixes(dependencies, kind)
    known = VALIDATION_SOURCE_SUFFIXES if kind == "validation" else TEST_SOURCE_SUFFIXES
    if not required:
        unexpected = [
            suffix for suffix in known
            if dependencies.store.exists(f"{attempt_dir}/{suffix}")
        ]
        if unexpected:
            raise ValueError("test fixture source policy forbids discovered source artifacts")
        return []
    paths = [f"{attempt_dir}/{suffix}" for suffix in required]
    missing = [path for path in paths if not dependencies.store.exists(path)]
    if missing:
        raise ValueError(f"required fixed source artifacts are missing: {missing}")
    fixed_root = f"{attempt_dir}/stable-{kind}-work"
    expected_children = (
        {"standard", "holdout"}
        if kind == "validation"
        else set(FROZEN_RECIPES)
    )
    if _requires_aux_layout(dependencies):
        expected_children.add("aux")
    if set(dependencies.store.listdir(fixed_root)) != expected_children:
        raise ValueError("fixed source root has missing or extra entries")
    expected_by_parent: dict[str, set[str]] = {}
    for path in paths:
        parent, name = str(Path(path).parent), Path(path).name
        expected_by_parent.setdefault(parent, set()).add(name)
    for parent, expected_names in expected_by_parent.items():
        if set(dependencies.store.listdir(parent)) != expected_names:
            raise ValueError("fixed source directory has missing or extra artifacts")
    return [
        dependencies.store.read_regular(path, label=f"{kind} source artifact")[1]
        for path in paths
    ]


def _requires_aux_layout(dependencies: CampaignDependencies) -> bool:
    return dependencies.source_policy in {
        PRODUCTION_SOURCE_POLICY,
        "production-layout-fixture",
    }


def _aux_identities(kind: str) -> tuple[str, ...]:
    return (
        ("standard", "holdout")
        if kind == "validation"
        else tuple(sorted(FROZEN_RECIPES))
    )


def _discover_aux_artifacts(
    dependencies: CampaignDependencies, attempt_dir: str, kind: str
) -> list[dict[str, object]]:
    if kind not in {"validation", "test"}:
        return []
    aux_root = f"{attempt_dir}/stable-{kind}-work/aux"
    if not _requires_aux_layout(dependencies):
        if dependencies.store.exists(aux_root):
            raise ValueError("fixture policy forbids an auxiliary output layout")
        return []
    identities = _aux_identities(kind)
    if set(dependencies.store.listdir(aux_root)) != set(identities):
        raise ValueError("fixed auxiliary root has missing or extra entries")
    names = set(_aux_names(kind))
    checkpoints: list[dict[str, object]] = []
    for identity in identities:
        parent = f"{aux_root}/{identity}"
        if set(dependencies.store.listdir(parent)) != names:
            raise ValueError("fixed auxiliary directory has missing or extra artifacts")
        checkpoint = f"{parent}/train-summary_best.pt"
        checkpoints.append(
            dependencies.store.read_regular(
                checkpoint, label=f"{kind} checkpoint artifact"
            )[1]
        )
    return checkpoints


def _validate_aux_artifacts(
    dependencies: CampaignDependencies,
    *,
    attempt_dir: str,
    kind: str,
    artifacts: Sequence[Mapping[str, object]],
) -> None:
    expected = (
        [
            f"{attempt_dir}/stable-{kind}-work/aux/{identity}/train-summary_best.pt"
            for identity in _aux_identities(kind)
        ]
        if kind in {"validation", "test"} and _requires_aux_layout(dependencies)
        else []
    )
    if [artifact.get("path") for artifact in artifacts] != expected:
        raise ValueError("checkpoint artifact registry mismatch")
    for artifact, path in zip(artifacts, expected, strict=True):
        if set(artifact) != ARTIFACT_FIELDS:
            raise ValueError("checkpoint artifact field set mismatch")
        _, current = dependencies.store.read_regular(
            path, label=f"{kind} checkpoint artifact"
        )
        if current != artifact:
            raise ValueError("checkpoint artifact identity/SHA mismatch")


def _validate_metric_sources(
    dependencies: CampaignDependencies,
    *,
    attempt_dir: str,
    kind: str,
    result: Mapping[str, object],
    artifacts: Sequence[Mapping[str, object]],
) -> None:
    required = _required_source_suffixes(dependencies, kind)
    if kind != "validation":
        if artifacts:
            if tuple(str(item.get("path", "")).removeprefix(f"{attempt_dir}/") for item in artifacts) != required:
                raise ValueError("non-validation attempt source registry mismatch")
        return
    expected_paths = [f"{attempt_dir}/{suffix}" for suffix in required]
    if len(artifacts) != len(expected_paths):
        raise ValueError("validation metric source artifact count mismatch")
    if not expected_paths:
        return
    for artifact, path in zip(artifacts, expected_paths, strict=True):
        if set(artifact) != ARTIFACT_FIELDS or artifact.get("path") != path:
            raise ValueError("validation metric source artifact registry mismatch")
        _, current = dependencies.store.read_regular(path, label="validation metric source")
        if current != artifact:
            raise ValueError("validation metric source artifact identity/SHA mismatch")
    sources: dict[tuple[str, str], dict[str, Any]] = {}
    for split in ("standard", "holdout"):
        for name in ("train-summary.json", "coverage-valid.json", "doa-valid.json"):
            path = f"{attempt_dir}/stable-validation-work/{split}/{name}"
            sources[(split, name)] = dependencies.store.read_json(path, label="validation metric source")
        train = sources[(split, "train-summary.json")]
        if (
            train.get("architecture_manifest") != result.get("architecture_manifest")
            or train.get("architecture_fingerprint") != result.get("architecture_fingerprint")
        ):
            raise ValueError("validation train source architecture mismatch")

    def coverage(split: str, scope: str) -> float:
        rows = sources[(split, "coverage-valid.json")].get("slices")
        if type(rows) is not list:
            raise ValueError("validation coverage metric source is invalid")
        selected = [row for row in rows if type(row) is dict and row.get("scope") == scope]
        if len(selected) != 1 or type(selected[0].get("auc")) is not float or not math.isfinite(selected[0]["auc"]):
            raise ValueError("validation coverage metric source is invalid")
        return selected[0]["auc"]

    doa_rows = sources[("standard", "doa-valid.json")].get("rows")
    if type(doa_rows) is not list or len(doa_rows) != 1 or type(doa_rows[0]) is not dict:
        raise ValueError("validation DOA metric source is invalid")
    recomputed = {
        "standard_overall_auc": coverage("standard", "overall"),
        "holdout_overall_auc": coverage("holdout", "overall"),
        "zero_auc": coverage("holdout", "bucket:zero"),
        "ordinary_doa": doa_rows[0].get("doa"),
        "weighted_doa": doa_rows[0].get("doa_weighted"),
    }
    validate_metrics(recomputed)
    if any(result.get(field) != value for field, value in recomputed.items()):
        raise ValueError("validation metric source differs from recomputed runner result")


def _validate_test_sources(
    dependencies: CampaignDependencies,
    result: Mapping[str, object],
    artifacts: Sequence[Mapping[str, object]],
) -> None:
    required = _required_source_suffixes(dependencies, "test")
    if not required:
        if artifacts:
            raise ValueError("aggregate-only test fixture has unexpected sources")
        return
    expected_paths = [f"attempts/test-once/{suffix}" for suffix in required]
    if [item.get("path") for item in artifacts] != expected_paths:
        raise ValueError("test source artifact registry mismatch")
    rows = result.get("rows")
    assert isinstance(rows, Mapping)
    for dataset in sorted(FROZEN_RECIPES):
        base = f"attempts/test-once/stable-test-work/{dataset}"
        train = dependencies.store.read_json(
            f"{base}/train-summary.json", label="test train source"
        )
        if (
            train.get("architecture_manifest") != result.get("architecture_manifest")
            or train.get("architecture_fingerprint") != result.get("architecture_fingerprint")
        ):
            raise ValueError("test train source architecture mismatch")
        coverage = dependencies.store.read_json(
            f"{base}/coverage-test.json", label="test coverage source"
        )
        slices = coverage.get("slices")
        if type(slices) is not list:
            raise ValueError("test coverage source is invalid")
        by_scope: dict[str, float] = {}
        for scope in ("overall", "bucket:zero"):
            selected = [
                row
                for row in slices
                if type(row) is dict and row.get("scope") == scope
            ]
            if len(selected) != 1:
                raise ValueError(
                    "test coverage source requires exactly one row per required scope"
                )
            by_scope[scope] = selected[0].get("auc")
        doa = dependencies.store.read_json(
            f"{base}/doa-test.json", label="test DOA source"
        )
        doa_rows = doa.get("rows")
        if type(doa_rows) is not list or len(doa_rows) != 1 or type(doa_rows[0]) is not dict:
            raise ValueError("test DOA source is invalid")
        recomputed = {
            "overall_auc": by_scope.get("overall"),
            "zero_auc": by_scope.get("bucket:zero"),
            "ordinary_doa": doa_rows[0].get("doa"),
            "weighted_doa": doa_rows[0].get("doa_weighted"),
        }
        for value in recomputed.values():
            if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("test source metric must be an exact finite float")
        result_row = rows[dataset]
        if any(result_row.get(field) != value for field, value in recomputed.items()):
            raise ValueError("test source metric differs from recomputed result")


def _run_registered(dependencies: CampaignDependencies, *, kind: str, architecture: str, dataset: str | None = None) -> dict[str, Any]:
    architecture_spec(architecture)
    store = dependencies.store
    _layout(dependencies)
    route_commit = dependencies.current_route()
    implementation_hash = dependencies.implementation_hash(route_commit)
    with _with_lock(dependencies):
        ledger = _load_ledger(dependencies)
        entries = ledger["entries"]
        if kind == "smoke" and any(row["kind"] == "smoke" and row["architecture"] == architecture for row in entries):
            raise ValueError(f"smoke for {architecture} has already been issued")
        if kind == "validation" and any(row["kind"] == kind and row["architecture"] == architecture and row.get("dataset_id") == dataset for row in entries):
            raise ValueError("validation attempt has already been issued")
        counter = len(entries) + 1
        nonce = secrets.token_hex(32)
        attempt_name = f"attempt-{counter:03d}"
        attempt_relative = f"attempts/{attempt_name}"
        store.mkdir_exclusive(attempt_relative)
        attempt_dir = dependencies.campaign_root / attempt_relative
        entry = {
            "counter": counter,
            "nonce": nonce,
            "kind": kind,
            "split_ids": ["standard", "holdout"] if kind == "validation" else ["smoke"],
            "architecture": architecture,
            "dataset_id": dataset,
            "attempt_dir": attempt_relative,
            "route_commit": route_commit,
            "implementation_code_sha256": implementation_hash,
            "status": "pending",
        }
        entries.append(entry)
        store.atomic_json("issuance-ledger.json", ledger)

    argv = _runner_argv(kind, architecture, attempt_dir, dataset)
    try:
        result = _validate_runner_result(kind, architecture, dataset, dependencies.runner(argv, attempt_dir))
        source_artifacts = _discover_source_artifacts(
            dependencies, attempt_relative, kind
        )
        aux_artifacts = _discover_aux_artifacts(
            dependencies, attempt_relative, kind
        )
        store.exclusive_json(f"{attempt_relative}/runner-result.json", result)
        _, artifact = store.read_regular(f"{attempt_relative}/runner-result.json", label="runner result")
        complete_entry = {**entry, "status": "complete", "proof_file_sha256": "0" * 64}
        proof = _canonical_attempt_proof(
            dependencies,
            ledger,
            complete_entry,
            result,
            [artifact, *source_artifacts, *aux_artifacts],
        )
        store.exclusive_json(f"{attempt_relative}/proof.json", proof)
    except BaseException:
        with _with_lock(dependencies):
            ledger = _load_ledger(dependencies)
            ledger["entries"][counter - 1]["status"] = "failed"
            store.atomic_json("issuance-ledger.json", ledger)
        raise
    with _with_lock(dependencies):
        ledger = _load_ledger(dependencies)
        current = ledger["entries"][counter - 1]
        if current["nonce"] != nonce or current["status"] != "pending":
            raise ValueError("issuance identity changed during runner execution")
        _, proof_record = store.read_regular(f"{attempt_relative}/proof.json", label="attempt proof")
        current["status"] = "complete"
        current["proof_file_sha256"] = proof_record["sha256"]
        store.atomic_json("issuance-ledger.json", ledger)
    print(json.dumps(proof, sort_keys=True))
    return proof


def _verify_attempts(dependencies: CampaignDependencies) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    store = dependencies.store
    _layout(dependencies)
    with _with_lock(dependencies):
        ledger = _load_ledger(dependencies)
        entries = json.loads(json.dumps(ledger["entries"]))
    proofs: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("status") != "complete":
            raise ValueError("status is invalid: issuance is not complete")
        attempt_relative = str(entry["attempt_dir"])
        proof = store.read_json(f"{attempt_relative}/proof.json", label="attempt proof")
        if set(proof) != ATTEMPT_PROOF_FIELDS:
            raise ValueError("canonical attempt proof field set mismatch")
        _verify_seal(proof, "attempt")
        _, proof_record = store.read_regular(f"{attempt_relative}/proof.json", label="attempt proof")
        if proof_record["sha256"] != entry.get("proof_file_sha256"):
            raise ValueError("attempt proof artifact SHA mismatch")
        artifacts = proof.get("artifacts")
        if type(artifacts) is not list or not artifacts:
            raise ValueError("attempt artifact registry is invalid")
        aggregate_path = f"{attempt_relative}/runner-result.json"
        _, aggregate = store.read_regular(aggregate_path, label="runner artifact")
        sources = _discover_source_artifacts(
            dependencies, attempt_relative, str(entry["kind"])
        )
        auxiliary = _discover_aux_artifacts(
            dependencies, attempt_relative, str(entry["kind"])
        )
        current_artifacts = [aggregate, *sources, *auxiliary]
        if artifacts != current_artifacts:
            raise ValueError(
                "proof artifact registry mismatch with fixed discovered artifacts"
            )
        result = store.read_json(aggregate_path, label="runner artifact")
        expected = _canonical_attempt_proof(
            dependencies, ledger, entry, result, current_artifacts
        )
        if proof != expected:
            raise ValueError("canonical attempt proof differs from recomputed immutable authority")
        proofs.append(proof)
    return ledger, proofs


def _write_decision(dependencies: CampaignDependencies, name: str, payload: Mapping[str, object]) -> dict[str, Any]:
    sealed = _seal(payload)
    dependencies.store.exclusive_json(f"decisions/{name}", sealed)
    return sealed


def _load_decision(dependencies: CampaignDependencies, name: str) -> dict[str, Any]:
    payload = dependencies.store.read_json(f"decisions/{name}", label=name)
    _verify_seal(payload, name)
    return payload


def _command_smoke(args: argparse.Namespace, deps: CampaignDependencies) -> int:
    _run_registered(deps, kind="smoke", architecture=args.architecture)
    return 0


def _command_validation(args: argparse.Namespace, deps: CampaignDependencies) -> int:
    _run_registered(deps, kind="validation", architecture=args.architecture, dataset=args.dataset)
    return 0


def _command_replay(args: argparse.Namespace, deps: CampaignDependencies) -> int:
    payload = _expected_replay(deps, args.architecture)
    decision = _write_decision(deps, f"replay-{args.architecture}.json", payload)
    print(json.dumps(decision, sort_keys=True))
    return 0


def _expected_replay(
    deps: CampaignDependencies,
    architecture: str,
    *,
    decision_route: str | None = None,
) -> dict[str, Any]:
    ledger, proofs = _verify_attempts(deps)
    selected = [proof for proof in proofs if proof["kind"] == "validation" and proof["architecture"] == architecture]
    if {proof["dataset_id"] for proof in selected} != set(FROZEN_RECIPES) or len(selected) != 3:
        raise ValueError("replay requires exactly one registered proof per frozen dataset")
    selected.sort(key=lambda proof: proof["counter"])
    return {
        "schema_version": 3,
        "campaign_id": CAMPAIGN_ID,
        "cohort_sha256": FROZEN_COHORT_SHA256,
        "architecture": architecture,
        "architecture_fingerprint": architecture_fingerprint(architecture),
        "route_commit": decision_route or deps.current_route(),
        "implementation_code_sha256": ledger["implementation_code_sha256"],
        "counters": [proof["counter"] for proof in selected],
        "nonces": [proof["nonce"] for proof in selected],
        "proof_sha256s": [proof["proof_sha256"] for proof in selected],
        "artifact_sha256s": [
            artifact["sha256"]
            for proof in selected
            for artifact in proof["artifacts"]
        ],
        "rows": {proof["dataset_id"]: proof["runner_result"] for proof in selected},
    }


def _verified_replay(deps: CampaignDependencies, architecture: str) -> dict[str, Any]:
    replay = _load_decision(deps, f"replay-{architecture}.json")
    _validate_route_chain(
        deps,
        attempt_commit=replay.get("route_commit"),
        attempt_code_hash=replay.get("implementation_code_sha256"),
        registered_code_hash=deps.implementation_hash(),
    )
    expected = _expected_replay(
        deps, architecture, decision_route=str(replay.get("route_commit"))
    )
    unsigned = dict(replay)
    unsigned.pop("proof_sha256", None)
    if unsigned != expected:
        raise ValueError("registered replay was forged, resealed, or is stale")
    return replay


def _command_freeze(args: argparse.Namespace, deps: CampaignDependencies) -> int:
    replay = _verified_replay(deps, "a0v4")
    payload = _write_decision(deps, "a0v4-frozen.json", {
        "schema_version": 3,
        "campaign_id": CAMPAIGN_ID,
        "cohort_sha256": FROZEN_COHORT_SHA256,
        "route_commit": deps.current_route(),
        "implementation_code_sha256": deps.implementation_hash(),
        "architecture": "a0v4",
        "architecture_fingerprint": architecture_fingerprint("a0v4"),
        "replay_proof_sha256": replay["proof_sha256"],
        "rows": replay["rows"],
        "counters": replay["counters"],
        "nonces": replay["nonces"],
        "proof_sha256s": replay["proof_sha256s"],
        "artifact_sha256s": replay["artifact_sha256s"],
    })
    print(json.dumps(payload, sort_keys=True))
    return 0


def _verified_frozen(deps: CampaignDependencies) -> dict[str, Any]:
    frozen = _load_decision(deps, "a0v4-frozen.json")
    replay = _verified_replay(deps, "a0v4")
    _validate_route_chain(
        deps,
        attempt_commit=frozen.get("route_commit"),
        attempt_code_hash=frozen.get("implementation_code_sha256"),
        registered_code_hash=deps.implementation_hash(),
    )
    if frozen.get("replay_proof_sha256") != replay.get("proof_sha256") or frozen.get("rows") != replay.get("rows"):
        raise ValueError("frozen A0v4 proof chain mismatch")
    return frozen


def relative_gate(deltas: Mapping[str, Mapping[str, object]]) -> dict[str, Any]:
    return stable_graph_relative_gate(deltas)


def _command_relative(args: argparse.Namespace, deps: CampaignDependencies) -> int:
    derived = _derive_relative(deps)
    payload = _write_decision(deps, "relative-gate.json", derived)
    print(json.dumps(payload, sort_keys=True))
    return 0


def _derive_relative(
    deps: CampaignDependencies,
    *,
    nonce: str | None = None,
    decision_route: str | None = None,
) -> dict[str, Any]:
    baseline = _verified_frozen(deps)
    candidate = _verified_replay(deps, "a2")
    deltas: dict[str, dict[str, float]] = {}
    for dataset in sorted(FROZEN_RECIPES):
        deltas[dataset] = {}
        for field in VALIDATION_METRICS:
            left = validate_metrics({name: baseline["rows"][dataset][name] for name in VALIDATION_METRICS})[field]
            right = validate_metrics({name: candidate["rows"][dataset][name] for name in VALIDATION_METRICS})[field]
            deltas[dataset][field] = right - left
    gate_input = {
        dataset: {
            "standard": rows["standard_overall_auc"],
            "holdout": rows["holdout_overall_auc"],
            "zero": rows["zero_auc"],
        }
        for dataset, rows in deltas.items()
    }
    decision = relative_gate(gate_input)
    gate_nonce = (nonce or secrets.token_hex(32)) if decision["passed"] else None
    return {
        "schema_version": 3,
        "campaign_id": CAMPAIGN_ID,
        "cohort_sha256": FROZEN_COHORT_SHA256,
        "route_commit": decision_route or deps.current_route(),
        "implementation_code_sha256": deps.implementation_hash(),
        **decision,
        "deltas": deltas,
        "a0v4_frozen_sha256": baseline["proof_sha256"],
        "a2_replay_sha256": candidate["proof_sha256"],
        "proof_sha256s": baseline["proof_sha256s"] + candidate["proof_sha256s"],
        "artifact_sha256s": baseline["artifact_sha256s"] + candidate["artifact_sha256s"],
        "counters": baseline["counters"] + candidate["counters"],
        "nonces": baseline["nonces"] + candidate["nonces"],
        "nonce": gate_nonce,
    }


def _verified_relative(deps: CampaignDependencies) -> dict[str, Any]:
    proof = _load_decision(deps, "relative-gate.json")
    _validate_route_chain(
        deps,
        attempt_commit=proof.get("route_commit"),
        attempt_code_hash=proof.get("implementation_code_sha256"),
        registered_code_hash=deps.implementation_hash(),
    )
    expected = _derive_relative(
        deps,
        nonce=proof.get("nonce") if isinstance(proof.get("nonce"), str) else None,
        decision_route=str(proof.get("route_commit")),
    )
    unsigned = dict(proof)
    unsigned.pop("proof_sha256", None)
    if unsigned != expected:
        raise ValueError("relative proof was forged or resealed")
    return proof


def load_verified_comparator_audit(path: Path) -> dict[str, Any]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        data = handle.read()
        after = os.fstat(handle.fileno())
    if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("comparator audit is not a stable regular file")
    audit = json.loads(data)
    if type(audit) is not dict:
        raise ValueError("comparator audit must be an object")
    registered = audit.get("audit_sha256")
    unsigned = dict(audit)
    unsigned.pop("audit_sha256", None)
    if registered != canonical_sha256(unsigned):
        raise ValueError("comparator audit canonical SHA-256 mismatch")
    if registered != FROZEN_COMPARATOR_AUDIT_SHA256:
        raise ValueError("comparator audit is not the frozen campaign audit")
    return audit


def _strongest_rows(audit: Mapping[str, object], dataset: str) -> dict[str, Mapping[str, object]]:
    strongest = audit.get("strongest_comparators")
    if not isinstance(strongest, Mapping) or not isinstance(strongest.get(dataset), Mapping):
        raise ValueError("comparator audit strongest registry is incomplete")
    source = strongest[dataset]
    assert isinstance(source, Mapping)
    try:
        rows = {
            "standard": source["standard"]["overall_auc"],
            "holdout": source["holdout"]["overall_auc"],
            "zero": source["holdout"]["zero_auc"],
        }
    except (KeyError, TypeError) as error:
        raise ValueError("same-protocol strongest comparator rows are incomplete") from error
    if any(not isinstance(row, Mapping) for row in rows.values()):
        raise ValueError("strongest comparator row identity is invalid")
    return rows


def _command_external(args: argparse.Namespace, deps: CampaignDependencies) -> int:
    relative = _verified_relative(deps)
    if relative.get("passed") is not True or type(relative.get("nonce")) is not str:
        raise ValueError("relative gate has not passed")
    candidate = _verified_replay(deps, "a2")
    audit = load_verified_comparator_audit(args.comparator_audit)
    derived = _derive_external(
        deps,
        relative=relative,
        candidate=candidate,
        audit=audit,
        audit_path=args.comparator_audit.absolute(),
    )
    payload = _write_decision(deps, "external-gate.json", derived)
    print(json.dumps(payload, sort_keys=True))
    return 0


def _derive_external(
    deps: CampaignDependencies,
    *,
    relative: Mapping[str, object],
    candidate: Mapping[str, object],
    audit: Mapping[str, object],
    audit_path: Path,
    nonce: str | None = None,
    decision_route: str | None = None,
) -> dict[str, Any]:
    deltas: dict[str, dict[str, float]] = {}
    identities: dict[str, dict[str, object]] = {}
    passed = True
    for dataset in sorted(FROZEN_RECIPES):
        rows = _strongest_rows(audit, dataset)
        identities[dataset] = {}
        deltas[dataset] = {}
        for condition, metric in (("standard", "standard_overall_auc"), ("holdout", "holdout_overall_auc"), ("zero", "zero_auc")):
            row = dict(rows[condition])
            value = row.get("value")
            if type(value) is not float or not math.isfinite(value):
                raise ValueError("comparator value must be an exact finite float")
            identities[dataset][condition] = {"row_sha256": canonical_sha256(row), "row": row}
            delta = candidate["rows"][dataset][metric] - value
            deltas[dataset][condition] = delta
            passed = passed and (delta > 0.0 if condition == "zero" else delta >= 0.0)
    return {
        "schema_version": 3,
        "campaign_id": CAMPAIGN_ID,
        "cohort_sha256": FROZEN_COHORT_SHA256,
        "route_commit": decision_route or deps.current_route(),
        "implementation_code_sha256": deps.implementation_hash(),
        "passed": passed,
        "nonce": (nonce or secrets.token_hex(32)) if passed else None,
        "relative_proof_sha256": relative["proof_sha256"],
        "a2_replay_sha256": candidate["proof_sha256"],
        "comparator_audit_sha256": audit["audit_sha256"],
        "comparator_audit_path": str(audit_path),
        "comparator_rows": identities,
        "deltas": deltas,
    }


def _verified_external(deps: CampaignDependencies) -> dict[str, Any]:
    proof = _load_decision(deps, "external-gate.json")
    audit_path = proof.get("comparator_audit_path")
    if type(audit_path) is not str:
        raise ValueError("external proof comparator audit path is invalid")
    audit = load_verified_comparator_audit(Path(audit_path))
    relative = _verified_relative(deps)
    candidate = _verified_replay(deps, "a2")
    _validate_route_chain(
        deps,
        attempt_commit=proof.get("route_commit"),
        attempt_code_hash=proof.get("implementation_code_sha256"),
        registered_code_hash=deps.implementation_hash(),
    )
    derived = _derive_external(
        deps,
        relative=relative,
        candidate=candidate,
        audit=audit,
        audit_path=Path(audit_path),
        nonce=proof.get("nonce") if isinstance(proof.get("nonce"), str) else None,
        decision_route=str(proof.get("route_commit")),
    )
    unsigned = dict(proof)
    unsigned.pop("proof_sha256", None)
    if unsigned != derived:
        raise ValueError("external proof was forged or resealed")
    return proof


def _command_test(args: argparse.Namespace, deps: CampaignDependencies) -> int:
    if args.architecture != "a2":
        raise ValueError("test remains closed for every architecture except a2")
    relative = _verified_relative(deps)
    external = _verified_external(deps)
    if relative.get("passed") is not True or external.get("passed") is not True or external.get("relative_proof_sha256") != relative.get("proof_sha256"):
        raise ValueError("test remains closed until registered gates pass")
    _verified_replay(deps, "a2")
    _layout(deps)
    route_commit = deps.current_route()
    implementation_hash = deps.implementation_hash(route_commit)
    nonce = secrets.token_hex(32)
    pending = _seal({
        "schema_version": 3,
        "campaign_id": CAMPAIGN_ID,
        "cohort_sha256": FROZEN_COHORT_SHA256,
        "state": "pending",
        "nonce": nonce,
        "route_commit": route_commit,
        "implementation_code_sha256": implementation_hash,
        "architecture": "a2",
        "split_id": "test",
        "architecture_manifest": architecture_spec("a2").manifest(),
        "architecture_fingerprint": architecture_fingerprint("a2"),
        "recipes": dict(FROZEN_RECIPES),
        "relative_proof_sha256": relative["proof_sha256"],
        "external_proof_sha256": external["proof_sha256"],
    })
    try:
        deps.store.exclusive_json("decisions/test-once.json", pending)
    except FileExistsError as error:
        raise ValueError("test-once nonce has already been consumed") from error
    state = "failed"
    argv: list[str] | None = None
    result: dict[str, Any] | None = None
    artifacts: list[dict[str, object]] | None = None
    error_text: str | None = None
    try:
        deps.fail("mkdir")
        deps.store.mkdir_exclusive("attempts/test-once")
        attempt_dir = deps.campaign_root / "attempts" / "test-once"
        deps.fail("argv")
        argv = _runner_argv("test", "a2", attempt_dir, None)
        deps.fail("runner")
        result = _validate_runner_result("test", "a2", None, deps.runner(argv, attempt_dir))
        deps.fail("artifact")
        source_artifacts = _discover_source_artifacts(
            deps, "attempts/test-once", "test"
        )
        aux_artifacts = _discover_aux_artifacts(
            deps, "attempts/test-once", "test"
        )
        deps.store.exclusive_json("attempts/test-once/runner-result.json", result)
        _, aggregate_artifact = deps.store.read_regular(
            "attempts/test-once/runner-result.json", label="test runner result"
        )
        artifacts = [aggregate_artifact, *source_artifacts, *aux_artifacts]
        _validate_test_sources(deps, result, source_artifacts)
        state = "succeeded"
    except BaseException as error:
        error_text = f"{type(error).__name__}: {error}"
        result = None
        artifacts = None
    finally:
        finalized = _seal({
            "schema_version": 3,
            "campaign_id": CAMPAIGN_ID,
            "cohort_sha256": FROZEN_COHORT_SHA256,
            "state": state,
            "nonce": nonce,
            "route_commit": route_commit,
            "implementation_code_sha256": implementation_hash,
            "architecture": "a2",
            "split_id": "test",
            "architecture_manifest": architecture_spec("a2").manifest(),
            "architecture_fingerprint": architecture_fingerprint("a2"),
            "recipes": dict(FROZEN_RECIPES),
            "relative_proof_sha256": relative["proof_sha256"],
            "external_proof_sha256": external["proof_sha256"],
            "runner_argv": argv,
            "runner_result": result,
            "artifacts": artifacts,
            "error": error_text,
        })
        try:
            deps.fail("finalize")
            deps.store.atomic_json("decisions/test-once.json", finalized)
        except BaseException as final_error:
            state = "failed"
            error_text = f"{type(final_error).__name__}: {final_error}"
            finalized = _seal({
                "schema_version": 3,
                "campaign_id": CAMPAIGN_ID,
                "cohort_sha256": FROZEN_COHORT_SHA256,
                "state": state,
                "nonce": nonce,
                "route_commit": route_commit,
                "implementation_code_sha256": implementation_hash,
                "architecture": "a2",
                "split_id": "test",
                "architecture_manifest": architecture_spec("a2").manifest(),
                "architecture_fingerprint": architecture_fingerprint("a2"),
                "recipes": dict(FROZEN_RECIPES),
                "relative_proof_sha256": relative["proof_sha256"],
                "external_proof_sha256": external["proof_sha256"],
                "runner_argv": argv,
                "runner_result": None,
                "artifacts": None,
                "error": error_text,
            })
            try:
                deps.store.atomic_json("decisions/test-once.json", finalized)
            except BaseException:
                pass
    if state == "failed":
        raise RuntimeError(error_text)
    print(json.dumps(finalized, sort_keys=True))
    return 0


def _verified_test_record(deps: CampaignDependencies) -> dict[str, Any]:
    record = deps.store.read_json("decisions/test-once.json", label="test-once.json")
    expected_fields = {
        "schema_version", "campaign_id", "cohort_sha256", "state", "nonce",
        "route_commit", "implementation_code_sha256", "architecture",
        "split_id",
        "architecture_manifest", "architecture_fingerprint", "recipes",
        "relative_proof_sha256", "external_proof_sha256", "runner_argv",
        "runner_result", "artifacts", "error", "proof_sha256",
    }
    if set(record) != expected_fields:
        raise ValueError("test-once record field set mismatch")
    _verify_seal(record, "test-once")
    relative = _verified_relative(deps)
    external = _verified_external(deps)
    _validate_route_chain(
        deps,
        attempt_commit=record.get("route_commit"),
        attempt_code_hash=record.get("implementation_code_sha256"),
        registered_code_hash=deps.implementation_hash(),
    )
    identity = (
        record.get("schema_version") == 3
        and record.get("campaign_id") == CAMPAIGN_ID
        and record.get("cohort_sha256") == FROZEN_COHORT_SHA256
        and record.get("architecture") == "a2"
        and record.get("split_id") == "test"
        and record.get("architecture_manifest") == architecture_spec("a2").manifest()
        and record.get("architecture_fingerprint") == architecture_fingerprint("a2")
        and record.get("recipes") == FROZEN_RECIPES
        and record.get("relative_proof_sha256") == relative.get("proof_sha256")
        and record.get("external_proof_sha256") == external.get("proof_sha256")
        and type(record.get("nonce")) is str
        and len(record["nonce"]) == 64
    )
    if not identity:
        raise ValueError("test-once identity/proof chain mismatch")
    state = record.get("state")
    if state == "failed":
        if record.get("runner_result") is not None or record.get("artifacts") is not None or type(record.get("error")) is not str:
            raise ValueError("failed test-once record contains a forged result/artifact")
        argv = record.get("runner_argv")
        if argv is not None and (type(argv) is not list or any(type(item) is not str for item in argv)):
            raise ValueError("failed test-once argv is invalid")
        return record
    if state != "succeeded":
        raise ValueError("test-once record is pending or has invalid state")
    expected_argv = _runner_argv(
        "test", "a2", deps.campaign_root / "attempts" / "test-once", None
    )
    if record.get("runner_argv") != expected_argv or record.get("error") is not None:
        raise ValueError("succeeded test-once argv/error mismatch")
    registered = record.get("artifacts")
    if type(registered) is not list or not registered:
        raise ValueError("succeeded test result artifact registry is invalid")
    aggregate_path = "attempts/test-once/runner-result.json"
    _, aggregate = deps.store.read_regular(aggregate_path, label="test result artifact")
    sources = _discover_source_artifacts(deps, "attempts/test-once", "test")
    auxiliary = _discover_aux_artifacts(deps, "attempts/test-once", "test")
    current_artifacts = [aggregate, *sources, *auxiliary]
    if registered != current_artifacts:
        raise ValueError("test proof artifacts differ from fixed discovered artifacts")
    result = deps.store.read_json(aggregate_path, label="test result artifact")
    normalized = _validate_runner_result("test", "a2", None, result)
    _validate_test_sources(deps, normalized, sources)
    if normalized != record.get("runner_result"):
        raise ValueError("test result differs from recomputed artifact")
    return record


def _command_status(args: argparse.Namespace, deps: CampaignDependencies) -> int:
    ledger, proofs = _verify_attempts(deps)
    decisions: dict[str, str] = {}
    for name in deps.store.listdir("decisions"):
        if name == "replay-a0v4.json":
            _verified_replay(deps, "a0v4")
        elif name == "replay-a2.json":
            _verified_replay(deps, "a2")
        elif name == "a0v4-frozen.json":
            _verified_frozen(deps)
        elif name == "relative-gate.json":
            _verified_relative(deps)
        elif name == "external-gate.json":
            _verified_external(deps)
        elif name == "test-once.json":
            _verified_test_record(deps)
        else:
            raise ValueError(f"status found an unregistered decision artifact: {name}")
        _, record = deps.store.read_regular(f"decisions/{name}", label=name)
        decisions[name] = str(record["sha256"])
    print(json.dumps({"campaign_id": CAMPAIGN_ID, "attempt_count": len(proofs), "ledger_entry_count": len(ledger["entries"]), "decisions": decisions}, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Secure stable graph campaign controller")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    smoke.add_argument("--seed", type=int, choices=(42,), default=42)
    smoke.add_argument("--epochs", type=int, choices=(1,), default=1)
    smoke.add_argument("--campaign-root", type=Path, choices=(DEFAULT_CAMPAIGN_ROOT,), default=DEFAULT_CAMPAIGN_ROOT)
    smoke.set_defaults(handler=_command_smoke)
    validation = subparsers.add_parser("run-validation")
    validation.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    validation.add_argument("--dataset", choices=FROZEN_RECIPES, required=True)
    validation.add_argument("--seed", type=int, choices=(42,), default=42)
    validation.add_argument("--split-seed", type=int, choices=(2024,), default=2024)
    validation.add_argument("--campaign-root", type=Path, choices=(DEFAULT_CAMPAIGN_ROOT,), default=DEFAULT_CAMPAIGN_ROOT)
    validation.set_defaults(handler=_command_validation)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    replay.set_defaults(handler=_command_replay)
    freeze = subparsers.add_parser("freeze-a0v4")
    freeze.set_defaults(handler=_command_freeze)
    relative = subparsers.add_parser("relative-gate")
    relative.set_defaults(handler=_command_relative)
    external = subparsers.add_parser("external-gate")
    external.add_argument("--comparator-audit", type=Path, default=DEFAULT_COMPARATOR_AUDIT)
    external.set_defaults(handler=_command_external)
    test_once = subparsers.add_parser("run-test-once")
    test_once.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    test_once.set_defaults(handler=_command_test)
    status = subparsers.add_parser("status")
    status.set_defaults(handler=_command_status)
    return parser


def execute(argv: Sequence[str] | None = None, *, dependencies: CampaignDependencies | None = None) -> int:
    args = build_parser().parse_args(argv)
    deps = dependencies or _production_dependencies()
    if dependencies is None and hasattr(args, "campaign_root") and args.campaign_root != DEFAULT_CAMPAIGN_ROOT:
        raise ValueError("exact production campaign root is required")
    return int(args.handler(args, deps))


def main(argv: Sequence[str] | None = None) -> int:
    return execute(argv)


if __name__ == "__main__":
    raise SystemExit(main())
