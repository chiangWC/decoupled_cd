from __future__ import annotations

import argparse
from dataclasses import dataclass
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
DEFAULT_CAMPAIGN_ROOT = Path(
    "/home/xph/jwc/research/local_data/decoupled_cd_codex_routes"
) / CAMPAIGN_ID
DEFAULT_COMPARATOR_AUDIT = Path(
    "/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/"
    "unified-mastery-20260712/audit/comparator-audit.json"
)
RUNNER = PROJECT_ROOT / "scripts" / "run_unified_validation.py"
Runner = Callable[[Sequence[str], Path], Mapping[str, object]]


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


def _ensure_nofollow_directory(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as error:
            raise ValueError(f"campaign directory is missing: {current}") from error
        if stat.S_ISLNK(mode):
            raise ValueError(f"campaign path must not contain a symlink: {current}")
        if not stat.S_ISDIR(mode):
            raise ValueError(f"campaign path parent is not a directory: {current}")
    descriptor = os.open(absolute, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError("campaign root is not a directory")
    finally:
        os.close(descriptor)


def _under(root: Path, path: Path) -> Path:
    root_absolute = root.absolute()
    candidate = path.absolute()
    try:
        candidate.relative_to(root_absolute)
    except ValueError as error:
        raise ValueError("artifact is outside the fixed campaign root") from error
    return candidate


def _mkdir(root: Path, relative: str) -> Path:
    path = _under(root, root / relative)
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise ValueError(f"campaign directory is invalid: {path}")
    return path


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_bytes(root: Path, path: Path, data: bytes) -> None:
    path = _under(root, path)
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
        0o600,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _exclusive_json(root: Path, path: Path, payload: object) -> None:
    _exclusive_bytes(root, path, _json_bytes(payload))


def _atomic_json(root: Path, path: Path, payload: object) -> None:
    path = _under(root, path)
    temporary = path.parent / f".{path.name}.{secrets.token_hex(16)}.tmp"
    _exclusive_bytes(root, temporary, _json_bytes(payload))
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _read_regular(root: Path, path: Path, *, label: str) -> tuple[bytes, dict[str, object]]:
    path = _under(root, path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} is not a regular no-follow file")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise ValueError(f"{label} identity changed while hashing")
    return b"".join(chunks), {
        "path": str(path.relative_to(root.absolute())),
        "size_bytes": before.st_size,
        "sha256": digest.hexdigest(),
        "device": before.st_dev,
        "inode": before.st_ino,
    }


def _read_json(root: Path, path: Path, *, label: str) -> dict[str, Any]:
    data, _ = _read_regular(root, path, label=label)
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON") from error
    if type(payload) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return payload


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
        "architecture_fingerprint": architecture_fingerprint(architecture),
        "dataset_id": dataset,
        "recipe_index": FROZEN_RECIPES[dataset],
        "seed": 42,
        "split_seed": 2024,
        "gpu_uuid": "GPU-fake",
        "peak_gpu_memory_gb": 0.25,
        **{field: 0.5 for field in VALIDATION_METRICS},
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
        payload = dict(records[0])
        payload["gpu_uuid"] = payload.pop("physical_gpu_uuid", None)
    return payload


@dataclass(frozen=True)
class CampaignDependencies:
    campaign_root: Path
    route_root: Path
    runner: Runner
    route_commit: str | None = None
    test_only: bool = False

    def __post_init__(self) -> None:
        root = self.campaign_root.absolute()
        if root.name != CAMPAIGN_ID:
            raise ValueError("exact campaign root name is required")
        if not self.test_only and root != DEFAULT_CAMPAIGN_ROOT:
            raise ValueError("exact production campaign root is required")
        _ensure_nofollow_directory(root)

    @classmethod
    def for_test(
        cls,
        *,
        campaign_root: Path,
        route_root: Path,
        runner: Runner,
        route_commit: str,
    ) -> CampaignDependencies:
        if type(route_commit) is not str or len(route_commit) != 40:
            raise ValueError("test route commit must be 40-hex")
        return cls(campaign_root, route_root, runner, route_commit, True)

    def current_route(self) -> str:
        return self.route_commit if self.route_commit is not None else _route_head(self.route_root)


def _production_dependencies() -> CampaignDependencies:
    return CampaignDependencies(DEFAULT_CAMPAIGN_ROOT, PROJECT_ROOT, _subprocess_runner)


def _layout(dependencies: CampaignDependencies) -> None:
    root = dependencies.campaign_root.absolute()
    _mkdir(root, "attempts")
    _mkdir(root, "decisions")
    lock = root / "issuance.lock"
    if not lock.exists():
        try:
            _exclusive_bytes(root, lock, b"")
        except FileExistsError:
            pass


def _load_ledger(root: Path) -> dict[str, Any]:
    path = root / "issuance-ledger.json"
    if not path.exists():
        return {"schema_version": 2, "campaign_id": CAMPAIGN_ID, "entries": []}
    ledger = _read_json(root, path, label="issuance ledger")
    entries = ledger.get("entries")
    if ledger.get("schema_version") != 2 or ledger.get("campaign_id") != CAMPAIGN_ID or type(entries) is not list:
        raise ValueError("issuance ledger identity is invalid")
    counters = [row.get("counter") for row in entries if type(row) is dict]
    nonces = [row.get("nonce") for row in entries if type(row) is dict]
    if counters != list(range(1, len(entries) + 1)):
        raise ValueError("issuance ledger has a counter gap or duplicate")
    if len(nonces) != len(set(nonces)) or any(type(nonce) is not str or len(nonce) != 64 for nonce in nonces):
        raise ValueError("issuance ledger has a duplicate or invalid nonce")
    return ledger


def _with_lock(root: Path):
    class Lock:
        def __enter__(self):
            descriptor = os.open(root / "issuance.lock", os.O_RDWR | os.O_NOFOLLOW)
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
    normalized = dict(result)
    if normalized.get("architecture_fingerprint") != architecture_fingerprint(architecture):
        raise ValueError("runner architecture fingerprint mismatch")
    peak = normalized.get("peak_gpu_memory_gb")
    if type(peak) is not float or not math.isfinite(peak) or peak <= 0.0:
        raise ValueError("runner peak GPU memory must be a positive finite float")
    if type(normalized.get("gpu_uuid")) is not str or not normalized["gpu_uuid"]:
        raise ValueError("runner GPU UUID is invalid")
    if kind == "validation":
        if normalized.get("dataset_id") != dataset or normalized.get("recipe_index") != FROZEN_RECIPES.get(str(dataset)):
            raise ValueError("runner frozen dataset/recipe mismatch")
        if normalized.get("seed") != 42 or normalized.get("split_seed") != 2024:
            raise ValueError("runner seed binding mismatch")
        metrics = {field: normalized.get(field) for field in VALIDATION_METRICS}
        normalized.update(validate_metrics(metrics))
    return normalized


def _run_registered(dependencies: CampaignDependencies, *, kind: str, architecture: str, dataset: str | None = None) -> dict[str, Any]:
    architecture_spec(architecture)
    root = dependencies.campaign_root.absolute()
    _layout(dependencies)
    route_commit = dependencies.current_route()
    with _with_lock(root):
        ledger = _load_ledger(root)
        entries = ledger["entries"]
        if kind == "smoke" and any(row["kind"] == "smoke" and row["architecture"] == architecture for row in entries):
            raise ValueError(f"smoke for {architecture} has already been issued")
        if kind == "validation" and any(row["kind"] == kind and row["architecture"] == architecture and row.get("dataset_id") == dataset for row in entries):
            raise ValueError("validation attempt has already been issued")
        counter = len(entries) + 1
        nonce = secrets.token_hex(32)
        attempt_name = f"attempt-{counter:03d}"
        attempt_dir = root / "attempts" / attempt_name
        os.mkdir(attempt_dir, 0o700)
        entry = {
            "counter": counter,
            "nonce": nonce,
            "kind": kind,
            "architecture": architecture,
            "dataset_id": dataset,
            "attempt_dir": f"attempts/{attempt_name}",
            "route_commit": route_commit,
            "status": "pending",
        }
        entries.append(entry)
        _atomic_json(root, root / "issuance-ledger.json", ledger)

    argv = _runner_argv(kind, architecture, attempt_dir, dataset)
    try:
        result = _validate_runner_result(kind, architecture, dataset, dependencies.runner(argv, attempt_dir))
        _exclusive_json(root, attempt_dir / "runner-result.json", result)
        _, artifact = _read_regular(root, attempt_dir / "runner-result.json", label="runner result")
        proof = _seal({
            "schema_version": 2,
            "campaign_id": CAMPAIGN_ID,
            "cohort_sha256": FROZEN_COHORT_SHA256,
            "kind": kind,
            "architecture": architecture,
            "architecture_manifest": architecture_spec(architecture).manifest(),
            "architecture_fingerprint": architecture_fingerprint(architecture),
            "dataset_id": dataset,
            "recipe_index": None if dataset is None else FROZEN_RECIPES[dataset],
            "seed": 42,
            "split_seed": 2024,
            "route_commit": route_commit,
            "counter": counter,
            "nonce": nonce,
            "runner_argv": argv,
            "runner_result": result,
            "artifacts": [artifact],
        })
        _exclusive_json(root, attempt_dir / "proof.json", proof)
    except BaseException:
        with _with_lock(root):
            ledger = _load_ledger(root)
            ledger["entries"][counter - 1]["status"] = "failed"
            _atomic_json(root, root / "issuance-ledger.json", ledger)
        raise
    with _with_lock(root):
        ledger = _load_ledger(root)
        current = ledger["entries"][counter - 1]
        if current["nonce"] != nonce or current["status"] != "pending":
            raise ValueError("issuance identity changed during runner execution")
        _, proof_record = _read_regular(root, attempt_dir / "proof.json", label="attempt proof")
        current["status"] = "complete"
        current["proof_sha256"] = proof_record["sha256"]
        _atomic_json(root, root / "issuance-ledger.json", ledger)
    print(json.dumps(proof, sort_keys=True))
    return proof


def _verify_attempts(dependencies: CampaignDependencies) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = dependencies.campaign_root.absolute()
    _layout(dependencies)
    route = dependencies.current_route()
    with _with_lock(root):
        ledger = _load_ledger(root)
        entries = json.loads(json.dumps(ledger["entries"]))
    proofs: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("status") != "complete":
            raise ValueError("status is invalid: issuance is not complete")
        attempt_dir = root / str(entry["attempt_dir"])
        proof = _read_json(root, attempt_dir / "proof.json", label="attempt proof")
        _verify_seal(proof, "attempt")
        if any(proof.get(field) != entry.get(field) for field in ("counter", "nonce", "kind", "architecture", "dataset_id", "route_commit")):
            raise ValueError("attempt proof and ledger identity mismatch")
        if proof.get("route_commit") != route:
            raise ValueError("stale route commit in registered attempt")
        _, proof_record = _read_regular(root, attempt_dir / "proof.json", label="attempt proof")
        if proof_record["sha256"] != entry.get("proof_sha256"):
            raise ValueError("attempt proof artifact SHA mismatch")
        artifacts = proof.get("artifacts")
        if type(artifacts) is not list or len(artifacts) != 1:
            raise ValueError("attempt artifact registry is invalid")
        artifact_path = root / str(artifacts[0].get("path"))
        _, current = _read_regular(root, artifact_path, label="runner artifact")
        if current != artifacts[0]:
            raise ValueError("runner artifact identity/SHA mismatch")
        result = _read_json(root, artifact_path, label="runner artifact")
        if result != proof.get("runner_result"):
            raise ValueError("runner artifact content mismatch")
        proofs.append(proof)
    return ledger, proofs


def _decision_path(root: Path, name: str) -> Path:
    return root.absolute() / "decisions" / name


def _write_decision(root: Path, name: str, payload: Mapping[str, object]) -> dict[str, Any]:
    sealed = _seal(payload)
    _exclusive_json(root.absolute(), _decision_path(root, name), sealed)
    return sealed


def _load_decision(root: Path, name: str) -> dict[str, Any]:
    payload = _read_json(root.absolute(), _decision_path(root, name), label=name)
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
    decision = _write_decision(deps.campaign_root, f"replay-{args.architecture}.json", payload)
    print(json.dumps(decision, sort_keys=True))
    return 0


def _expected_replay(deps: CampaignDependencies, architecture: str) -> dict[str, Any]:
    ledger, proofs = _verify_attempts(deps)
    selected = [proof for proof in proofs if proof["kind"] == "validation" and proof["architecture"] == architecture]
    if {proof["dataset_id"] for proof in selected} != set(FROZEN_RECIPES) or len(selected) != 3:
        raise ValueError("replay requires exactly one registered proof per frozen dataset")
    selected.sort(key=lambda proof: proof["counter"])
    return {
        "schema_version": 2,
        "campaign_id": CAMPAIGN_ID,
        "cohort_sha256": FROZEN_COHORT_SHA256,
        "architecture": architecture,
        "architecture_fingerprint": architecture_fingerprint(architecture),
        "route_commit": deps.current_route(),
        "counters": [proof["counter"] for proof in selected],
        "nonces": [proof["nonce"] for proof in selected],
        "proof_sha256s": [proof["proof_sha256"] for proof in selected],
        "artifact_sha256s": [proof["artifacts"][0]["sha256"] for proof in selected],
        "rows": {proof["dataset_id"]: proof["runner_result"] for proof in selected},
    }


def _verified_replay(deps: CampaignDependencies, architecture: str) -> dict[str, Any]:
    replay = _load_decision(deps.campaign_root, f"replay-{architecture}.json")
    expected = _expected_replay(deps, architecture)
    unsigned = dict(replay)
    unsigned.pop("proof_sha256", None)
    if unsigned != expected:
        raise ValueError("registered replay was forged, resealed, or is stale")
    return replay


def _command_freeze(args: argparse.Namespace, deps: CampaignDependencies) -> int:
    replay = _verified_replay(deps, "a0v4")
    payload = _write_decision(deps.campaign_root, "a0v4-frozen.json", {
        "schema_version": 2,
        "campaign_id": CAMPAIGN_ID,
        "cohort_sha256": FROZEN_COHORT_SHA256,
        "route_commit": deps.current_route(),
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
    frozen = _load_decision(deps.campaign_root, "a0v4-frozen.json")
    replay = _verified_replay(deps, "a0v4")
    if frozen.get("replay_proof_sha256") != replay.get("proof_sha256") or frozen.get("rows") != replay.get("rows"):
        raise ValueError("frozen A0v4 proof chain mismatch")
    return frozen


def relative_gate(deltas: Mapping[str, Mapping[str, object]]) -> dict[str, Any]:
    return stable_graph_relative_gate(deltas)


def _command_relative(args: argparse.Namespace, deps: CampaignDependencies) -> int:
    derived = _derive_relative(deps)
    payload = _write_decision(deps.campaign_root, "relative-gate.json", derived)
    print(json.dumps(payload, sort_keys=True))
    return 0


def _derive_relative(deps: CampaignDependencies, *, nonce: str | None = None) -> dict[str, Any]:
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
        "schema_version": 2,
        "campaign_id": CAMPAIGN_ID,
        "cohort_sha256": FROZEN_COHORT_SHA256,
        "route_commit": deps.current_route(),
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
    proof = _load_decision(deps.campaign_root, "relative-gate.json")
    expected = _derive_relative(deps, nonce=proof.get("nonce") if isinstance(proof.get("nonce"), str) else None)
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
    payload = _write_decision(deps.campaign_root, "external-gate.json", derived)
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
        "schema_version": 2,
        "campaign_id": CAMPAIGN_ID,
        "cohort_sha256": FROZEN_COHORT_SHA256,
        "route_commit": deps.current_route(),
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
    proof = _load_decision(deps.campaign_root, "external-gate.json")
    audit_path = proof.get("comparator_audit_path")
    if type(audit_path) is not str:
        raise ValueError("external proof comparator audit path is invalid")
    audit = load_verified_comparator_audit(Path(audit_path))
    relative = _verified_relative(deps)
    candidate = _verified_replay(deps, "a2")
    derived = _derive_external(
        deps,
        relative=relative,
        candidate=candidate,
        audit=audit,
        audit_path=Path(audit_path),
        nonce=proof.get("nonce") if isinstance(proof.get("nonce"), str) else None,
    )
    unsigned = dict(proof)
    unsigned.pop("proof_sha256", None)
    if unsigned != derived:
        raise ValueError("external proof was forged or resealed")
    return proof


def _command_test(args: argparse.Namespace, deps: CampaignDependencies) -> int:
    if args.architecture != "a2":
        raise ValueError("test remains closed for every architecture except a2")
    root = deps.campaign_root.absolute()
    relative = _verified_relative(deps)
    external = _verified_external(deps)
    if relative.get("passed") is not True or external.get("passed") is not True or external.get("relative_proof_sha256") != relative.get("proof_sha256"):
        raise ValueError("test remains closed until registered gates pass")
    _verified_replay(deps, "a2")
    _layout(deps)
    pending_path = _decision_path(root, "test-once.json")
    nonce = secrets.token_hex(32)
    pending = _seal({
        "schema_version": 2,
        "campaign_id": CAMPAIGN_ID,
        "state": "pending",
        "nonce": nonce,
        "route_commit": deps.current_route(),
        "relative_proof_sha256": relative["proof_sha256"],
        "external_proof_sha256": external["proof_sha256"],
    })
    try:
        _exclusive_json(root, pending_path, pending)
    except FileExistsError as error:
        raise ValueError("test-once nonce has already been consumed") from error
    attempt_dir = root / "attempts" / "test-once"
    os.mkdir(attempt_dir, 0o700)
    argv = _runner_argv("test", "a2", attempt_dir, None)
    try:
        result = _validate_runner_result("test", "a2", None, deps.runner(argv, attempt_dir))
        state = "succeeded"
        error_text = None
    except BaseException as error:
        result = None
        state = "failed"
        error_text = f"{type(error).__name__}: {error}"
    finalized = _seal({
        "schema_version": 2,
        "campaign_id": CAMPAIGN_ID,
        "state": state,
        "nonce": nonce,
        "route_commit": deps.current_route(),
        "relative_proof_sha256": relative["proof_sha256"],
        "external_proof_sha256": external["proof_sha256"],
        "runner_argv": argv,
        "runner_result": result,
        "error": error_text,
    })
    _atomic_json(root, pending_path, finalized)
    if state == "failed":
        raise RuntimeError(error_text)
    print(json.dumps(finalized, sort_keys=True))
    return 0


def _command_status(args: argparse.Namespace, deps: CampaignDependencies) -> int:
    ledger, proofs = _verify_attempts(deps)
    decisions: dict[str, str] = {}
    for path in sorted((deps.campaign_root / "decisions").iterdir()):
        if path.name == "replay-a0v4.json":
            _verified_replay(deps, "a0v4")
        elif path.name == "replay-a2.json":
            _verified_replay(deps, "a2")
        elif path.name == "a0v4-frozen.json":
            _verified_frozen(deps)
        elif path.name == "relative-gate.json":
            _verified_relative(deps)
        elif path.name == "external-gate.json":
            _verified_external(deps)
        elif path.name == "test-once.json":
            payload = _read_json(deps.campaign_root, path, label=path.name)
            _verify_seal(payload, path.name)
            if payload.get("route_commit") != deps.current_route():
                raise ValueError("test-once record has a stale route")
            if payload.get("state") not in {"pending", "succeeded", "failed"}:
                raise ValueError("test-once record state is invalid")
            relative = _verified_relative(deps)
            external = _verified_external(deps)
            if payload.get("relative_proof_sha256") != relative.get("proof_sha256") or payload.get("external_proof_sha256") != external.get("proof_sha256"):
                raise ValueError("test-once proof chain mismatch")
        else:
            raise ValueError(f"status found an unregistered decision artifact: {path.name}")
        _, record = _read_regular(deps.campaign_root, path, label=path.name)
        decisions[path.name] = str(record["sha256"])
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
