from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import secrets
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.unified_v2_spec import UnifiedArchitectureSpec
from scripts.unified_campaign import stable_graph_relative_gate
from scripts.unified_dataset_audit import canonical_sha256
from scripts.unified_validation_controller import (
    _atomic_json,
    _exclusive_json,
    _fsync_directory,
    _load_json,
    _route_head,
    _sha256_file,
)


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
SPLITS = frozenset({"standard", "holdout"})
NONCE_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
DEFAULT_CAMPAIGN_ROOT = Path(
    "/home/xph/jwc/research/local_data/decoupled_cd_codex_routes"
) / CAMPAIGN_ID


def architecture_spec(architecture: str) -> UnifiedArchitectureSpec:
    try:
        completion, _ = ARCHITECTURES[architecture]
    except KeyError as error:
        raise ValueError(f"unknown stable graph architecture: {architecture}") from error
    return UnifiedArchitectureSpec(completion=completion)


def architecture_fingerprint(architecture: str) -> str:
    return architecture_spec(architecture).fingerprint()


def _validated_nonce(value: object, *, label: str) -> str:
    if not isinstance(value, str) or NONCE_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase 64-hex")
    return value


def load_verified_comparator_audit(path: Path) -> dict[str, Any]:
    audit = _load_json(path, label="comparator audit")
    registered = audit.get("audit_sha256")
    unsigned = dict(audit)
    unsigned.pop("audit_sha256", None)
    if registered != canonical_sha256(unsigned):
        raise ValueError("comparator audit canonical SHA-256 mismatch")
    if registered != FROZEN_COMPARATOR_AUDIT_SHA256:
        raise ValueError("comparator audit is not the frozen campaign audit")
    strongest = audit.get("strongest_comparators")
    if not isinstance(strongest, Mapping) or not set(FROZEN_RECIPES).issubset(
        strongest
    ):
        raise ValueError("comparator audit strongest registry is incomplete")
    return audit


def _seal(payload: Mapping[str, object]) -> dict[str, Any]:
    result = dict(payload)
    if "proof_sha256" in result:
        raise ValueError("proof payload already contains proof_sha256")
    result["proof_sha256"] = canonical_sha256(result)
    return result


def _verify_seal(payload: Mapping[str, object], *, label: str) -> None:
    proof_sha256 = payload.get("proof_sha256")
    if not isinstance(proof_sha256, str) or NONCE_PATTERN.fullmatch(proof_sha256) is None:
        raise ValueError(f"{label} proof SHA-256 is invalid")
    unsigned = dict(payload)
    unsigned.pop("proof_sha256")
    if canonical_sha256(unsigned) != proof_sha256:
        raise ValueError(f"{label} proof SHA-256 mismatch")


def issue_attempt(
    *,
    architecture: str,
    dataset: str,
    split: str,
    route_commit: str | None = None,
    counter: int | None = None,
    nonce: str | None = None,
    test_authorization: Mapping[str, object] | None = None,
    campaign_root: Path | None = None,
) -> dict[str, Any]:
    """Issue an immutable attempt description; execution stays in the trusted runner."""
    architecture_spec(architecture)
    if dataset not in FROZEN_RECIPES:
        raise ValueError(f"dataset is outside the frozen cohort: {dataset}")
    if split == "test":
        if test_authorization is None or campaign_root is None:
            raise ValueError("test remains closed without relative and external nonces")
        _verify_test_authorization(test_authorization)
        _require_registered_gate(
            campaign_root,
            "test-once-authorization.json",
            test_authorization,
        )
    elif split not in SPLITS:
        raise ValueError(f"unknown validation split: {split}")
    if route_commit is not None and COMMIT_PATTERN.fullmatch(route_commit) is None:
        raise ValueError("route commit must be lowercase 40-hex")
    if counter is not None and (type(counter) is not int or counter < 1):
        raise ValueError("proof counter must be a positive integer")
    capability_nonce = nonce or secrets.token_hex(32)
    _validated_nonce(capability_nonce, label="attempt nonce")
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "cohort_sha256": FROZEN_COHORT_SHA256,
        "architecture": architecture,
        "architecture_fingerprint": architecture_fingerprint(architecture),
        "dataset_id": dataset,
        "split_id": split,
        "recipe_index": FROZEN_RECIPES[dataset],
        "route_commit": route_commit,
        "counter": counter,
        "nonce": capability_nonce,
    }


def _artifact_record(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"artifact is not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def build_validation_proof(
    *,
    architecture: str,
    dataset: str,
    route_commit: str,
    counter: int,
    artifact_path: Path,
    nonce: str | None = None,
    metrics: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    capability = issue_attempt(
        architecture=architecture,
        dataset=dataset,
        split="standard",
        route_commit=route_commit,
        counter=counter,
        nonce=nonce,
    )
    proof = {
        key: capability[key]
        for key in (
            "schema_version",
            "campaign_id",
            "cohort_sha256",
            "architecture",
            "architecture_fingerprint",
            "dataset_id",
            "recipe_index",
            "route_commit",
            "counter",
            "nonce",
        )
    }
    proof["artifact"] = _artifact_record(artifact_path)
    if metrics is not None:
        proof["metrics"] = _validated_metrics(metrics)
    return _seal(proof)


def _validated_metrics(value: Mapping[str, object]) -> dict[str, float]:
    required = {"standard", "holdout", "zero", "ordinary_doa", "weighted_doa"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("validation proof metrics are invalid")
    result: dict[str, float] = {}
    for field in sorted(required):
        metric = value[field]
        if type(metric) is not float or not math.isfinite(metric) or not 0.0 <= metric <= 1.0:
            raise ValueError(f"validation proof metric is invalid: {field}")
        result[field] = metric
    return result


def verify_validation_proof(
    proof: Mapping[str, object],
    *,
    architecture: str,
    dataset: str,
    route_commit: str,
    counter: int,
    artifact_path: Path,
) -> dict[str, Any]:
    if not isinstance(proof, Mapping):
        raise ValueError("validation proof must be an object")
    _verify_seal(proof, label="validation")
    expected_fields = {
        "schema_version",
        "campaign_id",
        "cohort_sha256",
        "architecture",
        "architecture_fingerprint",
        "dataset_id",
        "recipe_index",
        "route_commit",
        "counter",
        "nonce",
        "artifact",
        "proof_sha256",
    }
    if "metrics" in proof:
        expected_fields.add("metrics")
    if set(proof) != expected_fields:
        raise ValueError("validation proof field set mismatch")
    if proof.get("schema_version") != 1 or proof.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("validation proof campaign identity mismatch")
    if proof.get("cohort_sha256") != FROZEN_COHORT_SHA256:
        raise ValueError("cohort sha256 mismatch")
    if proof.get("architecture") != architecture:
        raise ValueError("architecture mismatch")
    if proof.get("architecture_fingerprint") != architecture_fingerprint(architecture):
        raise ValueError("architecture fingerprint mismatch")
    if proof.get("dataset_id") != dataset:
        raise ValueError("dataset identity mismatch")
    if proof.get("recipe_index") != FROZEN_RECIPES.get(dataset):
        raise ValueError("frozen recipe mismatch")
    if proof.get("route_commit") != route_commit:
        raise ValueError("route commit mismatch")
    if proof.get("counter") != counter:
        raise ValueError("proof counter mismatch")
    _validated_nonce(proof.get("nonce"), label="proof nonce")
    expected_artifact = _artifact_record(artifact_path)
    artifact = proof.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("artifact record mismatch")
    if artifact.get("sha256") != expected_artifact["sha256"]:
        raise ValueError("artifact SHA-256 mismatch")
    if dict(artifact) != expected_artifact:
        raise ValueError("artifact identity mismatch")
    if "metrics" in proof:
        _validated_metrics(proof["metrics"])  # type: ignore[arg-type]
    return dict(proof)


def relative_gate(
    deltas: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    return stable_graph_relative_gate(deltas)


def issue_relative_gate(
    deltas: Mapping[str, Mapping[str, object]], *, nonce: str | None = None
) -> dict[str, Any]:
    decision = relative_gate(deltas)
    gate_nonce: str | None = None
    if decision["passed"]:
        gate_nonce = nonce or secrets.token_hex(32)
        _validated_nonce(gate_nonce, label="relative gate nonce")
    payload = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "cohort_sha256": FROZEN_COHORT_SHA256,
        **decision,
        "nonce": gate_nonce,
    }
    return _seal(payload)


def _verify_gate(
    proof: Mapping[str, object], *, label: str, require_pass: bool
) -> None:
    if not isinstance(proof, Mapping):
        raise ValueError(f"{label} gate proof must be an object")
    _verify_seal(proof, label=f"{label} gate")
    if (
        proof.get("schema_version") != 1
        or proof.get("campaign_id") != CAMPAIGN_ID
        or proof.get("cohort_sha256") != FROZEN_COHORT_SHA256
    ):
        raise ValueError(f"{label} gate identity mismatch")
    if type(proof.get("passed")) is not bool:
        raise ValueError(f"{label} gate pass state is invalid")
    if require_pass and not proof["passed"]:
        raise ValueError(f"{label} gate has not passed")
    if proof["passed"]:
        _validated_nonce(proof.get("nonce"), label=f"{label} gate nonce")
    elif proof.get("nonce") is not None:
        raise ValueError(f"failed {label} gate must not issue a nonce")


def external_gate(
    relative_proof: Mapping[str, object],
    comparator_deltas: Mapping[str, Mapping[str, object]],
    *,
    nonce: str | None = None,
    comparator_audit_sha256: str | None = None,
) -> dict[str, Any]:
    _verify_gate(relative_proof, label="relative", require_pass=True)
    if not isinstance(comparator_deltas, Mapping) or set(comparator_deltas) != set(FROZEN_RECIPES):
        raise ValueError("external deltas must contain the exact frozen dataset cohort")
    normalized: dict[str, dict[str, float]] = {}
    for dataset in sorted(FROZEN_RECIPES):
        row = comparator_deltas[dataset]
        if not isinstance(row, Mapping) or set(row) != {"standard", "holdout", "zero"}:
            raise ValueError(f"external delta fields are invalid: {dataset}")
        normalized[dataset] = {}
        for field in ("standard", "holdout", "zero"):
            value = row[field]
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"external delta must be a finite float: {dataset}.{field}")
            normalized[dataset][field] = value
    passed = all(
        row["standard"] >= 0.0
        and row["holdout"] >= 0.0
        and row["zero"] > 0.0
        for row in normalized.values()
    )
    gate_nonce: str | None = None
    if passed:
        gate_nonce = nonce or secrets.token_hex(32)
        _validated_nonce(gate_nonce, label="external gate nonce")
    if comparator_audit_sha256 is not None:
        _validated_nonce(comparator_audit_sha256, label="comparator audit SHA-256")
    return _seal(
        {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "cohort_sha256": FROZEN_COHORT_SHA256,
            "passed": passed,
            "relative_nonce": relative_proof["nonce"],
            "comparator_audit_sha256": comparator_audit_sha256,
            "deltas": normalized,
            "nonce": gate_nonce,
        }
    )


def _verify_test_authorization(proof: Mapping[str, object]) -> None:
    _verify_gate(proof, label="test authorization", require_pass=True)
    for field in ("relative_nonce", "external_nonce", "nonce"):
        _validated_nonce(proof.get(field), label=field.replace("_", " "))


def authorize_test_once(
    relative_proof: Mapping[str, object],
    external_proof: Mapping[str, object],
    *,
    campaign_root: Path | None = None,
    nonce: str | None = None,
) -> dict[str, Any]:
    try:
        _verify_gate(relative_proof, label="relative", require_pass=True)
        _verify_gate(external_proof, label="external", require_pass=True)
    except ValueError as error:
        raise ValueError("test remains closed until relative and external gates pass") from error
    if external_proof.get("relative_nonce") != relative_proof.get("nonce"):
        raise ValueError("test remains closed: external relative nonce mismatch")
    if campaign_root is None:
        raise ValueError("registered gate proofs are required for test authorization")
    _require_registered_gate(campaign_root, "relative-gate.json", relative_proof)
    _require_registered_gate(campaign_root, "external-gate.json", external_proof)
    authorization_nonce = nonce or secrets.token_hex(32)
    _validated_nonce(authorization_nonce, label="test-once nonce")
    authorization = _seal(
        {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "cohort_sha256": FROZEN_COHORT_SHA256,
            "passed": True,
            "relative_nonce": relative_proof["nonce"],
            "external_nonce": external_proof["nonce"],
            "nonce": authorization_nonce,
        }
    )
    decision_dir = campaign_root.resolve() / "decisions"
    decision_dir.mkdir(parents=True, exist_ok=True)
    path = decision_dir / "test-once-authorization.json"
    if path.exists():
        raise ValueError("test-once authorization has already been issued")
    _exclusive_json(path, authorization)
    _fsync_directory(decision_dir)
    return authorization


def _campaign_path(root: Path, name: str) -> Path:
    return root.resolve() / "decisions" / name


def _write_gate(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _exclusive_json(path, payload)


def register_gate_proof(
    campaign_root: Path,
    name: str,
    proof: Mapping[str, object],
) -> None:
    if name not in {"relative-gate.json", "external-gate.json"}:
        raise ValueError("gate registry name is invalid")
    label = name.removesuffix("-gate.json")
    _verify_gate(proof, label=label, require_pass=False)
    _write_gate(_campaign_path(campaign_root, name), proof)


def _require_registered_gate(
    campaign_root: Path,
    name: str,
    proof: Mapping[str, object],
) -> None:
    path = _campaign_path(campaign_root, name)
    try:
        registered = _load_json(path, label=f"registered {name}")
    except ValueError as error:
        raise ValueError(f"registered {name} is required") from error
    if registered != dict(proof):
        raise ValueError(f"registered {name} mismatch")


def _load_gate(path: Path, *, label: str) -> dict[str, Any]:
    return _load_json(path, label=label)


def _command_status(args: argparse.Namespace) -> int:
    root = args.campaign_root.resolve()
    decisions: dict[str, object] = {}
    for name in (
        "a0v4-frozen.json",
        "relative-gate.json",
        "external-gate.json",
        "test-once-authorization.json",
        "test-once-consumed.json",
    ):
        path = root / "decisions" / name
        decisions[name] = _sha256_file(path) if path.is_file() else None
    print(json.dumps({"campaign_id": CAMPAIGN_ID, "decisions": decisions}, sort_keys=True))
    return 0


def _command_relative_gate(args: argparse.Namespace) -> int:
    deltas_path = args.deltas or (
        args.campaign_root.resolve() / "replay" / "relative-deltas.json"
    )
    deltas = _load_json(deltas_path, label="relative deltas")
    proof = issue_relative_gate(deltas)
    _write_gate(_campaign_path(args.campaign_root, "relative-gate.json"), proof)
    print(json.dumps(proof, sort_keys=True))
    return 0


def _command_external_gate(args: argparse.Namespace) -> int:
    relative = _load_gate(
        _campaign_path(args.campaign_root, "relative-gate.json"),
        label="relative gate",
    )
    deltas_path = args.deltas or (
        args.campaign_root.resolve() / "replay" / "external-deltas.json"
    )
    deltas = _load_json(deltas_path, label="external deltas")
    audit = load_verified_comparator_audit(args.comparator_audit)
    audit_sha = str(audit["audit_sha256"])
    proof = external_gate(
        relative,
        deltas,
        comparator_audit_sha256=audit_sha,
    )
    _write_gate(_campaign_path(args.campaign_root, "external-gate.json"), proof)
    print(json.dumps(proof, sort_keys=True))
    return 0


def _command_run_test_once(args: argparse.Namespace) -> int:
    if args.architecture != "a2":
        raise ValueError("test remains closed for every architecture except a2")
    relative = _load_gate(
        _campaign_path(args.campaign_root, "relative-gate.json"),
        label="relative gate",
    )
    external = _load_gate(
        _campaign_path(args.campaign_root, "external-gate.json"),
        label="external gate",
    )
    authorization = authorize_test_once(
        relative, external, campaign_root=args.campaign_root
    )
    print(json.dumps(authorization, sort_keys=True))
    return 0


def _command_issue(args: argparse.Namespace) -> int:
    route_commit = _route_head(PROJECT_ROOT)
    split = "standard" if args.command == "run-validation" else "holdout"
    dataset = getattr(args, "dataset", "ASSIST17")
    capability = issue_attempt(
        architecture=args.architecture,
        dataset=dataset,
        split=split,
        route_commit=route_commit,
        counter=1,
    )
    print(json.dumps(capability, sort_keys=True))
    return 0


def _command_replay(args: argparse.Namespace) -> int:
    route_commit = _route_head(PROJECT_ROOT)
    replayed: list[str] = []
    for counter, path in enumerate(args.proof, 1):
        proof = _load_json(path, label="validation proof")
        artifact = proof.get("artifact")
        if not isinstance(artifact, Mapping) or not isinstance(artifact.get("path"), str):
            raise ValueError("validation proof artifact path is invalid")
        verify_validation_proof(
            proof,
            architecture=args.architecture,
            dataset=str(proof.get("dataset_id")),
            route_commit=route_commit,
            counter=counter,
            artifact_path=Path(str(artifact["path"])),
        )
        replayed.append(str(proof["proof_sha256"]))
    print(json.dumps({"campaign_id": CAMPAIGN_ID, "replayed": replayed}, sort_keys=True))
    return 0


def _command_freeze(args: argparse.Namespace) -> int:
    if len(args.proof) != len(FROZEN_RECIPES):
        raise ValueError("A0v4 freeze requires exactly three replayed proofs")
    payload = _seal(
        {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "cohort_sha256": FROZEN_COHORT_SHA256,
            "architecture": "a0v4",
            "proof_sha256s": [_sha256_file(path) for path in args.proof],
        }
    )
    _write_gate(_campaign_path(args.campaign_root, "a0v4-frozen.json"), payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate the stable graph validation campaign.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def campaign_argument(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT
        )

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument("--epochs", type=int, default=1)
    campaign_argument(smoke)
    smoke.set_defaults(handler=_command_issue)

    validation = subparsers.add_parser("run-validation")
    validation.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    validation.add_argument("--dataset", choices=FROZEN_RECIPES, required=True)
    validation.add_argument("--seed", type=int, default=42)
    validation.add_argument("--split-seed", type=int, default=2024)
    campaign_argument(validation)
    validation.set_defaults(handler=_command_issue)

    replay = subparsers.add_parser("replay")
    replay.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    replay.add_argument("--proof", type=Path, action="append", default=[])
    campaign_argument(replay)
    replay.set_defaults(handler=_command_replay)

    freeze = subparsers.add_parser("freeze-a0v4")
    freeze.add_argument("--proof", type=Path, action="append", default=[])
    campaign_argument(freeze)
    freeze.set_defaults(handler=_command_freeze)

    relative = subparsers.add_parser("relative-gate")
    relative.add_argument("--deltas", type=Path)
    campaign_argument(relative)
    relative.set_defaults(handler=_command_relative_gate)

    external = subparsers.add_parser("external-gate")
    external.add_argument("--comparator-audit", type=Path, required=True)
    external.add_argument("--deltas", type=Path)
    campaign_argument(external)
    external.set_defaults(handler=_command_external_gate)

    test_once = subparsers.add_parser("run-test-once")
    test_once.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    campaign_argument(test_once)
    test_once.set_defaults(handler=_command_run_test_once)

    status = subparsers.add_parser("status")
    campaign_argument(status)
    status.set_defaults(handler=_command_status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
