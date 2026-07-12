from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from models.unified_v2_spec import UnifiedArchitectureSpec
from scripts.run_unified_validation import RECIPES
from scripts.unified_baseline_audit import audit_baseline_rows
from scripts.unified_dataset_audit import canonical_sha256
from scripts.unified_validation_controller import (
    CAMPAIGN_ID,
    authorize_next,
    initialize_controller,
    run_registered_pair,
)


def _verify_hash(payload: Mapping[str, Any], field: str, label: str) -> str:
    stored = payload.get(field)
    unhashed = dict(payload)
    unhashed.pop(field, None)
    if not isinstance(stored, str) or stored != canonical_sha256(unhashed):
        raise ValueError(f"{label} canonical SHA-256 mismatch")
    return stored


def build_exploration_payloads(
    dataset_audit: Mapping[str, Any],
    baseline_audit: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_audit_sha256 = _verify_hash(
        dataset_audit, "audit_sha256", "dataset audit"
    )
    datasets = dataset_audit.get("datasets")
    eligible = dataset_audit.get("eligible_dataset_ids")
    if not isinstance(datasets, Mapping) or not isinstance(eligible, list):
        raise ValueError("dataset audit has no eligible registry")
    dataset_ids = [
        dataset_id
        for dataset_id in eligible
        if isinstance(dataset_id, str)
        and isinstance(datasets.get(dataset_id), Mapping)
        and datasets[dataset_id].get("eligible") is True
    ]
    if len(dataset_ids) < 3:
        raise ValueError("fewer than three eligible auditable datasets")
    if len(dataset_ids) != len(eligible):
        raise ValueError("dataset eligible registry does not match audited records")

    _verify_hash(baseline_audit, "audit_sha256", "baseline audit")
    accepted = baseline_audit.get("accepted_rows")
    strongest = baseline_audit.get("strongest_comparators")
    if not isinstance(accepted, list) or not isinstance(strongest, Mapping):
        raise ValueError("baseline audit has no accepted strongest registry")
    verified = audit_baseline_rows(accepted, dataset_audit)
    if verified["rejected_count"] or verified["strongest_comparators"] != strongest:
        raise ValueError("baseline accepted provenance or strongest registry mismatch")

    references: dict[str, Any] = {}
    guards: list[dict[str, Any]] = []
    record_hashes: dict[str, str] = {}
    for dataset_id in dataset_ids:
        record = datasets[dataset_id]
        assert isinstance(record, Mapping)
        record_hashes[dataset_id] = _verify_hash(
            record, "audit_sha256", f"dataset record {dataset_id}"
        )
        try:
            standard_auc = strongest[dataset_id]["standard"]["auc"]
            holdout_auc = strongest[dataset_id]["holdout"]["auc"]
            zero_auc = strongest[dataset_id]["holdout"]["zero_auc"]
            selected = {
                "standard_overall_auc": standard_auc,
                "holdout_overall_auc": holdout_auc,
                "zero_auc": zero_auc,
            }
            values = {name: float(row["value"]) for name, row in selected.items()}
            sources = {
                name: str(row["source_path"]) for name, row in selected.items()
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"baseline-incomplete: {dataset_id}") from error
        references[dataset_id] = selected
        guards.append({
            "dataset_id": dataset_id,
            "standard_overall_auc": values["standard_overall_auc"],
            "holdout_overall_auc": values["holdout_overall_auc"],
            "zero_auc": values["zero_auc"],
            "comparator_sources": sources,
        })

    registry: dict[str, Any] = {
        "schema_version": 1,
        "dataset_ids": dataset_ids,
        "audit_sha256": dataset_audit_sha256,
        "dataset_audit_sha256": record_hashes,
        "b0_validation_references": references,
    }
    registry["cohort_sha256"] = canonical_sha256(registry)
    for row in guards:
        row["cohort_sha256"] = registry["cohort_sha256"]
    return registry, {"rows": guards}


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _exclusive_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _init(args: argparse.Namespace) -> None:
    if args.campaign_id != CAMPAIGN_ID or args.architecture != "a0":
        raise ValueError("A0 exploration requires the registered campaign and architecture")
    dataset_audit = _read_json(args.dataset_audit)
    baseline_audit = _read_json(args.baseline_audit)
    if not isinstance(dataset_audit, Mapping) or not isinstance(baseline_audit, Mapping):
        raise ValueError("audits must be JSON objects")
    registry, guards = build_exploration_payloads(dataset_audit, baseline_audit)
    state_path = args.state.resolve()
    stem = state_path.with_suffix("")
    registry_path = stem.parent / f"{stem.name}-provisional.json"
    guards_path = stem.parent / f"{stem.name}-external-guards.json"
    manifest_path = stem.parent / f"{stem.name}-manifest.json"
    controller_dir = stem.parent / f"{stem.name}-controller"
    artifact_root = state_path.parent.parent / "a0"
    rows_path = stem.parent / "a0-validation-rows.json"
    manifest = UnifiedArchitectureSpec(completion="prior").manifest()
    _exclusive_json(registry_path, registry)
    _exclusive_json(guards_path, guards)
    _exclusive_json(manifest_path, manifest)
    initialize_controller(
        state_dir=controller_dir,
        repo_root=args.repo_root.resolve(),
        cohort_path=registry_path,
        manifest_path=manifest_path,
        architecture="a0",
        baseline_rows_path=guards_path,
        data_root=Path(str(dataset_audit["root"])),
        artifact_root=artifact_root,
        exploration=True,
    )
    _exclusive_json(state_path, {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "architecture": "a0",
        "controller_state_dir": str(controller_dir),
        "repo_root": str(args.repo_root.resolve()),
        "provisional_registry": str(registry_path),
        "provisional_registry_sha256": registry["cohort_sha256"],
        "dataset_audit": str(args.dataset_audit.resolve()),
        "baseline_audit": str(args.baseline_audit.resolve()),
        "rows_output": str(rows_path),
    })


def _validated_ranked_proofs(
    controller_dir: Path,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]], list[dict[str, Any]]]:
    state = _read_json(controller_dir / "state.json")
    if not isinstance(state, dict) or not state.get("complete"):
        raise ValueError("exploration controller is not complete")
    if state.get("mode") != "a0_exploration" or state.get("active_pair") is not None:
        raise ValueError("controller is not in finalizable A0 exploration state")
    dataset_ids = state.get("dataset_ids")
    issuance_counter = state.get("issuance_counter")
    if not isinstance(dataset_ids, list) or type(issuance_counter) is not int:
        raise ValueError("exploration controller registry is invalid")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    proof_registry: list[dict[str, Any]] = []
    for path in sorted((controller_dir / "proofs").glob("*.json")):
        proof = _read_json(path)
        if not isinstance(proof, Mapping):
            raise ValueError(f"exploration proof is not an object: {path}")
        dataset_id = proof.get("dataset_id")
        counter = proof.get("counter")
        if (
            dataset_id not in dataset_ids
            or type(counter) is not int
            or path.name != f"{counter:06d}-{dataset_id}.json"
        ):
            raise ValueError(f"exploration proof filename/counter mismatch: {path}")
        for field in ("controller_id", "route_commit", "baseline_sha256"):
            if proof.get(field) != state.get(field):
                raise ValueError(f"exploration proof {field} mismatch: {path}")
        split_proofs = proof.get("split_proofs")
        if not isinstance(split_proofs, Mapping):
            raise ValueError(f"exploration split proofs are invalid: {path}")
        standard = split_proofs.get("standard")
        holdout = split_proofs.get("holdout")
        if not isinstance(standard, Mapping) or not isinstance(holdout, Mapping):
            raise ValueError(f"exploration split proof pair is incomplete: {path}")
        recipe_index = standard.get("recipe_index")
        numerical_recipe = standard.get("numerical_recipe")
        if (
            type(recipe_index) is not int
            or holdout.get("recipe_index") != recipe_index
            or holdout.get("numerical_recipe") != numerical_recipe
            or recipe_index < 0
            or recipe_index >= len(RECIPES[str(dataset_id)])
            or numerical_recipe != RECIPES[str(dataset_id)][recipe_index].__dict__
        ):
            raise ValueError(f"exploration proof recipe mismatch: {path}")
        proof_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        proof_registry.append({
            "counter": counter,
            "dataset_id": dataset_id,
            "proof_sha256": proof_sha256,
        })
        grouped.setdefault(str(dataset_id), []).append(proof)
    counters = [entry["counter"] for entry in proof_registry]
    if counters != list(range(1, issuance_counter + 1)):
        raise ValueError("exploration proof counters are not complete and contiguous")
    proof_registry_sha256 = canonical_sha256(proof_registry)
    registered_sha256 = state.get("finalized_proof_registry_sha256")
    if registered_sha256 is not None and registered_sha256 != proof_registry_sha256:
        raise ValueError("finalized exploration proof registry SHA-256 mismatch")
    selected_proofs: dict[str, Mapping[str, Any]] = {}
    for dataset_id in dataset_ids:
        proofs = grouped.get(dataset_id, [])
        if not proofs:
            raise ValueError(f"missing exploration proof: {dataset_id}")
        selected_proofs[dataset_id] = max(
            proofs,
            key=lambda proof: (
                min(
                    float(proof["deltas"]["standard_overall_auc"]),
                    float(proof["deltas"]["holdout_overall_auc"]),
                ),
                float(proof["deltas"]["zero_auc"]),
                -int(proof["counter"]),
            ),
        )
    return state, selected_proofs, proof_registry


def _rows_from_selected(
    state: Mapping[str, Any],
    selected_proofs: Mapping[str, Mapping[str, Any]],
    proof_registry: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    proof_hashes = {
        int(entry["counter"]): str(entry["proof_sha256"])
        for entry in proof_registry
    }
    rows: list[dict[str, Any]] = []
    for dataset_id in state["dataset_ids"]:
        selected = selected_proofs[dataset_id]
        failed_attempt_count = sum(
            entry["dataset_id"] == dataset_id for entry in proof_registry
        ) - 1
        standard = selected["split_proofs"]["standard"]
        holdout = selected["split_proofs"]["holdout"]
        rows.append({
            "dataset_id": dataset_id,
            "a0_standard_auc": float(standard["metrics"]["overall_auc"]),
            "a0_holdout_auc": float(holdout["metrics"]["overall_auc"]),
            "a0_zero_auc": float(holdout["metrics"]["zero_auc"]),
            "a0_ordinary_doa": float(standard["metrics"]["ordinary_doa"]),
            "a0_weighted_doa": float(standard["metrics"]["weighted_doa"]),
            "a0_fingerprint": state["architecture_fingerprint"],
            "recipe_index": int(standard["recipe_index"]),
            "numerical_recipe": standard["numerical_recipe"],
            "recipe_sha256": canonical_sha256(standard["numerical_recipe"]),
            "failed_attempt_count": failed_attempt_count,
            "proof_counter": int(selected["counter"]),
            "proof_sha256": proof_hashes[int(selected["counter"])],
            "provisional_registry_sha256": state["cohort_sha256"],
        })
    return rows


def finalize_exploration(controller_dir: Path) -> list[dict[str, Any]]:
    state, selected_proofs, proof_registry = _validated_ranked_proofs(controller_dir)
    rows = _rows_from_selected(state, selected_proofs, proof_registry)
    selected_recipes: dict[str, dict[str, Any]] = {}
    for row in rows:
        selected_recipes[str(row["dataset_id"])] = {
            "recipe_index": int(row["recipe_index"]),
            "numerical_recipe": row["numerical_recipe"],
            "recipe_sha256": str(row["recipe_sha256"]),
            "proof_counter": int(row["proof_counter"]),
            "proof_sha256": str(row["proof_sha256"]),
        }
    state["selected_recipes"] = selected_recipes
    state["finalized_proof_registry"] = proof_registry
    state["finalized_proof_registry_sha256"] = canonical_sha256(proof_registry)
    _atomic_json(controller_dir / "state.json", state)
    return rows


def _assemble_rows(controller_dir: Path) -> list[dict[str, Any]]:
    return finalize_exploration(controller_dir)


def _run_validation(args: argparse.Namespace) -> None:
    wrapper = _read_json(args.state.resolve())
    if not isinstance(wrapper, Mapping):
        raise ValueError("exploration state must be a JSON object")
    controller_dir = Path(str(wrapper["controller_state_dir"]))
    repo_root = Path(str(wrapper["repo_root"]))
    tokens = args.state.resolve().parent / "tokens"
    tokens.mkdir(parents=True, exist_ok=True)
    while True:
        state = _read_json(controller_dir / "state.json")
        if state.get("complete"):
            break
        counter = int(state.get("issuance_counter", 0)) + 1
        token_path = tokens / f"a0-{counter:06d}.json"
        authorize_next(
            state_dir=controller_dir,
            repo_root=repo_root,
            output_path=token_path,
        )
        run_registered_pair(state_dir=controller_dir, repo_root=repo_root)
    _exclusive_json(
        Path(str(wrapper["rows_output"])),
        {"rows": finalize_exploration(controller_dir)},
    )


def _finalize(args: argparse.Namespace) -> None:
    wrapper = _read_json(args.state.resolve())
    if not isinstance(wrapper, Mapping):
        raise ValueError("exploration state must be a JSON object")
    rows_payload = {
        "rows": finalize_exploration(Path(str(wrapper["controller_state_dir"])))
    }
    rows_path = Path(str(wrapper["rows_output"]))
    if rows_path.exists():
        existing = _read_json(rows_path)
        legacy_payload = json.loads(json.dumps(rows_payload))
        for row in legacy_payload["rows"]:
            row.pop("proof_sha256")
        if existing not in (rows_payload, legacy_payload):
            raise ValueError("existing A0 rows do not match replayed immutable proofs")
        if existing != rows_payload:
            _atomic_json(rows_path, rows_payload)
    else:
        _exclusive_json(rows_path, rows_payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orchestrate audited A0 exploration.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init")
    init.add_argument("--campaign-id", required=True)
    init.add_argument("--architecture", choices=("a0",), required=True)
    init.add_argument("--dataset-audit", type=Path, required=True)
    init.add_argument("--baseline-audit", type=Path, required=True)
    init.add_argument("--state", type=Path, required=True)
    init.add_argument("--repo-root", type=Path, default=Path.cwd())
    init.set_defaults(handler=_init)
    run = subparsers.add_parser("run-validation")
    run.add_argument("--state", type=Path, required=True)
    run.add_argument("--parallel-gpus", action="store_true")
    run.set_defaults(handler=_run_validation)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--state", type=Path, required=True)
    finalize.set_defaults(handler=_finalize)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
