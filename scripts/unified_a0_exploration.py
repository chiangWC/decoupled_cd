from __future__ import annotations

import argparse
import json
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


def _assemble_rows(controller_dir: Path) -> list[dict[str, Any]]:
    state = _read_json(controller_dir / "state.json")
    if not isinstance(state, Mapping) or not state.get("complete"):
        raise ValueError("exploration controller is not complete")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for path in sorted((controller_dir / "proofs").glob("*.json")):
        proof = _read_json(path)
        if isinstance(proof, Mapping):
            grouped.setdefault(str(proof["dataset_id"]), []).append(proof)
    rows: list[dict[str, Any]] = []
    for dataset_id in state["dataset_ids"]:
        proofs = grouped.get(dataset_id, [])
        if not proofs:
            raise ValueError(f"missing exploration proof: {dataset_id}")
        selected = max(
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
            "failed_attempt_count": len(proofs) - 1,
            "proof_counter": int(selected["counter"]),
            "provisional_registry_sha256": state["cohort_sha256"],
        })
    return rows


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
    _exclusive_json(Path(str(wrapper["rows_output"])), {"rows": _assemble_rows(controller_dir)})


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
