from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plugin_campaign import compute_frozen_config_id, sha256_file
from scripts.select_plugin_checkpoint import (
    SelectionError,
    _finite_float,
    index_candidates,
    load_candidate_manifest,
    load_selection,
    load_validation_doa_rows,
    validate_candidate_artifacts,
    validate_doa_binding,
    validate_manifest_protocol,
)
from scripts.select_standard_checkpoint import (
    _require_recipe,
    standard_selection_payload,
)


@dataclass(frozen=True)
class JointCandidate:
    recipe_id: str
    standard: dict[str, Any]
    holdout: dict[str, Any]
    holdout_doa: dict[str, Any]
    standard_auc_delta: float
    holdout_auc_delta: float
    holdout_weighted_doa_delta: float
    holdout_doa_delta: float

    @property
    def min_auc_delta(self) -> float:
        return min(self.standard_auc_delta, self.holdout_auc_delta)


def _atomic_write_jsons(payloads: dict[Path, dict[str, Any]]) -> None:
    if not payloads:
        return
    parents = {path.parent for path in payloads}
    for parent in parents:
        parent.mkdir(parents=True, exist_ok=True)
    existing = [path for path in payloads if path.exists()]
    if existing:
        raise SelectionError(f"selection output already exists: {existing[0]}")
    temporary_paths: dict[Path, Path] = {}
    try:
        for destination, payload in payloads.items():
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
                temporary_paths[destination] = Path(handle.name)
        for destination, temporary_path in temporary_paths.items():
            os.replace(temporary_path, destination)
    finally:
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)


def _input_hashes(
    *,
    standard_manifest_path: Path,
    holdout_manifest_path: Path,
    holdout_doa_csv_path: Path,
    standard_baseline_path: Path,
    holdout_baseline_path: Path,
) -> dict[str, str]:
    return {
        "standard_manifest": sha256_file(standard_manifest_path),
        "holdout_manifest": sha256_file(holdout_manifest_path),
        "holdout_doa_csv": sha256_file(holdout_doa_csv_path),
        "standard_baseline_selection": sha256_file(standard_baseline_path),
        "holdout_baseline_selection": sha256_file(holdout_baseline_path),
    }


def _baseline_auc(selection: dict[str, Any], field: str) -> float:
    try:
        return _finite_float(selection["validation"]["auc"], field)
    except (KeyError, TypeError) as exc:
        raise SelectionError(f"{field} is missing") from exc


def _holdout_baseline_doa(
    selection: dict[str, Any],
) -> tuple[float, float]:
    try:
        metrics = selection["validation_doa"]
        return (
            _finite_float(
                metrics["holdout_doa_weighted"],
                "holdout baseline validation_doa.holdout_doa_weighted",
            ),
            _finite_float(
                metrics["holdout_doa"],
                "holdout baseline validation_doa.holdout_doa",
            ),
        )
    except (KeyError, TypeError) as exc:
        raise SelectionError("holdout baseline validation DOA is missing") from exc


def _holdout_selection_payload(
    *,
    candidate: dict[str, Any],
    doa: dict[str, Any],
    protocol: dict[str, Any],
    baseline_auc: float,
    baseline_weighted_doa: float,
    baseline_doa: float,
) -> dict[str, Any]:
    payload = {
        "selection_mode": "plugin",
        "epoch": candidate["epoch"],
        "model_name": candidate["model_name"],
        "checkpoint_path": candidate["checkpoint_path"],
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "mastery_path": candidate["mastery_path"],
        "mastery_sha256": candidate["mastery_sha256"],
        "id_maps_path": candidate["id_maps_path"],
        "id_maps_sha256": candidate["id_maps_sha256"],
        "validation": candidate["validation"],
        "validation_doa": doa,
        "dataset": doa["dataset"],
        "holdout_assignments_sha256": doa["holdout_assignments_sha256"],
        "plugin_config": candidate["plugin_config"],
        "recipe_config": candidate["recipe_config"],
        "recipe_id": candidate["recipe_id"],
        "backbone_config": candidate["backbone_config"],
        "protocol": protocol,
        "constraints": {
            "baseline_auc": baseline_auc,
            "baseline_holdout_doa_weighted": baseline_weighted_doa,
            "baseline_holdout_doa": baseline_doa,
            "auc_tolerance": 0.0,
        },
        "constraint_deltas": {
            "auc": candidate["validation"]["auc"] - baseline_auc,
            "holdout_doa_weighted": (
                doa["holdout_doa_weighted"] - baseline_weighted_doa
            ),
            "holdout_doa": doa["holdout_doa"] - baseline_doa,
        },
    }
    payload["frozen_config_id"] = compute_frozen_config_id(
        checkpoint_sha256=payload["checkpoint_sha256"],
        id_maps_sha256=payload["id_maps_sha256"],
        plugin_config=payload["plugin_config"],
        backbone_config=payload["backbone_config"],
        protocol=payload["protocol"],
    )
    return payload


def _candidate_diagnostic(candidate: JointCandidate) -> dict[str, Any]:
    return {
        "recipe_id": candidate.recipe_id,
        "standard_model_name": candidate.standard["model_name"],
        "standard_epoch": candidate.standard["epoch"],
        "holdout_model_name": candidate.holdout["model_name"],
        "holdout_epoch": candidate.holdout["epoch"],
        "standard_auc_delta": candidate.standard_auc_delta,
        "holdout_auc_delta": candidate.holdout_auc_delta,
        "min_validation_auc_delta": candidate.min_auc_delta,
        "holdout_weighted_doa_delta": candidate.holdout_weighted_doa_delta,
        "holdout_doa_delta": candidate.holdout_doa_delta,
    }


def _needs_adapter_diagnostics(
    *,
    joint_candidates: list[JointCandidate],
    hard_feasible: list[JointCandidate],
    output_dir: Path,
    auc_safety_margin: float,
    input_sha256: dict[str, str],
) -> dict[str, Any]:
    best = (
        min(
            hard_feasible,
            key=lambda candidate: (
                -candidate.min_auc_delta,
                -candidate.holdout_doa_delta,
                candidate.standard["epoch"],
                candidate.holdout["epoch"],
            ),
        )
        if hard_feasible
        else None
    )
    payload = {
        "decision": "needs_adapter",
        "auc_safety_margin": auc_safety_margin,
        "joint_candidate_count": len(joint_candidates),
        "hard_feasible_count": len(hard_feasible),
        "best_hard_feasible_min_auc_delta": (
            best.min_auc_delta if best is not None else None
        ),
        "input_sha256": input_sha256,
        "candidates": [
            _candidate_diagnostic(candidate) for candidate in joint_candidates
        ],
    }
    _atomic_write_jsons({output_dir / "joint_diagnostics.json": payload})
    return payload


def select_joint_recipe(
    *,
    standard_manifest_path: Path,
    holdout_manifest_path: Path,
    holdout_doa_csv_path: Path,
    standard_baseline_path: Path,
    holdout_baseline_path: Path,
    output_dir: Path,
    auc_safety_margin: float = 0.001,
) -> dict[str, Any]:
    auc_safety_margin = _finite_float(
        auc_safety_margin,
        "auc_safety_margin",
    )
    if auc_safety_margin < 0.0:
        raise SelectionError("auc_safety_margin must be non-negative")
    standard_manifest_path = Path(standard_manifest_path)
    holdout_manifest_path = Path(holdout_manifest_path)
    holdout_doa_csv_path = Path(holdout_doa_csv_path)
    standard_baseline_path = Path(standard_baseline_path)
    holdout_baseline_path = Path(holdout_baseline_path)
    output_dir = Path(output_dir)

    standard_rows = load_candidate_manifest(standard_manifest_path)
    holdout_rows = load_candidate_manifest(holdout_manifest_path)
    standard_protocol = validate_manifest_protocol(
        standard_rows,
        expected_data_protocol="standard",
    )
    holdout_protocol = validate_manifest_protocol(
        holdout_rows,
        expected_data_protocol="holdout",
    )
    standard_identity = {
        key: value
        for key, value in standard_protocol.items()
        if key != "data_protocol"
    }
    holdout_identity = {
        key: value
        for key, value in holdout_protocol.items()
        if key != "data_protocol"
    }
    if standard_identity != holdout_identity:
        raise SelectionError("standard and holdout manifests use different protocols")

    standard_candidates = index_candidates(standard_rows)
    holdout_candidates = index_candidates(holdout_rows)
    for candidate in (*standard_candidates.values(), *holdout_candidates.values()):
        _require_recipe(candidate)
    validate_candidate_artifacts(
        candidates=standard_candidates,
        manifest_path=standard_manifest_path,
    )
    holdout_doa_rows = load_validation_doa_rows(holdout_doa_csv_path)
    missing_doa = sorted(set(holdout_candidates) - set(holdout_doa_rows))
    if missing_doa:
        raise SelectionError(
            f"DOA CSV is missing candidates: {', '.join(missing_doa)}"
        )
    validate_doa_binding(
        candidates=holdout_candidates,
        doa_rows=holdout_doa_rows,
        protocol=holdout_protocol,
        manifest_path=holdout_manifest_path,
    )
    holdout_identities = {
        (row["dataset"], row["holdout_assignments_sha256"])
        for row in holdout_doa_rows.values()
    }
    if len(holdout_identities) != 1:
        raise SelectionError(
            "validation DOA rows contain mixed dataset/holdout assignments"
        )
    holdout_dataset, holdout_assignments_sha256 = next(iter(holdout_identities))
    if holdout_dataset != holdout_protocol["dataset_name"]:
        raise SelectionError("holdout DOA and manifest use different datasets")

    standard_baseline = load_selection(standard_baseline_path)
    holdout_baseline = load_selection(holdout_baseline_path)
    if standard_baseline.get("protocol") != standard_protocol:
        raise SelectionError("standard baseline uses a different protocol")
    if holdout_baseline.get("protocol") != holdout_protocol:
        raise SelectionError("holdout baseline uses a different protocol")
    if standard_baseline.get("dataset") != holdout_dataset:
        raise SelectionError("standard baseline uses a different dataset")
    if holdout_baseline.get("dataset") != holdout_dataset:
        raise SelectionError("holdout baseline uses a different dataset")
    if (
        holdout_baseline.get("holdout_assignments_sha256")
        != holdout_assignments_sha256
    ):
        raise SelectionError("holdout baseline uses different holdout assignments")

    standard_baseline_auc = _baseline_auc(
        standard_baseline,
        "standard baseline validation.auc",
    )
    holdout_baseline_auc = _baseline_auc(
        holdout_baseline,
        "holdout baseline validation.auc",
    )
    holdout_baseline_weighted_doa, holdout_baseline_doa = _holdout_baseline_doa(
        holdout_baseline
    )
    input_sha256 = _input_hashes(
        standard_manifest_path=standard_manifest_path,
        holdout_manifest_path=holdout_manifest_path,
        holdout_doa_csv_path=holdout_doa_csv_path,
        standard_baseline_path=standard_baseline_path,
        holdout_baseline_path=holdout_baseline_path,
    )

    standard_by_recipe: dict[str, list[dict[str, Any]]] = {}
    holdout_by_recipe: dict[str, list[dict[str, Any]]] = {}
    for candidate in standard_candidates.values():
        standard_by_recipe.setdefault(candidate["recipe_id"], []).append(candidate)
    for candidate in holdout_candidates.values():
        holdout_by_recipe.setdefault(candidate["recipe_id"], []).append(candidate)
    joint_candidates = [
        JointCandidate(
            recipe_id=recipe_id,
            standard=standard,
            holdout=holdout,
            holdout_doa=holdout_doa_rows[holdout["model_name"]],
            standard_auc_delta=(
                standard["validation"]["auc"] - standard_baseline_auc
            ),
            holdout_auc_delta=(
                holdout["validation"]["auc"] - holdout_baseline_auc
            ),
            holdout_weighted_doa_delta=(
                holdout_doa_rows[holdout["model_name"]]["holdout_doa_weighted"]
                - holdout_baseline_weighted_doa
            ),
            holdout_doa_delta=(
                holdout_doa_rows[holdout["model_name"]]["holdout_doa"]
                - holdout_baseline_doa
            ),
        )
        for recipe_id in sorted(set(standard_by_recipe) & set(holdout_by_recipe))
        for standard in standard_by_recipe[recipe_id]
        for holdout in holdout_by_recipe[recipe_id]
    ]
    hard_feasible = [
        candidate
        for candidate in joint_candidates
        if candidate.standard_auc_delta >= 0.0
        and candidate.holdout_auc_delta >= 0.0
        and candidate.holdout_weighted_doa_delta >= 0.0
        and candidate.holdout_doa_delta > 0.0
    ]
    safe = [
        candidate
        for candidate in hard_feasible
        if candidate.min_auc_delta >= auc_safety_margin
    ]
    if not safe:
        return _needs_adapter_diagnostics(
            joint_candidates=joint_candidates,
            hard_feasible=hard_feasible,
            output_dir=output_dir,
            auc_safety_margin=auc_safety_margin,
            input_sha256=input_sha256,
        )
    selected = min(
        safe,
        key=lambda candidate: (
            -candidate.min_auc_delta,
            -candidate.holdout_doa_delta,
            candidate.standard["epoch"],
            candidate.holdout["epoch"],
        ),
    )

    standard_selection = standard_selection_payload(
        mode="plugin",
        candidate=selected.standard,
        protocol=standard_protocol,
        baseline_auc=standard_baseline_auc,
    )
    holdout_selection = _holdout_selection_payload(
        candidate=selected.holdout,
        doa=selected.holdout_doa,
        protocol=holdout_protocol,
        baseline_auc=holdout_baseline_auc,
        baseline_weighted_doa=holdout_baseline_weighted_doa,
        baseline_doa=holdout_baseline_doa,
    )
    standard_selection["input_sha256"] = input_sha256
    holdout_selection["input_sha256"] = input_sha256
    joint_selection = {
        "decision": "shared",
        "recipe_id": selected.recipe_id,
        "standard_model_name": selected.standard["model_name"],
        "holdout_model_name": selected.holdout["model_name"],
        "standard_epoch": selected.standard["epoch"],
        "holdout_epoch": selected.holdout["epoch"],
        "standard_frozen_config_id": standard_selection["frozen_config_id"],
        "holdout_frozen_config_id": holdout_selection["frozen_config_id"],
        "standard_validation_auc_delta": selected.standard_auc_delta,
        "holdout_validation_auc_delta": selected.holdout_auc_delta,
        "min_validation_auc_delta": selected.min_auc_delta,
        "holdout_validation_weighted_doa_delta": (
            selected.holdout_weighted_doa_delta
        ),
        "holdout_validation_doa_delta": selected.holdout_doa_delta,
        "auc_safety_margin": auc_safety_margin,
        "input_sha256": input_sha256,
    }
    _atomic_write_jsons({
        output_dir / "standard_selection.json": standard_selection,
        output_dir / "holdout_selection.json": holdout_selection,
        output_dir / "joint_selection.json": joint_selection,
    })
    return joint_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select one plugin recipe across standard and holdout splits"
    )
    parser.add_argument("--standard-manifest", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--holdout-valid-doa-csv", type=Path, required=True)
    parser.add_argument("--standard-baseline-selection", type=Path, required=True)
    parser.add_argument("--holdout-baseline-selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--auc-safety-margin", type=float, default=0.001)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = select_joint_recipe(
            standard_manifest_path=args.standard_manifest,
            holdout_manifest_path=args.holdout_manifest,
            holdout_doa_csv_path=args.holdout_valid_doa_csv,
            standard_baseline_path=args.standard_baseline_selection,
            holdout_baseline_path=args.holdout_baseline_selection,
            output_dir=args.output_dir,
            auc_safety_margin=args.auc_safety_margin,
        )
    except SelectionError as exc:
        print(f"selection failed: {exc}", file=sys.stderr)
        return 2
    return 3 if result["decision"] == "needs_adapter" else 0


if __name__ == "__main__":
    raise SystemExit(main())
