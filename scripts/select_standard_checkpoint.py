from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plugin_campaign import (
    compute_frozen_config_id,
    compute_recipe_id,
    sha256_file,
)
from scripts.select_plugin_checkpoint import (
    SelectionError,
    _finite_float,
    index_candidates,
    load_candidate_manifest,
    load_selection,
    validate_candidate_artifacts,
    validate_manifest_protocol,
)


def _require_recipe(candidate: dict[str, Any]) -> None:
    model_name = candidate["model_name"]
    recipe_config = candidate.get("recipe_config")
    recipe_id = candidate.get("recipe_id")
    if not isinstance(recipe_config, dict):
        raise SelectionError(f"candidate {model_name} lacks recipe_config")
    if not isinstance(recipe_id, str):
        raise SelectionError(f"candidate {model_name} lacks recipe_id")
    expected = compute_recipe_id(recipe_config, candidate["backbone_config"])
    if recipe_id != expected:
        raise SelectionError(f"candidate {model_name} recipe_id is invalid")


def standard_selection_payload(
    *,
    mode: str,
    candidate: dict[str, Any],
    protocol: dict[str, Any],
    baseline_auc: float,
) -> dict[str, Any]:
    payload = {
        "selection_mode": mode,
        "epoch": candidate["epoch"],
        "model_name": candidate["model_name"],
        "checkpoint_path": candidate["checkpoint_path"],
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "mastery_path": candidate["mastery_path"],
        "mastery_sha256": candidate["mastery_sha256"],
        "id_maps_path": candidate["id_maps_path"],
        "id_maps_sha256": candidate["id_maps_sha256"],
        "validation": candidate["validation"],
        "dataset": protocol["dataset_name"],
        "plugin_config": candidate["plugin_config"],
        "recipe_config": candidate["recipe_config"],
        "recipe_id": candidate["recipe_id"],
        "backbone_config": candidate["backbone_config"],
        "protocol": protocol,
        "constraints": {
            "baseline_auc": baseline_auc,
            "auc_tolerance": 0.0,
        },
        "constraint_deltas": {
            "auc": candidate["validation"]["auc"] - baseline_auc,
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


def _write_exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise SelectionError(f"selection output already exists: {path}") from exc


def select_standard_checkpoint(
    *,
    mode: str,
    manifest_path: Path,
    output_path: Path,
    baseline_selection_path: Path | None = None,
) -> dict[str, Any]:
    if mode not in {"baseline", "plugin"}:
        raise SelectionError("mode must be 'baseline' or 'plugin'")
    manifest_path = Path(manifest_path)
    rows = load_candidate_manifest(manifest_path)
    protocol = validate_manifest_protocol(
        rows,
        expected_data_protocol="standard",
    )
    candidates = index_candidates(rows)
    for candidate in candidates.values():
        _require_recipe(candidate)
    validate_candidate_artifacts(
        candidates=candidates,
        manifest_path=manifest_path,
    )

    if mode == "baseline":
        eligible = [
            candidate
            for candidate in candidates.values()
            if _finite_float(
                candidate["plugin_config"].get("aux_weight"),
                f"{candidate['model_name']}.plugin_config.aux_weight",
            )
            == 0.0
        ]
        if not eligible:
            raise SelectionError("baseline mode found no aux_weight=0 candidates")
        selected = min(
            eligible,
            key=lambda row: (-row["validation"]["auc"], row["epoch"]),
        )
        baseline_auc = selected["validation"]["auc"]
    else:
        baseline = load_selection(
            Path(baseline_selection_path) if baseline_selection_path else None
        )
        if baseline.get("protocol") != protocol:
            raise SelectionError("plugin and baseline selections use different protocols")
        if baseline.get("dataset") != protocol["dataset_name"]:
            raise SelectionError("plugin and baseline selections use different datasets")
        try:
            baseline_auc = _finite_float(
                baseline["validation"]["auc"],
                "baseline.validation.auc",
            )
        except (KeyError, TypeError) as exc:
            raise SelectionError("baseline selection lacks constraint metrics") from exc
        eligible = [
            candidate
            for candidate in candidates.values()
            if candidate["validation"]["auc"] >= baseline_auc
        ]
        if not eligible:
            raise SelectionError(
                "no standard plugin candidate satisfies baseline AUC"
            )
        selected = min(
            eligible,
            key=lambda row: (-row["validation"]["auc"], row["epoch"]),
        )

    payload = standard_selection_payload(
        mode=mode,
        candidate=selected,
        protocol=protocol,
        baseline_auc=baseline_auc,
    )
    payload["manifest_sha256"] = sha256_file(manifest_path)
    _write_exclusive_json(Path(output_path), payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a standard-protocol validation checkpoint"
    )
    parser.add_argument("--mode", choices=("baseline", "plugin"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-selection", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        select_standard_checkpoint(
            mode=args.mode,
            manifest_path=args.manifest,
            baseline_selection_path=args.baseline_selection,
            output_path=args.output_json,
        )
    except SelectionError as exc:
        print(f"selection failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
