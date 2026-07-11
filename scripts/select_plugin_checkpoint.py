from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plugin_campaign import compute_frozen_config_id, sha256_file


AUC_TOLERANCE = 0.002
PROTOCOL_FIELDS = (
    "split",
    "seed",
    "doa_seed",
    "min_responses",
    "max_pairs_per_concept",
    "split_seed",
    "q_matrix_sha256",
)
APPROVED_PROTOCOL = {
    "split": "valid",
    "seed": 42,
    "doa_seed": 42,
    "min_responses": 3,
    "split_seed": 2024,
}


class SelectionError(RuntimeError):
    """Raised when validation-only checkpoint selection cannot be completed."""


def _finite_float(value: Any, field: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise SelectionError(f"{field} must be numeric") from exc
    if not math.isfinite(normalized):
        raise SelectionError(f"{field} must be finite")
    return normalized


def _integer(value: Any, field: str) -> int:
    normalized = _finite_float(value, field)
    if not normalized.is_integer():
        raise SelectionError(f"{field} must be an integer")
    return int(normalized)


def _sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SelectionError(f"{field} must be a lowercase SHA-256")
    return value


def load_candidate_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SelectionError(f"cannot read candidate manifest: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SelectionError(
                f"invalid candidate JSON on line {line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise SelectionError(f"candidate line {line_number} must be an object")
        rows.append(row)
    if not rows:
        raise SelectionError("candidate manifest is empty")
    return rows


def validate_manifest_protocol(
    candidate_rows: list[dict[str, Any]],
    *,
    expected_data_protocol: str | None = None,
) -> dict[str, Any]:
    first = candidate_rows[0].get("protocol")
    if not isinstance(first, dict):
        raise SelectionError("candidate manifest must bind a validation protocol")
    missing = [field for field in PROTOCOL_FIELDS if field not in first]
    if missing:
        raise SelectionError(f"candidate protocol missing: {', '.join(missing)}")
    has_data_protocol = "data_protocol" in first
    has_dataset_name = "dataset_name" in first
    if has_data_protocol != has_dataset_name:
        raise SelectionError(
            "candidate protocol data_protocol and dataset_name must be provided together"
        )
    if has_data_protocol:
        if first["data_protocol"] not in {"standard", "holdout"}:
            raise SelectionError(
                "candidate protocol data_protocol must be 'standard' or 'holdout'"
            )
        if (
            not isinstance(first["dataset_name"], str)
            or not first["dataset_name"].strip()
        ):
            raise SelectionError(
                "candidate protocol dataset_name must be a non-empty string"
            )
    if expected_data_protocol is not None:
        if first.get("data_protocol") != expected_data_protocol:
            raise SelectionError(
                f"candidate protocol must be {expected_data_protocol!r}"
            )
    for field, expected in APPROVED_PROTOCOL.items():
        if first[field] != expected:
            raise SelectionError(
                f"candidate protocol {field} must be {expected!r}, "
                f"got {first[field]!r}"
            )
    _sha256(first["q_matrix_sha256"], "candidate protocol q_matrix_sha256")
    for row in candidate_rows[1:]:
        if row.get("protocol") != first:
            raise SelectionError("candidate manifest contains mixed protocols")
    return dict(first)


def index_candidates(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        model_name = row.get("model_name")
        if not isinstance(model_name, str) or not model_name:
            raise SelectionError("every candidate requires a non-empty model_name")
        if model_name in indexed:
            raise SelectionError(f"duplicate candidate model_name: {model_name}")
        try:
            epoch = int(row["epoch"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SelectionError(f"candidate {model_name} has invalid epoch") from exc
        validation = row.get("validation")
        if not isinstance(validation, dict):
            raise SelectionError(f"candidate {model_name} lacks validation metrics")
        normalized = dict(row)
        normalized["epoch"] = epoch
        normalized["validation"] = {
            name: _finite_float(validation.get(name), f"{model_name}.validation.{name}")
            for name in ("auc", "acc", "rmse")
        }
        plugin_config = row.get("plugin_config")
        if not isinstance(plugin_config, dict):
            raise SelectionError(f"candidate {model_name} lacks plugin_config")
        normalized["plugin_config"] = dict(plugin_config)
        backbone_config = row.get("backbone_config")
        if not isinstance(backbone_config, dict):
            raise SelectionError(f"candidate {model_name} lacks backbone_config")
        normalized["backbone_config"] = dict(backbone_config)
        for field in ("checkpoint_path", "mastery_path", "id_maps_path"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise SelectionError(f"candidate {model_name} lacks {field}")
        for field in (
            "checkpoint_sha256",
            "mastery_sha256",
            "id_maps_sha256",
        ):
            normalized[field] = _sha256(row.get(field), f"{model_name}.{field}")
        indexed[model_name] = normalized
    return indexed


def load_validation_doa_rows(path: Path) -> dict[str, dict[str, Any]]:
    try:
        handle = path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise SelectionError(f"cannot read validation DOA CSV: {path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        rows: dict[str, dict[str, Any]] = {}
        for row in reader:
            model_name = row.get("model")
            if not model_name:
                raise SelectionError("every DOA row requires model")
            if model_name in rows:
                raise SelectionError(f"duplicate DOA model: {model_name}")
            normalized = {
                name: _finite_float(row.get(name), f"{model_name}.{name}")
                for name in (
                    "holdout_doa",
                    "holdout_doa_weighted",
                    "holdout_num_concepts_evaluated",
                    "holdout_num_pairs",
                )
            }
            if normalized["holdout_num_concepts_evaluated"] <= 0:
                raise SelectionError(f"{model_name} has no evaluated holdout concepts")
            if normalized["holdout_num_pairs"] <= 0:
                raise SelectionError(f"{model_name} has no evaluated holdout pairs")
            normalized.update({
                "dataset": row.get("dataset"),
                "holdout_assignments_sha256": _sha256(
                    row.get("holdout_assignments_sha256"),
                    f"{model_name}.holdout_assignments_sha256",
                ),
                "split": row.get("split"),
                "doa_seed": _integer(row.get("doa_seed"), f"{model_name}.doa_seed"),
                "min_responses": _integer(
                    row.get("min_responses"), f"{model_name}.min_responses"
                ),
                "max_pairs_per_concept": _integer(
                    row.get("max_pairs_per_concept"),
                    f"{model_name}.max_pairs_per_concept",
                ),
                "split_seed": _integer(
                    row.get("split_seed"), f"{model_name}.split_seed"
                ),
                "mastery_sha256": _sha256(
                    row.get("mastery_sha256"), f"{model_name}.mastery_sha256"
                ),
                "id_maps_sha256": _sha256(
                    row.get("id_maps_sha256"), f"{model_name}.id_maps_sha256"
                ),
            })
            if not isinstance(normalized["dataset"], str) or not normalized[
                "dataset"
            ].strip():
                raise SelectionError(f"{model_name}.dataset must be non-empty")
            rows[model_name] = normalized
    if not rows:
        raise SelectionError("validation DOA CSV is empty")
    return rows


def load_selection(path: Path | None) -> dict[str, Any]:
    if path is None:
        raise SelectionError("plugin mode requires --baseline-selection")
    try:
        baseline = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectionError(f"cannot read baseline selection: {path}") from exc
    if not isinstance(baseline, dict):
        raise SelectionError("baseline selection must be an object")
    return baseline


def _checkpoint_path(manifest_path: Path, relative_path: str) -> Path:
    root = manifest_path.parent.parent
    path = root / relative_path
    if not path.is_file():
        raise SelectionError(f"selected checkpoint does not exist: {path}")
    return path


def validate_candidate_artifacts(
    *,
    candidates: dict[str, dict[str, Any]],
    manifest_path: Path,
) -> None:
    root = manifest_path.parent.parent
    for model_name, candidate in candidates.items():
        for path_field, hash_field in (
            ("checkpoint_path", "checkpoint_sha256"),
            ("mastery_path", "mastery_sha256"),
            ("id_maps_path", "id_maps_sha256"),
        ):
            artifact_path = root / candidate[path_field]
            if not artifact_path.is_file():
                raise SelectionError(
                    f"candidate {model_name} artifact does not exist: {artifact_path}"
                )
            actual_sha256 = sha256_file(artifact_path)
            if actual_sha256 != candidate[hash_field]:
                raise SelectionError(
                    f"candidate {model_name} {hash_field} does not match actual artifact"
                )


def validate_doa_binding(
    *,
    candidates: dict[str, dict[str, Any]],
    doa_rows: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
    manifest_path: Path,
) -> None:
    validate_candidate_artifacts(
        candidates=candidates,
        manifest_path=manifest_path,
    )
    for model_name, candidate in candidates.items():
        doa = doa_rows[model_name]
        for field in (
            "split",
            "doa_seed",
            "min_responses",
            "max_pairs_per_concept",
            "split_seed",
        ):
            if doa[field] != protocol[field]:
                raise SelectionError(
                    f"{model_name} DOA {field} does not match candidate protocol"
                )
        for hash_field in ("mastery_sha256", "id_maps_sha256"):
            if candidate[hash_field] != doa[hash_field]:
                raise SelectionError(
                    f"{model_name} DOA {hash_field} does not match candidate artifact"
                )


def _selection_payload(
    *,
    mode: str,
    candidate: dict[str, Any],
    doa: dict[str, Any],
    protocol: dict[str, Any],
    manifest_path: Path,
    baseline_auc: float,
    baseline_weighted: float,
) -> dict[str, Any]:
    checkpoint_path = _checkpoint_path(manifest_path, candidate["checkpoint_path"])
    payload = {
        "selection_mode": mode,
        "epoch": candidate["epoch"],
        "model_name": candidate["model_name"],
        "checkpoint_path": candidate["checkpoint_path"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "mastery_path": candidate["mastery_path"],
        "id_maps_path": candidate["id_maps_path"],
        "mastery_sha256": candidate["mastery_sha256"],
        "id_maps_sha256": candidate["id_maps_sha256"],
        "validation": candidate["validation"],
        "validation_doa": doa,
        "dataset": doa["dataset"],
        "holdout_assignments_sha256": doa["holdout_assignments_sha256"],
        "plugin_config": candidate["plugin_config"],
        "backbone_config": candidate["backbone_config"],
        "protocol": protocol,
        "constraints": {
            "baseline_auc": baseline_auc,
            "baseline_holdout_doa_weighted": baseline_weighted,
            "auc_tolerance": AUC_TOLERANCE if mode == "plugin" else 0.0,
        },
        "constraint_deltas": {
            "auc": candidate["validation"]["auc"] - baseline_auc,
            "holdout_doa_weighted": (
                doa["holdout_doa_weighted"] - baseline_weighted
            ),
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


def select_checkpoint(
    *,
    mode: str,
    manifest_path: Path,
    doa_csv_path: Path,
    output_path: Path,
    baseline_selection_path: Path | None = None,
) -> dict[str, Any]:
    if mode not in {"baseline", "plugin"}:
        raise SelectionError("mode must be 'baseline' or 'plugin'")
    manifest_path = Path(manifest_path)
    candidate_rows = load_candidate_manifest(manifest_path)
    protocol = validate_manifest_protocol(candidate_rows)
    candidates = index_candidates(candidate_rows)
    doa_rows = load_validation_doa_rows(Path(doa_csv_path))
    validation_identities = {
        (row["dataset"], row["holdout_assignments_sha256"])
        for row in doa_rows.values()
    }
    if len(validation_identities) != 1:
        raise SelectionError(
            "validation DOA rows contain mixed dataset/holdout assignments"
        )
    validation_dataset, validation_holdout_sha256 = next(
        iter(validation_identities)
    )
    missing = sorted(set(candidates) - set(doa_rows))
    if missing:
        raise SelectionError(f"DOA CSV is missing candidates: {', '.join(missing)}")
    validate_doa_binding(
        candidates=candidates,
        doa_rows=doa_rows,
        protocol=protocol,
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
        selected_doa = doa_rows[selected["model_name"]]
        baseline_auc = selected["validation"]["auc"]
        baseline_weighted = selected_doa["holdout_doa_weighted"]
    else:
        baseline = load_selection(
            Path(baseline_selection_path) if baseline_selection_path else None
        )
        if baseline.get("protocol") != protocol:
            raise SelectionError("plugin and baseline selections use different protocols")
        if baseline.get("dataset") != validation_dataset:
            raise SelectionError("plugin and baseline selections use different datasets")
        if (
            baseline.get("holdout_assignments_sha256")
            != validation_holdout_sha256
        ):
            raise SelectionError(
                "plugin and baseline selections use different holdout assignments"
            )
        try:
            baseline_auc = _finite_float(
                baseline["validation"]["auc"], "baseline.validation.auc"
            )
            baseline_weighted = _finite_float(
                baseline["validation_doa"]["holdout_doa_weighted"],
                "baseline.validation_doa.holdout_doa_weighted",
            )
        except (KeyError, TypeError) as exc:
            raise SelectionError("baseline selection lacks constraint metrics") from exc
        eligible = [
            candidate
            for candidate in candidates.values()
            if candidate["validation"]["auc"] >= baseline_auc - AUC_TOLERANCE
            and doa_rows[candidate["model_name"]]["holdout_doa_weighted"]
            >= baseline_weighted
        ]
        if not eligible:
            raise SelectionError("no plugin candidate satisfies baseline constraints")
        selected = min(
            eligible,
            key=lambda row: (
                -doa_rows[row["model_name"]]["holdout_doa"],
                -row["validation"]["auc"],
                row["epoch"],
            ),
        )
        selected_doa = doa_rows[selected["model_name"]]

    payload = _selection_payload(
        mode=mode,
        candidate=selected,
        doa=selected_doa,
        protocol=protocol,
        manifest_path=manifest_path,
        baseline_auc=baseline_auc,
        baseline_weighted=baseline_weighted,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise SelectionError(f"selection output already exists: {output_path}") from exc
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a validation-only plugin checkpoint")
    parser.add_argument("--mode", choices=("baseline", "plugin"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--valid-doa-csv", type=Path, required=True)
    parser.add_argument("--baseline-selection", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        select_checkpoint(
            mode=args.mode,
            manifest_path=args.manifest,
            doa_csv_path=args.valid_doa_csv,
            output_path=args.output_json,
            baseline_selection_path=args.baseline_selection,
        )
    except SelectionError as exc:
        print(f"selection failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
