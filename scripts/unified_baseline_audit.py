from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts.unified_dataset_audit import canonical_sha256


REQUIRED_BASELINE_FIELDS = (
    "dataset_id", "model", "seed", "split_seed", "split", "metric", "value",
    "data_sha256", "q_sha256", "prediction_sha256",
    "prediction_order_sha256", "config_sha256", "checkpoint_sha256", "source_path",
)
SAME_PROTOCOL = {"seed": 42, "split_seed": 2024}
BASELINE_SOURCE_PATHS = (
    Path("/home/xph/jwc/research/local_data/pyedmine_cd_baselines/job_outputs/"),
    Path("/home/xph/jwc/research/local_data/svgcd_baselines/job_outputs/"),
    Path("/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/formal-aaai-20260710/"),
    Path("/home/xph/jwc/research/decoupled_cd_v2/results/r14/ext_a17.csv"),
    Path("/home/xph/jwc/research/decoupled_cd_v2/results/r14/ext_moo.csv"),
    Path("/home/xph/jwc/research/decoupled_cd_v2/results/r14/ext_xes.csv"),
)
_HASH_FIELDS = (
    "data_sha256", "q_sha256", "prediction_sha256",
    "prediction_order_sha256", "config_sha256", "checkpoint_sha256",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory_baseline_sources(
    paths: Iterable[str | Path] = BASELINE_SOURCE_PATHS,
) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            inventory.append({
                "source_path": str(path), "source_sha256": None,
                "reasons": ["source path does not exist"],
            })
            continue
        files = (
            sorted(item for item in path.rglob("*") if item.is_file())
            if path.is_dir()
            else [path]
        )
        if not files:
            inventory.append({
                "source_path": str(path), "source_sha256": None,
                "reasons": ["source directory contains no files"],
            })
            continue
        for source_file in files:
            inventory.append({
                "source_path": str(source_file.resolve()),
                "source_sha256": _file_sha256(source_file), "reasons": [],
            })
    return inventory


def _dataset_records(dataset_audits: Mapping[str, Any]) -> Mapping[str, Any]:
    datasets = dataset_audits.get("datasets")
    return datasets if isinstance(datasets, Mapping) else dataset_audits


def _row_reasons(row: Mapping[str, Any], datasets: Mapping[str, Any]) -> list[str]:
    missing = [field for field in REQUIRED_BASELINE_FIELDS if field not in row]
    if missing:
        return ["missing required fields: " + ", ".join(missing)]
    reasons: list[str] = []
    for field, expected in SAME_PROTOCOL.items():
        if row[field] != expected:
            reasons.append(f"{field} mismatch: expected {expected}")
    value = row["value"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        reasons.append("value must be finite")
    for field in _HASH_FIELDS:
        value = row[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value.lower())
        ):
            reasons.append(f"{field} must be a SHA-256 hex digest")
    dataset_id = row["dataset_id"]
    dataset = datasets.get(dataset_id)
    if not isinstance(dataset, Mapping):
        reasons.append(f"dataset audit is missing: {dataset_id}")
        return reasons
    if not dataset.get("eligible"):
        reasons.append(f"dataset is not eligible: {dataset_id}")
        return reasons
    split_name = row["split"]
    split = dataset.get(split_name)
    if not isinstance(split, Mapping):
        reasons.append(f"dataset split audit is missing: {dataset_id}/{split_name}")
        return reasons
    for field in ("data_sha256", "q_sha256", "prediction_order_sha256"):
        if row[field] != split.get(field):
            reasons.append(f"{field} mismatch for {dataset_id}/{split_name}")
    return reasons


def audit_baseline_rows(
    rows: Iterable[Mapping[str, Any]], dataset_audits: Mapping[str, Any]
) -> dict[str, Any]:
    datasets = _dataset_records(dataset_audits)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    strongest: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for source_row in rows:
        row = dict(source_row)
        reasons = _row_reasons(row, datasets)
        if reasons:
            rejected.append({"row": row, "reasons": reasons})
            continue
        accepted.append(row)
        slot = strongest.setdefault(row["dataset_id"], {}).setdefault(
            row["split"], {}
        )
        current = slot.get(row["metric"])
        if current is None or row["value"] > current["value"]:
            slot[row["metric"]] = row
    result: dict[str, Any] = {
        "schema_version": 1,
        "same_protocol": dict(SAME_PROTOCOL),
        "required_fields": list(REQUIRED_BASELINE_FIELDS),
        "accepted_count": len(accepted), "rejected_count": len(rejected),
        "accepted_rows": accepted, "rejected_rows": rejected,
        "strongest_comparators": strongest,
        "source_files": inventory_baseline_sources(
            sorted({
                row.get("source_path")
                for row in [*accepted, *(item["row"] for item in rejected)]
                if row.get("source_path")
            })
        ),
    }
    result["audit_sha256"] = canonical_sha256(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit same-protocol external baseline rows."
    )
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", action="append", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = json.loads(args.rows.read_text(encoding="utf-8"))
    audits = json.loads(args.dataset_audit.read_text(encoding="utf-8"))
    result = audit_baseline_rows(rows, audits)
    result["source_inventory"] = inventory_baseline_sources(
        args.source or BASELINE_SOURCE_PATHS
    )
    result.pop("audit_sha256")
    result["audit_sha256"] = canonical_sha256(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
