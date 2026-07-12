from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def _expand_source_paths(paths: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(sorted(item for item in path.rglob("*") if item.is_file()))
        else:
            files.append(path)
    return files


def _canonicalize_artifact_row(row: Mapping[str, Any]) -> dict[str, Any]:
    canonical = dict(row)
    for field in ("seed", "split_seed"):
        value = canonical.get(field)
        if isinstance(value, str):
            try:
                canonical[field] = int(value)
            except ValueError:
                pass
    value = canonical.get("value")
    if isinstance(value, str):
        try:
            canonical["value"] = float(value)
        except ValueError:
            pass
    return canonical


def _json_rows(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        nested = payload.get("rows")
        if isinstance(nested, list):
            payload = nested
        else:
            payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("JSON artifact must contain an object or list of objects")
    if not all(isinstance(row, Mapping) for row in payload):
        raise ValueError("JSON artifact rows must be objects")
    return payload


def _read_artifact_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not set(REQUIRED_BASELINE_FIELDS).issubset(reader.fieldnames or ()):
                raise ValueError(
                    "artifact does not declare required baseline fields"
                )
            return [_canonicalize_artifact_row(row) for row in reader]
    if suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = [_canonicalize_artifact_row(row) for row in _json_rows(payload)]
        if rows and any(
            not set(REQUIRED_BASELINE_FIELDS).issubset(row) for row in rows
        ):
            raise ValueError("artifact does not declare required baseline fields")
        return rows
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError(f"JSONL row {line_number} is not an object")
                row = _canonicalize_artifact_row(payload)
                if not set(REQUIRED_BASELINE_FIELDS).issubset(row):
                    raise ValueError(
                        "artifact does not declare required baseline fields"
                    )
                rows.append(row)
        return rows
    shown_suffix = suffix if suffix else "(none)"
    raise ValueError(f"unsupported source artifact type: {shown_suffix}")


def audit_baseline_sources(
    paths: Iterable[str | Path], dataset_audits: Mapping[str, Any]
) -> dict[str, Any]:
    normalized_paths = [Path(path) for path in paths]
    source_roots: list[dict[str, Any]] = []
    for path in normalized_paths:
        reasons: list[str] = []
        if not path.exists():
            kind = "missing"
            reasons.append("source path does not exist")
        elif path.is_dir():
            kind = "directory"
            if not any(item.is_file() for item in path.rglob("*")):
                reasons.append("source directory contains no files")
        else:
            kind = "file"
        source_roots.append({
            "source_path": str(path.resolve()),
            "kind": kind,
            "reasons": reasons,
        })
    discovered: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    binding_rejected: list[dict[str, Any]] = []
    for source_path in _expand_source_paths(normalized_paths):
        source: dict[str, Any] = {
            "source_path": str(source_path.resolve()),
            "source_sha256": None,
            "rows": [],
            "row_reasons": [],
            "reasons": [],
        }
        if not source_path.is_file():
            source["reasons"].append("source path does not exist")
            discovered.append(source)
            continue
        source["source_sha256"] = _file_sha256(source_path)
        try:
            source["rows"] = _read_artifact_rows(source_path)
        except (OSError, UnicodeError, csv.Error, json.JSONDecodeError, ValueError) as error:
            source["reasons"].append(str(error))
        for row in source["rows"]:
            declared_path = row.get("source_path")
            declared_resolved = (
                str(Path(declared_path).resolve())
                if isinstance(declared_path, str)
                else None
            )
            if declared_resolved != source["source_path"]:
                reasons = [
                    "source_path does not match containing artifact: expected "
                    + source["source_path"]
                ]
                source["row_reasons"].append(reasons)
                binding_rejected.append({"row": row, "reasons": reasons})
            else:
                source["row_reasons"].append([])
                rows.append(row)
        discovered.append(source)

    result = audit_baseline_rows(rows, dataset_audits)
    result["rejected_rows"].extend(binding_rejected)
    result["rejected_count"] = len(result["rejected_rows"])
    accepted_sources: list[dict[str, Any]] = []
    rejected_sources: list[dict[str, Any]] = []
    for source in discovered:
        source_record = {
            "source_path": source["source_path"],
            "source_sha256": source["source_sha256"],
            "row_count": len(source["rows"]),
            "reasons": list(source["reasons"]),
        }
        if not source_record["row_count"] and not source_record["reasons"]:
            source_record["reasons"].append("source artifact contains no rows")
        for index, row in enumerate(source["rows"], start=1):
            reasons = source["row_reasons"][index - 1]
            if not reasons:
                row_audit = audit_baseline_rows([row], dataset_audits)
                if row_audit["rejected_rows"]:
                    reasons = row_audit["rejected_rows"][0]["reasons"]
            if reasons:
                source_record["reasons"].append(
                    f"row {index}: " + "; ".join(reasons)
                )
        target = accepted_sources if not source_record["reasons"] else rejected_sources
        target.append(source_record)
    result["discovered_source_count"] = len(discovered)
    result["source_root_count"] = len(source_roots)
    result["source_roots"] = source_roots
    result["accepted_source_records"] = accepted_sources
    result["rejected_source_records"] = rejected_sources
    result.pop("audit_sha256")
    result["audit_sha256"] = canonical_sha256(result)
    return result


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
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", action="append", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    audits = json.loads(args.dataset_audit.read_text(encoding="utf-8"))
    paths = [*BASELINE_SOURCE_PATHS, *(args.source or ())]
    result = audit_baseline_sources(paths, audits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
