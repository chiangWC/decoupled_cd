from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.unified_baseline_audit import (
    REQUIRED_BASELINE_FIELDS,
    SAME_PROTOCOL,
    audit_baseline_rows,
)
from scripts.unified_dataset_audit import canonical_sha256


_SCHEMA_V1_FIELDS = frozenset({
    "schema_version",
    "dataset_ids",
    "audit_sha256",
    "dataset_audit_sha256",
    "b0_validation_references",
    "cohort_sha256",
})
_SCHEMA_V2_FIELDS = frozenset({
    "schema_version",
    "dataset_ids",
    "a0_fingerprints",
    "comparator_audit_sha256",
    "dataset_audit_sha256",
    "rankings",
    "cohort_sha256",
})


def _verify_canonical_hash(
    payload: Mapping[str, Any],
    *,
    hash_field: str,
    mismatch_message: str,
) -> str:
    stored_hash = payload.get(hash_field)
    unhashed = dict(payload)
    unhashed.pop(hash_field, None)
    if not isinstance(stored_hash, str) or stored_hash != canonical_sha256(unhashed):
        raise ValueError(mismatch_message)
    return stored_hash


def _build_cohort(
    dataset_ids: Sequence[str],
    *,
    audit: Mapping[str, Any],
    b0_validation_references: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_ids = list(dataset_ids)
    if len(normalized_ids) != 3:
        raise ValueError("primary cohort requires exactly three datasets")
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("primary cohort dataset IDs must be unique")

    datasets = audit.get("datasets")
    if not isinstance(datasets, Mapping):
        raise ValueError("audit has no dataset records")
    audit_sha256 = _verify_canonical_hash(
        audit,
        hash_field="audit_sha256",
        mismatch_message="pool audit canonical SHA-256 mismatch",
    )

    dataset_hashes: dict[str, str] = {}
    selected_references: dict[str, Any] = {}
    for dataset_id in normalized_ids:
        record = datasets.get(dataset_id)
        if not isinstance(record, Mapping):
            raise ValueError(f"audit has no dataset record: {dataset_id}")
        record_hash = _verify_canonical_hash(
            record,
            hash_field="audit_sha256",
            mismatch_message=(
                f"dataset audit canonical SHA-256 mismatch: {dataset_id}"
            ),
        )
        if not record.get("eligible"):
            raise ValueError(f"cohort dataset is not eligible: {dataset_id}")
        if dataset_id not in b0_validation_references:
            raise ValueError(f"missing B0 validation reference: {dataset_id}")
        dataset_hashes[dataset_id] = record_hash
        selected_references[dataset_id] = b0_validation_references[dataset_id]

    cohort: dict[str, Any] = {
        "schema_version": 1,
        "dataset_ids": normalized_ids,
        "audit_sha256": audit_sha256,
        "dataset_audit_sha256": dataset_hashes,
        "b0_validation_references": selected_references,
    }
    cohort["cohort_sha256"] = canonical_sha256(cohort)
    return cohort


def load_verified_cohort(
    path: str | Path,
    *,
    expected_dataset_ids: Sequence[str] | None = None,
    expected_a0_fingerprints: Mapping[str, str] | None = None,
    expected_comparator_audit_sha256: str | None = None,
    expected_dataset_audit_sha256: str | None = None,
) -> dict[str, Any]:
    cohort_path = Path(path)
    try:
        existing = json.loads(cohort_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"existing cohort is unreadable: {error}") from error
    if not isinstance(existing, dict):
        raise ValueError("existing cohort is not a JSON object")
    unhashed = dict(existing)
    stored_hash = unhashed.pop("cohort_sha256", None)
    if stored_hash != canonical_sha256(unhashed):
        raise ValueError("canonical SHA-256 mismatch for frozen cohort")
    expectations = (
        expected_dataset_ids,
        expected_a0_fingerprints,
        expected_comparator_audit_sha256,
        expected_dataset_audit_sha256,
    )
    schema_version = existing.get("schema_version")
    if schema_version == 1:
        if any(expectation is not None for expectation in expectations):
            raise ValueError(
                "schema version 1 cannot use trusted schema-v2 expectations"
            )
        if set(existing) != _SCHEMA_V1_FIELDS:
            raise ValueError("schema version 1 field set mismatch")
        dataset_ids = existing.get("dataset_ids")
        if (
            not isinstance(dataset_ids, list)
            or len(dataset_ids) != 3
            or not all(
                isinstance(dataset_id, str) and bool(dataset_id)
                for dataset_id in dataset_ids
            )
            or len(set(dataset_ids)) != 3
            or not isinstance(existing.get("dataset_audit_sha256"), dict)
            or not isinstance(existing.get("b0_validation_references"), dict)
        ):
            raise ValueError("schema version 1 payload mismatch")
    elif schema_version == 2:
        if set(existing) != _SCHEMA_V2_FIELDS:
            raise ValueError("schema version 2 field set mismatch")
        if any(expectation is None for expectation in expectations):
            raise ValueError("trusted schema-v2 expectations are required")
        if existing.get("dataset_ids") != list(expected_dataset_ids or ()):
            raise ValueError("trusted dataset ID mismatch for frozen cohort")
        if existing.get("a0_fingerprints") != dict(expected_a0_fingerprints or {}):
            raise ValueError("trusted A0 fingerprint mismatch for frozen cohort")
        if (
            existing.get("comparator_audit_sha256")
            != expected_comparator_audit_sha256
        ):
            raise ValueError("trusted comparator audit hash mismatch for frozen cohort")
        if existing.get("dataset_audit_sha256") != expected_dataset_audit_sha256:
            raise ValueError("trusted dataset audit hash mismatch for frozen cohort")
    else:
        raise ValueError(f"unsupported or missing cohort schema version: {schema_version}")
    return existing


def _verify_existing(existing: dict[str, Any], requested: dict[str, Any]) -> None:
    if existing.get("dataset_ids") != requested["dataset_ids"]:
        raise ValueError("frozen cohort dataset list mismatch")
    if existing != requested:
        raise ValueError("frozen cohort metadata mismatch")


def _fsync_parent(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(path.parent, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def freeze_cohort(
    path: str | Path,
    dataset_ids: Sequence[str],
    *,
    audit: Mapping[str, Any],
    b0_validation_references: Mapping[str, Any],
) -> dict[str, Any]:
    cohort_path = Path(path)
    cohort = _build_cohort(
        dataset_ids,
        audit=audit,
        b0_validation_references=b0_validation_references,
    )
    cohort_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            cohort,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")

    try:
        file_descriptor = os.open(
            cohort_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o644,
        )
    except FileExistsError:
        existing = load_verified_cohort(cohort_path)
        _verify_existing(existing, cohort)
        return existing

    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            cohort_path.unlink()
        except OSError:
            pass
        raise
    _fsync_parent(cohort_path)
    return cohort


def _rows_by_dataset(
    rows: Sequence[Mapping[str, Any]], *, row_kind: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        dataset_id = row.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError(f"{row_kind} row has no dataset_id")
        if dataset_id in indexed:
            raise ValueError(f"duplicate {row_kind} row: {dataset_id}")
        indexed[dataset_id] = row
    return indexed


def _verified_audit_records(
    audit_rows: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], str]:
    audit_sha256 = _verify_canonical_hash(
        audit_rows,
        hash_field="audit_sha256",
        mismatch_message="dataset audit canonical SHA-256 mismatch",
    )
    source = audit_rows.get("datasets")
    if not isinstance(source, Mapping):
        raise ValueError("dataset audit has no dataset records")
    records: dict[str, Mapping[str, Any]] = {}
    for dataset_id, record in source.items():
        if not isinstance(dataset_id, str) or not isinstance(record, Mapping):
            raise ValueError("dataset audit records must be keyed objects")
        _verify_canonical_hash(
            record,
            hash_field="audit_sha256",
            mismatch_message=f"dataset record canonical SHA-256 mismatch: {dataset_id}",
        )
        records[dataset_id] = record
    return records, audit_sha256


def _verified_strongest_comparators(
    baseline_audit: Mapping[str, Any], dataset_audit: Mapping[str, Any]
) -> tuple[Mapping[str, Any], str]:
    audit_sha256 = _verify_canonical_hash(
        baseline_audit,
        hash_field="audit_sha256",
        mismatch_message="baseline audit canonical SHA-256 mismatch",
    )
    accepted_rows = baseline_audit.get("accepted_rows")
    rejected_rows = baseline_audit.get("rejected_rows")
    strongest = baseline_audit.get("strongest_comparators")
    if (
        baseline_audit.get("schema_version") != 1
        or baseline_audit.get("same_protocol") != SAME_PROTOCOL
        or baseline_audit.get("required_fields") != list(REQUIRED_BASELINE_FIELDS)
        or not isinstance(accepted_rows, list)
        or not isinstance(rejected_rows, list)
        or baseline_audit.get("accepted_count") != len(accepted_rows)
        or baseline_audit.get("rejected_count") != len(rejected_rows)
        or not isinstance(strongest, Mapping)
    ):
        raise ValueError("baseline audit has no accepted comparator registry")
    verified = audit_baseline_rows(accepted_rows, dataset_audit)
    if verified["rejected_count"]:
        raise ValueError("baseline audit contains invalid accepted baseline provenance")
    if verified["strongest_comparators"] != strongest:
        raise ValueError("baseline strongest comparator registry mismatch")
    return strongest, audit_sha256


def _strongest_value(
    strongest: Mapping[str, Any], dataset_id: str, split: str, metric: str
) -> float:
    try:
        row = strongest[dataset_id][split][metric]
        value = float(row["value"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"missing strongest comparator: {dataset_id}/{split}/{metric}"
        ) from error
    if not math.isfinite(value):
        raise ValueError(f"non-finite strongest comparator: {dataset_id}/{split}/{metric}")
    return value


def _primary_cohort_record(
    a0_rows: Sequence[Mapping[str, Any]],
    baseline_audit: Mapping[str, Any],
    audit_rows: Mapping[str, Any],
) -> dict[str, Any]:
    a0_by_dataset = _rows_by_dataset(a0_rows, row_kind="A0")
    audits, dataset_audit_sha256 = _verified_audit_records(audit_rows)
    strongest, comparator_audit_sha256 = _verified_strongest_comparators(
        baseline_audit, audit_rows
    )
    ranking: list[tuple[tuple[float, float, int, int], str]] = []
    fingerprints: dict[str, str] = {}
    for dataset_id, a0 in a0_by_dataset.items():
        audit = audits.get(dataset_id)
        if audit is None or not audit.get("eligible"):
            continue
        try:
            zero_margin = float(a0["a0_zero_auc"]) - _strongest_value(
                strongest, dataset_id, "holdout", "zero_auc"
            )
            overall_margin = min(
                float(a0["a0_standard_auc"])
                - _strongest_value(strongest, dataset_id, "standard", "auc"),
                float(a0["a0_holdout_auc"])
                - _strongest_value(strongest, dataset_id, "holdout", "auc"),
            )
            zero_count = int(audit["zero_count"])
            failed_attempt_count = int(a0["failed_attempt_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid primary cohort row: {dataset_id}") from error
        rank_key = (zero_margin, overall_margin, zero_count, -failed_attempt_count)
        if not all(math.isfinite(value) for value in (zero_margin, overall_margin)):
            raise ValueError(f"non-finite primary cohort metric: {dataset_id}")
        fingerprint = a0.get("a0_fingerprint")
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise ValueError(f"invalid A0 fingerprint: {dataset_id}")
        fingerprints[dataset_id] = fingerprint
        ranking.append((rank_key, dataset_id))
    ranking.sort(key=lambda item: item[0], reverse=True)
    if len(ranking) < 3:
        raise ValueError("primary cohort requires exactly three eligible datasets")
    dataset_ids = [dataset_id for _, dataset_id in ranking[:3]]
    cohort: dict[str, Any] = {
        "schema_version": 2,
        "dataset_ids": dataset_ids,
        "a0_fingerprints": {
            dataset_id: fingerprints[dataset_id] for dataset_id in dataset_ids
        },
        "comparator_audit_sha256": comparator_audit_sha256,
        "dataset_audit_sha256": dataset_audit_sha256,
        "rankings": [
            {"dataset_id": dataset_id, "rank_key": list(rank_key)}
            for rank_key, dataset_id in ranking
        ],
    }
    cohort["cohort_sha256"] = canonical_sha256(cohort)
    return cohort


def freeze_primary_cohort(
    a0_rows: Sequence[Mapping[str, Any]],
    comparator_rows: Mapping[str, Any],
    audit_rows: Mapping[str, Any],
    *,
    path: str | Path | None = None,
) -> dict[str, Any]:
    cohort = _primary_cohort_record(a0_rows, comparator_rows, audit_rows)
    if path is None:
        return cohort
    cohort_path = Path(path)
    cohort_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        cohort, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ) + "\n"
    try:
        with cohort_path.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        existing = load_verified_cohort(
            cohort_path,
            expected_dataset_ids=cohort["dataset_ids"],
            expected_a0_fingerprints=cohort["a0_fingerprints"],
            expected_comparator_audit_sha256=cohort[
                "comparator_audit_sha256"
            ],
            expected_dataset_audit_sha256=cohort["dataset_audit_sha256"],
        )
        _verify_existing(existing, cohort)
        return existing
    except BaseException:
        try:
            cohort_path.unlink()
        except OSError:
            pass
        raise
    _fsync_parent(cohort_path)
    return cohort


def verify_primary_cohort(
    a0_rows: Sequence[Mapping[str, Any]],
    comparator_rows: Mapping[str, Any],
    audit_rows: Mapping[str, Any],
    *,
    path: str | Path,
) -> dict[str, Any]:
    """Verify an existing primary cohort without mutating its path."""
    expected = _primary_cohort_record(a0_rows, comparator_rows, audit_rows)
    existing = load_verified_cohort(
        path,
        expected_dataset_ids=expected["dataset_ids"],
        expected_a0_fingerprints=expected["a0_fingerprints"],
        expected_comparator_audit_sha256=expected["comparator_audit_sha256"],
        expected_dataset_audit_sha256=expected["dataset_audit_sha256"],
    )
    _verify_existing(existing, expected)
    return existing


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _trusted_freeze(args: argparse.Namespace) -> dict[str, Any]:
    a0_payload = _load_json(args.a0_metrics)
    a0_rows = (
        a0_payload.get("rows")
        if isinstance(a0_payload, Mapping)
        else a0_payload
    )
    if not isinstance(a0_rows, list):
        raise ValueError("A0 metrics must be a row list or an object containing rows")
    baseline_audit = _load_json(args.baseline_audit)
    dataset_audit = _load_json(args.dataset_audit)
    if not isinstance(baseline_audit, Mapping) or not isinstance(
        dataset_audit, Mapping
    ):
        raise ValueError("trusted audits must be JSON objects")
    return freeze_primary_cohort(
        a0_rows,
        baseline_audit,
        dataset_audit,
        path=args.output if hasattr(args, "output") else args.cohort,
    )


def _trusted_verify(args: argparse.Namespace) -> dict[str, Any]:
    a0_payload = _load_json(args.a0_metrics)
    a0_rows = (
        a0_payload.get("rows")
        if isinstance(a0_payload, Mapping)
        else a0_payload
    )
    if not isinstance(a0_rows, list):
        raise ValueError("A0 metrics must be a row list or an object containing rows")
    baseline_audit = _load_json(args.baseline_audit)
    dataset_audit = _load_json(args.dataset_audit)
    if not isinstance(baseline_audit, Mapping) or not isinstance(
        dataset_audit, Mapping
    ):
        raise ValueError("trusted audits must be JSON objects")
    return verify_primary_cohort(
        a0_rows,
        baseline_audit,
        dataset_audit,
        path=args.cohort,
    )


def _print_cohort(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze or verify the Unified A0 cohort."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("freeze", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("--a0-metrics", type=Path, required=True)
        child.add_argument("--baseline-audit", type=Path, required=True)
        child.add_argument("--dataset-audit", type=Path, required=True)
        if command == "freeze":
            child.add_argument("--output", type=Path, required=True)
        else:
            child.add_argument("--cohort", type=Path, required=True)
        child.set_defaults(
            handler=_trusted_freeze if command == "freeze" else _trusted_verify
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _print_cohort(args.handler(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
