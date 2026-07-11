from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.unified_dataset_audit import canonical_sha256


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
    if len(normalized_ids) < 3:
        raise ValueError("primary cohort requires at least three datasets")
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


def _load_existing(path: Path) -> dict[str, Any]:
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"existing cohort is unreadable: {error}") from error
    if not isinstance(existing, dict):
        raise ValueError("existing cohort is not a JSON object")
    unhashed = dict(existing)
    stored_hash = unhashed.pop("cohort_sha256", None)
    if stored_hash != canonical_sha256(unhashed):
        raise ValueError("existing cohort canonical SHA-256 mismatch")
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
        existing = _load_existing(cohort_path)
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
