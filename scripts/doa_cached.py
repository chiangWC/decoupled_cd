from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.q_matrix import normalize_concept_sequence
from scripts.plugin_campaign import canonical_json_bytes, compute_frozen_config_id
from utils import compute_doa


_CACHE_COLUMNS = ["row_index", "stu_id", "exer_id", "cpt_seq", "label", "prob"]
_APPROVED_PROTOCOL = {
    "split": "valid",
    "seed": 42,
    "doa_seed": 42,
    "min_responses": 3,
    "max_pairs_per_concept": 100_000,
    "split_seed": 2024,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute final holdout DOA from the hash-bound cache emitted by a "
            "guarded plugin test evaluation. This command has no test/split input."
        )
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--evaluation-dir",
        action="append",
        required=True,
        help="Directory containing evaluation_cache_manifest.json and its artifacts.",
    )
    parser.add_argument("--model-name", action="append", required=True)
    parser.add_argument("--holdout-assignments", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def _read_bytes(path: Path) -> bytes:
    return Path(path).read_bytes()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validated_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _load_json_bytes(payload: bytes, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot decode {description} JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _validated_protocol(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("cache protocol must be a JSON object")
    required = (*_APPROVED_PROTOCOL, "q_matrix_sha256")
    missing = [field for field in required if field not in value]
    if missing:
        raise ValueError(f"cache protocol is missing: {', '.join(missing)}")
    for field, expected in _APPROVED_PROTOCOL.items():
        if value[field] != expected:
            raise ValueError(
                f"cache protocol {field} must be {expected!r}, got {value[field]!r}"
            )
    _validated_sha256(value["q_matrix_sha256"], "q_matrix_sha256")
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _verified_payload(
    *,
    path: Path,
    expected_sha256: Any,
    field: str,
) -> bytes:
    expected = _validated_sha256(expected_sha256, field)
    payload = _read_bytes(path)
    actual = _sha256_bytes(payload)
    if actual != expected:
        raise ValueError(
            f"{path.name} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return payload


def _token(value: Any) -> str:
    if pd.isna(value):
        raise ValueError("cached identifiers must not be missing")
    return str(value)


def _load_cache_frame(payload: bytes, *, row_count: int) -> pd.DataFrame:
    try:
        frame = pd.read_csv(
            io.BytesIO(payload),
            dtype={"stu_id": "string", "exer_id": "string", "cpt_seq": "string"},
        )
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        raise ValueError("cannot parse evaluation cache CSV") from exc
    if list(frame.columns) != _CACHE_COLUMNS:
        raise ValueError(
            f"evaluation cache columns must be exactly {_CACHE_COLUMNS!r}"
        )
    if len(frame) != row_count:
        raise ValueError(
            f"evaluation cache row count must be {row_count}, got {len(frame)}"
        )
    try:
        indices = pd.to_numeric(frame["row_index"], errors="raise").to_numpy(
            dtype=np.int64
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluation cache row_index must be integer") from exc
    if not np.array_equal(indices, np.arange(row_count, dtype=np.int64)):
        raise ValueError("evaluation cache row_index must be contiguous from zero")
    labels = pd.to_numeric(frame["label"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    probabilities = pd.to_numeric(frame["prob"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(labels).all() or not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("evaluation cache labels must be finite binary values")
    if not np.isfinite(probabilities).all() or not (
        (probabilities >= 0.0) & (probabilities <= 1.0)
    ).all():
        raise ValueError("evaluation cache probabilities must be finite and in [0, 1]")
    if frame[["stu_id", "exer_id"]].isna().any().any():
        raise ValueError("evaluation cache student/exercise IDs must not be missing")
    frame = frame.copy()
    frame["label"] = labels
    frame["prob"] = probabilities
    return frame


def _validated_id_maps(payload: bytes) -> dict[str, list[str]]:
    value = _load_json_bytes(payload, description="ID maps")
    normalized = {}
    for field in ("stu_ids", "exer_ids", "cpt_ids"):
        identifiers = value.get(field)
        if not isinstance(identifiers, list) or not identifiers:
            raise ValueError(f"ID maps require a non-empty {field} list")
        tokens = [str(identifier) for identifier in identifiers]
        if len(tokens) != len(set(tokens)):
            raise ValueError(f"ID maps contain duplicate {field}")
        normalized[field] = tokens
    return normalized


def _load_evaluation(evaluation_dir: Path) -> dict[str, Any]:
    evaluation_dir = Path(evaluation_dir)
    manifest_path = evaluation_dir / "evaluation_cache_manifest.json"
    manifest_bytes = _read_bytes(manifest_path)
    manifest = _load_json_bytes(manifest_bytes, description="evaluation cache manifest")
    if manifest.get("format_version") != 1:
        raise ValueError("evaluation cache format_version must be 1")
    if manifest.get("evaluation_split") != "test":
        raise ValueError("cached DOA requires evaluation_split='test'")
    for field, expected in {
        "cache_file": "evaluation_cache.csv",
        "mastery_file": "mastery.npy",
        "id_maps_file": "id_maps.json",
    }.items():
        if manifest.get(field) != expected:
            raise ValueError(f"cache manifest {field} must be {expected!r}")
    row_count = manifest.get("row_count")
    if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 0:
        raise ValueError("cache manifest row_count must be a non-negative integer")
    protocol = _validated_protocol(manifest.get("protocol"))
    checkpoint_sha256 = _validated_sha256(
        manifest.get("checkpoint_sha256"), "checkpoint_sha256"
    )
    source_id_maps_sha256 = _validated_sha256(
        manifest.get("source_id_maps_sha256"), "source_id_maps_sha256"
    )
    test_claim = manifest.get("test_claim")
    if not isinstance(test_claim, dict):
        raise ValueError("cache manifest requires a guarded test claim")
    claim_bytes = canonical_json_bytes(test_claim)
    test_claim_sha256 = _validated_sha256(
        manifest.get("test_claim_sha256"), "test_claim_sha256"
    )
    if _sha256_bytes(claim_bytes) != test_claim_sha256:
        raise ValueError("test claim SHA-256 does not match embedded claim")
    for field, expected in (
        ("checkpoint_sha256", checkpoint_sha256),
        ("id_maps_sha256", source_id_maps_sha256),
        ("plugin_config", manifest.get("plugin_config")),
        ("backbone_config", manifest.get("backbone_config")),
        ("protocol", protocol),
        ("frozen_config_id", manifest.get("frozen_config_id")),
        ("selection_sha256", manifest.get("selection_sha256")),
        ("dataset", manifest.get("dataset")),
        (
            "holdout_assignments_sha256",
            manifest.get("holdout_assignments_sha256"),
        ),
    ):
        if test_claim.get(field) != expected:
            raise ValueError(f"test claim {field} does not match cache manifest")
    frozen_config_id = compute_frozen_config_id(
        checkpoint_sha256=checkpoint_sha256,
        id_maps_sha256=source_id_maps_sha256,
        plugin_config=test_claim["plugin_config"],
        backbone_config=test_claim["backbone_config"],
        protocol=protocol,
    )
    if test_claim.get("frozen_config_id") != frozen_config_id:
        raise ValueError("test claim frozen_config_id is not canonical")
    selection_sha256 = _validated_sha256(
        test_claim.get("selection_sha256"), "selection_sha256"
    )
    dataset = test_claim.get("dataset")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("test claim dataset must be a non-empty string")
    holdout_assignments_sha256 = _validated_sha256(
        test_claim.get("holdout_assignments_sha256"),
        "holdout_assignments_sha256",
    )

    cache_path = evaluation_dir / "evaluation_cache.csv"
    mastery_path = evaluation_dir / "mastery.npy"
    id_maps_path = evaluation_dir / "id_maps.json"
    cache_bytes = _verified_payload(
        path=cache_path,
        expected_sha256=manifest.get("cache_sha256"),
        field="cache_sha256",
    )
    mastery_bytes = _verified_payload(
        path=mastery_path,
        expected_sha256=manifest.get("mastery_sha256"),
        field="mastery_sha256",
    )
    id_maps_bytes = _verified_payload(
        path=id_maps_path,
        expected_sha256=manifest.get("id_maps_sha256"),
        field="id_maps_sha256",
    )
    frame = _load_cache_frame(cache_bytes, row_count=row_count)
    id_maps = _validated_id_maps(id_maps_bytes)
    try:
        mastery = np.load(io.BytesIO(mastery_bytes), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("cannot load cached mastery.npy") from exc
    expected_shape = (len(id_maps["stu_ids"]), len(id_maps["cpt_ids"]))
    if mastery.shape != expected_shape:
        raise ValueError(
            f"cached mastery shape must be {expected_shape}, got {mastery.shape}"
        )
    if not np.isfinite(mastery).all():
        raise ValueError("cached mastery values must be finite")
    return {
        "manifest": manifest,
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "protocol": protocol,
        "checkpoint_sha256": checkpoint_sha256,
        "source_id_maps_sha256": source_id_maps_sha256,
        "frozen_config_id": frozen_config_id,
        "selection_sha256": selection_sha256,
        "test_claim_sha256": test_claim_sha256,
        "dataset": dataset,
        "holdout_assignments_sha256": holdout_assignments_sha256,
        "frame": frame,
        "mastery": mastery,
        "id_maps": id_maps,
    }


def _indexed_responses(
    frame: pd.DataFrame,
    id_maps: dict[str, list[str]],
) -> tuple[list[int], list[list[int]], list[float]]:
    stu2row = {token: index for index, token in enumerate(id_maps["stu_ids"])}
    exer_ids = set(id_maps["exer_ids"])
    cpt2col = {token: index for index, token in enumerate(id_maps["cpt_ids"])}
    student_ids = []
    concept_lists = []
    labels = []
    for index, row in enumerate(frame.itertuples(index=False)):
        student = _token(row.stu_id)
        exercise = _token(row.exer_id)
        if student not in stu2row:
            raise ValueError(f"cache row {index} has unknown student ID {student!r}")
        if exercise not in exer_ids:
            raise ValueError(f"cache row {index} has unknown exercise ID {exercise!r}")
        concepts = normalize_concept_sequence(row.cpt_seq)
        unknown = [concept for concept in concepts if concept not in cpt2col]
        if unknown:
            raise ValueError(
                f"cache row {index} has unknown concept ID {unknown[0]!r}"
            )
        student_ids.append(stu2row[student])
        concept_lists.append([cpt2col[concept] for concept in concepts])
        labels.append(float(row.label))
    return student_ids, concept_lists, labels


def _holdout_map(
    assignments: pd.DataFrame,
    id_maps: dict[str, list[str]],
) -> dict[int, set[int]]:
    required = {"stu_id", "holdout_concepts"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(
            f"holdout assignments are missing: {', '.join(sorted(missing))}"
        )
    stu2row = {token: index for index, token in enumerate(id_maps["stu_ids"])}
    cpt2col = {token: index for index, token in enumerate(id_maps["cpt_ids"])}
    result = {}
    for row in assignments.itertuples(index=False):
        student = _token(row.stu_id)
        student_index = stu2row.get(student)
        if student_index is None or pd.isna(row.holdout_concepts):
            continue
        concepts = normalize_concept_sequence(row.holdout_concepts)
        unknown = [concept for concept in concepts if concept not in cpt2col]
        if unknown:
            raise ValueError(
                f"holdout assignments have unknown concept ID {unknown[0]!r}"
            )
        held = {cpt2col[concept] for concept in concepts}
        if held:
            result[student_index] = held
    return result


def main() -> None:
    args = parse_args()
    if len(args.evaluation_dir) != len(args.model_name):
        raise ValueError("--evaluation-dir and --model-name counts must match")
    assignments_path = Path(args.holdout_assignments)
    assignments_bytes = _read_bytes(assignments_path)
    try:
        assignments = pd.read_csv(
            io.BytesIO(assignments_bytes),
            dtype={"stu_id": "string", "holdout_concepts": "string"},
        )
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        raise ValueError("cannot parse holdout assignments CSV") from exc
    assignments_sha256 = _sha256_bytes(assignments_bytes)

    rows = []
    for raw_dir, model_name in zip(
        args.evaluation_dir, args.model_name, strict=True
    ):
        evaluation = _load_evaluation(Path(raw_dir))
        if args.dataset_name != evaluation["dataset"]:
            raise ValueError(
                f"dataset must be {evaluation['dataset']!r}, got {args.dataset_name!r}"
            )
        if assignments_sha256 != evaluation["holdout_assignments_sha256"]:
            raise ValueError(
                "holdout assignments SHA-256 does not match frozen selection"
            )
        student_ids, concept_lists, labels = _indexed_responses(
            evaluation["frame"], evaluation["id_maps"]
        )
        protocol = evaluation["protocol"]
        result = {
            "dataset": args.dataset_name,
            "model": model_name,
            "split": "test",
            "seed": protocol["seed"],
            "doa_seed": protocol["doa_seed"],
            "min_responses": protocol["min_responses"],
            "max_pairs_per_concept": protocol["max_pairs_per_concept"],
            "split_seed": protocol["split_seed"],
            "q_matrix_sha256": protocol["q_matrix_sha256"],
            "checkpoint_sha256": evaluation["checkpoint_sha256"],
            "source_id_maps_sha256": evaluation["source_id_maps_sha256"],
            "frozen_config_id": evaluation["frozen_config_id"],
            "selection_sha256": evaluation["selection_sha256"],
            "test_claim_sha256": evaluation["test_claim_sha256"],
            "cache_manifest_sha256": evaluation["manifest_sha256"],
            "cache_sha256": evaluation["manifest"]["cache_sha256"],
            "mastery_sha256": evaluation["manifest"]["mastery_sha256"],
            "id_maps_sha256": evaluation["manifest"]["id_maps_sha256"],
            "holdout_assignments_sha256": assignments_sha256,
        }
        result.update(
            compute_doa(
                mastery=evaluation["mastery"],
                student_ids=student_ids,
                concept_lists=concept_lists,
                labels=labels,
                min_responses=protocol["min_responses"],
                max_pairs_per_concept=protocol["max_pairs_per_concept"],
                seed=protocol["doa_seed"],
            )
        )

        held_by_student = _holdout_map(assignments, evaluation["id_maps"])
        held_students = []
        held_concepts = []
        held_labels = []
        for student, concepts, label in zip(
            student_ids, concept_lists, labels, strict=True
        ):
            held = held_by_student.get(student)
            if not held:
                continue
            kept = [concept for concept in concepts if concept in held]
            if kept:
                held_students.append(student)
                held_concepts.append(kept)
                held_labels.append(label)
        holdout_result = compute_doa(
            mastery=evaluation["mastery"],
            student_ids=held_students,
            concept_lists=held_concepts,
            labels=held_labels,
            min_responses=protocol["min_responses"],
            max_pairs_per_concept=protocol["max_pairs_per_concept"],
            seed=protocol["doa_seed"],
        )
        result.update({f"holdout_{key}": value for key, value in holdout_result.items()})
        rows.append(result)
        print(result)

    pd.DataFrame(rows).to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
