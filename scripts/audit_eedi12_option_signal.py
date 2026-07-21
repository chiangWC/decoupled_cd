from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.pool_protocol import sha256_file
from scripts.audit_option_contrast_signal import (
    BOOTSTRAP_REPLICATES,
    LoadedProtocol,
    _target_mask,
    fast_student_cluster_bootstrap,
    run_dataset,
)


SPLIT_SEED = 2024
MODEL_SEED = 42
MIN_STUDENT_ROWS = 30
SELECTED_STUDENTS = 5_000
CHUNK_ROWS = 500_000
SOURCE_SHA256 = {
    "interactions": "721ebae1c5ddb3f8a4c85a437893216bbba1d8b2ca950ee0681d1f3e98ebdc0e",
    "questions": "673aabe79e8dba2cf82e4bf87221796f2672c1ad264c9e08e2a99bf235704a12",
    "subjects": "d576a6eccc171d8eb82a284a9586c27b2bd9941f39f3fcd6ae1110470766f202",
    "enem_summary": "0f255d1383696f0c96eb7ad5caf6ceab90f798bd84a3b6c66f05bb15ada8d1aa",
}
SOURCE_COLUMNS = (
    "QuestionId",
    "UserId",
    "AnswerId",
    "IsCorrect",
    "CorrectAnswer",
    "AnswerValue",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered Eedi Tasks 1&2 option-signal extension."
    )
    parser.add_argument("--interactions", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--subjects", required=True)
    parser.add_argument("--enem-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=BOOTSTRAP_REPLICATES,
    )
    args = parser.parse_args()
    if args.bootstrap_replicates != BOOTSTRAP_REPLICATES:
        raise ValueError("The formal extension fixes 2,000 bootstrap replicates.")
    return args


def stable_student_key(raw_user_id: int) -> str:
    payload = f"{SPLIT_SEED}\x1feedi12-student\x1f{raw_user_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def stable_row_id(raw_user_id: int, raw_question_id: int) -> str:
    payload = f"eedi12\x1f{raw_user_id}\x1f{raw_question_id}".encode()
    return hashlib.sha256(payload).hexdigest()[:32]


def hash_values(values: Any) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def verify_input(path: Path, key: str) -> str:
    digest = sha256_file(path)
    expected = SOURCE_SHA256[key]
    if digest != expected:
        raise RuntimeError(f"Source hash mismatch for {path}: {digest} != {expected}")
    return digest


def repository_provenance() -> dict[str, str]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError("Formal Eedi signal audit requires a clean worktree.")
    return {
        "git_commit": head,
        "script_sha256": sha256_file(Path(__file__)),
        "shared_signal_script_sha256": sha256_file(
            PROJECT_ROOT / "scripts/audit_option_contrast_signal.py"
        ),
    }


def select_students(interactions_path: Path) -> tuple[list[int], dict[str, Any]]:
    counts: Counter[int] = Counter()
    total_rows = 0
    for chunk in pd.read_csv(
        interactions_path,
        usecols=["UserId"],
        chunksize=CHUNK_ROWS,
    ):
        total_rows += len(chunk)
        counts.update(
            {
                int(user): int(count)
                for user, count in chunk["UserId"].value_counts().items()
            }
        )
    eligible = [user for user, count in counts.items() if count >= MIN_STUDENT_ROWS]
    if len(eligible) < SELECTED_STUDENTS:
        raise RuntimeError(f"Only {len(eligible)} students satisfy the frozen minimum.")
    selected = sorted(eligible, key=lambda user: (stable_student_key(user), user))[
        :SELECTED_STUDENTS
    ]
    return selected, {
        "source_rows": total_rows,
        "source_students": len(counts),
        "eligible_students": len(eligible),
        "minimum_student_rows": MIN_STUDENT_ROWS,
        "selected_students": len(selected),
        "selection_key_uses_label_correctness_question_or_option": False,
        "selected_raw_user_sha256": hash_values(sorted(selected)),
        "selected_key_order_sha256": hash_values(selected),
        "selected_source_rows_expected": int(sum(counts[user] for user in selected)),
    }


def load_selected_rows(
    interactions_path: Path,
    selected_students: list[int],
) -> pd.DataFrame:
    selected = set(selected_students)
    chunks = []
    for chunk in pd.read_csv(
        interactions_path,
        usecols=list(SOURCE_COLUMNS),
        chunksize=CHUNK_ROWS,
    ):
        keep = chunk["UserId"].isin(selected)
        if bool(keep.any()):
            chunks.append(chunk.loc[keep, list(SOURCE_COLUMNS)].copy())
    if not chunks:
        raise RuntimeError("Stable student cohort contains no source rows.")
    frame = pd.concat(chunks, ignore_index=True)
    if frame["UserId"].nunique() != SELECTED_STUDENTS:
        raise RuntimeError("Not every selected student was recovered from the source.")
    if frame.duplicated(["UserId", "QuestionId"]).any():
        raise RuntimeError("Eedi cohort unexpectedly contains duplicate student-question rows.")
    for column in SOURCE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")
    valid_options = frame["AnswerValue"].between(1, 4) & frame[
        "CorrectAnswer"
    ].between(1, 4)
    if not bool(valid_options.all()):
        raise RuntimeError("Eedi cohort has missing or out-of-range option values.")
    expected_label = (frame["AnswerValue"] == frame["CorrectAnswer"]).astype(int)
    if not expected_label.equals(frame["IsCorrect"].astype(int)):
        raise RuntimeError("Eedi cohort option identity disagrees with IsCorrect.")
    return frame


def parse_subjects(value: object) -> tuple[int, ...]:
    text = str(value).strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError(f"Invalid Eedi SubjectId value: {value!r}")
    values = tuple(int(token.strip()) for token in text[1:-1].split(",") if token.strip())
    if not values:
        raise ValueError("Eedi question has no subjects.")
    return tuple(sorted(set(values)))


def build_protocol(
    source: pd.DataFrame,
    question_path: Path,
    subject_path: Path,
    *,
    input_audit: dict[str, Any],
) -> tuple[LoadedProtocol, dict[str, Any]]:
    questions = pd.read_csv(question_path, usecols=["QuestionId", "SubjectId"])
    subjects = pd.read_csv(subject_path, encoding="utf-8-sig")
    if questions["QuestionId"].duplicated().any():
        raise RuntimeError("Eedi question metadata contains duplicate QuestionId rows.")
    official_subjects = set(pd.to_numeric(subjects["SubjectId"]).astype(int))
    raw_q = {
        int(row.QuestionId): parse_subjects(row.SubjectId)
        for row in questions.itertuples(index=False)
    }
    cohort_questions = set(source["QuestionId"].astype(int))
    if not cohort_questions.issubset(raw_q):
        raise RuntimeError("Selected cohort contains questions absent from metadata.")
    used_subjects = sorted(
        {subject for question in cohort_questions for subject in raw_q[question]}
    )
    if not set(used_subjects).issubset(official_subjects):
        raise RuntimeError("Question metadata references an unknown subject.")

    raw_users = sorted(source["UserId"].astype(int).unique())
    raw_questions = sorted(cohort_questions)
    user_map = {value: index for index, value in enumerate(raw_users)}
    question_map = {value: index for index, value in enumerate(raw_questions)}
    subject_map = {value: index for index, value in enumerate(used_subjects)}
    concept_sequence = {
        raw_question: ",".join(
            str(subject_map[subject]) for subject in raw_q[raw_question]
        )
        for raw_question in raw_questions
    }

    core = pd.DataFrame(
        {
            "source_row_id": [
                stable_row_id(int(user), int(question))
                for user, question in source[["UserId", "QuestionId"]].itertuples(
                    index=False, name=None
                )
            ],
            "stu_id": source["UserId"].map(user_map).astype(int),
            "exer_id": source["QuestionId"].map(question_map).astype(int),
            "cpt_seq": source["QuestionId"].map(concept_sequence),
            "label": source["IsCorrect"].astype(int),
            "selected_option": source["AnswerValue"].astype(int) - 1,
            "correct_option": source["CorrectAnswer"].astype(int) - 1,
            "option_count": 4,
        }
    )
    if core.isna().any().any() or core["source_row_id"].duplicated().any():
        raise RuntimeError("Dense Eedi cohort construction is incomplete or non-unique.")
    q_matrix = pd.DataFrame(
        {
            "exer_id": [question_map[value] for value in raw_questions],
            "cpt_seq": [concept_sequence[value] for value in raw_questions],
        }
    )
    q_lookup = {
        str(row.exer_id): tuple(str(row.cpt_seq).split(","))
        for row in q_matrix.itertuples(index=False)
    }
    cohort_audit = {
        "rows": int(len(core)),
        "students": int(core["stu_id"].nunique()),
        "questions": int(core["exer_id"].nunique()),
        "concepts": len(used_subjects),
        "option_coverage": 1.0,
        "option_label_agreement": 1.0,
        "duplicate_student_question_rows": 0,
        "row_identity_sha256": hash_values(sorted(core["source_row_id"])),
        "dense_user_map_sha256": hash_values(f"{raw}:{dense}" for raw, dense in user_map.items()),
        "dense_question_map_sha256": hash_values(
            f"{raw}:{dense}" for raw, dense in question_map.items()
        ),
        "dense_subject_map_sha256": hash_values(
            f"{raw}:{dense}" for raw, dense in subject_map.items()
        ),
        "q_mapping_sha256": hash_values(
            f"{row.exer_id}:{row.cpt_seq}" for row in q_matrix.itertuples(index=False)
        ),
        "all_official_subject_ids_used_without_hierarchy": True,
    }
    protocol = LoadedProtocol(
        name="eedi12",
        train=core,
        q_matrix=q_matrix,
        q_lookup=q_lookup,
        target_scope="low_coverage",
        input_audit={**input_audit, "cohort": cohort_audit},
    )
    return protocol, cohort_audit


def main() -> None:
    args = parse_args()
    interactions = Path(args.interactions)
    questions = Path(args.questions)
    subjects = Path(args.subjects)
    enem_summary_path = Path(args.enem_summary)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("Formal Eedi signal output directory must be empty.")
    output_dir.mkdir(parents=True, exist_ok=True)

    provenance = repository_provenance()
    source_hashes = {
        "interactions": verify_input(interactions, "interactions"),
        "questions": verify_input(questions, "questions"),
        "subjects": verify_input(subjects, "subjects"),
        "enem_summary": verify_input(enem_summary_path, "enem_summary"),
    }
    selected, selection_audit = select_students(interactions)
    source = load_selected_rows(interactions, selected)
    if len(source) != selection_audit["selected_source_rows_expected"]:
        raise RuntimeError("Recovered cohort size disagrees with the selection scan.")
    input_audit = {
        "source_hashes": source_hashes,
        "selection": selection_audit,
        "opened_files": [
            str(interactions.resolve()),
            str(questions.resolve()),
            str(subjects.resolve()),
        ],
        "validation_or_test_files_opened": False,
        "answer_metadata_opened": False,
    }
    protocol, cohort_audit = build_protocol(
        source,
        questions,
        subjects,
        input_audit=input_audit,
    )
    eedi_summary, predictions = run_dataset(protocol, output_dir=output_dir)
    bootstrap = None
    if eedi_summary.get("deterministic_dataset_pass"):
        target = _target_mask(predictions, eedi_summary["target_scope"])
        control = eedi_summary["stronger_control"]
        bootstrap = fast_student_cluster_bootstrap(
            labels=predictions.loc[target, "label"].to_numpy(dtype=int),
            students=predictions.loc[target, "stu_id"].astype(str).to_numpy(),
            full_probability=predictions.loc[target, "prob_full"].to_numpy(dtype=float),
            control_probability=predictions.loc[
                target, f"prob_{control}"
            ].to_numpy(dtype=float),
            replicates=args.bootstrap_replicates,
            seed=SPLIT_SEED,
        )
        eedi_summary["target_student_cluster_bootstrap"] = bootstrap
        (output_dir / "eedi12_summary.json").write_text(
            json.dumps(eedi_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    enem_summary = json.loads(enem_summary_path.read_text(encoding="utf-8"))
    enem_passed = bool(enem_summary.get("deterministic_dataset_pass"))
    eedi_passed = bool(
        eedi_summary.get("deterministic_dataset_pass")
        and bootstrap is not None
        and bootstrap["ci_low"] > 0.0
    )
    aggregate = {
        "schema_version": 1,
        "route": "eedi12_option_signal_extension_v1",
        "repository_provenance": provenance,
        "source_hashes": source_hashes,
        "cohort": cohort_audit,
        "eedi12": eedi_summary,
        "frozen_enem": {
            "summary_path": str(enem_summary_path.resolve()),
            "summary_sha256": source_hashes["enem_summary"],
            "deterministic_dataset_pass": enem_passed,
            "stronger_control": enem_summary.get("stronger_control"),
            "deltas_full_minus_control": enem_summary.get(
                "deltas_full_minus_control"
            ),
        },
        "eedi12_passed": eedi_passed,
        "enem_passed": enem_passed,
        "two_dataset_option_signal": bool(eedi_passed and enem_passed),
        "module_implementation_activated": bool(eedi_passed and enem_passed),
        "model_seed": MODEL_SEED,
        "split_seed": SPLIT_SEED,
        "no_multi_seed": True,
    }
    aggregate_path = output_dir / "aggregate.json"
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "eedi12_passed": eedi_passed,
                "enem_passed": enem_passed,
                "two_dataset_option_signal": aggregate[
                    "two_dataset_option_signal"
                ],
                "eedi12_signal_eligible": eedi_summary.get(
                    "signal_eligible", True
                ),
                "eedi12_control": eedi_summary.get("stronger_control"),
                "eedi12_deltas": eedi_summary.get(
                    "deltas_full_minus_control"
                ),
                "eedi12_bootstrap": bootstrap,
                "output": str(aggregate_path.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
