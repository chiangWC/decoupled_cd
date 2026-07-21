from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
from zipfile import ZipFile
import zlib

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.pool_protocol import (
    assign_atomic_holdout_split,
    assign_atomic_standard_split,
    canonicalize_interactions,
    derive_q_matrix,
    sha256_file,
    write_dataset_variant,
)
from scripts.audit_static_metadata_admission import reconstruct_nips


FROZEN_SPLIT_SEED = 2024
FROZEN_EDNET_STUDENTS = 5_000
FROZEN_MIN_STUDENT_INTERACTIONS = 30
FROZEN_SOURCE_SHA256 = {
    "nips_interactions": (
        "8bdbe55a310641f9e59caffbff2eac85da0b2f6c6b2bb99fa69922296f52a4e1"
    ),
    "nips_questions": (
        "44204e3450d0a2298fef0d940677fcd7c373b43f994cd5fec1d7381aff95001b"
    ),
    "nips_subjects": (
        "d576a6eccc171d8eb82a284a9586c27b2bd9941f39f3fcd6ae1110470766f202"
    ),
    "ednet_interactions": (
        "f69582c9c731215c5c7c2024641aaf55982914bbe3d4dd83b7c35152a3261f44"
    ),
    "ednet_questions": (
        "137d2936b746c98e1586ef134bf9f9bc4b5a878c96b88e630b2241fe9f15d3cd"
    ),
    "ednet_archive": (
        "0d13933f90201c5101c7fe8659e44474fa049e3fb93181a8ba6fb3e63267b535"
    ),
    "enem_archive": (
        "b1180eccce79df5a4dae25a5a0d66e62b60f3351c0a61817c8639a2ce5553967"
    ),
}
FROZEN_NIPS_PROTOCOL_SHA256 = {
    "standard": {
        "data.csv": "b1259ecfc3a49d0f40bd988d2f57bbff7684c73d915826c81b64cb9f829acaf8",
        "train.csv": "e54b4f302d19c00c41d9ff2d69c9152145a7bd59d5c43ffa411b47ce90f36df5",
        "valid.csv": "391f29f69f894fe342c0246cea24a119feb2c5143e30f5d61cf3470671be0838",
        "test.csv": "84ec6efffa7d428b8fdd4466f5b761d31d4e198f869177c4d2eeafaac23fd04b",
        "Q_matrix.csv": "cb11593103b55b184fc4e1eaea41449c49fce0b4783f5ed44563c23ca818088c",
    },
    "holdout": {
        "data.csv": "b1259ecfc3a49d0f40bd988d2f57bbff7684c73d915826c81b64cb9f829acaf8",
        "train.csv": "64c8d02067cffe87ec8b1b1b747604425a5d1223d9097c9e904c61350b4312d7",
        "valid.csv": "0ba37ad102168651e1d405492ee0186a6e2ebf295c49188b582b4df1450e0ce4",
        "test.csv": "7e40f50bce6e7bc60c1125ff98d18c69818a59d7973eef53fbb56015407e50c3",
        "Q_matrix.csv": "cb11593103b55b184fc4e1eaea41449c49fce0b4783f5ed44563c23ca818088c",
    },
}
OPTION_COLUMNS = (
    "source_row_id",
    "stu_id",
    "exer_id",
    "selected_option",
    "correct_option",
    "option_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare provenance-locked option-aware CD protocols."
    )
    parser.add_argument(
        "--dataset",
        choices=("all", "nips34", "ednet", "enem"),
        default="all",
    )
    parser.add_argument("--nips-standard", required=True)
    parser.add_argument("--nips-holdout", required=True)
    parser.add_argument("--nips-interactions", required=True)
    parser.add_argument("--nips-questions", required=True)
    parser.add_argument("--nips-subjects", required=True)
    parser.add_argument("--ednet-interactions", required=True)
    parser.add_argument("--ednet-questions", required=True)
    parser.add_argument("--ednet-archive", required=True)
    parser.add_argument("--ednet-kt1-dir", required=True)
    parser.add_argument("--enem-archive", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split-seed", type=int, default=2024)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stable_key(seed: int, dataset: str, value: object) -> str:
    payload = f"{seed}\x1f{dataset}\x1f{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def label_free_row_id(student: object, exercise: object) -> str:
    payload = f"option-v1\x1f{student}\x1f{exercise}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def repository_provenance() -> dict[str, str]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(
            "Formal option protocol generation requires a clean committed worktree."
        )
    return {
        "git_commit": head,
        "prepare_script_sha256": sha256_file(Path(__file__)),
    }


def require_hash(path: str | Path, key: str) -> str:
    digest = sha256_file(path)
    expected = FROZEN_SOURCE_SHA256[key]
    if digest != expected:
        raise RuntimeError(
            f"Frozen source mismatch for {path}: {digest} != {expected}"
        )
    return digest


def normalize_ednet_records(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame[
        ["timestamp", "question_id", "user_answer"]
    ].copy()
    output["timestamp"] = pd.to_numeric(
        output["timestamp"], errors="raise"
    ).astype("int64")
    output["question_id"] = output["question_id"].astype(str)
    output["user_answer"] = (
        output["user_answer"].astype(str).str.lower()
    )
    return output.sort_values(
        ["timestamp", "question_id", "user_answer"],
        kind="stable",
    ).reset_index(drop=True)


def file_crc32(path: Path) -> tuple[int, int]:
    checksum = 0
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum = zlib.crc32(chunk, checksum)
            size += len(chunk)
    return checksum & 0xFFFFFFFF, size


def verify_ednet_snapshot(
    snapshot: pd.DataFrame,
    *,
    students: list[object],
    kt1_dir: str | Path,
    archive: str | Path,
) -> dict[str, Any]:
    grouped = snapshot.groupby("user_id", sort=False)
    digest = hashlib.sha256()
    archive_digest = hashlib.sha256()
    official_rows = 0
    with ZipFile(archive) as source_archive:
        for student in sorted(students, key=lambda value: int(value)):
            member_name = f"KT1/u{int(student)}.csv"
            member = source_archive.getinfo(member_name)
            official_path = Path(kt1_dir) / f"u{int(student)}.csv"
            extracted_crc, extracted_size = file_crc32(official_path)
            if extracted_crc != member.CRC or extracted_size != member.file_size:
                raise RuntimeError(
                    f"Extracted EdNet file differs from verified archive: {member_name}."
                )
            official = pd.read_csv(
                official_path,
                usecols=["timestamp", "question_id", "user_answer"],
            )
            official = normalize_ednet_records(official)
            frozen = normalize_ednet_records(grouped.get_group(student))
            if not official.equals(frozen):
                raise RuntimeError(
                    f"Frozen EdNet snapshot differs from official KT1 for user {student}."
                )
            official_rows += len(official)
            digest.update(str(int(student)).encode("utf-8"))
            digest.update(
                pd.util.hash_pandas_object(official, index=False).values.tobytes()
            )
            archive_digest.update(member_name.encode("utf-8"))
            archive_digest.update(str(member.CRC).encode("ascii"))
            archive_digest.update(str(member.file_size).encode("ascii"))
    return {
        "rows": int(official_rows),
        "canonical_sha256": digest.hexdigest(),
        "archive_member_binding_sha256": archive_digest.hexdigest(),
        "verified_archive_members": int(len(students)),
    }


def replace_directory(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to rebuild it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_tags(value: object) -> tuple[int, ...]:
    if pd.isna(value):
        return ()
    return tuple(
        sorted(
            {
                int(token.strip())
                for token in str(value).split(";")
                if token.strip()
            }
        )
    )


def format_concepts(values: list[int] | tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


def canonicalize_option_rows(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "stu_id",
        "exer_id",
        "cpt_seq",
        "label",
        "selected_option",
        "correct_option",
        "option_count",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Option frame is missing columns: {sorted(missing)}")
    if frame.duplicated(["stu_id", "exer_id"]).any():
        raise RuntimeError("Option protocols require one row per student-exercise pair.")
    for column in ("label", "selected_option", "correct_option", "option_count"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    expected_label = (frame["selected_option"] == frame["correct_option"]).astype(int)
    if not expected_label.equals(frame["label"].reset_index(drop=True)):
        raise RuntimeError("Binary label is inconsistent with selected/correct options.")
    valid = (
        frame["option_count"].between(2, 7)
        & frame["selected_option"].ge(0)
        & frame["correct_option"].ge(0)
        & frame["selected_option"].lt(frame["option_count"])
        & frame["correct_option"].lt(frame["option_count"])
    )
    if not bool(valid.all()):
        raise RuntimeError(f"Invalid option rows: {int((~valid).sum())}")

    core, removed = canonicalize_interactions(frame)
    if removed:
        raise RuntimeError(f"Unexpected duplicate canonical interactions: {removed}")
    core["source_row_id"] = [
        label_free_row_id(student, exercise)
        for student, exercise in core[["stu_id", "exer_id"]].itertuples(
            index=False, name=None
        )
    ]
    if core["source_row_id"].duplicated().any():
        raise RuntimeError("Label-free source-row identity collision detected.")
    option_values = frame[
        [
            "stu_id",
            "exer_id",
            "selected_option",
            "correct_option",
            "option_count",
        ]
    ].copy()
    option_values["stu_id"] = option_values["stu_id"].astype(int).astype(str)
    option_values["exer_id"] = option_values["exer_id"].astype(int).astype(str)
    options = core[["source_row_id", "stu_id", "exer_id"]].merge(
        option_values,
        on=["stu_id", "exer_id"],
        how="left",
        validate="one_to_one",
    )
    if options[list(OPTION_COLUMNS[3:])].isna().any().any():
        raise RuntimeError("Option sidecar failed to align with canonical rows.")
    for column in OPTION_COLUMNS[3:]:
        options[column] = options[column].astype(int)
    return core, options.loc[:, OPTION_COLUMNS]


def write_option_variants(
    *,
    dataset: str,
    core: pd.DataFrame,
    options: pd.DataFrame,
    output_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    q_matrix, conflicts = derive_q_matrix(core)
    if conflicts:
        raise RuntimeError(f"{dataset} has {conflicts} exercise-to-Q conflicts.")
    standard_dir = output_root / f"{dataset}_option_v1"
    holdout_dir = output_root / f"{dataset}_option_chold_v1"
    audit_dir = output_root / f"{dataset}_option_source_v1"
    replace_directory(standard_dir, overwrite=overwrite)
    replace_directory(holdout_dir, overwrite=overwrite)
    replace_directory(audit_dir, overwrite=overwrite)
    audit_option_path = audit_dir / "audit_options.csv"
    options.to_csv(audit_option_path, index=False)
    standard_splits = assign_atomic_standard_split(core, seed=FROZEN_SPLIT_SEED)
    holdout_splits, assignments = assign_atomic_holdout_split(
        core,
        seed=FROZEN_SPLIT_SEED,
    )
    reports: dict[str, Any] = {}
    for name, directory, splits, split_assignments in (
        ("standard", standard_dir, standard_splits, None),
        ("holdout", holdout_dir, holdout_splits, assignments),
    ):
        report = write_dataset_variant(
            directory,
            full_frame=core,
            splits=splits,
            q_matrix=q_matrix,
            assignments=split_assignments,
        )
        assignment_path = directory / "student_concept_holdout_assignments.csv"
        if split_assignments is not None:
            report["files"][assignment_path.name] = {
                "sha256": sha256_file(assignment_path),
                "rows": int(len(split_assignments)),
            }
        train_ids = set(splits["train"]["source_row_id"].astype(str))
        train_options = options.loc[
            options["source_row_id"].astype(str).isin(train_ids)
        ].copy()
        if len(train_options) != len(splits["train"]):
            raise RuntimeError(
                f"{dataset} {name} train option sidecar is not row-exact."
            )
        train_option_path = directory / "train_options.csv"
        train_options.to_csv(train_option_path, index=False)
        report["files"]["train_options.csv"] = {
            "sha256": sha256_file(train_option_path),
            "rows": int(len(train_options)),
        }
        reports[name] = report
    return {
        "q_mapping_conflicts": conflicts,
        "students": int(core["stu_id"].nunique()),
        "items": int(core["exer_id"].nunique()),
        "concepts": int(
            core["cpt_seq"].str.split(",").explode().nunique()
        ),
        "rows": int(len(core)),
        "option_file": {
            "path": str(audit_option_path.resolve()),
            "sha256": sha256_file(audit_option_path),
            "rows": int(len(options)),
            "training_access": False,
        },
        **reports,
    }


def write_audit_artifacts(
    directory: Path,
    artifacts: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for filename, frame in artifacts.items():
        path = directory / filename
        frame.to_csv(path, index=False)
        reports[filename] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "rows": int(len(frame)),
            "training_access": False,
        }
    return reports


def prepare_nips(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    source_hashes = {
        "interactions": require_hash(args.nips_interactions, "nips_interactions"),
        "questions": require_hash(args.nips_questions, "nips_questions"),
        "subjects": require_hash(args.nips_subjects, "nips_subjects"),
    }
    protocol_dirs = {
        "standard": Path(args.nips_standard),
        "holdout": Path(args.nips_holdout),
    }
    protocol_hashes: dict[str, dict[str, str]] = {}
    for variant, directory in protocol_dirs.items():
        protocol_hashes[variant] = {}
        for filename, expected in FROZEN_NIPS_PROTOCOL_SHA256[variant].items():
            digest = sha256_file(directory / filename)
            if digest != expected:
                raise RuntimeError(
                    f"NIPS {variant}/{filename} drifted: {digest} != {expected}"
                )
            protocol_hashes[variant][filename] = digest
    identity, _ = reconstruct_nips(
        args.nips_questions,
        args.nips_subjects,
        args.nips_standard,
    )
    if not identity["identity_exact"]:
        raise RuntimeError(f"NIPS Q provenance failed: {identity}")

    raw = pd.read_csv(
        args.nips_interactions,
        usecols=[
            "UserId",
            "QuestionId",
            "AnswerValue",
            "CorrectAnswer",
            "IsCorrect",
        ],
    )
    if raw.duplicated(["UserId", "QuestionId"]).any():
        raise RuntimeError("NIPS source is not unique by student-question.")
    students = sorted(raw["UserId"].unique())
    student_map = {value: index for index, value in enumerate(students)}
    mapped = pd.DataFrame(
        {
            "stu_id": raw["UserId"].map(student_map).astype(int),
            "exer_id": raw["QuestionId"].astype(int),
            "label": raw["IsCorrect"].astype(int),
            "selected_option": raw["AnswerValue"].astype(int) - 1,
            "correct_option": raw["CorrectAnswer"].astype(int) - 1,
            "option_count": 4,
        }
    )
    current = pd.read_csv(Path(args.nips_standard) / "data.csv")
    raw_rows = Counter(
        mapped[["stu_id", "exer_id", "label"]].itertuples(
            index=False, name=None
        )
    )
    current_rows = Counter(
        current[["stu_id", "exer_id", "label"]].itertuples(
            index=False, name=None
        )
    )
    if raw_rows != current_rows:
        raise RuntimeError("NIPS raw interactions do not reconstruct current data.csv.")
    binary_consistency = (
        (mapped["selected_option"] == mapped["correct_option"]).astype(int)
        == mapped["label"]
    )
    if not bool(binary_consistency.all()):
        raise RuntimeError("NIPS per-row answer and correctness fields disagree.")
    audit_dir = output_root / "nips34_option_source_v1"
    train_option_dir = output_root / "nips34_option_train_v1"
    replace_directory(audit_dir, overwrite=args.overwrite)
    replace_directory(train_option_dir, overwrite=args.overwrite)
    options = mapped[
        [
            "stu_id",
            "exer_id",
            "selected_option",
            "correct_option",
            "option_count",
        ]
    ].sort_values(["stu_id", "exer_id"])
    audit_option_path = audit_dir / "audit_options.csv"
    options.to_csv(audit_option_path, index=False)
    train_option_files: dict[str, Any] = {}
    for variant, directory in protocol_dirs.items():
        train_keys = pd.read_csv(
            directory / "train.csv",
            usecols=["stu_id", "exer_id"],
        ).drop_duplicates()
        train_options = train_keys.merge(
            options,
            on=["stu_id", "exer_id"],
            how="left",
            validate="one_to_one",
        )
        if (
            len(train_options) != len(train_keys)
            or train_options["selected_option"].isna().any()
        ):
            raise RuntimeError(
                f"NIPS {variant} train option sidecar is not row-exact."
            )
        train_option_path = train_option_dir / f"{variant}_train_options.csv"
        train_options.to_csv(train_option_path, index=False)
        train_option_files[variant] = {
            "path": str(train_option_path.resolve()),
            "sha256": sha256_file(train_option_path),
            "rows": int(len(train_options)),
        }
    return {
        "source_hashes": source_hashes,
        "protocol_hashes": protocol_hashes,
        "source_to_protocol_identity_exact": True,
        "q_identity": identity,
        "rows": int(len(options)),
        "students": int(options["stu_id"].nunique()),
        "items": int(options["exer_id"].nunique()),
        "option_file": {
            "path": str(audit_option_path.resolve()),
            "sha256": sha256_file(audit_option_path),
            "rows": int(len(options)),
            "training_access": False,
        },
        "train_option_files": train_option_files,
        "standard_dir": str(protocol_dirs["standard"].resolve()),
        "holdout_dir": str(protocol_dirs["holdout"].resolve()),
        "target_scope": "low_coverage",
        "correct_answer_policy": (
            "per-interaction raw CorrectAnswer; four version-conflict items are retained"
        ),
    }


def prepare_ednet(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    source_hashes = {
        "interactions": require_hash(args.ednet_interactions, "ednet_interactions"),
        "questions": require_hash(args.ednet_questions, "ednet_questions"),
        "official_archive": require_hash(args.ednet_archive, "ednet_archive"),
    }
    raw = pd.read_csv(
        args.ednet_interactions,
        usecols=[
            "user_id",
            "timestamp",
            "question_id",
            "user_answer",
        ],
        low_memory=False,
    )
    questions = pd.read_csv(
        args.ednet_questions,
        usecols=["question_id", "correct_answer", "tags"],
        low_memory=False,
    )
    snapshot = raw.copy()
    source_rows = len(raw)
    raw["question_id"] = raw["question_id"].astype(str)
    questions["question_id"] = questions["question_id"].astype(str)
    raw = raw.merge(questions, on="question_id", how="left", validate="many_to_one")
    raw["user_answer"] = raw["user_answer"].astype(str).str.lower()
    raw["correct_answer"] = raw["correct_answer"].astype(str).str.lower()
    raw["tags_parsed"] = raw["tags"].map(parse_tags)
    valid = (
        raw["user_answer"].isin(list("abcd"))
        & raw["correct_answer"].isin(list("abcd"))
        & raw["tags_parsed"].map(bool)
        & raw["timestamp"].notna()
    )
    invalid_rows = int((~valid).sum())
    raw = raw.loc[valid].copy()
    raw["timestamp"] = pd.to_numeric(raw["timestamp"], errors="raise")
    raw = raw.sort_values(
        ["user_id", "timestamp", "question_id"],
        kind="stable",
    )
    before_first = len(raw)
    raw = raw.drop_duplicates(["user_id", "question_id"], keep="first")
    repeated_attempts_removed = before_first - len(raw)
    counts = raw.groupby("user_id").size()
    eligible = counts[counts >= FROZEN_MIN_STUDENT_INTERACTIONS].index.tolist()
    selected_students = sorted(
        eligible,
        key=lambda value: stable_key(
            FROZEN_SPLIT_SEED, "ednet-student", value
        ),
    )[:FROZEN_EDNET_STUDENTS]
    if len(selected_students) != FROZEN_EDNET_STUDENTS:
        raise RuntimeError(
            f"EdNet has only {len(selected_students)} eligible students."
        )
    official_identity = verify_ednet_snapshot(
        snapshot,
        students=selected_students,
        kt1_dir=args.ednet_kt1_dir,
        archive=args.ednet_archive,
    )
    raw = raw.loc[raw["user_id"].isin(selected_students)].copy()
    raw = raw.reset_index(drop=True)
    student_values = sorted(raw["user_id"].unique(), key=lambda value: int(value))
    item_values = sorted(
        raw["question_id"].unique(),
        key=lambda value: int(str(value).removeprefix("q")),
    )
    concepts = sorted(
        {concept for values in raw["tags_parsed"] for concept in values}
    )
    student_map = {value: index for index, value in enumerate(student_values)}
    item_map = {value: index for index, value in enumerate(item_values)}
    concept_map = {value: index for index, value in enumerate(concepts)}
    answer_map = {value: index for index, value in enumerate("abcd")}
    frame = pd.DataFrame(
        {
            "stu_id": raw["user_id"].map(student_map),
            "exer_id": raw["question_id"].map(item_map),
            "cpt_seq": raw["tags_parsed"].map(
                lambda values: format_concepts(
                    [concept_map[value] for value in values]
                )
            ),
            "label": (
                raw["user_answer"] == raw["correct_answer"]
            ).astype(int),
            "selected_option": raw["user_answer"].map(answer_map),
            "correct_option": raw["correct_answer"].map(answer_map),
            "option_count": 4,
        }
    ).reset_index(drop=True)
    core, options = canonicalize_option_rows(frame)
    variants = write_option_variants(
        dataset="ednet",
        core=core,
        options=options,
        output_root=output_root,
        overwrite=args.overwrite,
    )
    source_row_map = pd.DataFrame(
        {
            "source_row_id": core["source_row_id"],
            "raw_user_id": raw["user_id"],
            "raw_question_id": raw["question_id"],
            "raw_timestamp": raw["timestamp"].astype("int64"),
        }
    )
    audit_artifacts = write_audit_artifacts(
        output_root / "ednet_option_source_v1",
        {
            "student_id_map.csv": pd.DataFrame(
                {
                    "raw_user_id": student_values,
                    "stu_id": [student_map[value] for value in student_values],
                }
            ),
            "exercise_id_map.csv": pd.DataFrame(
                {
                    "raw_question_id": item_values,
                    "exer_id": [item_map[value] for value in item_values],
                }
            ),
            "concept_id_map.csv": pd.DataFrame(
                {
                    "raw_tag_id": concepts,
                    "cpt_id": [concept_map[value] for value in concepts],
                }
            ),
            "source_row_map.csv": source_row_map,
        },
    )
    return {
        "source_hashes": source_hashes,
        "source_to_protocol_identity_exact": True,
        "official_snapshot_identity": official_identity,
        "source_rows": int(source_rows),
        "invalid_rows_removed": invalid_rows,
        "repeated_attempts_removed": int(repeated_attempts_removed),
        "eligible_students": int(len(eligible)),
        "student_selection": {
            "method": "lowest sha256(seed=2024,dataset,raw_user_id)",
            "selected": FROZEN_EDNET_STUDENTS,
            "minimum_first_attempt_interactions": (
                FROZEN_MIN_STUDENT_INTERACTIONS
            ),
        },
        "target_scope": "bucket:zero",
        "audit_artifacts": audit_artifacts,
        **variants,
    }


def question_sort_key(value: str) -> tuple[int, int]:
    left, right = value.split("-", maxsplit=1)
    return int(left), int(right)


def prepare_enem(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    source_hash = require_hash(args.enem_archive, "enem_archive")
    raw = pd.read_csv(
        args.enem_archive,
        compression="zip",
        usecols=[
            "ID-STUDENT",
            "ID-QUESTION",
            "STUDENT-CHOICE",
            "CORRECT-CHOICE",
            "CORRECT?",
        ],
        low_memory=False,
    )
    source_rows = len(raw)
    for column in ("STUDENT-CHOICE", "CORRECT-CHOICE"):
        raw[column] = raw[column].astype(str).str.upper()
    valid = (
        raw["STUDENT-CHOICE"].isin(list("ABCDE"))
        & raw["CORRECT-CHOICE"].isin(list("ABCDE"))
        & raw["ID-STUDENT"].notna()
        & raw["ID-QUESTION"].astype(str).str.fullmatch(r"\d+-\d+")
    )
    invalid_rows = int((~valid).sum())
    raw = raw.loc[valid].copy()
    source_label = raw["CORRECT?"].astype(str).str.lower().map(
        {"true": 1, "false": 0, "1": 1, "0": 0}
    )
    derived_label = (
        raw["STUDENT-CHOICE"] == raw["CORRECT-CHOICE"]
    ).astype(int)
    if source_label.isna().any() or not source_label.astype(int).equals(
        derived_label
    ):
        raise RuntimeError("ENEM CORRECT? disagrees with selected/correct choices.")
    before_unique = len(raw)
    raw = raw.drop_duplicates(["ID-STUDENT", "ID-QUESTION"], keep="first")
    duplicate_pairs_removed = before_unique - len(raw)
    counts = raw.groupby("ID-STUDENT").size()
    eligible_students = set(
        counts[counts >= FROZEN_MIN_STUDENT_INTERACTIONS].index
    )
    raw = raw.loc[raw["ID-STUDENT"].isin(eligible_students)].copy()
    raw = raw.reset_index(drop=True)
    student_values = sorted(raw["ID-STUDENT"].unique(), key=int)
    item_values = sorted(
        raw["ID-QUESTION"].astype(str).unique(),
        key=question_sort_key,
    )
    student_map = {value: index for index, value in enumerate(student_values)}
    item_map = {value: index for index, value in enumerate(item_values)}
    answer_map = {value: index for index, value in enumerate("ABCDE")}
    item_ids = raw["ID-QUESTION"].astype(str).map(item_map)
    frame = pd.DataFrame(
        {
            "stu_id": raw["ID-STUDENT"].map(student_map),
            "exer_id": item_ids,
            "cpt_seq": item_ids.astype(str),
            "label": (
                raw["STUDENT-CHOICE"] == raw["CORRECT-CHOICE"]
            ).astype(int),
            "selected_option": raw["STUDENT-CHOICE"].map(answer_map),
            "correct_option": raw["CORRECT-CHOICE"].map(answer_map),
            "option_count": 5,
        }
    ).reset_index(drop=True)
    core, options = canonicalize_option_rows(frame)
    variants = write_option_variants(
        dataset="enem",
        core=core,
        options=options,
        output_root=output_root,
        overwrite=args.overwrite,
    )
    source_row_map = pd.DataFrame(
        {
            "source_row_id": core["source_row_id"],
            "raw_user_id": raw["ID-STUDENT"],
            "raw_question_id": raw["ID-QUESTION"],
        }
    )
    audit_artifacts = write_audit_artifacts(
        output_root / "enem_option_source_v1",
        {
            "student_id_map.csv": pd.DataFrame(
                {
                    "raw_user_id": student_values,
                    "stu_id": [student_map[value] for value in student_values],
                }
            ),
            "exercise_id_map.csv": pd.DataFrame(
                {
                    "raw_question_id": item_values,
                    "exer_id": [item_map[value] for value in item_values],
                }
            ),
            "source_row_map.csv": source_row_map,
        },
    )
    return {
        "source_hashes": {"archive": source_hash},
        "source_to_protocol_identity_exact": True,
        "source_repository_commit": (
            "cf48f58100a119af8a3be405e6f96bc63a16915d"
        ),
        "source_rows": int(source_rows),
        "invalid_rows_removed": invalid_rows,
        "duplicate_student_item_rows_removed": int(duplicate_pairs_removed),
        "minimum_student_interactions": FROZEN_MIN_STUDENT_INTERACTIONS,
        "target_scope": "bucket:zero",
        "q_semantics": "one exercise equals one concept",
        "audit_artifacts": audit_artifacts,
        **variants,
    }


def main() -> None:
    args = parse_args()
    if args.split_seed != FROZEN_SPLIT_SEED:
        raise ValueError("Option protocols are frozen to split_seed=2024.")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "split_seed": FROZEN_SPLIT_SEED,
        "model_seed": 42,
        "repository": repository_provenance(),
        "datasets": {},
    }
    selected = (
        ("nips34", "ednet", "enem")
        if args.dataset == "all"
        else (args.dataset,)
    )
    for dataset in selected:
        if dataset == "nips34":
            payload["datasets"][dataset] = prepare_nips(args, output_root)
        elif dataset == "ednet":
            payload["datasets"][dataset] = prepare_ednet(args, output_root)
        elif dataset == "enem":
            payload["datasets"][dataset] = prepare_enem(args, output_root)
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
