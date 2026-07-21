from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import subprocess
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.pool_protocol import (
    canonical_concepts,
    derive_q_matrix,
    sha256_file,
    stable_fraction,
)
from scripts.audit_option_contrast_admission import attach_options
from scripts.evaluate_coverage_slice import add_target_coverage


MODEL_SEED = 42
SPLIT_SEED = 2024
NUM_FOLDS = 5
HIDE_FRACTION = 0.20
MIN_HISTORY_ROWS = 10
MIN_PSEUDO_TARGET_ROWS = 3
MIN_SIGNAL_TARGET_ROWS = 500
MIN_SIGNAL_TARGET_STUDENTS = 100
MIN_SIGNAL_TARGET_PER_LABEL = 100
BETA_PRIOR = 1.0
CELL_SHRINKAGE = 20.0
MIN_OPTION_CELL_STUDENTS = 5
OPTION_RELIABILITY_SHRINKAGE = 3.0
ITEM_EASE_SHRINKAGE = 20.0
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_BATCH = 8
VARIANTS = ("direct", "full", "shuffle")
COMMON_FEATURE_NAMES = (
    "log_history_rows",
    "history_balance",
    "observed_concept_fraction",
    "log_target_q_size",
    "log_reference_item_count",
    "reference_item_ease_logit",
    "target_q_seen_fraction",
    "log_target_q_attempts",
    "direct_state_mean",
    "direct_state_min",
    "direct_state_max",
    "direct_reliability_mean",
    "direct_reliability_max",
)
OPTION_FEATURE_NAMES = (
    "option_residual_mean",
    "option_residual_min",
    "option_residual_max",
    "option_reliability_mean",
    "option_reliability_max",
    "log_supported_wrong_rows",
)
FEATURE_NAMES = (*COMMON_FEATURE_NAMES, *OPTION_FEATURE_NAMES)
TARGET_OPTION_COLUMNS = ("selected_option", "correct_option", "option_count")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered train-only option-contrast signal gate."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=BOOTSTRAP_REPLICATES,
    )
    args = parser.parse_args()
    if args.bootstrap_replicates != BOOTSTRAP_REPLICATES:
        raise ValueError("The formal option signal gate fixes 2,000 bootstraps.")
    return args


def _hash_values(values: Any) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def _logit(value: float | np.ndarray) -> float | np.ndarray:
    clipped = np.clip(value, 1e-8, 1.0 - 1e-8)
    return np.log(clipped / (1.0 - clipped))


def _concepts(value: object) -> tuple[str, ...]:
    return tuple(token for token in canonical_concepts(value).split(",") if token)


def _stable_row_id(dataset: str, student: object, exercise: object) -> str:
    payload = f"{dataset}\x1f{student}\x1f{exercise}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


@dataclass(frozen=True)
class LoadedProtocol:
    name: str
    train: pd.DataFrame
    q_matrix: pd.DataFrame
    q_lookup: dict[str, tuple[str, ...]]
    target_scope: str
    input_audit: dict[str, Any]


@dataclass(frozen=True)
class PseudoProtocol:
    support: pd.DataFrame
    target_public: pd.DataFrame
    target_labels: np.ndarray
    target_scope: str
    audit: dict[str, Any]


@dataclass
class StudentStates:
    students: tuple[str, ...]
    student_index: dict[str, int]
    history_rows: np.ndarray
    history_correct: np.ndarray
    concept_attempts: np.ndarray
    concept_seen: np.ndarray
    direct_state: np.ndarray
    direct_reliability: np.ndarray
    option_residual: np.ndarray
    option_reliability: np.ndarray
    supported_wrong_rows: np.ndarray


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={
            "source_row_id": str,
            "stu_id": str,
            "exer_id": str,
            "cpt_seq": str,
        },
        low_memory=False,
    )


def _verify_file(path: Path, expected: str) -> str:
    digest = sha256_file(path)
    if digest != expected:
        raise RuntimeError(f"Hash mismatch for {path}: {digest} != {expected}")
    return digest


def load_protocol(
    manifest_path: Path,
    admission_path: Path,
    dataset: str,
) -> LoadedProtocol:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    if sha256_file(manifest_path) != admission.get("manifest_sha256"):
        raise RuntimeError("Admission result is not bound to this manifest.")
    if admission.get("route_activated") is not True:
        raise RuntimeError("Option data admission did not activate the route.")
    admission_by_dataset = {
        report["dataset"]: report for report in admission["reports"]
    }
    if admission_by_dataset.get(dataset, {}).get("passed") is not True:
        raise RuntimeError(f"{dataset} did not pass option data admission.")
    spec = manifest["datasets"][dataset]
    if dataset == "nips34":
        directory = Path(spec["holdout_dir"])
        hashes = spec["protocol_hashes"]["holdout"]
        option_path = Path(spec["train_option_files"]["holdout"]["path"])
        option_hash = spec["train_option_files"]["holdout"]["sha256"]
    else:
        holdout = spec["holdout"]
        directory = Path(holdout["directory"])
        hashes = {
            filename: details["sha256"]
            for filename, details in holdout["files"].items()
        }
        option_path = directory / "train_options.csv"
        option_hash = hashes["train_options.csv"]
    train_path = directory / "train.csv"
    q_path = directory / "Q_matrix.csv"
    verified = {
        "train.csv": _verify_file(train_path, hashes["train.csv"]),
        "Q_matrix.csv": _verify_file(q_path, hashes["Q_matrix.csv"]),
        "train_options.csv": _verify_file(option_path, option_hash),
    }
    train = _read_csv(train_path)
    options = _read_csv(option_path)
    if len(options) != len(train):
        raise RuntimeError("Train option sidecar is not row-exact.")
    attached, coverage = attach_options(train, options)
    if coverage != 1.0 or len(attached) != len(train):
        raise RuntimeError("Train option sidecar does not cover train rows exactly.")
    for column in TARGET_OPTION_COLUMNS:
        attached[column] = pd.to_numeric(
            attached[column], errors="raise"
        ).astype(int)
    q_raw = _read_csv(q_path)
    q_matrix, _ = derive_q_matrix(q_raw)
    q_lookup = {
        str(row.exer_id): _concepts(row.cpt_seq)
        for row in q_matrix.itertuples(index=False)
    }
    missing_items = set(attached["exer_id"].astype(str)) - set(q_lookup)
    if missing_items:
        raise RuntimeError(f"Train items absent from Q: {len(missing_items)}")
    opened = [str(train_path.resolve()), str(option_path.resolve()), str(q_path.resolve())]
    forbidden = [
        str((directory / filename).resolve())
        for filename in ("valid.csv", "test.csv")
    ]
    if set(opened) & set(forbidden):
        raise RuntimeError("Signal gate opened validation or test data.")
    return LoadedProtocol(
        name=dataset,
        train=attached.reset_index(drop=True),
        q_matrix=q_matrix,
        q_lookup=q_lookup,
        target_scope=spec["target_scope"],
        input_audit={
            "manifest_sha256": sha256_file(manifest_path),
            "admission_sha256": sha256_file(admission_path),
            "opened_files": opened,
            "opened_file_hashes": verified,
            "forbidden_files_not_opened": forbidden,
            "audit_option_source_opened": False,
            "train_rows": int(len(attached)),
        },
    )


def build_pseudo_protocol(protocol: LoadedProtocol) -> PseudoProtocol:
    support_indices: list[int] = []
    target_indices: list[int] = []
    fold_by_student: dict[str, int] = {}
    skipped = 0
    k_values: list[int] = []
    for student, group in protocol.train.groupby(
        protocol.train["stu_id"].astype(str),
        sort=False,
    ):
        student = str(student)
        rows = list(group.index)
        k = max(MIN_PSEUDO_TARGET_ROWS, int(np.floor(HIDE_FRACTION * len(rows))))
        if len(rows) - k < MIN_HISTORY_ROWS:
            skipped += 1
            continue
        ordered = sorted(
            rows,
            key=lambda index: (
                stable_fraction(
                    SPLIT_SEED,
                    "option-signal-hide",
                    protocol.name,
                    student,
                    str(protocol.train.at[index, "exer_id"]),
                ),
                str(protocol.train.at[index, "exer_id"]),
            ),
        )
        target_indices.extend(ordered[:k])
        support_indices.extend(ordered[k:])
        k_values.append(k)
        fold_by_student[student] = min(
            int(
                NUM_FOLDS
                * stable_fraction(
                    SPLIT_SEED,
                    "option-signal-fold",
                    protocol.name,
                    student,
                )
            ),
            NUM_FOLDS - 1,
        )
    support = protocol.train.loc[support_indices].copy().reset_index(drop=True)
    hidden = protocol.train.loc[target_indices].copy().reset_index(drop=True)
    support_groups = set(
        zip(
            support["stu_id"].astype(str),
            support["exer_id"].astype(str),
            strict=True,
        )
    )
    target_groups = set(
        zip(
            hidden["stu_id"].astype(str),
            hidden["exer_id"].astype(str),
            strict=True,
        )
    )
    if support_groups & target_groups:
        raise RuntimeError("Pseudo support and target student-item groups overlap.")
    labels = hidden["label"].to_numpy(dtype=np.int64, copy=True)
    public_columns = [
        column
        for column in (
            "source_row_id",
            "stu_id",
            "exer_id",
            "cpt_seq",
        )
        if column in hidden.columns
    ]
    target_public = hidden.loc[:, public_columns].copy()
    if "source_row_id" not in target_public:
        target_public.insert(
            0,
            "source_row_id",
            [
                _stable_row_id(protocol.name, student, exercise)
                for student, exercise in target_public[
                    ["stu_id", "exer_id"]
                ].itertuples(index=False, name=None)
            ],
        )
    target_public["fold"] = target_public["stu_id"].astype(str).map(
        fold_by_student
    ).astype(int)
    coverage_input = target_public.copy()
    coverage_input["label"] = labels
    coverage = add_target_coverage(
        coverage_input,
        student_history_frame=support,
        q_matrix=protocol.q_matrix,
    )
    target_public = coverage.drop(columns="label")
    if any(column in target_public for column in (*TARGET_OPTION_COLUMNS, "label")):
        raise RuntimeError("Pseudo-target feature frame retains a forbidden field.")
    fold_counts = {
        str(fold): int((target_public["fold"] == fold).sum())
        for fold in range(NUM_FOLDS)
    }
    if any(count == 0 for count in fold_counts.values()):
        raise RuntimeError(f"Pseudo target fold is empty: {fold_counts}")
    selection_hash = _hash_values(
        f"{row.source_row_id}:{int(row.fold)}"
        for row in target_public.sort_values("source_row_id").itertuples(index=False)
    )
    return PseudoProtocol(
        support=support,
        target_public=target_public.reset_index(drop=True),
        target_labels=labels,
        target_scope=protocol.target_scope,
        audit={
            "eligible_students": int(len(fold_by_student)),
            "skipped_students": int(skipped),
            "support_rows": int(len(support)),
            "pseudo_target_rows": int(len(target_public)),
            "minimum_k": int(min(k_values)),
            "maximum_k": int(max(k_values)),
            "fold_counts": fold_counts,
            "selection_sha256": selection_hash,
            "support_target_group_overlap": 0,
            "selection_key_uses_label_or_option": False,
            "target_option_columns_present": False,
        },
    )


def _target_mask(frame: pd.DataFrame, scope: str) -> np.ndarray:
    if scope == "bucket:zero":
        return frame["coverage_bucket"].astype(str).to_numpy() == "zero"
    if scope == "low_coverage":
        return (
            frame["coverage_group"].astype(str).to_numpy()
            == "low_coverage"
        )
    raise ValueError(f"Unsupported target scope: {scope}")


class OptionSignatureModel:
    def __init__(
        self,
        *,
        dataset: str,
        fold: int,
        reference_support: pd.DataFrame,
        q_lookup: dict[str, tuple[str, ...]],
    ) -> None:
        self.dataset = dataset
        self.fold = fold
        self.reference_support = reference_support.reset_index(drop=True)
        self.items = tuple(sorted(q_lookup))
        self.item_index = {item: index for index, item in enumerate(self.items)}
        self.concepts = tuple(
            sorted({concept for values in q_lookup.values() for concept in values})
        )
        self.concept_index = {
            concept: index for index, concept in enumerate(self.concepts)
        }
        self.q_lookup = q_lookup
        self.max_options = int(reference_support["option_count"].max())
        self.q_matrix = self._build_q_matrix()
        self.reference_students = tuple(
            sorted(reference_support["stu_id"].astype(str).unique())
        )
        self.reference_student_index = {
            student: index for index, student in enumerate(self.reference_students)
        }
        (
            self.mastery,
            self.seen,
            self.concept_attempts,
            self.concept_correct,
            self.global_concept_prior,
        ) = self._reference_mastery()
        self.global_concept_attempts = self.concept_attempts.sum(axis=0)
        self.global_concept_correct = self.concept_correct.sum(axis=0)
        self.reference_row_count = int(len(self.reference_support))
        self.reference_correct_count = float(
            self.reference_support["label"].sum()
        )
        (
            self.binary_numerator,
            self.binary_denominator,
            self.binary_state,
            self.binary_count,
        ) = self._binary_signatures()
        (
            self.option_numerator,
            self.option_denominator,
            self.option_count,
            self.option_delta,
        ) = self._option_signatures()
        self.item_count, self.item_correct, self.item_ease = (
            self._item_statistics()
        )
        (
            self.wrong_option_counts,
            self.fallback_option_counts,
            self.student_item_wrong_counts,
            self.student_fallback_wrong_counts,
        ) = (
            self._shuffle_distributions()
        )

    def _build_q_matrix(self) -> sparse.csr_matrix:
        rows: list[int] = []
        columns: list[int] = []
        for item, item_index in self.item_index.items():
            for concept in self.q_lookup[item]:
                rows.append(item_index)
                columns.append(self.concept_index[concept])
        return sparse.csr_matrix(
            (
                np.ones(len(rows), dtype=np.float64),
                (rows, columns),
            ),
            shape=(len(self.items), len(self.concepts)),
        )

    def _history_matrices(
        self,
        frame: pd.DataFrame,
        students: tuple[str, ...],
    ) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        student_index = {student: index for index, student in enumerate(students)}
        row_students = frame["stu_id"].astype(str).map(student_index).to_numpy()
        row_items = frame["exer_id"].astype(str).map(self.item_index).to_numpy()
        if pd.isna(row_students).any() or pd.isna(row_items).any():
            raise RuntimeError("History contains an unmapped student or item.")
        attempts = sparse.csr_matrix(
            (
                np.ones(len(frame), dtype=np.float64),
                (row_students.astype(int), row_items.astype(int)),
            ),
            shape=(len(students), len(self.items)),
        )
        correct = sparse.csr_matrix(
            (
                frame["label"].to_numpy(dtype=np.float64),
                (row_students.astype(int), row_items.astype(int)),
            ),
            shape=(len(students), len(self.items)),
        )
        return attempts, correct

    def _reference_mastery(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        attempts, correct = self._history_matrices(
            self.reference_support,
            self.reference_students,
        )
        concept_attempts = attempts.dot(self.q_matrix).toarray()
        concept_correct = correct.dot(self.q_matrix).toarray()
        seen = concept_attempts > 0
        mastery = np.zeros_like(concept_attempts, dtype=np.float64)
        mastery[seen] = (
            concept_correct[seen] + BETA_PRIOR
        ) / (concept_attempts[seen] + 2.0 * BETA_PRIOR)
        global_prior = (
            concept_correct.sum(axis=0) + BETA_PRIOR
        ) / (concept_attempts.sum(axis=0) + 2.0 * BETA_PRIOR)
        return (
            mastery,
            seen.astype(np.float64),
            concept_attempts,
            concept_correct,
            global_prior,
        )

    def _cell_matrix(
        self,
        cell_ids: np.ndarray,
        *,
        cell_count: int,
        row_mask: np.ndarray | None = None,
    ) -> sparse.csr_matrix:
        if row_mask is None:
            row_mask = np.ones(len(self.reference_support), dtype=bool)
        students = self.reference_support.loc[
            row_mask, "stu_id"
        ].astype(str).map(self.reference_student_index).to_numpy(dtype=int)
        cells = cell_ids[row_mask]
        return sparse.csr_matrix(
            (
                np.ones(len(cells), dtype=np.float64),
                (students, cells),
            ),
            shape=(len(self.reference_students), cell_count),
        )

    def _binary_signatures(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        item_ids = self.reference_support["exer_id"].astype(str).map(
            self.item_index
        ).to_numpy(dtype=int)
        labels = self.reference_support["label"].to_numpy(dtype=int)
        cells = item_ids * 2 + labels
        history = self._cell_matrix(cells, cell_count=len(self.items) * 2)
        weighted_mastery = self.mastery * self.seen
        numerator = np.asarray(history.T.dot(weighted_mastery))
        denominator = np.asarray(history.T.dot(self.seen))
        state = (
            numerator
            + CELL_SHRINKAGE * self.global_concept_prior[None, :]
        ) / (denominator + CELL_SHRINKAGE)
        count = np.asarray(history.sum(axis=0)).ravel()
        return numerator, denominator, state, count

    def _option_signatures(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        item_ids = self.reference_support["exer_id"].astype(str).map(
            self.item_index
        ).to_numpy(dtype=int)
        selected = self.reference_support["selected_option"].to_numpy(dtype=int)
        cells = item_ids * self.max_options + selected
        wrong = self.reference_support["label"].to_numpy(dtype=int) == 0
        history = self._cell_matrix(
            cells,
            cell_count=len(self.items) * self.max_options,
            row_mask=wrong,
        )
        weighted_mastery = self.mastery * self.seen
        numerator = np.asarray(history.T.dot(weighted_mastery))
        denominator = np.asarray(history.T.dot(self.seen))
        count = np.asarray(history.sum(axis=0)).ravel()
        cell_items = np.repeat(np.arange(len(self.items)), self.max_options)
        binary_wrong = self.binary_state[cell_items * 2]
        option_state = (
            numerator + CELL_SHRINKAGE * binary_wrong
        ) / (denominator + CELL_SHRINKAGE)
        confidence = denominator / (denominator + CELL_SHRINKAGE)
        supported = (
            (count[:, None] >= MIN_OPTION_CELL_STUDENTS)
            & (denominator >= MIN_OPTION_CELL_STUDENTS)
        )
        delta = option_state - binary_wrong
        delta[~supported] = 0.0
        return numerator, denominator, count, delta

    def _item_statistics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        item_ids = self.reference_support["exer_id"].astype(str).map(
            self.item_index
        ).to_numpy(dtype=int)
        labels = self.reference_support["label"].to_numpy(dtype=np.float64)
        count = np.bincount(item_ids, minlength=len(self.items)).astype(float)
        correct = np.bincount(
            item_ids,
            weights=labels,
            minlength=len(self.items),
        )
        global_rate = float((labels.sum() + 1.0) / (len(labels) + 2.0))
        ease = (
            correct + ITEM_EASE_SHRINKAGE * global_rate
        ) / (count + ITEM_EASE_SHRINKAGE)
        return count, correct, ease

    def _shuffle_distributions(
        self,
    ) -> tuple[
        np.ndarray,
        dict[int, np.ndarray],
        dict[tuple[str, int], np.ndarray],
        dict[tuple[str, int], np.ndarray],
    ]:
        counts = np.zeros(
            (len(self.items), self.max_options),
            dtype=np.int64,
        )
        wrong = self.reference_support.loc[
            self.reference_support["label"].to_numpy(dtype=int) == 0
        ]
        for row in wrong.itertuples(index=False):
            counts[
                self.item_index[str(row.exer_id)],
                int(row.selected_option),
            ] += 1
        fallback: dict[int, np.ndarray] = {}
        student_item: dict[tuple[str, int], np.ndarray] = {}
        student_fallback: dict[tuple[str, int], np.ndarray] = {}
        for option_count, group in wrong.groupby("option_count", sort=True):
            values = np.zeros(self.max_options, dtype=np.int64)
            for option, count in group["selected_option"].value_counts().items():
                values[int(option)] = int(count)
            fallback[int(option_count)] = values
        for (student, exercise), group in wrong.groupby(
            ["stu_id", "exer_id"], sort=False
        ):
            values = np.zeros(self.max_options, dtype=np.int64)
            for option, count in group["selected_option"].value_counts().items():
                values[int(option)] += int(count)
            student_item[(str(student), self.item_index[str(exercise)])] = values
        for (student, option_count), group in wrong.groupby(
            ["stu_id", "option_count"], sort=False
        ):
            values = np.zeros(self.max_options, dtype=np.int64)
            for option, count in group["selected_option"].value_counts().items():
                values[int(option)] += int(count)
            student_fallback[(str(student), int(option_count))] = values
        return counts, fallback, student_item, student_fallback

    def shuffled_options(
        self,
        frame: pd.DataFrame,
        *,
        leave_one_student_out: bool,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        output = frame["selected_option"].to_numpy(dtype=int, copy=True)
        primary = 0
        fallback = 0
        uniform = 0
        wrong_indices = np.flatnonzero(
            frame["label"].to_numpy(dtype=int) == 0
        )
        for index in wrong_indices:
            row = frame.iloc[index]
            item = self.item_index[str(row.exer_id)]
            option_count = int(row.option_count)
            correct_option = int(row.correct_option)
            weights = self.wrong_option_counts[item].astype(float).copy()
            if leave_one_student_out:
                weights -= self.student_item_wrong_counts.get(
                    (str(row.stu_id), item),
                    np.zeros(self.max_options, dtype=np.int64),
                )
            legal = np.arange(self.max_options) < option_count
            legal &= np.arange(self.max_options) != correct_option
            weights[~legal] = 0.0
            if weights.sum() > 0:
                primary += 1
            else:
                weights = self.fallback_option_counts.get(
                    option_count,
                    np.zeros(self.max_options, dtype=np.int64),
                ).astype(float).copy()
                if leave_one_student_out:
                    weights -= self.student_fallback_wrong_counts.get(
                        (str(row.stu_id), option_count),
                        np.zeros(self.max_options, dtype=np.int64),
                    )
                weights[~legal] = 0.0
                if weights.sum() > 0:
                    fallback += 1
                else:
                    weights = legal.astype(float)
                    uniform += 1
            if weights.sum() <= 0:
                raise RuntimeError("No legal wrong option is available for shuffle.")
            threshold = stable_fraction(
                SPLIT_SEED,
                "option-signal-shuffle",
                self.dataset,
                str(row.stu_id),
                str(row.exer_id),
            ) * weights.sum()
            output[index] = int(
                np.searchsorted(np.cumsum(weights), threshold, side="right")
            )
        return output, {
            "wrong_rows": int(len(wrong_indices)),
            "primary_item_distribution": int(primary),
            "option_count_fallback": int(fallback),
            "uniform_legal_fallback": int(uniform),
            "hash_uses_real_option": False,
        }

    def _student_history_summary(
        self,
        frame: pd.DataFrame,
        students: tuple[str, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        attempts, correct = self._history_matrices(frame, students)
        concept_attempts = attempts.dot(self.q_matrix).toarray()
        concept_seen = concept_attempts > 0
        history_rows = np.asarray(attempts.sum(axis=1)).ravel()
        history_correct = np.asarray(correct.sum(axis=1)).ravel()
        return history_rows, history_correct, concept_attempts, concept_seen

    def states(
        self,
        frame: pd.DataFrame,
        *,
        chosen_options: np.ndarray,
        leave_one_student_out: bool,
    ) -> StudentStates:
        frame = frame.reset_index(drop=True)
        students = tuple(sorted(frame["stu_id"].astype(str).unique()))
        student_index = {student: index for index, student in enumerate(students)}
        (
            history_rows,
            history_correct,
            concept_attempts,
            concept_seen,
        ) = self._student_history_summary(frame, students)
        num_students = len(students)
        num_concepts = len(self.concepts)
        direct_sum = np.zeros((num_students, num_concepts), dtype=np.float64)
        direct_rel_sum = np.zeros_like(direct_sum)
        option_sum = np.zeros_like(direct_sum)
        option_rel_sum = np.zeros_like(direct_sum)
        wrong_rows = np.zeros(num_students, dtype=np.float64)
        supported_wrong = np.zeros(num_students, dtype=np.float64)
        item_ids = frame["exer_id"].astype(str).map(
            self.item_index
        ).to_numpy(dtype=int)
        labels = frame["label"].to_numpy(dtype=int)
        actual_options = frame["selected_option"].to_numpy(dtype=int)
        local_students = frame["stu_id"].astype(str).map(
            student_index
        ).to_numpy(dtype=int)
        if leave_one_student_out:
            reference_students = frame["stu_id"].astype(str).map(
                self.reference_student_index
            ).to_numpy(dtype=int)
            if pd.isna(reference_students).any():
                raise RuntimeError("LOO history includes a non-reference student.")
        else:
            reference_students = np.full(len(frame), -1, dtype=int)
        chunk_size = 2_000
        for start in range(0, len(frame), chunk_size):
            stop = min(start + chunk_size, len(frame))
            sl = slice(start, stop)
            row_items = item_ids[sl]
            row_labels = labels[sl]
            binary_cells = row_items * 2 + row_labels
            numerator = self.binary_numerator[binary_cells].copy()
            denominator = self.binary_denominator[binary_cells].copy()
            if leave_one_student_out:
                ref = reference_students[sl]
                numerator -= self.mastery[ref] * self.seen[ref]
                denominator -= self.seen[ref]
                prior_numerator = (
                    self.global_concept_correct[None, :]
                    - self.concept_correct[ref]
                    + BETA_PRIOR
                )
                prior_denominator = (
                    self.global_concept_attempts[None, :]
                    - self.concept_attempts[ref]
                    + 2.0 * BETA_PRIOR
                )
                prior = prior_numerator / prior_denominator
            else:
                prior = self.global_concept_prior[None, :]
            binary_state = (
                numerator
                + CELL_SHRINKAGE * prior
            ) / (denominator + CELL_SHRINKAGE)
            binary_reliability = denominator / (
                denominator + CELL_SHRINKAGE
            )
            np.add.at(direct_sum, local_students[sl], binary_state)
            np.add.at(
                direct_rel_sum,
                local_students[sl],
                binary_reliability,
            )

            local_wrong = np.flatnonzero(row_labels == 0)
            if len(local_wrong) == 0:
                continue
            global_rows = start + local_wrong
            selected = chosen_options[global_rows]
            option_cells = (
                item_ids[global_rows] * self.max_options + selected
            )
            option_num = self.option_numerator[option_cells].copy()
            option_den = self.option_denominator[option_cells].copy()
            effective_count = self.option_count[option_cells].copy()
            if leave_one_student_out:
                ref = reference_students[global_rows]
                same_cell = selected == actual_options[global_rows]
                if same_cell.any():
                    option_num[same_cell] -= (
                        self.mastery[ref[same_cell]]
                        * self.seen[ref[same_cell]]
                    )
                    option_den[same_cell] -= self.seen[ref[same_cell]]
                    effective_count[same_cell] -= 1.0
            binary_wrong = binary_state[local_wrong]
            option_state = (
                option_num + CELL_SHRINKAGE * binary_wrong
            ) / (option_den + CELL_SHRINKAGE)
            confidence = option_den / (option_den + CELL_SHRINKAGE)
            supported = (
                (effective_count[:, None] >= MIN_OPTION_CELL_STUDENTS)
                & (option_den >= MIN_OPTION_CELL_STUDENTS)
            )
            delta = option_state - binary_wrong
            delta[~supported] = 0.0
            target_students = local_students[global_rows]
            np.add.at(option_sum, target_students, delta)
            np.add.at(
                option_rel_sum,
                target_students,
                confidence * supported,
            )
            np.add.at(wrong_rows, target_students, 1.0)
            np.add.at(
                supported_wrong,
                target_students,
                supported.any(axis=1).astype(float),
            )
        history_denominator = np.maximum(history_rows[:, None], 1.0)
        direct_state = direct_sum / history_denominator
        direct_reliability = direct_rel_sum / history_denominator
        wrong_denominator = np.maximum(wrong_rows[:, None], 1.0)
        option_reliability = option_rel_sum / wrong_denominator
        option_residual = option_sum / wrong_denominator
        student_reliability = supported_wrong / (
            supported_wrong + OPTION_RELIABILITY_SHRINKAGE
        )
        option_residual *= student_reliability[:, None]
        return StudentStates(
            students=students,
            student_index=student_index,
            history_rows=history_rows,
            history_correct=history_correct,
            concept_attempts=concept_attempts,
            concept_seen=concept_seen,
            direct_state=direct_state,
            direct_reliability=direct_reliability,
            option_residual=option_residual,
            option_reliability=option_reliability,
            supported_wrong_rows=supported_wrong,
        )

    def feature_matrices(
        self,
        target_public: pd.DataFrame,
        *,
        states_full: StudentStates,
        states_shuffle: StudentStates,
        leave_one_student_out: bool,
    ) -> dict[str, np.ndarray]:
        if states_full.students != states_shuffle.students:
            raise RuntimeError("Full and Shuffle student order differs.")
        rows = len(target_public)
        common = np.zeros((rows, len(COMMON_FEATURE_NAMES)), dtype=np.float64)
        full_option = np.zeros(
            (rows, len(OPTION_FEATURE_NAMES)),
            dtype=np.float64,
        )
        shuffle_option = np.zeros_like(full_option)
        for row_index, row in enumerate(target_public.itertuples(index=False)):
            student = str(row.stu_id)
            student_index = states_full.student_index[student]
            item = self.item_index[str(row.exer_id)]
            q_indices = np.asarray(
                [
                    self.concept_index[concept]
                    for concept in self.q_lookup[str(row.exer_id)]
                ],
                dtype=int,
            )
            history_rows = states_full.history_rows[student_index]
            history_correct = states_full.history_correct[student_index]
            posterior = (
                history_correct + BETA_PRIOR
            ) / (history_rows + 2.0 * BETA_PRIOR)
            attempts = states_full.concept_attempts[student_index, q_indices]
            seen = states_full.concept_seen[student_index, q_indices]
            direct = states_full.direct_state[student_index, q_indices]
            direct_rel = states_full.direct_reliability[
                student_index, q_indices
            ]
            if leave_one_student_out:
                ref_index = self.reference_student_index[student]
                global_rate = (
                    self.reference_correct_count
                    - history_correct
                    + BETA_PRIOR
                ) / (
                    self.reference_row_count
                    - history_rows
                    + 2.0 * BETA_PRIOR
                )
                item_ease = (
                    self.item_correct[item]
                    + ITEM_EASE_SHRINKAGE * global_rate
                ) / (self.item_count[item] + ITEM_EASE_SHRINKAGE)
            else:
                item_ease = self.item_ease[item]
            common[row_index] = (
                np.log1p(history_rows),
                2.0 * posterior - 1.0,
                float(states_full.concept_seen[student_index].mean()),
                np.log1p(len(q_indices)),
                np.log1p(self.item_count[item]),
                float(_logit(item_ease)),
                float(seen.mean()),
                np.log1p(attempts.sum()),
                float(direct.mean()),
                float(direct.min()),
                float(direct.max()),
                float(direct_rel.mean()),
                float(direct_rel.max()),
            )
            full_values = states_full.option_residual[
                student_index, q_indices
            ]
            full_rel = states_full.option_reliability[
                student_index, q_indices
            ]
            shuffle_values = states_shuffle.option_residual[
                student_index, q_indices
            ]
            shuffle_rel = states_shuffle.option_reliability[
                student_index, q_indices
            ]
            full_option[row_index] = (
                float(full_values.mean()),
                float(full_values.min()),
                float(full_values.max()),
                float(full_rel.mean()),
                float(full_rel.max()),
                np.log1p(states_full.supported_wrong_rows[student_index]),
            )
            shuffle_option[row_index] = (
                float(shuffle_values.mean()),
                float(shuffle_values.min()),
                float(shuffle_values.max()),
                float(shuffle_rel.mean()),
                float(shuffle_rel.max()),
                np.log1p(states_shuffle.supported_wrong_rows[student_index]),
            )
        zeros = np.zeros_like(full_option)
        return {
            "direct": np.hstack((common, zeros)),
            "full": np.hstack((common, full_option)),
            "shuffle": np.hstack((common, shuffle_option)),
        }


def fit_fold(
    protocol: LoadedProtocol,
    pseudo: PseudoProtocol,
    fold: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    reference_students = set(
        pseudo.target_public.loc[
            pseudo.target_public["fold"] != fold,
            "stu_id",
        ].astype(str)
    )
    held_students = set(
        pseudo.target_public.loc[
            pseudo.target_public["fold"] == fold,
            "stu_id",
        ].astype(str)
    )
    if reference_students & held_students:
        raise RuntimeError("Reference and held students overlap.")
    reference_support = pseudo.support.loc[
        pseudo.support["stu_id"].astype(str).isin(reference_students)
    ].copy()
    held_support = pseudo.support.loc[
        pseudo.support["stu_id"].astype(str).isin(held_students)
    ].copy()
    reference_mask = (
        pseudo.target_public["fold"].to_numpy(dtype=int) != fold
    )
    held_mask = ~reference_mask
    reference_target = pseudo.target_public.loc[
        reference_mask
    ].reset_index(drop=True)
    held_target = pseudo.target_public.loc[held_mask].reset_index(drop=True)
    reference_labels = pseudo.target_labels[reference_mask]
    held_labels = pseudo.target_labels[held_mask]
    model = OptionSignatureModel(
        dataset=protocol.name,
        fold=fold,
        reference_support=reference_support,
        q_lookup=protocol.q_lookup,
    )
    ref_shuffle, ref_shuffle_audit = model.shuffled_options(
        reference_support,
        leave_one_student_out=True,
    )
    held_shuffle, held_shuffle_audit = model.shuffled_options(
        held_support,
        leave_one_student_out=False,
    )
    ref_full_states = model.states(
        reference_support,
        chosen_options=reference_support["selected_option"].to_numpy(dtype=int),
        leave_one_student_out=True,
    )
    ref_shuffle_states = model.states(
        reference_support,
        chosen_options=ref_shuffle,
        leave_one_student_out=True,
    )
    held_full_states = model.states(
        held_support,
        chosen_options=held_support["selected_option"].to_numpy(dtype=int),
        leave_one_student_out=False,
    )
    held_shuffle_states = model.states(
        held_support,
        chosen_options=held_shuffle,
        leave_one_student_out=False,
    )
    reference_features = model.feature_matrices(
        reference_target,
        states_full=ref_full_states,
        states_shuffle=ref_shuffle_states,
        leave_one_student_out=True,
    )
    held_features = model.feature_matrices(
        held_target,
        states_full=held_full_states,
        states_shuffle=held_shuffle_states,
        leave_one_student_out=False,
    )
    dimensions = {
        variant: int(reference_features[variant].shape[1])
        for variant in VARIANTS
    }
    if len(set(dimensions.values())) != 1:
        raise RuntimeError(f"Feature dimensions differ: {dimensions}")
    scaler = StandardScaler()
    scaler.fit(
        np.vstack([reference_features[variant] for variant in VARIANTS])
    )
    predictions: dict[str, np.ndarray] = {}
    if np.unique(reference_labels).size != 2:
        raise RuntimeError(f"Reference fold {fold} lacks both labels.")
    for variant in VARIANTS:
        estimator = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="liblinear",
            max_iter=1_000,
            random_state=MODEL_SEED,
        )
        estimator.fit(
            scaler.transform(reference_features[variant]),
            reference_labels,
        )
        predictions[variant] = estimator.predict_proba(
            scaler.transform(held_features[variant])
        )[:, 1]
    result = held_target.copy()
    result["label"] = held_labels
    for variant in VARIANTS:
        result[f"prob_{variant}"] = predictions[variant]
    target = _target_mask(result, protocol.target_scope)
    fold_audit = {
        "fold": int(fold),
        "reference_students": int(len(reference_students)),
        "held_students": int(len(held_students)),
        "reference_support_rows": int(len(reference_support)),
        "held_support_rows": int(len(held_support)),
        "reference_target_rows": int(len(reference_target)),
        "held_target_rows": int(len(held_target)),
        "held_pseudo_t_rows": int(target.sum()),
        "reference_student_sha256": _hash_values(sorted(reference_students)),
        "held_student_sha256": _hash_values(sorted(held_students)),
        "feature_names": list(FEATURE_NAMES),
        "feature_dimensions": dimensions,
        "full_shuffle_same_width": (
            dimensions["full"] == dimensions["shuffle"]
        ),
        "scaler_sha256": _hash_values(
            np.concatenate((scaler.mean_, scaler.scale_))
        ),
        "classifier": (
            "L2 LogisticRegression(C=1, solver=liblinear, max_iter=1000)"
        ),
        "ref_shuffle": ref_shuffle_audit,
        "held_shuffle": held_shuffle_audit,
        "target_option_fields_consumed": False,
    }
    return result, fold_audit


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    if len(labels) == 0 or np.unique(labels).size != 2:
        raise RuntimeError("Formal metric scope must contain both labels.")
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "acc": float(accuracy_score(labels, probabilities >= 0.5)),
        "rmse": float(np.sqrt(mean_squared_error(labels, probabilities))),
        "rows": int(len(labels)),
        "label_0": int((labels == 0).sum()),
        "label_1": int((labels == 1).sum()),
    }


def summarize_predictions(
    predictions: pd.DataFrame,
    *,
    target_scope: str,
) -> dict[str, Any]:
    labels = predictions["label"].to_numpy(dtype=int)
    target = _target_mask(predictions, target_scope)
    metrics: dict[str, Any] = {}
    for variant in VARIANTS:
        probability = predictions[f"prob_{variant}"].to_numpy(dtype=float)
        metrics[variant] = {
            "overall": _metrics(labels, probability),
            "target": _metrics(labels[target], probability[target]),
        }
    direct_auc = metrics["direct"]["target"]["auc"]
    shuffle_auc = metrics["shuffle"]["target"]["auc"]
    stronger = "direct" if direct_auc >= shuffle_auc else "shuffle"
    full_target = metrics["full"]["target"]
    control_target = metrics[stronger]["target"]
    full_overall = metrics["full"]["overall"]
    control_overall = metrics[stronger]["overall"]
    deltas = {
        "target_auc": full_target["auc"] - control_target["auc"],
        "overall_auc": full_overall["auc"] - control_overall["auc"],
        "target_brier": full_target["brier"] - control_target["brier"],
        "overall_brier": full_overall["brier"] - control_overall["brier"],
    }
    deterministic_pass = (
        deltas["target_auc"] >= 0.005
        and deltas["overall_auc"] >= -0.001
        and deltas["target_brier"] <= 0.0002
    )
    return {
        "target_scope": target_scope,
        "metrics": metrics,
        "stronger_control": stronger,
        "control_selected_once_by_target_auc": True,
        "deltas_full_minus_control": deltas,
        "deterministic_dataset_pass": bool(deterministic_pass),
    }


def _weighted_auc_batch(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    order = np.argsort(probabilities, kind="mergesort")
    ordered_probabilities = probabilities[order]
    ordered_labels = labels[order]
    starts = np.r_[
        0,
        np.flatnonzero(np.diff(ordered_probabilities) != 0.0) + 1,
    ]
    ordered_weights = weights[:, order]
    positive = np.add.reduceat(
        ordered_weights * ordered_labels[None, :],
        starts,
        axis=1,
    )
    negative = np.add.reduceat(
        ordered_weights * (1.0 - ordered_labels)[None, :],
        starts,
        axis=1,
    )
    negative_before = np.cumsum(negative, axis=1) - negative
    numerator = np.sum(
        positive * (negative_before + 0.5 * negative),
        axis=1,
    )
    denominator = positive.sum(axis=1) * negative.sum(axis=1)
    output = np.full(len(weights), np.nan, dtype=np.float64)
    valid = denominator > 0
    output[valid] = numerator[valid] / denominator[valid]
    return output


def fast_student_cluster_bootstrap(
    *,
    labels: np.ndarray,
    students: np.ndarray,
    full_probability: np.ndarray,
    control_probability: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.float64)
    students = np.asarray(students).astype(str)
    full_probability = np.asarray(full_probability, dtype=np.float64)
    control_probability = np.asarray(control_probability, dtype=np.float64)
    if np.unique(labels).size != 2:
        raise RuntimeError("Bootstrap scope lacks both labels.")
    unique_students, inverse = np.unique(students, return_inverse=True)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for start in range(0, replicates, BOOTSTRAP_BATCH):
        batch = min(BOOTSTRAP_BATCH, replicates - start)
        counts = np.empty((batch, len(unique_students)), dtype=np.int32)
        for index in range(batch):
            sample = rng.integers(
                0,
                len(unique_students),
                size=len(unique_students),
            )
            counts[index] = np.bincount(
                sample,
                minlength=len(unique_students),
            )
        row_weights = counts[:, inverse].astype(np.float64)
        full_auc = _weighted_auc_batch(
            labels,
            full_probability,
            row_weights,
        )
        control_auc = _weighted_auc_batch(
            labels,
            control_probability,
            row_weights,
        )
        valid = np.isfinite(full_auc) & np.isfinite(control_auc)
        deltas.extend((full_auc[valid] - control_auc[valid]).tolist())
    if len(deltas) < max(100, int(0.9 * replicates)):
        raise RuntimeError("Too many bootstrap samples lacked both labels.")
    values = np.asarray(deltas)
    observed_full = float(roc_auc_score(labels, full_probability))
    observed_control = float(roc_auc_score(labels, control_probability))
    return {
        "full_auc": observed_full,
        "control_auc": observed_control,
        "delta_auc": observed_full - observed_control,
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "probability_delta_positive": float((values > 0.0).mean()),
        "student_count": int(len(unique_students)),
        "row_count": int(len(labels)),
        "replicates_requested": int(replicates),
        "replicates_used": int(len(values)),
        "bootstrap_seed": int(seed),
        "bootstrap_is_multi_seed": False,
    }


def deterministic_aggregate_gate(
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    passing = {
        name
        for name, summary in summaries.items()
        if summary.get("signal_eligible", True)
        and summary.get("deterministic_dataset_pass", False)
    }
    deltas = {
        name: summary["deltas_full_minus_control"]["target_auc"]
        for name, summary in summaries.items()
        if summary.get("signal_eligible", True)
    }
    return {
        "passing_datasets": sorted(passing),
        "at_least_two_datasets": len(passing) >= 2,
        "ednet_required": "ednet" in passing,
        "nips_or_enem_required": bool(passing & {"nips34", "enem"}),
        "ineligible_datasets": sorted(name for name, summary in summaries.items() if not summary.get("signal_eligible", True)),
        "one_target_delta_at_least_0_010": (
            max((deltas[name] for name in passing), default=float("-inf"))
            >= 0.010
        ),
        "bootstrap_needed": (
            len(passing) >= 2
            and "ednet" in passing
            and bool(passing & {"nips34", "enem"})
            and max(
                (deltas[name] for name in passing),
                default=float("-inf"),
            )
            >= 0.010
        ),
    }


def pseudo_target_feasibility(
    pseudo: PseudoProtocol,
) -> dict[str, Any]:
    target = _target_mask(pseudo.target_public, pseudo.target_scope)
    labels = pseudo.target_labels[target]
    students = pseudo.target_public.loc[target, "stu_id"].astype(str)
    label_0 = int((labels == 0).sum())
    label_1 = int((labels == 1).sum())
    report = {
        "scope": pseudo.target_scope,
        "rows": int(target.sum()),
        "students": int(students.nunique()),
        "label_0": label_0,
        "label_1": label_1,
        "minimum_rows": MIN_SIGNAL_TARGET_ROWS,
        "minimum_students": MIN_SIGNAL_TARGET_STUDENTS,
        "minimum_per_label": MIN_SIGNAL_TARGET_PER_LABEL,
    }
    report["eligible"] = bool(
        report["rows"] >= MIN_SIGNAL_TARGET_ROWS
        and report["students"] >= MIN_SIGNAL_TARGET_STUDENTS
        and label_0 >= MIN_SIGNAL_TARGET_PER_LABEL
        and label_1 >= MIN_SIGNAL_TARGET_PER_LABEL
    )
    report["reason"] = (
        "ok" if report["eligible"] else "insufficient_target_support"
    )
    return report


def run_dataset(
    protocol: LoadedProtocol,
    *,
    output_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    pseudo = build_pseudo_protocol(protocol)
    feasibility = pseudo_target_feasibility(pseudo)
    if not feasibility["eligible"]:
        summary = {
            "dataset": protocol.name,
            "signal_eligible": False,
            "ineligibility": feasibility,
            "target_scope": protocol.target_scope,
            "input_audit": protocol.input_audit,
            "pseudo_protocol": pseudo.audit,
            "target_option_or_label_used_as_feature": False,
            "model_seed": MODEL_SEED,
            "split_seed": SPLIT_SEED,
            "deterministic_dataset_pass": False,
        }
        summary_path = output_dir / f"{protocol.name}_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return summary, pd.DataFrame()
    fold_predictions = []
    fold_audits = []
    for fold in range(NUM_FOLDS):
        prediction, audit = fit_fold(protocol, pseudo, fold)
        fold_predictions.append(prediction)
        fold_audits.append(audit)
    predictions = pd.concat(fold_predictions, ignore_index=True)
    if predictions["source_row_id"].duplicated().any():
        raise RuntimeError("OOF predictions contain duplicate source rows.")
    expected_ids = set(pseudo.target_public["source_row_id"].astype(str))
    actual_ids = set(predictions["source_row_id"].astype(str))
    if actual_ids != expected_ids or len(predictions) != len(expected_ids):
        raise RuntimeError("OOF predictions do not exactly cover pseudo targets.")
    predictions = predictions.sort_values(
        "source_row_id",
        kind="stable",
    ).reset_index(drop=True)
    summary = summarize_predictions(
        predictions,
        target_scope=protocol.target_scope,
    )
    prediction_path = output_dir / f"{protocol.name}_predictions.csv"
    predictions.to_csv(prediction_path, index=False)
    summary.update(
        {
            "dataset": protocol.name,
            "input_audit": protocol.input_audit,
            "pseudo_protocol": pseudo.audit,
            "signal_eligibility": feasibility,
            "folds": fold_audits,
            "prediction_path": str(prediction_path.resolve()),
            "prediction_sha256": sha256_file(prediction_path),
            "prediction_rows": int(len(predictions)),
            "target_option_or_label_used_as_feature": False,
            "model_seed": MODEL_SEED,
            "split_seed": SPLIT_SEED,
        }
    )
    summary_path = output_dir / f"{protocol.name}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary, predictions


def repository_provenance() -> dict[str, Any]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(
            "Formal signal gate requires a clean committed worktree."
        )
    tracked_scripts = {
        "signal_script_sha256": sha256_file(Path(__file__)),
        "admission_script_sha256": sha256_file(
            PROJECT_ROOT / "scripts/audit_option_contrast_admission.py"
        ),
        "coverage_script_sha256": sha256_file(
            PROJECT_ROOT / "scripts/evaluate_coverage_slice.py"
        ),
    }
    return {"git_commit": head, **tracked_scripts}


def main() -> None:
    args = parse_args()
    provenance = repository_provenance()
    manifest_path = Path(args.manifest)
    admission_path = Path(args.admission)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("Formal signal output directory must be empty.")
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}
    predictions: dict[str, pd.DataFrame] = {}
    for dataset in ("nips34", "ednet", "enem"):
        protocol = load_protocol(
            manifest_path,
            admission_path,
            dataset,
        )
        summary, prediction = run_dataset(
            protocol,
            output_dir=output_dir,
        )
        summaries[dataset] = summary
        predictions[dataset] = prediction
    deterministic_gate = deterministic_aggregate_gate(summaries)
    bootstraps: dict[str, Any] = {}
    if deterministic_gate["bootstrap_needed"]:
        for dataset in deterministic_gate["passing_datasets"]:
            frame = predictions[dataset]
            summary = summaries[dataset]
            target = _target_mask(frame, summary["target_scope"])
            control = summary["stronger_control"]
            bootstrap = fast_student_cluster_bootstrap(
                labels=frame.loc[target, "label"].to_numpy(dtype=int),
                students=frame.loc[target, "stu_id"].astype(str).to_numpy(),
                full_probability=frame.loc[
                    target, "prob_full"
                ].to_numpy(dtype=float),
                control_probability=frame.loc[
                    target, f"prob_{control}"
                ].to_numpy(dtype=float),
                replicates=args.bootstrap_replicates,
                seed=SPLIT_SEED,
            )
            bootstraps[dataset] = bootstrap
            summaries[dataset]["target_student_cluster_bootstrap"] = bootstrap
    ci_pass = any(
        result["ci_low"] > 0.0 for result in bootstraps.values()
    )
    route_passed = bool(
        deterministic_gate["bootstrap_needed"] and ci_pass
    )
    aggregate = {
        "schema_version": 1,
        "route": "option_contrast_signal_v1",
        "manifest_sha256": sha256_file(manifest_path),
        "admission_sha256": sha256_file(admission_path),
        "repository_provenance": provenance,
        "datasets": summaries,
        "deterministic_gate": deterministic_gate,
        "bootstraps": bootstraps,
        "at_least_one_student_cluster_ci_low_above_zero": ci_pass,
        "signal_gate_passed": route_passed,
        "module_implementation_activated": route_passed,
        "no_multi_seed": True,
        "bootstrap_replicates": args.bootstrap_replicates,
    }
    aggregate_path = output_dir / "aggregate.json"
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "signal_gate_passed": route_passed,
                "deterministic_gate": deterministic_gate,
                "dataset_results": {
                    name: {
                        "signal_eligible": value.get("signal_eligible", True),
                        "stronger_control": value.get("stronger_control"),
                        "deltas": value.get("deltas_full_minus_control"),
                        "deterministic_pass": value[
                            "deterministic_dataset_pass"
                        ],
                        "target_rows": value.get("metrics", {}).get("full", {}).get("target", {}).get("rows"),
                    }
                    for name, value in summaries.items()
                },
                "bootstrap_ci_low": {
                    name: value["ci_low"]
                    for name, value in bootstraps.items()
                },
                "aggregate_path": str(aggregate_path.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
