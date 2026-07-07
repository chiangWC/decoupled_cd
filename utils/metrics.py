from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

CalibrationBin = dict[str, float | int | None]


def _compute_calibration_bins(labels: np.ndarray, probs: np.ndarray, num_bins: int = 10) -> tuple[float, list[CalibrationBin]]:
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bins: list[CalibrationBin] = []
    ece = 0.0
    total = labels.size

    for bin_index in range(num_bins):
        lower = float(bin_edges[bin_index])
        upper = float(bin_edges[bin_index + 1])
        if bin_index == num_bins - 1:
            mask = (probs >= lower) & (probs <= upper)
        else:
            mask = (probs >= lower) & (probs < upper)
        count = int(mask.sum())

        if count > 0:
            accuracy = float(labels[mask].mean())
            confidence = float(probs[mask].mean())
            gap = abs(accuracy - confidence)
            if total > 0:
                ece += (count / total) * gap
        else:
            accuracy = None
            confidence = None
            gap = None

        bins.append(
            {
                "bin_lower": round(lower, 10),
                "bin_upper": round(upper, 10),
                "count": count,
                "accuracy": accuracy,
                "confidence": confidence,
                "gap": gap,
            }
        )

    return float(ece), bins


def compute_doa(
    *,
    mastery: np.ndarray,
    student_ids: np.ndarray,
    concept_lists: list[list[int]],
    labels: np.ndarray,
    min_responses: int = 1,
    max_pairs_per_concept: int = 100_000,
    seed: int = 2024,
) -> dict[str, float | int]:
    """
    Degree of Agreement: for each concept, over student pairs whose observed
    accuracy on that concept differs, the fraction where the model's mastery
    ordering agrees (mastery ties count 0.5). Reported as the unweighted mean
    over concepts and a pair-count-weighted mean. Pairs are subsampled per
    concept beyond ``max_pairs_per_concept`` for tractability.

    mastery: (num_students, num_concepts) array of diagnosed mastery levels.
    student_ids / concept_lists / labels: one entry per response; concept_lists
    holds the concept indices of the answered exercise.
    """
    mastery = np.asarray(mastery, dtype=np.float64)
    student_ids = np.asarray(student_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.float64)
    num_concepts = mastery.shape[1]

    correct_sums: dict[int, dict[int, float]] = {}
    attempt_sums: dict[int, dict[int, int]] = {}
    for row_index in range(student_ids.size):
        student = int(student_ids[row_index])
        for concept in concept_lists[row_index]:
            concept = int(concept)
            if concept < 0 or concept >= num_concepts:
                continue
            correct_sums.setdefault(concept, {})
            attempt_sums.setdefault(concept, {})
            correct_sums[concept][student] = correct_sums[concept].get(student, 0.0) + float(labels[row_index])
            attempt_sums[concept][student] = attempt_sums[concept].get(student, 0) + 1

    rng = np.random.default_rng(seed)
    per_concept_doa: list[float] = []
    per_concept_pairs: list[int] = []
    per_concept_spearman: list[float] = []
    for concept, attempts_by_student in attempt_sums.items():
        students = np.array(
            [s for s, n in attempts_by_student.items() if n >= min_responses],
            dtype=np.int64,
        )
        if students.size < 2:
            continue
        accuracy = np.array(
            [correct_sums[concept][int(s)] / attempts_by_student[int(s)] for s in students],
            dtype=np.float64,
        )
        concept_mastery = mastery[students, concept]

        total_pairs = students.size * (students.size - 1) // 2
        if total_pairs <= max_pairs_per_concept:
            left_idx, right_idx = np.triu_indices(students.size, k=1)
        else:
            left_idx = rng.integers(0, students.size, size=max_pairs_per_concept)
            right_idx = rng.integers(0, students.size, size=max_pairs_per_concept)
            keep = left_idx != right_idx
            left_idx, right_idx = left_idx[keep], right_idx[keep]

        acc_diff = accuracy[left_idx] - accuracy[right_idx]
        informative = acc_diff != 0.0
        if not informative.any():
            continue
        mastery_diff = concept_mastery[left_idx[informative]] - concept_mastery[right_idx[informative]]
        acc_diff = acc_diff[informative]
        concordant = (np.sign(mastery_diff) == np.sign(acc_diff)).astype(np.float64)
        concordant[mastery_diff == 0.0] = 0.5
        per_concept_doa.append(float(concordant.mean()))
        per_concept_pairs.append(int(concordant.size))
        # Continuous companion: rank correlation between mastery and observed accuracy.
        if np.std(accuracy) > 0 and np.std(concept_mastery) > 0:
            acc_rank = np.argsort(np.argsort(accuracy)).astype(np.float64)
            mas_rank = np.argsort(np.argsort(concept_mastery)).astype(np.float64)
            rho = np.corrcoef(acc_rank, mas_rank)[0, 1]
            if np.isfinite(rho):
                per_concept_spearman.append(float(rho))

    if not per_concept_doa:
        return {"doa": float("nan"), "doa_weighted": float("nan"), "num_concepts_evaluated": 0, "num_pairs": 0}
    doa_values = np.array(per_concept_doa, dtype=np.float64)
    pair_counts = np.array(per_concept_pairs, dtype=np.float64)
    # Bootstrap CI over concepts (resample the per-concept DOA set).
    boot = rng.choice(doa_values, size=(1000, doa_values.size), replace=True).mean(axis=1)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    spearman = float(np.mean(per_concept_spearman)) if per_concept_spearman else float("nan")
    return {
        "doa": float(doa_values.mean()),
        "doa_weighted": float((doa_values * pair_counts).sum() / pair_counts.sum()),
        "doa_ci_low": float(ci_low),
        "doa_ci_high": float(ci_high),
        "doa_spearman": spearman,
        "num_concepts_evaluated": int(doa_values.size),
        "num_pairs": int(pair_counts.sum()),
    }


def compute_metrics(labels, probs) -> dict[str, float | list[CalibrationBin]]:
    labels = np.asarray(labels, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    preds = (probs >= 0.5).astype(np.int64)

    try:
        auc = float(roc_auc_score(labels, probs))
    except Exception:
        auc = 0.5

    acc = float(accuracy_score(labels, preds))
    brier = float(mean_squared_error(labels, probs))
    rmse = float(math.sqrt(brier))
    ece, calibration_bins = _compute_calibration_bins(labels, probs)
    return {
        "auc": auc,
        "acc": acc,
        "rmse": rmse,
        "brier": brier,
        "ece": ece,
        "calibration_bins": calibration_bins,
    }
