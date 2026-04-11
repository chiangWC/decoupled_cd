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
