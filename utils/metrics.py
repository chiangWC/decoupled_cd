from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score


def compute_metrics(labels, probs) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    preds = (probs >= 0.5).astype(np.int64)

    try:
        auc = float(roc_auc_score(labels, probs))
    except Exception:
        auc = 0.5

    acc = float(accuracy_score(labels, preds))
    rmse = float(math.sqrt(mean_squared_error(labels, probs)))
    return {"auc": auc, "acc": acc, "rmse": rmse}
