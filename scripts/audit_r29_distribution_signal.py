from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.r29_protocol import stable_fraction


RFF_DIM = 32
MASK_FRACTION = 0.2
RFF_BANDWIDTH = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train-only response-distribution activation audit for the post-r29 round."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-fraction", type=float, default=MASK_FRACTION)
    parser.add_argument("--rff-dim", type=int, default=RFF_DIM)
    parser.add_argument("--rff-bandwidth", type=float, default=RFF_BANDWIDTH)
    parser.add_argument("--max-heldout-rows", type=int, default=200000)
    return parser.parse_args()


def rff_parameters(*, dim: int, bandwidth: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if dim <= 0 or bandwidth <= 0:
        raise ValueError("RFF dimension and bandwidth must be positive.")
    rng = np.random.default_rng(seed)
    weights = rng.normal(size=(2, dim)) / bandwidth
    phases = rng.uniform(0.0, 2.0 * math.pi, size=dim)
    return weights, phases


def rff_features(
    values: np.ndarray,
    *,
    weights: np.ndarray,
    phases: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Distribution tokens must have shape [rows, 2].")
    return math.sqrt(2.0 / weights.shape[1]) * np.cos(values @ weights + phases)


def prepare_context(
    frame: pd.DataFrame,
    *,
    seed: int,
    mask_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"stu_id", "exer_id", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Training frame is missing columns: {sorted(required - set(frame))}")
    work = frame.copy().reset_index(drop=True)
    row_keys = (
        work["source_row_id"].astype(str)
        if "source_row_id" in work.columns
        else work.index.astype(str)
    )
    work["_heldout"] = [
        stable_fraction(seed, "distribution-proxy-mask", row_key) < mask_fraction
        for row_key in row_keys
    ]
    context = work[~work["_heldout"]].copy()
    heldout = work[work["_heldout"]].copy()
    if context.empty or heldout.empty:
        raise RuntimeError("Distribution proxy requires nonempty context and pseudo-holdout rows.")
    return context, heldout


def item_statistics(context: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    global_rate = float(context["label"].mean())
    grouped = context.groupby("exer_id")["label"].agg(["sum", "count"])
    grouped["item_rate"] = (grouped["sum"] + 2.0 * global_rate) / (grouped["count"] + 2.0)
    grouped["item_support"] = np.log1p(grouped["count"])
    return grouped[["item_rate", "item_support"]], global_rate


def student_representations(
    context: pd.DataFrame,
    *,
    item_stats: pd.DataFrame,
    global_rate: float,
    weights: np.ndarray,
    phases: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, tuple[float, float]]]:
    enriched = context.join(item_stats, on="exer_id")
    enriched["item_rate"] = enriched["item_rate"].fillna(global_rate)
    enriched["signed_item_rate"] = 2.0 * enriched["item_rate"] - 1.0
    enriched["signed_response"] = 2.0 * enriched["label"].astype(float) - 1.0
    tokens = enriched[["signed_item_rate", "signed_response"]].to_numpy(dtype=np.float64)
    embedded = rff_features(tokens, weights=weights, phases=phases)
    full: dict[str, np.ndarray] = {}
    control: dict[str, np.ndarray] = {}
    summaries: dict[str, tuple[float, float]] = {}
    enriched["stu_id"] = enriched["stu_id"].astype(str)
    for student, indices_value in enriched.groupby("stu_id", sort=False).indices.items():
        indices = np.asarray(indices_value, dtype=np.int64)
        student_tokens = tokens[indices]
        full[student] = embedded[indices].mean(axis=0)
        mean_token = student_tokens.mean(axis=0, keepdims=True)
        control[student] = rff_features(
            mean_token, weights=weights, phases=phases
        ).reshape(-1)
        summaries[student] = (
            float(enriched.iloc[indices]["label"].mean()),
            float(np.log1p(len(indices))),
        )
    return full, control, summaries


def design_matrix(
    rows: pd.DataFrame,
    *,
    representations: dict[str, np.ndarray],
    summaries: dict[str, tuple[float, float]],
    item_stats: pd.DataFrame,
    global_rate: float,
    rff_dim: int,
) -> np.ndarray:
    enriched = rows.join(item_stats, on="exer_id")
    enriched["item_rate"] = enriched["item_rate"].fillna(global_rate)
    enriched["item_support"] = enriched["item_support"].fillna(0.0)
    output: list[np.ndarray] = []
    zero = np.zeros(rff_dim, dtype=np.float64)
    for row in enriched.itertuples(index=False):
        student = str(row.stu_id)
        success, support = summaries.get(student, (global_rate, 0.0))
        output.append(
            np.concatenate(
                [
                    representations.get(student, zero),
                    np.asarray(
                        [success, support, float(row.item_rate), float(row.item_support)],
                        dtype=np.float64,
                    ),
                ]
            )
        )
    return np.stack(output)


def evaluate_representation(
    *,
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    evaluation_x: np.ndarray,
    evaluation_y: np.ndarray,
    seed: int,
) -> dict[str, float]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=500, random_state=seed),
    )
    model.fit(fit_x, fit_y)
    probabilities = model.predict_proba(evaluation_x)[:, 1]
    return {
        "auc": float(roc_auc_score(evaluation_y, probabilities)),
        "brier": float(brier_score_loss(evaluation_y, probabilities)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    frame = pd.read_csv(args.train, dtype={"stu_id": str, "exer_id": str})
    context, heldout = prepare_context(
        frame, seed=args.seed, mask_fraction=args.mask_fraction
    )
    heldout["_sample_key"] = [
        stable_fraction(args.seed, "distribution-proxy-sample", index)
        for index in heldout.index.astype(str)
    ]
    heldout = heldout.nsmallest(min(args.max_heldout_rows, len(heldout)), "_sample_key")
    heldout["_fit"] = [
        stable_fraction(args.seed, "distribution-proxy-fit", student) < 0.7
        for student in heldout["stu_id"].astype(str)
    ]
    fit = heldout[heldout["_fit"]]
    evaluation = heldout[~heldout["_fit"]]
    if min(len(fit), len(evaluation)) < 100:
        raise RuntimeError("Distribution proxy has insufficient fit/evaluation rows.")
    if set(fit["label"].astype(int)) != {0, 1} or set(evaluation["label"].astype(int)) != {0, 1}:
        raise RuntimeError("Distribution proxy fit/evaluation labels must both be binary.")

    stats, global_rate = item_statistics(context)
    weights, phases = rff_parameters(
        dim=args.rff_dim, bandwidth=args.rff_bandwidth, seed=args.seed
    )
    full, control, summaries = student_representations(
        context,
        item_stats=stats,
        global_rate=global_rate,
        weights=weights,
        phases=phases,
    )
    results: dict[str, dict[str, float]] = {}
    for name, representations in (("distribution", full), ("marginal_control", control)):
        fit_x = design_matrix(
            fit,
            representations=representations,
            summaries=summaries,
            item_stats=stats,
            global_rate=global_rate,
            rff_dim=args.rff_dim,
        )
        evaluation_x = design_matrix(
            evaluation,
            representations=representations,
            summaries=summaries,
            item_stats=stats,
            global_rate=global_rate,
            rff_dim=args.rff_dim,
        )
        results[name] = evaluate_representation(
            fit_x=fit_x,
            fit_y=fit["label"].astype(int).to_numpy(),
            evaluation_x=evaluation_x,
            evaluation_y=evaluation["label"].astype(int).to_numpy(),
            seed=args.seed,
        )
    delta_auc = results["distribution"]["auc"] - results["marginal_control"]["auc"]
    delta_brier = (
        results["distribution"]["brier"] - results["marginal_control"]["brier"]
    )
    return {
        "schema_version": 1,
        "dataset": args.dataset,
        "seed": args.seed,
        "history_source": "train_only",
        "mask_fraction": args.mask_fraction,
        "fit_students_disjoint_from_evaluation_students": True,
        "fit_rows": len(fit),
        "evaluation_rows": len(evaluation),
        "representation_dim": args.rff_dim,
        "downstream_parameter_count_equal": True,
        "distribution": results["distribution"],
        "marginal_control": results["marginal_control"],
        "delta_auc": delta_auc,
        "delta_brier": delta_brier,
        "activation_threshold": {"min_delta_auc": 0.002, "max_delta_brier": 0.0},
        "passed": delta_auc >= 0.002 and delta_brier <= 0.0,
    }


def main() -> None:
    args = parse_args()
    if (
        args.seed != 42
        or args.mask_fraction != MASK_FRACTION
        or args.rff_dim != RFF_DIM
        or args.rff_bandwidth != RFF_BANDWIDTH
    ):
        raise ValueError(
            "The response-distribution audit is fixed to seed42, mask=0.2, "
            "rff_dim=32 and bandwidth=0.5."
        )
    payload = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
