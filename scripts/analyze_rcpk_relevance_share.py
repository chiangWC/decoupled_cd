from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.mappings import build_unified_id_mappings
from data.q_matrix import build_q_matrix_tensor
from data.readers import read_interactions, read_q_matrix
from models import load_path_kernel_graph
from scripts.paired_student_bootstrap import paired_student_cluster_bootstrap


BIN_NAMES = ("low", "middle", "high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether target-conditioned RCPK gains track the share of "
            "train-history items connected to each target."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--full-predictions", required=True)
    parser.add_argument("--global-predictions", required=True)
    parser.add_argument("--train-interactions", required=True)
    parser.add_argument("--valid-interactions", required=True)
    parser.add_argument("--q-matrix", required=True)
    parser.add_argument("--relation-graph", required=True)
    parser.add_argument("--hops", type=int, default=4)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--output", required=True)
    parser.add_argument("--row-output", required=True)
    args = parser.parse_args()
    if args.hops <= 0 or args.replicates < 100:
        raise ValueError("hops must be positive and replicates at least 100")
    return args


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _incoming_neighbors(transition_t) -> list[list[int]]:
    transition_t = transition_t.coalesce()
    incoming: list[list[int]] = [[] for _ in range(transition_t.shape[0])]
    targets, sources = transition_t.indices().cpu().numpy()
    for target, source in zip(targets.tolist(), sources.tolist(), strict=True):
        incoming[int(target)].append(int(source))
    return incoming


def reachable_history_sources(
    *,
    target: int,
    incoming_q: list[list[int]],
    incoming_metadata: list[list[int]],
    num_exercises: int,
    hops: int,
) -> set[int]:
    """Return exercise sources reaching target through at least one metadata edge."""

    frontier = {(int(target), False)}
    visited = set(frontier)
    reachable: set[int] = set()
    for _ in range(hops):
        next_frontier: set[tuple[int, bool]] = set()
        for node, used_metadata in frontier:
            for source in incoming_q[node]:
                state = (source, used_metadata)
                if state not in visited:
                    next_frontier.add(state)
            for source in incoming_metadata[node]:
                state = (source, True)
                if state not in visited:
                    next_frontier.add(state)
        for source, used_metadata in next_frontier:
            if used_metadata and source < num_exercises:
                reachable.add(source)
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return reachable


def _validate_predictions(
    full: pd.DataFrame,
    control: pd.DataFrame,
    valid: pd.DataFrame,
    student_map: dict[str, int],
    exercise_map: dict[str, int],
) -> None:
    required = {"stu_id", "exer_id", "label", "prob"}
    if not required.issubset(full) or not required.issubset(control):
        raise ValueError("Prediction files lack required aligned columns")
    if len(full) != len(control) or len(full) != len(valid):
        raise ValueError("Prediction and validation row counts differ")
    for column in ("stu_id", "exer_id", "label"):
        if not np.array_equal(full[column].to_numpy(), control[column].to_numpy()):
            raise ValueError(f"Full/control rows differ in {column}")
    expected_students = np.asarray(
        [student_map[str(value)] for value in valid["stu_id"]], dtype=int
    )
    expected_exercises = np.asarray(
        [exercise_map[str(value)] for value in valid["exer_id"]], dtype=int
    )
    expected_labels = valid["label"].to_numpy(dtype=float)
    if not np.array_equal(full["stu_id"].to_numpy(dtype=int), expected_students):
        raise ValueError("Prediction student IDs do not match dense validation mapping")
    if not np.array_equal(full["exer_id"].to_numpy(dtype=int), expected_exercises):
        raise ValueError("Prediction exercise IDs do not match dense validation mapping")
    if not np.array_equal(full["label"].to_numpy(dtype=float), expected_labels):
        raise ValueError("Prediction labels do not match validation rows")


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    labels = frame["label"].to_numpy(dtype=float)
    full_probs = frame["full_prob"].to_numpy(dtype=float)
    global_probs = frame["global_prob"].to_numpy(dtype=float)
    both_labels = np.unique(labels).size == 2
    return {
        "rows": int(len(frame)),
        "students": int(frame["stu_id"].nunique()),
        "mean_relevance_share": float(frame["relevance_share"].mean()),
        "mean_history_count": float(frame["history_count"].mean()),
        "same_item_history_rate": float(frame["same_item_history"].mean()),
        "full_auc": float(roc_auc_score(labels, full_probs)) if both_labels else None,
        "global_auc": (
            float(roc_auc_score(labels, global_probs)) if both_labels else None
        ),
        "full_log_loss": float(log_loss(labels, full_probs, labels=[0, 1])),
        "global_log_loss": float(log_loss(labels, global_probs, labels=[0, 1])),
        "mean_global_damage": float(frame["global_damage"].mean()),
        "full_brier": float(np.mean((full_probs - labels) ** 2)),
        "global_brier": float(np.mean((global_probs - labels) ** 2)),
    }


def _clustered_low_high_bootstrap(
    frame: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    eligible = frame[frame["relevance_bin"].isin(["low", "high"])].copy()
    students, inverse = np.unique(
        eligible["stu_id"].astype(str).to_numpy(), return_inverse=True
    )
    low = eligible["relevance_bin"].to_numpy() == "low"
    high = eligible["relevance_bin"].to_numpy() == "high"
    damage = eligible["global_damage"].to_numpy(dtype=float)

    def contrast(weights: np.ndarray) -> float:
        low_weight = weights[low].sum()
        high_weight = weights[high].sum()
        if low_weight <= 0.0 or high_weight <= 0.0:
            return float("nan")
        return float(
            np.dot(weights[low], damage[low]) / low_weight
            - np.dot(weights[high], damage[high]) / high_weight
        )

    observed = contrast(np.ones(len(eligible), dtype=float))
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(replicates):
        sampled = rng.integers(0, len(students), size=len(students))
        counts = np.bincount(sampled, minlength=len(students))
        value = contrast(counts[inverse].astype(float))
        if np.isfinite(value):
            values.append(value)
    if len(values) < max(100, int(0.9 * replicates)):
        raise RuntimeError("Too many low/high bootstrap replicates were invalid")
    samples = np.asarray(values)
    return {
        "estimand": "mean_global_damage_low_minus_high",
        "observed": observed,
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "probability_positive": float(np.mean(samples > 0.0)),
        "replicates_requested": int(replicates),
        "replicates_used": int(len(samples)),
        "bootstrap_seed": int(seed),
    }


def main() -> None:
    args = parse_args()
    train = read_interactions(args.train_interactions)
    valid = read_interactions(args.valid_interactions)
    q_matrix = read_q_matrix(args.q_matrix)
    mappings = build_unified_id_mappings(
        pd.concat([train, valid], ignore_index=True), q_matrix
    )
    q_tensor = build_q_matrix_tensor(
        q_matrix,
        mappings["exercise_id_map"],
        mappings["concept_id_map"],
    )
    graph = load_path_kernel_graph(
        args.relation_graph,
        mode="relational",
        q_matrix=q_tensor,
        exercise_id_map=mappings["exercise_id_map"],
        concept_id_map=mappings["concept_id_map"],
    )
    full = pd.read_csv(args.full_predictions)
    control = pd.read_csv(args.global_predictions)
    _validate_predictions(
        full,
        control,
        valid,
        mappings["student_id_map"],
        mappings["exercise_id_map"],
    )

    histories = [set() for _ in mappings["student_id_map"]]
    for row in train.itertuples(index=False):
        student = mappings["student_id_map"][str(row.stu_id)]
        exercise = mappings["exercise_id_map"][str(row.exer_id)]
        histories[student].add(exercise)

    incoming_q = _incoming_neighbors(graph.q_transition_t)
    incoming_metadata = _incoming_neighbors(graph.metadata_transition_t)
    targets = full["exer_id"].to_numpy(dtype=int)
    reachable_by_target = {
        int(target): reachable_history_sources(
            target=int(target),
            incoming_q=incoming_q,
            incoming_metadata=incoming_metadata,
            num_exercises=graph.num_exercises,
            hops=args.hops,
        )
        for target in np.unique(targets)
    }

    students = full["stu_id"].to_numpy(dtype=int)
    history_count = np.asarray([len(histories[s]) for s in students], dtype=int)
    reachable_count = np.asarray(
        [
            len(histories[s].intersection(reachable_by_target[int(target)]))
            for s, target in zip(students, targets, strict=True)
        ],
        dtype=int,
    )
    relevance_share = reachable_count / np.maximum(history_count, 1)
    same_item = np.asarray(
        [target in histories[s] for s, target in zip(students, targets, strict=True)]
    )
    labels = full["label"].to_numpy(dtype=float)
    full_probs = np.clip(full["prob"].to_numpy(dtype=float), 1e-7, 1.0 - 1e-7)
    global_probs = np.clip(
        control["prob"].to_numpy(dtype=float), 1e-7, 1.0 - 1e-7
    )
    full_loss = -(labels * np.log(full_probs) + (1.0 - labels) * np.log(1.0 - full_probs))
    global_loss = -(
        labels * np.log(global_probs)
        + (1.0 - labels) * np.log(1.0 - global_probs)
    )

    rows = pd.DataFrame(
        {
            "row_index": np.arange(len(full)),
            "stu_id": students,
            "exer_id": targets,
            "label": labels,
            "full_prob": full_probs,
            "global_prob": global_probs,
            "history_count": history_count,
            "reachable_history_count": reachable_count,
            "relevance_share": relevance_share,
            "same_item_history": same_item,
            "global_damage": global_loss - full_loss,
            "relevance_bin": "ineligible",
        }
    )
    positive = (rows["history_count"] >= 5) & (rows["reachable_history_count"] > 0)
    ranked = rows.loc[positive, "relevance_share"].rank(method="first")
    rows.loc[positive, "relevance_bin"] = pd.qcut(
        ranked, q=3, labels=BIN_NAMES
    ).astype(str)
    rows.loc[(rows["history_count"] >= 5) & ~positive, "relevance_bin"] = "zero"

    bin_metrics = {
        name: _metrics(rows[rows["relevance_bin"] == name])
        for name in (*BIN_NAMES, "zero", "ineligible")
        if (rows["relevance_bin"] == name).any()
    }
    overall_bootstrap = paired_student_cluster_bootstrap(
        full=full,
        control=control,
        scope="overall",
        replicates=args.replicates,
        seed=args.seed,
    )
    trend_bootstrap = _clustered_low_high_bootstrap(
        rows,
        replicates=args.replicates,
        seed=args.seed,
    )
    result = {
        "dataset": args.dataset,
        "definition": (
            "distinct train-history items reaching target within at most four "
            "RCPK edges and using at least one metadata edge / distinct history items"
        ),
        "hops": int(args.hops),
        "graph_source_sha256": graph.source_sha256,
        "graph_edge_sha256": graph.edge_sha256,
        "full_prediction_sha256": _sha256(args.full_predictions),
        "global_prediction_sha256": _sha256(args.global_predictions),
        "overall": _metrics(rows),
        "positive_share_rows": int(positive.sum()),
        "zero_share_rows": int(((rows["history_count"] >= 5) & ~positive).sum()),
        "short_history_rows": int((rows["history_count"] < 5).sum()),
        "bins": bin_metrics,
        "paired_auc_bootstrap": overall_bootstrap,
        "low_minus_high_damage_bootstrap": trend_bootstrap,
    }
    output = Path(args.output)
    row_output = Path(args.row_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    row_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    rows.to_csv(row_output, index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
