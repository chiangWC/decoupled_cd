from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


GRAPH_MODELS = ("ORCDF-NCD", "SVGCD", "RCD", "HyperCD")
ALIGNMENT = ("audit_row_id", "stu_id", "exer_id", "label")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Gate D graph-family and graph-vs-KaNCD effects."
    )
    parser.add_argument(
        "--graph-model",
        action="append",
        nargs=3,
        required=True,
        metavar=("MODEL", "PAIRED_DIR", "SUMMARY_JSON"),
    )
    parser.add_argument(
        "--non-graph",
        nargs=3,
        required=True,
        metavar=("MODEL", "PAIRED_DIR", "SUMMARY_JSON"),
    )
    parser.add_argument(
        "--model-failure",
        action="append",
        nargs=3,
        default=[],
        metavar=("MODEL", "DATASET", "REASON"),
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["dataset"]): row for row in payload["datasets"]}


def load_rows(directory: Path, dataset: str) -> pd.DataFrame:
    path = directory / f"{dataset}_paired_rows.csv"
    frame = pd.read_csv(path)
    required = {*ALIGNMENT, "concept_prob", "random_prob", "log_loss_damage"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing {missing}.")
    for column in ALIGNMENT[:3]:
        frame[column] = frame[column].astype(str)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if frame["audit_row_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate audit rows.")
    return frame


def assert_aligned(left: pd.DataFrame, right: pd.DataFrame) -> None:
    if len(left) != len(right):
        raise ValueError("Graph/non-graph paired-row counts differ.")
    for column in ALIGNMENT:
        if not np.array_equal(left[column].to_numpy(), right[column].to_numpy()):
            raise ValueError(f"Graph/non-graph alignment failed for {column}.")


def intervals(values: list[float]) -> tuple[float, float]:
    finite = np.asarray([value for value in values if np.isfinite(value)])
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def interaction(
    graph: pd.DataFrame,
    non_graph: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    assert_aligned(graph, non_graph)
    log_values = (
        graph["log_loss_damage"].to_numpy(float)
        - non_graph["log_loss_damage"].to_numpy(float)
    )
    point_log = float(log_values.mean())
    point_auc = float(
        (
            roc_auc_score(graph["label"], graph["random_prob"])
            - roc_auc_score(graph["label"], graph["concept_prob"])
        )
        - (
            roc_auc_score(non_graph["label"], non_graph["random_prob"])
            - roc_auc_score(non_graph["label"], non_graph["concept_prob"])
        )
    )
    students = graph["stu_id"].drop_duplicates().to_numpy()
    by_student = {
        student: graph.index[graph["stu_id"].eq(student)].to_numpy()
        for student in students
    }
    rng = np.random.default_rng(seed)
    boot_log: list[float] = []
    boot_auc: list[float] = []
    for _ in range(replicates):
        sampled = rng.choice(students, size=len(students), replace=True)
        indices = np.concatenate([by_student[student] for student in sampled])
        boot_log.append(float(log_values[indices].mean()))
        labels = graph.loc[indices, "label"]
        if labels.nunique() == 2:
            boot_auc.append(
                float(
                    (
                        roc_auc_score(labels, graph.loc[indices, "random_prob"])
                        - roc_auc_score(
                            labels, graph.loc[indices, "concept_prob"]
                        )
                    )
                    - (
                        roc_auc_score(
                            labels, non_graph.loc[indices, "random_prob"]
                        )
                        - roc_auc_score(
                            labels, non_graph.loc[indices, "concept_prob"]
                        )
                    )
                )
            )
    log_low, log_high = intervals(boot_log)
    auc_low, auc_high = intervals(boot_auc)
    return {
        "log_loss_interaction": point_log,
        "log_loss_ci_low": log_low,
        "log_loss_ci_high": log_high,
        "log_loss_support": point_log > 0 and log_low > 0,
        "auc_interaction": point_auc,
        "auc_ci_low": auc_low,
        "auc_ci_high": auc_high,
    }


def analyze(args: argparse.Namespace) -> tuple[dict[str, Any], pd.DataFrame]:
    if args.seed != 2024 or args.bootstrap != 2000:
        raise ValueError("Gate D fixes bootstrap=2000 and seed=2024.")
    graph_specs = {
        model: (Path(directory), Path(summary))
        for model, directory, summary in args.graph_model
    }
    if set(graph_specs) != set(GRAPH_MODELS):
        raise ValueError(f"Gate D requires exactly {GRAPH_MODELS}.")
    non_name, non_dir_raw, non_summary_raw = args.non_graph
    if non_name != "EduCDM-KaNCD-GMF":
        raise ValueError("Gate D non-graph reference is EduCDM-KaNCD-GMF.")
    failures = {
        (model, dataset): reason
        for model, dataset, reason in args.model_failure
    }
    if len(failures) != len(args.model_failure):
        raise ValueError("Duplicate Gate D model failure.")
    if any(model not in GRAPH_MODELS for model, _ in failures):
        raise ValueError("Gate D failure names an unknown graph model.")
    summaries = {
        model: load_summary(summary)
        for model, (_, summary) in graph_specs.items()
    }
    non_summary = load_summary(Path(non_summary_raw))
    datasets = set(non_summary)
    for model, rows in summaries.items():
        failed = {
            dataset
            for failed_model, dataset in failures
            if failed_model == model
        }
        if set(rows).intersection(failed):
            raise ValueError(f"{model}: result and failure overlap.")
        if set(rows).union(failed) != datasets:
            raise ValueError(
                f"{model}: results plus failures do not cover Gate D datasets."
            )
    output_rows: list[dict[str, Any]] = []
    admitted_datasets: list[str] = []
    for dataset in sorted(datasets):
        non_rows = load_rows(Path(non_dir_raw), dataset)
        graph_support_count = 0
        interaction_support_count = 0
        for model in GRAPH_MODELS:
            failure = failures.get((model, dataset))
            if failure is not None:
                output_rows.append(
                    {
                        "dataset": dataset,
                        "graph_model": model,
                        "graph_damage": None,
                        "graph_support": False,
                        "non_graph_damage": non_summary[dataset][
                            "log_loss_damage_concept_minus_random"
                        ],
                        "model_failure": failure,
                        "log_loss_support": False,
                    }
                )
                continue
            graph_rows = load_rows(graph_specs[model][0], dataset)
            effect = interaction(
                graph_rows,
                non_rows,
                replicates=args.bootstrap,
                seed=args.seed,
            )
            supports = bool(
                summaries[model][dataset][
                    "supports_concept_specific_damage"
                ]
            )
            graph_support_count += int(supports)
            interaction_support_count += int(effect["log_loss_support"])
            output_rows.append(
                {
                    "dataset": dataset,
                    "graph_model": model,
                    "graph_damage": summaries[model][dataset][
                        "log_loss_damage_concept_minus_random"
                    ],
                    "graph_support": supports,
                    "non_graph_damage": non_summary[dataset][
                        "log_loss_damage_concept_minus_random"
                    ],
                    "model_failure": None,
                    **effect,
                }
            )
        supports_family = (
            graph_support_count >= 3 and interaction_support_count >= 2
        )
        if supports_family:
            admitted_datasets.append(dataset)
        for row in output_rows[-len(GRAPH_MODELS) :]:
            row["graph_support_count"] = graph_support_count
            row["interaction_support_count"] = interaction_support_count
            row["dataset_supports_graph_family"] = supports_family
    payload = {
        "schema_version": 1,
        "gate": "graph_family_concept_depletion_gate_d",
        "graph_models": list(GRAPH_MODELS),
        "non_graph_model": non_name,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "model_failures": [
            {"model": model, "dataset": dataset, "reason": reason}
            for (model, dataset), reason in sorted(failures.items())
        ],
        "dataset_rule": {
            "minimum_graph_support": 3,
            "minimum_positive_interactions": 2,
        },
        "admitted_datasets": admitted_datasets,
        "admitted_dataset_count": len(admitted_datasets),
        "required_datasets": 3,
        "admitted": len(admitted_datasets) >= 3,
        "decision": (
            "admit_graph_family_problem"
            if len(admitted_datasets) >= 3
            else "reject_graph_family_problem"
        ),
    }
    return payload, pd.DataFrame(output_rows)


def main() -> None:
    args = parse_args()
    payload, table = analyze(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "gate_d_model_interactions.csv", index=False)
    (args.output_dir / "gate_d_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
