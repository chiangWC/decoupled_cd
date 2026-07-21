from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.pool_protocol import sha256_file
from scripts.audit_option_contrast_signal import (
    _metrics,
    _target_mask,
    fast_student_cluster_bootstrap,
)
from scripts.audit_static_metadata_signal import (
    BOOTSTRAP_REPLICATES,
    MIN_TARGET_PER_LABEL,
    MIN_TARGET_ROWS,
    MIN_TARGET_STUDENTS,
    MODEL_SEED,
    NUM_FOLDS,
    RelationEdge,
    Protocol,
    SPLIT_SEED,
    _load_protocols,
    _q_edges,
    _q_frame,
    _read_protocol_csv,
    build_graph_kernel,
    build_pseudo_protocol,
    fit_fold,
    rewire_metadata,
)


XES_ADMISSION_SHA256 = (
    "1c896e30d410e92deaab803eb5ad7cf420ef667ddcb3a61839577e18974bf2ec"
)
PRIOR_RESULT_SHA256 = (
    "7829e3fdf31597ebcf469f0c0714fe6933b8158982a015c956a2f93305527406"
)
PRIOR_PREDICTION_SHA256 = {
    "assist09": (
        "035b43808c9b3987bc896e3ad2007fad9e04dc02170cd59668b13307d8f4f0a8"
    ),
    "junyi": (
        "b3bc9a9add8fc7303a6f60b9d6f24b1842d814ca783e73a72824adbc112db5b0"
    ),
}
XES_PROTOCOL_SHA256 = {
    "train.csv": (
        "27d0f8715176f83040048498f345ca1ca3af9a1dbaef9ea3083d1821f6621740"
    ),
    "Q_matrix.csv": (
        "13965c21cc2728281df235877805fcbf137bf851a4f9e232de6ec14620546df7"
    ),
}
TARGET_MAXIMUM_REGRESSION = 0.001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overall representation signal from static curriculum relations."
    )
    parser.add_argument("--static-admission-audit", required=True)
    parser.add_argument("--assist09-protocol", required=True)
    parser.add_argument("--assist09-raw", required=True)
    parser.add_argument("--nips-protocol", required=True)
    parser.add_argument("--nips-question-metadata", required=True)
    parser.add_argument("--nips-subject-metadata", required=True)
    parser.add_argument("--junyi-protocol", required=True)
    parser.add_argument("--junyi-official-log", required=True)
    parser.add_argument("--junyi-directed", required=True)
    parser.add_argument("--junyi-undirected", required=True)
    parser.add_argument("--junyi-source-repo", required=True)
    parser.add_argument("--xes-protocol", required=True)
    parser.add_argument("--xes-admission-audit", required=True)
    parser.add_argument("--prior-result-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=BOOTSTRAP_REPLICATES,
    )
    args = parser.parse_args()
    if args.bootstrap_replicates != BOOTSTRAP_REPLICATES:
        raise ValueError("The formal gate fixes 2,000 bootstrap replicates.")
    return args


def load_xes_protocol(
    protocol_dir: Path,
    admission_path: Path,
) -> Protocol:
    if sha256_file(admission_path) != XES_ADMISSION_SHA256:
        raise RuntimeError("XES admission audit hash mismatch.")
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    if admission.get("admitted") is not True:
        raise RuntimeError("XES static metadata was not admitted.")
    verified = {}
    for filename, expected in XES_PROTOCOL_SHA256.items():
        path = protocol_dir / filename
        digest = sha256_file(path)
        if digest != expected:
            raise RuntimeError(f"XES protocol hash mismatch for {filename}.")
        verified[filename] = digest
    train = _read_protocol_csv(protocol_dir / "train.csv")
    train["label"] = pd.to_numeric(train["label"], errors="raise").astype(int)
    q_frame, q_lookup = _q_frame(protocol_dir)
    metadata_edges = tuple(
        RelationEdge(
            relation=str(edge["relation"]),
            source=str(edge["source"]),
            target=str(edge["target"]),
            directed=bool(edge["directed"]),
        )
        for edge in admission["metadata_edges"]
    )
    return Protocol(
        name="xes3g5m",
        train=train.reset_index(drop=True),
        q_frame=q_frame,
        q_lookup=q_lookup,
        target_scope="bucket:zero",
        q_edges=_q_edges(q_lookup),
        metadata_edges=metadata_edges,
        audit={
            "xes_admission_sha256": XES_ADMISSION_SHA256,
            "protocol_hashes": verified,
            "tree_audit": admission["audit"],
            "opened_protocol_files": [
                str((protocol_dir / name).resolve())
                for name in ("train.csv", "Q_matrix.csv")
            ],
            "forbidden_protocol_files_not_opened": [
                str((protocol_dir / name).resolve())
                for name in ("valid.csv", "test.csv")
            ],
            "validation_or_test_opened": False,
        },
    )


def overall_feasibility(predictions: pd.DataFrame) -> dict[str, Any]:
    labels = predictions["label"].to_numpy(dtype=int)
    report = {
        "rows": len(predictions),
        "students": int(predictions["stu_id"].astype(str).nunique()),
        "label_0": int((labels == 0).sum()),
        "label_1": int((labels == 1).sum()),
        "minimum_rows": MIN_TARGET_ROWS,
        "minimum_students": MIN_TARGET_STUDENTS,
        "minimum_per_label": MIN_TARGET_PER_LABEL,
    }
    report["eligible"] = bool(
        report["rows"] >= MIN_TARGET_ROWS
        and report["students"] >= MIN_TARGET_STUDENTS
        and report["label_0"] >= MIN_TARGET_PER_LABEL
        and report["label_1"] >= MIN_TARGET_PER_LABEL
    )
    return report


def target_diagnostic(
    predictions: pd.DataFrame,
    *,
    scope: str,
    metrics: dict[str, dict[str, Any]],
    stronger_control: str,
) -> dict[str, Any]:
    target = _target_mask(predictions, scope)
    labels = predictions.loc[target, "label"].to_numpy(dtype=int)
    students = predictions.loc[target, "stu_id"].astype(str)
    feasibility = {
        "rows": int(target.sum()),
        "students": int(students.nunique()),
        "label_0": int((labels == 0).sum()),
        "label_1": int((labels == 1).sum()),
    }
    feasibility["eligible"] = bool(
        feasibility["rows"] >= MIN_TARGET_ROWS
        and feasibility["students"] >= MIN_TARGET_STUDENTS
        and feasibility["label_0"] >= MIN_TARGET_PER_LABEL
        and feasibility["label_1"] >= MIN_TARGET_PER_LABEL
    )
    if not feasibility["eligible"]:
        return {"scope": scope, "feasibility": feasibility, "eligible": False}
    target_metrics = {}
    for variant in metrics:
        target_metrics[variant] = _metrics(
            labels,
            predictions.loc[target, f"prob_{variant}"].to_numpy(dtype=float),
        )
    delta = (
        target_metrics["full"]["auc"]
        - target_metrics[stronger_control]["auc"]
    )
    return {
        "scope": scope,
        "feasibility": feasibility,
        "eligible": True,
        "metrics": target_metrics,
        "delta_auc_full_minus_primary_control": delta,
        "safety_pass": delta >= -TARGET_MAXIMUM_REGRESSION,
    }


def summarize_overall(
    predictions: pd.DataFrame,
    *,
    target_scope: str,
) -> dict[str, Any]:
    variants = [
        column.removeprefix("prob_")
        for column in predictions
        if column.startswith("prob_")
    ]
    labels = predictions["label"].to_numpy(dtype=int)
    metrics = {
        variant: {
            "overall": _metrics(
                labels,
                predictions[f"prob_{variant}"].to_numpy(dtype=float),
            )
        }
        for variant in variants
    }
    shuffles = [name for name in variants if name.startswith("shuffle_")]
    strongest_shuffle = max(
        shuffles,
        key=lambda name: metrics[name]["overall"]["auc"],
    )
    stronger_control = max(
        ("q_only", strongest_shuffle),
        key=lambda name: metrics[name]["overall"]["auc"],
    )
    full = metrics["full"]["overall"]
    control = metrics[stronger_control]["overall"]
    delta = {
        "overall_auc": full["auc"] - control["auc"],
        "overall_brier": full["brier"] - control["brier"],
    }
    full_beats_every_control = all(
        full["auc"] > metrics[name]["overall"]["auc"]
        for name in ("q_only", *shuffles)
    )
    diagnostic = target_diagnostic(
        predictions,
        scope=target_scope,
        metrics=metrics,
        stronger_control=stronger_control,
    )
    deterministic_pass = bool(
        delta["overall_auc"] >= 0.005
        and delta["overall_brier"] <= 0.0002
        and full_beats_every_control
        and (
            not diagnostic["eligible"]
            or diagnostic["safety_pass"]
        )
    )
    return {
        "primary_metric": "pseudo_overall_auc",
        "metrics": metrics,
        "strongest_shuffle": strongest_shuffle,
        "stronger_control": stronger_control,
        "deltas_full_minus_control": delta,
        "full_overall_auc_exceeds_every_control": full_beats_every_control,
        "target_diagnostic": diagnostic,
        "deterministic_dataset_pass": deterministic_pass,
    }


def reuse_prior_predictions(
    *,
    dataset: str,
    prior_dir: Path,
    target_scope: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    result_path = prior_dir / "result.json"
    if sha256_file(result_path) != PRIOR_RESULT_SHA256:
        raise RuntimeError("Prior static-signal result hash mismatch.")
    path = prior_dir / f"{dataset}_predictions.csv"
    if sha256_file(path) != PRIOR_PREDICTION_SHA256[dataset]:
        raise RuntimeError(f"Prior {dataset} prediction hash mismatch.")
    predictions = pd.read_csv(path, dtype={"source_row_id": str, "stu_id": str})
    summary = summarize_overall(predictions, target_scope=target_scope)
    summary.update(
        {
            "dataset": dataset,
            "prediction_path": str(path.resolve()),
            "prediction_sha256": PRIOR_PREDICTION_SHA256[dataset],
            "prediction_reused": True,
            "prior_result_sha256": PRIOR_RESULT_SHA256,
            "overall_feasibility": overall_feasibility(predictions),
        }
    )
    return summary, predictions


def run_fresh_dataset(
    protocol: Protocol,
    *,
    output_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    pseudo = build_pseudo_protocol(protocol)
    shuffles = {}
    shuffle_audits = {}
    for replicate in range(3):
        edges, audit = rewire_metadata(
            protocol.metadata_edges,
            dataset=protocol.name,
            replicate=replicate,
        )
        shuffles[f"shuffle_{replicate}"] = edges
        shuffle_audits[f"shuffle_{replicate}"] = audit
    kernels = {
        "full": build_graph_kernel(protocol.q_edges, protocol.metadata_edges),
        **{
            name: build_graph_kernel(protocol.q_edges, edges)
            for name, edges in shuffles.items()
        },
    }
    predictions = []
    fold_audits = []
    for fold in range(NUM_FOLDS):
        frame, audit = fit_fold(protocol, pseudo, kernels, fold)
        predictions.append(frame)
        fold_audits.append(audit)
    output = pd.concat(predictions, ignore_index=True)
    if output["source_row_id"].duplicated().any():
        raise RuntimeError("OOF predictions contain duplicate source rows.")
    output = output.sort_values("source_row_id", kind="stable").reset_index(drop=True)
    feasibility = overall_feasibility(output)
    if not feasibility["eligible"]:
        raise RuntimeError(
            f"{protocol.name} overall pseudo-target is ineligible: {feasibility}"
        )
    summary = summarize_overall(output, target_scope=protocol.target_scope)
    prediction_path = output_dir / f"{protocol.name}_predictions.csv"
    output.to_csv(prediction_path, index=False)
    summary.update(
        {
            "dataset": protocol.name,
            "prediction_path": str(prediction_path.resolve()),
            "prediction_sha256": sha256_file(prediction_path),
            "prediction_reused": False,
            "overall_feasibility": feasibility,
            "input_audit": protocol.audit,
            "pseudo_protocol": pseudo.audit,
            "full_graph": kernels["full"].audit,
            "shuffle_audits": shuffle_audits,
            "folds": fold_audits,
        }
    )
    (output_dir / f"{protocol.name}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary, output


def aggregate_gate(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    passing = {
        name
        for name, summary in summaries.items()
        if summary["deterministic_dataset_pass"]
    }
    maximum = max(
        (
            summaries[name]["deltas_full_minus_control"]["overall_auc"]
            for name in passing
        ),
        default=float("-inf"),
    )
    return {
        "passing_datasets": sorted(passing),
        "at_least_two_datasets": len(passing) >= 2,
        "new_xes_or_nips_required": bool(passing & {"xes3g5m", "nips34"}),
        "one_overall_delta_at_least_0_010": maximum >= 0.010,
        "bootstrap_needed": (
            len(passing) >= 2
            and bool(passing & {"xes3g5m", "nips34"})
            and maximum >= 0.010
        ),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocols = _load_protocols(args)
    protocols["xes3g5m"] = load_xes_protocol(
        Path(args.xes_protocol),
        Path(args.xes_admission_audit),
    )
    summaries = {}
    predictions = {}
    for name in ("assist09", "junyi"):
        summary, frame = reuse_prior_predictions(
            dataset=name,
            prior_dir=Path(args.prior_result_dir),
            target_scope=protocols[name].target_scope,
        )
        summaries[name] = summary
        predictions[name] = frame
    for name in ("nips34", "xes3g5m"):
        summary, frame = run_fresh_dataset(
            protocols[name],
            output_dir=output_dir,
        )
        summaries[name] = summary
        predictions[name] = frame
    aggregate = aggregate_gate(summaries)
    bootstraps = {}
    if aggregate["bootstrap_needed"]:
        for index, name in enumerate(aggregate["passing_datasets"]):
            frame = predictions[name]
            control = summaries[name]["stronger_control"]
            bootstraps[name] = fast_student_cluster_bootstrap(
                labels=frame["label"].to_numpy(dtype=int),
                students=frame["stu_id"].astype(str).to_numpy(),
                full_probability=frame["prob_full"].to_numpy(dtype=float),
                control_probability=frame[f"prob_{control}"].to_numpy(dtype=float),
                replicates=BOOTSTRAP_REPLICATES,
                seed=MODEL_SEED + index,
            )
    aggregate["bootstrap_ci_lower_bound_above_zero"] = any(
        report["ci_low"] > 0.0 for report in bootstraps.values()
    )
    aggregate["route_activated"] = bool(
        aggregate["at_least_two_datasets"]
        and aggregate["new_xes_or_nips_required"]
        and aggregate["one_overall_delta_at_least_0_010"]
        and aggregate["bootstrap_ci_lower_bound_above_zero"]
    )
    payload = {
        "schema_version": 1,
        "policy": {
            "primary_metric": "pseudo_overall_auc",
            "model_seed": MODEL_SEED,
            "split_seed": SPLIT_SEED,
            "student_disjoint_oof_folds": NUM_FOLDS,
            "minimum_overall_auc_delta": 0.005,
            "one_dataset_overall_auc_delta": 0.010,
            "maximum_overall_brier_increase": 0.0002,
            "maximum_target_auc_regression": TARGET_MAXIMUM_REGRESSION,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "multi_seed": False,
            "validation_or_test_opened": False,
        },
        "summaries": summaries,
        "aggregate_gate": aggregate,
        "bootstraps": bootstraps,
    }
    path = output_dir / "result.json"
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
