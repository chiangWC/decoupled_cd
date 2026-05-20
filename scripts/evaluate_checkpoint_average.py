from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import prepare_experiment_split_bundles
from scripts.analyze_prediction_slices import load_model, predict_bundle
from utils import resolve_device, write_json
from utils.logging import setup_logging
from utils.metrics import compute_metrics


DATA_KEYS = (
    "train_interactions",
    "valid_interactions",
    "test_interactions",
    "q_matrix",
    "concept_graph",
    "graph_mode",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a pure checkpoint-average ensemble on a prepared train/valid/test split."
    )
    parser.add_argument("--summaries", nargs="+", required=True, help="Training summary JSON files to average.")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--average", choices=("prob", "logit"), default="prob")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _load_summary(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if "best_checkpoint_path" not in summary or not summary["best_checkpoint_path"]:
        raise ValueError(f"{path} is missing best_checkpoint_path.")
    if "concept_dim" not in summary:
        raise ValueError(f"{path} is missing concept_dim.")
    return summary


def _summary_data_signature(summary: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(summary.get(key) for key in DATA_KEYS)


def _validate_summary_compatibility(summary_paths: list[str], summaries: list[dict[str, Any]]) -> None:
    if len(summaries) < 2:
        raise ValueError("--summaries requires at least two checkpoint summaries.")
    reference = _summary_data_signature(summaries[0])
    for path, summary in zip(summary_paths[1:], summaries[1:]):
        if _summary_data_signature(summary) != reference:
            raise ValueError(f"{path} does not match the split/data signature of {summary_paths[0]}.")


def _average_predictions(predictions: list[np.ndarray], mode: str) -> np.ndarray:
    if not predictions:
        raise ValueError("At least one prediction array is required.")
    stacked = np.stack([np.asarray(prediction, dtype=np.float64) for prediction in predictions], axis=0)
    if mode == "prob":
        return stacked.mean(axis=0)
    if mode == "logit":
        clipped = np.clip(stacked, 1e-6, 1.0 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped))
        return 1.0 / (1.0 + np.exp(-logits.mean(axis=0)))
    raise ValueError(f"Unsupported average mode: {mode}")


def main() -> None:
    args = parse_args()
    logger, log_path = setup_logging(args.log_dir, name="checkpoint_average")
    summary_paths = [str(path) for path in args.summaries]
    summaries = [_load_summary(path) for path in summary_paths]
    _validate_summary_compatibility(summary_paths, summaries)

    first_summary = summaries[0]
    graph_mode = str(first_summary.get("graph_mode", "single"))
    if graph_mode != "single":
        raise ValueError("evaluate_checkpoint_average currently supports only graph_mode=single.")

    device = str(resolve_device(args.device, args.gpus))
    logger.info("Resolved device: %s", device)
    logger.info("Split: %s", args.split)
    logger.info("Average mode: %s", args.average)
    logger.info("Summaries: %s", ", ".join(summary_paths))

    bundles = prepare_experiment_split_bundles(
        train_interactions_path=first_summary["train_interactions"],
        valid_interactions_path=first_summary["valid_interactions"],
        test_interactions_path=first_summary["test_interactions"],
        q_matrix_path=first_summary["q_matrix"],
        concept_graph_path=first_summary["concept_graph"],
        prerequisite_graph_path=None,
        similarity_graph_path=None,
    )
    bundle = bundles[args.split]

    predictions: list[np.ndarray] = []
    labels: np.ndarray | None = None
    for summary in summaries:
        model = load_model(
            summary=summary,
            checkpoint_path=str(summary["best_checkpoint_path"]),
            bundles=bundles,
            concept_dim=int(summary["concept_dim"]),
            device=device,
        )
        frame = predict_bundle(bundle=bundle, model=model, device=device)
        predictions.append(frame["prob"].to_numpy(dtype=np.float64))
        if labels is None:
            labels = frame["label"].to_numpy(dtype=np.float64)

    if labels is None:
        raise ValueError("No labels were produced for evaluation.")

    averaged_probs = _average_predictions(predictions, args.average)
    metrics = compute_metrics(labels, averaged_probs)
    payload = {
        "summaries": summary_paths,
        "split": args.split,
        "average": args.average,
        "device": device,
        "log_path": log_path,
        "metrics": metrics,
    }
    write_json(payload, args.output)
    logger.info(
        "Finished checkpoint average: auc=%.6f acc=%.6f rmse=%.6f brier=%.6f ece=%.6f",
        metrics["auc"],
        metrics["acc"],
        metrics["rmse"],
        metrics["brier"],
        metrics["ece"],
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
