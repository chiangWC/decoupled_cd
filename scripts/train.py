from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, UTC

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import apply_dataset_defaults
from data import prepare_experiment_split_bundles, prepare_step_data_bundle
from models import DecoupledCDM
from trainers import evaluate_model, train_model
from utils import append_summary_csv, resolve_device, save_history_csv, set_global_seed, setup_logging, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the minimal decoupled CDM pipeline.")
    parser.add_argument("--dataset", default=None, help="Optional dataset key for default paths and hyperparameters.")
    parser.add_argument("--interactions", default=None, help="Single interaction CSV with stu_id/exer_id/cpt_seq/label.")
    parser.add_argument("--train-interactions", default=None, help="Train split CSV.")
    parser.add_argument("--valid-interactions", default=None, help="Validation split CSV.")
    parser.add_argument("--test-interactions", default=None, help="Test split CSV.")
    parser.add_argument(
        "--q-matrix",
        dest="q_matrix",
        default=None,
        help="Optional Q-matrix CSV. If omitted, it is derived from unique exer_id/cpt_seq pairs in interactions.",
    )
    parser.add_argument("--concept-graph", default=None, help="Optional external concept graph CSV.")
    parser.add_argument("--prerequisite-graph", default=None, help="Optional prerequisite graph CSV for dual-graph mode.")
    parser.add_argument("--similarity-graph", default=None, help="Optional similarity graph CSV for dual-graph mode.")
    parser.add_argument("--graph-mode", choices=["single", "dual"], default="single")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--concept-dim", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gs-mode", choices=["constant", "conditional"], default="conditional")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpus", default=None, help="Optional comma-separated GPU candidates when --device auto.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap for quick smoke runs.")
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--lr-scheduler-patience", type=int, default=10)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--output", default="results/train_summary.json")
    args = parser.parse_args()
    return apply_dataset_defaults(args, parser)


def derive_q_matrix_if_needed(interactions_path: str, q_matrix_path: str | None) -> str:
    if q_matrix_path is not None:
        return q_matrix_path
    interactions = pd.read_csv(interactions_path)
    q_df = interactions[["exer_id", "cpt_seq"]].drop_duplicates().reset_index(drop=True)
    output_path = Path("results") / "derived_q_matrix.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    q_df.to_csv(output_path, index=False)
    return str(output_path)


def derive_q_matrix_from_splits_if_needed(
    train_path: str,
    valid_path: str,
    test_path: str,
    q_matrix_path: str | None,
) -> str:
    if q_matrix_path is not None:
        return q_matrix_path
    frames = [
        pd.read_csv(train_path, usecols=["exer_id", "cpt_seq"]),
        pd.read_csv(valid_path, usecols=["exer_id", "cpt_seq"]),
        pd.read_csv(test_path, usecols=["exer_id", "cpt_seq"]),
    ]
    q_df = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    output_path = Path("results") / "derived_q_matrix_splits.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    q_df.to_csv(output_path, index=False)
    return str(output_path)


def materialize_subset_if_needed(interactions_path: str, max_rows: int | None) -> str:
    if max_rows is None:
        return interactions_path
    interactions = pd.read_csv(interactions_path, nrows=max_rows)
    source_stem = Path(interactions_path).stem
    output_path = Path("results") / f"{source_stem}_subset_{max_rows}_interactions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    interactions.to_csv(output_path, index=False)
    return str(output_path)


def main() -> None:
    args = parse_args()
    if args.graph_mode == "dual" and (args.prerequisite_graph is None or args.similarity_graph is None):
        raise ValueError("Dual graph mode requires both --prerequisite-graph and --similarity-graph.")
    set_global_seed(args.seed)
    logger, log_path = setup_logging(args.log_dir, name="train")
    resolved_device = str(resolve_device(args.device, args.gpus))
    logger.info("Resolved device: %s", resolved_device)
    logger.info("Graph mode: %s", args.graph_mode)
    logger.info("Seed: %s", args.seed)
    using_splits = all([args.train_interactions, args.valid_interactions, args.test_interactions])
    if not using_splits and not args.interactions:
        raise ValueError("Provide either --interactions or all of --train-interactions/--valid-interactions/--test-interactions.")

    if using_splits:
        train_path = materialize_subset_if_needed(args.train_interactions, args.max_rows)
        valid_path = materialize_subset_if_needed(args.valid_interactions, args.max_rows)
        test_path = materialize_subset_if_needed(args.test_interactions, args.max_rows)
        q_matrix_source = derive_q_matrix_from_splits_if_needed(train_path, valid_path, test_path, args.q_matrix)
        logger.info("Using split mode with train=%s valid=%s test=%s", train_path, valid_path, test_path)
        bundles = prepare_experiment_split_bundles(
            train_interactions_path=train_path,
            valid_interactions_path=valid_path,
            test_interactions_path=test_path,
            q_matrix_path=q_matrix_source,
            concept_graph_path=args.concept_graph,
            prerequisite_graph_path=args.prerequisite_graph if args.graph_mode == "dual" else None,
            similarity_graph_path=args.similarity_graph if args.graph_mode == "dual" else None,
        )
        train_bundle = bundles["train"]
        valid_bundle = bundles["valid"]
        test_bundle = bundles["test"]
    else:
        interactions_path = materialize_subset_if_needed(args.interactions, args.max_rows)
        q_matrix_path = derive_q_matrix_if_needed(interactions_path, args.q_matrix)
        logger.info("Using single-file mode with interactions=%s", interactions_path)
        single_bundle = prepare_step_data_bundle(
            interactions_path=interactions_path,
            q_matrix_path=q_matrix_path,
            concept_graph_path=args.concept_graph,
            prerequisite_graph_path=args.prerequisite_graph if args.graph_mode == "dual" else None,
            similarity_graph_path=args.similarity_graph if args.graph_mode == "dual" else None,
        )
        train_bundle = single_bundle
        valid_bundle = None
        test_bundle = single_bundle

    model = DecoupledCDM(
        num_students=train_bundle.num_students,
        num_exercises=train_bundle.num_exercises,
        num_concepts=train_bundle.num_concepts,
        concept_dim=args.concept_dim,
        alpha=args.alpha,
        beta=args.beta,
        gs_mode=args.gs_mode,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(output_path.with_name(output_path.stem + "_best.pt"))
    result = train_model(
        train_bundle=train_bundle,
        valid_bundle=valid_bundle,
        model=model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=resolved_device,
        early_stop_patience=args.early_stop_patience,
        lr_scheduler_patience=args.lr_scheduler_patience,
        lr_scheduler_factor=args.lr_scheduler_factor,
        min_learning_rate=args.min_learning_rate,
        checkpoint_path=checkpoint_path,
    )
    test_metrics = evaluate_model(bundle=test_bundle, model=model, device=resolved_device)
    valid_metrics = evaluate_model(bundle=valid_bundle, model=model, device=resolved_device) if valid_bundle is not None else None

    output = {
        "dataset": args.dataset,
        "num_students": train_bundle.num_students,
        "num_exercises": train_bundle.num_exercises,
        "num_concepts": train_bundle.num_concepts,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "gs_mode": args.gs_mode,
        "graph_mode": args.graph_mode,
        "seed": args.seed,
        "device": resolved_device,
        "max_rows": args.max_rows,
        "final_loss": result.final_loss,
        "best_val_auc": result.best_val_auc,
        "best_epoch": result.best_epoch,
        "best_checkpoint_path": result.best_checkpoint_path,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "history": result.history,
        "log_path": log_path,
    }

    write_json(output, args.output)
    history_path = str(output_path.with_name(output_path.stem + "_history.csv"))
    save_history_csv(result.history, history_path)

    summary_row = {
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
        "train_interactions": args.train_interactions or args.interactions,
        "valid_interactions": args.valid_interactions,
        "test_interactions": args.test_interactions or args.interactions,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "lr_scheduler_patience": args.lr_scheduler_patience,
        "lr_scheduler_factor": args.lr_scheduler_factor,
        "min_learning_rate": args.min_learning_rate,
        "device": resolved_device,
        "concept_dim": args.concept_dim,
        "gs_mode": args.gs_mode,
        "graph_mode": args.graph_mode,
        "seed": args.seed,
        "best_epoch": result.best_epoch,
        "best_val_auc": result.best_val_auc,
        "test_auc": test_metrics["auc"],
        "test_acc": test_metrics["acc"],
        "test_rmse": test_metrics["rmse"],
        "best_checkpoint_path": str(Path(checkpoint_path).resolve()),
        "output_json": str(output_path.resolve()),
        "history_csv": str(Path(history_path).resolve()),
    }
    append_summary_csv(summary_row, "results/experiment_results.csv")
    logger.info(
        "Finished run: best_val_auc=%s test_auc=%.6f test_acc=%.6f test_rmse=%.6f",
        result.best_val_auc,
        test_metrics["auc"],
        test_metrics["acc"],
        test_metrics["rmse"],
    )

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
