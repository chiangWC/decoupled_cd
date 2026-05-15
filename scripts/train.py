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
    parser.add_argument(
        "--prerequisite-graph",
        default=None,
        help="Legacy dual-graph ablation input. Not used by the current single-graph mainline.",
    )
    parser.add_argument(
        "--similarity-graph",
        default=None,
        help="Legacy dual-graph ablation input. Not used by the current single-graph mainline.",
    )
    parser.add_argument(
        "--graph-mode",
        choices=["single", "dual"],
        default="single",
        help="Use 'single' for the current mainline. 'dual' is kept only for historical ablations.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Mini-batch size used only when --training-mode recompute_minibatch.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--training-mode",
        choices=["full_batch", "recompute_minibatch"],
        default="full_batch",
        help="Training loop semantics. full_batch preserves the current mainline; recompute_minibatch makes --batch-size effective.",
    )
    parser.add_argument("--concept-dim", type=int, default=32)
    parser.add_argument(
        "--student-gate-prior-alpha",
        type=float,
        default=None,
        help="TKC prior used only to initialize the student fusion gate; training remains adaptive.",
    )
    parser.add_argument(
        "--student-gate-prior-beta",
        type=float,
        default=None,
        help="UKC prior used only to initialize the student fusion gate; training remains adaptive.",
    )
    parser.add_argument(
        "--alpha",
        dest="legacy_alpha",
        type=float,
        default=None,
        help="Deprecated alias for --student-gate-prior-alpha.",
    )
    parser.add_argument(
        "--beta",
        dest="legacy_beta",
        type=float,
        default=None,
        help="Deprecated alias for --student-gate-prior-beta.",
    )
    parser.add_argument("--gs-mode", choices=["constant", "conditional"], default="conditional")
    parser.add_argument(
        "--high-concept-logit-adapter",
        action="store_true",
        help="Enable a zero-init logit residual applied only to higher concept-count exercises.",
    )
    parser.add_argument(
        "--high-concept-logit-min-count",
        type=int,
        default=3,
        help="Minimum concept count required before the high-concept logit adapter is applied.",
    )
    parser.add_argument(
        "--pairwise-history-interaction-adapter",
        action="store_true",
        help="Enable a shared pairwise concept scorer driven by explicit per-concept history statistics.",
    )
    parser.add_argument(
        "--pairwise-history-interaction-min-count",
        type=int,
        default=2,
        help="Minimum concept count required before the history-carrier pairwise residual is applied.",
    )
    parser.add_argument(
        "--gs-difficulty-adapter",
        action="store_true",
        help="Enable a zero-init difficulty residual on the conditional guess/slip branch.",
    )
    parser.add_argument(
        "--interpretable-readout-expert-adapter",
        action="store_true",
        help="Enable a zero-init expert residual gated only by interpretable slice features.",
    )
    parser.add_argument(
        "--interpretable-readout-expert-count",
        type=int,
        default=3,
        help="Number of experts used by the interpretable readout residual.",
    )
    parser.add_argument(
        "--student-conditioned-ukc-readout-residual",
        action="store_true",
        help="Enable a zero-init none-seen readout residual from graph-adjacent student TKC states.",
    )
    parser.add_argument(
        "--evidence-calibrated-behavior-gate",
        action="store_true",
        help="Enable a zero-init student-concept evidence residual on the TKC correct/incorrect behavior gate.",
    )
    parser.add_argument(
        "--evidence-behavior-gate-max-logit",
        type=float,
        default=0.5,
        help="Absolute logit scale used by tanh bounding for the evidence behavior-gate residual.",
    )
    parser.add_argument(
        "--evidence-behavior-gate-trigger",
        choices=["all", "low_evidence"],
        default="all",
        help="Which student-concept cells receive the evidence behavior-gate residual.",
    )
    parser.add_argument(
        "--evidence-behavior-gate-low-attempt-threshold",
        type=float,
        default=3.0,
        help="Maximum attempt count treated as low evidence when --evidence-behavior-gate-trigger=low_evidence.",
    )
    parser.add_argument(
        "--concept-evidence-readout-residual",
        action="store_true",
        help="Enable a zero-init target-local student-concept evidence residual on the cognitive readout.",
    )
    parser.add_argument(
        "--concept-evidence-readout-min-count",
        type=int,
        default=2,
        help="Minimum target concept count required before the concept evidence readout residual is applied.",
    )
    parser.add_argument(
        "--concept-evidence-readout-min-seen-ratio",
        type=float,
        default=1.0,
        help="Minimum fraction of target concepts with train-history evidence before applying the residual.",
    )
    parser.add_argument(
        "--concept-evidence-readout-max-logit",
        type=float,
        default=0.5,
        help="Absolute logit scale used by tanh bounding for the concept evidence readout residual.",
    )
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
    args = apply_dataset_defaults(args, parser)
    args.student_gate_prior_alpha = resolve_student_gate_prior_arg(
        explicit=args.student_gate_prior_alpha,
        legacy=args.legacy_alpha,
        default=1.0,
        new_flag="--student-gate-prior-alpha",
        legacy_flag="--alpha",
    )
    args.student_gate_prior_beta = resolve_student_gate_prior_arg(
        explicit=args.student_gate_prior_beta,
        legacy=args.legacy_beta,
        default=1.0,
        new_flag="--student-gate-prior-beta",
        legacy_flag="--beta",
    )
    if args.evidence_behavior_gate_max_logit <= 0.0:
        raise ValueError("--evidence-behavior-gate-max-logit must be positive.")
    if args.evidence_behavior_gate_low_attempt_threshold < 0.0:
        raise ValueError("--evidence-behavior-gate-low-attempt-threshold must be non-negative.")
    if args.concept_evidence_readout_min_count < 1:
        raise ValueError("--concept-evidence-readout-min-count must be positive.")
    if args.concept_evidence_readout_min_seen_ratio < 0.0 or args.concept_evidence_readout_min_seen_ratio > 1.0:
        raise ValueError("--concept-evidence-readout-min-seen-ratio must be in [0, 1].")
    if args.concept_evidence_readout_max_logit <= 0.0:
        raise ValueError("--concept-evidence-readout-max-logit must be positive.")
    return args


def resolve_student_gate_prior_arg(
    *,
    explicit: float | None,
    legacy: float | None,
    default: float,
    new_flag: str,
    legacy_flag: str,
) -> float:
    resolved_explicit = None if explicit is None else float(explicit)
    resolved_legacy = None if legacy is None else float(legacy)
    if resolved_legacy is not None and resolved_explicit == float(default):
        resolved_explicit = None
    if resolved_explicit is not None and resolved_legacy is not None and resolved_explicit != resolved_legacy:
        raise ValueError(f"Received conflicting values for {new_flag} and deprecated {legacy_flag}.")
    if resolved_explicit is not None:
        return resolved_explicit
    if resolved_legacy is not None:
        return resolved_legacy
    return float(default)


def validate_graph_args(args: argparse.Namespace) -> None:
    has_prerequisite_graph = args.prerequisite_graph is not None
    has_similarity_graph = args.similarity_graph is not None
    if args.graph_mode == "dual":
        if not has_prerequisite_graph or not has_similarity_graph:
            raise ValueError("Dual graph mode requires both --prerequisite-graph and --similarity-graph.")
        return
    if has_prerequisite_graph or has_similarity_graph:
        raise ValueError(
            "Single graph mode does not accept --prerequisite-graph or --similarity-graph. "
            "Use --graph-mode dual for the legacy dual-graph ablation."
        )


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
    validate_graph_args(args)
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
        graph_mode=args.graph_mode,
        student_gate_prior_alpha=args.student_gate_prior_alpha,
        student_gate_prior_beta=args.student_gate_prior_beta,
        gs_mode=args.gs_mode,
        high_concept_logit_adapter=args.high_concept_logit_adapter,
        high_concept_logit_min_count=args.high_concept_logit_min_count,
        pairwise_history_interaction_adapter=args.pairwise_history_interaction_adapter,
        pairwise_history_interaction_min_count=args.pairwise_history_interaction_min_count,
        gs_difficulty_adapter=args.gs_difficulty_adapter,
        interpretable_readout_expert_adapter=args.interpretable_readout_expert_adapter,
        interpretable_readout_expert_count=args.interpretable_readout_expert_count,
        student_conditioned_ukc_readout_residual=args.student_conditioned_ukc_readout_residual,
        evidence_calibrated_behavior_gate=args.evidence_calibrated_behavior_gate,
        evidence_behavior_gate_max_logit=args.evidence_behavior_gate_max_logit,
        evidence_behavior_gate_trigger=args.evidence_behavior_gate_trigger,
        evidence_behavior_gate_low_attempt_threshold=args.evidence_behavior_gate_low_attempt_threshold,
        concept_evidence_readout_residual=args.concept_evidence_readout_residual,
        concept_evidence_readout_min_count=args.concept_evidence_readout_min_count,
        concept_evidence_readout_min_seen_ratio=args.concept_evidence_readout_min_seen_ratio,
        concept_evidence_readout_max_logit=args.concept_evidence_readout_max_logit,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(output_path.with_name(output_path.stem + "_best.pt"))
    result = train_model(
        train_bundle=train_bundle,
        valid_bundle=valid_bundle,
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        training_mode=args.training_mode,
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
        "train_interactions": args.train_interactions or args.interactions,
        "valid_interactions": args.valid_interactions,
        "test_interactions": args.test_interactions or args.interactions,
        "q_matrix": args.q_matrix,
        "concept_graph": args.concept_graph,
        "prerequisite_graph": args.prerequisite_graph,
        "similarity_graph": args.similarity_graph,
        "num_students": train_bundle.num_students,
        "num_exercises": train_bundle.num_exercises,
        "num_concepts": train_bundle.num_concepts,
        "concept_dim": args.concept_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "training_mode": args.training_mode,
        "gs_mode": args.gs_mode,
        "graph_mode": args.graph_mode,
        "student_gate_prior_alpha": args.student_gate_prior_alpha,
        "student_gate_prior_beta": args.student_gate_prior_beta,
        "high_concept_logit_adapter": args.high_concept_logit_adapter,
        "high_concept_logit_min_count": args.high_concept_logit_min_count,
        "pairwise_history_interaction_adapter": args.pairwise_history_interaction_adapter,
        "pairwise_history_interaction_min_count": args.pairwise_history_interaction_min_count,
        "gs_difficulty_adapter": args.gs_difficulty_adapter,
        "interpretable_readout_expert_adapter": args.interpretable_readout_expert_adapter,
        "interpretable_readout_expert_count": args.interpretable_readout_expert_count,
        "student_conditioned_ukc_readout_residual": args.student_conditioned_ukc_readout_residual,
        "evidence_calibrated_behavior_gate": args.evidence_calibrated_behavior_gate,
        "evidence_behavior_gate_max_logit": args.evidence_behavior_gate_max_logit,
        "evidence_behavior_gate_trigger": args.evidence_behavior_gate_trigger,
        "evidence_behavior_gate_low_attempt_threshold": args.evidence_behavior_gate_low_attempt_threshold,
        "concept_evidence_readout_residual": args.concept_evidence_readout_residual,
        "concept_evidence_readout_min_count": args.concept_evidence_readout_min_count,
        "concept_evidence_readout_min_seen_ratio": args.concept_evidence_readout_min_seen_ratio,
        "concept_evidence_readout_max_logit": args.concept_evidence_readout_max_logit,
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
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "training_mode": args.training_mode,
        "lr_scheduler_patience": args.lr_scheduler_patience,
        "lr_scheduler_factor": args.lr_scheduler_factor,
        "min_learning_rate": args.min_learning_rate,
        "device": resolved_device,
        "concept_dim": args.concept_dim,
        "gs_mode": args.gs_mode,
        "graph_mode": args.graph_mode,
        "student_gate_prior_alpha": args.student_gate_prior_alpha,
        "student_gate_prior_beta": args.student_gate_prior_beta,
        "high_concept_logit_adapter": args.high_concept_logit_adapter,
        "high_concept_logit_min_count": args.high_concept_logit_min_count,
        "pairwise_history_interaction_adapter": args.pairwise_history_interaction_adapter,
        "pairwise_history_interaction_min_count": args.pairwise_history_interaction_min_count,
        "gs_difficulty_adapter": args.gs_difficulty_adapter,
        "interpretable_readout_expert_adapter": args.interpretable_readout_expert_adapter,
        "interpretable_readout_expert_count": args.interpretable_readout_expert_count,
        "student_conditioned_ukc_readout_residual": args.student_conditioned_ukc_readout_residual,
        "evidence_calibrated_behavior_gate": args.evidence_calibrated_behavior_gate,
        "evidence_behavior_gate_max_logit": args.evidence_behavior_gate_max_logit,
        "evidence_behavior_gate_trigger": args.evidence_behavior_gate_trigger,
        "evidence_behavior_gate_low_attempt_threshold": args.evidence_behavior_gate_low_attempt_threshold,
        "concept_evidence_readout_residual": args.concept_evidence_readout_residual,
        "concept_evidence_readout_min_count": args.concept_evidence_readout_min_count,
        "concept_evidence_readout_min_seen_ratio": args.concept_evidence_readout_min_seen_ratio,
        "concept_evidence_readout_max_logit": args.concept_evidence_readout_max_logit,
        "seed": args.seed,
        "best_epoch": result.best_epoch,
        "best_val_auc": result.best_val_auc,
        "test_auc": test_metrics["auc"],
        "test_acc": test_metrics["acc"],
        "test_rmse": test_metrics["rmse"],
        "test_brier": test_metrics["brier"],
        "test_ece": test_metrics["ece"],
        "best_checkpoint_path": str(Path(checkpoint_path).resolve()),
        "output_json": str(output_path.resolve()),
        "history_csv": str(Path(history_path).resolve()),
    }
    if valid_metrics is not None:
        summary_row["valid_brier"] = valid_metrics["brier"]
        summary_row["valid_ece"] = valid_metrics["ece"]
    append_summary_csv(summary_row, "results/experiment_results.csv")
    logger.info(
        "Finished run: best_val_auc=%s test_auc=%.6f test_acc=%.6f test_rmse=%.6f test_brier=%.6f test_ece=%.6f",
        result.best_val_auc,
        test_metrics["auc"],
        test_metrics["acc"],
        test_metrics["rmse"],
        test_metrics["brier"],
        test_metrics["ece"],
    )

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
