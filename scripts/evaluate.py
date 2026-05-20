from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import apply_dataset_defaults
from data import prepare_experiment_split_bundles
from models import DecoupledCDM
from trainers import evaluate_model
from utils import resolve_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation on train/valid/test splits.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--train-interactions", default=None)
    parser.add_argument("--valid-interactions", default=None)
    parser.add_argument("--test-interactions", default=None)
    parser.add_argument("--q-matrix", default=None)
    parser.add_argument("--concept-graph", default=None)
    parser.add_argument("--prerequisite-graph", default=None, help="Legacy dual-graph ablation input.")
    parser.add_argument("--similarity-graph", default=None, help="Legacy dual-graph ablation input.")
    parser.add_argument(
        "--graph-mode",
        choices=["single", "dual"],
        default="single",
        help="Use 'single' for the current mainline. 'dual' is retained only for historical ablations.",
    )
    parser.add_argument("--concept-dim", type=int, default=16)
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
    parser.add_argument("--alpha", dest="legacy_alpha", type=float, default=None, help="Deprecated alias for --student-gate-prior-alpha.")
    parser.add_argument("--beta", dest="legacy_beta", type=float, default=None, help="Deprecated alias for --student-gate-prior-beta.")
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
    parser.add_argument("--evidence-behavior-gate-max-logit", type=float, default=0.5)
    parser.add_argument(
        "--evidence-behavior-gate-trigger",
        choices=["all", "low_evidence"],
        default="all",
    )
    parser.add_argument("--evidence-behavior-gate-low-attempt-threshold", type=float, default=3.0)
    parser.add_argument(
        "--concept-evidence-readout-residual",
        action="store_true",
        help="Enable a zero-init target-local student-concept evidence residual on the cognitive readout.",
    )
    parser.add_argument("--concept-evidence-readout-min-count", type=int, default=2)
    parser.add_argument("--concept-evidence-readout-max-count", type=int, default=0)
    parser.add_argument("--concept-evidence-readout-min-seen-ratio", type=float, default=1.0)
    parser.add_argument("--concept-evidence-readout-max-logit", type=float, default=0.5)
    parser.add_argument(
        "--concept-evidence-prior-residual",
        action="store_true",
        help="Enable a deterministic target-local student-concept mastery prior on the cognitive readout.",
    )
    parser.add_argument("--concept-evidence-prior-min-count", type=int, default=2)
    parser.add_argument("--concept-evidence-prior-min-seen-ratio", type=float, default=1.0)
    parser.add_argument("--concept-evidence-prior-max-logit", type=float, default=0.5)
    parser.add_argument("--concept-evidence-prior-strength", type=float, default=2.0)
    parser.add_argument("--concept-evidence-prior-confidence-cap", type=float, default=20.0)
    parser.add_argument("--concept-evidence-prior-min-confidence", type=float, default=0.0)
    parser.add_argument("--concept-evidence-prior-min-abs-mastery", type=float, default=0.0)
    parser.add_argument("--concept-evidence-prior-positive-scale", type=float, default=1.0)
    parser.add_argument("--concept-evidence-prior-negative-scale", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--output", default="results/eval_summary.json")
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
    if args.concept_evidence_readout_max_count < 0:
        raise ValueError("--concept-evidence-readout-max-count must be non-negative.")
    if (
        args.concept_evidence_readout_max_count > 0
        and args.concept_evidence_readout_max_count < args.concept_evidence_readout_min_count
    ):
        raise ValueError("--concept-evidence-readout-max-count must be zero or at least --concept-evidence-readout-min-count.")
    if args.concept_evidence_readout_min_seen_ratio < 0.0 or args.concept_evidence_readout_min_seen_ratio > 1.0:
        raise ValueError("--concept-evidence-readout-min-seen-ratio must be in [0, 1].")
    if args.concept_evidence_readout_max_logit <= 0.0:
        raise ValueError("--concept-evidence-readout-max-logit must be positive.")
    if args.concept_evidence_prior_min_count < 1:
        raise ValueError("--concept-evidence-prior-min-count must be positive.")
    if args.concept_evidence_prior_min_seen_ratio < 0.0 or args.concept_evidence_prior_min_seen_ratio > 1.0:
        raise ValueError("--concept-evidence-prior-min-seen-ratio must be in [0, 1].")
    if args.concept_evidence_prior_max_logit <= 0.0:
        raise ValueError("--concept-evidence-prior-max-logit must be positive.")
    if args.concept_evidence_prior_strength <= 0.0:
        raise ValueError("--concept-evidence-prior-strength must be positive.")
    if args.concept_evidence_prior_confidence_cap <= 0.0:
        raise ValueError("--concept-evidence-prior-confidence-cap must be positive.")
    if args.concept_evidence_prior_min_confidence < 0.0 or args.concept_evidence_prior_min_confidence > 1.0:
        raise ValueError("--concept-evidence-prior-min-confidence must be in [0, 1].")
    if args.concept_evidence_prior_min_abs_mastery < 0.0 or args.concept_evidence_prior_min_abs_mastery > 1.0:
        raise ValueError("--concept-evidence-prior-min-abs-mastery must be in [0, 1].")
    if args.concept_evidence_prior_positive_scale < 0.0:
        raise ValueError("--concept-evidence-prior-positive-scale must be non-negative.")
    if args.concept_evidence_prior_negative_scale < 0.0:
        raise ValueError("--concept-evidence-prior-negative-scale must be non-negative.")
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


def derive_q_matrix_from_splits_if_needed(train_path: str, valid_path: str, test_path: str, q_matrix_path: str | None) -> str:
    if q_matrix_path is not None:
        return q_matrix_path
    import pandas as pd

    frames = [
        pd.read_csv(train_path, usecols=["exer_id", "cpt_seq"]),
        pd.read_csv(valid_path, usecols=["exer_id", "cpt_seq"]),
        pd.read_csv(test_path, usecols=["exer_id", "cpt_seq"]),
    ]
    q_df = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    output_path = Path("results") / "derived_q_matrix_eval.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    q_df.to_csv(output_path, index=False)
    return str(output_path)


def main() -> None:
    args = parse_args()
    validate_graph_args(args)
    if not all([args.train_interactions, args.valid_interactions, args.test_interactions]):
        raise ValueError("Evaluation requires --train-interactions, --valid-interactions, and --test-interactions.")

    q_matrix_path = derive_q_matrix_from_splits_if_needed(
        args.train_interactions,
        args.valid_interactions,
        args.test_interactions,
        args.q_matrix,
    )
    bundles = prepare_experiment_split_bundles(
        train_interactions_path=args.train_interactions,
        valid_interactions_path=args.valid_interactions,
        test_interactions_path=args.test_interactions,
        q_matrix_path=q_matrix_path,
        concept_graph_path=args.concept_graph,
        prerequisite_graph_path=args.prerequisite_graph if args.graph_mode == "dual" else None,
        similarity_graph_path=args.similarity_graph if args.graph_mode == "dual" else None,
    )
    device = str(resolve_device(args.device, args.gpus))
    model = DecoupledCDM(
        num_students=bundles["train"].num_students,
        num_exercises=bundles["train"].num_exercises,
        num_concepts=bundles["train"].num_concepts,
        concept_dim=args.concept_dim,
        graph_mode=args.graph_mode,
        student_gate_prior_alpha=args.student_gate_prior_alpha,
        student_gate_prior_beta=args.student_gate_prior_beta,
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
        concept_evidence_readout_max_count=args.concept_evidence_readout_max_count,
        concept_evidence_readout_min_seen_ratio=args.concept_evidence_readout_min_seen_ratio,
        concept_evidence_readout_max_logit=args.concept_evidence_readout_max_logit,
        concept_evidence_prior_residual=args.concept_evidence_prior_residual,
        concept_evidence_prior_min_count=args.concept_evidence_prior_min_count,
        concept_evidence_prior_min_seen_ratio=args.concept_evidence_prior_min_seen_ratio,
        concept_evidence_prior_max_logit=args.concept_evidence_prior_max_logit,
        concept_evidence_prior_strength=args.concept_evidence_prior_strength,
        concept_evidence_prior_confidence_cap=args.concept_evidence_prior_confidence_cap,
        concept_evidence_prior_min_confidence=args.concept_evidence_prior_min_confidence,
        concept_evidence_prior_min_abs_mastery=args.concept_evidence_prior_min_abs_mastery,
        concept_evidence_prior_positive_scale=args.concept_evidence_prior_positive_scale,
        concept_evidence_prior_negative_scale=args.concept_evidence_prior_negative_scale,
    )

    payload = {
        "dataset": args.dataset,
        "train_interactions": args.train_interactions,
        "valid_interactions": args.valid_interactions,
        "test_interactions": args.test_interactions,
        "q_matrix": args.q_matrix,
        "concept_graph": args.concept_graph,
        "prerequisite_graph": args.prerequisite_graph,
        "similarity_graph": args.similarity_graph,
        "device": device,
        "concept_dim": args.concept_dim,
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
        "concept_evidence_readout_max_count": args.concept_evidence_readout_max_count,
        "concept_evidence_readout_min_seen_ratio": args.concept_evidence_readout_min_seen_ratio,
        "concept_evidence_readout_max_logit": args.concept_evidence_readout_max_logit,
        "concept_evidence_prior_residual": args.concept_evidence_prior_residual,
        "concept_evidence_prior_min_count": args.concept_evidence_prior_min_count,
        "concept_evidence_prior_min_seen_ratio": args.concept_evidence_prior_min_seen_ratio,
        "concept_evidence_prior_max_logit": args.concept_evidence_prior_max_logit,
        "concept_evidence_prior_strength": args.concept_evidence_prior_strength,
        "concept_evidence_prior_confidence_cap": args.concept_evidence_prior_confidence_cap,
        "concept_evidence_prior_min_confidence": args.concept_evidence_prior_min_confidence,
        "concept_evidence_prior_min_abs_mastery": args.concept_evidence_prior_min_abs_mastery,
        "concept_evidence_prior_positive_scale": args.concept_evidence_prior_positive_scale,
        "concept_evidence_prior_negative_scale": args.concept_evidence_prior_negative_scale,
        "train_metrics": evaluate_model(bundle=bundles["train"], model=model, device=device),
        "valid_metrics": evaluate_model(bundle=bundles["valid"], model=model, device=device),
        "test_metrics": evaluate_model(bundle=bundles["test"], model=model, device=device),
    }
    write_json(payload, args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
