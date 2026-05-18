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
        "--concept-evidence-readout-max-count",
        type=int,
        default=0,
        help="Optional maximum target concept count allowed before the concept evidence readout residual is applied. Zero disables the upper bound.",
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
    parser.add_argument(
        "--concept-evidence-prior-residual",
        action="store_true",
        help="Enable a deterministic target-local student-concept mastery prior on the cognitive readout.",
    )
    parser.add_argument("--concept-evidence-prior-min-count", type=int, default=2)
    parser.add_argument("--concept-evidence-prior-max-count", type=int, default=0)
    parser.add_argument("--concept-evidence-prior-min-seen-ratio", type=float, default=1.0)
    parser.add_argument("--concept-evidence-prior-max-logit", type=float, default=0.5)
    parser.add_argument("--concept-evidence-prior-strength", type=float, default=2.0)
    parser.add_argument("--concept-evidence-prior-confidence-cap", type=float, default=20.0)
    parser.add_argument(
        "--student-evidence-ability-prior-residual",
        action="store_true",
        help="Enable a deterministic train-history student global ability prior on the cognitive readout.",
    )
    parser.add_argument("--student-evidence-ability-prior-min-attempts", type=int, default=1)
    parser.add_argument("--student-evidence-ability-prior-max-logit", type=float, default=0.25)
    parser.add_argument("--student-evidence-ability-prior-strength", type=float, default=2.0)
    parser.add_argument("--student-evidence-ability-prior-confidence-cap", type=float, default=200.0)
    parser.add_argument(
        "--student-evidence-gs-prior-residual",
        action="store_true",
        help="Enable a deterministic student ability prior on guess/slip logits.",
    )
    parser.add_argument("--student-evidence-gs-prior-min-attempts", type=int, default=1)
    parser.add_argument("--student-evidence-gs-prior-max-logit", type=float, default=0.25)
    parser.add_argument("--student-evidence-gs-prior-strength", type=float, default=2.0)
    parser.add_argument("--student-evidence-gs-prior-confidence-cap", type=float, default=200.0)
    parser.add_argument(
        "--concept-evidence-calibrated-readout",
        action="store_true",
        help="Enable a bounded pure-CDM readout residual from student/target concept evidence summaries.",
    )
    parser.add_argument("--concept-evidence-calibrated-readout-min-count", type=int, default=1)
    parser.add_argument("--concept-evidence-calibrated-readout-max-count", type=int, default=0)
    parser.add_argument("--concept-evidence-calibrated-readout-min-seen-ratio", type=float, default=1.0)
    parser.add_argument("--concept-evidence-calibrated-readout-max-logit", type=float, default=0.35)
    parser.add_argument("--concept-evidence-calibrated-readout-prior-strength", type=float, default=2.0)
    parser.add_argument("--concept-evidence-calibrated-readout-confidence-cap", type=float, default=20.0)
    parser.add_argument(
        "--history-evidence-fusion-readout",
        action="store_true",
        help="Enable a zero-init bounded cognitive readout over train-history concept, student, and exercise evidence.",
    )
    parser.add_argument("--history-evidence-fusion-min-count", type=int, default=1)
    parser.add_argument("--history-evidence-fusion-max-count", type=int, default=0)
    parser.add_argument("--history-evidence-fusion-min-seen-ratio", type=float, default=0.0)
    parser.add_argument("--history-evidence-fusion-max-logit", type=float, default=0.5)
    parser.add_argument("--history-evidence-fusion-prior-strength", type=float, default=2.0)
    parser.add_argument("--history-evidence-fusion-concept-confidence-cap", type=float, default=20.0)
    parser.add_argument("--history-evidence-fusion-exercise-confidence-cap", type=float, default=200.0)
    parser.add_argument("--history-evidence-fusion-student-confidence-cap", type=float, default=200.0)
    parser.add_argument(
        "--history-evidence-linear-readout",
        action="store_true",
        help="Enable a zero-init 3-weight linear readout over concept, student, and exercise evidence priors.",
    )
    parser.add_argument("--history-evidence-linear-min-count", type=int, default=1)
    parser.add_argument("--history-evidence-linear-max-count", type=int, default=0)
    parser.add_argument("--history-evidence-linear-min-seen-ratio", type=float, default=0.0)
    parser.add_argument("--history-evidence-linear-max-logit", type=float, default=0.5)
    parser.add_argument("--history-evidence-linear-prior-strength", type=float, default=2.0)
    parser.add_argument("--history-evidence-linear-concept-confidence-cap", type=float, default=20.0)
    parser.add_argument("--history-evidence-linear-exercise-confidence-cap", type=float, default=200.0)
    parser.add_argument("--history-evidence-linear-student-confidence-cap", type=float, default=200.0)
    parser.add_argument(
        "--history-evidence-logit-prior-residual",
        action="store_true",
        help="Enable a fixed interpretable log-odds prior from train-history evidence on the cognitive readout.",
    )
    parser.add_argument(
        "--history-evidence-logit-prior-location",
        choices=["cognitive", "output", "loss_only"],
        default="cognitive",
        help="Apply the fixed log-odds evidence prior in forward pass, or expose it for train-time loss only.",
    )
    parser.add_argument("--history-evidence-logit-prior-min-count", type=int, default=1)
    parser.add_argument("--history-evidence-logit-prior-max-count", type=int, default=0)
    parser.add_argument("--history-evidence-logit-prior-min-seen-ratio", type=float, default=0.0)
    parser.add_argument("--history-evidence-logit-prior-max-logit", type=float, default=3.0)
    parser.add_argument("--history-evidence-logit-prior-component-cap", type=float, default=3.0)
    parser.add_argument("--history-evidence-logit-prior-weight-student", type=float, default=1.0)
    parser.add_argument("--history-evidence-logit-prior-weight-exercise", type=float, default=1.0)
    parser.add_argument("--history-evidence-logit-prior-weight-target-concept", type=float, default=0.6)
    parser.add_argument("--history-evidence-logit-prior-weight-concept", type=float, default=0.35)
    parser.add_argument("--history-evidence-logit-prior-weight-mastery", type=float, default=0.0)
    parser.add_argument("--history-evidence-logit-prior-prior-weight", type=float, default=5.0)
    parser.add_argument("--history-evidence-logit-prior-mastery-confidence-cap", type=float, default=20.0)
    parser.add_argument(
        "--history-evidence-cognitive-alignment-weight",
        type=float,
        default=0.0,
        help="Standardized MSE loss weight aligning cognitive logits to loss-only history evidence logit prior.",
    )
    parser.add_argument(
        "--history-evidence-output-calibration",
        action="store_true",
        help="Enable a zero-init bounded output-logit calibration from pure train-history evidence features.",
    )
    parser.add_argument("--history-evidence-output-calibration-min-count", type=int, default=1)
    parser.add_argument("--history-evidence-output-calibration-max-count", type=int, default=0)
    parser.add_argument("--history-evidence-output-calibration-min-seen-ratio", type=float, default=0.0)
    parser.add_argument("--history-evidence-output-calibration-max-logit", type=float, default=0.5)
    parser.add_argument("--history-evidence-output-calibration-prior-strength", type=float, default=2.0)
    parser.add_argument("--history-evidence-output-calibration-concept-confidence-cap", type=float, default=20.0)
    parser.add_argument("--history-evidence-output-calibration-exercise-confidence-cap", type=float, default=200.0)
    parser.add_argument("--history-evidence-output-calibration-student-confidence-cap", type=float, default=200.0)
    parser.add_argument(
        "--exercise-evidence-prior-residual",
        action="store_true",
        help="Enable a bounded train-history exercise ease prior on the cognitive readout.",
    )
    parser.add_argument("--exercise-evidence-prior-min-count", type=int, default=1)
    parser.add_argument("--exercise-evidence-prior-max-logit", type=float, default=0.25)
    parser.add_argument("--exercise-evidence-prior-strength", type=float, default=2.0)
    parser.add_argument("--exercise-evidence-prior-confidence-cap", type=float, default=200.0)
    parser.add_argument(
        "--exercise-evidence-difficulty-adapter",
        action="store_true",
        help="Enable a zero-init learnable global slope for bounded train-history exercise ease evidence.",
    )
    parser.add_argument("--exercise-evidence-difficulty-adapter-min-count", type=int, default=1)
    parser.add_argument("--exercise-evidence-difficulty-adapter-max-logit", type=float, default=0.5)
    parser.add_argument("--exercise-evidence-difficulty-adapter-strength", type=float, default=2.0)
    parser.add_argument("--exercise-evidence-difficulty-adapter-confidence-cap", type=float, default=200.0)
    parser.add_argument(
        "--exercise-evidence-difficulty-init",
        action="store_true",
        help="Initialize learned exercise difficulty from bounded train-history exercise ease evidence.",
    )
    parser.add_argument("--exercise-evidence-difficulty-init-min-count", type=int, default=1)
    parser.add_argument("--exercise-evidence-difficulty-init-max-abs-logit", type=float, default=0.25)
    parser.add_argument("--exercise-evidence-difficulty-init-strength", type=float, default=2.0)
    parser.add_argument("--exercise-evidence-difficulty-init-confidence-cap", type=float, default=200.0)
    parser.add_argument(
        "--exercise-evidence-difficulty-regularization-weight",
        type=float,
        default=0.0,
        help="MSE penalty weight tying learned exercise difficulty to bounded train-history difficulty evidence.",
    )
    parser.add_argument("--exercise-evidence-difficulty-regularization-min-count", type=int, default=1)
    parser.add_argument("--exercise-evidence-difficulty-regularization-max-abs-logit", type=float, default=0.25)
    parser.add_argument("--exercise-evidence-difficulty-regularization-strength", type=float, default=2.0)
    parser.add_argument("--exercise-evidence-difficulty-regularization-confidence-cap", type=float, default=200.0)
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
    if args.concept_evidence_prior_max_count < 0:
        raise ValueError("--concept-evidence-prior-max-count must be non-negative.")
    if (
        args.concept_evidence_prior_max_count > 0
        and args.concept_evidence_prior_max_count < args.concept_evidence_prior_min_count
    ):
        raise ValueError("--concept-evidence-prior-max-count must be zero or at least --concept-evidence-prior-min-count.")
    if args.concept_evidence_prior_min_seen_ratio < 0.0 or args.concept_evidence_prior_min_seen_ratio > 1.0:
        raise ValueError("--concept-evidence-prior-min-seen-ratio must be in [0, 1].")
    if args.concept_evidence_prior_max_logit <= 0.0:
        raise ValueError("--concept-evidence-prior-max-logit must be positive.")
    if args.concept_evidence_prior_strength <= 0.0:
        raise ValueError("--concept-evidence-prior-strength must be positive.")
    if args.concept_evidence_prior_confidence_cap <= 0.0:
        raise ValueError("--concept-evidence-prior-confidence-cap must be positive.")
    if args.student_evidence_ability_prior_min_attempts < 1:
        raise ValueError("--student-evidence-ability-prior-min-attempts must be positive.")
    if args.student_evidence_ability_prior_max_logit <= 0.0:
        raise ValueError("--student-evidence-ability-prior-max-logit must be positive.")
    if args.student_evidence_ability_prior_strength <= 0.0:
        raise ValueError("--student-evidence-ability-prior-strength must be positive.")
    if args.student_evidence_ability_prior_confidence_cap <= 0.0:
        raise ValueError("--student-evidence-ability-prior-confidence-cap must be positive.")
    if args.student_evidence_gs_prior_min_attempts < 1:
        raise ValueError("--student-evidence-gs-prior-min-attempts must be positive.")
    if args.student_evidence_gs_prior_max_logit <= 0.0:
        raise ValueError("--student-evidence-gs-prior-max-logit must be positive.")
    if args.student_evidence_gs_prior_strength <= 0.0:
        raise ValueError("--student-evidence-gs-prior-strength must be positive.")
    if args.student_evidence_gs_prior_confidence_cap <= 0.0:
        raise ValueError("--student-evidence-gs-prior-confidence-cap must be positive.")
    if args.concept_evidence_calibrated_readout_min_count < 1:
        raise ValueError("--concept-evidence-calibrated-readout-min-count must be positive.")
    if args.concept_evidence_calibrated_readout_max_count < 0:
        raise ValueError("--concept-evidence-calibrated-readout-max-count must be non-negative.")
    if (
        args.concept_evidence_calibrated_readout_max_count > 0
        and args.concept_evidence_calibrated_readout_max_count < args.concept_evidence_calibrated_readout_min_count
    ):
        raise ValueError(
            "--concept-evidence-calibrated-readout-max-count must be zero or at least "
            "--concept-evidence-calibrated-readout-min-count."
        )
    if (
        args.concept_evidence_calibrated_readout_min_seen_ratio < 0.0
        or args.concept_evidence_calibrated_readout_min_seen_ratio > 1.0
    ):
        raise ValueError("--concept-evidence-calibrated-readout-min-seen-ratio must be in [0, 1].")
    if args.concept_evidence_calibrated_readout_max_logit <= 0.0:
        raise ValueError("--concept-evidence-calibrated-readout-max-logit must be positive.")
    if args.concept_evidence_calibrated_readout_prior_strength <= 0.0:
        raise ValueError("--concept-evidence-calibrated-readout-prior-strength must be positive.")
    if args.concept_evidence_calibrated_readout_confidence_cap <= 0.0:
        raise ValueError("--concept-evidence-calibrated-readout-confidence-cap must be positive.")
    if args.history_evidence_fusion_min_count < 1:
        raise ValueError("--history-evidence-fusion-min-count must be positive.")
    if args.history_evidence_fusion_max_count < 0:
        raise ValueError("--history-evidence-fusion-max-count must be non-negative.")
    if (
        args.history_evidence_fusion_max_count > 0
        and args.history_evidence_fusion_max_count < args.history_evidence_fusion_min_count
    ):
        raise ValueError("--history-evidence-fusion-max-count must be zero or at least --history-evidence-fusion-min-count.")
    if args.history_evidence_fusion_min_seen_ratio < 0.0 or args.history_evidence_fusion_min_seen_ratio > 1.0:
        raise ValueError("--history-evidence-fusion-min-seen-ratio must be in [0, 1].")
    if args.history_evidence_fusion_max_logit <= 0.0:
        raise ValueError("--history-evidence-fusion-max-logit must be positive.")
    if args.history_evidence_fusion_prior_strength <= 0.0:
        raise ValueError("--history-evidence-fusion-prior-strength must be positive.")
    if args.history_evidence_fusion_concept_confidence_cap <= 0.0:
        raise ValueError("--history-evidence-fusion-concept-confidence-cap must be positive.")
    if args.history_evidence_fusion_exercise_confidence_cap <= 0.0:
        raise ValueError("--history-evidence-fusion-exercise-confidence-cap must be positive.")
    if args.history_evidence_fusion_student_confidence_cap <= 0.0:
        raise ValueError("--history-evidence-fusion-student-confidence-cap must be positive.")
    if args.history_evidence_linear_min_count < 1:
        raise ValueError("--history-evidence-linear-min-count must be positive.")
    if args.history_evidence_linear_max_count < 0:
        raise ValueError("--history-evidence-linear-max-count must be non-negative.")
    if (
        args.history_evidence_linear_max_count > 0
        and args.history_evidence_linear_max_count < args.history_evidence_linear_min_count
    ):
        raise ValueError("--history-evidence-linear-max-count must be zero or at least --history-evidence-linear-min-count.")
    if args.history_evidence_linear_min_seen_ratio < 0.0 or args.history_evidence_linear_min_seen_ratio > 1.0:
        raise ValueError("--history-evidence-linear-min-seen-ratio must be in [0, 1].")
    if args.history_evidence_linear_max_logit <= 0.0:
        raise ValueError("--history-evidence-linear-max-logit must be positive.")
    if args.history_evidence_linear_prior_strength <= 0.0:
        raise ValueError("--history-evidence-linear-prior-strength must be positive.")
    if args.history_evidence_linear_concept_confidence_cap <= 0.0:
        raise ValueError("--history-evidence-linear-concept-confidence-cap must be positive.")
    if args.history_evidence_linear_exercise_confidence_cap <= 0.0:
        raise ValueError("--history-evidence-linear-exercise-confidence-cap must be positive.")
    if args.history_evidence_linear_student_confidence_cap <= 0.0:
        raise ValueError("--history-evidence-linear-student-confidence-cap must be positive.")
    if args.history_evidence_logit_prior_min_count < 1:
        raise ValueError("--history-evidence-logit-prior-min-count must be positive.")
    if args.history_evidence_logit_prior_max_count < 0:
        raise ValueError("--history-evidence-logit-prior-max-count must be non-negative.")
    if (
        args.history_evidence_logit_prior_max_count > 0
        and args.history_evidence_logit_prior_max_count < args.history_evidence_logit_prior_min_count
    ):
        raise ValueError(
            "--history-evidence-logit-prior-max-count must be zero or at least "
            "--history-evidence-logit-prior-min-count."
        )
    if (
        args.history_evidence_logit_prior_min_seen_ratio < 0.0
        or args.history_evidence_logit_prior_min_seen_ratio > 1.0
    ):
        raise ValueError("--history-evidence-logit-prior-min-seen-ratio must be in [0, 1].")
    if args.history_evidence_logit_prior_max_logit <= 0.0:
        raise ValueError("--history-evidence-logit-prior-max-logit must be positive.")
    if args.history_evidence_logit_prior_component_cap <= 0.0:
        raise ValueError("--history-evidence-logit-prior-component-cap must be positive.")
    if args.history_evidence_logit_prior_prior_weight <= 0.0:
        raise ValueError("--history-evidence-logit-prior-prior-weight must be positive.")
    if args.history_evidence_logit_prior_mastery_confidence_cap <= 0.0:
        raise ValueError("--history-evidence-logit-prior-mastery-confidence-cap must be positive.")
    if args.history_evidence_cognitive_alignment_weight < 0.0:
        raise ValueError("--history-evidence-cognitive-alignment-weight must be non-negative.")
    if (
        args.history_evidence_cognitive_alignment_weight > 0.0
        and not args.history_evidence_logit_prior_residual
    ):
        raise ValueError(
            "--history-evidence-logit-prior-residual must be enabled with "
            "--history-evidence-cognitive-alignment-weight."
        )
    if (
        args.history_evidence_cognitive_alignment_weight > 0.0
        and args.history_evidence_logit_prior_location != "loss_only"
    ):
        raise ValueError(
            "--history-evidence-logit-prior-location must be loss_only with "
            "--history-evidence-cognitive-alignment-weight."
        )
    if args.history_evidence_output_calibration_min_count < 1:
        raise ValueError("--history-evidence-output-calibration-min-count must be positive.")
    if args.history_evidence_output_calibration_max_count < 0:
        raise ValueError("--history-evidence-output-calibration-max-count must be non-negative.")
    if (
        args.history_evidence_output_calibration_max_count > 0
        and args.history_evidence_output_calibration_max_count < args.history_evidence_output_calibration_min_count
    ):
        raise ValueError(
            "--history-evidence-output-calibration-max-count must be zero or at least min-count."
        )
    if (
        args.history_evidence_output_calibration_min_seen_ratio < 0.0
        or args.history_evidence_output_calibration_min_seen_ratio > 1.0
    ):
        raise ValueError("--history-evidence-output-calibration-min-seen-ratio must be in [0, 1].")
    if args.history_evidence_output_calibration_max_logit <= 0.0:
        raise ValueError("--history-evidence-output-calibration-max-logit must be positive.")
    if args.history_evidence_output_calibration_prior_strength <= 0.0:
        raise ValueError("--history-evidence-output-calibration-prior-strength must be positive.")
    if args.history_evidence_output_calibration_concept_confidence_cap <= 0.0:
        raise ValueError("--history-evidence-output-calibration-concept-confidence-cap must be positive.")
    if args.history_evidence_output_calibration_exercise_confidence_cap <= 0.0:
        raise ValueError("--history-evidence-output-calibration-exercise-confidence-cap must be positive.")
    if args.history_evidence_output_calibration_student_confidence_cap <= 0.0:
        raise ValueError("--history-evidence-output-calibration-student-confidence-cap must be positive.")
    if args.exercise_evidence_prior_min_count < 1:
        raise ValueError("--exercise-evidence-prior-min-count must be positive.")
    if args.exercise_evidence_prior_max_logit <= 0.0:
        raise ValueError("--exercise-evidence-prior-max-logit must be positive.")
    if args.exercise_evidence_prior_strength <= 0.0:
        raise ValueError("--exercise-evidence-prior-strength must be positive.")
    if args.exercise_evidence_prior_confidence_cap <= 0.0:
        raise ValueError("--exercise-evidence-prior-confidence-cap must be positive.")
    if args.exercise_evidence_difficulty_adapter_min_count < 1:
        raise ValueError("--exercise-evidence-difficulty-adapter-min-count must be positive.")
    if args.exercise_evidence_difficulty_adapter_max_logit <= 0.0:
        raise ValueError("--exercise-evidence-difficulty-adapter-max-logit must be positive.")
    if args.exercise_evidence_difficulty_adapter_strength <= 0.0:
        raise ValueError("--exercise-evidence-difficulty-adapter-strength must be positive.")
    if args.exercise_evidence_difficulty_adapter_confidence_cap <= 0.0:
        raise ValueError("--exercise-evidence-difficulty-adapter-confidence-cap must be positive.")
    if args.exercise_evidence_difficulty_init_min_count < 1:
        raise ValueError("--exercise-evidence-difficulty-init-min-count must be positive.")
    if args.exercise_evidence_difficulty_init_max_abs_logit <= 0.0:
        raise ValueError("--exercise-evidence-difficulty-init-max-abs-logit must be positive.")
    if args.exercise_evidence_difficulty_init_strength <= 0.0:
        raise ValueError("--exercise-evidence-difficulty-init-strength must be positive.")
    if args.exercise_evidence_difficulty_init_confidence_cap <= 0.0:
        raise ValueError("--exercise-evidence-difficulty-init-confidence-cap must be positive.")
    if args.exercise_evidence_difficulty_regularization_weight < 0.0:
        raise ValueError("--exercise-evidence-difficulty-regularization-weight must be non-negative.")
    if args.exercise_evidence_difficulty_regularization_min_count < 1:
        raise ValueError("--exercise-evidence-difficulty-regularization-min-count must be positive.")
    if args.exercise_evidence_difficulty_regularization_max_abs_logit <= 0.0:
        raise ValueError("--exercise-evidence-difficulty-regularization-max-abs-logit must be positive.")
    if args.exercise_evidence_difficulty_regularization_strength <= 0.0:
        raise ValueError("--exercise-evidence-difficulty-regularization-strength must be positive.")
    if args.exercise_evidence_difficulty_regularization_confidence_cap <= 0.0:
        raise ValueError("--exercise-evidence-difficulty-regularization-confidence-cap must be positive.")
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
        concept_evidence_readout_max_count=args.concept_evidence_readout_max_count,
        concept_evidence_readout_min_seen_ratio=args.concept_evidence_readout_min_seen_ratio,
        concept_evidence_readout_max_logit=args.concept_evidence_readout_max_logit,
        concept_evidence_prior_residual=args.concept_evidence_prior_residual,
        concept_evidence_prior_min_count=args.concept_evidence_prior_min_count,
        concept_evidence_prior_max_count=args.concept_evidence_prior_max_count,
        concept_evidence_prior_min_seen_ratio=args.concept_evidence_prior_min_seen_ratio,
        concept_evidence_prior_max_logit=args.concept_evidence_prior_max_logit,
        concept_evidence_prior_strength=args.concept_evidence_prior_strength,
        concept_evidence_prior_confidence_cap=args.concept_evidence_prior_confidence_cap,
        student_evidence_ability_prior_residual=args.student_evidence_ability_prior_residual,
        student_evidence_ability_prior_min_attempts=args.student_evidence_ability_prior_min_attempts,
        student_evidence_ability_prior_max_logit=args.student_evidence_ability_prior_max_logit,
        student_evidence_ability_prior_strength=args.student_evidence_ability_prior_strength,
        student_evidence_ability_prior_confidence_cap=args.student_evidence_ability_prior_confidence_cap,
        student_evidence_gs_prior_residual=args.student_evidence_gs_prior_residual,
        student_evidence_gs_prior_min_attempts=args.student_evidence_gs_prior_min_attempts,
        student_evidence_gs_prior_max_logit=args.student_evidence_gs_prior_max_logit,
        student_evidence_gs_prior_strength=args.student_evidence_gs_prior_strength,
        student_evidence_gs_prior_confidence_cap=args.student_evidence_gs_prior_confidence_cap,
        concept_evidence_calibrated_readout=args.concept_evidence_calibrated_readout,
        concept_evidence_calibrated_readout_min_count=args.concept_evidence_calibrated_readout_min_count,
        concept_evidence_calibrated_readout_max_count=args.concept_evidence_calibrated_readout_max_count,
        concept_evidence_calibrated_readout_min_seen_ratio=args.concept_evidence_calibrated_readout_min_seen_ratio,
        concept_evidence_calibrated_readout_max_logit=args.concept_evidence_calibrated_readout_max_logit,
        concept_evidence_calibrated_readout_prior_strength=args.concept_evidence_calibrated_readout_prior_strength,
        concept_evidence_calibrated_readout_confidence_cap=args.concept_evidence_calibrated_readout_confidence_cap,
        history_evidence_fusion_readout=args.history_evidence_fusion_readout,
        history_evidence_fusion_min_count=args.history_evidence_fusion_min_count,
        history_evidence_fusion_max_count=args.history_evidence_fusion_max_count,
        history_evidence_fusion_min_seen_ratio=args.history_evidence_fusion_min_seen_ratio,
        history_evidence_fusion_max_logit=args.history_evidence_fusion_max_logit,
        history_evidence_fusion_prior_strength=args.history_evidence_fusion_prior_strength,
        history_evidence_fusion_concept_confidence_cap=args.history_evidence_fusion_concept_confidence_cap,
        history_evidence_fusion_exercise_confidence_cap=args.history_evidence_fusion_exercise_confidence_cap,
        history_evidence_fusion_student_confidence_cap=args.history_evidence_fusion_student_confidence_cap,
        history_evidence_linear_readout=args.history_evidence_linear_readout,
        history_evidence_linear_min_count=args.history_evidence_linear_min_count,
        history_evidence_linear_max_count=args.history_evidence_linear_max_count,
        history_evidence_linear_min_seen_ratio=args.history_evidence_linear_min_seen_ratio,
        history_evidence_linear_max_logit=args.history_evidence_linear_max_logit,
        history_evidence_linear_prior_strength=args.history_evidence_linear_prior_strength,
        history_evidence_linear_concept_confidence_cap=args.history_evidence_linear_concept_confidence_cap,
        history_evidence_linear_exercise_confidence_cap=args.history_evidence_linear_exercise_confidence_cap,
        history_evidence_linear_student_confidence_cap=args.history_evidence_linear_student_confidence_cap,
        history_evidence_logit_prior_residual=args.history_evidence_logit_prior_residual,
        history_evidence_logit_prior_location=args.history_evidence_logit_prior_location,
        history_evidence_logit_prior_min_count=args.history_evidence_logit_prior_min_count,
        history_evidence_logit_prior_max_count=args.history_evidence_logit_prior_max_count,
        history_evidence_logit_prior_min_seen_ratio=args.history_evidence_logit_prior_min_seen_ratio,
        history_evidence_logit_prior_max_logit=args.history_evidence_logit_prior_max_logit,
        history_evidence_logit_prior_component_cap=args.history_evidence_logit_prior_component_cap,
        history_evidence_logit_prior_weight_student=args.history_evidence_logit_prior_weight_student,
        history_evidence_logit_prior_weight_exercise=args.history_evidence_logit_prior_weight_exercise,
        history_evidence_logit_prior_weight_target_concept=args.history_evidence_logit_prior_weight_target_concept,
        history_evidence_logit_prior_weight_concept=args.history_evidence_logit_prior_weight_concept,
        history_evidence_logit_prior_weight_mastery=args.history_evidence_logit_prior_weight_mastery,
        history_evidence_logit_prior_prior_weight=args.history_evidence_logit_prior_prior_weight,
        history_evidence_logit_prior_mastery_confidence_cap=args.history_evidence_logit_prior_mastery_confidence_cap,
        history_evidence_output_calibration=args.history_evidence_output_calibration,
        history_evidence_output_calibration_min_count=args.history_evidence_output_calibration_min_count,
        history_evidence_output_calibration_max_count=args.history_evidence_output_calibration_max_count,
        history_evidence_output_calibration_min_seen_ratio=(
            args.history_evidence_output_calibration_min_seen_ratio
        ),
        history_evidence_output_calibration_max_logit=args.history_evidence_output_calibration_max_logit,
        history_evidence_output_calibration_prior_strength=(
            args.history_evidence_output_calibration_prior_strength
        ),
        history_evidence_output_calibration_concept_confidence_cap=(
            args.history_evidence_output_calibration_concept_confidence_cap
        ),
        history_evidence_output_calibration_exercise_confidence_cap=(
            args.history_evidence_output_calibration_exercise_confidence_cap
        ),
        history_evidence_output_calibration_student_confidence_cap=(
            args.history_evidence_output_calibration_student_confidence_cap
        ),
        exercise_evidence_prior_residual=args.exercise_evidence_prior_residual,
        exercise_evidence_prior_min_count=args.exercise_evidence_prior_min_count,
        exercise_evidence_prior_max_logit=args.exercise_evidence_prior_max_logit,
        exercise_evidence_prior_strength=args.exercise_evidence_prior_strength,
        exercise_evidence_prior_confidence_cap=args.exercise_evidence_prior_confidence_cap,
        exercise_evidence_difficulty_adapter=args.exercise_evidence_difficulty_adapter,
        exercise_evidence_difficulty_adapter_min_count=args.exercise_evidence_difficulty_adapter_min_count,
        exercise_evidence_difficulty_adapter_max_logit=args.exercise_evidence_difficulty_adapter_max_logit,
        exercise_evidence_difficulty_adapter_strength=args.exercise_evidence_difficulty_adapter_strength,
        exercise_evidence_difficulty_adapter_confidence_cap=args.exercise_evidence_difficulty_adapter_confidence_cap,
    )
    difficulty_init_count = 0
    if args.exercise_evidence_difficulty_init:
        difficulty_init_count = model.initialize_exercise_difficulty_from_evidence(
            exercise_evidence=train_bundle.exercise_evidence_tensor,
            min_count=args.exercise_evidence_difficulty_init_min_count,
            max_abs_logit=args.exercise_evidence_difficulty_init_max_abs_logit,
            strength=args.exercise_evidence_difficulty_init_strength,
            confidence_cap=args.exercise_evidence_difficulty_init_confidence_cap,
        )
        logger.info("Initialized exercise difficulty from evidence for %s exercises.", difficulty_init_count)
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
        exercise_evidence_difficulty_regularization_weight=args.exercise_evidence_difficulty_regularization_weight,
        exercise_evidence_difficulty_regularization_min_count=args.exercise_evidence_difficulty_regularization_min_count,
        exercise_evidence_difficulty_regularization_max_abs_logit=args.exercise_evidence_difficulty_regularization_max_abs_logit,
        exercise_evidence_difficulty_regularization_strength=args.exercise_evidence_difficulty_regularization_strength,
        exercise_evidence_difficulty_regularization_confidence_cap=(
            args.exercise_evidence_difficulty_regularization_confidence_cap
        ),
        history_evidence_cognitive_alignment_weight=args.history_evidence_cognitive_alignment_weight,
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
        "concept_evidence_readout_max_count": args.concept_evidence_readout_max_count,
        "concept_evidence_readout_min_seen_ratio": args.concept_evidence_readout_min_seen_ratio,
        "concept_evidence_readout_max_logit": args.concept_evidence_readout_max_logit,
        "concept_evidence_prior_residual": args.concept_evidence_prior_residual,
        "concept_evidence_prior_min_count": args.concept_evidence_prior_min_count,
        "concept_evidence_prior_max_count": args.concept_evidence_prior_max_count,
        "concept_evidence_prior_min_seen_ratio": args.concept_evidence_prior_min_seen_ratio,
        "concept_evidence_prior_max_logit": args.concept_evidence_prior_max_logit,
        "concept_evidence_prior_strength": args.concept_evidence_prior_strength,
        "concept_evidence_prior_confidence_cap": args.concept_evidence_prior_confidence_cap,
        "student_evidence_ability_prior_residual": args.student_evidence_ability_prior_residual,
        "student_evidence_ability_prior_min_attempts": args.student_evidence_ability_prior_min_attempts,
        "student_evidence_ability_prior_max_logit": args.student_evidence_ability_prior_max_logit,
        "student_evidence_ability_prior_strength": args.student_evidence_ability_prior_strength,
        "student_evidence_ability_prior_confidence_cap": args.student_evidence_ability_prior_confidence_cap,
        "student_evidence_gs_prior_residual": args.student_evidence_gs_prior_residual,
        "student_evidence_gs_prior_min_attempts": args.student_evidence_gs_prior_min_attempts,
        "student_evidence_gs_prior_max_logit": args.student_evidence_gs_prior_max_logit,
        "student_evidence_gs_prior_strength": args.student_evidence_gs_prior_strength,
        "student_evidence_gs_prior_confidence_cap": args.student_evidence_gs_prior_confidence_cap,
        "concept_evidence_calibrated_readout": args.concept_evidence_calibrated_readout,
        "concept_evidence_calibrated_readout_min_count": args.concept_evidence_calibrated_readout_min_count,
        "concept_evidence_calibrated_readout_max_count": args.concept_evidence_calibrated_readout_max_count,
        "concept_evidence_calibrated_readout_min_seen_ratio": args.concept_evidence_calibrated_readout_min_seen_ratio,
        "concept_evidence_calibrated_readout_max_logit": args.concept_evidence_calibrated_readout_max_logit,
        "concept_evidence_calibrated_readout_prior_strength": args.concept_evidence_calibrated_readout_prior_strength,
        "concept_evidence_calibrated_readout_confidence_cap": args.concept_evidence_calibrated_readout_confidence_cap,
        "history_evidence_fusion_readout": args.history_evidence_fusion_readout,
        "history_evidence_fusion_min_count": args.history_evidence_fusion_min_count,
        "history_evidence_fusion_max_count": args.history_evidence_fusion_max_count,
        "history_evidence_fusion_min_seen_ratio": args.history_evidence_fusion_min_seen_ratio,
        "history_evidence_fusion_max_logit": args.history_evidence_fusion_max_logit,
        "history_evidence_fusion_prior_strength": args.history_evidence_fusion_prior_strength,
        "history_evidence_fusion_concept_confidence_cap": args.history_evidence_fusion_concept_confidence_cap,
        "history_evidence_fusion_exercise_confidence_cap": args.history_evidence_fusion_exercise_confidence_cap,
        "history_evidence_fusion_student_confidence_cap": args.history_evidence_fusion_student_confidence_cap,
        "history_evidence_linear_readout": args.history_evidence_linear_readout,
        "history_evidence_linear_min_count": args.history_evidence_linear_min_count,
        "history_evidence_linear_max_count": args.history_evidence_linear_max_count,
        "history_evidence_linear_min_seen_ratio": args.history_evidence_linear_min_seen_ratio,
        "history_evidence_linear_max_logit": args.history_evidence_linear_max_logit,
        "history_evidence_linear_prior_strength": args.history_evidence_linear_prior_strength,
        "history_evidence_linear_concept_confidence_cap": args.history_evidence_linear_concept_confidence_cap,
        "history_evidence_linear_exercise_confidence_cap": args.history_evidence_linear_exercise_confidence_cap,
        "history_evidence_linear_student_confidence_cap": args.history_evidence_linear_student_confidence_cap,
        "history_evidence_logit_prior_residual": args.history_evidence_logit_prior_residual,
        "history_evidence_logit_prior_location": args.history_evidence_logit_prior_location,
        "history_evidence_logit_prior_min_count": args.history_evidence_logit_prior_min_count,
        "history_evidence_logit_prior_max_count": args.history_evidence_logit_prior_max_count,
        "history_evidence_logit_prior_min_seen_ratio": args.history_evidence_logit_prior_min_seen_ratio,
        "history_evidence_logit_prior_max_logit": args.history_evidence_logit_prior_max_logit,
        "history_evidence_logit_prior_component_cap": args.history_evidence_logit_prior_component_cap,
        "history_evidence_logit_prior_weight_student": args.history_evidence_logit_prior_weight_student,
        "history_evidence_logit_prior_weight_exercise": args.history_evidence_logit_prior_weight_exercise,
        "history_evidence_logit_prior_weight_target_concept": args.history_evidence_logit_prior_weight_target_concept,
        "history_evidence_logit_prior_weight_concept": args.history_evidence_logit_prior_weight_concept,
        "history_evidence_logit_prior_weight_mastery": args.history_evidence_logit_prior_weight_mastery,
        "history_evidence_logit_prior_prior_weight": args.history_evidence_logit_prior_prior_weight,
        "history_evidence_logit_prior_mastery_confidence_cap": (
            args.history_evidence_logit_prior_mastery_confidence_cap
        ),
        "history_evidence_cognitive_alignment_weight": args.history_evidence_cognitive_alignment_weight,
        "history_evidence_output_calibration": args.history_evidence_output_calibration,
        "history_evidence_output_calibration_min_count": args.history_evidence_output_calibration_min_count,
        "history_evidence_output_calibration_max_count": args.history_evidence_output_calibration_max_count,
        "history_evidence_output_calibration_min_seen_ratio": (
            args.history_evidence_output_calibration_min_seen_ratio
        ),
        "history_evidence_output_calibration_max_logit": args.history_evidence_output_calibration_max_logit,
        "history_evidence_output_calibration_prior_strength": (
            args.history_evidence_output_calibration_prior_strength
        ),
        "history_evidence_output_calibration_concept_confidence_cap": (
            args.history_evidence_output_calibration_concept_confidence_cap
        ),
        "history_evidence_output_calibration_exercise_confidence_cap": (
            args.history_evidence_output_calibration_exercise_confidence_cap
        ),
        "history_evidence_output_calibration_student_confidence_cap": (
            args.history_evidence_output_calibration_student_confidence_cap
        ),
        "exercise_evidence_prior_residual": args.exercise_evidence_prior_residual,
        "exercise_evidence_prior_min_count": args.exercise_evidence_prior_min_count,
        "exercise_evidence_prior_max_logit": args.exercise_evidence_prior_max_logit,
        "exercise_evidence_prior_strength": args.exercise_evidence_prior_strength,
        "exercise_evidence_prior_confidence_cap": args.exercise_evidence_prior_confidence_cap,
        "exercise_evidence_difficulty_adapter": args.exercise_evidence_difficulty_adapter,
        "exercise_evidence_difficulty_adapter_min_count": args.exercise_evidence_difficulty_adapter_min_count,
        "exercise_evidence_difficulty_adapter_max_logit": args.exercise_evidence_difficulty_adapter_max_logit,
        "exercise_evidence_difficulty_adapter_strength": args.exercise_evidence_difficulty_adapter_strength,
        "exercise_evidence_difficulty_adapter_confidence_cap": args.exercise_evidence_difficulty_adapter_confidence_cap,
        "exercise_evidence_difficulty_init": args.exercise_evidence_difficulty_init,
        "exercise_evidence_difficulty_init_min_count": args.exercise_evidence_difficulty_init_min_count,
        "exercise_evidence_difficulty_init_max_abs_logit": args.exercise_evidence_difficulty_init_max_abs_logit,
        "exercise_evidence_difficulty_init_strength": args.exercise_evidence_difficulty_init_strength,
        "exercise_evidence_difficulty_init_confidence_cap": args.exercise_evidence_difficulty_init_confidence_cap,
        "exercise_evidence_difficulty_init_count": difficulty_init_count,
        "exercise_evidence_difficulty_regularization_weight": args.exercise_evidence_difficulty_regularization_weight,
        "exercise_evidence_difficulty_regularization_min_count": args.exercise_evidence_difficulty_regularization_min_count,
        "exercise_evidence_difficulty_regularization_max_abs_logit": (
            args.exercise_evidence_difficulty_regularization_max_abs_logit
        ),
        "exercise_evidence_difficulty_regularization_strength": (
            args.exercise_evidence_difficulty_regularization_strength
        ),
        "exercise_evidence_difficulty_regularization_confidence_cap": (
            args.exercise_evidence_difficulty_regularization_confidence_cap
        ),
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
        "concept_evidence_readout_max_count": args.concept_evidence_readout_max_count,
        "concept_evidence_readout_min_seen_ratio": args.concept_evidence_readout_min_seen_ratio,
        "concept_evidence_readout_max_logit": args.concept_evidence_readout_max_logit,
        "concept_evidence_prior_residual": args.concept_evidence_prior_residual,
        "concept_evidence_prior_min_count": args.concept_evidence_prior_min_count,
        "concept_evidence_prior_max_count": args.concept_evidence_prior_max_count,
        "concept_evidence_prior_min_seen_ratio": args.concept_evidence_prior_min_seen_ratio,
        "concept_evidence_prior_max_logit": args.concept_evidence_prior_max_logit,
        "concept_evidence_prior_strength": args.concept_evidence_prior_strength,
        "concept_evidence_prior_confidence_cap": args.concept_evidence_prior_confidence_cap,
        "student_evidence_ability_prior_residual": args.student_evidence_ability_prior_residual,
        "student_evidence_ability_prior_min_attempts": args.student_evidence_ability_prior_min_attempts,
        "student_evidence_ability_prior_max_logit": args.student_evidence_ability_prior_max_logit,
        "student_evidence_ability_prior_strength": args.student_evidence_ability_prior_strength,
        "student_evidence_ability_prior_confidence_cap": args.student_evidence_ability_prior_confidence_cap,
        "student_evidence_gs_prior_residual": args.student_evidence_gs_prior_residual,
        "student_evidence_gs_prior_min_attempts": args.student_evidence_gs_prior_min_attempts,
        "student_evidence_gs_prior_max_logit": args.student_evidence_gs_prior_max_logit,
        "student_evidence_gs_prior_strength": args.student_evidence_gs_prior_strength,
        "student_evidence_gs_prior_confidence_cap": args.student_evidence_gs_prior_confidence_cap,
        "concept_evidence_calibrated_readout": args.concept_evidence_calibrated_readout,
        "concept_evidence_calibrated_readout_min_count": args.concept_evidence_calibrated_readout_min_count,
        "concept_evidence_calibrated_readout_max_count": args.concept_evidence_calibrated_readout_max_count,
        "concept_evidence_calibrated_readout_min_seen_ratio": args.concept_evidence_calibrated_readout_min_seen_ratio,
        "concept_evidence_calibrated_readout_max_logit": args.concept_evidence_calibrated_readout_max_logit,
        "concept_evidence_calibrated_readout_prior_strength": args.concept_evidence_calibrated_readout_prior_strength,
        "concept_evidence_calibrated_readout_confidence_cap": args.concept_evidence_calibrated_readout_confidence_cap,
        "history_evidence_fusion_readout": args.history_evidence_fusion_readout,
        "history_evidence_fusion_min_count": args.history_evidence_fusion_min_count,
        "history_evidence_fusion_max_count": args.history_evidence_fusion_max_count,
        "history_evidence_fusion_min_seen_ratio": args.history_evidence_fusion_min_seen_ratio,
        "history_evidence_fusion_max_logit": args.history_evidence_fusion_max_logit,
        "history_evidence_fusion_prior_strength": args.history_evidence_fusion_prior_strength,
        "history_evidence_fusion_concept_confidence_cap": args.history_evidence_fusion_concept_confidence_cap,
        "history_evidence_fusion_exercise_confidence_cap": args.history_evidence_fusion_exercise_confidence_cap,
        "history_evidence_fusion_student_confidence_cap": args.history_evidence_fusion_student_confidence_cap,
        "history_evidence_linear_readout": args.history_evidence_linear_readout,
        "history_evidence_linear_min_count": args.history_evidence_linear_min_count,
        "history_evidence_linear_max_count": args.history_evidence_linear_max_count,
        "history_evidence_linear_min_seen_ratio": args.history_evidence_linear_min_seen_ratio,
        "history_evidence_linear_max_logit": args.history_evidence_linear_max_logit,
        "history_evidence_linear_prior_strength": args.history_evidence_linear_prior_strength,
        "history_evidence_linear_concept_confidence_cap": args.history_evidence_linear_concept_confidence_cap,
        "history_evidence_linear_exercise_confidence_cap": args.history_evidence_linear_exercise_confidence_cap,
        "history_evidence_linear_student_confidence_cap": args.history_evidence_linear_student_confidence_cap,
        "history_evidence_logit_prior_residual": args.history_evidence_logit_prior_residual,
        "history_evidence_logit_prior_location": args.history_evidence_logit_prior_location,
        "history_evidence_logit_prior_min_count": args.history_evidence_logit_prior_min_count,
        "history_evidence_logit_prior_max_count": args.history_evidence_logit_prior_max_count,
        "history_evidence_logit_prior_min_seen_ratio": args.history_evidence_logit_prior_min_seen_ratio,
        "history_evidence_logit_prior_max_logit": args.history_evidence_logit_prior_max_logit,
        "history_evidence_logit_prior_component_cap": args.history_evidence_logit_prior_component_cap,
        "history_evidence_logit_prior_weight_student": args.history_evidence_logit_prior_weight_student,
        "history_evidence_logit_prior_weight_exercise": args.history_evidence_logit_prior_weight_exercise,
        "history_evidence_logit_prior_weight_target_concept": args.history_evidence_logit_prior_weight_target_concept,
        "history_evidence_logit_prior_weight_concept": args.history_evidence_logit_prior_weight_concept,
        "history_evidence_logit_prior_weight_mastery": args.history_evidence_logit_prior_weight_mastery,
        "history_evidence_logit_prior_prior_weight": args.history_evidence_logit_prior_prior_weight,
        "history_evidence_logit_prior_mastery_confidence_cap": (
            args.history_evidence_logit_prior_mastery_confidence_cap
        ),
        "history_evidence_cognitive_alignment_weight": args.history_evidence_cognitive_alignment_weight,
        "history_evidence_output_calibration": args.history_evidence_output_calibration,
        "history_evidence_output_calibration_min_count": args.history_evidence_output_calibration_min_count,
        "history_evidence_output_calibration_max_count": args.history_evidence_output_calibration_max_count,
        "history_evidence_output_calibration_min_seen_ratio": (
            args.history_evidence_output_calibration_min_seen_ratio
        ),
        "history_evidence_output_calibration_max_logit": args.history_evidence_output_calibration_max_logit,
        "history_evidence_output_calibration_prior_strength": (
            args.history_evidence_output_calibration_prior_strength
        ),
        "history_evidence_output_calibration_concept_confidence_cap": (
            args.history_evidence_output_calibration_concept_confidence_cap
        ),
        "history_evidence_output_calibration_exercise_confidence_cap": (
            args.history_evidence_output_calibration_exercise_confidence_cap
        ),
        "history_evidence_output_calibration_student_confidence_cap": (
            args.history_evidence_output_calibration_student_confidence_cap
        ),
        "exercise_evidence_prior_residual": args.exercise_evidence_prior_residual,
        "exercise_evidence_prior_min_count": args.exercise_evidence_prior_min_count,
        "exercise_evidence_prior_max_logit": args.exercise_evidence_prior_max_logit,
        "exercise_evidence_prior_strength": args.exercise_evidence_prior_strength,
        "exercise_evidence_prior_confidence_cap": args.exercise_evidence_prior_confidence_cap,
        "exercise_evidence_difficulty_adapter": args.exercise_evidence_difficulty_adapter,
        "exercise_evidence_difficulty_adapter_min_count": args.exercise_evidence_difficulty_adapter_min_count,
        "exercise_evidence_difficulty_adapter_max_logit": args.exercise_evidence_difficulty_adapter_max_logit,
        "exercise_evidence_difficulty_adapter_strength": args.exercise_evidence_difficulty_adapter_strength,
        "exercise_evidence_difficulty_adapter_confidence_cap": args.exercise_evidence_difficulty_adapter_confidence_cap,
        "exercise_evidence_difficulty_init": args.exercise_evidence_difficulty_init,
        "exercise_evidence_difficulty_init_min_count": args.exercise_evidence_difficulty_init_min_count,
        "exercise_evidence_difficulty_init_max_abs_logit": args.exercise_evidence_difficulty_init_max_abs_logit,
        "exercise_evidence_difficulty_init_strength": args.exercise_evidence_difficulty_init_strength,
        "exercise_evidence_difficulty_init_confidence_cap": args.exercise_evidence_difficulty_init_confidence_cap,
        "exercise_evidence_difficulty_init_count": difficulty_init_count,
        "exercise_evidence_difficulty_regularization_weight": args.exercise_evidence_difficulty_regularization_weight,
        "exercise_evidence_difficulty_regularization_min_count": args.exercise_evidence_difficulty_regularization_min_count,
        "exercise_evidence_difficulty_regularization_max_abs_logit": (
            args.exercise_evidence_difficulty_regularization_max_abs_logit
        ),
        "exercise_evidence_difficulty_regularization_strength": (
            args.exercise_evidence_difficulty_regularization_strength
        ),
        "exercise_evidence_difficulty_regularization_confidence_cap": (
            args.exercise_evidence_difficulty_regularization_confidence_cap
        ),
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
