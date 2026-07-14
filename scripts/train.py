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
from models import (
    CountPriorBaseline,
    DecoupledCDM,
    DecoupledCDMEnsemble,
    DecoupledCDMV2,
    KaNCDBaseline,
    TwoStageTKCUKCCDM,
)
from trainers import evaluate_model, train_model
from utils import append_summary_csv, resolve_device, save_history_csv, set_global_seed, setup_logging, write_json


def _reset_cuda_peak_memory_if_available(device: str) -> None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return
    torch.cuda.set_device(torch.device(device))
    torch.cuda.reset_peak_memory_stats()


def _max_cuda_memory_allocated_gb(device: str) -> float | None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    torch.cuda.set_device(torch.device(device))
    return torch.cuda.max_memory_allocated() / (1024**3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the minimal decoupled CDM pipeline.")
    parser.add_argument("--dataset", default=None, help="Optional dataset key for default paths and hyperparameters.")
    parser.add_argument(
        "--model",
        choices=["v1", "v2", "b0", "kancd", "two_stage_tkc_ukc"],
        default="v1",
        help=(
            "Model variant. v1 is the frozen mainline (adapters allowed). v2 is the clean core with "
            "independent module flags for single-module attribution runs. b0 is the count-prior "
            "logistic baseline over train-history statistics. kancd is a faithful in-harness "
            "KaNCD reimplementation (low-rank mastery extrapolation baseline). "
            "two_stage_tkc_ukc is the clean two-module evidence/completion candidate."
        ),
    )
    parser.add_argument(
        "--kancd-latent-dim",
        type=int,
        default=64,
        help="Latent dimension for the KaNCD baseline's low-rank factorization.",
    )
    parser.add_argument(
        "--semantic-node-mode",
        choices=[
            "bidirectional_q",
            "raw_identity_control",
            "global_context_control",
        ],
        default="bidirectional_q",
        help="Q-specific semantic alignment and its two capacity-equal controls.",
    )
    parser.add_argument(
        "--evidence-representation-mode",
        choices=[
            "calibrated_history",
            "identity_raw_control",
            "calibrated_summary_control",
            "raw_summary_control",
        ],
        default="calibrated_history",
        help="Module 1 path; raw_summary_control is its clean capacity control.",
    )
    parser.add_argument(
        "--concept-prior-mode",
        choices=[
            "population_q",
            "semantic_q_control",
            "global_population_control",
        ],
        default="population_q",
        help="Concept-side population prior and its two clean controls.",
    )
    parser.add_argument(
        "--state-completion-mode",
        choices=[
            "query_attentive_field",
            "global_attentive_control",
            "personalized_interaction",
            "additive_personalized_control",
            "direct_prior_control",
        ],
        default="personalized_interaction",
        help="State-completion module path and clean capacity control.",
    )
    parser.add_argument(
        "--diagnosis-mode",
        choices=["target_conditioned", "monotonic_control"],
        default="target_conditioned",
        help="Diagnosis module path; monotonic_control is the standard decoder ablation.",
    )
    parser.add_argument("--completion-evidence-cap", type=float, default=20.0)
    parser.add_argument(
        "--context-target-frac",
        type=float,
        default=0.0,
        help="Hide this fraction of each student's train history and supervise only hidden responses.",
    )
    parser.add_argument(
        "--v2-ukc-propagation",
        action="store_true",
        help="V2 module 1: student-conditioned TKC->UKC propagation (zero-init residual on the static UKC).",
    )
    parser.add_argument(
        "--v2-ukc-layers",
        type=int,
        default=1,
        help="Propagation hops for --v2-ukc-propagation.",
    )
    parser.add_argument(
        "--v2-ukc-evidence-cap",
        type=float,
        default=20.0,
        help="Attempt-count cap for the evidence confidence weighting in v2 modules.",
    )
    parser.add_argument(
        "--v2-target-aware-readout",
        action="store_true",
        help="V2 module 2: per-concept mastery head plus target-concept-aware readout.",
    )
    parser.add_argument(
        "--v2-monotonic-readout",
        action="store_true",
        help="V2 module 3a: monotone-in-mastery interaction. Requires --v2-target-aware-readout.",
    )
    parser.add_argument(
        "--v2-bounded-gs",
        action="store_true",
        help="V2 module 3b: bounded guess/slip conditioned only on the exercise representation.",
    )
    parser.add_argument(
        "--v2-hybrid-readout",
        action="store_true",
        help=(
            "V2 module 2h: keep the pooled NCF match and add a zero-init target-concept-aware "
            "residual head plus the mastery head. Mutually exclusive with --v2-target-aware-readout."
        ),
    )
    parser.add_argument(
        "--v2-lowrank-mastery",
        action="store_true",
        help=(
            "V2 mastery-head surgery: low-rank student x concept extrapolation base plus a "
            "zero-init graph-state correction. Requires a mastery head "
            "(--v2-target-aware-readout or --v2-hybrid-readout)."
        ),
    )
    parser.add_argument("--v2-lowrank-dim", type=int, default=64)
    parser.add_argument(
        "--v2-mastery-aux-weight",
        type=float,
        default=0.0,
        help=(
            "Surgery v2: weight of an auxiliary BCE that predicts each response from the "
            "monotone mastery term ALONE (fixed scale 4.0), giving the mastery head direct "
            "training pressure. Requires --v2-monotonic-readout."
        ),
    )
    parser.add_argument(
        "--v2-response-graph",
        action="store_true",
        help=(
            "V2 encoder upgrade: ORCDF-style right/wrong response-graph LightGCN encoder feeding "
            "zero-init residuals into the exercise embeddings and the fused student state."
        ),
    )
    parser.add_argument("--v2-rg-layers", type=int, default=2)
    parser.add_argument("--v2-dual-graph", action="store_true", help="Mo-1: student-exercise response-graph co-propagation channel (gated fusion).")
    parser.add_argument("--v2-dual-graph-adaptive", action="store_true", help="Mo-1b: density-conditioned dual-graph gate (suppress on dense concept graphs).")
    parser.add_argument("--v2-router", action="store_true", help="Meta-router: structure-signal (density,coverage) gate over decoupling vs response-graph state.")
    parser.add_argument("--v2-attn-readout", action="store_true", help="Mo-2: target-conditioned attention pooling of per-concept states.")
    parser.add_argument("--v2-irt-head", action="store_true", help="Mo-3: MIRT structured cognitive logit term.")
    parser.add_argument("--v2-contrastive-weight", type=float, default=0.0, help="Tr-1: InfoNCE contrastive loss on student states (two dropout views).")
    parser.add_argument("--v2-consistency-weight", type=float, default=0.0, help="Tr-2: full-vs-masked-history prediction consistency loss.")
    parser.add_argument("--v2-consistency-adaptive", action="store_true", help="Tr-2b: coverage-scaled per-student masking rate.")
    parser.add_argument("--v2-curriculum", action="store_true", help="Tr-3: evidence-density curriculum ordering of minibatches.")
    parser.add_argument(
        "--v2-rg-primary",
        action="store_true",
        help=(
            "Make the student-exercise response-graph encoder a primary (xavier-init) co-encoder "
            "of the student state, not a starved zero-init residual. Supplies signal on low-density "
            "datasets where the concept co-occurrence graph is empty."
        ),
    )
    parser.add_argument(
        "--v2-history-dropout-frac",
        type=float,
        default=0.0,
        help=(
            "Training recipe: each epoch randomly drop this fraction of observed history entries "
            "(masks/evidence re-derived) before encoding — augmentation for inference from "
            "incomplete evidence. full_batch only."
        ),
    )
    parser.add_argument(
        "--v2-masked-response-weight",
        type=float,
        default=0.0,
        help=(
            "Training recipe: weight of the masked-response self-supervision — hide a fraction of "
            "history entries and predict the labels of training interactions on the hidden entries. "
            "full_batch only."
        ),
    )
    parser.add_argument("--v2-masked-response-frac", type=float, default=0.15)
    parser.add_argument(
        "--v2-rg-mastery",
        action="store_true",
        help=(
            "Deep response-graph integration: K-dim student/exercise embeddings propagated over "
            "the train-only right/wrong response graphs directly form the mastery logit base "
            "(replacement, not residual). Requires a mastery head."
        ),
    )
    parser.add_argument(
        "--v2-target-fusion",
        action="store_true",
        help=(
            "V2 module 2f: zero-init parallel match head over a target-local TKC/UKC state fused "
            "per exercise by its own concept composition."
        ),
    )
    parser.add_argument("--v2-gs-max-guess", type=float, default=0.3)
    parser.add_argument("--v2-gs-max-slip", type=float, default=0.3)
    parser.add_argument(
        "--v2-readout-dropout",
        type=float,
        default=0.0,
        help="Dropout inside the v2 prediction heads (NCF match / concept scorer). Regularizes sparse data.",
    )
    parser.add_argument(
        "--v2-ukc-consistency-weight",
        type=float,
        default=0.0,
        help=(
            "V2 module 4 (weak form): weight of the masked-concept state consistency loss. "
            "Demotes random tested concepts to pseudo-UKC and pulls the inferred state toward the "
            "stop-gradient observed TKC state. No response labels are used. Requires "
            "--v2-ukc-propagation and full_batch training."
        ),
    )
    parser.add_argument(
        "--v2-ukc-consistency-drop-frac",
        type=float,
        default=0.2,
        help="Fraction of each student's tested concepts demoted per epoch for the consistency loss.",
    )
    parser.add_argument("--b0-prior-weight", type=float, default=5.0)
    parser.add_argument("--b0-component-cap", type=float, default=3.0)
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
    parser.add_argument(
        "--student-batch-size",
        type=int,
        default=None,
        help=(
            "Number of students per optimizer step for --training-mode "
            "student_recompute_minibatch. This avoids recomputing propagation "
            "for every student in Junyi-scale runs."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Adam optimizer weight decay. Defaults to 0.0 to preserve existing experiment baselines.",
    )
    parser.add_argument(
        "--training-mode",
        choices=["full_batch", "recompute_minibatch", "student_recompute_minibatch"],
        default="full_batch",
        help=(
            "Training loop semantics. full_batch preserves the current mainline; "
            "recompute_minibatch makes --batch-size effective; "
            "student_recompute_minibatch batches interactions by student IDs."
        ),
    )
    parser.add_argument(
        "--checkpoint-selection-metric",
        choices=["auc", "acc", "loss", "rmse", "brier", "ece"],
        default="auc",
        help="Validation metric used to choose the saved checkpoint. auc preserves current behavior.",
    )
    parser.add_argument(
        "--checkpoint-selection-start-epoch",
        type=int,
        default=1,
        help="First epoch eligible for validation checkpoint selection.",
    )
    parser.add_argument(
        "--checkpoint-selection-window",
        type=int,
        default=1,
        help=(
            "Number of recent eligible validation metrics averaged for checkpoint selection. "
            "One preserves current behavior."
        ),
    )
    parser.add_argument("--concept-dim", type=int, default=32)
    parser.add_argument(
        "--dual-cdm-ensemble",
        action="store_true",
        help="Train a single-checkpoint two-tower CDM whose probabilities are averaged inside the model.",
    )
    parser.add_argument(
        "--dual-cdm-secondary-concept-dim",
        type=int,
        default=80,
        help="Concept dimension for the secondary tower when --dual-cdm-ensemble is enabled.",
    )
    parser.add_argument(
        "--dual-cdm-branch-bce-weight",
        type=float,
        default=0.0,
        help="Auxiliary BCE weight applied to each dual CDM tower before probability mixing.",
    )
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
        "--student-fusion-mode",
        choices=["adaptive", "tkc_only", "ukc_only", "mean"],
        default="adaptive",
        help="Student-level TKC/UKC fusion mode. Use tkc_only/ukc_only/mean for direct ablations.",
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
    parser.add_argument("--concept-evidence-prior-min-confidence", type=float, default=0.0)
    parser.add_argument("--concept-evidence-prior-min-abs-mastery", type=float, default=0.0)
    parser.add_argument("--concept-evidence-prior-positive-scale", type=float, default=1.0)
    parser.add_argument("--concept-evidence-prior-negative-scale", type=float, default=1.0)
    parser.add_argument(
        "--concept-evidence-prior-train-start-epoch",
        type=int,
        default=1,
        help="First training epoch where the concept evidence prior can affect train-mode forward passes.",
    )
    parser.add_argument(
        "--concept-evidence-prior-train-warmup-epochs",
        type=int,
        default=0,
        help=(
            "Number of training epochs used to linearly ramp the concept evidence prior max-logit after "
            "--concept-evidence-prior-train-start-epoch. Zero applies the full prior immediately."
        ),
    )
    parser.add_argument(
        "--concept-evidence-prior-apply-mode",
        choices=["all", "eval_only", "train_only"],
        default="all",
    )
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
        "--history-evidence-cognitive-alignment-final-weight",
        type=float,
        default=None,
        help="Linearly anneal cognitive alignment weight to this value by the final epoch; omit to keep it fixed.",
    )
    parser.add_argument(
        "--history-evidence-cognitive-alignment-anneal-start-epoch",
        type=int,
        default=1,
        help="Epoch where cognitive alignment weight annealing starts; earlier epochs keep the initial weight.",
    )
    parser.add_argument(
        "--history-evidence-cognitive-alignment-anneal-end-epoch",
        type=int,
        default=None,
        help="Epoch where cognitive alignment weight annealing reaches the final weight; defaults to --epochs.",
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
    parser.add_argument(
        "--evaluation-stage",
        choices=["validation", "confirmation"],
        default="confirmation",
        help="validation skips test evaluation; confirmation evaluates the frozen model on test.",
    )
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
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be non-negative.")
    if args.dual_cdm_secondary_concept_dim < 1:
        raise ValueError("--dual-cdm-secondary-concept-dim must be positive.")
    if args.dual_cdm_branch_bce_weight < 0.0:
        raise ValueError("--dual-cdm-branch-bce-weight must be non-negative.")
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
    if args.concept_evidence_prior_min_confidence < 0.0 or args.concept_evidence_prior_min_confidence > 1.0:
        raise ValueError("--concept-evidence-prior-min-confidence must be in [0, 1].")
    if args.concept_evidence_prior_min_abs_mastery < 0.0 or args.concept_evidence_prior_min_abs_mastery > 1.0:
        raise ValueError("--concept-evidence-prior-min-abs-mastery must be in [0, 1].")
    if args.concept_evidence_prior_positive_scale < 0.0:
        raise ValueError("--concept-evidence-prior-positive-scale must be non-negative.")
    if args.concept_evidence_prior_negative_scale < 0.0:
        raise ValueError("--concept-evidence-prior-negative-scale must be non-negative.")
    if args.concept_evidence_prior_train_start_epoch < 1:
        raise ValueError("--concept-evidence-prior-train-start-epoch must be positive.")
    if args.concept_evidence_prior_train_warmup_epochs < 0:
        raise ValueError("--concept-evidence-prior-train-warmup-epochs must be non-negative.")
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
        args.history_evidence_cognitive_alignment_final_weight is not None
        and args.history_evidence_cognitive_alignment_final_weight < 0.0
    ):
        raise ValueError("--history-evidence-cognitive-alignment-final-weight must be non-negative.")
    if args.history_evidence_cognitive_alignment_anneal_start_epoch < 1:
        raise ValueError("--history-evidence-cognitive-alignment-anneal-start-epoch must be positive.")
    if (
        args.history_evidence_cognitive_alignment_anneal_end_epoch is not None
        and args.history_evidence_cognitive_alignment_anneal_end_epoch
        < args.history_evidence_cognitive_alignment_anneal_start_epoch
    ):
        raise ValueError(
            "--history-evidence-cognitive-alignment-anneal-end-epoch must be at least "
            "--history-evidence-cognitive-alignment-anneal-start-epoch."
        )
    if (
        args.history_evidence_cognitive_alignment_weight > 0.0
        and not args.history_evidence_logit_prior_residual
    ):
        raise ValueError(
            "--history-evidence-logit-prior-residual must be enabled with "
            "history evidence alignment losses."
        )
    if (
        args.history_evidence_cognitive_alignment_weight > 0.0
        and args.history_evidence_logit_prior_location != "loss_only"
    ):
        raise ValueError(
            "--history-evidence-logit-prior-location must be loss_only with "
            "history evidence alignment losses."
        )
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


V1_ONLY_FLAG_ATTRS = (
    "dual_cdm_ensemble",
    "high_concept_logit_adapter",
    "pairwise_history_interaction_adapter",
    "gs_difficulty_adapter",
    "interpretable_readout_expert_adapter",
    "student_conditioned_ukc_readout_residual",
    "concept_evidence_readout_residual",
    "concept_evidence_prior_residual",
    "history_evidence_logit_prior_residual",
)

V2_ONLY_FLAG_ATTRS = (
    "v2_ukc_propagation",
    "v2_target_aware_readout",
    "v2_monotonic_readout",
    "v2_bounded_gs",
    "v2_hybrid_readout",
    "v2_target_fusion",
    "v2_lowrank_mastery",
    "v2_response_graph",
    "v2_rg_primary",
    "v2_rg_mastery",
)


def validate_model_args(args: argparse.Namespace) -> None:
    if args.model != "v1":
        enabled_v1_flags = [name for name in V1_ONLY_FLAG_ATTRS if getattr(args, name)]
        if enabled_v1_flags:
            raise ValueError(
                f"--model {args.model} does not accept v1 adapter flags (keep attribution clean): "
                + ", ".join(enabled_v1_flags)
            )
        if args.history_evidence_cognitive_alignment_weight > 0.0:
            raise ValueError(f"--model {args.model} does not support history evidence alignment losses.")
        if args.dual_cdm_branch_bce_weight > 0.0:
            raise ValueError(f"--model {args.model} does not accept --dual-cdm-branch-bce-weight.")
        if args.graph_mode != "single":
            raise ValueError(f"--model {args.model} supports single-graph mode only.")
    if args.model != "v2":
        enabled_v2_flags = [name for name in V2_ONLY_FLAG_ATTRS if getattr(args, name)]
        if enabled_v2_flags:
            raise ValueError(
                f"v2 module flags require --model v2: " + ", ".join(enabled_v2_flags)
            )
    completion_nondefaults = (
        args.semantic_node_mode != "bidirectional_q"
        or args.evidence_representation_mode != "calibrated_history"
        or args.concept_prior_mode != "population_q"
        or args.state_completion_mode != "personalized_interaction"
        or args.diagnosis_mode != "target_conditioned"
        or args.completion_evidence_cap != 20.0
        or args.context_target_frac != 0.0
    )
    if args.model != "two_stage_tkc_ukc" and completion_nondefaults:
        raise ValueError(
            "Two-stage module flags require --model two_stage_tkc_ukc."
        )
    if args.model == "two_stage_tkc_ukc":
        if args.seed != 42:
            raise ValueError("The active research protocol fixes this model to --seed 42.")
        if args.completion_evidence_cap <= 0.0:
            raise ValueError("--completion-evidence-cap must be positive.")
        if not 0.0 <= args.context_target_frac < 1.0:
            raise ValueError("--context-target-frac must be in [0, 1).")
        if (
            args.context_target_frac > 0.0
            and args.training_mode != "student_recompute_minibatch"
        ):
            raise ValueError(
                "--context-target-frac requires student_recompute_minibatch."
            )
    if args.v2_monotonic_readout and not (args.v2_target_aware_readout or args.v2_hybrid_readout):
        raise ValueError("--v2-monotonic-readout requires --v2-target-aware-readout or --v2-hybrid-readout.")
    if args.v2_hybrid_readout and args.v2_target_aware_readout:
        raise ValueError("--v2-hybrid-readout and --v2-target-aware-readout are mutually exclusive.")
    if args.v2_lowrank_mastery and not (args.v2_target_aware_readout or args.v2_hybrid_readout):
        raise ValueError("--v2-lowrank-mastery requires --v2-target-aware-readout or --v2-hybrid-readout.")
    if args.v2_rg_mastery and not (args.v2_target_aware_readout or args.v2_hybrid_readout):
        raise ValueError("--v2-rg-mastery requires --v2-target-aware-readout or --v2-hybrid-readout.")
    if args.v2_rg_mastery and args.v2_lowrank_mastery:
        raise ValueError("--v2-rg-mastery and --v2-lowrank-mastery are mutually exclusive.")
    if args.v2_lowrank_dim < 1:
        raise ValueError("--v2-lowrank-dim must be positive.")
    if args.v2_mastery_aux_weight < 0.0:
        raise ValueError("--v2-mastery-aux-weight must be non-negative.")
    if args.v2_mastery_aux_weight > 0.0:
        if args.model != "v2" or not args.v2_monotonic_readout:
            raise ValueError("--v2-mastery-aux-weight requires --model v2 with --v2-monotonic-readout.")
        if args.training_mode not in {"full_batch", "student_recompute_minibatch"}:
            raise ValueError(
                "--v2-mastery-aux-weight supports full_batch and student_recompute_minibatch training."
            )
    if args.v2_ukc_consistency_weight < 0.0:
        raise ValueError("--v2-ukc-consistency-weight must be non-negative.")
    if args.v2_ukc_consistency_weight > 0.0:
        if args.model != "v2" or not args.v2_ukc_propagation:
            raise ValueError("--v2-ukc-consistency-weight requires --model v2 with --v2-ukc-propagation.")
        if args.training_mode != "full_batch":
            raise ValueError("--v2-ukc-consistency-weight is only implemented for full_batch training.")
    if not 0.0 < args.v2_ukc_consistency_drop_frac < 1.0:
        raise ValueError("--v2-ukc-consistency-drop-frac must be in (0, 1).")
    if not 0.0 <= args.v2_history_dropout_frac < 1.0:
        raise ValueError("--v2-history-dropout-frac must be in [0, 1).")
    if args.v2_masked_response_weight < 0.0:
        raise ValueError("--v2-masked-response-weight must be non-negative.")
    if not 0.0 < args.v2_masked_response_frac < 1.0:
        raise ValueError("--v2-masked-response-frac must be in (0, 1).")
    if (args.v2_history_dropout_frac > 0.0 or args.v2_masked_response_weight > 0.0):
        if args.model != "v2":
            raise ValueError("history dropout / masked response recipes require --model v2.")
        if args.training_mode != "full_batch":
            raise ValueError("history dropout / masked response recipes are only implemented for full_batch training.")
    if args.v2_ukc_layers < 1:
        raise ValueError("--v2-ukc-layers must be positive.")
    if args.v2_ukc_evidence_cap <= 0.0:
        raise ValueError("--v2-ukc-evidence-cap must be positive.")


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
    validate_model_args(args)
    set_global_seed(args.seed)
    logger, log_path = setup_logging(args.log_dir, name="train")
    resolved_device = str(resolve_device(args.device, args.gpus))
    _reset_cuda_peak_memory_if_available(resolved_device)
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
        bundles = {"train": single_bundle, "valid": single_bundle, "test": single_bundle}

    model_kwargs = dict(
        num_students=train_bundle.num_students,
        num_exercises=train_bundle.num_exercises,
        num_concepts=train_bundle.num_concepts,
        concept_dim=args.concept_dim,
        graph_mode=args.graph_mode,
        student_fusion_mode=args.student_fusion_mode,
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
        concept_evidence_prior_min_confidence=args.concept_evidence_prior_min_confidence,
        concept_evidence_prior_min_abs_mastery=args.concept_evidence_prior_min_abs_mastery,
        concept_evidence_prior_positive_scale=args.concept_evidence_prior_positive_scale,
        concept_evidence_prior_negative_scale=args.concept_evidence_prior_negative_scale,
        concept_evidence_prior_apply_mode=args.concept_evidence_prior_apply_mode,
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
    )
    if args.model == "two_stage_tkc_ukc":
        model = TwoStageTKCUKCCDM(
            num_students=train_bundle.num_students,
            num_exercises=train_bundle.num_exercises,
            num_concepts=train_bundle.num_concepts,
            concept_dim=args.concept_dim,
            semantic_node_mode=args.semantic_node_mode,
            evidence_mode=args.evidence_representation_mode,
            concept_prior_mode=args.concept_prior_mode,
            completion_mode=args.state_completion_mode,
            diagnosis_mode=args.diagnosis_mode,
            evidence_cap=args.completion_evidence_cap,
            readout_dropout=args.v2_readout_dropout,
            max_guess=args.v2_gs_max_guess,
            max_slip=args.v2_gs_max_slip,
        )
    elif args.model == "v2":
        model = DecoupledCDMV2(
            num_students=train_bundle.num_students,
            num_exercises=train_bundle.num_exercises,
            num_concepts=train_bundle.num_concepts,
            concept_dim=args.concept_dim,
            student_fusion_mode=args.student_fusion_mode,
            student_gate_prior_alpha=args.student_gate_prior_alpha,
            student_gate_prior_beta=args.student_gate_prior_beta,
            gs_mode=args.gs_mode,
            ukc_propagation=args.v2_ukc_propagation,
            ukc_propagation_layers=args.v2_ukc_layers,
            ukc_evidence_cap=args.v2_ukc_evidence_cap,
            target_aware_readout=args.v2_target_aware_readout,
            monotonic_readout=args.v2_monotonic_readout,
            bounded_gs=args.v2_bounded_gs,
            hybrid_readout=args.v2_hybrid_readout,
            target_fusion=args.v2_target_fusion,
            lowrank_mastery=args.v2_lowrank_mastery,
            lowrank_dim=args.v2_lowrank_dim,
            mastery_aux_head=args.v2_mastery_aux_weight > 0.0,
            response_graph_encoder=args.v2_response_graph,
            response_graph_layers=args.v2_rg_layers,
            rg_primary=args.v2_rg_primary,
            rg_mastery=args.v2_rg_mastery,
            dual_graph=args.v2_dual_graph,
            dual_graph_adaptive=args.v2_dual_graph_adaptive,
            router=args.v2_router,
            attn_readout=args.v2_attn_readout,
            irt_head=args.v2_irt_head,
            readout_dropout=args.v2_readout_dropout,
            gs_max_guess=args.v2_gs_max_guess,
            gs_max_slip=args.v2_gs_max_slip,
        )
    elif args.model == "b0":
        model = CountPriorBaseline(
            num_students=train_bundle.num_students,
            num_exercises=train_bundle.num_exercises,
            num_concepts=train_bundle.num_concepts,
            prior_weight=args.b0_prior_weight,
            component_cap=args.b0_component_cap,
        )
    elif args.model == "kancd":
        model = KaNCDBaseline(
            num_students=train_bundle.num_students,
            num_exercises=train_bundle.num_exercises,
            num_concepts=train_bundle.num_concepts,
            latent_dim=args.kancd_latent_dim,
        )
    elif args.dual_cdm_ensemble:
        model_kwargs["secondary_concept_dim"] = args.dual_cdm_secondary_concept_dim
        model = DecoupledCDMEnsemble(**model_kwargs)
    else:
        model = DecoupledCDM(**model_kwargs)
    architecture_fingerprint = getattr(model, "architecture_fingerprint", None)
    initialization_hash = (
        model.initialization_hash() if hasattr(model, "initialization_hash") else None
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
        student_batch_size=args.student_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        training_mode=args.training_mode,
        device=resolved_device,
        early_stop_patience=args.early_stop_patience,
        lr_scheduler_patience=args.lr_scheduler_patience,
        lr_scheduler_factor=args.lr_scheduler_factor,
        min_learning_rate=args.min_learning_rate,
        checkpoint_selection_metric=args.checkpoint_selection_metric,
        checkpoint_selection_start_epoch=args.checkpoint_selection_start_epoch,
        checkpoint_selection_window=args.checkpoint_selection_window,
        checkpoint_path=checkpoint_path,
        history_evidence_cognitive_alignment_weight=args.history_evidence_cognitive_alignment_weight,
        history_evidence_cognitive_alignment_final_weight=args.history_evidence_cognitive_alignment_final_weight,
        history_evidence_cognitive_alignment_anneal_start_epoch=(
            args.history_evidence_cognitive_alignment_anneal_start_epoch
        ),
        history_evidence_cognitive_alignment_anneal_end_epoch=(
            args.history_evidence_cognitive_alignment_anneal_end_epoch
        ),
        dual_tower_branch_bce_weight=args.dual_cdm_branch_bce_weight,
        concept_evidence_prior_train_start_epoch=args.concept_evidence_prior_train_start_epoch,
        concept_evidence_prior_train_warmup_epochs=args.concept_evidence_prior_train_warmup_epochs,
        ukc_consistency_weight=args.v2_ukc_consistency_weight,
        ukc_consistency_drop_frac=args.v2_ukc_consistency_drop_frac,
        mastery_aux_bce_weight=args.v2_mastery_aux_weight,
        contrastive_weight=args.v2_contrastive_weight,
        consistency_weight=args.v2_consistency_weight,
        consistency_adaptive=args.v2_consistency_adaptive,
        curriculum=args.v2_curriculum,
        history_dropout_frac=args.v2_history_dropout_frac,
        masked_response_weight=args.v2_masked_response_weight,
        masked_response_frac=args.v2_masked_response_frac,
        context_target_frac=args.context_target_frac,
    )
    evaluation_student_batch_size = (
        args.student_batch_size if args.training_mode == "student_recompute_minibatch" else None
    )
    test_metrics = (
        evaluate_model(
            bundle=test_bundle,
            model=model,
            device=resolved_device,
            student_batch_size=evaluation_student_batch_size,
        )
        if args.evaluation_stage == "confirmation"
        else None
    )
    valid_metrics = (
        evaluate_model(
            bundle=valid_bundle,
            model=model,
            device=resolved_device,
            student_batch_size=evaluation_student_batch_size,
        )
        if valid_bundle is not None
        else None
    )
    max_cuda_memory_allocated_gb = _max_cuda_memory_allocated_gb(resolved_device)

    v2_flag_snapshot = {
        "model": args.model,
        "v2_ukc_propagation": args.v2_ukc_propagation,
        "v2_ukc_layers": args.v2_ukc_layers,
        "v2_ukc_evidence_cap": args.v2_ukc_evidence_cap,
        "v2_target_aware_readout": args.v2_target_aware_readout,
        "v2_monotonic_readout": args.v2_monotonic_readout,
        "v2_bounded_gs": args.v2_bounded_gs,
        "v2_hybrid_readout": args.v2_hybrid_readout,
        "v2_target_fusion": args.v2_target_fusion,
        "v2_ukc_consistency_weight": args.v2_ukc_consistency_weight,
        "v2_ukc_consistency_drop_frac": args.v2_ukc_consistency_drop_frac,
        "v2_lowrank_mastery": args.v2_lowrank_mastery,
        "v2_lowrank_dim": args.v2_lowrank_dim,
        "v2_mastery_aux_weight": args.v2_mastery_aux_weight,
        "v2_history_dropout_frac": args.v2_history_dropout_frac,
        "v2_masked_response_weight": args.v2_masked_response_weight,
        "v2_masked_response_frac": args.v2_masked_response_frac,
        "v2_response_graph": args.v2_response_graph,
        "v2_rg_layers": args.v2_rg_layers,
        "v2_rg_primary": args.v2_rg_primary,
        "v2_dual_graph": args.v2_dual_graph,
        "v2_dual_graph_adaptive": args.v2_dual_graph_adaptive,
        "v2_router": args.v2_router,
        "v2_consistency_adaptive": args.v2_consistency_adaptive,
        "v2_attn_readout": args.v2_attn_readout,
        "v2_irt_head": args.v2_irt_head,
        "v2_contrastive_weight": args.v2_contrastive_weight,
        "v2_consistency_weight": args.v2_consistency_weight,
        "v2_curriculum": args.v2_curriculum,
        "v2_rg_mastery": args.v2_rg_mastery,
        "v2_readout_dropout": args.v2_readout_dropout,
        "v2_gs_max_guess": args.v2_gs_max_guess,
        "v2_gs_max_slip": args.v2_gs_max_slip,
        "b0_prior_weight": args.b0_prior_weight,
        "b0_component_cap": args.b0_component_cap,
        "kancd_latent_dim": args.kancd_latent_dim,
        "semantic_node_mode": args.semantic_node_mode,
        "evidence_representation_mode": args.evidence_representation_mode,
        "concept_prior_mode": args.concept_prior_mode,
        "state_completion_mode": args.state_completion_mode,
        "diagnosis_mode": args.diagnosis_mode,
        "completion_evidence_cap": args.completion_evidence_cap,
        "context_target_frac": args.context_target_frac,
        "architecture_fingerprint": architecture_fingerprint,
        "initialization_hash": initialization_hash,
    }
    output = {
        "dataset": args.dataset,
        "evaluation_stage": args.evaluation_stage,
        **v2_flag_snapshot,
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
        "dual_cdm_ensemble": args.dual_cdm_ensemble,
        "dual_cdm_secondary_concept_dim": args.dual_cdm_secondary_concept_dim,
        "dual_cdm_branch_bce_weight": args.dual_cdm_branch_bce_weight,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "student_batch_size": args.student_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "training_mode": args.training_mode,
        "checkpoint_selection_metric": result.checkpoint_selection_metric,
        "checkpoint_selection_start_epoch": result.checkpoint_selection_start_epoch,
        "checkpoint_selection_window": result.checkpoint_selection_window,
        "best_validation_score": result.best_validation_score,
        "gs_mode": args.gs_mode,
        "graph_mode": args.graph_mode,
        "student_fusion_mode": args.student_fusion_mode,
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
        "concept_evidence_prior_min_confidence": args.concept_evidence_prior_min_confidence,
        "concept_evidence_prior_min_abs_mastery": args.concept_evidence_prior_min_abs_mastery,
        "concept_evidence_prior_positive_scale": args.concept_evidence_prior_positive_scale,
        "concept_evidence_prior_negative_scale": args.concept_evidence_prior_negative_scale,
        "concept_evidence_prior_train_start_epoch": args.concept_evidence_prior_train_start_epoch,
        "concept_evidence_prior_train_warmup_epochs": args.concept_evidence_prior_train_warmup_epochs,
        "concept_evidence_prior_apply_mode": args.concept_evidence_prior_apply_mode,
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
        "history_evidence_cognitive_alignment_final_weight": (
            args.history_evidence_cognitive_alignment_final_weight
        ),
        "history_evidence_cognitive_alignment_anneal_start_epoch": (
            args.history_evidence_cognitive_alignment_anneal_start_epoch
        ),
        "history_evidence_cognitive_alignment_anneal_end_epoch": (
            args.history_evidence_cognitive_alignment_anneal_end_epoch
        ),
        "seed": args.seed,
        "device": resolved_device,
        "max_rows": args.max_rows,
        "final_loss": result.final_loss,
        "best_val_auc": result.best_val_auc,
        "best_epoch": result.best_epoch,
        "max_cuda_memory_allocated_gb": max_cuda_memory_allocated_gb,
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
        # v1 keeps the historical experiment_results.csv column set unchanged;
        # v2/b0 rows carry the model/module columns and go to a separate CSV.
        **(v2_flag_snapshot if args.model != "v1" else {}),
        "train_interactions": args.train_interactions or args.interactions,
        "valid_interactions": args.valid_interactions,
        "test_interactions": args.test_interactions or args.interactions,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "student_batch_size": args.student_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "training_mode": args.training_mode,
        "lr_scheduler_patience": args.lr_scheduler_patience,
        "lr_scheduler_factor": args.lr_scheduler_factor,
        "min_learning_rate": args.min_learning_rate,
        "device": resolved_device,
        "concept_dim": args.concept_dim,
        "dual_cdm_ensemble": args.dual_cdm_ensemble,
        "dual_cdm_secondary_concept_dim": args.dual_cdm_secondary_concept_dim,
        "dual_cdm_branch_bce_weight": args.dual_cdm_branch_bce_weight,
        "gs_mode": args.gs_mode,
        "graph_mode": args.graph_mode,
        "student_fusion_mode": args.student_fusion_mode,
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
        "concept_evidence_prior_min_confidence": args.concept_evidence_prior_min_confidence,
        "concept_evidence_prior_min_abs_mastery": args.concept_evidence_prior_min_abs_mastery,
        "concept_evidence_prior_positive_scale": args.concept_evidence_prior_positive_scale,
        "concept_evidence_prior_negative_scale": args.concept_evidence_prior_negative_scale,
        "concept_evidence_prior_train_start_epoch": args.concept_evidence_prior_train_start_epoch,
        "concept_evidence_prior_train_warmup_epochs": args.concept_evidence_prior_train_warmup_epochs,
        "concept_evidence_prior_apply_mode": args.concept_evidence_prior_apply_mode,
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
        "history_evidence_cognitive_alignment_final_weight": (
            args.history_evidence_cognitive_alignment_final_weight
        ),
        "history_evidence_cognitive_alignment_anneal_start_epoch": (
            args.history_evidence_cognitive_alignment_anneal_start_epoch
        ),
        "history_evidence_cognitive_alignment_anneal_end_epoch": (
            args.history_evidence_cognitive_alignment_anneal_end_epoch
        ),
        "seed": args.seed,
        "best_epoch": result.best_epoch,
        "max_cuda_memory_allocated_gb": max_cuda_memory_allocated_gb,
        "best_val_auc": result.best_val_auc,
        "evaluation_stage": args.evaluation_stage,
        "test_auc": None if test_metrics is None else test_metrics["auc"],
        "test_acc": None if test_metrics is None else test_metrics["acc"],
        "test_rmse": None if test_metrics is None else test_metrics["rmse"],
        "test_brier": None if test_metrics is None else test_metrics["brier"],
        "test_ece": None if test_metrics is None else test_metrics["ece"],
        "best_checkpoint_path": str(Path(checkpoint_path).resolve()),
        "output_json": str(output_path.resolve()),
        "history_csv": str(Path(history_path).resolve()),
    }
    if valid_metrics is not None:
        summary_row["valid_brier"] = valid_metrics["brier"]
        summary_row["valid_ece"] = valid_metrics["ece"]
    summary_csv_path = (
        "results/experiment_results.csv" if args.model == "v1" else "results/experiment_results_v2.csv"
    )
    append_summary_csv(summary_row, summary_csv_path)
    if test_metrics is None:
        logger.info(
            "Finished validation run: best_val_auc=%s; test evaluation intentionally skipped.",
            result.best_val_auc,
        )
    else:
        logger.info(
            "Finished confirmation run: best_val_auc=%s test_auc=%.6f test_acc=%.6f "
            "test_rmse=%.6f test_brier=%.6f test_ece=%.6f",
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
