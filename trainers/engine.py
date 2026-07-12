from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd
import torch
import torch.nn.functional as F

from data import StepDataBundle
from models import DecoupledCDM, DecoupledForwardOutput, UnifiedDecoupledCDM
from utils import compute_metrics


@dataclass
class TrainResult:
    history: list[dict[str, float]]
    final_loss: float
    best_val_auc: float
    best_epoch: int
    best_checkpoint_path: str | None
    checkpoint_selection_metric: str
    checkpoint_selection_start_epoch: int
    checkpoint_selection_window: int
    best_validation_score: float
    swa_start_epoch: int
    swa_epoch_count: int
    swa_checkpoint_path: str | None
    ema_start_epoch: int
    ema_decay: float
    ema_epoch_count: int
    ema_checkpoint_path: str | None


@dataclass
class EpochTrainStats:
    mean_loss: float
    optimizer_steps: int


def _bundle_tensors(bundle: StepDataBundle, device: torch.device) -> dict[str, torch.Tensor | None]:
    exercise_evidence_tensor = getattr(bundle, "exercise_evidence_tensor", None)
    return {
        "q_matrix": bundle.q_matrix_tensor.to(device),
        "concept_graph": bundle.concept_graph.to(device),
        "prerequisite_graph": bundle.prerequisite_graph.to(device) if bundle.prerequisite_graph is not None else None,
        "similarity_graph": bundle.similarity_graph.to(device) if bundle.similarity_graph is not None else None,
        "student_exercise_mask": bundle.student_exercise_mask.to(device),
        "student_tkc_mask": bundle.student_tkc_mask.to(device),
        "student_ukc_mask": bundle.student_ukc_mask.to(device),
        "student_concept_evidence": (
            bundle.student_concept_evidence_tensor.to(device)
            if bundle.student_concept_evidence_tensor is not None
            else None
        ),
        "exercise_evidence": (
            exercise_evidence_tensor.to(device) if exercise_evidence_tensor is not None else None
        ),
        "interaction_student_ids": bundle.interaction_student_ids.to(device),
        "interaction_exercise_ids": bundle.interaction_exercise_ids.to(device),
        "interaction_labels": bundle.interaction_labels.to(device),
        "response_matrix": bundle.response_matrix_tensor.to(device),
    }


def _forward_model(*, model: DecoupledCDM, **kwargs):
    """Call legacy and unified models without widening either public forward API."""
    if isinstance(model, UnifiedDecoupledCDM):
        kwargs.pop("prerequisite_graph", None)
        kwargs.pop("similarity_graph", None)
        kwargs.pop("exercise_evidence", None)
    return model(**kwargs)


def observed_mastery_evidence_loss(
    output: DecoupledForwardOutput,
    evidence: torch.Tensor,
    student_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if output.observed_mastery is None or output.mastery_observed_mask is None:
        raise ValueError("observed mastery output is required")
    selected = evidence if student_ids is None else evidence[student_ids]
    attempts, correct = selected.unbind(dim=-1)
    target = (correct + 1.0) / (attempts + 2.0)
    mask = output.mastery_observed_mask
    if not bool(mask.any()):
        raise ValueError("at least one observed mastery cell is required")
    return F.binary_cross_entropy(
        output.observed_mastery[mask],
        target[mask],
    )


def masked_completion_loss(
    output: DecoupledForwardOutput,
    evidence: torch.Tensor,
    student_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if (
        output.completion_predictions is None
        or output.mastery_observed_mask is None
    ):
        raise ValueError("completion predictions are required")
    selected = evidence if student_ids is None else evidence[student_ids]
    attempts, correct = selected[..., :2].unbind(dim=-1)
    target = (correct + 1.0) / (attempts + 2.0)
    mask = output.mastery_observed_mask
    if not bool(mask.any()):
        raise ValueError(
            "at least one observed completion target is required"
        )
    return F.binary_cross_entropy(
        output.completion_predictions[mask],
        target[mask],
    )


def _hash_interaction_rows(frame: pd.DataFrame) -> set[int]:
    if frame.empty:
        return set()
    normalized = frame.fillna("<NA>").astype(str)
    row_hashes = pd.util.hash_pandas_object(normalized, index=False)
    return {int(value) for value in row_hashes.tolist()}


def _validate_history_visibility(bundle: StepDataBundle) -> None:
    if bundle.allow_target_in_history:
        return
    target_hashes = _hash_interaction_rows(bundle.interactions)
    history_hashes = _hash_interaction_rows(bundle.history_interactions)
    if target_hashes.isdisjoint(history_hashes):
        return
    raise ValueError(
        f"{bundle.split_name} bundle reuses target interactions inside propagation history. "
        "Evaluation bundles must use history-only tensors built from past-visible interactions."
    )


def evaluate_model(
    *,
    bundle: StepDataBundle,
    model: DecoupledCDM,
    device: str = "cpu",
    student_batch_size: int | None = None,
) -> dict[str, float]:
    if student_batch_size is not None and student_batch_size <= 0:
        raise ValueError("student_batch_size must be positive when provided.")
    _validate_history_visibility(bundle)
    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()
    tensors = _bundle_tensors(bundle, torch_device)

    with torch.no_grad():
        if student_batch_size is None:
            output = _forward_model(
                model=model,
                q_matrix=tensors["q_matrix"],
                concept_graph=tensors["concept_graph"],
                prerequisite_graph=tensors["prerequisite_graph"],
                similarity_graph=tensors["similarity_graph"],
                student_exercise_mask=tensors["student_exercise_mask"],
                response_matrix=tensors["response_matrix"],
                student_tkc_mask=tensors["student_tkc_mask"],
                student_ukc_mask=tensors["student_ukc_mask"],
                student_concept_evidence=tensors["student_concept_evidence"],
                exercise_evidence=tensors["exercise_evidence"],
                target_student_ids=tensors["interaction_student_ids"],
                target_exercise_ids=tensors["interaction_exercise_ids"],
            )
            loss = F.binary_cross_entropy(output.probs, tensors["interaction_labels"])
            probs = output.probs
            labels = tensors["interaction_labels"]
        else:
            labels_parts = []
            probs_parts = []
            total_loss = tensors["interaction_labels"].new_zeros(())
            unique_student_ids = torch.unique(tensors["interaction_student_ids"], sorted=False)
            for start in range(0, unique_student_ids.numel(), student_batch_size):
                student_ids = unique_student_ids[start : start + student_batch_size]
                batch_indices = _interaction_indices_for_student_ids(
                    interaction_student_ids=tensors["interaction_student_ids"],
                    student_ids=student_ids,
                    num_students=tensors["student_exercise_mask"].size(0),
                )
                if batch_indices.numel() == 0:
                    continue
                batch_labels = tensors["interaction_labels"][batch_indices]
                output = _forward_model(
                    model=model,
                    q_matrix=tensors["q_matrix"],
                    concept_graph=tensors["concept_graph"],
                    prerequisite_graph=tensors["prerequisite_graph"],
                    similarity_graph=tensors["similarity_graph"],
                    student_exercise_mask=tensors["student_exercise_mask"],
                    response_matrix=tensors["response_matrix"],
                    student_tkc_mask=tensors["student_tkc_mask"],
                    student_ukc_mask=tensors["student_ukc_mask"],
                    student_concept_evidence=tensors["student_concept_evidence"],
                    exercise_evidence=tensors["exercise_evidence"],
                    target_student_ids=tensors["interaction_student_ids"][batch_indices],
                    target_exercise_ids=tensors["interaction_exercise_ids"][batch_indices],
                    use_student_subset=True,
                )
                total_loss = total_loss + F.binary_cross_entropy(output.probs, batch_labels, reduction="sum")
                labels_parts.append(batch_labels)
                probs_parts.append(output.probs)
            labels = torch.cat(labels_parts, dim=0)
            probs = torch.cat(probs_parts, dim=0)
            loss = total_loss / labels.numel()

    metrics = compute_metrics(
        labels=labels.detach().cpu().numpy(),
        probs=probs.detach().cpu().numpy(),
    )
    metrics["loss"] = float(loss.item())
    return metrics


def _interaction_indices_for_student_ids(
    *,
    interaction_student_ids: torch.Tensor,
    student_ids: torch.Tensor,
    num_students: int,
) -> torch.Tensor:
    selected_students = torch.zeros(num_students, dtype=torch.bool, device=interaction_student_ids.device)
    selected_students[student_ids] = True
    return torch.nonzero(selected_students[interaction_student_ids], as_tuple=False).squeeze(-1)


def _resolve_linear_epoch_weight(
    *,
    start_weight: float,
    final_weight: float | None,
    epoch: int,
    epochs: int,
    anneal_start_epoch: int = 1,
    anneal_end_epoch: int | None = None,
) -> float:
    if final_weight is None or epochs <= 1:
        return float(start_weight)
    end_epoch = epochs if anneal_end_epoch is None else anneal_end_epoch
    if epoch < anneal_start_epoch:
        return float(start_weight)
    if epoch >= end_epoch:
        return float(final_weight)
    if end_epoch == anneal_start_epoch:
        return float(final_weight)
    progress = float(epoch - anneal_start_epoch) / float(end_epoch - anneal_start_epoch)
    return float(start_weight) + (float(final_weight) - float(start_weight)) * progress


def _checkpoint_metric_mode(metric: str) -> str:
    if metric in {"auc", "acc"}:
        return "max"
    if metric in {"loss", "rmse", "brier", "ece"}:
        return "min"
    raise ValueError("checkpoint_selection_metric must be one of auc, acc, loss, rmse, brier, ece.")


def _is_checkpoint_metric_improved(*, metric: str, value: float, best_value: float) -> bool:
    mode = _checkpoint_metric_mode(metric)
    if mode == "max":
        return value > best_value
    return value < best_value


def train_model(
    *,
    train_bundle: StepDataBundle,
    model: DecoupledCDM,
    valid_bundle: StepDataBundle | None = None,
    epochs: int = 5,
    batch_size: int | None = None,
    student_batch_size: int | None = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    training_mode: str = "full_batch",
    device: str = "cpu",
    early_stop_patience: int = 5,
    lr_scheduler_patience: int = 10,
    lr_scheduler_factor: float = 0.5,
    min_learning_rate: float = 1e-5,
    checkpoint_selection_metric: str = "auc",
    checkpoint_selection_start_epoch: int = 1,
    checkpoint_selection_window: int = 1,
    checkpoint_path: str | None = None,
    swa_start_epoch: int = 0,
    swa_checkpoint_path: str | None = None,
    ema_start_epoch: int = 0,
    ema_decay: float = 0.99,
    ema_checkpoint_path: str | None = None,
    exercise_evidence_difficulty_regularization_weight: float = 0.0,
    exercise_evidence_difficulty_regularization_min_count: int = 1,
    exercise_evidence_difficulty_regularization_max_abs_logit: float = 0.25,
    exercise_evidence_difficulty_regularization_strength: float = 2.0,
    exercise_evidence_difficulty_regularization_confidence_cap: float = 200.0,
    history_evidence_cognitive_alignment_weight: float = 0.0,
    history_evidence_cognitive_alignment_final_weight: float | None = None,
    history_evidence_cognitive_alignment_anneal_start_epoch: int = 1,
    history_evidence_cognitive_alignment_anneal_end_epoch: int | None = None,
    history_evidence_cognitive_alignment_confidence_power: float = 0.0,
    history_evidence_cognitive_alignment_confidence_cap: float = 20.0,
    history_evidence_cognitive_alignment_confidence_floor: float = 0.0,
    history_evidence_cognitive_alignment_residual_power: float = 0.0,
    history_evidence_cognitive_alignment_residual_floor: float = 0.0,
    history_evidence_cognitive_rank_alignment_weight: float = 0.0,
    history_evidence_cognitive_rank_alignment_pair_count: int = 8192,
    history_evidence_cognitive_rank_alignment_min_target_gap: float = 0.0,
    history_evidence_output_alignment_weight: float = 0.0,
    history_evidence_output_alignment_confidence_power: float = 0.0,
    history_evidence_output_alignment_confidence_cap: float = 20.0,
    history_evidence_output_alignment_confidence_floor: float = 0.0,
    checkpoint_distillation_targets: torch.Tensor | None = None,
    checkpoint_distillation_weight: float = 0.0,
    checkpoint_distillation_start_epoch: int = 1,
    checkpoint_distillation_loss: str = "bce",
    dual_tower_branch_bce_weight: float = 0.0,
    concept_evidence_prior_train_start_epoch: int = 1,
    concept_evidence_prior_train_warmup_epochs: int = 0,
    ukc_consistency_weight: float = 0.0,
    ukc_consistency_drop_frac: float = 0.2,
    mastery_aux_bce_weight: float = 0.0,
    unified_evidence_loss_weight: float = 0.0,
    unified_completion_loss_weight: float = 0.0,
    history_dropout_frac: float = 0.0,
    masked_response_weight: float = 0.0,
    masked_response_frac: float = 0.15,
    contrastive_weight: float = 0.0,
    consistency_weight: float = 0.0,
    consistency_adaptive: bool = False,
    curriculum: bool = False,
) -> TrainResult:
    if training_mode not in {"full_batch", "recompute_minibatch", "student_recompute_minibatch"}:
        raise ValueError(f"Unsupported training_mode: {training_mode}")
    if ukc_consistency_weight < 0.0:
        raise ValueError("ukc_consistency_weight must be non-negative.")
    if ukc_consistency_weight > 0.0 and training_mode != "full_batch":
        raise ValueError("ukc_consistency_weight is only implemented for full_batch training.")
    if ukc_consistency_weight > 0.0 and not hasattr(model, "compute_ukc_consistency_loss"):
        raise ValueError("ukc_consistency_weight requires a model with compute_ukc_consistency_loss.")
    if isinstance(model, UnifiedDecoupledCDM):
        if (
            not math.isfinite(unified_evidence_loss_weight)
            or unified_evidence_loss_weight <= 0.0
        ):
            raise ValueError(
                "unified_evidence_loss_weight must be finite and positive "
                "for unified models."
            )
        if (
            not math.isfinite(unified_completion_loss_weight)
            or unified_completion_loss_weight < 0.0
        ):
            raise ValueError(
                "unified_completion_loss_weight must be finite and "
                "non-negative."
            )
        if (
            model.architecture.completion == "prior"
            and unified_completion_loss_weight != 0.0
        ):
            raise ValueError(
                "unified_completion_loss_weight must be zero for prior "
                "completion."
            )
        if (
            model.architecture.completion == "lowrank"
            and unified_completion_loss_weight <= 0.0
        ):
            raise ValueError(
                "unified_completion_loss_weight must be positive for "
                "lowrank completion."
            )
        if training_mode not in {"full_batch", "student_recompute_minibatch"}:
            raise ValueError(
                "unified mastery supervision supports full_batch and "
                "student_recompute_minibatch training."
            )
        model.set_checkpoint_loss_weights(
            evidence_loss_weight=unified_evidence_loss_weight,
            completion_loss_weight=unified_completion_loss_weight,
        )
    else:
        for name, value in (
            ("unified_evidence_loss_weight", unified_evidence_loss_weight),
            (
                "unified_completion_loss_weight",
                unified_completion_loss_weight,
            ),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if (
            unified_evidence_loss_weight > 0.0
            or unified_completion_loss_weight > 0.0
        ):
            raise ValueError(
                "unified loss weights are only supported by unified models."
            )
    if not 0.0 <= history_dropout_frac < 1.0:
        raise ValueError("history_dropout_frac must be in [0, 1).")
    if masked_response_weight < 0.0:
        raise ValueError("masked_response_weight must be non-negative.")
    if not 0.0 < masked_response_frac < 1.0:
        raise ValueError("masked_response_frac must be in (0, 1).")
    if (history_dropout_frac > 0.0 or masked_response_weight > 0.0) and training_mode != "full_batch":
        raise ValueError("history dropout / masked response losses are only implemented for full_batch training.")
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive when provided.")
    if student_batch_size is not None and student_batch_size <= 0:
        raise ValueError("student_batch_size must be positive when provided.")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative.")
    if swa_start_epoch < 0:
        raise ValueError("swa_start_epoch must be non-negative.")
    if ema_start_epoch < 0:
        raise ValueError("ema_start_epoch must be non-negative.")
    if ema_decay < 0.0 or ema_decay >= 1.0:
        raise ValueError("ema_decay must be in [0, 1).")
    if swa_start_epoch > 0 and ema_start_epoch > 0:
        raise ValueError("swa_start_epoch and ema_start_epoch cannot both be enabled.")
    if training_mode == "full_batch" and batch_size is not None:
        raise ValueError("full_batch training does not consume --batch-size; use --training-mode recompute_minibatch.")
    if training_mode == "full_batch" and student_batch_size is not None:
        raise ValueError(
            "full_batch training does not consume --student-batch-size; "
            "use --training-mode student_recompute_minibatch."
        )
    if training_mode == "recompute_minibatch" and batch_size is None:
        raise ValueError("recompute_minibatch training requires --batch-size.")
    if training_mode == "recompute_minibatch" and student_batch_size is not None:
        raise ValueError("recompute_minibatch training does not consume --student-batch-size.")
    if training_mode == "student_recompute_minibatch" and student_batch_size is None:
        raise ValueError("student_recompute_minibatch training requires --student-batch-size.")
    if training_mode == "student_recompute_minibatch" and batch_size is not None:
        raise ValueError("student_recompute_minibatch training does not consume --batch-size.")
    if exercise_evidence_difficulty_regularization_weight < 0.0:
        raise ValueError("exercise_evidence_difficulty_regularization_weight must be non-negative.")
    if exercise_evidence_difficulty_regularization_min_count < 1:
        raise ValueError("exercise_evidence_difficulty_regularization_min_count must be positive.")
    if exercise_evidence_difficulty_regularization_max_abs_logit <= 0.0:
        raise ValueError("exercise_evidence_difficulty_regularization_max_abs_logit must be positive.")
    if exercise_evidence_difficulty_regularization_strength <= 0.0:
        raise ValueError("exercise_evidence_difficulty_regularization_strength must be positive.")
    if exercise_evidence_difficulty_regularization_confidence_cap <= 0.0:
        raise ValueError("exercise_evidence_difficulty_regularization_confidence_cap must be positive.")
    if history_evidence_cognitive_alignment_weight < 0.0:
        raise ValueError("history_evidence_cognitive_alignment_weight must be non-negative.")
    if (
        history_evidence_cognitive_alignment_final_weight is not None
        and history_evidence_cognitive_alignment_final_weight < 0.0
    ):
        raise ValueError("history_evidence_cognitive_alignment_final_weight must be non-negative.")
    if history_evidence_cognitive_alignment_anneal_start_epoch < 1:
        raise ValueError("history_evidence_cognitive_alignment_anneal_start_epoch must be positive.")
    if (
        history_evidence_cognitive_alignment_anneal_end_epoch is not None
        and history_evidence_cognitive_alignment_anneal_end_epoch
        < history_evidence_cognitive_alignment_anneal_start_epoch
    ):
        raise ValueError(
            "history_evidence_cognitive_alignment_anneal_end_epoch must be at least "
            "history_evidence_cognitive_alignment_anneal_start_epoch."
        )
    if history_evidence_cognitive_alignment_confidence_power < 0.0:
        raise ValueError("history_evidence_cognitive_alignment_confidence_power must be non-negative.")
    if history_evidence_cognitive_alignment_confidence_cap <= 0.0:
        raise ValueError("history_evidence_cognitive_alignment_confidence_cap must be positive.")
    if (
        history_evidence_cognitive_alignment_confidence_floor < 0.0
        or history_evidence_cognitive_alignment_confidence_floor > 1.0
    ):
        raise ValueError("history_evidence_cognitive_alignment_confidence_floor must be in [0, 1].")
    if history_evidence_cognitive_alignment_residual_power < 0.0:
        raise ValueError("history_evidence_cognitive_alignment_residual_power must be non-negative.")
    if (
        history_evidence_cognitive_alignment_residual_floor < 0.0
        or history_evidence_cognitive_alignment_residual_floor > 1.0
    ):
        raise ValueError("history_evidence_cognitive_alignment_residual_floor must be in [0, 1].")
    if history_evidence_cognitive_rank_alignment_weight < 0.0:
        raise ValueError("history_evidence_cognitive_rank_alignment_weight must be non-negative.")
    if history_evidence_cognitive_rank_alignment_pair_count < 1:
        raise ValueError("history_evidence_cognitive_rank_alignment_pair_count must be positive.")
    if history_evidence_cognitive_rank_alignment_min_target_gap < 0.0:
        raise ValueError("history_evidence_cognitive_rank_alignment_min_target_gap must be non-negative.")
    if history_evidence_output_alignment_weight < 0.0:
        raise ValueError("history_evidence_output_alignment_weight must be non-negative.")
    if history_evidence_output_alignment_confidence_power < 0.0:
        raise ValueError("history_evidence_output_alignment_confidence_power must be non-negative.")
    if history_evidence_output_alignment_confidence_cap <= 0.0:
        raise ValueError("history_evidence_output_alignment_confidence_cap must be positive.")
    if (
        history_evidence_output_alignment_confidence_floor < 0.0
        or history_evidence_output_alignment_confidence_floor > 1.0
    ):
        raise ValueError("history_evidence_output_alignment_confidence_floor must be in [0, 1].")
    if checkpoint_distillation_weight < 0.0:
        raise ValueError("checkpoint_distillation_weight must be non-negative.")
    if dual_tower_branch_bce_weight < 0.0:
        raise ValueError("dual_tower_branch_bce_weight must be non-negative.")
    if checkpoint_distillation_start_epoch < 1:
        raise ValueError("checkpoint_distillation_start_epoch must be positive.")
    if checkpoint_distillation_loss not in {"bce", "standardized_logit_mse"}:
        raise ValueError("checkpoint_distillation_loss must be one of bce, standardized_logit_mse.")
    if checkpoint_selection_start_epoch < 1:
        raise ValueError("checkpoint_selection_start_epoch must be positive.")
    if checkpoint_selection_window < 1:
        raise ValueError("checkpoint_selection_window must be positive.")
    if concept_evidence_prior_train_start_epoch < 1:
        raise ValueError("concept_evidence_prior_train_start_epoch must be positive.")
    if concept_evidence_prior_train_warmup_epochs < 0:
        raise ValueError("concept_evidence_prior_train_warmup_epochs must be non-negative.")
    checkpoint_metric_mode = _checkpoint_metric_mode(checkpoint_selection_metric)

    torch_device = torch.device(device)
    model = model.to(torch_device)
    train_tensors = _bundle_tensors(train_bundle, torch_device)
    checkpoint_distillation_targets_tensor = None
    if checkpoint_distillation_targets is not None:
        checkpoint_distillation_targets_tensor = checkpoint_distillation_targets.to(torch_device)
        if checkpoint_distillation_targets_tensor.shape != train_tensors["interaction_labels"].shape:
            raise ValueError("checkpoint_distillation_targets must match train interaction label shape.")
        if (
            torch.any(checkpoint_distillation_targets_tensor < 0.0)
            or torch.any(checkpoint_distillation_targets_tensor > 1.0)
        ):
            raise ValueError("checkpoint_distillation_targets must contain probabilities in [0, 1].")
    if checkpoint_distillation_weight > 0.0 and checkpoint_distillation_targets_tensor is None:
        raise ValueError("checkpoint_distillation_targets are required when checkpoint_distillation_weight is positive.")
    difficulty_prior_target = None
    difficulty_prior_mask = None
    if exercise_evidence_difficulty_regularization_weight > 0.0:
        difficulty_prior_target, difficulty_prior_mask = model.build_exercise_difficulty_prior_target(
            exercise_evidence=train_tensors["exercise_evidence"],
            min_count=exercise_evidence_difficulty_regularization_min_count,
            max_abs_logit=exercise_evidence_difficulty_regularization_max_abs_logit,
            strength=exercise_evidence_difficulty_regularization_strength,
            confidence_cap=exercise_evidence_difficulty_regularization_confidence_cap,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        min_lr=min_learning_rate,
    )
    history: list[dict[str, float]] = []
    best_state = None
    best_val_auc = float("nan")
    best_validation_score = float("-inf") if checkpoint_metric_mode == "max" else float("inf")
    best_epoch = 0
    patience_counter = 0
    checkpoint_score_window: list[float] = []
    swa_state: dict[str, torch.Tensor] | None = None
    swa_epoch_count = 0
    ema_state: dict[str, torch.Tensor] | None = None
    ema_epoch_count = 0

    for epoch in range(1, epochs + 1):
        cognitive_alignment_epoch_weight = _resolve_linear_epoch_weight(
            start_weight=history_evidence_cognitive_alignment_weight,
            final_weight=history_evidence_cognitive_alignment_final_weight,
            epoch=epoch,
            epochs=epochs,
            anneal_start_epoch=history_evidence_cognitive_alignment_anneal_start_epoch,
            anneal_end_epoch=history_evidence_cognitive_alignment_anneal_end_epoch,
        )
        checkpoint_distillation_epoch_weight = (
            checkpoint_distillation_weight
            if epoch >= checkpoint_distillation_start_epoch
            else 0.0
        )
        prior_attr = "concept_evidence_prior_residual"
        max_logit_attr = "concept_evidence_prior_max_logit"
        concept_prior_train_scale = 0.0
        prior_enabled = bool(getattr(model, prior_attr, False))
        should_defer_concept_prior = epoch < concept_evidence_prior_train_start_epoch and prior_enabled
        if prior_enabled and not should_defer_concept_prior:
            if concept_evidence_prior_train_warmup_epochs > 0:
                warmup_step = epoch - concept_evidence_prior_train_start_epoch + 1
                concept_prior_train_scale = min(
                    1.0,
                    max(0.0, float(warmup_step) / float(concept_evidence_prior_train_warmup_epochs)),
                )
            else:
                concept_prior_train_scale = 1.0
        original_concept_prior_enabled = None
        original_concept_prior_max_logit = None
        if should_defer_concept_prior:
            original_concept_prior_enabled = getattr(model, prior_attr)
            setattr(model, prior_attr, False)
        elif (
            prior_enabled
            and concept_prior_train_scale < 1.0
            and hasattr(model, max_logit_attr)
        ):
            original_concept_prior_max_logit = getattr(model, max_logit_attr)
            setattr(model, max_logit_attr, float(original_concept_prior_max_logit) * concept_prior_train_scale)
        try:
            if training_mode == "full_batch":
                train_stats = _train_full_batch_epoch(
                    model=model,
                    tensors=train_tensors,
                    optimizer=optimizer,
                    ukc_consistency_weight=ukc_consistency_weight,
                    ukc_consistency_drop_frac=ukc_consistency_drop_frac,
                    mastery_aux_bce_weight=mastery_aux_bce_weight,
                    unified_evidence_loss_weight=(
                        unified_evidence_loss_weight
                    ),
                    unified_completion_loss_weight=(
                        unified_completion_loss_weight
                    ),
                    history_dropout_frac=history_dropout_frac,
                    masked_response_weight=masked_response_weight,
                    masked_response_frac=masked_response_frac,
                    exercise_evidence_difficulty_regularization_weight=exercise_evidence_difficulty_regularization_weight,
                    difficulty_prior_target=difficulty_prior_target,
                    difficulty_prior_mask=difficulty_prior_mask,
                    history_evidence_cognitive_alignment_weight=cognitive_alignment_epoch_weight,
                    history_evidence_cognitive_alignment_confidence_power=(
                        history_evidence_cognitive_alignment_confidence_power
                    ),
                    history_evidence_cognitive_alignment_confidence_cap=(
                        history_evidence_cognitive_alignment_confidence_cap
                    ),
                    history_evidence_cognitive_alignment_confidence_floor=(
                        history_evidence_cognitive_alignment_confidence_floor
                    ),
                    history_evidence_cognitive_alignment_residual_power=(
                        history_evidence_cognitive_alignment_residual_power
                    ),
                    history_evidence_cognitive_alignment_residual_floor=(
                        history_evidence_cognitive_alignment_residual_floor
                    ),
                    history_evidence_cognitive_rank_alignment_weight=history_evidence_cognitive_rank_alignment_weight,
                    history_evidence_cognitive_rank_alignment_pair_count=(
                        history_evidence_cognitive_rank_alignment_pair_count
                    ),
                    history_evidence_cognitive_rank_alignment_min_target_gap=(
                        history_evidence_cognitive_rank_alignment_min_target_gap
                    ),
                    history_evidence_output_alignment_weight=history_evidence_output_alignment_weight,
                    history_evidence_output_alignment_confidence_power=(
                        history_evidence_output_alignment_confidence_power
                    ),
                    history_evidence_output_alignment_confidence_cap=(
                        history_evidence_output_alignment_confidence_cap
                    ),
                    history_evidence_output_alignment_confidence_floor=(
                        history_evidence_output_alignment_confidence_floor
                    ),
                    checkpoint_distillation_targets=checkpoint_distillation_targets_tensor,
                    checkpoint_distillation_weight=checkpoint_distillation_epoch_weight,
                    checkpoint_distillation_loss=checkpoint_distillation_loss,
                    dual_tower_branch_bce_weight=dual_tower_branch_bce_weight,
                )
            elif training_mode == "recompute_minibatch":
                train_stats = _train_recompute_minibatch_epoch(
                    model=model,
                    tensors=train_tensors,
                    optimizer=optimizer,
                    batch_size=int(batch_size),
                    device=torch_device,
                    exercise_evidence_difficulty_regularization_weight=exercise_evidence_difficulty_regularization_weight,
                    difficulty_prior_target=difficulty_prior_target,
                    difficulty_prior_mask=difficulty_prior_mask,
                    history_evidence_cognitive_alignment_weight=cognitive_alignment_epoch_weight,
                    history_evidence_cognitive_alignment_confidence_power=(
                        history_evidence_cognitive_alignment_confidence_power
                    ),
                    history_evidence_cognitive_alignment_confidence_cap=(
                        history_evidence_cognitive_alignment_confidence_cap
                    ),
                    history_evidence_cognitive_alignment_confidence_floor=(
                        history_evidence_cognitive_alignment_confidence_floor
                    ),
                    history_evidence_cognitive_alignment_residual_power=(
                        history_evidence_cognitive_alignment_residual_power
                    ),
                    history_evidence_cognitive_alignment_residual_floor=(
                        history_evidence_cognitive_alignment_residual_floor
                    ),
                    history_evidence_cognitive_rank_alignment_weight=history_evidence_cognitive_rank_alignment_weight,
                    history_evidence_cognitive_rank_alignment_pair_count=(
                        history_evidence_cognitive_rank_alignment_pair_count
                    ),
                    history_evidence_cognitive_rank_alignment_min_target_gap=(
                        history_evidence_cognitive_rank_alignment_min_target_gap
                    ),
                    history_evidence_output_alignment_weight=history_evidence_output_alignment_weight,
                    history_evidence_output_alignment_confidence_power=(
                        history_evidence_output_alignment_confidence_power
                    ),
                    history_evidence_output_alignment_confidence_cap=(
                        history_evidence_output_alignment_confidence_cap
                    ),
                    history_evidence_output_alignment_confidence_floor=(
                        history_evidence_output_alignment_confidence_floor
                    ),
                    checkpoint_distillation_targets=checkpoint_distillation_targets_tensor,
                    checkpoint_distillation_weight=checkpoint_distillation_epoch_weight,
                    checkpoint_distillation_loss=checkpoint_distillation_loss,
                    dual_tower_branch_bce_weight=dual_tower_branch_bce_weight,
                )
            else:
                train_stats = _train_student_recompute_minibatch_epoch(
                    model=model,
                    tensors=train_tensors,
                    optimizer=optimizer,
                    student_batch_size=int(student_batch_size),
                    mastery_aux_bce_weight=mastery_aux_bce_weight,
                    unified_evidence_loss_weight=(
                        unified_evidence_loss_weight
                    ),
                    unified_completion_loss_weight=(
                        unified_completion_loss_weight
                    ),
                    contrastive_weight=contrastive_weight,
                    consistency_weight=consistency_weight,
                    consistency_adaptive=consistency_adaptive,
                    curriculum=curriculum,
                    exercise_evidence_difficulty_regularization_weight=exercise_evidence_difficulty_regularization_weight,
                    difficulty_prior_target=difficulty_prior_target,
                    difficulty_prior_mask=difficulty_prior_mask,
                    history_evidence_cognitive_alignment_weight=cognitive_alignment_epoch_weight,
                    history_evidence_cognitive_alignment_confidence_power=(
                        history_evidence_cognitive_alignment_confidence_power
                    ),
                    history_evidence_cognitive_alignment_confidence_cap=(
                        history_evidence_cognitive_alignment_confidence_cap
                    ),
                    history_evidence_cognitive_alignment_confidence_floor=(
                        history_evidence_cognitive_alignment_confidence_floor
                    ),
                    history_evidence_cognitive_alignment_residual_power=(
                        history_evidence_cognitive_alignment_residual_power
                    ),
                    history_evidence_cognitive_alignment_residual_floor=(
                        history_evidence_cognitive_alignment_residual_floor
                    ),
                    history_evidence_cognitive_rank_alignment_weight=history_evidence_cognitive_rank_alignment_weight,
                    history_evidence_cognitive_rank_alignment_pair_count=(
                        history_evidence_cognitive_rank_alignment_pair_count
                    ),
                    history_evidence_cognitive_rank_alignment_min_target_gap=(
                        history_evidence_cognitive_rank_alignment_min_target_gap
                    ),
                    history_evidence_output_alignment_weight=history_evidence_output_alignment_weight,
                    history_evidence_output_alignment_confidence_power=(
                        history_evidence_output_alignment_confidence_power
                    ),
                    history_evidence_output_alignment_confidence_cap=(
                        history_evidence_output_alignment_confidence_cap
                    ),
                    history_evidence_output_alignment_confidence_floor=(
                        history_evidence_output_alignment_confidence_floor
                    ),
                    checkpoint_distillation_targets=checkpoint_distillation_targets_tensor,
                    checkpoint_distillation_weight=checkpoint_distillation_epoch_weight,
                    checkpoint_distillation_loss=checkpoint_distillation_loss,
                    dual_tower_branch_bce_weight=dual_tower_branch_bce_weight,
                )
        finally:
            if original_concept_prior_max_logit is not None:
                setattr(model, max_logit_attr, original_concept_prior_max_logit)
            if original_concept_prior_enabled is not None:
                setattr(model, prior_attr, original_concept_prior_enabled)
        row = {
            "epoch": float(epoch),
            "train_loss": float(train_stats.mean_loss),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "training_mode": training_mode,
            "optimizer_steps": float(train_stats.optimizer_steps),
            "history_evidence_cognitive_alignment_weight": float(cognitive_alignment_epoch_weight),
            "checkpoint_distillation_weight": float(checkpoint_distillation_epoch_weight),
            "concept_evidence_prior_train_scale": float(concept_prior_train_scale),
        }

        if valid_bundle is not None:
            evaluate_kwargs = {
                "bundle": valid_bundle,
                "model": model,
                "device": device,
            }
            if training_mode == "student_recompute_minibatch":
                evaluate_kwargs["student_batch_size"] = student_batch_size
            val_metrics = evaluate_model(**evaluate_kwargs)
            row["val_loss"] = float(val_metrics["loss"])
            row["val_auc"] = float(val_metrics["auc"])
            row["val_acc"] = float(val_metrics["acc"])
            row["val_rmse"] = float(val_metrics["rmse"])
            row["val_brier"] = float(val_metrics["brier"])
            row["val_ece"] = float(val_metrics["ece"])
            scheduler.step(row["val_auc"])

            checkpoint_score = float(row[f"val_{checkpoint_selection_metric}"])
            eligible_for_checkpoint = epoch >= checkpoint_selection_start_epoch
            if eligible_for_checkpoint:
                checkpoint_score_window.append(checkpoint_score)
                if len(checkpoint_score_window) > checkpoint_selection_window:
                    checkpoint_score_window.pop(0)
            smoothed_checkpoint_score = (
                sum(checkpoint_score_window) / len(checkpoint_score_window)
                if eligible_for_checkpoint
                else checkpoint_score
            )
            row[f"val_{checkpoint_selection_metric}_selection_score"] = float(smoothed_checkpoint_score)
            if eligible_for_checkpoint and _is_checkpoint_metric_improved(
                metric=checkpoint_selection_metric,
                value=smoothed_checkpoint_score,
                best_value=best_validation_score,
            ):
                best_validation_score = smoothed_checkpoint_score
                best_val_auc = row["val_auc"]
                best_epoch = epoch
                patience_counter = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if checkpoint_path is not None:
                    torch.save(best_state, checkpoint_path)
            else:
                if eligible_for_checkpoint:
                    patience_counter += 1
                if patience_counter >= early_stop_patience:
                    history.append(row)
                    break
        else:
            if epoch == epochs:
                best_epoch = epoch
                best_val_auc = float("nan")
                best_validation_score = float("nan")

        history.append(row)
        if swa_start_epoch > 0 and epoch >= swa_start_epoch:
            current_state = model.state_dict()
            if swa_state is None:
                swa_state = {
                    key: value.detach().cpu().clone()
                    for key, value in current_state.items()
                }
                swa_epoch_count = 1
            else:
                swa_epoch_count += 1
                for key, value in current_state.items():
                    detached = value.detach().cpu()
                    if torch.is_floating_point(swa_state[key]):
                        swa_state[key].add_((detached - swa_state[key]) / float(swa_epoch_count))
                    else:
                        swa_state[key].copy_(detached)
        if ema_start_epoch > 0 and epoch >= ema_start_epoch:
            current_state = model.state_dict()
            if ema_state is None:
                ema_state = {
                    key: value.detach().cpu().clone()
                    for key, value in current_state.items()
                }
                ema_epoch_count = 1
            else:
                ema_epoch_count += 1
                for key, value in current_state.items():
                    detached = value.detach().cpu()
                    if torch.is_floating_point(ema_state[key]):
                        ema_state[key].mul_(ema_decay).add_(detached, alpha=1.0 - ema_decay)
                    else:
                        ema_state[key].copy_(detached)

    selected_swa_checkpoint_path = None
    selected_ema_checkpoint_path = None
    if swa_state is not None and swa_epoch_count > 0:
        model.load_state_dict(swa_state)
        if swa_checkpoint_path is not None:
            torch.save(swa_state, swa_checkpoint_path)
            selected_swa_checkpoint_path = swa_checkpoint_path
    elif ema_state is not None and ema_epoch_count > 0:
        model.load_state_dict(ema_state)
        if ema_checkpoint_path is not None:
            torch.save(ema_state, ema_checkpoint_path)
            selected_ema_checkpoint_path = ema_checkpoint_path
    elif best_state is not None:
        model.load_state_dict(best_state)

    final_loss = history[-1]["train_loss"] if history else 0.0
    return TrainResult(
        history=history,
        final_loss=final_loss,
        best_val_auc=best_val_auc,
        best_epoch=best_epoch,
        best_checkpoint_path=checkpoint_path if best_state is not None else None,
        checkpoint_selection_metric=checkpoint_selection_metric,
        checkpoint_selection_start_epoch=checkpoint_selection_start_epoch,
        checkpoint_selection_window=checkpoint_selection_window,
        best_validation_score=best_validation_score,
        swa_start_epoch=swa_start_epoch,
        swa_epoch_count=swa_epoch_count,
        swa_checkpoint_path=selected_swa_checkpoint_path,
        ema_start_epoch=ema_start_epoch,
        ema_decay=ema_decay,
        ema_epoch_count=ema_epoch_count,
        ema_checkpoint_path=selected_ema_checkpoint_path,
    )


def _perturb_history_tensors(
    tensors: dict[str, torch.Tensor | None],
    *,
    drop_frac: float | torch.Tensor,
) -> tuple[dict[str, torch.Tensor | None], torch.Tensor]:
    """
    Randomly drop a fraction of observed (student, exercise) history entries and
    re-derive the dependent tensors in tensor space (TKC/UKC masks, per-concept
    evidence counts approximated by distinct-exercise counts). Returns the
    perturbed tensor dict and the boolean keep-matrix over (S, E).

    drop_frac may be a scalar or a per-student (S,) tensor (adaptive dropping,
    e.g. scaled by coverage so sparse students are barely touched).
    """
    mask = tensors["student_exercise_mask"]
    threshold = drop_frac.unsqueeze(-1) if torch.is_tensor(drop_frac) else drop_frac
    keep = torch.rand_like(mask) >= threshold
    dropped_mask = mask * keep.to(mask.dtype)
    q_binary = (tensors["q_matrix"] > 0).to(mask.dtype)
    attempts = dropped_mask @ q_binary
    correct = (dropped_mask * tensors["response_matrix"]) @ q_binary
    tkc_mask = (attempts > 0).to(mask.dtype)
    ukc_mask = (1.0 - tkc_mask).clamp(min=0.0, max=1.0)
    perturbed = dict(tensors)
    perturbed["student_exercise_mask"] = dropped_mask
    perturbed["student_tkc_mask"] = tkc_mask
    perturbed["student_ukc_mask"] = ukc_mask
    if tensors["student_concept_evidence"] is not None:
        perturbed["student_concept_evidence"] = torch.stack([attempts, correct], dim=-1)
    return perturbed, keep


def _masked_response_loss(
    *,
    model: DecoupledCDM,
    tensors: dict[str, torch.Tensor | None],
    mask_frac: float,
) -> torch.Tensor:
    """
    Masked-response self-supervision: hide a random fraction of observed
    history entries, then predict the labels of the training interactions
    that fall on the hidden entries using the masked history. Blocks the
    transductive look-up shortcut and directly trains inference from
    incomplete evidence. Supervision stays on observed (tested) responses.
    """
    perturbed, keep = _perturb_history_tensors(tensors, drop_frac=mask_frac)
    student_ids = tensors["interaction_student_ids"]
    exercise_ids = tensors["interaction_exercise_ids"]
    hidden_rows = ~keep[student_ids, exercise_ids]
    if int(hidden_rows.sum().item()) == 0:
        return tensors["interaction_labels"].new_zeros(())
    output = model(
        q_matrix=perturbed["q_matrix"],
        concept_graph=perturbed["concept_graph"],
        prerequisite_graph=perturbed["prerequisite_graph"],
        similarity_graph=perturbed["similarity_graph"],
        student_exercise_mask=perturbed["student_exercise_mask"],
        response_matrix=perturbed["response_matrix"],
        student_tkc_mask=perturbed["student_tkc_mask"],
        student_ukc_mask=perturbed["student_ukc_mask"],
        student_concept_evidence=perturbed["student_concept_evidence"],
        exercise_evidence=perturbed["exercise_evidence"],
        target_student_ids=student_ids[hidden_rows],
        target_exercise_ids=exercise_ids[hidden_rows],
    )
    return F.binary_cross_entropy(output.probs, tensors["interaction_labels"][hidden_rows])


def _train_full_batch_epoch(
    *,
    model: DecoupledCDM,
    tensors: dict[str, torch.Tensor | None],
    optimizer: torch.optim.Optimizer,
    exercise_evidence_difficulty_regularization_weight: float = 0.0,
    difficulty_prior_target: torch.Tensor | None = None,
    difficulty_prior_mask: torch.Tensor | None = None,
    history_evidence_cognitive_alignment_weight: float = 0.0,
    history_evidence_cognitive_alignment_confidence_power: float = 0.0,
    history_evidence_cognitive_alignment_confidence_cap: float = 20.0,
    history_evidence_cognitive_alignment_confidence_floor: float = 0.0,
    history_evidence_cognitive_alignment_residual_power: float = 0.0,
    history_evidence_cognitive_alignment_residual_floor: float = 0.0,
    history_evidence_cognitive_rank_alignment_weight: float = 0.0,
    history_evidence_cognitive_rank_alignment_pair_count: int = 8192,
    history_evidence_cognitive_rank_alignment_min_target_gap: float = 0.0,
    history_evidence_output_alignment_weight: float = 0.0,
    history_evidence_output_alignment_confidence_power: float = 0.0,
    history_evidence_output_alignment_confidence_cap: float = 20.0,
    history_evidence_output_alignment_confidence_floor: float = 0.0,
    checkpoint_distillation_targets: torch.Tensor | None = None,
    checkpoint_distillation_weight: float = 0.0,
    checkpoint_distillation_loss: str = "bce",
    dual_tower_branch_bce_weight: float = 0.0,
    ukc_consistency_weight: float = 0.0,
    ukc_consistency_drop_frac: float = 0.2,
    mastery_aux_bce_weight: float = 0.0,
    unified_evidence_loss_weight: float = 0.0,
    unified_completion_loss_weight: float = 0.0,
    history_dropout_frac: float = 0.0,
    masked_response_weight: float = 0.0,
    masked_response_frac: float = 0.15,
) -> EpochTrainStats:
    model.train()
    optimizer.zero_grad()

    forward_tensors = tensors
    if history_dropout_frac > 0.0:
        forward_tensors, _ = _perturb_history_tensors(tensors, drop_frac=history_dropout_frac)

    output = _forward_model(
        model=model,
        q_matrix=forward_tensors["q_matrix"],
        concept_graph=forward_tensors["concept_graph"],
        prerequisite_graph=forward_tensors["prerequisite_graph"],
        similarity_graph=forward_tensors["similarity_graph"],
        student_exercise_mask=forward_tensors["student_exercise_mask"],
        response_matrix=forward_tensors["response_matrix"],
        student_tkc_mask=forward_tensors["student_tkc_mask"],
        student_ukc_mask=forward_tensors["student_ukc_mask"],
        student_concept_evidence=forward_tensors["student_concept_evidence"],
        exercise_evidence=forward_tensors["exercise_evidence"],
        target_student_ids=forward_tensors["interaction_student_ids"],
        target_exercise_ids=forward_tensors["interaction_exercise_ids"],
    )
    loss = F.binary_cross_entropy(output.probs, tensors["interaction_labels"])
    if mastery_aux_bce_weight > 0.0:
        if getattr(output, "mastery_aux_logits", None) is None:
            raise ValueError("mastery_aux_bce_weight requires a model that emits mastery_aux_logits.")
        loss = loss + mastery_aux_bce_weight * F.binary_cross_entropy_with_logits(
            output.mastery_aux_logits, tensors["interaction_labels"]
        )
    if unified_evidence_loss_weight > 0.0:
        loss = (
            loss
            + unified_evidence_loss_weight
            * observed_mastery_evidence_loss(
                output,
                tensors["student_concept_evidence"][..., :2],
            )
        )
    if unified_completion_loss_weight > 0.0:
        loss = (
            loss
            + unified_completion_loss_weight
            * masked_completion_loss(
                output,
                tensors["student_concept_evidence"],
            )
        )
    if masked_response_weight > 0.0:
        loss = loss + masked_response_weight * _masked_response_loss(
            model=model,
            tensors=tensors,
            mask_frac=masked_response_frac,
        )
    if ukc_consistency_weight > 0.0:
        loss = loss + ukc_consistency_weight * model.compute_ukc_consistency_loss(
            q_matrix=tensors["q_matrix"],
            concept_graph=tensors["concept_graph"],
            student_exercise_mask=tensors["student_exercise_mask"],
            response_matrix=tensors["response_matrix"],
            student_tkc_mask=tensors["student_tkc_mask"],
            student_concept_evidence=tensors["student_concept_evidence"],
            drop_frac=ukc_consistency_drop_frac,
        )
    if dual_tower_branch_bce_weight > 0.0:
        loss = loss + _dual_tower_branch_bce_loss(
            output=output,
            labels=tensors["interaction_labels"],
            weight=dual_tower_branch_bce_weight,
        )
    if history_evidence_cognitive_alignment_weight > 0.0:
        loss = loss + _history_evidence_cognitive_alignment_loss(
            model=model,
            output=output,
            tensors=tensors,
            target_student_ids=tensors["interaction_student_ids"],
            target_exercise_ids=tensors["interaction_exercise_ids"],
            weight=history_evidence_cognitive_alignment_weight,
            confidence_power=history_evidence_cognitive_alignment_confidence_power,
            confidence_cap=history_evidence_cognitive_alignment_confidence_cap,
            confidence_floor=history_evidence_cognitive_alignment_confidence_floor,
            residual_power=history_evidence_cognitive_alignment_residual_power,
            residual_floor=history_evidence_cognitive_alignment_residual_floor,
        )
    if history_evidence_cognitive_rank_alignment_weight > 0.0:
        loss = loss + _history_evidence_cognitive_rank_alignment_loss(
            model=model,
            output=output,
            tensors=tensors,
            target_student_ids=tensors["interaction_student_ids"],
            target_exercise_ids=tensors["interaction_exercise_ids"],
            weight=history_evidence_cognitive_rank_alignment_weight,
            pair_count=history_evidence_cognitive_rank_alignment_pair_count,
            min_target_gap=history_evidence_cognitive_rank_alignment_min_target_gap,
        )
    if history_evidence_output_alignment_weight > 0.0:
        loss = loss + _history_evidence_output_alignment_loss(
            model=model,
            output=output,
            tensors=tensors,
            target_student_ids=tensors["interaction_student_ids"],
            target_exercise_ids=tensors["interaction_exercise_ids"],
            weight=history_evidence_output_alignment_weight,
            confidence_power=history_evidence_output_alignment_confidence_power,
            confidence_cap=history_evidence_output_alignment_confidence_cap,
            confidence_floor=history_evidence_output_alignment_confidence_floor,
        )
    if checkpoint_distillation_weight > 0.0:
        loss = loss + _checkpoint_distillation_loss(
            output=output,
            target_probs=checkpoint_distillation_targets,
            weight=checkpoint_distillation_weight,
            loss_type=checkpoint_distillation_loss,
        )
    if exercise_evidence_difficulty_regularization_weight > 0.0:
        loss = loss + _exercise_difficulty_regularization_loss(
            model=model,
            weight=exercise_evidence_difficulty_regularization_weight,
            difficulty_prior_target=difficulty_prior_target,
            difficulty_prior_mask=difficulty_prior_mask,
        )
    loss.backward()
    optimizer.step()
    return EpochTrainStats(mean_loss=float(loss.item()), optimizer_steps=1)


def _train_recompute_minibatch_epoch(
    *,
    model: DecoupledCDM,
    tensors: dict[str, torch.Tensor | None],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
    exercise_evidence_difficulty_regularization_weight: float = 0.0,
    difficulty_prior_target: torch.Tensor | None = None,
    difficulty_prior_mask: torch.Tensor | None = None,
    history_evidence_cognitive_alignment_weight: float = 0.0,
    history_evidence_cognitive_alignment_confidence_power: float = 0.0,
    history_evidence_cognitive_alignment_confidence_cap: float = 20.0,
    history_evidence_cognitive_alignment_confidence_floor: float = 0.0,
    history_evidence_cognitive_alignment_residual_power: float = 0.0,
    history_evidence_cognitive_alignment_residual_floor: float = 0.0,
    history_evidence_cognitive_rank_alignment_weight: float = 0.0,
    history_evidence_cognitive_rank_alignment_pair_count: int = 8192,
    history_evidence_cognitive_rank_alignment_min_target_gap: float = 0.0,
    history_evidence_output_alignment_weight: float = 0.0,
    history_evidence_output_alignment_confidence_power: float = 0.0,
    history_evidence_output_alignment_confidence_cap: float = 20.0,
    history_evidence_output_alignment_confidence_floor: float = 0.0,
    checkpoint_distillation_targets: torch.Tensor | None = None,
    checkpoint_distillation_weight: float = 0.0,
    checkpoint_distillation_loss: str = "bce",
    dual_tower_branch_bce_weight: float = 0.0,
) -> EpochTrainStats:
    model.train()
    num_targets = int(tensors["interaction_labels"].size(0))
    permutation = torch.randperm(num_targets, device=device)
    total_loss = 0.0
    optimizer_steps = 0

    for start in range(0, num_targets, batch_size):
        batch_indices = permutation[start : start + batch_size]
        batch_labels = tensors["interaction_labels"][batch_indices]
        optimizer.zero_grad()
        output = model(
            q_matrix=tensors["q_matrix"],
            concept_graph=tensors["concept_graph"],
            prerequisite_graph=tensors["prerequisite_graph"],
            similarity_graph=tensors["similarity_graph"],
            student_exercise_mask=tensors["student_exercise_mask"],
            response_matrix=tensors["response_matrix"],
            student_tkc_mask=tensors["student_tkc_mask"],
            student_ukc_mask=tensors["student_ukc_mask"],
            student_concept_evidence=tensors["student_concept_evidence"],
            exercise_evidence=tensors["exercise_evidence"],
            target_student_ids=tensors["interaction_student_ids"][batch_indices],
            target_exercise_ids=tensors["interaction_exercise_ids"][batch_indices],
            use_student_subset=True,
        )
        loss = F.binary_cross_entropy(output.probs, batch_labels)
        if dual_tower_branch_bce_weight > 0.0:
            loss = loss + _dual_tower_branch_bce_loss(
                output=output,
                labels=batch_labels,
                weight=dual_tower_branch_bce_weight,
            )
        if history_evidence_cognitive_alignment_weight > 0.0:
            loss = loss + _history_evidence_cognitive_alignment_loss(
                model=model,
                output=output,
                tensors=tensors,
                target_student_ids=tensors["interaction_student_ids"][batch_indices],
                target_exercise_ids=tensors["interaction_exercise_ids"][batch_indices],
                weight=history_evidence_cognitive_alignment_weight,
                confidence_power=history_evidence_cognitive_alignment_confidence_power,
                confidence_cap=history_evidence_cognitive_alignment_confidence_cap,
                confidence_floor=history_evidence_cognitive_alignment_confidence_floor,
                residual_power=history_evidence_cognitive_alignment_residual_power,
                residual_floor=history_evidence_cognitive_alignment_residual_floor,
            )
        if history_evidence_cognitive_rank_alignment_weight > 0.0:
            loss = loss + _history_evidence_cognitive_rank_alignment_loss(
                model=model,
                output=output,
                tensors=tensors,
                target_student_ids=tensors["interaction_student_ids"][batch_indices],
                target_exercise_ids=tensors["interaction_exercise_ids"][batch_indices],
                weight=history_evidence_cognitive_rank_alignment_weight,
                pair_count=history_evidence_cognitive_rank_alignment_pair_count,
                min_target_gap=history_evidence_cognitive_rank_alignment_min_target_gap,
            )
        if history_evidence_output_alignment_weight > 0.0:
            loss = loss + _history_evidence_output_alignment_loss(
                model=model,
                output=output,
                tensors=tensors,
                target_student_ids=tensors["interaction_student_ids"][batch_indices],
                target_exercise_ids=tensors["interaction_exercise_ids"][batch_indices],
                weight=history_evidence_output_alignment_weight,
                confidence_power=history_evidence_output_alignment_confidence_power,
                confidence_cap=history_evidence_output_alignment_confidence_cap,
                confidence_floor=history_evidence_output_alignment_confidence_floor,
            )
        if checkpoint_distillation_weight > 0.0:
            loss = loss + _checkpoint_distillation_loss(
                output=output,
                target_probs=checkpoint_distillation_targets[batch_indices],
                weight=checkpoint_distillation_weight,
                loss_type=checkpoint_distillation_loss,
            )
        if exercise_evidence_difficulty_regularization_weight > 0.0:
            loss = loss + _exercise_difficulty_regularization_loss(
                model=model,
                weight=exercise_evidence_difficulty_regularization_weight,
                difficulty_prior_target=difficulty_prior_target,
                difficulty_prior_mask=difficulty_prior_mask,
            )
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * float(batch_labels.numel())
        optimizer_steps += 1

    return EpochTrainStats(mean_loss=total_loss / float(num_targets), optimizer_steps=optimizer_steps)


def _train_student_recompute_minibatch_epoch(
    *,
    model: DecoupledCDM,
    tensors: dict[str, torch.Tensor | None],
    optimizer: torch.optim.Optimizer,
    student_batch_size: int,
    exercise_evidence_difficulty_regularization_weight: float = 0.0,
    difficulty_prior_target: torch.Tensor | None = None,
    difficulty_prior_mask: torch.Tensor | None = None,
    history_evidence_cognitive_alignment_weight: float = 0.0,
    history_evidence_cognitive_alignment_confidence_power: float = 0.0,
    history_evidence_cognitive_alignment_confidence_cap: float = 20.0,
    history_evidence_cognitive_alignment_confidence_floor: float = 0.0,
    history_evidence_cognitive_alignment_residual_power: float = 0.0,
    history_evidence_cognitive_alignment_residual_floor: float = 0.0,
    history_evidence_cognitive_rank_alignment_weight: float = 0.0,
    history_evidence_cognitive_rank_alignment_pair_count: int = 8192,
    history_evidence_cognitive_rank_alignment_min_target_gap: float = 0.0,
    history_evidence_output_alignment_weight: float = 0.0,
    history_evidence_output_alignment_confidence_power: float = 0.0,
    history_evidence_output_alignment_confidence_cap: float = 20.0,
    history_evidence_output_alignment_confidence_floor: float = 0.0,
    checkpoint_distillation_targets: torch.Tensor | None = None,
    checkpoint_distillation_weight: float = 0.0,
    checkpoint_distillation_loss: str = "bce",
    dual_tower_branch_bce_weight: float = 0.0,
    mastery_aux_bce_weight: float = 0.0,
    unified_evidence_loss_weight: float = 0.0,
    unified_completion_loss_weight: float = 0.0,
    contrastive_weight: float = 0.0,
    consistency_weight: float = 0.0,
    consistency_adaptive: bool = False,
    curriculum: bool = False,
) -> EpochTrainStats:
    model.train()
    num_targets = int(tensors["interaction_labels"].size(0))
    unique_student_ids = torch.unique(tensors["interaction_student_ids"], sorted=False)
    if curriculum:
        # Tr-3: order students by evidence density (attempts), easy (dense) first.
        evidence = tensors["student_concept_evidence"]
        density = (
            evidence[..., 0].sum(dim=1)
            if evidence is not None
            else tensors["student_exercise_mask"].sum(dim=1)
        )
        student_permutation = unique_student_ids[
            torch.argsort(density[unique_student_ids], descending=True)
        ]
    else:
        student_permutation = unique_student_ids[
            torch.randperm(unique_student_ids.numel(), device=unique_student_ids.device)
        ]
    total_loss = 0.0
    optimizer_steps = 0

    for start in range(0, student_permutation.numel(), student_batch_size):
        student_ids = student_permutation[start : start + student_batch_size]
        batch_indices = _interaction_indices_for_student_ids(
            interaction_student_ids=tensors["interaction_student_ids"],
            student_ids=student_ids,
            num_students=tensors["student_exercise_mask"].size(0),
        )
        if batch_indices.numel() == 0:
            continue
        batch_labels = tensors["interaction_labels"][batch_indices]
        optimizer.zero_grad()
        output = _forward_model(
            model=model,
            q_matrix=tensors["q_matrix"],
            concept_graph=tensors["concept_graph"],
            prerequisite_graph=tensors["prerequisite_graph"],
            similarity_graph=tensors["similarity_graph"],
            student_exercise_mask=tensors["student_exercise_mask"],
            response_matrix=tensors["response_matrix"],
            student_tkc_mask=tensors["student_tkc_mask"],
            student_ukc_mask=tensors["student_ukc_mask"],
            student_concept_evidence=tensors["student_concept_evidence"],
            exercise_evidence=tensors["exercise_evidence"],
            target_student_ids=tensors["interaction_student_ids"][batch_indices],
            target_exercise_ids=tensors["interaction_exercise_ids"][batch_indices],
            use_student_subset=True,
        )
        loss = F.binary_cross_entropy(output.probs, batch_labels)
        if mastery_aux_bce_weight > 0.0:
            if getattr(output, "mastery_aux_logits", None) is None:
                raise ValueError("mastery_aux_bce_weight requires a model that emits mastery_aux_logits.")
            loss = loss + mastery_aux_bce_weight * F.binary_cross_entropy_with_logits(
                output.mastery_aux_logits, batch_labels
            )
        if unified_evidence_loss_weight > 0.0:
            mastery_student_ids = torch.unique(
                tensors["interaction_student_ids"][batch_indices],
                sorted=True,
            )
            loss = (
                loss
                + unified_evidence_loss_weight
                * observed_mastery_evidence_loss(
                    output,
                    tensors["student_concept_evidence"][..., :2],
                    student_ids=mastery_student_ids,
                )
            )
        if unified_completion_loss_weight > 0.0:
            completion_student_ids = torch.unique(
                tensors["interaction_student_ids"][batch_indices],
                sorted=True,
            )
            loss = (
                loss
                + unified_completion_loss_weight
                * masked_completion_loss(
                    output,
                    tensors["student_concept_evidence"],
                    student_ids=completion_student_ids,
                )
            )
        if consistency_weight > 0.0 or contrastive_weight > 0.0:
            if consistency_adaptive:
                # Tr-2-adaptive: per-student drop scaled by coverage so
                # high-coverage students are perturbed hard and sparse students
                # (whose evidence we cannot afford to erase) are barely touched.
                coverage = tensors["student_tkc_mask"].mean(dim=1)
                drop = 0.6 * coverage / coverage.clamp_min(1e-6).max()
                perturbed, _ = _perturb_history_tensors(tensors, drop_frac=drop)
            else:
                perturbed, _ = _perturb_history_tensors(tensors, drop_frac=0.3)
            view2 = model(
                q_matrix=perturbed["q_matrix"],
                concept_graph=perturbed["concept_graph"],
                prerequisite_graph=perturbed["prerequisite_graph"],
                similarity_graph=perturbed["similarity_graph"],
                student_exercise_mask=perturbed["student_exercise_mask"],
                response_matrix=perturbed["response_matrix"],
                student_tkc_mask=perturbed["student_tkc_mask"],
                student_ukc_mask=perturbed["student_ukc_mask"],
                student_concept_evidence=perturbed["student_concept_evidence"],
                exercise_evidence=perturbed["exercise_evidence"],
                target_student_ids=tensors["interaction_student_ids"][batch_indices],
                target_exercise_ids=tensors["interaction_exercise_ids"][batch_indices],
                use_student_subset=True,
            )
            if consistency_weight > 0.0:
                # Tr-2: predictions from full vs masked history should agree.
                loss = loss + consistency_weight * F.mse_loss(view2.probs, output.probs.detach())
            if contrastive_weight > 0.0:
                # Tr-1: InfoNCE pulling the two views of the same student together.
                # With use_student_subset, student_state is already the batch's
                # unique-student subset, aligned across both views.
                z1 = F.normalize(output.student_state, dim=-1)
                z2 = F.normalize(view2.student_state, dim=-1)
                logits = z1 @ z2.t() / 0.2
                targets = torch.arange(z1.size(0), device=z1.device)
                loss = loss + contrastive_weight * F.cross_entropy(logits, targets)
        if dual_tower_branch_bce_weight > 0.0:
            loss = loss + _dual_tower_branch_bce_loss(
                output=output,
                labels=batch_labels,
                weight=dual_tower_branch_bce_weight,
            )
        if history_evidence_cognitive_alignment_weight > 0.0:
            loss = loss + _history_evidence_cognitive_alignment_loss(
                model=model,
                output=output,
                tensors=tensors,
                target_student_ids=tensors["interaction_student_ids"][batch_indices],
                target_exercise_ids=tensors["interaction_exercise_ids"][batch_indices],
                weight=history_evidence_cognitive_alignment_weight,
                confidence_power=history_evidence_cognitive_alignment_confidence_power,
                confidence_cap=history_evidence_cognitive_alignment_confidence_cap,
                confidence_floor=history_evidence_cognitive_alignment_confidence_floor,
                residual_power=history_evidence_cognitive_alignment_residual_power,
                residual_floor=history_evidence_cognitive_alignment_residual_floor,
            )
        if history_evidence_cognitive_rank_alignment_weight > 0.0:
            loss = loss + _history_evidence_cognitive_rank_alignment_loss(
                model=model,
                output=output,
                tensors=tensors,
                target_student_ids=tensors["interaction_student_ids"][batch_indices],
                target_exercise_ids=tensors["interaction_exercise_ids"][batch_indices],
                weight=history_evidence_cognitive_rank_alignment_weight,
                pair_count=history_evidence_cognitive_rank_alignment_pair_count,
                min_target_gap=history_evidence_cognitive_rank_alignment_min_target_gap,
            )
        if history_evidence_output_alignment_weight > 0.0:
            loss = loss + _history_evidence_output_alignment_loss(
                model=model,
                output=output,
                tensors=tensors,
                target_student_ids=tensors["interaction_student_ids"][batch_indices],
                target_exercise_ids=tensors["interaction_exercise_ids"][batch_indices],
                weight=history_evidence_output_alignment_weight,
                confidence_power=history_evidence_output_alignment_confidence_power,
                confidence_cap=history_evidence_output_alignment_confidence_cap,
                confidence_floor=history_evidence_output_alignment_confidence_floor,
            )
        if checkpoint_distillation_weight > 0.0:
            loss = loss + _checkpoint_distillation_loss(
                output=output,
                target_probs=checkpoint_distillation_targets[batch_indices],
                weight=checkpoint_distillation_weight,
                loss_type=checkpoint_distillation_loss,
            )
        if exercise_evidence_difficulty_regularization_weight > 0.0:
            loss = loss + _exercise_difficulty_regularization_loss(
                model=model,
                weight=exercise_evidence_difficulty_regularization_weight,
                difficulty_prior_target=difficulty_prior_target,
                difficulty_prior_mask=difficulty_prior_mask,
            )
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * float(batch_labels.numel())
        optimizer_steps += 1

    return EpochTrainStats(mean_loss=total_loss / float(num_targets), optimizer_steps=optimizer_steps)


def _build_history_evidence_cognitive_alignment_prior(
    *,
    model: DecoupledCDM,
    tensors: dict[str, torch.Tensor | None],
    target_student_ids: torch.Tensor,
    target_exercise_ids: torch.Tensor,
) -> torch.Tensor:
    if not getattr(model, "history_evidence_logit_prior_residual", False):
        raise ValueError(
            "history_evidence_logit_prior_residual must be enabled when cognitive alignment loss is enabled."
        )
    if getattr(model, "history_evidence_logit_prior_location", None) != "loss_only":
        raise ValueError(
            "history_evidence_logit_prior_location must be loss_only when cognitive alignment loss is enabled."
        )
    q_matrix = tensors["q_matrix"]
    if q_matrix is None:
        raise ValueError("q_matrix is required when cognitive alignment loss is enabled.")
    q_vectors = q_matrix[target_exercise_ids]
    concept_counts = q_vectors.sum(dim=1, keepdim=True)
    prior = model._build_history_evidence_logit_prior_residual(
        q_vectors=q_vectors,
        target_student_ids=target_student_ids,
        target_exercise_ids=target_exercise_ids,
        student_concept_evidence=tensors["student_concept_evidence"],
        exercise_evidence=tensors["exercise_evidence"],
        student_exercise_mask=tensors["student_exercise_mask"],
        response_matrix=tensors["response_matrix"],
        concept_summary=(
            concept_counts,
            q_vectors.new_zeros(q_vectors.size(0), 1),
            q_vectors.new_zeros(q_vectors.size(0), 1),
        ),
    ).detach()
    return prior


def _history_evidence_cognitive_alignment_loss(
    *,
    model: DecoupledCDM,
    output: object,
    tensors: dict[str, torch.Tensor | None],
    target_student_ids: torch.Tensor,
    target_exercise_ids: torch.Tensor,
    weight: float,
    confidence_power: float = 0.0,
    confidence_cap: float = 20.0,
    confidence_floor: float = 0.0,
    residual_power: float = 0.0,
    residual_floor: float = 0.0,
) -> torch.Tensor:
    if weight <= 0.0:
        return target_student_ids.new_zeros((), dtype=torch.float32)
    if not hasattr(output, "cognitive_probs"):
        raise ValueError("model output must expose cognitive_probs when cognitive alignment loss is enabled.")

    prior = _build_history_evidence_cognitive_alignment_prior(
        model=model,
        tensors=tensors,
        target_student_ids=target_student_ids,
        target_exercise_ids=target_exercise_ids,
    )
    cognitive_logits = torch.logit(output.cognitive_probs.clamp(min=1e-6, max=1.0 - 1e-6))
    q_matrix = tensors["q_matrix"]
    if q_matrix is None:
        raise ValueError("q_matrix is required when cognitive alignment loss is enabled.")
    reliability_weights = _history_evidence_alignment_reliability_weights(
        q_vectors=q_matrix[target_exercise_ids],
        target_student_ids=target_student_ids,
        student_concept_evidence=tensors["student_concept_evidence"],
        confidence_power=confidence_power,
        confidence_cap=confidence_cap,
        confidence_floor=confidence_floor,
    )
    return _standardized_mse_alignment_loss(
        cognitive_logits,
        prior,
        sample_weights=reliability_weights,
        residual_focus_power=residual_power,
        residual_focus_floor=residual_floor,
    ) * float(weight)


def _history_evidence_alignment_reliability_weights(
    *,
    q_vectors: torch.Tensor,
    target_student_ids: torch.Tensor,
    student_concept_evidence: torch.Tensor | None,
    confidence_power: float,
    confidence_cap: float,
    confidence_floor: float,
) -> torch.Tensor | None:
    if confidence_power <= 0.0:
        return None
    if student_concept_evidence is None:
        raise ValueError(
            "student_concept_evidence is required when history evidence alignment confidence weighting is enabled."
        )
    q_mask = (q_vectors > 0).to(dtype=q_vectors.dtype)
    concept_counts = q_mask.sum(dim=1).clamp_min(1.0)
    target_concept_attempts = student_concept_evidence[target_student_ids, :, 0].to(dtype=q_vectors.dtype) * q_mask
    confidence = (target_concept_attempts.clamp_max(confidence_cap) / float(confidence_cap)) * q_mask
    mean_confidence = confidence.sum(dim=1) / concept_counts
    scaled_confidence = mean_confidence.clamp(0.0, 1.0).pow(float(confidence_power))
    return float(confidence_floor) + (1.0 - float(confidence_floor)) * scaled_confidence


def _history_evidence_cognitive_rank_alignment_loss(
    *,
    model: DecoupledCDM,
    output: object,
    tensors: dict[str, torch.Tensor | None],
    target_student_ids: torch.Tensor,
    target_exercise_ids: torch.Tensor,
    weight: float,
    pair_count: int,
    min_target_gap: float,
) -> torch.Tensor:
    if weight <= 0.0:
        return target_student_ids.new_zeros((), dtype=torch.float32)
    if not hasattr(output, "cognitive_probs"):
        raise ValueError("model output must expose cognitive_probs when cognitive rank alignment loss is enabled.")

    prior = _build_history_evidence_cognitive_alignment_prior(
        model=model,
        tensors=tensors,
        target_student_ids=target_student_ids,
        target_exercise_ids=target_exercise_ids,
    )
    cognitive_logits = torch.logit(output.cognitive_probs.clamp(min=1e-6, max=1.0 - 1e-6))
    return _standardized_pairwise_rank_alignment_loss(
        prediction=cognitive_logits,
        target=prior,
        pair_count=pair_count,
        min_target_gap=min_target_gap,
    ) * float(weight)


def _history_evidence_output_alignment_loss(
    *,
    model: DecoupledCDM,
    output: object,
    tensors: dict[str, torch.Tensor | None],
    target_student_ids: torch.Tensor,
    target_exercise_ids: torch.Tensor,
    weight: float,
    confidence_power: float = 0.0,
    confidence_cap: float = 20.0,
    confidence_floor: float = 0.0,
) -> torch.Tensor:
    if weight <= 0.0:
        return target_student_ids.new_zeros((), dtype=torch.float32)
    if not hasattr(output, "probs"):
        raise ValueError("model output must expose probs when output alignment loss is enabled.")

    prior = _build_history_evidence_cognitive_alignment_prior(
        model=model,
        tensors=tensors,
        target_student_ids=target_student_ids,
        target_exercise_ids=target_exercise_ids,
    )
    output_logits = torch.logit(output.probs.clamp(min=1e-6, max=1.0 - 1e-6))
    q_matrix = tensors["q_matrix"]
    if q_matrix is None:
        raise ValueError("q_matrix is required when output alignment loss is enabled.")
    reliability_weights = _history_evidence_alignment_reliability_weights(
        q_vectors=q_matrix[target_exercise_ids],
        target_student_ids=target_student_ids,
        student_concept_evidence=tensors["student_concept_evidence"],
        confidence_power=confidence_power,
        confidence_cap=confidence_cap,
        confidence_floor=confidence_floor,
    )
    return _standardized_mse_alignment_loss(output_logits, prior, sample_weights=reliability_weights) * float(weight)


def _checkpoint_distillation_loss(
    *,
    output: object,
    target_probs: torch.Tensor | None,
    weight: float,
    loss_type: str = "bce",
) -> torch.Tensor:
    if weight <= 0.0:
        if hasattr(output, "probs"):
            return output.probs.new_zeros(())
        return torch.tensor(0.0)
    if target_probs is None:
        raise ValueError("checkpoint_distillation_targets are required when checkpoint distillation is enabled.")
    if not hasattr(output, "probs"):
        raise ValueError("model output must expose probs when checkpoint distillation is enabled.")
    target_probs = target_probs.to(output.probs.device)
    if loss_type == "bce":
        return F.binary_cross_entropy(output.probs, target_probs) * float(weight)
    if loss_type == "standardized_logit_mse":
        output_logits = torch.logit(output.probs.clamp(1e-6, 1.0 - 1e-6))
        target_logits = torch.logit(target_probs.clamp(1e-6, 1.0 - 1e-6))
        return _standardized_mse_alignment_loss(output_logits, target_logits) * float(weight)
    raise ValueError("checkpoint_distillation_loss must be one of bce, standardized_logit_mse.")


def _dual_tower_branch_bce_loss(
    *,
    output: object,
    labels: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    if weight <= 0.0:
        return labels.new_zeros(())
    primary_probs = getattr(output, "primary_probs", None)
    secondary_probs = getattr(output, "secondary_probs", None)
    if primary_probs is None or secondary_probs is None:
        raise ValueError("dual tower branch BCE requires output.primary_probs and output.secondary_probs.")
    primary_loss = F.binary_cross_entropy(primary_probs, labels)
    secondary_loss = F.binary_cross_entropy(secondary_probs, labels)
    return (primary_loss + secondary_loss) * (0.5 * float(weight))


def _standardized_mse_alignment_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    min_std: float = 1e-6,
    residual_focus_power: float = 0.0,
    residual_focus_floor: float = 0.0,
) -> torch.Tensor:
    if target.numel() < 2:
        return prediction.new_zeros(())
    if sample_weights is None:
        prediction_mean = prediction.mean()
        target_mean = target.mean()
        prediction_var = (prediction - prediction_mean).square().mean()
        target_var = (target - target_mean).square().mean()
        normalized_weights = None
    else:
        if sample_weights.shape != target.shape:
            raise ValueError("sample_weights must match target shape for cognitive alignment loss.")
        clipped_weights = sample_weights.to(dtype=target.dtype, device=target.device).clamp_min(0.0)
        weight_sum = clipped_weights.sum()
        if float(weight_sum.detach().cpu().item()) <= min_std:
            return prediction.new_zeros(())
        normalized_weights = clipped_weights / weight_sum
        prediction_mean = (prediction * normalized_weights).sum()
        target_mean = (target * normalized_weights).sum()
        prediction_var = ((prediction - prediction_mean).square() * normalized_weights).sum()
        target_var = ((target - target_mean).square() * normalized_weights).sum()
    target_std = target_var.sqrt()
    if float(target_std.detach().cpu().item()) <= min_std:
        return prediction.new_zeros(())
    prediction_std = prediction_var.sqrt().clamp_min(min_std)
    prediction_z = (prediction - prediction_mean) / prediction_std
    target_z = (target - target_mean) / target_std.clamp_min(min_std)
    if residual_focus_power > 0.0:
        residual_magnitude = (prediction_z.detach() - target_z.detach()).abs()
        scale = residual_magnitude.max().clamp_min(min_std)
        residual_weights = (residual_magnitude / scale).clamp(0.0, 1.0).pow(float(residual_focus_power))
        residual_weights = float(residual_focus_floor) + (1.0 - float(residual_focus_floor)) * residual_weights
        if normalized_weights is None:
            normalized_weights = residual_weights / residual_weights.sum().clamp_min(min_std)
        else:
            combined_weights = normalized_weights * residual_weights
            normalized_weights = combined_weights / combined_weights.sum().clamp_min(min_std)
    residual = (prediction_z - target_z).square()
    if normalized_weights is None:
        return residual.mean()
    return (residual * normalized_weights).sum()


def _standardized_pairwise_rank_alignment_loss(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    pair_count: int,
    min_target_gap: float = 0.0,
    min_std: float = 1e-6,
) -> torch.Tensor:
    if target.numel() < 2:
        return prediction.new_zeros(())
    target_std = target.std(unbiased=False)
    if float(target_std.detach().cpu().item()) <= min_std:
        return prediction.new_zeros(())
    prediction_std = prediction.std(unbiased=False).clamp_min(min_std)
    prediction_z = (prediction - prediction.mean()) / prediction_std
    target_z = (target - target.mean()) / target_std.clamp_min(min_std)

    item_count = int(target_z.numel())
    left_indices = torch.randint(item_count, (int(pair_count),), device=target_z.device)
    right_indices = torch.randint(item_count, (int(pair_count),), device=target_z.device)
    target_diff = target_z[left_indices] - target_z[right_indices]
    pair_mask = left_indices != right_indices
    if min_target_gap > 0.0:
        pair_mask = pair_mask & (target_diff.abs() >= float(min_target_gap))
    rank_sign = target_diff.sign()
    pair_mask = pair_mask & (rank_sign != 0.0)
    if torch.count_nonzero(pair_mask).item() == 0:
        return prediction.new_zeros(())

    prediction_diff = prediction_z[left_indices] - prediction_z[right_indices]
    pair_loss = F.softplus(-prediction_diff[pair_mask] * rank_sign[pair_mask])
    pair_weight = target_diff[pair_mask].abs().clamp(max=1.0)
    return (pair_loss * pair_weight).sum() / pair_weight.sum().clamp_min(min_std)


def _exercise_difficulty_regularization_loss(
    *,
    model: DecoupledCDM,
    weight: float,
    difficulty_prior_target: torch.Tensor | None,
    difficulty_prior_mask: torch.Tensor | None,
) -> torch.Tensor:
    if weight <= 0.0:
        return model.exercise_difficulty.weight.new_zeros(())
    if difficulty_prior_target is None or difficulty_prior_mask is None:
        raise ValueError("difficulty prior target and mask are required when difficulty regularization is enabled.")
    if torch.count_nonzero(difficulty_prior_mask).item() == 0:
        return model.exercise_difficulty.weight.new_zeros(())
    current_difficulty = model.exercise_difficulty.weight[:, 0]
    residual = current_difficulty[difficulty_prior_mask] - difficulty_prior_target[difficulty_prior_mask]
    return residual.square().mean() * float(weight)
