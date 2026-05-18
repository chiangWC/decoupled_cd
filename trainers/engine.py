from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn.functional as F

from data import StepDataBundle
from models import DecoupledCDM
from utils import compute_metrics


@dataclass
class TrainResult:
    history: list[dict[str, float]]
    final_loss: float
    best_val_auc: float
    best_epoch: int
    best_checkpoint_path: str | None


@dataclass
class EpochTrainStats:
    mean_loss: float
    optimizer_steps: int


def _bundle_tensors(bundle: StepDataBundle, device: torch.device) -> dict[str, torch.Tensor | None]:
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
            bundle.exercise_evidence_tensor.to(device) if bundle.exercise_evidence_tensor is not None else None
        ),
        "interaction_student_ids": bundle.interaction_student_ids.to(device),
        "interaction_exercise_ids": bundle.interaction_exercise_ids.to(device),
        "interaction_labels": bundle.interaction_labels.to(device),
        "response_matrix": bundle.response_matrix_tensor.to(device),
    }


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
) -> dict[str, float]:
    _validate_history_visibility(bundle)
    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()
    tensors = _bundle_tensors(bundle, torch_device)

    with torch.no_grad():
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
            target_student_ids=tensors["interaction_student_ids"],
            target_exercise_ids=tensors["interaction_exercise_ids"],
        )
        loss = F.binary_cross_entropy(output.probs, tensors["interaction_labels"])

    metrics = compute_metrics(
        labels=tensors["interaction_labels"].detach().cpu().numpy(),
        probs=output.probs.detach().cpu().numpy(),
    )
    metrics["loss"] = float(loss.item())
    return metrics


def train_model(
    *,
    train_bundle: StepDataBundle,
    model: DecoupledCDM,
    valid_bundle: StepDataBundle | None = None,
    epochs: int = 5,
    batch_size: int | None = None,
    learning_rate: float = 1e-3,
    training_mode: str = "full_batch",
    device: str = "cpu",
    early_stop_patience: int = 5,
    lr_scheduler_patience: int = 10,
    lr_scheduler_factor: float = 0.5,
    min_learning_rate: float = 1e-5,
    checkpoint_path: str | None = None,
    exercise_evidence_difficulty_regularization_weight: float = 0.0,
    exercise_evidence_difficulty_regularization_min_count: int = 1,
    exercise_evidence_difficulty_regularization_max_abs_logit: float = 0.25,
    exercise_evidence_difficulty_regularization_strength: float = 2.0,
    exercise_evidence_difficulty_regularization_confidence_cap: float = 200.0,
    history_evidence_cognitive_alignment_weight: float = 0.0,
) -> TrainResult:
    if training_mode not in {"full_batch", "recompute_minibatch"}:
        raise ValueError(f"Unsupported training_mode: {training_mode}")
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive when provided.")
    if training_mode == "full_batch" and batch_size is not None:
        raise ValueError("full_batch training does not consume --batch-size; use --training-mode recompute_minibatch.")
    if training_mode == "recompute_minibatch" and batch_size is None:
        raise ValueError("recompute_minibatch training requires --batch-size.")
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

    torch_device = torch.device(device)
    model = model.to(torch_device)
    train_tensors = _bundle_tensors(train_bundle, torch_device)
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

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        min_lr=min_learning_rate,
    )
    history: list[dict[str, float]] = []
    best_state = None
    best_val_auc = float("-inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        if training_mode == "full_batch":
            train_stats = _train_full_batch_epoch(
                model=model,
                tensors=train_tensors,
                optimizer=optimizer,
                exercise_evidence_difficulty_regularization_weight=exercise_evidence_difficulty_regularization_weight,
                difficulty_prior_target=difficulty_prior_target,
                difficulty_prior_mask=difficulty_prior_mask,
                history_evidence_cognitive_alignment_weight=history_evidence_cognitive_alignment_weight,
            )
        else:
            train_stats = _train_recompute_minibatch_epoch(
                model=model,
                tensors=train_tensors,
                optimizer=optimizer,
                batch_size=int(batch_size),
                device=torch_device,
                exercise_evidence_difficulty_regularization_weight=exercise_evidence_difficulty_regularization_weight,
                difficulty_prior_target=difficulty_prior_target,
                difficulty_prior_mask=difficulty_prior_mask,
                history_evidence_cognitive_alignment_weight=history_evidence_cognitive_alignment_weight,
            )
        row = {
            "epoch": float(epoch),
            "train_loss": float(train_stats.mean_loss),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "training_mode": training_mode,
            "optimizer_steps": float(train_stats.optimizer_steps),
        }

        if valid_bundle is not None:
            val_metrics = evaluate_model(bundle=valid_bundle, model=model, device=device)
            row["val_loss"] = float(val_metrics["loss"])
            row["val_auc"] = float(val_metrics["auc"])
            row["val_acc"] = float(val_metrics["acc"])
            row["val_rmse"] = float(val_metrics["rmse"])
            row["val_brier"] = float(val_metrics["brier"])
            row["val_ece"] = float(val_metrics["ece"])
            scheduler.step(row["val_auc"])

            if row["val_auc"] > best_val_auc:
                best_val_auc = row["val_auc"]
                best_epoch = epoch
                patience_counter = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if checkpoint_path is not None:
                    torch.save(best_state, checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    history.append(row)
                    break
        else:
            if epoch == epochs:
                best_epoch = epoch
                best_val_auc = float("nan")

        history.append(row)

    if best_state is not None:
        model.load_state_dict(best_state)

    final_loss = history[-1]["train_loss"] if history else 0.0
    return TrainResult(
        history=history,
        final_loss=final_loss,
        best_val_auc=best_val_auc,
        best_epoch=best_epoch,
        best_checkpoint_path=checkpoint_path if best_state is not None else None,
    )


def _train_full_batch_epoch(
    *,
    model: DecoupledCDM,
    tensors: dict[str, torch.Tensor | None],
    optimizer: torch.optim.Optimizer,
    exercise_evidence_difficulty_regularization_weight: float = 0.0,
    difficulty_prior_target: torch.Tensor | None = None,
    difficulty_prior_mask: torch.Tensor | None = None,
    history_evidence_cognitive_alignment_weight: float = 0.0,
) -> EpochTrainStats:
    model.train()
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
        target_student_ids=tensors["interaction_student_ids"],
        target_exercise_ids=tensors["interaction_exercise_ids"],
    )
    loss = F.binary_cross_entropy(output.probs, tensors["interaction_labels"])
    if history_evidence_cognitive_alignment_weight > 0.0:
        loss = loss + _history_evidence_cognitive_alignment_loss(
            model=model,
            output=output,
            tensors=tensors,
            target_student_ids=tensors["interaction_student_ids"],
            target_exercise_ids=tensors["interaction_exercise_ids"],
            weight=history_evidence_cognitive_alignment_weight,
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
        )
        loss = F.binary_cross_entropy(output.probs, batch_labels)
        if history_evidence_cognitive_alignment_weight > 0.0:
            loss = loss + _history_evidence_cognitive_alignment_loss(
                model=model,
                output=output,
                tensors=tensors,
                target_student_ids=tensors["interaction_student_ids"][batch_indices],
                target_exercise_ids=tensors["interaction_exercise_ids"][batch_indices],
                weight=history_evidence_cognitive_alignment_weight,
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


def _history_evidence_cognitive_alignment_loss(
    *,
    model: DecoupledCDM,
    output: object,
    tensors: dict[str, torch.Tensor | None],
    target_student_ids: torch.Tensor,
    target_exercise_ids: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    if weight <= 0.0:
        return target_student_ids.new_zeros((), dtype=torch.float32)
    if not getattr(model, "history_evidence_logit_prior_residual", False):
        raise ValueError(
            "history_evidence_logit_prior_residual must be enabled when cognitive alignment loss is enabled."
        )
    if getattr(model, "history_evidence_logit_prior_location", None) != "loss_only":
        raise ValueError(
            "history_evidence_logit_prior_location must be loss_only when cognitive alignment loss is enabled."
        )
    if not hasattr(output, "cognitive_probs"):
        raise ValueError("model output must expose cognitive_probs when cognitive alignment loss is enabled.")

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
    cognitive_logits = torch.logit(output.cognitive_probs.clamp(min=1e-6, max=1.0 - 1e-6))
    return _standardized_mse_alignment_loss(cognitive_logits, prior) * float(weight)


def _standardized_mse_alignment_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
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
    return F.mse_loss(prediction_z, target_z)


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
