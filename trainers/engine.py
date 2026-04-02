from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from data import StepDataBundle, build_response_matrix
from models import DecoupledCDM
from utils import compute_metrics


@dataclass
class TrainResult:
    history: list[dict[str, float]]
    final_loss: float
    best_val_auc: float
    best_epoch: int
    best_checkpoint_path: str | None


def _bundle_tensors(bundle: StepDataBundle, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "q_matrix": bundle.q_matrix_tensor.to(device),
        "concept_graph": bundle.concept_graph.to(device),
        "student_exercise_mask": bundle.student_exercise_mask.to(device),
        "student_tkc_mask": bundle.student_tkc_mask.to(device),
        "student_ukc_mask": bundle.student_ukc_mask.to(device),
        "interaction_student_ids": bundle.interaction_student_ids.to(device),
        "interaction_exercise_ids": bundle.interaction_exercise_ids.to(device),
        "interaction_labels": bundle.interaction_labels.to(device),
        "response_matrix": build_response_matrix(
            interactions=bundle.interactions,
            student_id_map=bundle.student_id_map,
            exercise_id_map=bundle.exercise_id_map,
        ).to(device),
    }


def evaluate_model(
    *,
    bundle: StepDataBundle,
    model: DecoupledCDM,
    device: str = "cpu",
) -> dict[str, float]:
    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()
    tensors = _bundle_tensors(bundle, torch_device)

    with torch.no_grad():
        output = model(
            q_matrix=tensors["q_matrix"],
            concept_graph=tensors["concept_graph"],
            student_exercise_mask=tensors["student_exercise_mask"],
            response_matrix=tensors["response_matrix"],
            student_tkc_mask=tensors["student_tkc_mask"],
            student_ukc_mask=tensors["student_ukc_mask"],
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
    learning_rate: float = 1e-3,
    device: str = "cpu",
    early_stop_patience: int = 5,
    lr_scheduler_patience: int = 10,
    lr_scheduler_factor: float = 0.5,
    min_learning_rate: float = 1e-5,
    checkpoint_path: str | None = None,
) -> TrainResult:
    torch_device = torch.device(device)
    model = model.to(torch_device)
    train_tensors = _bundle_tensors(train_bundle, torch_device)

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
        model.train()
        optimizer.zero_grad()

        output = model(
            q_matrix=train_tensors["q_matrix"],
            concept_graph=train_tensors["concept_graph"],
            student_exercise_mask=train_tensors["student_exercise_mask"],
            response_matrix=train_tensors["response_matrix"],
            student_tkc_mask=train_tensors["student_tkc_mask"],
            student_ukc_mask=train_tensors["student_ukc_mask"],
            target_student_ids=train_tensors["interaction_student_ids"],
            target_exercise_ids=train_tensors["interaction_exercise_ids"],
        )
        loss = F.binary_cross_entropy(output.probs, train_tensors["interaction_labels"])
        loss.backward()
        optimizer.step()

        mean_loss = float(loss.item())
        row = {
            "epoch": float(epoch),
            "train_loss": float(mean_loss),
            "alpha": float(model.propagation.alpha.detach().cpu().item()),
            "beta": float(model.propagation.beta.detach().cpu().item()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }

        if valid_bundle is not None:
            val_metrics = evaluate_model(bundle=valid_bundle, model=model, device=device)
            row["val_loss"] = float(val_metrics["loss"])
            row["val_auc"] = float(val_metrics["auc"])
            row["val_acc"] = float(val_metrics["acc"])
            row["val_rmse"] = float(val_metrics["rmse"])
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
