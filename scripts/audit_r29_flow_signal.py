from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.r29_protocol import canonical_concepts, stable_fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Masked-cell proxy gate for r29 accuracy-oriented flow completion.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-fraction", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-train-cells", type=int, default=100000)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def expand_cells(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    interactions = frame.copy()
    interactions["cpt_seq"] = interactions["cpt_seq"].map(canonical_concepts).str.split(",")
    interactions = interactions.explode("cpt_seq", ignore_index=True).rename(columns={"cpt_seq": "concept"})
    cells = (
        interactions.groupby(["stu_id", "concept"], as_index=False)
        .agg(correct=("label", "sum"), attempts=("label", "size"))
    )
    cells["accuracy"] = cells["correct"] / cells["attempts"]
    return interactions, cells


def build_features(cells: pd.DataFrame, *, seed: int, mask_fraction: float) -> pd.DataFrame:
    output = cells.copy()
    output["heldout"] = [
        stable_fraction(seed, "flow-proxy-mask", row.stu_id, row.concept) < mask_fraction
        for row in output.itertuples(index=False)
    ]
    kept = output[~output["heldout"]]
    global_correct = float(kept["correct"].sum())
    global_attempts = float(kept["attempts"].sum())
    global_rate = global_correct / max(global_attempts, 1.0)
    student_stats = kept.groupby("stu_id").agg(
        student_correct=("correct", "sum"), student_attempts=("attempts", "sum")
    )
    concept_stats = kept.groupby("concept").agg(
        concept_correct=("correct", "sum"), concept_attempts=("attempts", "sum")
    )
    output = output.join(student_stats, on="stu_id").join(concept_stats, on="concept")
    for column in ("student_correct", "student_attempts", "concept_correct", "concept_attempts"):
        output[column] = output[column].fillna(0.0)
    train_cell = (~output["heldout"]).astype(float)
    student_correct = output["student_correct"] - train_cell * output["correct"]
    student_attempts = output["student_attempts"] - train_cell * output["attempts"]
    concept_correct = output["concept_correct"] - train_cell * output["correct"]
    concept_attempts = output["concept_attempts"] - train_cell * output["attempts"]
    output["student_rate"] = (student_correct + global_rate * 2.0) / (student_attempts + 2.0)
    output["concept_rate"] = (concept_correct + global_rate * 2.0) / (concept_attempts + 2.0)
    output["global_rate"] = global_rate
    output["student_support"] = np.log1p(student_attempts) / math.log1p(max(student_attempts.max(), 1.0))
    output["concept_support"] = np.log1p(concept_attempts) / math.log1p(max(concept_attempts.max(), 1.0))
    output["interaction_support"] = np.log1p(output["attempts"]) / math.log1p(max(output["attempts"].max(), 1.0))
    return output


FEATURE_COLUMNS = (
    "student_rate",
    "concept_rate",
    "global_rate",
    "student_support",
    "concept_support",
    "interaction_support",
)


class MatchedProxy(nn.Module):
    def __init__(self, *, flow: bool, steps: int) -> None:
        super().__init__()
        self.flow = bool(flow)
        self.steps = int(steps)
        self.feature_encoder = nn.Sequential(nn.Linear(6, 32), nn.Tanh())
        self.score = nn.Sequential(nn.Linear(33, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.feature_encoder(features)
        initial_probability = (features[:, 0] + features[:, 1]) * 0.5
        state = torch.logit(initial_probability.clamp(1e-4, 1.0 - 1e-4)).unsqueeze(-1)
        if not self.flow:
            return (state + self.score(torch.cat([encoded, torch.sigmoid(state)], dim=-1))).squeeze(-1)
        bandwidth = torch.cdist(encoded.detach(), encoded.detach()).square().median().clamp_min(1e-4)
        kernel = torch.exp(-torch.cdist(encoded.detach(), encoded.detach()).square() / bandwidth)
        for _ in range(self.steps):
            particle_score = self.score(torch.cat([encoded, torch.sigmoid(state)], dim=-1))
            attraction = kernel @ particle_score / kernel.sum(dim=1, keepdim=True).clamp_min(1.0)
            differences = state - state.transpose(0, 1)
            concentration = (kernel * differences).sum(dim=1, keepdim=True)
            concentration = concentration / kernel.sum(dim=1, keepdim=True).clamp_min(1.0)
            state = state + 0.25 * (attraction - 0.1 * concentration)
        return state.squeeze(-1)


def state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_proxy(
    *,
    flow: bool,
    steps: int,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    train_weights: torch.Tensor,
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[MatchedProxy, str]:
    set_seed(seed)
    model = MatchedProxy(flow=flow, steps=steps)
    initial_hash = state_hash(model)
    model = model.to(device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_features, train_targets, train_weights),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        for features, targets, weights in loader:
            features = features.to(device)
            targets = targets.to(device)
            weights = weights.to(device)
            optimizer.zero_grad()
            logits = model(features)
            losses = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            loss = (losses * weights).sum() / weights.sum().clamp_min(1.0)
            loss.backward()
            optimizer.step()
    return model, initial_hash


def predict_in_batches(
    model: nn.Module,
    features: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = features[start : start + batch_size].to(device)
            output.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(output)


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    )
    frame = pd.read_csv(args.train, dtype={"stu_id": str, "exer_id": str, "cpt_seq": str})
    interactions, cells = expand_cells(frame)
    features = build_features(cells, seed=args.seed, mask_fraction=args.mask_fraction)
    train = features[~features["heldout"]].copy()
    train["sample_key"] = [
        stable_fraction(args.seed, "flow-proxy-train", row.stu_id, row.concept)
        for row in train.itertuples(index=False)
    ]
    train = train.nsmallest(min(args.max_train_cells, len(train)), "sample_key")
    heldout = features[features["heldout"]].copy()
    if len(heldout) < 100 or set((heldout["correct"] > 0).astype(int)) != {0, 1}:
        raise RuntimeError("Masked flow proxy has insufficient heldout cells or labels.")
    train_x = torch.tensor(train.loc[:, FEATURE_COLUMNS].to_numpy(), dtype=torch.float32)
    train_y = torch.tensor(train["accuracy"].to_numpy(), dtype=torch.float32)
    train_w = torch.tensor(train["attempts"].to_numpy(), dtype=torch.float32)
    heldout_x = torch.tensor(heldout.loc[:, FEATURE_COLUMNS].to_numpy(), dtype=torch.float32)
    models: dict[str, MatchedProxy] = {}
    hashes: dict[str, str] = {}
    for name, flow in (("flow", True), ("capacity", False)):
        models[name], hashes[name] = train_proxy(
            flow=flow,
            steps=args.steps,
            train_features=train_x,
            train_targets=train_y,
            train_weights=train_w,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
        )
    if hashes["flow"] != hashes["capacity"]:
        raise RuntimeError("Flow and capacity proxy common initialization differs.")
    cell_predictions = {
        name: predict_in_batches(
            model, heldout_x, batch_size=args.batch_size, device=device
        )
        for name, model in models.items()
    }
    prediction_maps = {
        name: {
            (str(row.stu_id), str(row.concept)): float(prediction)
            for row, prediction in zip(heldout.itertuples(index=False), values, strict=True)
        }
        for name, values in cell_predictions.items()
    }
    heldout_keys = set(prediction_maps["flow"])
    selected = interactions[
        [
            (student, concept) in heldout_keys
            for student, concept in zip(
                interactions["stu_id"].astype(str), interactions["concept"].astype(str), strict=True
            )
        ]
    ]
    labels = selected["label"].astype(int).to_numpy()
    metrics: dict[str, Any] = {}
    for name, mapping in prediction_maps.items():
        predictions = np.asarray(
            [mapping[(student, concept)] for student, concept in zip(
                selected["stu_id"].astype(str), selected["concept"].astype(str), strict=True
            )]
        )
        metrics[name] = {
            "auc": float(roc_auc_score(labels, predictions)),
            "brier": float(brier_score_loss(labels, predictions)),
        }
    delta_auc = metrics["flow"]["auc"] - metrics["capacity"]["auc"]
    delta_brier = metrics["flow"]["brier"] - metrics["capacity"]["brier"]
    return {
        "schema_version": 1,
        "dataset": args.dataset,
        "seed": args.seed,
        "history_source": "train_only",
        "mask_fraction": args.mask_fraction,
        "steps": args.steps,
        "train_cells": len(train),
        "heldout_cells": len(heldout),
        "heldout_interaction_rows": len(selected),
        "parameter_count": sum(parameter.numel() for parameter in models["flow"].parameters()),
        "common_initialization_hash": hashes["flow"],
        "flow": metrics["flow"],
        "capacity": metrics["capacity"],
        "delta_auc": delta_auc,
        "delta_brier": delta_brier,
        "activation_threshold": {"min_delta_auc": 0.002, "max_delta_brier": 0.0},
        "passed": delta_auc >= 0.002 and delta_brier <= 0.0,
    }


def main() -> None:
    args = parse_args()
    if args.seed != 42 or args.steps != 4 or args.mask_fraction != 0.2:
        raise ValueError("r29 flow proxy is fixed to seed=42, steps=4 and mask_fraction=0.2.")
    payload = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
