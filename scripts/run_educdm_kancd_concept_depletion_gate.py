from __future__ import annotations

import argparse
import copy
import inspect
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import prepare_experiment_split_bundles
from data.pool_protocol import sha256_file
from scripts.run_kancd_concept_depletion_gate import (
    align_and_write,
    verify_protocol_arm,
)
from utils import set_global_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-check Gate C with the official EduCDM KaNCD network "
            "under the shared row-aligned harness."
        )
    )
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument(
        "--arm",
        choices=("concept_depleted", "random_depleted"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prediction-batch-size", type=int, default=2048)
    return parser.parse_args()


def load_official_net_class():
    from EduCDM.KaNCD.KaNCD import Net

    return Net


def predict_bundle(
    *,
    model: torch.nn.Module,
    bundle,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    q_matrix = bundle.q_matrix_tensor.to(device)
    students = bundle.interaction_student_ids
    exercises = bundle.interaction_exercise_ids
    labels = bundle.interaction_labels
    probabilities: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, labels.numel(), batch_size):
            stop = min(start + batch_size, labels.numel())
            student_batch = students[start:stop].to(device)
            exercise_batch = exercises[start:stop].to(device)
            probabilities.append(
                model(
                    student_batch,
                    exercise_batch,
                    q_matrix[exercise_batch],
                ).detach().cpu()
            )
    return torch.cat(probabilities).numpy(), labels.numpy()


def validation_auc(
    *,
    model: torch.nn.Module,
    bundle,
    device: torch.device,
    batch_size: int,
) -> float:
    probabilities, labels = predict_bundle(
        model=model,
        bundle=bundle,
        device=device,
        batch_size=batch_size,
    )
    return float(roc_auc_score(labels, probabilities))


def train_official_net(
    *,
    model: torch.nn.Module,
    train_bundle,
    valid_bundle,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[list[dict[str, float]], int, float]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    students = train_bundle.interaction_student_ids
    exercises = train_bundle.interaction_exercise_ids
    labels = train_bundle.interaction_labels
    q_matrix = train_bundle.q_matrix_tensor.to(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_auc = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(labels.numel(), generator=generator)
        total_loss = 0.0
        seen = 0
        for start in range(0, labels.numel(), batch_size):
            indices = permutation[start : start + batch_size]
            student_batch = students[indices].to(device)
            exercise_batch = exercises[indices].to(device)
            label_batch = labels[indices].to(device)
            probabilities = model(
                student_batch,
                exercise_batch,
                q_matrix[exercise_batch],
            )
            loss = F.binary_cross_entropy(probabilities, label_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = int(label_batch.numel())
            total_loss += float(loss.detach()) * count
            seen += count

        auc = validation_auc(
            model=model,
            bundle=valid_bundle,
            device=device,
            batch_size=batch_size,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": total_loss / max(seen, 1),
                "valid_auc": auc,
            }
        )
        print(
            f"epoch={epoch} train_loss={history[-1]['train_loss']:.6f} "
            f"valid_auc={auc:.6f}",
            flush=True,
        )
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("EduCDM KaNCD training produced no checkpoint.")
    model.load_state_dict(best_state)
    return history, best_epoch, best_auc


def run(args: argparse.Namespace) -> dict[str, Any]:
    frozen = {
        "seed": 42,
        "epochs": 30,
        "patience": 10,
        "batch_size": 1024,
        "learning_rate": 2e-3,
        "latent_dim": 64,
        "mf_type": "gmf",
    }
    actual = {
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "latent_dim": args.latent_dim,
        "mf_type": "gmf",
    }
    if actual != frozen:
        raise ValueError(f"EduCDM KaNCD cross-check recipe is frozen: {frozen}.")
    manifest, arm_dir, arm_hashes = verify_protocol_arm(
        args.protocol_dir, args.arm
    )
    set_global_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundles = prepare_experiment_split_bundles(
        train_interactions_path=arm_dir / "train.csv",
        valid_interactions_path=arm_dir / "valid.csv",
        test_interactions_path=arm_dir / "test.csv",
        q_matrix_path=arm_dir / "Q_matrix.csv",
    )
    train_bundle = bundles["train"]
    net_class = load_official_net_class()
    model = net_class(
        train_bundle.num_exercises,
        train_bundle.num_students,
        train_bundle.num_concepts,
        "gmf",
        args.latent_dim,
    )
    history, best_epoch, best_auc = train_official_net(
        model=model,
        train_bundle=train_bundle,
        valid_bundle=bundles["valid"],
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    checkpoint = args.output_dir / "best_model.pt"
    torch.save(model.state_dict(), checkpoint)
    summaries: dict[str, Any] = {}
    for split in ("valid", "test"):
        probabilities, labels = predict_bundle(
            model=model,
            bundle=bundles[split],
            device=device,
            batch_size=args.prediction_batch_size,
        )
        summaries[split] = align_and_write(
            source=arm_dir / f"{split}.csv",
            probabilities=probabilities,
            labels=labels,
            output=args.output_dir / f"{split}_predictions.csv",
        )
    source_path = Path(inspect.getsourcefile(net_class) or "")
    payload = {
        "schema_version": 1,
        "model": "EduCDM-KaNCD-GMF",
        "model_family": "non_graph_id_factorized_official_network",
        "dataset": manifest["dataset"],
        "arm": args.arm,
        "seed": args.seed,
        "selection": "best early-stop validation AUC in shared harness",
        "settings": {
            **frozen,
            "prediction_batch_size": args.prediction_batch_size,
        },
        "dimensions": {
            "students": train_bundle.num_students,
            "exercises": train_bundle.num_exercises,
            "concepts": train_bundle.num_concepts,
        },
        "train_result": {
            "history": history,
            "best_epoch": best_epoch,
            "best_valid_auc": best_auc,
        },
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_source": {
            "class": "EduCDM.KaNCD.KaNCD.Net",
            "path": str(source_path.resolve()),
            "sha256": sha256_file(source_path),
            "adaptation": (
                "Official network class; row-aligned shared harness, "
                "validation checkpointing, and larger frozen batch."
            ),
        },
        "protocol": {
            "manifest": str((args.protocol_dir / "manifest.json").resolve()),
            "manifest_sha256": sha256_file(
                args.protocol_dir / "manifest.json"
            ),
            "arm_hashes": arm_hashes,
            "source_test_opened": False,
        },
        "splits": summaries,
    }
    write_json(payload, args.output_dir / "summary.json")
    return payload


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
