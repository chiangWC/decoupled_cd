from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import prepare_experiment_split_bundles
from data.pool_protocol import sha256_file
from models import KaNCDBaseline
from trainers import train_model
from trainers.engine import _bundle_tensors
from utils import compute_metrics, set_global_seed, write_json


FILES = ("train.csv", "valid.csv", "test.csv", "Q_matrix.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen non-graph KaNCD Gate C arm."
    )
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument(
        "--arm",
        choices=("concept_depleted", "random_depleted"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prediction-batch-size", type=int, default=8192)
    return parser.parse_args()


def verify_protocol_arm(
    protocol_dir: Path, arm: str
) -> tuple[dict[str, Any], Path, dict[str, str]]:
    manifest_path = protocol_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "paired_concept_depletion_gate_b":
        raise ValueError(f"{manifest_path}: unexpected protocol.")
    if manifest.get("source_test_opened") is not False:
        raise ValueError(f"{manifest_path}: source-test boundary is not locked.")
    if not manifest.get("qualified_for_screen"):
        raise ValueError(f"{manifest_path}: protocol did not qualify.")
    arm_dir = protocol_dir / arm
    expected = manifest["generated_hashes"][arm]
    actual = {name: sha256_file(arm_dir / name) for name in FILES}
    if actual != expected:
        raise ValueError(
            f"{protocol_dir}: generated-arm hashes differ from Gate B."
        )
    return manifest, arm_dir, actual


def predict_bundle(
    *,
    model: KaNCDBaseline,
    bundle,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if batch_size < 1:
        raise ValueError("prediction batch size must be positive.")
    model.eval()
    tensors = _bundle_tensors(bundle, device)
    labels = tensors["interaction_labels"]
    probabilities: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, labels.numel(), batch_size):
            stop = min(start + batch_size, labels.numel())
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
                target_student_ids=tensors["interaction_student_ids"][start:stop],
                target_exercise_ids=tensors["interaction_exercise_ids"][start:stop],
            )
            probabilities.append(output.probs.detach().cpu())
    return (
        torch.cat(probabilities).numpy(),
        labels.detach().cpu().numpy(),
    )


def align_and_write(
    *,
    source: Path,
    probabilities: np.ndarray,
    labels: np.ndarray,
    output: Path,
) -> dict[str, Any]:
    frame = pd.read_csv(source).reset_index(drop=True)
    expected = pd.to_numeric(frame["label"], errors="raise").to_numpy(float)
    if (
        len(frame) != len(probabilities)
        or not np.array_equal(expected, labels)
    ):
        raise RuntimeError(f"KaNCD predictions do not align with {source}.")
    frame["prob"] = probabilities
    frame.to_csv(output, index=False)
    return {
        "metrics": compute_metrics(expected, probabilities),
        "prediction_path": str(output.resolve()),
        "prediction_sha256": sha256_file(output),
        "rows": len(frame),
        "row_aligned": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed != 42:
        raise ValueError("Gate C fixes model seed to 42.")
    frozen = {
        "epochs": 50,
        "patience": 10,
        "batch_size": 4096,
        "learning_rate": 1e-3,
        "latent_dim": 64,
    }
    actual = {
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "latent_dim": args.latent_dim,
    }
    if actual != frozen:
        raise ValueError(f"Gate C KaNCD recipe is frozen: {frozen}.")
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
    model = KaNCDBaseline(
        num_students=train_bundle.num_students,
        num_exercises=train_bundle.num_exercises,
        num_concepts=train_bundle.num_concepts,
        latent_dim=args.latent_dim,
    )
    checkpoint = args.output_dir / "best_model.pt"
    result = train_model(
        train_bundle=train_bundle,
        valid_bundle=bundles["valid"],
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        training_mode="recompute_minibatch",
        device=str(device),
        early_stop_patience=args.patience,
        checkpoint_selection_metric="auc",
        checkpoint_path=str(checkpoint),
    )
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
    payload = {
        "schema_version": 1,
        "model": "KaNCD",
        "model_family": "non_graph_id_factorized",
        "dataset": manifest["dataset"],
        "arm": args.arm,
        "seed": args.seed,
        "selection": "best early-stop validation AUC",
        "settings": {
            **frozen,
            "training_mode": "recompute_minibatch",
            "prediction_batch_size": args.prediction_batch_size,
        },
        "dimensions": {
            "students": train_bundle.num_students,
            "exercises": train_bundle.num_exercises,
            "concepts": train_bundle.num_concepts,
        },
        "train_result": asdict(result),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_source": {
            "path": "models/kancd_baseline.py",
            "sha256": sha256_file(REPO_ROOT / "models/kancd_baseline.py"),
            "publication_caveat": (
                "in-harness reimplementation; cross-check against EduCDM "
                "before publication"
            ),
        },
        "protocol": {
            "manifest": str(
                (args.protocol_dir / "manifest.json").resolve()
            ),
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
    args = parse_args()
    payload = run(args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
