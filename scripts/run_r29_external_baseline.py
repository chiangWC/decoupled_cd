from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_coverage_slice import add_target_coverage, compute_slice_rows
from utils import compute_metrics, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and row-align a seed-42 external r29 baseline.")
    parser.add_argument("--model", choices=["orcdf", "svgcd"], required=True)
    parser.add_argument("--external-root", required=True, help="Directory containing ORCDF/ and SVGCD/ packages.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-variant", choices=["standard", "holdout"], required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--target-scope", choices=["bucket:zero", "low_coverage"], default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def external_args(args: argparse.Namespace) -> SimpleNamespace:
    common: dict[str, Any] = {
        "data_dir": str(Path(args.data_dir).resolve()),
        "log_dir": str(Path(args.output_dir).resolve()),
        "train_file": "train.csv",
        "valid_file": "valid.csv",
        "test_file": "test.csv",
        "lr": 0.004 if args.model == "orcdf" else 0.001,
        "weight_decay": 0.0,
        "batch_size": args.batch_size,
        "epochs": args.epochs or (30 if args.model == "orcdf" else 100),
        "patience": args.patience or (5 if args.model == "orcdf" else 10),
        "seed": 42,
        "experiment_name": f"r29_external_{args.model}_{args.dataset}_{args.dataset_variant}",
    }
    if args.model == "orcdf":
        common.update(
            {
                "latent_dim": 32,
                "gcn_layers": 3,
                "keep_prob": 0.9,
                "if_type": "ncd",
                "mode": "all",
                "flip_ratio": 0.15,
                "ssl_temp": 0.5,
                "ssl_weight": 0.001,
                "prednet_len1": 512,
                "prednet_len2": 256,
                "dropout": 0.0,
                "conflict_num_groups": 10,
            }
        )
    else:
        common.update(
            {
                "emb_dim": 128,
                "dnn_units": [256, 128],
                "dropout_rate": 0.5,
                "n_gnn_layer": 2,
                "cl_tau": 0.7,
                "cl_weight": 0.5,
                "beta": 0.4,
                "eps": 1e-8,
                "eval_batch_size": args.eval_batch_size,
                "longtail_bin_edges": "0,20,40,60,80,100,150,250,500",
            }
        )
    return SimpleNamespace(**common)


def build_components(args: argparse.Namespace, settings: SimpleNamespace):
    external_root = Path(args.external_root).resolve()
    sys.path.insert(0, str(external_root))
    package = args.model.upper()
    dataset_module = importlib.import_module(f"{package}.dataset")
    model_module = importlib.import_module(f"{package}.model")
    trainer_module = importlib.import_module(f"{package}.trainer")
    utils_module = importlib.import_module(f"{package}.utils")
    utils_module.set_seed(42)
    device = utils_module.get_device()
    logger = logging.getLogger(settings.experiment_name)
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    processor = dataset_module.CognitiveDataProcessor(settings, logger)
    loaders = processor.get_loaders()
    if args.model == "orcdf":
        model = model_module.ORCDFNet(
            student_n=processor.num_students,
            exer_n=processor.num_exercises,
            knowledge_n=processor.num_concepts,
            latent_dim=settings.latent_dim,
            gcn_layers=settings.gcn_layers,
            keep_prob=settings.keep_prob,
            if_type=settings.if_type,
            mode=settings.mode,
            flip_ratio=settings.flip_ratio,
            ssl_temp=settings.ssl_temp,
            ssl_weight=settings.ssl_weight,
            prednet_len1=settings.prednet_len1,
            prednet_len2=settings.prednet_len2,
            dropout=settings.dropout,
            device=device,
        ).to(device)
        model.get_graph_dict(processor.graph_dict)
    else:
        model = model_module.SVGCDNet(
            student_n=processor.num_students,
            exer_n=processor.num_exercises,
            knowledge_n=processor.num_concepts,
            args=settings,
            pos_graph=processor.correct_adj,
            neg_graph=processor.wrong_adj,
            device=device,
        ).to(device)
    trainer = trainer_module.Trainer(model, loaders, processor, settings, logger)
    return external_root, processor, model, trainer, device


def predict(model_name: str, model: torch.nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: list[float] = []
    labels: list[float] = []
    with torch.no_grad():
        for batch in loader:
            if model_name == "orcdf":
                student, exercise, q_rows, label = batch
                prediction, _ = model(student, exercise, q_rows)
            else:
                moved = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
                prediction = model.forward_test(
                    moved["stu_id"], moved["exer_id"], moved["Q_mat"]
                )
                label = moved["label"]
            probs.extend(prediction.detach().cpu().reshape(-1).tolist())
            labels.extend(label.detach().cpu().reshape(-1).tolist())
    return np.asarray(probs), np.asarray(labels)


def align_predictions(
    *,
    source: Path,
    probs: np.ndarray,
    labels: np.ndarray,
    train_frame: pd.DataFrame,
    q_matrix: pd.DataFrame,
    dataset: str,
    model_name: str,
    target_scope: str | None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any] | None, list[dict[str, Any]]]:
    frame = pd.read_csv(source).reset_index(drop=True)
    expected_labels = pd.to_numeric(frame["label"], errors="raise").to_numpy(dtype=float)
    if len(frame) != len(probs) or not np.array_equal(expected_labels, labels):
        raise RuntimeError(f"External {model_name} predictions are not row-aligned with {source}.")
    frame["prob"] = probs
    metrics = compute_metrics(expected_labels, probs)
    target = None
    slice_rows: list[dict[str, Any]] = []
    if target_scope is not None:
        frame = add_target_coverage(frame, train_frame=train_frame, q_matrix=q_matrix)
        slice_rows = compute_slice_rows(frame, dataset_name=dataset, model_name=model_name)
        by_scope = {row["scope"]: row for row in slice_rows}
        if target_scope not in by_scope:
            raise RuntimeError(f"Target scope {target_scope} is unavailable for {dataset}.")
        target = by_scope[target_scope]
    return frame, metrics, target, slice_rows


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("r29 external baselines are fixed to seed=42.")
    if args.dataset_variant == "holdout" and args.target_scope is None:
        raise ValueError("holdout reproduction requires --target-scope.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = external_args(args)
    external_root, processor, model, trainer, device = build_components(args, settings)
    checkpoint = output_dir / "best_model.pth"
    best_auc = float("-inf")
    patience = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, settings.epochs + 1):
        train_output = trainer.train_epoch(epoch)
        validation = trainer.evaluate(trainer.val_loader, "Valid")
        history.append({"epoch": epoch, "validation": validation, "train_output": str(train_output)})
        if float(validation["auc"]) > best_auc:
            best_auc = float(validation["auc"])
            patience = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            patience += 1
            if patience >= settings.patience:
                break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False), strict=True)
    train_frame = pd.read_csv(Path(args.data_dir) / "train.csv")
    q_matrix = pd.read_csv(Path(args.data_dir) / "Q_matrix.csv")
    summaries: dict[str, Any] = {}
    for split, loader in (("valid", trainer.val_loader), ("test", trainer.test_loader)):
        probs, labels = predict(args.model, model, loader, device)
        aligned, metrics, target, slice_rows = align_predictions(
            source=Path(args.data_dir) / f"{split}.csv",
            probs=probs,
            labels=labels,
            train_frame=train_frame,
            q_matrix=q_matrix,
            dataset=args.dataset,
            model_name=args.model.upper(),
            target_scope=args.target_scope if args.dataset_variant == "holdout" else None,
        )
        prediction_path = output_dir / f"{split}_predictions.csv"
        aligned.to_csv(prediction_path, index=False)
        if slice_rows:
            pd.DataFrame(slice_rows).to_csv(output_dir / f"{split}_slices.csv", index=False)
        summaries[split] = {
            "metrics": metrics,
            "target_metrics": target,
            "prediction_path": str(prediction_path.resolve()),
            "prediction_sha256": sha256_file(prediction_path),
            "rows": len(aligned),
            "row_aligned": True,
        }
    package_dir = external_root / args.model.upper()
    source_hashes = {
        filename: sha256_file(package_dir / filename)
        for filename in ("model.py", "dataset.py", "trainer.py")
    }
    payload = {
        "schema_version": 2,
        "external_model": args.model.upper(),
        "dataset": args.dataset,
        "dataset_variant": args.dataset_variant,
        "seed": 42,
        "selection": "best validation overall AUC",
        "target_history_source": "train_only",
        "settings": vars(settings),
        "external_source": {
            "mode": "read-only package import; no external model source copied",
            "root": str(package_dir),
            "source_hashes": source_hashes,
        },
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "best_validation_auc": best_auc,
        "history": history,
        "splits": summaries,
    }
    write_json(payload, output_dir / "summary.json")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
