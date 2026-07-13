from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import StepDataBundle, prepare_experiment_split_bundles
from models import (
    R29_COMPLETER_MODES,
    R29_COMPLETION_OBJECTIVES,
    R29CompletionCDM,
)
from scripts.train_r28 import (
    compute_doa_from_predictions,
    load_json,
    predict_bundle,
    sha256_file,
    sha256_payload,
    target_scope_metrics,
)
from trainers import train_model
from utils import compute_metrics, resolve_device, set_global_seed, write_json


EXECUTABLE_COMPLETERS = (
    "marginal_anchor",
    "meta_implicit",
    "direct_prior",
    "capacity_control",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-first r29 completion trainer.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-variant", choices=["standard", "holdout"], required=True)
    parser.add_argument("--model", choices=["r29_completion"], default="r29_completion")
    parser.add_argument("--state-completer", choices=R29_COMPLETER_MODES, required=True)
    parser.add_argument(
        "--completion-objective", choices=R29_COMPLETION_OBJECTIVES, default="none"
    )
    parser.add_argument(
        "--evaluation-stage", choices=["validation", "test-confirmation"], default="validation"
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-root", default=os.environ.get("KNOFIELD_DATA_ROOT", "data"))
    parser.add_argument("--registry", default=str(PROJECT_ROOT / "configs/r29_registry.json"))
    parser.add_argument("--recipes", default=str(PROJECT_ROOT / "configs/r29_recipes.json"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--student-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--concept-dim", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--scheduler-patience", type=int, default=None)
    parser.add_argument("--max-target-rows", type=int, default=None, help="Smoke-only cap.")
    return parser.parse_args()


def resolve_recipe(args: argparse.Namespace, recipes: dict[str, Any]) -> dict[str, Any]:
    recipe = dict(recipes["defaults"])
    recipe.update(recipes.get("dataset_overrides", {}).get(args.dataset, {}))
    for key in (
        "epochs",
        "student_batch_size",
        "learning_rate",
        "weight_decay",
        "concept_dim",
        "early_stop_patience",
        "scheduler_patience",
    ):
        value = getattr(args, key)
        if value is not None:
            recipe[key] = value
    recipe["seed"] = args.seed
    return recipe


def dataset_spec(registry: dict[str, Any], dataset: str) -> dict[str, Any]:
    entry = registry["datasets"].get(dataset)
    if entry is None:
        raise ValueError(f"Unknown r29 dataset: {dataset}")
    if entry.get("source") == "r28_registry":
        base_path = PROJECT_ROOT / registry["base_registry"]
        return load_json(base_path)["datasets"][dataset]
    return entry


def resolve_data_paths(
    *,
    registry: dict[str, Any],
    dataset: str,
    dataset_variant: str,
    data_root: str | Path,
) -> tuple[dict[str, Path], dict[str, Any]]:
    r29_entry = registry["datasets"].get(dataset)
    if r29_entry is None:
        raise ValueError(f"Unknown r29 dataset: {dataset}")
    if r29_entry.get("status") not in {"active", "external_reproduction_pending"}:
        raise ValueError(f"Dataset {dataset!r} is not data-ready in r29.")
    spec = dataset_spec(registry, dataset)
    directory_key = "standard_dir" if dataset_variant == "standard" else "holdout_dir"
    directory = Path(data_root) / spec[directory_key]
    paths = {
        "train": directory / "train.csv",
        "valid": directory / "valid.csv",
        "test": directory / "test.csv",
        "q_matrix": directory / "Q_matrix.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing r29 data files: {missing}")
    return paths, spec


def verify_fingerprints(
    *,
    spec: dict[str, Any],
    dataset_variant: str,
    paths: dict[str, Path],
) -> dict[str, str]:
    expected = spec["fingerprints"][dataset_variant]
    filenames = {
        "train": "train.csv",
        "valid": "valid.csv",
        "test": "test.csv",
        "q_matrix": "Q_matrix.csv",
    }
    actual = {filename: sha256_file(paths[key]) for key, filename in filenames.items()}
    for filename, digest in actual.items():
        if digest != expected[filename]:
            raise RuntimeError(
                f"Fingerprint mismatch for {filename}: expected={expected[filename]}, actual={digest}"
            )
    return actual


def build_model(
    *, bundle: StepDataBundle, args: argparse.Namespace, recipe: dict[str, Any]
) -> R29CompletionCDM:
    return R29CompletionCDM(
        num_students=bundle.num_students,
        num_exercises=bundle.num_exercises,
        num_concepts=bundle.num_concepts,
        concept_dim=int(recipe["concept_dim"]),
        state_completer=args.state_completer,
        completion_objective=args.completion_objective,
        completion_mask_frac=float(recipe["completion_mask_frac"]),
        evidence_cap=float(recipe["evidence_cap"]),
        readout_dropout=float(recipe["readout_dropout"]),
        max_guess=float(recipe["max_guess"]),
        max_slip=float(recipe["max_slip"]),
    )


def external_axes(
    *, registry: dict[str, Any], dataset: str, spec: dict[str, Any]
) -> dict[str, dict[str, Any] | None]:
    entry = registry["datasets"][dataset]
    if entry.get("source") == "r28_registry":
        return {
            "S": spec["standard_external"],
            "H": spec["holdout_external"],
            "T": spec["holdout_external"],
        }
    return entry["external_axes"]


def external_auc(axis: dict[str, Any], *, stage: str, target: bool = False) -> float:
    key = f"{stage}_{'target_' if target else ''}auc"
    if key in axis:
        return float(axis[key])
    stage_payload = axis.get(stage)
    if isinstance(stage_payload, dict) and "auc" in stage_payload:
        return float(stage_payload["auc"])
    raise KeyError(f"External axis is missing {key}: {axis}")


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("r29 model experiments are fixed to seed=42; multi-seed is forbidden.")
    if args.state_completer not in EXECUTABLE_COMPLETERS:
        raise ValueError(f"{args.state_completer} is not executable; active modes={EXECUTABLE_COMPLETERS}")
    if args.evaluation_stage == "test-confirmation" and args.checkpoint is None:
        raise ValueError("test-confirmation requires a frozen validation checkpoint.")
    if args.evaluation_stage == "validation" and args.checkpoint is not None:
        raise ValueError("validation trains its own checkpoint.")

    registry = load_json(args.registry)
    recipes = load_json(args.recipes)
    recipe = resolve_recipe(args, recipes)
    paths, spec = resolve_data_paths(
        registry=registry,
        dataset=args.dataset,
        dataset_variant=args.dataset_variant,
        data_root=args.data_root,
    )
    fingerprints = verify_fingerprints(spec=spec, dataset_variant=args.dataset_variant, paths=paths)
    bundles = prepare_experiment_split_bundles(
        train_interactions_path=paths["train"],
        valid_interactions_path=paths["valid"],
        test_interactions_path=paths["test"],
        q_matrix_path=paths["q_matrix"],
    )
    set_global_seed(42)
    device = resolve_device(args.device, args.gpus)
    model = build_model(bundle=bundles["train"], args=args, recipe=recipe).to(device)
    initial_hash = model.common_initialization_hash()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    training_result = None
    if args.evaluation_stage == "validation":
        training_result = train_model(
            train_bundle=bundles["train"],
            valid_bundle=bundles["valid"],
            model=model,
            epochs=int(recipe["epochs"]),
            student_batch_size=int(recipe["student_batch_size"]),
            learning_rate=float(recipe["learning_rate"]),
            weight_decay=float(recipe["weight_decay"]),
            training_mode="student_recompute_minibatch",
            device=str(device),
            early_stop_patience=int(recipe["early_stop_patience"]),
            lr_scheduler_patience=int(recipe["scheduler_patience"]),
            lr_scheduler_factor=float(recipe["scheduler_factor"]),
            min_learning_rate=float(recipe["min_learning_rate"]),
            checkpoint_selection_metric="auc",
            checkpoint_path=str(checkpoint_path),
        )
        evaluation_bundle = bundles["valid"]
    else:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
        checkpoint_path = Path(args.checkpoint)
        evaluation_bundle = bundles["test"]

    predictions, mastery = predict_bundle(
        model=model,
        bundle=evaluation_bundle,
        device=device,
        student_batch_size=int(recipe["student_batch_size"]),
        max_target_rows=args.max_target_rows,
    )
    metrics = compute_metrics(predictions["label"].to_numpy(), predictions["prob"].to_numpy())
    target_metrics = None
    slice_rows: list[dict[str, Any]] = []
    if args.dataset_variant == "holdout":
        predictions, slice_rows, target_metrics = target_scope_metrics(
            predictions=predictions,
            bundle=evaluation_bundle,
            target_scope=spec["target_scope"],
            dataset=args.dataset,
            model_name=f"r29_{args.state_completer}",
            minimum_rows=1 if args.max_target_rows is not None else 100,
        )
    prediction_path = output_dir / f"{args.evaluation_stage}_predictions.csv"
    predictions.to_csv(prediction_path, index=False)
    mastery_path = output_dir / f"{args.evaluation_stage}_mastery.npy"
    np.save(mastery_path, mastery)
    doa = compute_doa_from_predictions(
        predictions=predictions, mastery=mastery, bundle=evaluation_bundle
    )

    axes = external_axes(registry=registry, dataset=args.dataset, spec=spec)
    stage = "validation" if args.evaluation_stage == "validation" else "test"
    axis_name = "S" if args.dataset_variant == "standard" else "H"
    axis = axes[axis_name]
    if axis is None:
        raise RuntimeError(f"{args.dataset} external {axis_name} reproduction is not registered.")
    overall_margin = float(metrics["auc"]) - external_auc(axis, stage=stage)
    target_margin = None
    if target_metrics is not None:
        target_axis = axes["T"]
        if target_axis is None:
            raise RuntimeError(f"{args.dataset} external T reproduction is not registered.")
        target_margin = float(target_metrics["auc"]) - external_auc(
            target_axis, stage=stage, target=True
        )
    payload = {
        "schema_version": 2,
        "model": "r29_completion",
        "dataset": args.dataset,
        "dataset_variant": args.dataset_variant,
        "evaluation_stage": args.evaluation_stage,
        "seed": 42,
        "state_completer": args.state_completer,
        "completion_objective": args.completion_objective,
        "architecture_fingerprint": model.architecture_fingerprint,
        "common_initialization_hash": initial_hash,
        "active_parameter_count": model.active_parameter_count(),
        "recipe": recipe,
        "recipe_sha256": sha256_payload(recipe),
        "data_fingerprints": fingerprints,
        "metrics": metrics,
        "target_metrics": target_metrics,
        "overall_external_margin": overall_margin,
        "target_external_margin": target_margin,
        "doa": doa,
        "slice_rows": slice_rows,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": sha256_file(prediction_path),
        "mastery_path": str(mastery_path.resolve()),
        "mastery_sha256": sha256_file(mastery_path),
        "training": (
            {
                "best_epoch": training_result.best_epoch,
                "best_val_auc": training_result.best_val_auc,
                "best_validation_score": training_result.best_validation_score,
                "final_loss": training_result.final_loss,
                "history": training_result.history,
            }
            if training_result is not None
            else None
        ),
    }
    write_json(payload, output_dir / "summary.json")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
