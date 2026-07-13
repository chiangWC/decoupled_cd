from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import StepDataBundle, prepare_experiment_split_bundles
from models import COMPLETER_MODES, COMPLETION_OBJECTIVES, R28CompletionCDM
from scripts.evaluate_coverage_slice import add_target_coverage, compute_slice_rows
from trainers import train_model
from utils import compute_doa, compute_metrics, resolve_device, set_global_seed, write_json


TRAINABLE_COMPLETERS = (
    "relational",
    "direct_prior",
    "capacity_mlp",
    "partial_vae",
    "difficulty_set",
    "difficulty_capacity",
    "poe_ability",
    "poe_capacity",
    "hierarchical_bayes",
    "hierarchical_capacity",
    "cohort_conditioned",
    "cohort_capacity",
    "bipolar_prototype",
    "bipolar_capacity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-first r28 completion trainer.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-variant", choices=["standard", "holdout"], required=True)
    parser.add_argument("--state-completer", choices=COMPLETER_MODES, required=True)
    parser.add_argument("--completion-objective", choices=COMPLETION_OBJECTIVES, default="none")
    parser.add_argument(
        "--evaluation-stage",
        choices=["validation", "test-confirmation"],
        default="validation",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-root", default=os.environ.get("KNOFIELD_DATA_ROOT", "data"))
    parser.add_argument("--registry", default=str(PROJECT_ROOT / "configs/r28_registry.json"))
    parser.add_argument("--recipes", default=str(PROJECT_ROOT / "configs/r28_recipes.json"))
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
    parser.add_argument("--max-target-rows", type=int, default=None, help="Smoke-only prediction cap.")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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


def resolve_data_paths(
    *,
    registry: dict[str, Any],
    dataset: str,
    dataset_variant: str,
    data_root: str | Path,
) -> dict[str, Path]:
    entry = registry["datasets"].get(dataset)
    if entry is None or entry.get("status") != "active":
        raise ValueError(f"Dataset {dataset!r} is not in the active r28 pool.")
    directory_key = "standard_dir" if dataset_variant == "standard" else "holdout_dir"
    directory = Path(data_root) / entry[directory_key]
    paths = {
        "train": directory / "train.csv",
        "valid": directory / "valid.csv",
        "test": directory / "test.csv",
        "q_matrix": directory / "Q_matrix.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing r28 data files: {missing}")
    return paths


def verify_fingerprints(
    *,
    registry: dict[str, Any],
    dataset: str,
    dataset_variant: str,
    paths: dict[str, Path],
) -> dict[str, str]:
    expected = registry["datasets"][dataset]["fingerprints"][dataset_variant]
    actual: dict[str, str] = {}
    names = {"train": "train.csv", "valid": "valid.csv", "test": "test.csv", "q_matrix": "Q_matrix.csv"}
    for key, filename in names.items():
        digest = sha256_file(paths[key])
        actual[filename] = digest
        if digest != expected[filename]:
            raise RuntimeError(
                f"Fingerprint mismatch for {paths[key]}: expected={expected[filename]}, actual={digest}"
            )
    return actual


def build_model(*, bundle: StepDataBundle, args: argparse.Namespace, recipe: dict[str, Any]) -> R28CompletionCDM:
    return R28CompletionCDM(
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
        partial_vae_kl_weight=float(recipe["partial_vae_kl_weight"]),
    )


def _bundle_tensors(bundle: StepDataBundle, device: torch.device) -> dict[str, torch.Tensor | None]:
    return {
        "q_matrix": bundle.q_matrix_tensor.to(device),
        "concept_graph": bundle.concept_graph.to(device),
        "student_exercise_mask": bundle.student_exercise_mask.to(device),
        "response_matrix": bundle.response_matrix_tensor.to(device),
        "student_tkc_mask": bundle.student_tkc_mask.to(device),
        "student_ukc_mask": bundle.student_ukc_mask.to(device),
        "student_concept_evidence": bundle.student_concept_evidence_tensor.to(device),
        "exercise_evidence": bundle.exercise_evidence_tensor.to(device),
        "student_ids": bundle.interaction_student_ids.to(device),
        "exercise_ids": bundle.interaction_exercise_ids.to(device),
        "labels": bundle.interaction_labels.to(device),
    }


def _forward_kwargs(tensors: dict[str, torch.Tensor | None]) -> dict[str, torch.Tensor | None]:
    return {
        "q_matrix": tensors["q_matrix"],
        "concept_graph": tensors["concept_graph"],
        "student_exercise_mask": tensors["student_exercise_mask"],
        "response_matrix": tensors["response_matrix"],
        "student_tkc_mask": tensors["student_tkc_mask"],
        "student_ukc_mask": tensors["student_ukc_mask"],
        "student_concept_evidence": tensors["student_concept_evidence"],
        "exercise_evidence": tensors["exercise_evidence"],
    }


def predict_bundle(
    *,
    model: R28CompletionCDM,
    bundle: StepDataBundle,
    device: torch.device,
    student_batch_size: int,
    max_target_rows: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    model.eval()
    tensors = _bundle_tensors(bundle, device)
    target_count = int(tensors["labels"].numel())
    if max_target_rows is not None:
        target_count = min(target_count, max_target_rows)
    student_ids = tensors["student_ids"][:target_count]
    exercise_ids = tensors["exercise_ids"][:target_count]
    labels = tensors["labels"][:target_count]
    unique_students = torch.unique(student_ids, sorted=True)

    probs = torch.empty(target_count, dtype=torch.float32, device=device)
    mastery = np.full((bundle.num_students, bundle.num_concepts), np.nan, dtype=np.float32)
    base_kwargs = _forward_kwargs(tensors)
    with torch.no_grad():
        for start in range(0, unique_students.numel(), student_batch_size):
            chunk = unique_students[start : start + student_batch_size]
            selected = torch.isin(student_ids, chunk)
            row_indices = torch.nonzero(selected, as_tuple=False).squeeze(-1)
            output = model(
                **base_kwargs,
                target_student_ids=student_ids[row_indices],
                target_exercise_ids=exercise_ids[row_indices],
                use_student_subset=True,
            )
            probs[row_indices] = output.probs
            sorted_chunk = torch.unique(student_ids[row_indices], sorted=True)
            mastery[sorted_chunk.cpu().numpy()] = output.mastery.cpu().numpy()

    frame = bundle.interactions.iloc[:target_count].copy().reset_index(drop=True)
    frame.insert(0, "row_index", np.arange(target_count, dtype=np.int64))
    frame["mapped_student_id"] = student_ids.cpu().numpy()
    frame["mapped_exercise_id"] = exercise_ids.cpu().numpy()
    frame["label"] = labels.cpu().numpy()
    frame["prob"] = probs.cpu().numpy()
    return frame, mastery


def target_scope_metrics(
    *,
    predictions: pd.DataFrame,
    bundle: StepDataBundle,
    target_scope: str,
    dataset: str,
    model_name: str,
    minimum_rows: int = 100,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    enriched = add_target_coverage(
        predictions,
        train_frame=bundle.history_interactions,
        q_matrix=bundle.q_matrix,
    )
    slice_rows = compute_slice_rows(enriched, dataset_name=dataset, model_name=model_name)
    by_scope = {row["scope"]: row for row in slice_rows}
    if target_scope not in by_scope:
        raise RuntimeError(f"Target scope {target_scope!r} is unavailable for {dataset}.")
    target = by_scope[target_scope]
    if int(target["count"]) < minimum_rows:
        raise RuntimeError(f"Target scope {target_scope} has only {target['count']} rows.")
    return enriched, slice_rows, target


def compute_doa_from_predictions(
    *,
    predictions: pd.DataFrame,
    mastery: np.ndarray,
    bundle: StepDataBundle,
) -> dict[str, float | int]:
    concept_lists: list[list[int]] = []
    q_tensor = bundle.q_matrix_tensor
    for exercise_id in predictions["mapped_exercise_id"].to_numpy(dtype=np.int64):
        concept_lists.append(torch.nonzero(q_tensor[int(exercise_id)] > 0, as_tuple=False).squeeze(-1).tolist())
    valid_mastery = np.nan_to_num(mastery, nan=0.5)
    return compute_doa(
        mastery=valid_mastery,
        student_ids=predictions["mapped_student_id"].to_numpy(dtype=np.int64),
        concept_lists=concept_lists,
        labels=predictions["label"].to_numpy(dtype=np.float64),
        min_responses=3,
        seed=42,
    )


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("r28 is fixed to seed=42; multi-seed runs are forbidden.")
    if args.state_completer not in TRAINABLE_COMPLETERS:
        raise ValueError(
            f"{args.state_completer} is not activated. Only {TRAINABLE_COMPLETERS} are executable in the first campaign."
        )
    if args.evaluation_stage == "test-confirmation" and args.checkpoint is None:
        raise ValueError("test-confirmation requires --checkpoint from a frozen validation candidate.")
    if args.evaluation_stage == "validation" and args.checkpoint is not None:
        raise ValueError("validation trains its own checkpoint; do not pass --checkpoint.")

    registry = load_json(args.registry)
    recipes = load_json(args.recipes)
    recipe = resolve_recipe(args, recipes)
    paths = resolve_data_paths(
        registry=registry,
        dataset=args.dataset,
        dataset_variant=args.dataset_variant,
        data_root=args.data_root,
    )
    fingerprints = verify_fingerprints(
        registry=registry,
        dataset=args.dataset,
        dataset_variant=args.dataset_variant,
        paths=paths,
    )
    bundles = prepare_experiment_split_bundles(
        train_interactions_path=paths["train"],
        valid_interactions_path=paths["valid"],
        test_interactions_path=paths["test"],
        q_matrix_path=paths["q_matrix"],
    )

    set_global_seed(42)
    device = resolve_device(args.device, args.gpus)
    model = build_model(bundle=bundles["train"], args=args, recipe=recipe).to(device)
    initial_common_hash = model.common_initialization_hash()
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
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
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
            target_scope=registry["datasets"][args.dataset]["target_scope"],
            dataset=args.dataset,
            model_name=f"r28_{args.state_completer}",
            minimum_rows=1 if args.max_target_rows is not None else 100,
        )

    prediction_path = output_dir / f"{args.evaluation_stage}_predictions.csv"
    predictions.to_csv(prediction_path, index=False)
    mastery_path = output_dir / f"{args.evaluation_stage}_mastery.npy"
    np.save(mastery_path, mastery)
    doa = compute_doa_from_predictions(predictions=predictions, mastery=mastery, bundle=evaluation_bundle)

    external_key = "standard_external" if args.dataset_variant == "standard" else "holdout_external"
    external = registry["datasets"][args.dataset][external_key]
    stage_prefix = "validation" if args.evaluation_stage == "validation" else "test"
    overall_margin = float(metrics["auc"]) - float(external[f"{stage_prefix}_auc"])
    target_margin = None
    if target_metrics is not None:
        target_margin = float(target_metrics["auc"]) - float(external[f"{stage_prefix}_target_auc"])

    payload = {
        "schema_version": 1,
        "dataset": args.dataset,
        "dataset_variant": args.dataset_variant,
        "evaluation_stage": args.evaluation_stage,
        "seed": 42,
        "state_completer": args.state_completer,
        "completion_objective": args.completion_objective,
        "architecture_fingerprint": model.architecture_fingerprint,
        "common_initialization_hash": initial_common_hash,
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
