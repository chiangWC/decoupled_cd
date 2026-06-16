from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PyEdmine CD checkpoint predictions by target train-history concept coverage."
    )
    parser.add_argument("--pyedmine-root", type=Path, default=Path("/home/xph/jwc/pyedmine"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--job-json", action="append", required=True, help="PyEdmine job output JSON. Repeat per model.")
    parser.add_argument("--model-name", action="append", default=None, help="Display name matching --job-json.")
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--slice-csv", type=Path, default=None)
    parser.add_argument("--prediction-dir", type=Path, default=None)
    parser.add_argument("--evaluate-batch-size", type=int, default=4096)
    parser.add_argument("--use-cpu", action="store_true")
    args = parser.parse_args()
    if args.model_name is not None and len(args.model_name) != len(args.job_json):
        raise ValueError("--model-name must be repeated the same number of times as --job-json.")
    return args


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_concept_sequence(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    preds = (probs >= 0.5).astype(np.int64)
    try:
        auc = float(roc_auc_score(labels, probs))
    except Exception:
        auc = 0.5
    brier = float(mean_squared_error(labels, probs))
    return {
        "auc": auc,
        "acc": float(accuracy_score(labels, preds)),
        "rmse": float(math.sqrt(brier)),
        "brier": brier,
        "ece": compute_ece(labels, probs),
    }


def compute_ece(labels: np.ndarray, probs: np.ndarray, num_bins: int = 10) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    total = len(labels)
    for index in range(num_bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == num_bins - 1:
            mask = (probs >= lower) & (probs <= upper)
        else:
            mask = (probs >= lower) & (probs < upper)
        count = int(mask.sum())
        if count:
            ece += (count / total) * abs(float(labels[mask].mean()) - float(probs[mask].mean()))
    return float(ece)


def load_manifest_dataset(manifest_path: Path, dataset_key: str) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    for item in manifest.get("datasets", []):
        if item.get("key") == dataset_key:
            return item
    raise ValueError(f"Dataset key not found in manifest: {dataset_key}")


def dataset_name_from_train_file(train_file_name: str) -> str:
    suffix = "_train.txt"
    if not train_file_name.endswith(suffix):
        raise ValueError(f"Cannot derive dataset name from train file: {train_file_name}")
    return train_file_name[: -len(suffix)]


def import_pyedmine(pyedmine_root: Path) -> tuple[Any, Any]:
    evaluate_dir = pyedmine_root / "examples" / "cognitive_diagnosis" / "evaluate"
    for path in (str(evaluate_dir), str(pyedmine_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from config import config_dlcd  # type: ignore
    from edmine.dataset.CognitiveDiagnosisDataset import BasicCognitiveDiagnosisDataset  # type: ignore

    return config_dlcd, BasicCognitiveDiagnosisDataset


def predict_pyedmine_job(
    *,
    pyedmine_root: Path,
    job: dict[str, Any],
    source_test: pd.DataFrame,
    batch_size: int,
    use_cpu: bool,
) -> pd.DataFrame:
    config_dlcd, dataset_cls = import_pyedmine(pyedmine_root)
    model_name = str(job["model"])
    params = {
        "model_dir_name": job["model_dir_name"],
        "model_file_name": "saved.ckt",
        "model_name_in_ckt": "best_valid",
        "dataset_name": dataset_name_from_train_file(str(job["train_file_name"])),
        "test_file_name": job["test_file_name"],
        "evaluate_batch_size": batch_size,
        "evaluate_overall": True,
        "user_cold_start": -1,
        "question_cold_start": -1,
        "save_log": False,
        "use_cpu": use_cpu,
    }
    global_params, global_objects = config_dlcd(params)
    dataset = dataset_cls(
        {
            "setting_name": job["setting_name"],
            "file_name": job["test_file_name"],
            "device": global_params["device"],
        },
        global_objects,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = global_objects["models"][model_name]
    model.eval()

    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            output = model.get_predict_score(batch)["predict_score"].detach().cpu().numpy()
            truth = batch["correctness"].detach().cpu().numpy()
            probs.append(np.asarray(output).reshape(-1))
            labels.append(np.asarray(truth).reshape(-1))
    prob_array = np.concatenate(probs, axis=0)
    label_array = np.concatenate(labels, axis=0)
    if len(source_test) != len(label_array):
        raise ValueError(f"Prediction row count mismatch: source={len(source_test)} predictions={len(label_array)}")
    source_labels = source_test["label"].to_numpy(dtype=np.float64)
    if not np.array_equal(source_labels.astype(int), label_array.astype(int)):
        raise ValueError("Source test labels do not align with PyEdmine test-file order.")

    frame = source_test.reset_index(drop=True).copy()
    frame["label"] = label_array.astype(float)
    frame["prob"] = prob_array.astype(float)
    return frame


def build_student_seen_concepts(train_frame: pd.DataFrame) -> dict[Any, set[str]]:
    seen: dict[Any, set[str]] = {}
    for row in train_frame.itertuples(index=False):
        seen.setdefault(row.stu_id, set()).update(normalize_concept_sequence(row.cpt_seq))
    return seen


def build_exercise_concepts(q_matrix: pd.DataFrame) -> dict[str, set[str]]:
    concepts_by_exercise: dict[str, set[str]] = {}
    for row in q_matrix.itertuples(index=False):
        concepts_by_exercise.setdefault(str(row.exer_id), set()).update(normalize_concept_sequence(row.cpt_seq))
    return concepts_by_exercise


def coverage_bucket(value: float | None) -> str:
    if value is None:
        return "no_concepts"
    if value == 0.0:
        return "zero"
    if value < 0.5:
        return "low"
    if value < 1.0:
        return "partial"
    return "full"


def add_target_coverage(frame: pd.DataFrame, *, train_frame: pd.DataFrame, q_matrix: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    student_seen = build_student_seen_concepts(train_frame)
    exercise_concepts = build_exercise_concepts(q_matrix)
    coverages: list[float | None] = []
    concept_counts: list[int] = []
    seen_counts: list[int] = []
    for row in enriched.itertuples(index=False):
        concepts = exercise_concepts.get(str(row.exer_id), set(normalize_concept_sequence(row.cpt_seq)))
        concept_count = len(concepts)
        seen_count = len(concepts & student_seen.get(row.stu_id, set()))
        concept_counts.append(concept_count)
        seen_counts.append(seen_count)
        coverages.append(None if concept_count == 0 else seen_count / concept_count)
    enriched["target_concept_count"] = concept_counts
    enriched["target_seen_concept_count"] = seen_counts
    enriched["target_coverage"] = coverages
    enriched["coverage_bucket"] = [coverage_bucket(value) for value in coverages]
    enriched["coverage_group"] = [
        "low_coverage" if value is not None and value < 0.5 else "full_coverage" if value == 1.0 else "other"
        for value in coverages
    ]
    return enriched


def metric_row(frame: pd.DataFrame, *, dataset_name: str, model_name: str, scope: str) -> dict[str, Any]:
    metrics = compute_metrics(frame["label"].to_numpy(), frame["prob"].to_numpy())
    return {
        "dataset": dataset_name,
        "model": model_name,
        "scope": scope,
        "count": int(len(frame)),
        "label_rate": float(frame["label"].mean()) if len(frame) else None,
        "mean_prob": float(frame["prob"].mean()) if len(frame) else None,
        **metrics,
    }


def compute_slice_rows(frame: pd.DataFrame, *, dataset_name: str, model_name: str) -> list[dict[str, Any]]:
    rows = [metric_row(frame, dataset_name=dataset_name, model_name=model_name, scope="overall")]
    for bucket in ["zero", "low", "partial", "full", "no_concepts"]:
        group = frame[frame["coverage_bucket"] == bucket]
        if len(group):
            rows.append(metric_row(group, dataset_name=dataset_name, model_name=model_name, scope=f"bucket:{bucket}"))
    for group_name in ["low_coverage", "full_coverage"]:
        group = frame[frame["coverage_group"] == group_name]
        if len(group):
            rows.append(metric_row(group, dataset_name=dataset_name, model_name=model_name, scope=group_name))
    return rows


def build_summary_row(slice_rows: list[dict[str, Any]], *, dataset_name: str, model_name: str) -> dict[str, Any]:
    by_scope = {row["scope"]: row for row in slice_rows}
    overall = by_scope.get("overall", {})
    low = by_scope.get("low_coverage", {})
    full = by_scope.get("full_coverage", {})
    return {
        "dataset": dataset_name,
        "model": model_name,
        "overall_auc": overall.get("auc"),
        "overall_brier": overall.get("brier"),
        "overall_ece": overall.get("ece"),
        "low_coverage_count": low.get("count", 0),
        "low_coverage_auc": low.get("auc"),
        "full_coverage_count": full.get("count", 0),
        "full_coverage_auc": full.get("auc"),
        "coverage_gap": (
            float(full["auc"]) - float(low["auc"]) if "auc" in full and "auc" in low else None
        ),
        "low_coverage_brier": low.get("brier"),
        "low_coverage_ece": low.get("ece"),
    }


def main() -> None:
    args = parse_args()
    dataset = load_manifest_dataset(args.manifest, args.dataset_key)
    source_dir = Path(dataset["source_dir"])
    train_frame = pd.read_csv(source_dir / "train.csv")
    test_frame = pd.read_csv(source_dir / "test.csv")
    q_matrix = pd.read_csv(source_dir / "Q_matrix.csv")

    model_names = args.model_name or [Path(path).stem for path in args.job_json]
    all_slices: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    prediction_dir = args.prediction_dir
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    for job_path_text, model_name in zip(args.job_json, model_names, strict=True):
        job_path = Path(job_path_text)
        job = load_json(job_path)
        predictions = predict_pyedmine_job(
            pyedmine_root=args.pyedmine_root,
            job=job,
            source_test=test_frame,
            batch_size=args.evaluate_batch_size,
            use_cpu=args.use_cpu,
        )
        enriched = add_target_coverage(predictions, train_frame=train_frame, q_matrix=q_matrix)
        if prediction_dir is not None:
            enriched.to_csv(prediction_dir / f"{args.dataset_key}__{model_name.replace(' ', '_')}_predictions.csv", index=False)
        slice_rows = compute_slice_rows(enriched, dataset_name=args.dataset_key, model_name=model_name)
        all_slices.extend(slice_rows)
        summaries.append(build_summary_row(slice_rows, dataset_name=args.dataset_key, model_name=model_name))
        runs.append(
            {
                "job_json": str(job_path),
                "model": model_name,
                "source_dir": str(source_dir),
                "test_rows": int(len(test_frame)),
                "coverage_bucket_counts": {
                    str(key): int(value) for key, value in enriched["coverage_bucket"].value_counts().items()
                },
                "original_job_metrics": job.get("test_metrics"),
            }
        )

    output = {
        "dataset": args.dataset_key,
        "source_dir": str(source_dir),
        "low_coverage_definition": "target_coverage < 0.5, including coverage == 0",
        "runs": runs,
        "slices": all_slices,
        "summary": summaries,
    }
    write_json(args.output, output)
    slice_csv = args.slice_csv or args.output.with_name(f"{args.output.stem}_slices.csv")
    summary_csv = args.summary_csv or args.output.with_name(f"{args.output.stem}_summary.csv")
    pd.DataFrame(all_slices).to_csv(slice_csv, index=False)
    pd.DataFrame(summaries).to_csv(summary_csv, index=False)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
