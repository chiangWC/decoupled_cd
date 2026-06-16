from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation-only RCD history-hiding stress test for PyEdmine.")
    parser.add_argument("--pyedmine-root", type=Path, default=Path("/home/xph/jwc/pyedmine"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--job-json", type=Path, required=True)
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--hide-ratios", default="0.8")
    parser.add_argument("--mask-seeds", default="11,13,17")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--per-run-csv", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--evaluate-batch-size", type=int, default=4096)
    parser.add_argument("--use-cpu", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_float_list(value: str) -> list[float]:
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("Expected at least one float value.")
    return result


def parse_int_list(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("Expected at least one integer value.")
    return result


def load_manifest_dataset(manifest_path: Path, dataset_key: str) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    for item in manifest.get("datasets", []):
        if item.get("key") == dataset_key:
            return item
    raise ValueError(f"Dataset key not found in manifest: {dataset_key}")


def mask_train_history_interactions(train_frame: pd.DataFrame, *, hide_ratio: float, seed: int) -> pd.DataFrame:
    rng = torch.Generator()
    rng.manual_seed(int(seed))
    keep_chunks: list[pd.DataFrame] = []
    for _, group in train_frame.groupby("stu_id", sort=False):
        random_values = torch.rand(len(group), generator=rng).numpy()
        keep_mask = random_values >= hide_ratio
        if keep_mask.any():
            keep_chunks.append(group.loc[keep_mask])
    if not keep_chunks:
        return train_frame.iloc[0:0].copy()
    return pd.concat(keep_chunks, ignore_index=True)


def load_id_map(path: Path, raw_column: str) -> dict[str, int]:
    frame = pd.read_csv(path)
    return {str(row[raw_column]): int(row["pyedmine_id"]) for _, row in frame.iterrows()}


def write_pyedmine_cd_file(
    *,
    source_frame: pd.DataFrame,
    output_path: Path,
    user_map: dict[str, int],
    question_map: dict[str, int],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("user_id,question_id,correctness\n")
        for row in source_frame.itertuples(index=False):
            handle.write(
                f"{user_map[str(row.stu_id)]},{question_map[str(row.exer_id)]},{int(row.label)}\n"
            )


def import_pyedmine(pyedmine_root: Path) -> tuple[Any, Any]:
    evaluate_dir = pyedmine_root / "examples" / "cognitive_diagnosis" / "evaluate"
    for path in (str(evaluate_dir), str(pyedmine_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from config import config_dlcd  # type: ignore
    from edmine.dataset.CognitiveDiagnosisDataset import BasicCognitiveDiagnosisDataset  # type: ignore

    return config_dlcd, BasicCognitiveDiagnosisDataset


def run_graph_builders(
    *,
    pyedmine_root: Path,
    setting_name: str,
    dataset_name: str,
    train_file_name: str,
) -> None:
    setting_dir = pyedmine_root / "meta_data" / "dataset" / "settings" / setting_name
    graph_dir = setting_dir / "RCD"
    for prefix in ("k_from_e", "e_from_k", "u_from_e", "e_from_u"):
        target = graph_dir / f"{prefix}_{train_file_name}"
        if target.exists():
            target.unlink()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(pyedmine_root)
    py = sys.executable
    for script in ("build_k_e_graph.py", "build_u_e_graph.py"):
        subprocess.run(
            [
                py,
                str(pyedmine_root / "examples" / "cognitive_diagnosis" / "rcd" / script),
                "--setting_name",
                setting_name,
                "--dataset_name",
                dataset_name,
                "--train_file_name",
                train_file_name,
            ],
            cwd=pyedmine_root,
            env=env,
            check=True,
        )


def create_masked_model_dir(
    *,
    pyedmine_root: Path,
    original_model_dir_name: str,
    setting_name: str,
    masked_train_file_name: str,
    dataset_key: str,
    hide_ratio: float,
    mask_seed: int,
) -> str:
    model_root = pyedmine_root / "saved_models"
    original_dir = model_root / original_model_dir_name
    if not original_dir.exists():
        raise FileNotFoundError(original_dir)
    masked_stem = masked_train_file_name.removesuffix(".txt")
    masked_dir_name = (
        f"RCD@@{setting_name}@@{masked_stem}@@evalhide{hide_ratio:g}_seed{mask_seed}@@{dataset_key}"
    )
    masked_dir = model_root / masked_dir_name
    if masked_dir.exists() or masked_dir.is_symlink():
        if masked_dir.is_symlink() or masked_dir.is_file():
            masked_dir.unlink()
        else:
            shutil.rmtree(masked_dir)
    masked_dir.symlink_to(original_dir, target_is_directory=True)
    return masked_dir_name


def predict_pyedmine_job(
    *,
    pyedmine_root: Path,
    job: dict[str, Any],
    model_dir_name: str,
    source_test: pd.DataFrame,
    batch_size: int,
    use_cpu: bool,
) -> np.ndarray:
    config_dlcd, dataset_cls = import_pyedmine(pyedmine_root)
    params = {
        "model_dir_name": model_dir_name,
        "model_file_name": "saved.ckt",
        "model_name_in_ckt": "best_valid",
        "dataset_name": job["train_file_name"].removesuffix("_train.txt"),
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
    model = global_objects["models"][job["model"]]
    model.eval()
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            probs.append(model.get_predict_score(batch)["predict_score"].detach().cpu().numpy().reshape(-1))
            labels.append(batch["correctness"].detach().cpu().numpy().reshape(-1))
    label_array = np.concatenate(labels, axis=0).astype(int)
    source_labels = source_test["label"].to_numpy(dtype=int)
    if not np.array_equal(source_labels, label_array):
        raise ValueError("Source test labels do not align with PyEdmine test-file order.")
    return np.concatenate(probs, axis=0).astype(float)


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
        "rmse": float(np.sqrt(brier)),
        "brier": brier,
        "ece": compute_ece(labels, probs),
    }


def compute_ece(labels: np.ndarray, probs: np.ndarray, num_bins: int = 10) -> float:
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


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for (dataset, model, hide_ratio), group in frame.groupby(["dataset", "model", "hide_ratio"], sort=False):
        row: dict[str, Any] = {
            "dataset": dataset,
            "model": model,
            "hide_ratio": float(hide_ratio),
            "n": int(len(group)),
            "original_auc": float(group["original_auc"].iloc[0]),
        }
        for col in ["hidden_auc", "delta_auc", "hidden_acc", "hidden_brier", "hidden_ece", "hidden_history_rows"]:
            values = group[col].astype(float)
            row[f"mean_{col}"] = float(values.mean())
            row[f"std_{col}"] = float(values.std(ddof=0)) if len(values) > 1 else 0.0
        summary_rows.append(row)
    return summary_rows


def main() -> None:
    args = parse_args()
    if args.use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    hide_ratios = parse_float_list(args.hide_ratios)
    mask_seeds = parse_int_list(args.mask_seeds)
    dataset = load_manifest_dataset(args.manifest, args.dataset_key)
    job = load_json(args.job_json)
    if job.get("model") != "RCD":
        raise ValueError("This evaluator only supports PyEdmine RCD jobs.")
    source_dir = Path(dataset["source_dir"])
    train_frame = pd.read_csv(source_dir / "train.csv")
    test_frame = pd.read_csv(source_dir / "test.csv")
    labels = test_frame["label"].to_numpy(dtype=np.float64)
    preprocessed_dir = (
        args.pyedmine_root / "meta_data" / "dataset" / "dataset_preprocessed" / dataset["pyedmine_dataset_name"]
    )
    user_map = load_id_map(preprocessed_dir / "user_id_map.csv", "raw_user_id")
    question_map = load_id_map(preprocessed_dir / "question_id_map.csv", "raw_question_id")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    setting_dir = args.pyedmine_root / "meta_data" / "dataset" / "settings" / job["setting_name"]
    rows: list[dict[str, Any]] = []
    original_auc = float(job["test_metrics"]["auc"])

    for hide_ratio in hide_ratios:
        for mask_seed in mask_seeds:
            masked = mask_train_history_interactions(train_frame, hide_ratio=hide_ratio, seed=mask_seed)
            masked_train_file_name = (
                f"{dataset['pyedmine_dataset_name']}_hide{hide_ratio:g}_mask{mask_seed}_train.txt"
            )
            write_pyedmine_cd_file(
                source_frame=masked,
                output_path=setting_dir / masked_train_file_name,
                user_map=user_map,
                question_map=question_map,
            )
            run_graph_builders(
                pyedmine_root=args.pyedmine_root,
                setting_name=job["setting_name"],
                dataset_name=dataset["pyedmine_dataset_name"],
                train_file_name=masked_train_file_name,
            )
            masked_model_dir_name = create_masked_model_dir(
                pyedmine_root=args.pyedmine_root,
                original_model_dir_name=job["model_dir_name"],
                setting_name=job["setting_name"],
                masked_train_file_name=masked_train_file_name,
                dataset_key=args.dataset_key,
                hide_ratio=hide_ratio,
                mask_seed=mask_seed,
            )
            probs = predict_pyedmine_job(
                pyedmine_root=args.pyedmine_root,
                job=job,
                model_dir_name=masked_model_dir_name,
                source_test=test_frame,
                batch_size=args.evaluate_batch_size,
                use_cpu=args.use_cpu,
            )
            metrics = compute_metrics(labels, probs)
            rows.append(
                {
                    "dataset": args.dataset_key,
                    "model": "RCD",
                    "hide_ratio": hide_ratio,
                    "mask_seed": mask_seed,
                    "original_auc": original_auc,
                    "hidden_auc": metrics["auc"],
                    "delta_auc": original_auc - metrics["auc"],
                    "hidden_acc": metrics["acc"],
                    "hidden_brier": metrics["brier"],
                    "hidden_ece": metrics["ece"],
                    "hidden_history_rows": int(len(masked)),
                    "masked_train_file_name": masked_train_file_name,
                    "masked_model_dir_name": masked_model_dir_name,
                }
            )

    summary_rows = summarize(rows)
    payload = {
        "dataset": args.dataset_key,
        "model": "RCD",
        "protocol": "evaluation-only masked train-history graph; checkpoint weights fixed",
        "job_json": str(args.job_json),
        "source_dir": str(source_dir),
        "hide_ratios": hide_ratios,
        "mask_seeds": mask_seeds,
        "rows": rows,
        "summary": summary_rows,
    }
    write_json(args.output, payload)
    per_run_csv = args.per_run_csv or args.output.with_name(f"{args.output.stem}_per_run.csv")
    summary_csv = args.summary_csv or args.output.with_name(f"{args.output.stem}_summary.csv")
    pd.DataFrame(rows).to_csv(per_run_csv, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
