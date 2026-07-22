from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
import torch
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export row-aligned predictions from a saved PyEdmine CD model."
    )
    parser.add_argument("--pyedmine-root", type=Path, required=True)
    parser.add_argument("--model-dir-name", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--file-name", required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    evaluate_dir = (
        args.pyedmine_root / "examples" / "cognitive_diagnosis" / "evaluate"
    )
    sys.path.insert(0, str(args.pyedmine_root))
    sys.path.insert(0, str(evaluate_dir))

    from config import config_dlcd
    from edmine.dataset.CognitiveDiagnosisDataset import (
        BasicCognitiveDiagnosisDataset,
    )
    from utils import get_model_info

    if args.model_dir_name.startswith("HyperCD@@"):
        module = importlib.import_module(
            "edmine.model.cognitive_diagnosis_model.HyperCD"
        )
        if module.MODEL_NAME != "HyperCDF":
            raise RuntimeError(
                f"Unexpected HyperCD model key: {module.MODEL_NAME}"
            )
        module.MODEL_NAME = "HyperCD"
        from edmine.model.registry import MODEL_REGISTRY

        MODEL_REGISTRY["HyperCD"] = module.HyperCD
    local_params = {
        "model_dir_name": args.model_dir_name,
        "model_file_name": "saved.ckt",
        "model_name_in_ckt": "best_valid",
        "dataset_name": args.dataset_name,
        "evaluate_batch_size": args.batch_size,
        "evaluate_overall": True,
        "user_cold_start": -1,
        "question_cold_start": -1,
        "save_log": False,
        "use_cpu": args.cpu,
    }
    global_params, global_objects = config_dlcd(local_params)
    dataset = BasicCognitiveDiagnosisDataset(
        {
            "setting_name": get_model_info(args.model_dir_name)[1],
            "file_name": args.file_name,
            "device": global_params["device"],
        },
        global_objects,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model_name = get_model_info(args.model_dir_name)[0]
    model = global_objects["models"][model_name]
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            probabilities.append(
                model.get_predict_score(batch)["predict_score"]
                .detach()
                .cpu()
                .numpy()
            )
            labels.append(batch["correctness"].detach().cpu().numpy())
    probability = np.concatenate(probabilities).astype(float)
    label = np.concatenate(labels).astype(int)
    source = pd.read_csv(args.source_csv).reset_index(drop=True)
    expected = pd.to_numeric(source["label"], errors="raise").to_numpy(int)
    if len(source) != len(probability) or not np.array_equal(expected, label):
        raise RuntimeError(
            f"PyEdmine prediction order does not align with {args.source_csv}."
        )
    if not np.isfinite(probability).all():
        raise RuntimeError("PyEdmine emitted non-finite probabilities.")
    source["prob"] = probability
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    source.to_csv(args.output_csv, index=False)
    metrics = {
        "auc": float(roc_auc_score(label, probability)),
        "acc": float(accuracy_score(label, probability >= 0.5)),
        "rmse": float(mean_squared_error(label, probability) ** 0.5),
    }
    payload: dict[str, object] = {
        "rows": len(source),
        "row_aligned": True,
        "metrics": metrics,
        "output": str(args.output_csv.resolve()),
    }
    print(json.dumps(payload, sort_keys=True))
    return payload


if __name__ == "__main__":
    run(parse_args())
