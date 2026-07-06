from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_coverage_slice import add_target_coverage, compute_slice_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Coverage-slice an external baseline's test predictions (prob,label CSV in "
            "test-row order) with the same bucket logic as evaluate_coverage_slice."
        )
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split-dir", required=True, help="Directory with train/valid/test.csv and Q_matrix.csv.")
    parser.add_argument("--predictions", action="append", required=True, help="prob,label CSV. Repeat per model.")
    parser.add_argument("--model-name", action="append", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_dir = Path(args.split_dir)
    test_frame = pd.read_csv(split_dir / "test.csv")
    train_frame = pd.read_csv(split_dir / "train.csv")
    q_matrix = pd.read_csv(split_dir / "Q_matrix.csv")

    all_rows = []
    for pred_path, model_name in zip(args.predictions, args.model_name, strict=True):
        preds = pd.read_csv(pred_path)
        if len(preds) != len(test_frame):
            raise ValueError(
                f"{model_name}: prediction rows ({len(preds)}) != test rows ({len(test_frame)})."
            )
        if not (preds["label"].round().astype(int).values == test_frame["label"].astype(int).values).all():
            raise ValueError(f"{model_name}: label sequence mismatch — predictions are not in test-row order.")
        frame = test_frame.reset_index(drop=True).copy()
        frame["label"] = preds["label"].values
        frame["prob"] = preds["prob"].values
        enriched = add_target_coverage(frame, train_frame=train_frame, q_matrix=q_matrix)
        all_rows.extend(
            compute_slice_rows(enriched, model_name=model_name, dataset_name=args.dataset_name)
        )

    out = pd.DataFrame(all_rows)
    out.to_csv(args.output_csv, index=False)
    print(out[out.scope.isin(["overall", "bucket:zero", "bucket:low", "low_coverage"])].to_string(index=False))


if __name__ == "__main__":
    main()
