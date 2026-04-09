from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import apply_dataset_defaults
from data import prepare_experiment_split_bundles
from models import DecoupledCDM
from trainers import evaluate_model
from utils import resolve_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation on train/valid/test splits.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--train-interactions", default=None)
    parser.add_argument("--valid-interactions", default=None)
    parser.add_argument("--test-interactions", default=None)
    parser.add_argument("--q-matrix", default=None)
    parser.add_argument("--concept-graph", default=None)
    parser.add_argument("--prerequisite-graph", default=None, help="Legacy dual-graph ablation input.")
    parser.add_argument("--similarity-graph", default=None, help="Legacy dual-graph ablation input.")
    parser.add_argument(
        "--graph-mode",
        choices=["single", "dual"],
        default="single",
        help="Use 'single' for the current mainline. 'dual' is retained only for historical ablations.",
    )
    parser.add_argument("--concept-dim", type=int, default=16)
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Legacy fusion-prior knob retained for reproducibility; the current mainline uses adaptive fusion.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Legacy fusion-prior knob retained for reproducibility; the current mainline uses adaptive fusion.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--output", default="results/eval_summary.json")
    args = parser.parse_args()
    return apply_dataset_defaults(args, parser)


def derive_q_matrix_from_splits_if_needed(train_path: str, valid_path: str, test_path: str, q_matrix_path: str | None) -> str:
    if q_matrix_path is not None:
        return q_matrix_path
    import pandas as pd

    frames = [
        pd.read_csv(train_path, usecols=["exer_id", "cpt_seq"]),
        pd.read_csv(valid_path, usecols=["exer_id", "cpt_seq"]),
        pd.read_csv(test_path, usecols=["exer_id", "cpt_seq"]),
    ]
    q_df = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    output_path = Path("results") / "derived_q_matrix_eval.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    q_df.to_csv(output_path, index=False)
    return str(output_path)


def main() -> None:
    args = parse_args()
    if args.graph_mode == "dual" and (args.prerequisite_graph is None or args.similarity_graph is None):
        raise ValueError("Dual graph mode requires both --prerequisite-graph and --similarity-graph.")
    if not all([args.train_interactions, args.valid_interactions, args.test_interactions]):
        raise ValueError("Evaluation requires --train-interactions, --valid-interactions, and --test-interactions.")

    q_matrix_path = derive_q_matrix_from_splits_if_needed(
        args.train_interactions,
        args.valid_interactions,
        args.test_interactions,
        args.q_matrix,
    )
    bundles = prepare_experiment_split_bundles(
        train_interactions_path=args.train_interactions,
        valid_interactions_path=args.valid_interactions,
        test_interactions_path=args.test_interactions,
        q_matrix_path=q_matrix_path,
        concept_graph_path=args.concept_graph,
        prerequisite_graph_path=args.prerequisite_graph if args.graph_mode == "dual" else None,
        similarity_graph_path=args.similarity_graph if args.graph_mode == "dual" else None,
    )
    device = str(resolve_device(args.device, args.gpus))
    model = DecoupledCDM(
        num_students=bundles["train"].num_students,
        num_exercises=bundles["train"].num_exercises,
        num_concepts=bundles["train"].num_concepts,
        concept_dim=args.concept_dim,
        alpha=args.alpha,
        beta=args.beta,
    )

    payload = {
        "device": device,
        "graph_mode": args.graph_mode,
        "train_metrics": evaluate_model(bundle=bundles["train"], model=model, device=device),
        "valid_metrics": evaluate_model(bundle=bundles["valid"], model=model, device=device),
        "test_metrics": evaluate_model(bundle=bundles["test"], model=model, device=device),
    }
    write_json(payload, args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
