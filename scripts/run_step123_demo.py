from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import build_response_matrix, prepare_step_data_bundle
from models import DecoupledCDM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal Step 1 + Step 2 + Step 3 forward pass.")
    parser.add_argument("--interactions", required=True, help="Interaction CSV with stu_id/exer_id/cpt_seq/label.")
    parser.add_argument(
        "--q-matrix",
        dest="q_matrix",
        default=None,
        help="Optional Q-matrix CSV. If omitted, it is derived from unique exer_id/cpt_seq pairs in interactions.",
    )
    parser.add_argument("--concept-dim", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--output", default=None, help="Optional JSON path for summary output.")
    return parser.parse_args()


def derive_q_matrix_if_needed(interactions_path: str, q_matrix_path: str | None) -> str:
    if q_matrix_path is not None:
        return q_matrix_path

    interactions = pd.read_csv(interactions_path)
    q_df = interactions[["exer_id", "cpt_seq"]].drop_duplicates().reset_index(drop=True)
    output_path = Path("results") / "derived_q_matrix.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    q_df.to_csv(output_path, index=False)
    return str(output_path)


def main() -> None:
    args = parse_args()
    q_matrix_path = derive_q_matrix_if_needed(args.interactions, args.q_matrix)
    bundle = prepare_step_data_bundle(
        interactions_path=args.interactions,
        q_matrix_path=q_matrix_path,
    )
    response_matrix = build_response_matrix(
        interactions=bundle.interactions,
        student_id_map=bundle.student_id_map,
        exercise_id_map=bundle.exercise_id_map,
    )

    model = DecoupledCDM(
        num_students=bundle.num_students,
        num_exercises=bundle.num_exercises,
        num_concepts=bundle.num_concepts,
        concept_dim=args.concept_dim,
        alpha=args.alpha,
        beta=args.beta,
    )

    with torch.no_grad():
        output = model(
            q_matrix=bundle.q_matrix_tensor,
            concept_graph=bundle.concept_graph,
            student_exercise_mask=bundle.student_exercise_mask,
            response_matrix=response_matrix,
            student_tkc_mask=bundle.student_tkc_mask,
            student_ukc_mask=bundle.student_ukc_mask,
            target_student_ids=bundle.interaction_student_ids,
            target_exercise_ids=bundle.interaction_exercise_ids,
        )

    summary = {
        "num_students": bundle.num_students,
        "num_exercises": bundle.num_exercises,
        "num_concepts": bundle.num_concepts,
        "student_state_shape": list(output.student_state.shape),
        "tkc_states_shape": list(output.tkc_states.shape),
        "ukc_states_shape": list(output.ukc_states.shape),
        "interaction_prob_shape": list(output.probs.shape),
        "students_with_nonempty_ukc": int((bundle.student_ukc_mask.sum(dim=1) > 0).sum().item()),
        "mean_tkc_count": float(bundle.student_tkc_mask.sum(dim=1).float().mean().item()),
        "mean_ukc_count": float(bundle.student_ukc_mask.sum(dim=1).float().mean().item()),
        "mean_cognitive_prob": float(output.cognitive_probs.mean().item()),
        "mean_final_prob": float(output.probs.mean().item()),
        "mean_guess_prob": float(output.guess_probs.mean().item()),
        "mean_slip_prob": float(output.slip_probs.mean().item()),
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
