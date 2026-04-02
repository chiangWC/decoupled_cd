from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import build_transition_matrices, save_concept_graph_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an RCD-style concept transition graph from ordered ASSIST09 data.")
    parser.add_argument(
        "--interactions",
        default="data/assist_09_ordered/data.csv",
        help="Ordered unsplit interaction file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/assist_09_ordered/transition_graph",
        help="Directory to write graph artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    interactions = pd.read_csv(args.interactions)
    num_concepts = int(
        max(
            int(token)
            for text in interactions["cpt_seq"].tolist()
            for token in str(text).split(",")
            if str(token).strip()
        )
        + 1
    )
    outputs = build_transition_matrices(interactions=interactions, num_concepts=num_concepts)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_concept_graph_csv(outputs["propagation_graph"], output_dir / "propagation_graph.csv")
    save_concept_graph_csv(outputs["correct_matrix"], output_dir / "correct_matrix.csv")
    save_concept_graph_csv(outputs["transition_scores"], output_dir / "transition_scores.csv")
    save_concept_graph_csv(outputs["transition_binary"], output_dir / "transition_binary.csv")
    save_concept_graph_csv(outputs["prerequisite_graph"], output_dir / "prerequisite_graph.csv")
    save_concept_graph_csv(outputs["similarity_graph"], output_dir / "similarity_graph.csv")

    summary = {
        "interactions_path": str(Path(args.interactions).resolve()),
        "num_concepts": num_concepts,
        "threshold": outputs["threshold"],
        "num_prerequisite_edges": outputs["num_prerequisite_edges"],
        "num_similarity_edges": outputs["num_similarity_edges"],
        "output_dir": str(output_dir.resolve()),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
