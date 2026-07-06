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
from data.mappings import _normalize_concept_sequence, build_concept_id_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an RCD-style concept transition graph from ordered ASSIST09 data.")
    parser.add_argument(
        "--interactions",
        default="data/assist_09_ordered/train.csv",
        help=(
            "Ordered interaction file used to count transitions. Must be the TRAIN split only: "
            "building from unsplit data leaks valid/test labels into the graph."
        ),
    )
    parser.add_argument(
        "--concept-universe",
        nargs="*",
        default=None,
        help=(
            "Optional interaction CSVs (e.g. train/valid/test) used only to size the concept set. "
            "Without this, concepts absent from --interactions shrink the matrix below the model's "
            "Q-matrix dimension and downstream K x K matmuls misalign."
        ),
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
    universe_frames = [interactions]
    if args.concept_universe:
        universe_frames += [pd.read_csv(path) for path in args.concept_universe]
    # Index the graph in the SAME concept order the training pipeline uses
    # (build_concept_id_map sorts concept tokens lexicographically as strings);
    # raw integer ids would misalign rows against the model's Q-matrix indices.
    concept_id_map = build_concept_id_map(pd.concat(universe_frames, ignore_index=True))
    num_concepts = len(concept_id_map)
    if not args.concept_universe:
        print(
            f"[warn] concept universe ({num_concepts} concepts) inferred from --interactions only; "
            "pass --concept-universe train.csv valid.csv test.csv so it matches the model's Q-matrix dimension.",
        )
    interactions = interactions.copy()
    interactions["cpt_seq"] = interactions["cpt_seq"].map(
        lambda raw: ",".join(str(concept_id_map[token]) for token in _normalize_concept_sequence(raw))
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

    with open(output_dir / "concept_id_map.json", "w", encoding="utf-8") as f:
        json.dump(concept_id_map, f, indent=2, ensure_ascii=False)

    summary = {
        "interactions_path": str(Path(args.interactions).resolve()),
        "concept_universe": [str(Path(p).resolve()) for p in (args.concept_universe or [])],
        "concept_index_order": "pipeline_lexicographic_string_sort",
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
