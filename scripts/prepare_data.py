from __future__ import annotations

import argparse
from pprint import pprint

from data import prepare_data_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare interaction and Q-matrix data.")
    parser.add_argument("--interactions", required=True, help="Path to the interaction CSV file.")
    parser.add_argument("--q-matrix", required=True, dest="q_matrix", help="Path to the Q-matrix CSV file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = prepare_data_bundle(
        interactions_path=args.interactions,
        q_matrix_path=args.q_matrix,
    )

    summary = {
        "num_rows_interactions": len(bundle["interactions"]),
        "num_rows_q_matrix": len(bundle["q_matrix"]),
        "num_students": bundle["num_students"],
        "num_exercises": bundle["num_exercises"],
        "num_concepts": bundle["num_concepts"],
    }
    pprint(summary)


if __name__ == "__main__":
    main()
