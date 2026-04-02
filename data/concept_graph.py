from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from .q_matrix import normalize_concept_sequence


def load_concept_graph_csv(path: str | Path) -> torch.Tensor:
    df = pd.read_csv(path, index_col=0)
    return torch.tensor(df.values, dtype=torch.float32)


def save_concept_graph_csv(graph: torch.Tensor, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(graph.detach().cpu().numpy())
    df.to_csv(output_path)


def build_transition_matrices(
    interactions: pd.DataFrame,
    num_concepts: int,
) -> dict[str, torch.Tensor | float | int]:
    """
    Build the RCD-style transition graph from ordered interactions.

    Multi-concept handling:
    - for two adjacent correct interactions,
      every concept in the previous item contributes to every concept
      in the next item, excluding self-pairs.
    """
    counts = torch.zeros(num_concepts, num_concepts, dtype=torch.float32)

    for _, stu_df in interactions.groupby("stu_id", sort=False):
        prev_concepts: list[str] | None = None
        prev_label: int | None = None

        for row in stu_df.itertuples(index=False):
            concepts = normalize_concept_sequence(row.cpt_seq)
            label = int(row.label)

            if prev_concepts is not None and prev_label == 1 and label == 1:
                for src in prev_concepts:
                    src_idx = int(src)
                    for dst in concepts:
                        dst_idx = int(dst)
                        if src_idx != dst_idx:
                            counts[src_idx, dst_idx] += 1.0

            prev_concepts = concepts
            prev_label = label

    row_sums = counts.sum(dim=1, keepdim=True)
    correct_matrix = torch.where(row_sums > 0, counts / row_sums.clamp(min=1.0), torch.zeros_like(counts))
    correct_matrix.fill_diagonal_(0.0)

    min_val = float(correct_matrix.min().item())
    max_val = float(correct_matrix.max().item())
    if max_val > min_val:
        transition_scores = (correct_matrix - min_val) / (max_val - min_val)
    else:
        transition_scores = torch.zeros_like(correct_matrix)
    transition_scores.fill_diagonal_(0.0)

    threshold = float(transition_scores.mean().item() ** 3)
    transition_binary = (transition_scores > threshold).to(dtype=torch.float32)
    transition_binary.fill_diagonal_(0.0)

    prereq = transition_binary * (1.0 - transition_binary.transpose(0, 1))
    similarity = transition_binary * transition_binary.transpose(0, 1)

    # For propagation we use a row-normalized union graph:
    # directed prerequisite edges + symmetric similarity edges.
    propagation_graph = torch.maximum(prereq, similarity)
    propagation_graph.fill_diagonal_(1.0)
    propagation_graph = propagation_graph / propagation_graph.sum(dim=1, keepdim=True).clamp(min=1.0)

    return {
        "counts": counts,
        "correct_matrix": correct_matrix,
        "transition_scores": transition_scores,
        "transition_binary": transition_binary,
        "prerequisite_graph": prereq,
        "similarity_graph": similarity,
        "propagation_graph": propagation_graph,
        "threshold": threshold,
        "num_prerequisite_edges": int(prereq.sum().item()),
        "num_similarity_edges": int(similarity.sum().item()),
    }
