from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.pool_protocol import sha256_file
from scripts.audit_static_metadata_signal import (
    MIN_SWAPS_PER_EDGE,
    RelationEdge,
    SPLIT_SEED,
    _degree_signature,
    _load_protocols,
    _stable_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize canonical CPC relation JSON."
    )
    parser.add_argument(
        "--dataset",
        choices=["assist09", "nips34", "junyi", "xes3g5m"],
        required=True,
    )
    parser.add_argument(
        "--variant",
        choices=["full", "rewire_0", "rewire_1", "rewire_2"],
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--static-admission-audit")
    parser.add_argument("--assist09-protocol")
    parser.add_argument("--assist09-raw")
    parser.add_argument("--nips-protocol")
    parser.add_argument("--nips-question-metadata")
    parser.add_argument("--nips-subject-metadata")
    parser.add_argument("--junyi-protocol")
    parser.add_argument("--junyi-official-log")
    parser.add_argument("--junyi-directed")
    parser.add_argument("--junyi-undirected")
    parser.add_argument("--junyi-source-repo")
    parser.add_argument("--xes-admission-audit")
    return parser.parse_args()


def _edge_payload(edge: RelationEdge) -> dict[str, Any]:
    return {
        "relation": edge.relation,
        "source": edge.source,
        "target": edge.target,
        "directed": edge.directed,
    }


def _edge_sha256(edges: list[RelationEdge]) -> str:
    digest = hashlib.sha256()
    for edge in sorted(edges, key=RelationEdge.canonical):
        digest.update(
            json.dumps(
                _edge_payload(edge),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _require(args: argparse.Namespace, names: tuple[str, ...]) -> None:
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        raise ValueError(
            "Missing source arguments: " + ", ".join(missing)
        )


def _load_edges(
    args: argparse.Namespace,
) -> tuple[list[RelationEdge], dict[str, Any]]:
    if args.dataset == "xes3g5m":
        _require(args, ("xes_admission_audit",))
        path = Path(args.xes_admission_audit)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("admitted") is not True:
            raise RuntimeError("XES relation source is not admitted.")
        edges = [
            RelationEdge(
                relation=str(edge["relation"]),
                source=str(edge["source"]),
                target=str(edge["target"]),
                directed=bool(edge["directed"]),
            )
            for edge in payload["metadata_edges"]
        ]
        return edges, {
            "admission_path": str(path.resolve()),
            "admission_sha256": sha256_file(path),
        }

    names = (
        "static_admission_audit",
        "assist09_protocol",
        "assist09_raw",
        "nips_protocol",
        "nips_question_metadata",
        "nips_subject_metadata",
        "junyi_protocol",
        "junyi_official_log",
        "junyi_directed",
        "junyi_undirected",
        "junyi_source_repo",
    )
    _require(args, names)
    protocols = _load_protocols(args)
    protocol = protocols[args.dataset]
    return list(protocol.metadata_edges), protocol.audit


def _rewire_pairs(
    edges: list[RelationEdge],
    *,
    rng: np.random.Generator,
    directed: bool,
) -> tuple[list[RelationEdge], int]:
    pairs = [
        (edge.source, edge.target)
        if directed else tuple(sorted((edge.source, edge.target)))
        for edge in edges
    ]
    occupied = set(pairs)
    target_swaps = MIN_SWAPS_PER_EDGE * len(edges)
    accepted = 0
    attempts = 0
    maximum_attempts = max(10_000, 300 * target_swaps)
    while accepted < target_swaps and attempts < maximum_attempts:
        attempts += 1
        first, second = rng.choice(len(pairs), size=2, replace=False)
        a, b = pairs[first]
        c, d = pairs[second]
        if not directed and rng.integers(0, 2):
            c, d = d, c
        if directed:
            proposed = ((a, d), (c, b))
            if a == c or b == d or a == d or c == b:
                continue
        else:
            if len({a, b, c, d}) < 4:
                continue
            proposed = (
                tuple(sorted((a, d))),
                tuple(sorted((c, b))),
            )
        if proposed[0] == proposed[1]:
            continue
        old_first = pairs[first]
        old_second = pairs[second]
        occupied.remove(old_first)
        occupied.remove(old_second)
        if proposed[0] in occupied or proposed[1] in occupied:
            occupied.add(old_first)
            occupied.add(old_second)
            continue
        pairs[first], pairs[second] = proposed
        occupied.add(proposed[0])
        occupied.add(proposed[1])
        accepted += 1
    if accepted < target_swaps:
        raise RuntimeError(
            f"Fast rewire accepted {accepted}/{target_swaps} swaps."
        )
    relation = edges[0].relation
    return [
        RelationEdge(relation, source, target, directed)
        for source, target in pairs
    ], accepted


def _fast_rewire_metadata(
    edges: list[RelationEdge],
    *,
    dataset: str,
    replicate: int,
) -> tuple[list[RelationEdge], dict[str, Any]]:
    by_relation: dict[str, list[RelationEdge]] = defaultdict(list)
    for edge in edges:
        by_relation[edge.relation].append(edge)
    shuffled: list[RelationEdge] = []
    accepted_by_relation: dict[str, int] = {}
    for relation, relation_edges in sorted(by_relation.items()):
        directions = {edge.directed for edge in relation_edges}
        if len(directions) != 1:
            raise RuntimeError(f"Mixed directionality in {relation}.")
        directed = relation_edges[0].directed
        rng = np.random.default_rng(
            _stable_seed(SPLIT_SEED, dataset, replicate, relation)
        )
        rewired, accepted = _rewire_pairs(
            relation_edges,
            rng=rng,
            directed=directed,
        )
        shuffled.extend(rewired)
        accepted_by_relation[relation] = accepted
    if _degree_signature(edges) != _degree_signature(shuffled):
        raise RuntimeError("Fast rewire changed a degree sequence.")
    if len({edge.canonical() for edge in shuffled}) != len(shuffled):
        raise RuntimeError("Fast rewire produced duplicate edges.")
    if any(edge.source == edge.target for edge in shuffled):
        raise RuntimeError("Fast rewire produced a self-loop.")
    return sorted(shuffled, key=RelationEdge.canonical), {
        "replicate": replicate,
        "edge_count": len(shuffled),
        "accepted_swaps_by_relation": accepted_by_relation,
        "minimum_swaps_per_edge": MIN_SWAPS_PER_EDGE,
        "degree_sequence_exact": True,
        "self_loops": 0,
        "duplicate_edges": 0,
        "implementation": "constant_time_membership_v1",
    }


def main() -> None:
    args = parse_args()
    edges, source_audit = _load_edges(args)
    rewire_audit: dict[str, Any] | None = None
    if args.variant != "full":
        replicate = int(args.variant.rsplit("_", 1)[1])
        rewired, rewire_audit = _fast_rewire_metadata(
            edges,
            dataset=args.dataset,
            replicate=replicate,
        )
        edges = list(rewired)
    ordered = sorted(edges, key=RelationEdge.canonical)
    payload = {
        "schema_version": 1,
        "dataset": args.dataset,
        "variant": args.variant,
        "source_audit": source_audit,
        "rewire_audit": rewire_audit,
        "metadata_edge_sha256": _edge_sha256(ordered),
        "metadata_edges": [_edge_payload(edge) for edge in ordered],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output.resolve()),
        "sha256": sha256_file(output),
        "metadata_edges": len(ordered),
        "variant": args.variant,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
