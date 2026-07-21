from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.pool_protocol import sha256_file


XES_REPO_COMMIT = "b5a35caf223c5c5d558da2fb836087fa6af2bef5"
DFCD_REPO_COMMIT = "6a3127c7dda4aab3a92077634803b48fc2b3e716"
FROZEN_SHA256 = {
    "official_archive": (
        "62d145bd995248f78726a0b6ab69612cf418e3118d0fe01bfe6f4c61b5072f73"
    ),
    "official_questions": (
        "57a3a63ef924e1f10bc83e673213841be4885f5ad52f0e606775a4fc030bc0d4"
    ),
    "official_routes": (
        "a3175aa6d4ff2d350315273b87c3175d4fa72d252d45815a56082adf6c4a27c8"
    ),
    "dfcd_total": (
        "997dee1b031274219f6145fb775af6c29db711ee2bf162b0e810abece654d497"
    ),
    "dfcd_q": (
        "f10d7ebf3c66ecb22790a35923ce76b995ac84d7420658cdab9bde20df53f98a"
    ),
    "dfcd_preprocess": (
        "26941d146a1b36d5f8746da5e94b5b3bc5582ce73fe5ce5d001d7b3b4b769222"
    ),
    "dfcd_source_zip": (
        "6bdda08dec262351ac8833b56fac53490f1344cfaf6da2bbc1126a2ececf9578"
    ),
    "reconstructed_map": (
        "56b27c2dd005438220a3b0827b14e981f35bc0822f201f10aae7cbb9858482ae"
    ),
    "protocol_data": (
        "5bb8910793c374039053f6b984c5217ac28fa7e47417cc9a51f59ec778e550f8"
    ),
    "protocol_train": (
        "27d0f8715176f83040048498f345ca1ca3af9a1dbaef9ea3083d1821f6621740"
    ),
    "protocol_q": (
        "13965c21cc2728281df235877805fcbf137bf851a4f9e232de6ec14620546df7"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Outcome-blind admission of official XES KC routes."
    )
    parser.add_argument("--official-xes-repo", required=True)
    parser.add_argument("--official-archive", required=True)
    parser.add_argument("--official-questions", required=True)
    parser.add_argument("--official-routes", required=True)
    parser.add_argument("--dfcd-repo", required=True)
    parser.add_argument("--dfcd-total", required=True)
    parser.add_argument("--dfcd-q", required=True)
    parser.add_argument("--dfcd-preprocess", required=True)
    parser.add_argument("--dfcd-source-zip", required=True)
    parser.add_argument("--reconstructed-map", required=True)
    parser.add_argument("--protocol-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _git_head(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def _verify_inputs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    protocol = Path(args.protocol_dir)
    paths = {
        "official_archive": Path(args.official_archive),
        "official_questions": Path(args.official_questions),
        "official_routes": Path(args.official_routes),
        "dfcd_total": Path(args.dfcd_total),
        "dfcd_q": Path(args.dfcd_q),
        "dfcd_preprocess": Path(args.dfcd_preprocess),
        "dfcd_source_zip": Path(args.dfcd_source_zip),
        "reconstructed_map": Path(args.reconstructed_map),
        "protocol_data": protocol / "data.csv",
        "protocol_train": protocol / "train.csv",
        "protocol_q": protocol / "Q_matrix.csv",
    }
    reports = {}
    for name, path in paths.items():
        digest = sha256_file(path)
        if digest != FROZEN_SHA256[name]:
            raise RuntimeError(
                f"Frozen hash mismatch for {name}: {digest} != "
                f"{FROZEN_SHA256[name]}"
            )
        reports[name] = {
            "path": str(path.resolve()),
            "sha256": digest,
            "size_bytes": path.stat().st_size,
        }
    heads = {
        "official_xes": _git_head(Path(args.official_xes_repo)),
        "dfcd": _git_head(Path(args.dfcd_repo)),
    }
    if heads != {
        "official_xes": XES_REPO_COMMIT,
        "dfcd": DFCD_REPO_COMMIT,
    }:
        raise RuntimeError(f"Frozen Git commit mismatch: {heads}")
    reports["git_commits"] = heads
    return reports


def _normalize_node(value: object) -> str:
    return str(value).strip()


def _parse_current_q(path: Path) -> dict[int, int]:
    frame = pd.read_csv(
        path,
        usecols=["exer_id", "cpt_seq"],
        dtype={"exer_id": int, "cpt_seq": str},
    )
    output = {}
    for row in frame.itertuples(index=False):
        text = str(row.cpt_seq).strip().strip("[]")
        parts = [
            part.strip()
            for part in text.replace("_", ",").split(",")
            if part.strip()
        ]
        if len(parts) != 1:
            raise RuntimeError(
                f"XES current item {row.exer_id} is not single-concept."
            )
        item = int(row.exer_id)
        concept = int(parts[0])
        if item in output and output[item] != concept:
            raise RuntimeError(f"Conflicting current Q row for item {item}.")
        output[item] = concept
    return output


def _pair_counter(path: Path, *, header: bool) -> Counter[tuple[int, int]]:
    if header:
        frame = pd.read_csv(
            path,
            usecols=["stu_id", "exer_id"],
            dtype={"stu_id": int, "exer_id": int},
        )
    else:
        frame = pd.read_csv(
            path,
            header=None,
            usecols=[0, 1],
            names=["stu_id", "exer_id"],
            dtype={"stu_id": float, "exer_id": float},
        )
        frame = frame.astype(int)
    return Counter(
        zip(
            frame["stu_id"].astype(int),
            frame["exer_id"].astype(int),
            strict=True,
        )
    )


def _aux_node(label: str) -> str:
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()[:20]
    return f"xes_aux:{digest}"


def materialize_relations(
    *,
    current_q: dict[int, int],
    reverse_question_map: dict[int, int],
    questions: dict[str, Any],
    route_labels: set[str],
) -> dict[str, Any]:
    if set(current_q) != set(reverse_question_map):
        raise RuntimeError("Current and reconstructed item maps differ.")
    leaf_by_concept: dict[int, str] = {}
    item_to_official: dict[int, int] = {}
    routes_by_concept: dict[int, set[tuple[str, ...]]] = defaultdict(set)
    for item, concept in sorted(current_q.items()):
        official = int(reverse_question_map[item])
        if str(official) not in questions:
            raise RuntimeError(f"Official question {official} is absent.")
        routes = questions[str(official)].get("kc_routes", [])
        if not routes:
            raise RuntimeError(f"Official question {official} has no KC route.")
        route = tuple(
            _normalize_node(part)
            for part in str(routes[0]).split("----")
            if _normalize_node(part)
        )
        if not route:
            raise RuntimeError(f"Official question {official} has an empty route.")
        missing = set(route) - route_labels
        if missing:
            raise RuntimeError(
                f"Route nodes absent from kc_routes_map: {sorted(missing)}"
            )
        leaf = route[-1]
        previous = leaf_by_concept.setdefault(concept, leaf)
        if previous != leaf:
            raise RuntimeError(
                f"Current concept {concept} maps to {previous!r} and {leaf!r}."
            )
        item_to_official[item] = official
        routes_by_concept[concept].add(route)
    concept_by_leaf: dict[str, int] = {}
    for concept, leaf in leaf_by_concept.items():
        if leaf in concept_by_leaf and concept_by_leaf[leaf] != concept:
            raise RuntimeError(
                f"Leaf {leaf!r} maps to multiple current concepts."
            )
        concept_by_leaf[leaf] = concept

    node_labels: dict[str, str] = {}
    edges: set[tuple[str, str]] = set()

    def node_id(label: str) -> str:
        if label in concept_by_leaf:
            node = f"concept:{concept_by_leaf[label]}"
        else:
            node = _aux_node(label)
        existing = node_labels.setdefault(node, label)
        if existing != label:
            raise RuntimeError("Auxiliary node hash collision.")
        return node

    for routes in routes_by_concept.values():
        for route in routes:
            nodes = [node_id(label) for label in route]
            edges.update(zip(nodes[:-1], nodes[1:], strict=True))
    if any(source == target for source, target in edges):
        raise RuntimeError("Materialized XES tree contains a self-loop.")
    incident_current = {
        int(node.split(":", 1)[1])
        for edge in edges
        for node in edge
        if node.startswith("concept:")
    }
    if incident_current != set(leaf_by_concept):
        missing = sorted(set(leaf_by_concept) - incident_current)
        raise RuntimeError(
            f"Current concepts absent from materialized paths: {missing}"
        )
    return {
        "item_to_official_question": {
            str(item): official
            for item, official in sorted(item_to_official.items())
        },
        "concept_to_official_leaf": {
            str(concept): leaf
            for concept, leaf in sorted(leaf_by_concept.items())
        },
        "node_labels": dict(sorted(node_labels.items())),
        "metadata_edges": [
            {
                "relation": "xes_kc_tree",
                "source": source,
                "target": target,
                "directed": True,
            }
            for source, target in sorted(edges)
        ],
        "audit": {
            "mapped_items": len(item_to_official),
            "mapped_concepts": len(leaf_by_concept),
            "unique_leaf_labels": len(concept_by_leaf),
            "concepts_with_multiple_parent_paths": sum(
                len(routes) > 1 for routes in routes_by_concept.values()
            ),
            "maximum_paths_per_concept": max(
                map(len, routes_by_concept.values())
            ),
            "materialized_nodes": len(node_labels),
            "auxiliary_nodes": sum(
                node.startswith("xes_aux:") for node in node_labels
            ),
            "tree_edges": len(edges),
            "incident_current_concepts": len(incident_current),
            "self_loops": 0,
            "duplicate_edges": 0,
        },
    }


def main() -> None:
    args = parse_args()
    fingerprints = _verify_inputs(args)
    with Path(args.reconstructed_map).open("rb") as handle:
        mapping = pickle.load(handle)
    required = {"reverse_question_map", "question_map", "student_map"}
    if not required <= set(mapping):
        raise RuntimeError(f"Reconstructed map lacks {required - set(mapping)}")
    reverse = {
        int(item): int(question)
        for item, question in mapping["reverse_question_map"].items()
    }
    current_q = _parse_current_q(Path(args.protocol_dir) / "Q_matrix.csv")
    current_pairs = _pair_counter(
        Path(args.protocol_dir) / "data.csv",
        header=True,
    )
    source_pairs = _pair_counter(Path(args.dfcd_total), header=False)
    pair_exact = current_pairs == source_pairs
    if not pair_exact:
        raise RuntimeError("Current and DFCD student-item multisets differ.")
    questions = json.loads(
        Path(args.official_questions).read_text(encoding="utf-8")
    )
    routes = json.loads(
        Path(args.official_routes).read_text(encoding="utf-8")
    )
    route_labels = {
        _normalize_node(label) for label in routes.values()
    }
    materialized = materialize_relations(
        current_q=current_q,
        reverse_question_map=reverse,
        questions=questions,
        route_labels=route_labels,
    )
    payload = {
        "schema_version": 1,
        "admitted": True,
        "outcome_blind": True,
        "response_label_columns_read": False,
        "source_identity": {
            "student_item_multiset_exact": pair_exact,
            "interaction_pairs": sum(current_pairs.values()),
            "current_items": len(current_q),
            "reconstructed_items": len(reverse),
            "current_concepts": len(set(current_q.values())),
        },
        "source_versions": {
            "official_xes_commit": XES_REPO_COMMIT,
            "dfcd_commit": DFCD_REPO_COMMIT,
            "dfcd_preprocess_parameters": {
                "seed": 0,
                "students": 2000,
                "exercises_requested": 2000,
                "top_concepts": 200,
                "minimum_responses": 50,
                "python_hash_seed": 0,
            },
        },
        "fingerprints": fingerprints,
        **materialized,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
