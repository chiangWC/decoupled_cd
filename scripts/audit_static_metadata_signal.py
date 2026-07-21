from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.pool_protocol import sha256_file, stable_fraction
from scripts.audit_option_contrast_signal import (
    _hash_values,
    _logit,
    _metrics,
    _target_mask,
    fast_student_cluster_bootstrap,
)
from scripts.audit_static_metadata_admission import (
    FROZEN_INPUT_SHA256,
    FROZEN_JUNYI_COMMIT,
    JUNYI_TRACKED_FILES,
    load_q_map,
    reconstruct_assist09,
    reconstruct_junyi,
    reconstruct_nips,
    sha256_bytes,
)
from scripts.evaluate_coverage_slice import add_target_coverage


MODEL_SEED = 42
SPLIT_SEED = 2024
NUM_FOLDS = 5
HIDE_FRACTION = 0.20
MIN_HISTORY_ROWS = 10
MIN_PSEUDO_TARGET_ITEMS = 3
MIN_TARGET_ROWS = 500
MIN_TARGET_STUDENTS = 100
MIN_TARGET_PER_LABEL = 100
ITEM_EASE_SHRINKAGE = 20.0
MAX_HOPS = 4
SHUFFLE_REPLICATES = 3
MIN_SWAPS_PER_EDGE = 10
BOOTSTRAP_REPLICATES = 2_000
STATIC_AUDIT_SHA256 = (
    "c034c24f2e9f0cde2ee5ff8a0ed13c47767c950fa93ac6a4ccd23770fc0bc2af"
)
COMMON_FEATURE_NAMES = (
    "log_history_rows",
    "history_balance",
    "observed_concept_fraction",
    "log_target_q_size",
    "target_q_seen_fraction",
    "log_target_q_attempts",
    "log_reference_item_count",
    "reference_item_ease_logit",
)
RELATION_FEATURE_NAMES = (
    "metadata_response_mean",
    "metadata_residual_mean",
    "metadata_abs_residual_mean",
    "log_metadata_path_mass",
    "inverse_first_metadata_hop",
    "metadata_path_reliability",
)


@dataclass(frozen=True)
class RelationEdge:
    relation: str
    source: str
    target: str
    directed: bool

    def canonical(self) -> tuple[str, str, str, bool]:
        if self.directed or self.source <= self.target:
            return self.relation, self.source, self.target, self.directed
        return self.relation, self.target, self.source, self.directed


@dataclass(frozen=True)
class Protocol:
    name: str
    train: pd.DataFrame
    q_frame: pd.DataFrame
    q_lookup: dict[str, tuple[str, ...]]
    target_scope: str
    q_edges: tuple[tuple[str, str], ...]
    metadata_edges: tuple[RelationEdge, ...]
    audit: dict[str, Any]


@dataclass(frozen=True)
class PseudoProtocol:
    support: pd.DataFrame
    target_public: pd.DataFrame
    target_labels: np.ndarray
    audit: dict[str, Any]


@dataclass(frozen=True)
class GraphKernel:
    nodes: tuple[str, ...]
    node_index: dict[str, int]
    q_transition: sparse.csr_matrix
    metadata_transition: sparse.csr_matrix
    all_transition: sparse.csr_matrix
    audit: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train-only predictive signal gate for static metadata."
    )
    parser.add_argument("--static-admission-audit", required=True)
    parser.add_argument("--assist09-protocol", required=True)
    parser.add_argument("--assist09-raw", required=True)
    parser.add_argument("--nips-protocol", required=True)
    parser.add_argument("--nips-question-metadata", required=True)
    parser.add_argument("--nips-subject-metadata", required=True)
    parser.add_argument("--junyi-protocol", required=True)
    parser.add_argument("--junyi-official-log", required=True)
    parser.add_argument("--junyi-directed", required=True)
    parser.add_argument("--junyi-undirected", required=True)
    parser.add_argument("--junyi-source-repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=BOOTSTRAP_REPLICATES,
    )
    args = parser.parse_args()
    if args.bootstrap_replicates != BOOTSTRAP_REPLICATES:
        raise ValueError("The formal gate fixes 2,000 bootstrap replicates.")
    return args


def _stable_seed(*parts: object) -> int:
    value = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _stable_row_id(dataset: str, student: object, item: object) -> str:
    value = f"{dataset}\x1f{student}\x1f{item}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:32]


def _concepts(value: object) -> tuple[str, ...]:
    text = str(value).strip().strip("[]")
    if not text or text.lower() == "nan":
        return ()
    separator = "," if "," in text else "_"
    return tuple(
        str(int(part.strip()))
        for part in text.split(separator)
        if part.strip()
    )


def _read_protocol_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={
            "source_row_id": str,
            "stu_id": str,
            "exer_id": str,
            "cpt_seq": str,
        },
        low_memory=False,
    )


def _verify(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"Hash mismatch for {path}: {actual} != {expected}")
    return actual


def _verify_junyi_git(
    repo: Path,
    paths: dict[str, Path],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, relative_path in JUNYI_TRACKED_FILES.items():
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "show",
                f"{FROZEN_JUNYI_COMMIT}:{relative_path}",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        digest = sha256_bytes(completed.stdout)
        if digest != FROZEN_INPUT_SHA256[name]:
            raise RuntimeError(f"Frozen Git blob mismatch: {relative_path}")
        if sha256_file(paths[name]) != digest:
            raise RuntimeError(f"Working source mismatch: {relative_path}")
        result[relative_path] = digest
    return result


def _q_frame(protocol_dir: Path) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    frame = _read_protocol_csv(protocol_dir / "Q_matrix.csv")
    if not {"exer_id", "cpt_seq"} <= set(frame):
        raise RuntimeError("Q_matrix.csv lacks exer_id or cpt_seq.")
    lookup: dict[str, set[str]] = defaultdict(set)
    for row in frame.itertuples(index=False):
        concepts = _concepts(row.cpt_seq)
        if not concepts:
            raise RuntimeError(f"Item {row.exer_id} has no concepts.")
        lookup[str(row.exer_id)].update(concepts)
    normalized = {
        item: tuple(sorted(values, key=int))
        for item, values in lookup.items()
    }
    q_rows = pd.DataFrame(
        [
            {"exer_id": item, "cpt_seq": ",".join(concepts)}
            for item, concepts in normalized.items()
        ]
    )
    return q_rows, normalized


def _q_edges(q_lookup: dict[str, tuple[str, ...]]) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            {
                (f"item:{item}", f"concept:{concept}")
                for item, concepts in q_lookup.items()
                for concept in concepts
            }
        )
    )


def _assist09_relations(
    metadata: dict[int, dict[str, set[str]]],
    *,
    num_items: int,
) -> tuple[RelationEdge, ...]:
    maximum = max(2, math.floor(0.10 * num_items))
    output: list[RelationEdge] = []
    for relation in ("template_id", "assistment_id"):
        items_by_value: dict[str, set[int]] = defaultdict(set)
        for item, fields in metadata.items():
            for value in fields.get(relation, set()):
                items_by_value[str(value)].add(int(item))
        for value, items in items_by_value.items():
            if not 2 <= len(items) <= maximum:
                continue
            group = f"group:{relation}:{value}"
            output.extend(
                RelationEdge(
                    relation=relation,
                    source=f"item:{item}",
                    target=group,
                    directed=True,
                )
                for item in sorted(items)
            )
    return tuple(sorted(output, key=RelationEdge.canonical))


def _load_protocols(args: argparse.Namespace) -> dict[str, Protocol]:
    admission_path = Path(args.static_admission_audit)
    _verify(admission_path, STATIC_AUDIT_SHA256)
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    if admission.get("frozen_inputs_verified") is not True:
        raise RuntimeError("Static admission provenance was not verified.")

    specs = {
        "assist09": {
            "directory": Path(args.assist09_protocol),
            "hash_prefix": "assist09",
            "scope": "bucket:zero",
        },
        "nips34": {
            "directory": Path(args.nips_protocol),
            "hash_prefix": "nips",
            "scope": "low_coverage",
        },
        "junyi": {
            "directory": Path(args.junyi_protocol),
            "hash_prefix": "junyi",
            "scope": "bucket:zero",
        },
    }
    source_paths = {
        "assist09_raw": Path(args.assist09_raw),
        "nips_question_metadata": Path(args.nips_question_metadata),
        "nips_subject_metadata": Path(args.nips_subject_metadata),
        "junyi_official_log": Path(args.junyi_official_log),
        "junyi_directed": Path(args.junyi_directed),
        "junyi_undirected": Path(args.junyi_undirected),
    }
    source_hashes = {
        name: _verify(path, FROZEN_INPUT_SHA256[name])
        for name, path in source_paths.items()
    }
    junyi_blobs = _verify_junyi_git(
        Path(args.junyi_source_repo),
        source_paths,
    )

    q_frames: dict[str, pd.DataFrame] = {}
    q_lookups: dict[str, dict[str, tuple[str, ...]]] = {}
    trains: dict[str, pd.DataFrame] = {}
    protocol_hashes: dict[str, dict[str, str]] = {}
    for name, spec in specs.items():
        directory = spec["directory"]
        prefix = spec["hash_prefix"]
        verified = {}
        for filename, suffix in (
            ("data.csv", "data"),
            ("train.csv", "train"),
            ("Q_matrix.csv", "Q_matrix"),
        ):
            verified[filename] = _verify(
                directory / filename,
                FROZEN_INPUT_SHA256[f"{prefix}_{suffix}"],
            )
        protocol_hashes[name] = verified
        train = _read_protocol_csv(directory / "train.csv")
        train["label"] = pd.to_numeric(train["label"], errors="raise").astype(int)
        if not set(train["label"].unique()) <= {0, 1}:
            raise RuntimeError(f"{name} has non-binary train labels.")
        q_frame, q_lookup = _q_frame(directory)
        missing = set(train["exer_id"].astype(str)) - set(q_lookup)
        if missing:
            raise RuntimeError(f"{name} train has {len(missing)} items absent from Q.")
        trains[name] = train.reset_index(drop=True)
        q_frames[name] = q_frame
        q_lookups[name] = q_lookup

    a09_identity, a09_metadata = reconstruct_assist09(
        args.assist09_raw,
        args.assist09_protocol,
    )
    nips_identity, nips_hierarchy = reconstruct_nips(
        args.nips_question_metadata,
        args.nips_subject_metadata,
        args.nips_protocol,
    )
    junyi_identity, junyi_directed, junyi_undirected = reconstruct_junyi(
        args.junyi_official_log,
        args.junyi_directed,
        args.junyi_undirected,
        args.junyi_protocol,
    )
    identities = {
        "assist09": a09_identity,
        "nips34": nips_identity,
        "junyi": junyi_identity,
    }
    if not all(report["identity_exact"] for report in identities.values()):
        raise RuntimeError("A source-to-protocol identity reconstruction failed.")
    relations = {
        "assist09": _assist09_relations(
            a09_metadata,
            num_items=len(q_lookups["assist09"]),
        ),
        "nips34": tuple(
            RelationEdge(
                "subject_hierarchy",
                f"concept:{child}",
                f"concept:{parent}",
                True,
            )
            for child, parent in nips_hierarchy
        ),
        "junyi": tuple(
            [
                *(
                    RelationEdge(
                        "prerequisite",
                        f"concept:{left}",
                        f"concept:{right}",
                        True,
                    )
                    for left, right in junyi_directed
                ),
                *(
                    RelationEdge(
                        "similarity",
                        f"concept:{left}",
                        f"concept:{right}",
                        False,
                    )
                    for left, right in junyi_undirected
                ),
            ]
        ),
    }
    output = {}
    for name, spec in specs.items():
        directory = spec["directory"]
        forbidden = [
            str((directory / filename).resolve())
            for filename in ("valid.csv", "test.csv")
        ]
        output[name] = Protocol(
            name=name,
            train=trains[name],
            q_frame=q_frames[name],
            q_lookup=q_lookups[name],
            target_scope=spec["scope"],
            q_edges=_q_edges(q_lookups[name]),
            metadata_edges=relations[name],
            audit={
                "static_admission_sha256": STATIC_AUDIT_SHA256,
                "identity": identities[name],
                "protocol_hashes": protocol_hashes[name],
                "source_hashes": source_hashes,
                "junyi_git_blobs": junyi_blobs if name == "junyi" else {},
                "opened_protocol_files": [
                    str((directory / filename).resolve())
                    for filename in ("data.csv", "train.csv", "Q_matrix.csv")
                ],
                "forbidden_protocol_files_not_opened": forbidden,
                "validation_or_test_opened": False,
            },
        )
    return output


def build_pseudo_protocol(protocol: Protocol) -> PseudoProtocol:
    support_indices: list[int] = []
    target_indices: list[int] = []
    fold_by_student: dict[str, int] = {}
    skipped_students = 0
    for student, frame in protocol.train.groupby(
        protocol.train["stu_id"].astype(str),
        sort=False,
    ):
        student = str(student)
        groups = {
            str(item): list(group.index)
            for item, group in frame.groupby(
                frame["exer_id"].astype(str),
                sort=False,
            )
        }
        ordered_items = sorted(
            groups,
            key=lambda item: (
                stable_fraction(
                    SPLIT_SEED,
                    "static-metadata-signal-hide",
                    protocol.name,
                    student,
                    item,
                ),
                item,
            ),
        )
        k = max(
            MIN_PSEUDO_TARGET_ITEMS,
            int(np.floor(HIDE_FRACTION * len(ordered_items))),
        )
        target_items = set(ordered_items[:k])
        support_rows = [
            index
            for item, rows in groups.items()
            if item not in target_items
            for index in rows
        ]
        if len(support_rows) < MIN_HISTORY_ROWS:
            skipped_students += 1
            continue
        support_indices.extend(support_rows)
        target_indices.extend(
            index
            for item, rows in groups.items()
            if item in target_items
            for index in rows
        )
        fold_by_student[student] = min(
            int(
                NUM_FOLDS
                * stable_fraction(
                    SPLIT_SEED,
                    "static-metadata-signal-fold",
                    protocol.name,
                    student,
                )
            ),
            NUM_FOLDS - 1,
        )
    support = protocol.train.loc[support_indices].copy().reset_index(drop=True)
    hidden = protocol.train.loc[target_indices].copy().reset_index(drop=True)
    support_groups = set(
        zip(
            support["stu_id"].astype(str),
            support["exer_id"].astype(str),
            strict=True,
        )
    )
    target_groups = set(
        zip(
            hidden["stu_id"].astype(str),
            hidden["exer_id"].astype(str),
            strict=True,
        )
    )
    if support_groups & target_groups:
        raise RuntimeError("Pseudo support and target student-item groups overlap.")
    labels = hidden["label"].to_numpy(dtype=np.int64, copy=True)
    public_columns = [
        column
        for column in ("source_row_id", "stu_id", "exer_id", "cpt_seq")
        if column in hidden
    ]
    public = hidden.loc[:, public_columns].copy()
    if "source_row_id" not in public:
        public.insert(
            0,
            "source_row_id",
            [
                _stable_row_id(protocol.name, student, item)
                for student, item in public[
                    ["stu_id", "exer_id"]
                ].itertuples(index=False, name=None)
            ],
        )
    public["fold"] = public["stu_id"].astype(str).map(fold_by_student).astype(int)
    coverage_input = public.copy()
    coverage_input["label"] = labels
    covered = add_target_coverage(
        coverage_input,
        student_history_frame=support,
        q_matrix=protocol.q_frame,
    )
    public = covered.drop(columns="label")
    if "label" in public:
        raise RuntimeError("Pseudo-target label leaked into public features.")
    fold_counts = {
        str(fold): int((public["fold"] == fold).sum())
        for fold in range(NUM_FOLDS)
    }
    if any(count == 0 for count in fold_counts.values()):
        raise RuntimeError(f"Empty pseudo-target fold: {fold_counts}")
    return PseudoProtocol(
        support=support,
        target_public=public.reset_index(drop=True),
        target_labels=labels,
        audit={
            "support_rows": int(len(support)),
            "pseudo_target_rows": int(len(public)),
            "eligible_students": int(len(fold_by_student)),
            "skipped_students": int(skipped_students),
            "fold_counts": fold_counts,
            "selection_sha256": _hash_values(
                f"{row.source_row_id}:{int(row.fold)}"
                for row in public.sort_values("source_row_id").itertuples(index=False)
            ),
            "support_target_group_overlap": 0,
            "selection_uses_label": False,
        },
    )


def pseudo_target_feasibility(
    pseudo: PseudoProtocol,
    scope: str,
) -> dict[str, Any]:
    target = _target_mask(pseudo.target_public, scope)
    labels = pseudo.target_labels[target]
    students = pseudo.target_public.loc[target, "stu_id"].astype(str)
    report = {
        "scope": scope,
        "rows": int(target.sum()),
        "students": int(students.nunique()),
        "label_0": int((labels == 0).sum()),
        "label_1": int((labels == 1).sum()),
        "minimum_rows": MIN_TARGET_ROWS,
        "minimum_students": MIN_TARGET_STUDENTS,
        "minimum_per_label": MIN_TARGET_PER_LABEL,
    }
    report["eligible"] = bool(
        report["rows"] >= MIN_TARGET_ROWS
        and report["students"] >= MIN_TARGET_STUDENTS
        and report["label_0"] >= MIN_TARGET_PER_LABEL
        and report["label_1"] >= MIN_TARGET_PER_LABEL
    )
    return report


def _degree_signature(
    edges: Iterable[RelationEdge],
) -> dict[str, dict[str, int]]:
    output: dict[str, Counter[str]] = defaultdict(Counter)
    for edge in edges:
        if edge.directed:
            output[f"{edge.relation}:out"][edge.source] += 1
            output[f"{edge.relation}:in"][edge.target] += 1
        else:
            output[f"{edge.relation}:degree"][edge.source] += 1
            output[f"{edge.relation}:degree"][edge.target] += 1
    return {
        relation: dict(sorted(values.items()))
        for relation, values in sorted(output.items())
    }


def _edge_hash(edges: Iterable[RelationEdge]) -> str:
    return _hash_values(
        ":".join(map(str, edge.canonical()))
        for edge in sorted(edges, key=RelationEdge.canonical)
    )


def _rewire_directed(
    edges: list[RelationEdge],
    *,
    rng: np.random.Generator,
) -> tuple[list[RelationEdge], int]:
    pairs = [(edge.source, edge.target) for edge in edges]
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
        proposed = ((a, d), (c, b))
        if a == c or b == d or a == d or c == b:
            continue
        if proposed[0] == proposed[1]:
            continue
        remaining = occupied - {pairs[first], pairs[second]}
        if proposed[0] in remaining or proposed[1] in remaining:
            continue
        occupied = remaining | set(proposed)
        pairs[first], pairs[second] = proposed
        accepted += 1
    if accepted < target_swaps:
        raise RuntimeError(
            f"Directed rewire accepted {accepted}/{target_swaps} swaps."
        )
    relation = edges[0].relation
    return [
        RelationEdge(relation, source, target, True)
        for source, target in pairs
    ], accepted


def _rewire_undirected(
    edges: list[RelationEdge],
    *,
    rng: np.random.Generator,
) -> tuple[list[RelationEdge], int]:
    pairs = [tuple(sorted((edge.source, edge.target))) for edge in edges]
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
        if rng.integers(0, 2):
            c, d = d, c
        if len({a, b, c, d}) < 4:
            continue
        proposed = (
            tuple(sorted((a, d))),
            tuple(sorted((c, b))),
        )
        if proposed[0] == proposed[1]:
            continue
        remaining = occupied - {pairs[first], pairs[second]}
        if proposed[0] in remaining or proposed[1] in remaining:
            continue
        occupied = remaining | set(proposed)
        pairs[first], pairs[second] = proposed
        accepted += 1
    if accepted < target_swaps:
        raise RuntimeError(
            f"Undirected rewire accepted {accepted}/{target_swaps} swaps."
        )
    relation = edges[0].relation
    return [
        RelationEdge(relation, source, target, False)
        for source, target in pairs
    ], accepted


def rewire_metadata(
    edges: tuple[RelationEdge, ...],
    *,
    dataset: str,
    replicate: int,
) -> tuple[tuple[RelationEdge, ...], dict[str, Any]]:
    by_relation: dict[str, list[RelationEdge]] = defaultdict(list)
    for edge in edges:
        by_relation[edge.relation].append(edge)
    shuffled: list[RelationEdge] = []
    accepted_by_relation = {}
    for relation, relation_edges in sorted(by_relation.items()):
        directed_values = {edge.directed for edge in relation_edges}
        if len(directed_values) != 1:
            raise RuntimeError(f"Mixed directionality in {relation}.")
        rng = np.random.default_rng(
            _stable_seed(SPLIT_SEED, dataset, replicate, relation)
        )
        if relation_edges[0].directed:
            rewired, accepted = _rewire_directed(relation_edges, rng=rng)
        else:
            rewired, accepted = _rewire_undirected(relation_edges, rng=rng)
        shuffled.extend(rewired)
        accepted_by_relation[relation] = accepted
    original_degrees = _degree_signature(edges)
    shuffled_degrees = _degree_signature(shuffled)
    if original_degrees != shuffled_degrees:
        raise RuntimeError("Degree-preserving shuffle changed a degree sequence.")
    if len({edge.canonical() for edge in shuffled}) != len(shuffled):
        raise RuntimeError("Degree-preserving shuffle produced duplicate edges.")
    if any(edge.source == edge.target for edge in shuffled):
        raise RuntimeError("Degree-preserving shuffle produced a self-loop.")
    return tuple(sorted(shuffled, key=RelationEdge.canonical)), {
        "replicate": replicate,
        "edge_count": len(shuffled),
        "accepted_swaps_by_relation": accepted_by_relation,
        "minimum_swaps_per_edge": MIN_SWAPS_PER_EDGE,
        "degree_signature_sha256": _hash_values(
            json.dumps(original_degrees, sort_keys=True)
        ),
        "original_edge_sha256": _edge_hash(edges),
        "shuffled_edge_sha256": _edge_hash(shuffled),
        "degree_sequence_exact": True,
        "self_loops": 0,
        "duplicate_edges": 0,
    }


def build_graph_kernel(
    q_edges: tuple[tuple[str, str], ...],
    metadata_edges: tuple[RelationEdge, ...],
) -> GraphKernel:
    nodes = tuple(
        sorted(
            {
                node
                for edge in q_edges
                for node in edge
            }
            | {
                node
                for edge in metadata_edges
                for node in (edge.source, edge.target)
            }
        )
    )
    node_index = {node: index for index, node in enumerate(nodes)}
    q_rows: list[int] = []
    q_columns: list[int] = []
    q_values: list[float] = []
    for left, right in q_edges:
        a, b = node_index[left], node_index[right]
        q_rows.extend((a, b))
        q_columns.extend((b, a))
        q_values.extend((1.0, 1.0))
    m_rows: list[int] = []
    m_columns: list[int] = []
    m_values: list[float] = []
    for edge in metadata_edges:
        a, b = node_index[edge.source], node_index[edge.target]
        m_rows.extend((a, b))
        m_columns.extend((b, a))
        if edge.directed:
            m_values.extend((1.0, 0.5))
        else:
            m_values.extend((1.0, 1.0))
    shape = (len(nodes), len(nodes))
    q = sparse.csr_matrix(
        (np.asarray(q_values), (q_rows, q_columns)),
        shape=shape,
        dtype=np.float64,
    )
    metadata = sparse.csr_matrix(
        (np.asarray(m_values), (m_rows, m_columns)),
        shape=shape,
        dtype=np.float64,
    )
    total = q + metadata
    degree = np.asarray(total.sum(axis=1)).ravel()
    inverse = np.divide(
        1.0,
        degree,
        out=np.zeros_like(degree),
        where=degree > 0,
    )
    normalization = sparse.diags(inverse)
    q_transition = (normalization @ q).tocsr()
    metadata_transition = (normalization @ metadata).tocsr()
    all_transition = (q_transition + metadata_transition).tocsr()
    return GraphKernel(
        nodes=nodes,
        node_index=node_index,
        q_transition=q_transition,
        metadata_transition=metadata_transition,
        all_transition=all_transition,
        audit={
            "nodes": len(nodes),
            "q_edges": len(q_edges),
            "metadata_edges": len(metadata_edges),
            "metadata_edge_sha256": _edge_hash(metadata_edges),
            "directed_forward_weight": 1.0,
            "directed_inverse_weight": 0.5,
            "max_hops": MAX_HOPS,
        },
    )


def _item_statistics(
    reference_support: pd.DataFrame,
    items: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, float]]:
    count = Counter(reference_support["exer_id"].astype(str))
    correct = Counter(
        reference_support.loc[
            reference_support["label"].to_numpy(dtype=int) == 1,
            "exer_id",
        ].astype(str)
    )
    global_rate = (
        float(reference_support["label"].sum()) + 1.0
    ) / (len(reference_support) + 2.0)
    ease = {
        item: (
            correct[item] + ITEM_EASE_SHRINKAGE * global_rate
        ) / (count[item] + ITEM_EASE_SHRINKAGE)
        for item in items
    }
    return {item: float(count[item]) for item in items}, ease


def common_features(
    *,
    support: pd.DataFrame,
    target: pd.DataFrame,
    q_lookup: dict[str, tuple[str, ...]],
    item_count: dict[str, float],
    item_ease: dict[str, float],
) -> np.ndarray:
    concepts = {
        concept for values in q_lookup.values() for concept in values
    }
    stats: dict[str, dict[str, Any]] = {}
    for student, frame in support.groupby(
        support["stu_id"].astype(str),
        sort=False,
    ):
        attempts: Counter[str] = Counter()
        for item in frame["exer_id"].astype(str):
            attempts.update(q_lookup[item])
        stats[str(student)] = {
            "rows": len(frame),
            "correct": float(frame["label"].sum()),
            "attempts": attempts,
            "seen": set(attempts),
        }
    output = np.zeros((len(target), len(COMMON_FEATURE_NAMES)), dtype=np.float64)
    for index, row in enumerate(target.itertuples(index=False)):
        student = str(row.stu_id)
        item = str(row.exer_id)
        state = stats[student]
        target_concepts = q_lookup[item]
        target_attempts = sum(
            state["attempts"][concept] for concept in target_concepts
        )
        target_seen = sum(
            concept in state["seen"] for concept in target_concepts
        )
        posterior = (state["correct"] + 1.0) / (state["rows"] + 2.0)
        output[index] = (
            np.log1p(state["rows"]),
            2.0 * posterior - 1.0,
            len(state["seen"]) / max(len(concepts), 1),
            np.log1p(len(target_concepts)),
            target_seen / len(target_concepts),
            np.log1p(target_attempts),
            np.log1p(item_count[item]),
            float(_logit(item_ease[item])),
        )
    return output


def relation_features(
    *,
    support: pd.DataFrame,
    target: pd.DataFrame,
    kernel: GraphKernel,
    item_ease: dict[str, float],
    batch_size: int = 32,
) -> np.ndarray:
    students = tuple(sorted(target["stu_id"].astype(str).unique()))
    student_index = {student: index for index, student in enumerate(students)}
    target_student = target["stu_id"].astype(str).map(student_index).to_numpy()
    target_nodes = np.asarray(
        [
            kernel.node_index[f"item:{item}"]
            for item in target["exer_id"].astype(str)
        ],
        dtype=int,
    )
    output = np.zeros(
        (len(target), len(RELATION_FEATURE_NAMES)),
        dtype=np.float64,
    )
    support_by_student = {
        str(student): frame
        for student, frame in support.groupby(
            support["stu_id"].astype(str),
            sort=False,
        )
    }
    for start in range(0, len(students), batch_size):
        batch_students = students[start : start + batch_size]
        batch = len(batch_students)
        local = {student: index for index, student in enumerate(batch_students)}
        initial = np.zeros((4 * batch, len(kernel.nodes)), dtype=np.float64)
        for student in batch_students:
            frame = support_by_student[student]
            row = local[student]
            items = frame["exer_id"].astype(str).to_numpy()
            nodes = np.asarray(
                [kernel.node_index[f"item:{item}"] for item in items],
                dtype=int,
            )
            labels = frame["label"].to_numpy(dtype=np.float64)
            ease = np.asarray([item_ease[item] for item in items])
            residual = labels - ease
            np.add.at(initial[row], nodes, 1.0)
            np.add.at(initial[batch + row], nodes, labels)
            np.add.at(initial[2 * batch + row], nodes, residual)
            np.add.at(initial[3 * batch + row], nodes, np.abs(residual))

        rows = np.flatnonzero(
            (target_student >= start)
            & (target_student < start + batch)
        )
        local_students = target_student[rows] - start
        local_nodes = target_nodes[rows]
        mass_total = np.zeros(len(rows), dtype=np.float64)
        correct_total = np.zeros(len(rows), dtype=np.float64)
        residual_total = np.zeros(len(rows), dtype=np.float64)
        absolute_total = np.zeros(len(rows), dtype=np.float64)
        first_hop = np.zeros(len(rows), dtype=np.float64)
        no_metadata = initial
        used_metadata = np.zeros_like(initial)
        for hop in range(1, MAX_HOPS + 1):
            next_no_metadata = no_metadata @ kernel.q_transition
            next_used_metadata = (
                used_metadata @ kernel.all_transition
                + no_metadata @ kernel.metadata_transition
            )
            no_metadata = np.asarray(next_no_metadata)
            used_metadata = np.asarray(next_used_metadata)
            mass = used_metadata[local_students, local_nodes]
            correct = used_metadata[batch + local_students, local_nodes]
            residual = used_metadata[2 * batch + local_students, local_nodes]
            absolute = used_metadata[3 * batch + local_students, local_nodes]
            newly_reached = (first_hop == 0.0) & (mass > 0.0)
            first_hop[newly_reached] = float(hop)
            mass_total += mass
            correct_total += correct
            residual_total += residual
            absolute_total += absolute
        positive = mass_total > 0.0
        block = np.zeros((len(rows), len(RELATION_FEATURE_NAMES)))
        block[positive, 0] = correct_total[positive] / mass_total[positive]
        block[positive, 1] = residual_total[positive] / mass_total[positive]
        block[positive, 2] = absolute_total[positive] / mass_total[positive]
        block[:, 3] = np.log1p(mass_total)
        block[positive, 4] = 1.0 / first_hop[positive]
        block[:, 5] = mass_total / (mass_total + 1.0)
        output[rows] = block
    return output


def fit_fold(
    protocol: Protocol,
    pseudo: PseudoProtocol,
    kernels: dict[str, GraphKernel],
    fold: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    reference_mask = pseudo.target_public["fold"].to_numpy(dtype=int) != fold
    held_mask = ~reference_mask
    reference_students = set(
        pseudo.target_public.loc[reference_mask, "stu_id"].astype(str)
    )
    held_students = set(
        pseudo.target_public.loc[held_mask, "stu_id"].astype(str)
    )
    if reference_students & held_students:
        raise RuntimeError("OOF reference and held students overlap.")
    reference_support = pseudo.support.loc[
        pseudo.support["stu_id"].astype(str).isin(reference_students)
    ]
    item_count, item_ease = _item_statistics(
        reference_support,
        tuple(sorted(protocol.q_lookup)),
    )
    reference_target = pseudo.target_public.loc[reference_mask].reset_index(drop=True)
    held_target = pseudo.target_public.loc[held_mask].reset_index(drop=True)
    reference_labels = pseudo.target_labels[reference_mask]
    held_labels = pseudo.target_labels[held_mask]
    common_reference = common_features(
        support=pseudo.support,
        target=reference_target,
        q_lookup=protocol.q_lookup,
        item_count=item_count,
        item_ease=item_ease,
    )
    common_held = common_features(
        support=pseudo.support,
        target=held_target,
        q_lookup=protocol.q_lookup,
        item_count=item_count,
        item_ease=item_ease,
    )
    zeros_reference = np.zeros(
        (len(reference_target), len(RELATION_FEATURE_NAMES))
    )
    zeros_held = np.zeros((len(held_target), len(RELATION_FEATURE_NAMES)))
    reference_features = {
        "q_only": np.hstack((common_reference, zeros_reference))
    }
    held_features = {"q_only": np.hstack((common_held, zeros_held))}
    for variant, kernel in kernels.items():
        reference_block = relation_features(
            support=pseudo.support,
            target=reference_target,
            kernel=kernel,
            item_ease=item_ease,
        )
        held_block = relation_features(
            support=pseudo.support,
            target=held_target,
            kernel=kernel,
            item_ease=item_ease,
        )
        reference_features[variant] = np.hstack(
            (common_reference, reference_block)
        )
        held_features[variant] = np.hstack((common_held, held_block))
    variants = tuple(reference_features)
    dimensions = {
        variant: int(reference_features[variant].shape[1])
        for variant in variants
    }
    if len(set(dimensions.values())) != 1:
        raise RuntimeError(f"Variant feature dimensions differ: {dimensions}")
    scaler = StandardScaler()
    scaler.fit(np.vstack([reference_features[name] for name in variants]))
    output = held_target.copy()
    output["label"] = held_labels
    if np.unique(reference_labels).size != 2:
        raise RuntimeError(f"Reference labels lack both classes in fold {fold}.")
    for variant in variants:
        estimator = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="liblinear",
            max_iter=1_000,
            random_state=MODEL_SEED,
        )
        estimator.fit(
            scaler.transform(reference_features[variant]),
            reference_labels,
        )
        output[f"prob_{variant}"] = estimator.predict_proba(
            scaler.transform(held_features[variant])
        )[:, 1]
    return output, {
        "fold": fold,
        "reference_students": len(reference_students),
        "held_students": len(held_students),
        "reference_target_rows": len(reference_target),
        "held_target_rows": len(held_target),
        "feature_names": [
            *COMMON_FEATURE_NAMES,
            *RELATION_FEATURE_NAMES,
        ],
        "feature_dimensions": dimensions,
        "shared_scaler": True,
        "classifier": (
            "L2 LogisticRegression(C=1, solver=liblinear, max_iter=1000)"
        ),
        "target_label_used_as_feature": False,
    }


def summarize(
    predictions: pd.DataFrame,
    *,
    target_scope: str,
) -> dict[str, Any]:
    variants = [
        column.removeprefix("prob_")
        for column in predictions
        if column.startswith("prob_")
    ]
    labels = predictions["label"].to_numpy(dtype=int)
    target = _target_mask(predictions, target_scope)
    metrics = {}
    for variant in variants:
        probabilities = predictions[f"prob_{variant}"].to_numpy(dtype=float)
        metrics[variant] = {
            "overall": _metrics(labels, probabilities),
            "target": _metrics(labels[target], probabilities[target]),
        }
    shuffles = [name for name in variants if name.startswith("shuffle_")]
    strongest_shuffle = max(
        shuffles,
        key=lambda name: metrics[name]["target"]["auc"],
    )
    stronger_control = max(
        ("q_only", strongest_shuffle),
        key=lambda name: metrics[name]["target"]["auc"],
    )
    full_target = metrics["full"]["target"]
    control_target = metrics[stronger_control]["target"]
    full_overall = metrics["full"]["overall"]
    control_overall = metrics[stronger_control]["overall"]
    delta = {
        "target_auc": full_target["auc"] - control_target["auc"],
        "overall_auc": full_overall["auc"] - control_overall["auc"],
        "target_brier": full_target["brier"] - control_target["brier"],
        "overall_brier": full_overall["brier"] - control_overall["brier"],
    }
    full_beats_every_control = all(
        full_target["auc"] > metrics[name]["target"]["auc"]
        for name in ("q_only", *shuffles)
    )
    passed = bool(
        delta["target_auc"] >= 0.005
        and delta["overall_auc"] >= -0.001
        and delta["target_brier"] <= 0.0002
        and full_beats_every_control
    )
    return {
        "target_scope": target_scope,
        "metrics": metrics,
        "strongest_shuffle": strongest_shuffle,
        "stronger_control": stronger_control,
        "control_selected_once_by_target_auc": True,
        "deltas_full_minus_control": delta,
        "full_target_auc_exceeds_every_control": full_beats_every_control,
        "deterministic_dataset_pass": passed,
    }


def aggregate_gate(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    passing = {
        name
        for name, summary in summaries.items()
        if summary.get("deterministic_dataset_pass", False)
    }
    maximum = max(
        (
            summaries[name]["deltas_full_minus_control"]["target_auc"]
            for name in passing
        ),
        default=float("-inf"),
    )
    return {
        "passing_datasets": sorted(passing),
        "at_least_two_datasets": len(passing) >= 2,
        "assist09_or_nips_required": bool(passing & {"assist09", "nips34"}),
        "one_target_delta_at_least_0_010": maximum >= 0.010,
        "bootstrap_needed": (
            len(passing) >= 2
            and bool(passing & {"assist09", "nips34"})
            and maximum >= 0.010
        ),
    }


def run_dataset(
    protocol: Protocol,
    *,
    output_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    pseudo = build_pseudo_protocol(protocol)
    feasibility = pseudo_target_feasibility(pseudo, protocol.target_scope)
    if not feasibility["eligible"]:
        summary = {
            "dataset": protocol.name,
            "signal_eligible": False,
            "signal_eligibility": feasibility,
            "pseudo_protocol": pseudo.audit,
            "input_audit": protocol.audit,
            "deterministic_dataset_pass": False,
        }
        (output_dir / f"{protocol.name}_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return summary, pd.DataFrame()
    shuffles = {}
    shuffle_audits = {}
    for replicate in range(SHUFFLE_REPLICATES):
        edges, audit = rewire_metadata(
            protocol.metadata_edges,
            dataset=protocol.name,
            replicate=replicate,
        )
        shuffles[f"shuffle_{replicate}"] = edges
        shuffle_audits[f"shuffle_{replicate}"] = audit
    kernels = {
        "full": build_graph_kernel(protocol.q_edges, protocol.metadata_edges),
        **{
            name: build_graph_kernel(protocol.q_edges, edges)
            for name, edges in shuffles.items()
        },
    }
    folds = []
    fold_audits = []
    for fold in range(NUM_FOLDS):
        prediction, audit = fit_fold(protocol, pseudo, kernels, fold)
        folds.append(prediction)
        fold_audits.append(audit)
    predictions = pd.concat(folds, ignore_index=True)
    if predictions["source_row_id"].duplicated().any():
        raise RuntimeError("OOF predictions contain duplicate source rows.")
    predictions = predictions.sort_values(
        "source_row_id",
        kind="stable",
    ).reset_index(drop=True)
    summary = summarize(predictions, target_scope=protocol.target_scope)
    prediction_path = output_dir / f"{protocol.name}_predictions.csv"
    predictions.to_csv(prediction_path, index=False)
    summary.update(
        {
            "dataset": protocol.name,
            "signal_eligible": True,
            "signal_eligibility": feasibility,
            "pseudo_protocol": pseudo.audit,
            "input_audit": protocol.audit,
            "full_graph": kernels["full"].audit,
            "shuffle_audits": shuffle_audits,
            "folds": fold_audits,
            "prediction_path": str(prediction_path.resolve()),
            "prediction_sha256": sha256_file(prediction_path),
            "prediction_rows": len(predictions),
            "model_seed": MODEL_SEED,
            "split_seed": SPLIT_SEED,
            "multi_seed": False,
        }
    )
    (output_dir / f"{protocol.name}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary, predictions


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocols = _load_protocols(args)
    summaries = {}
    predictions = {}
    for name in ("assist09", "nips34", "junyi"):
        summary, prediction = run_dataset(
            protocols[name],
            output_dir=output_dir,
        )
        summaries[name] = summary
        predictions[name] = prediction
    aggregate = aggregate_gate(summaries)
    bootstraps = {}
    if aggregate["bootstrap_needed"]:
        for index, name in enumerate(aggregate["passing_datasets"]):
            prediction = predictions[name]
            target = _target_mask(prediction, protocols[name].target_scope)
            control = summaries[name]["stronger_control"]
            bootstraps[name] = fast_student_cluster_bootstrap(
                labels=prediction.loc[target, "label"].to_numpy(dtype=int),
                students=prediction.loc[target, "stu_id"].astype(str).to_numpy(),
                full_probability=prediction.loc[
                    target, "prob_full"
                ].to_numpy(dtype=float),
                control_probability=prediction.loc[
                    target, f"prob_{control}"
                ].to_numpy(dtype=float),
                replicates=BOOTSTRAP_REPLICATES,
                seed=MODEL_SEED + index,
            )
    aggregate["bootstrap_ci_lower_bound_above_zero"] = any(
        report["ci_low"] > 0.0 for report in bootstraps.values()
    )
    aggregate["route_activated"] = bool(
        aggregate["at_least_two_datasets"]
        and aggregate["assist09_or_nips_required"]
        and aggregate["one_target_delta_at_least_0_010"]
        and aggregate["bootstrap_ci_lower_bound_above_zero"]
    )
    payload = {
        "schema_version": 1,
        "policy": {
            "train_only_pseudo_target": True,
            "validation_or_test_opened": False,
            "model_seed": MODEL_SEED,
            "split_seed": SPLIT_SEED,
            "student_disjoint_oof_folds": NUM_FOLDS,
            "hide_fraction": HIDE_FRACTION,
            "max_hops": MAX_HOPS,
            "shuffle_replicates": SHUFFLE_REPLICATES,
            "minimum_swaps_per_edge": MIN_SWAPS_PER_EDGE,
            "target_auc_minimum_delta": 0.005,
            "one_dataset_target_auc_delta": 0.010,
            "overall_auc_maximum_regression": 0.001,
            "target_brier_maximum_increase": 0.0002,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "summaries": summaries,
        "aggregate_gate": aggregate,
        "bootstraps": bootstraps,
    }
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
