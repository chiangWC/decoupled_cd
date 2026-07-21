from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Hashable, Iterable

import pandas as pd


Node = tuple[str, Hashable]

FROZEN_MAX_HOPS = 4
FROZEN_MAX_GROUP_FRACTION = 0.10
FROZEN_MIN_REACHABLE_FRACTION = 0.25
FROZEN_MIN_PASSING_DATASETS = 3
FROZEN_JUNYI_COMMIT = (
    "178e6a9d1485265cbc5f2f5dc93ecbe5f3c28ef0"
)
FROZEN_INPUT_SHA256 = {
    "assist09_raw": (
        "162ef8d2d28bcbfea6591a282994062bd8d5eaa00636544292a0d268dca6e5da"
    ),
    "assist17_raw": (
        "b5b366b11d9250af319f3117c5ccb39544fd72379026bfd33514e4bbd047bf73"
    ),
    "nips_question_metadata": (
        "44204e3450d0a2298fef0d940677fcd7c373b43f994cd5fec1d7381aff95001b"
    ),
    "nips_subject_metadata": (
        "d576a6eccc171d8eb82a284a9586c27b2bd9941f39f3fcd6ae1110470766f202"
    ),
    "junyi_official_log": (
        "50802f681b7191194c4bfbb20a6af81cb7a2239c58e33b939093702a770278ca"
    ),
    "junyi_directed": (
        "32cda83a7d36e25791f442ccaf50a32ebf70680fcd6a0e17ab11ec87d5407c82"
    ),
    "junyi_undirected": (
        "bafb696991b7ab3572d6333421e1d832da1c5a2a1a63d871779a709382742251"
    ),
    "assist09_data": "998d5e80d9e1fb28b286926fd2ced2c2947de2e919d20375285b020b357ae4d7",
    "assist09_train": "e266351a038366cda2f380bf649e8f6cb520e508ce1f3ad2b698890e724dab51",
    "assist09_valid": "33e813b197fbb4b7618f3b89c260caf2f2b0c4293508b0287df8aba5b1f9853c",
    "assist09_Q_matrix": "1d729d53e4285f555e2c747285d154997c77a2601b883f035e7466550e4ce0a3",
    "assist17_data": "8442eb7de28f4a01462712b3f01976904c62c07f74e0dc717ad604690e3c45cf",
    "assist17_train": "e3c281f01d2fedaafa289c6d950c6ae64c48e2d2d3b2f6a69bd28e88a3ef236b",
    "assist17_valid": "07b74bcfcd2c28693469d9eedddbe8b8d0fcc89d4659b079d75198886267fe70",
    "assist17_Q_matrix": "23a59ec57c3b454d2d3fece3760aa91ec65a297e357766265466591b5bb0f9e3",
    "nips_data": "b1259ecfc3a49d0f40bd988d2f57bbff7684c73d915826c81b64cb9f829acaf8",
    "nips_train": "64c8d02067cffe87ec8b1b1b747604425a5d1223d9097c9e904c61350b4312d7",
    "nips_valid": "0ba37ad102168651e1d405492ee0186a6e2ebf295c49188b582b4df1450e0ce4",
    "nips_Q_matrix": "cb11593103b55b184fc4e1eaea41449c49fce0b4783f5ed44563c23ca818088c",
    "junyi_data": "d079a8d80ab557d2de5dbe6200566f8171ebe1fa5f572d4e85a9f9ce8aee3174",
    "junyi_train": "85eeac0b393d01debf43afb922940cdd3efca167b30a0cb8fda8c1eb37d82500",
    "junyi_valid": "b7893f25c7f506dbb39bbed282e494ff5dd4f4327f7e5a52c11efef65e3cb4f5",
    "junyi_Q_matrix": "64b1039bf9a022069b1df5248f5335081427056aebac7902d8ce5b4f5bcfc13e",
}
JUNYI_TRACKED_FILES = {
    "junyi_official_log": "data/junyi/log_data.json",
    "junyi_directed": "data/junyi/graph/K_Directed.txt",
    "junyi_undirected": "data/junyi/graph/K_Undirected.txt",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def parse_concepts(value: Any) -> tuple[int, ...]:
    if isinstance(value, (int, float)) and not pd.isna(value):
        return (int(value),)
    text = str(value).strip().strip("[]")
    if not text or text.lower() == "nan":
        return ()
    separator = "," if "," in text else "_"
    return tuple(int(part.strip()) for part in text.split(separator) if part.strip())


def load_q_map(path: str | Path) -> dict[int, set[int]]:
    frame = pd.read_csv(path, usecols=["exer_id", "cpt_seq"])
    result: dict[int, set[int]] = defaultdict(set)
    seen_pairs: set[tuple[int, int]] = set()
    for row in frame.itertuples(index=False):
        concepts = parse_concepts(row.cpt_seq)
        if not concepts:
            raise ValueError(
                "Empty concept assignment for exercise "
                f"{row.exer_id} in {path}"
            )
        item = int(row.exer_id)
        for concept in concepts:
            pair = (item, concept)
            if pair in seen_pairs:
                raise ValueError(
                    f"Duplicate Q pair {pair} in {path}"
                )
            seen_pairs.add(pair)
            result[item].add(concept)
    return dict(result)


def load_protocol_ids(
    directory: str | Path,
    *,
    include_concepts: bool,
) -> tuple[dict[str, pd.DataFrame], dict[int, set[int]]]:
    """Load only train/validation IDs used by the activation statistic."""
    root = Path(directory)
    columns = (
        ["stu_id", "exer_id", "cpt_seq"]
        if include_concepts
        else ["stu_id", "exer_id"]
    )
    splits = {
        split: pd.read_csv(root / f"{split}.csv", usecols=columns)
        for split in ("train", "valid")
    }
    q_map = load_q_map(root / "Q_matrix.csv")
    q_items = set(q_map)
    for split, frame in splits.items():
        unknown = set(frame["exer_id"].astype(int)) - q_items
        if unknown:
            raise ValueError(
                f"{split} contains {len(unknown)} items absent from Q"
            )
    return splits, q_map


def load_protocol_data(
    directory: str | Path,
    *,
    include_concepts: bool,
) -> pd.DataFrame:
    columns = ["stu_id", "exer_id"]
    if include_concepts:
        columns.append("cpt_seq")
    return pd.read_csv(Path(directory) / "data.csv", usecols=columns)


def expanded_triplets(
    frames: Iterable[pd.DataFrame],
) -> Counter[tuple[int, int, int]]:
    result: Counter[tuple[int, int, int]] = Counter()
    for frame in frames:
        for row in frame.itertuples(index=False):
            for concept in parse_concepts(row.cpt_seq):
                result[(int(row.stu_id), int(row.exer_id), concept)] += 1
    return result


def interaction_pairs(
    frames: Iterable[pd.DataFrame],
) -> Counter[tuple[int, int]]:
    result: Counter[tuple[int, int]] = Counter()
    for frame in frames:
        result.update(
            (int(row.stu_id), int(row.exer_id))
            for row in frame.itertuples(index=False)
        )
    return result


def q_pairs(q_map: dict[int, set[int]]) -> set[tuple[int, int]]:
    return {
        (item, concept)
        for item, concepts in q_map.items()
        for concept in concepts
    }


@dataclass
class TypedGraph:
    adjacency: dict[Node, dict[Node, bool]]

    @classmethod
    def empty(cls) -> "TypedGraph":
        return cls(defaultdict(dict))

    def add_edge(
        self,
        left: Node,
        right: Node,
        *,
        metadata: bool,
    ) -> None:
        self.adjacency[left][right] = (
            self.adjacency[left].get(right, False) or metadata
        )
        self.adjacency[right][left] = (
            self.adjacency[right].get(left, False) or metadata
        )

    def add_q(self, q_map: dict[int, set[int]]) -> None:
        for item, concepts in q_map.items():
            for concept in concepts:
                self.add_edge(
                    ("item", item),
                    ("concept", concept),
                    metadata=False,
                )

    def reachable_concepts(
        self,
        source_items: Iterable[int],
        *,
        max_hops: int,
        require_metadata: bool,
    ) -> dict[int, int]:
        queue: deque[tuple[Node, bool, int]] = deque()
        visited: dict[tuple[Node, bool], int] = {}
        for item in source_items:
            node = ("item", int(item))
            visited[(node, False)] = 0
            queue.append((node, False, 0))
        reached: dict[int, int] = {}
        while queue:
            node, used_metadata, depth = queue.popleft()
            if (
                node[0] == "concept"
                and (
                    used_metadata or not require_metadata
                )
            ):
                concept = int(node[1])
                reached[concept] = min(
                    reached.get(concept, max_hops + 1),
                    depth,
                )
            if depth >= max_hops:
                continue
            for neighbor, edge_is_metadata in self.adjacency.get(
                node, {}
            ).items():
                next_used = used_metadata or edge_is_metadata
                state = (neighbor, next_used)
                next_depth = depth + 1
                if visited.get(state, max_hops + 1) <= next_depth:
                    continue
                visited[state] = next_depth
                queue.append((neighbor, next_used, next_depth))
        return reached


def add_filtered_group_edges(
    graph: TypedGraph,
    values_by_item: dict[int, set[str]],
    *,
    relation: str,
    num_items: int,
    max_group_fraction: float,
) -> dict[str, Any]:
    items_by_value: dict[str, set[int]] = defaultdict(set)
    for item, values in values_by_item.items():
        for value in values:
            items_by_value[str(value)].add(int(item))
    max_size = max(2, math.floor(num_items * max_group_fraction))
    admitted = {
        value: items
        for value, items in items_by_value.items()
        if 2 <= len(items) <= max_size
    }
    incident: set[int] = set()
    for value, items in admitted.items():
        metadata_node = (f"metadata:{relation}", value)
        for item in items:
            graph.add_edge(
                ("item", item),
                metadata_node,
                metadata=True,
            )
            incident.add(item)
    return {
        "raw_values": len(items_by_value),
        "admitted_values": len(admitted),
        "max_group_size": max_size,
        "incident_items": len(incident),
        "incident_item_fraction": (
            len(incident) / num_items if num_items else 0.0
        ),
    }


def _metadata_sets(
    frame: pd.DataFrame,
    *,
    raw_item_column: str,
    item_map: dict[Any, int],
    field: str,
    require_invariant: bool,
) -> tuple[dict[int, set[str]], dict[str, Any]]:
    values_by_item: dict[int, set[str]] = {}
    ambiguous = 0
    for raw_item, group in frame.dropna(subset=[field]).groupby(
        raw_item_column
    ):
        if raw_item not in item_map:
            continue
        values = {
            str(value)
            for value in group[field].dropna().tolist()
        }
        if require_invariant and len(values) != 1:
            ambiguous += 1
            continue
        if values:
            values_by_item[item_map[raw_item]] = values
    return values_by_item, {
        "mapped_items_with_values": len(values_by_item),
        "ambiguous_items_excluded": ambiguous,
        "require_invariant": require_invariant,
    }


def reconstruct_assist09(
    raw_path: str | Path,
    protocol_dir: str | Path,
) -> tuple[dict[str, Any], dict[int, dict[str, set[str]]]]:
    fields = [
        "user_id",
        "problem_id",
        "skill_id",
        "template_id",
        "assistment_id",
    ]
    raw = pd.read_csv(
        raw_path,
        usecols=fields,
        encoding="latin1",
        low_memory=False,
    )
    base = (
        raw.dropna(subset=["skill_id"])
        .groupby(
            "user_id",
            group_keys=False,
            sort=True,
        )
        .apply(
            lambda group: group.drop_duplicates(
                "problem_id",
                keep="first",
            )
        )
        .reset_index(drop=True)
    )
    user_counts = base.groupby("user_id").size()
    base = base[
        base["user_id"].isin(user_counts[user_counts >= 15].index)
    ].copy()
    student_values = list(pd.unique(base["user_id"]))
    item_values = list(pd.unique(base["problem_id"]))
    student_map = {
        value: index for index, value in enumerate(student_values)
    }
    item_map = {
        value: index for index, value in enumerate(item_values)
    }
    raw_concepts = sorted(
        {
            concept
            for value in base["skill_id"]
            for concept in parse_concepts(value)
        }
    )
    concept_map = {
        value: index for index, value in enumerate(raw_concepts)
    }
    reconstructed: Counter[tuple[int, int, int]] = Counter()
    for row in base.itertuples(index=False):
        for raw_concept in parse_concepts(row.skill_id):
            reconstructed[
                (
                    student_map[row.user_id],
                    item_map[row.problem_id],
                    concept_map[raw_concept],
                )
            ] += 1

    current_q = load_q_map(Path(protocol_dir) / "Q_matrix.csv")
    current = expanded_triplets(
        [load_protocol_data(protocol_dir, include_concepts=True)]
    )
    reconstructed_q = {
        (item_map[row.problem_id], concept_map[raw_concept])
        for row in base.itertuples(index=False)
        for raw_concept in parse_concepts(row.skill_id)
    }
    metadata: dict[int, dict[str, set[str]]] = defaultdict(dict)
    field_reports: dict[str, Any] = {}
    for field in ("template_id", "assistment_id"):
        values, report = _metadata_sets(
            raw,
            raw_item_column="problem_id",
            item_map=item_map,
            field=field,
            require_invariant=True,
        )
        field_reports[field] = report
        for item, item_values_set in values.items():
            metadata[item][field] = item_values_set
    exact = (
        reconstructed == current
        and reconstructed_q == q_pairs(current_q)
    )
    return {
        "identity_exact": exact,
        "expanded_interaction_triplets": sum(current.values()),
        "students": len(student_map),
        "mapped_items": len(item_map),
        "current_items": len(current_q),
        "mapped_item_fraction": (
            len(set(item_map.values()) & set(current_q))
            / len(current_q)
        ),
        "concepts": len(concept_map),
        "triplet_multiset_exact": reconstructed == current,
        "q_pair_set_exact": reconstructed_q == q_pairs(current_q),
        "metadata_fields": field_reports,
    }, dict(metadata)


def reconstruct_assist17(
    raw_path: str | Path,
    protocol_dir: str | Path,
) -> tuple[dict[str, Any], dict[int, dict[str, set[str]]]]:
    fields = [
        "studentId",
        "problemId",
        "skill",
        "startTime",
        "problemType",
    ]
    raw = pd.read_csv(
        raw_path,
        usecols=fields,
        encoding="utf-8",
        low_memory=False,
    )
    raw["startTime"] = pd.to_datetime(
        raw["startTime"],
        errors="coerce",
    )
    base = (
        raw.groupby(
            "studentId",
            group_keys=False,
            sort=True,
        )
        .apply(
            lambda group: group.sort_values("startTime").drop_duplicates(
                "problemId",
                keep="first",
            )
        )
        .reset_index(drop=True)
    )
    base = (
        base.dropna(subset=["skill"])
        .drop_duplicates(["studentId", "problemId"], keep="first")
    )
    user_counts = base.groupby("studentId").size()
    base = base[
        base["studentId"].isin(user_counts[user_counts >= 15].index)
    ].copy()
    student_values = list(pd.unique(base["studentId"]))
    item_values = list(pd.unique(base["problemId"]))
    concept_values = list(pd.unique(base["skill"]))
    student_map = {
        value: index for index, value in enumerate(student_values)
    }
    item_map = {
        value: index for index, value in enumerate(item_values)
    }
    concept_map = {
        value: index for index, value in enumerate(concept_values)
    }
    reconstructed = Counter(
        (
            student_map[row.studentId],
            item_map[row.problemId],
            concept_map[row.skill],
        )
        for row in base.itertuples(index=False)
    )
    current_q = load_q_map(Path(protocol_dir) / "Q_matrix.csv")
    current = expanded_triplets(
        [load_protocol_data(protocol_dir, include_concepts=True)]
    )
    reconstructed_q = {
        (item_map[row.problemId], concept_map[row.skill])
        for row in base.itertuples(index=False)
    }
    values, field_report = _metadata_sets(
        raw,
        raw_item_column="problemId",
        item_map=item_map,
        field="problemType",
        require_invariant=True,
    )
    metadata = {
        item: {"problem_type": item_values_set}
        for item, item_values_set in values.items()
    }
    exact = (
        reconstructed == current
        and reconstructed_q == q_pairs(current_q)
    )
    return {
        "identity_exact": exact,
        "interactions": sum(current.values()),
        "students": len(student_map),
        "mapped_items": len(item_map),
        "current_items": len(current_q),
        "mapped_item_fraction": (
            len(set(item_map.values()) & set(current_q))
            / len(current_q)
        ),
        "concepts": len(concept_map),
        "triplet_multiset_exact": reconstructed == current,
        "q_pair_set_exact": reconstructed_q == q_pairs(current_q),
        "metadata_fields": {"problem_type": field_report},
    }, metadata


def parse_subjects(value: Any) -> set[int]:
    return set(parse_concepts(str(value).replace(" ", "")))


def reconstruct_nips(
    question_path: str | Path,
    subject_path: str | Path,
    protocol_dir: str | Path,
) -> tuple[dict[str, Any], list[tuple[int, int]]]:
    questions = pd.read_csv(
        question_path,
        usecols=["QuestionId", "SubjectId"],
    )
    subjects = pd.read_csv(
        subject_path,
        usecols=["SubjectId", "ParentId", "Level"],
        encoding="utf-8-sig",
    )
    current_q = load_q_map(Path(protocol_dir) / "Q_matrix.csv")
    raw_q = {
        int(row.QuestionId): parse_subjects(row.SubjectId)
        for row in questions.itertuples(index=False)
    }
    current_items = set(current_q)
    item_identity = current_items == set(raw_q)
    raw_concepts = sorted(set().union(*raw_q.values()))
    solutions: list[tuple[int | None, dict[int, int]]] = []
    for excluded in [None, *raw_concepts]:
        retained = [
            value for value in raw_concepts if value != excluded
        ]
        if len(retained) != len(set().union(*current_q.values())):
            continue
        mapping = {
            raw_value: index
            for index, raw_value in enumerate(retained)
        }
        if all(
            {
                mapping[value]
                for value in raw_q[item]
                if value in mapping
            }
            == current_q[item]
            for item in current_items
        ):
            solutions.append((excluded, mapping))
    if len(solutions) != 1:
        raise RuntimeError(
            "NIPS concept encoding is not unique: "
            f"{len(solutions)} solutions"
        )
    excluded, mapping = solutions[0]
    parent_by_subject: dict[int, int] = {}
    for row in subjects.itertuples(index=False):
        if pd.isna(row.ParentId):
            continue
        parent_by_subject[int(row.SubjectId)] = int(row.ParentId)
    hierarchy_edges = sorted(
        {
            (mapping[child], mapping[parent])
            for child, parent in parent_by_subject.items()
            if child in mapping
            and parent in mapping
            and child != parent
        }
    )
    subject_ids = set(subjects["SubjectId"].astype(int))
    subject_coverage = len(set(mapping) & subject_ids) / len(mapping)
    return {
        "identity_exact": (
            item_identity
            and len(solutions) == 1
            and subject_coverage == 1.0
        ),
        "mapped_items": len(current_items & set(raw_q)),
        "current_items": len(current_items),
        "mapped_item_fraction": (
            len(current_items & set(raw_q)) / len(current_items)
        ),
        "concepts": len(mapping),
        "excluded_raw_subject": excluded,
        "subject_metadata_coverage": subject_coverage,
        "hierarchy_edges": len(hierarchy_edges),
    }, hierarchy_edges


def read_flat_pairs(path: str | Path) -> list[tuple[int, int]]:
    tokens = [
        int(token)
        for token in Path(path).read_text(encoding="utf-8").split()
    ]
    if len(tokens) % 2:
        raise ValueError(f"Odd number of graph tokens in {path}")
    return list(zip(tokens[0::2], tokens[1::2]))


def reconstruct_junyi(
    official_log_path: str | Path,
    directed_path: str | Path,
    undirected_path: str | Path,
    protocol_dir: str | Path,
) -> tuple[
    dict[str, Any],
    list[tuple[int, int]],
    list[tuple[int, int]],
]:
    current_q = load_q_map(Path(protocol_dir) / "Q_matrix.csv")
    current_pairs = interaction_pairs(
        [load_protocol_data(protocol_dir, include_concepts=False)]
    )
    official = json.loads(
        Path(official_log_path).read_text(encoding="utf-8")
    )
    source_pairs: Counter[tuple[int, int]] = Counter()
    source_item_concept_identity = True
    for user_record in official:
        student = int(user_record["user_id"]) - 1
        for record in user_record["logs"]:
            item = int(record["exer_id"]) - 1
            concept = int(record["knowledge_code"]) - 1
            source_item_concept_identity &= item == concept
            source_pairs[
                (student, item)
            ] += 1
    identity_q = all(
        concepts == {item}
        for item, concepts in current_q.items()
    )
    current_concepts = set(current_q)
    directed_raw = read_flat_pairs(directed_path)
    undirected_raw = read_flat_pairs(undirected_path)
    directed = sorted(
        {
            (left, right)
            for left, right in directed_raw
            if left != right
            and left in current_concepts
            and right in current_concepts
        }
    )
    undirected = sorted(
        {
            tuple(sorted((left, right)))
            for left, right in undirected_raw
            if left != right
            and left in current_concepts
            and right in current_concepts
        }
    )
    incident = {
        node
        for edge in [*directed, *undirected]
        for node in edge
    }
    exact = (
        source_pairs == current_pairs
        and identity_q
        and source_item_concept_identity
    )
    return {
        "identity_exact": exact,
        "pair_multiset_exact": source_pairs == current_pairs,
        "item_concept_identity_exact": identity_q,
        "source_item_concept_identity_exact": (
            source_item_concept_identity
        ),
        "mapped_items": len(current_q),
        "current_items": len(current_q),
        "mapped_item_fraction": 1.0,
        "directed_raw_pairs": len(directed_raw),
        "directed_induced_unique_nonself": len(directed),
        "undirected_raw_pairs": len(undirected_raw),
        "undirected_induced_unique_nonself": len(undirected),
        "relation_incident_items": len(incident),
        "relation_incident_item_fraction": (
            len(incident) / len(current_q)
        ),
    }, directed, undirected


def target_reachability(
    graph: TypedGraph,
    q_only_graph: TypedGraph,
    splits: dict[str, pd.DataFrame],
    q_map: dict[int, set[int]],
    *,
    target_scope: str,
    max_hops: int,
) -> dict[str, Any]:
    history_items: dict[int, set[int]] = defaultdict(set)
    seen_concepts: dict[int, set[int]] = defaultdict(set)
    for row in splits["train"].itertuples(index=False):
        student = int(row.stu_id)
        item = int(row.exer_id)
        history_items[student].add(item)
        seen_concepts[student].update(q_map.get(item, set()))

    target_records: list[tuple[int, set[int]]] = []
    for row in splits["valid"].itertuples(index=False):
        student = int(row.stu_id)
        target = q_map.get(int(row.exer_id), set())
        if not target:
            continue
        seen = seen_concepts.get(student, set())
        coverage = len(target & seen) / len(target)
        eligible = (
            coverage == 0.0
            if target_scope == "exact_zero"
            else coverage < 0.5
        )
        if eligible:
            target_records.append((student, target - seen))

    target_students = {
        student for student, _ in target_records
    }
    metadata_reached_by_student = {
        student: graph.reachable_concepts(
            items,
            max_hops=max_hops,
            require_metadata=True,
        )
        for student, items in history_items.items()
        if student in target_students
    }
    q_reached_by_student = {
        student: q_only_graph.reachable_concepts(
            items,
            max_hops=max_hops,
            require_metadata=False,
        )
        for student, items in history_items.items()
        if student in target_students
    }
    stats: dict[str, dict[str, Any]] = {
        name: {
            "any_rows": 0,
            "all_rows": 0,
            "reachable_concepts": 0,
            "shortest_path_rows": Counter(),
        }
        for name in ("metadata_path", "q_only", "incremental")
    }
    missing_total = 0
    for student, missing in target_records:
        metadata_distances = metadata_reached_by_student.get(
            student, {}
        )
        q_distances = q_reached_by_student.get(student, {})
        metadata_reachable = set(missing) & set(metadata_distances)
        q_reachable = set(missing) & set(q_distances)
        incremental = metadata_reachable - q_reachable
        missing_total += len(missing)
        for name, reachable, distances in (
            (
                "metadata_path",
                metadata_reachable,
                metadata_distances,
            ),
            ("q_only", q_reachable, q_distances),
            ("incremental", incremental, metadata_distances),
        ):
            current = stats[name]
            current["reachable_concepts"] += len(reachable)
            if reachable:
                current["any_rows"] += 1
                current["shortest_path_rows"][
                    min(distances[concept] for concept in reachable)
                ] += 1
            if missing and len(reachable) == len(missing):
                current["all_rows"] += 1

    count = len(target_records)
    finalized: dict[str, dict[str, Any]] = {}
    for name, values in stats.items():
        finalized[name] = {
            "any_rows": values["any_rows"],
            "any_row_fraction": (
                values["any_rows"] / count if count else 0.0
            ),
            "all_rows": values["all_rows"],
            "all_row_fraction": (
                values["all_rows"] / count if count else 0.0
            ),
            "reachable_concepts": values["reachable_concepts"],
            "reachable_concept_fraction": (
                values["reachable_concepts"] / missing_total
                if missing_total
                else 0.0
            ),
            "shortest_path_rows": {
                str(key): value
                for key, value in sorted(
                    values["shortest_path_rows"].items()
                )
            },
        }
    return {
        "scope": target_scope,
        "validation_target_rows": count,
        "missing_concepts": missing_total,
        **finalized,
        "max_hops": max_hops,
    }


def dataset_gate(
    identity: dict[str, Any],
    reachability: dict[str, Any],
    *,
    min_reachable_fraction: float,
) -> dict[str, Any]:
    checks = {
        "identity_exact": bool(identity["identity_exact"]),
        "mapped_item_fraction_at_least_0_99": (
            identity["mapped_item_fraction"] >= 0.99
        ),
        "target_rows_at_least_100": (
            reachability["validation_target_rows"] >= 100
        ),
        "incremental_reachable_fraction_at_least_threshold": (
            reachability["incremental"]["any_row_fraction"]
            >= min_reachable_fraction
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Outcome-blind static metadata admission audit."
        )
    )
    parser.add_argument("--assist09-protocol", required=True)
    parser.add_argument("--assist09-raw", required=True)
    parser.add_argument("--assist17-protocol", required=True)
    parser.add_argument("--assist17-raw", required=True)
    parser.add_argument("--nips-protocol", required=True)
    parser.add_argument(
        "--nips-question-metadata",
        required=True,
    )
    parser.add_argument(
        "--nips-subject-metadata",
        required=True,
    )
    parser.add_argument("--junyi-protocol", required=True)
    parser.add_argument("--junyi-official-log", required=True)
    parser.add_argument("--junyi-directed", required=True)
    parser.add_argument("--junyi-undirected", required=True)
    parser.add_argument("--junyi-source-repo", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def fingerprint_paths(
    paths: dict[str, str],
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "path": str(Path(path).resolve()),
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }


def protocol_fingerprint_paths(
    prefix: str,
    directory: str | Path,
) -> dict[str, str]:
    root = Path(directory)
    return {
        f"{prefix}_{name.removesuffix('.csv')}": str(root / name)
        for name in (
            "data.csv",
            "train.csv",
            "valid.csv",
            "Q_matrix.csv",
        )
    }


def audit_input_paths(args: argparse.Namespace) -> dict[str, str]:
    paths = {
        "assist09_raw": args.assist09_raw,
        "assist17_raw": args.assist17_raw,
        "nips_question_metadata": args.nips_question_metadata,
        "nips_subject_metadata": args.nips_subject_metadata,
        "junyi_official_log": args.junyi_official_log,
        "junyi_directed": args.junyi_directed,
        "junyi_undirected": args.junyi_undirected,
    }
    for prefix, directory in (
        ("assist09", args.assist09_protocol),
        ("assist17", args.assist17_protocol),
        ("nips", args.nips_protocol),
        ("junyi", args.junyi_protocol),
    ):
        paths.update(protocol_fingerprint_paths(prefix, directory))
    return paths


def verify_frozen_inputs(paths: dict[str, str]) -> None:
    if set(paths) != set(FROZEN_INPUT_SHA256):
        missing = sorted(set(FROZEN_INPUT_SHA256) - set(paths))
        unexpected = sorted(set(paths) - set(FROZEN_INPUT_SHA256))
        raise RuntimeError(
            "Frozen input key mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )
    mismatches = {
        name: {
            "expected": FROZEN_INPUT_SHA256[name],
            "actual": sha256_file(path),
        }
        for name, path in paths.items()
        if sha256_file(path) != FROZEN_INPUT_SHA256[name]
    }
    if mismatches:
        raise RuntimeError(
            "Frozen input SHA-256 mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )


def verify_junyi_git_source(
    repo: str | Path,
    paths: dict[str, str],
) -> dict[str, str]:
    verified: dict[str, str] = {}
    for name, relative_path in JUNYI_TRACKED_FILES.items():
        result = subprocess.run(
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
        blob_sha256 = sha256_bytes(result.stdout)
        if blob_sha256 != FROZEN_INPUT_SHA256[name]:
            raise RuntimeError(
                f"Frozen Junyi Git blob mismatch for {relative_path}"
            )
        if sha256_file(paths[name]) != blob_sha256:
            raise RuntimeError(
                f"Junyi working source differs from {relative_path}"
            )
        verified[relative_path] = blob_sha256
    return verified


def main() -> None:
    args = parse_args()
    input_paths = audit_input_paths(args)
    verify_frozen_inputs(input_paths)
    junyi_git_blobs = verify_junyi_git_source(
        args.junyi_source_repo,
        input_paths,
    )
    reports: dict[str, Any] = {}

    a09_identity, a09_metadata = reconstruct_assist09(
        args.assist09_raw,
        args.assist09_protocol,
    )
    a09_splits, a09_q = load_protocol_ids(
        args.assist09_protocol,
        include_concepts=False,
    )
    a09_graph = TypedGraph.empty()
    a09_graph.add_q(a09_q)
    a09_q_only_graph = TypedGraph.empty()
    a09_q_only_graph.add_q(a09_q)
    a09_relation_reports = {}
    for relation in ("template_id", "assistment_id"):
        values = {
            item: fields[relation]
            for item, fields in a09_metadata.items()
            if relation in fields
        }
        a09_relation_reports[relation] = (
            add_filtered_group_edges(
                a09_graph,
                values,
                relation=relation,
                num_items=len(a09_q),
                max_group_fraction=FROZEN_MAX_GROUP_FRACTION,
            )
        )
    a09_reach = target_reachability(
        a09_graph,
        a09_q_only_graph,
        a09_splits,
        a09_q,
        target_scope="exact_zero",
        max_hops=FROZEN_MAX_HOPS,
    )
    reports["ASSIST09"] = {
        "identity": a09_identity,
        "relations": a09_relation_reports,
        "reachability": a09_reach,
    }

    a17_identity, a17_metadata = reconstruct_assist17(
        args.assist17_raw,
        args.assist17_protocol,
    )
    a17_splits, a17_q = load_protocol_ids(
        args.assist17_protocol,
        include_concepts=False,
    )
    a17_graph = TypedGraph.empty()
    a17_graph.add_q(a17_q)
    a17_q_only_graph = TypedGraph.empty()
    a17_q_only_graph.add_q(a17_q)
    a17_values = {
        item: fields["problem_type"]
        for item, fields in a17_metadata.items()
        if "problem_type" in fields
    }
    a17_relations = {
        "problem_type": add_filtered_group_edges(
            a17_graph,
            a17_values,
            relation="problem_type",
            num_items=len(a17_q),
            max_group_fraction=FROZEN_MAX_GROUP_FRACTION,
        )
    }
    a17_reach = target_reachability(
        a17_graph,
        a17_q_only_graph,
        a17_splits,
        a17_q,
        target_scope="exact_zero",
        max_hops=FROZEN_MAX_HOPS,
    )
    reports["ASSIST17"] = {
        "identity": a17_identity,
        "relations": a17_relations,
        "reachability": a17_reach,
    }

    nips_identity, nips_edges = reconstruct_nips(
        args.nips_question_metadata,
        args.nips_subject_metadata,
        args.nips_protocol,
    )
    nips_splits, nips_q = load_protocol_ids(
        args.nips_protocol,
        include_concepts=False,
    )
    nips_graph = TypedGraph.empty()
    nips_graph.add_q(nips_q)
    nips_q_only_graph = TypedGraph.empty()
    nips_q_only_graph.add_q(nips_q)
    for child, parent in nips_edges:
        nips_graph.add_edge(
            ("concept", child),
            ("concept", parent),
            metadata=True,
        )
    nips_reach = target_reachability(
        nips_graph,
        nips_q_only_graph,
        nips_splits,
        nips_q,
        target_scope="low_coverage",
        max_hops=FROZEN_MAX_HOPS,
    )
    reports["NIPS34"] = {
        "identity": nips_identity,
        "relations": {
            "subject_hierarchy": {"edges": len(nips_edges)}
        },
        "reachability": nips_reach,
    }

    (
        junyi_identity,
        junyi_directed,
        junyi_undirected,
    ) = reconstruct_junyi(
        args.junyi_official_log,
        args.junyi_directed,
        args.junyi_undirected,
        args.junyi_protocol,
    )
    junyi_splits, junyi_q = load_protocol_ids(
        args.junyi_protocol,
        include_concepts=False,
    )
    junyi_graph = TypedGraph.empty()
    junyi_graph.add_q(junyi_q)
    junyi_q_only_graph = TypedGraph.empty()
    junyi_q_only_graph.add_q(junyi_q)
    for left, right in [
        *junyi_directed,
        *junyi_undirected,
    ]:
        junyi_graph.add_edge(
            ("concept", left),
            ("concept", right),
            metadata=True,
        )
    junyi_reach = target_reachability(
        junyi_graph,
        junyi_q_only_graph,
        junyi_splits,
        junyi_q,
        target_scope="exact_zero",
        max_hops=FROZEN_MAX_HOPS,
    )
    reports["Junyi"] = {
        "identity": junyi_identity,
        "relations": {
            "prerequisite": {"edges": len(junyi_directed)},
            "similarity": {"edges": len(junyi_undirected)},
        },
        "reachability": junyi_reach,
    }

    for report in reports.values():
        report["gate"] = dataset_gate(
            report["identity"],
            report["reachability"],
            min_reachable_fraction=FROZEN_MIN_REACHABLE_FRACTION,
        )
    passing = sum(
        report["gate"]["passed"]
        for report in reports.values()
    )
    payload = {
        "schema_version": 1,
        "audit_policy": {
            "outcome_blind": True,
            "response_outcomes_extracted_inspected_or_used": False,
            "whole_files_hashed_for_provenance": True,
            "provenance_uses_test_identifiers": False,
            "reachability_split": "validation",
            "reachability_gate": "incremental_over_q_only",
            "max_hops": FROZEN_MAX_HOPS,
            "max_group_fraction": FROZEN_MAX_GROUP_FRACTION,
            "min_reachable_fraction": FROZEN_MIN_REACHABLE_FRACTION,
            "min_passing_datasets": FROZEN_MIN_PASSING_DATASETS,
        },
        "source_versions": {
            "junyi_rcd_commit": FROZEN_JUNYI_COMMIT,
            "junyi_git_blobs": junyi_git_blobs,
        },
        "frozen_inputs_verified": True,
        "fingerprints": fingerprint_paths(input_paths),
        "reports": reports,
        "passing_datasets": passing,
        "route_activated": (
            passing >= FROZEN_MIN_PASSING_DATASETS
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
