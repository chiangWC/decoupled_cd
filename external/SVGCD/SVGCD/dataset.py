import csv
import io
import json
import os
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, default_collate


class ResponseDataset(Dataset):
    def __init__(self, triplets, q_matrix):
        self.triplets = triplets
        self.q_matrix = q_matrix

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        stu, exer, label = self.triplets[idx]
        return {
            "stu_id": torch.tensor(stu, dtype=torch.long),
            "exer_id": torch.tensor(exer, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float32),
            "Q_mat": self.q_matrix,
        }

    @staticmethod
    def collate_fn(batch):
        elem = batch[0]
        ret = {key: default_collate([d[key] for d in batch]) for key in elem if key != "Q_mat"}
        ret["Q_mat"] = elem["Q_mat"]
        return ret


class CognitiveDataProcessor:
    def __init__(
        self,
        args,
        logger,
        *,
        split_mode="all",
        id_maps_path=None,
        q_matrix_path=None,
        id_maps_payload=None,
        q_matrix_bytes=None,
    ):
        if split_mode not in {"all", "train", "valid", "test"}:
            raise ValueError("split_mode must be all, train, valid, or test")
        if split_mode == "test" and id_maps_path is None and id_maps_payload is None:
            raise ValueError("test mode requires frozen ID maps")
        if split_mode != "all" and q_matrix_path is None and q_matrix_bytes is None:
            raise ValueError("plugin split mode requires Q-matrix bytes or path")
        self.args = args
        self.logger = logger
        self.data_dir = args.data_dir
        self.split_mode = split_mode
        self.id_maps_path = id_maps_path
        self.q_matrix_path = q_matrix_path
        self.id_maps_payload = id_maps_payload
        self.q_matrix_bytes = q_matrix_bytes

        self.logger.info(">>> Processing Data...")
        self.train_rows = self._read_csv(args.train_file)
        self.valid_rows = self._read_csv(args.valid_file)
        schema_rows = self.train_rows + self.valid_rows
        self.test_rows = (
            self._read_csv(args.test_file)
            if split_mode in {"all", "test"}
            else []
        )
        mapping_rows = (
            schema_rows + self.test_rows if split_mode == "all" else schema_rows
        )

        if split_mode == "all":
            q_rows = mapping_rows
            if id_maps_path is not None or id_maps_payload is not None:
                frozen = self._load_frozen_ids()
                self.stu_ids = frozen["stu_ids"]
                self.exer_ids = frozen["exer_ids"]
                self.cpt_ids = frozen["cpt_ids"]
            else:
                self.stu_ids = [
                    str(value)
                    for value in sorted(
                        {int(row["stu_id"]) for row in mapping_rows}
                    )
                ]
                self.exer_ids = [
                    str(value)
                    for value in sorted(
                        {int(row["exer_id"]) for row in mapping_rows}
                    )
                ]
                all_concepts = set()
                for row in mapping_rows:
                    all_concepts.update(self._parse_cpts(row["cpt_seq"]))
                self.cpt_ids = [str(value) for value in sorted(all_concepts)]
        else:
            q_rows = self._read_q_matrix()
            derived_stu_ids = self._sorted_tokens(r["stu_id"] for r in schema_rows)
            derived_exer_ids = self._sorted_tokens(r["exer_id"] for r in q_rows)
            all_concepts = set()
            for row in q_rows:
                all_concepts.update(self._parse_cpts(row["cpt_seq"]))
            derived_cpt_ids = self._sorted_tokens(all_concepts)
            derived = {
                "stu_ids": derived_stu_ids,
                "exer_ids": derived_exer_ids,
                "cpt_ids": derived_cpt_ids,
            }
            if id_maps_path is not None or id_maps_payload is not None:
                frozen = self._load_frozen_ids()
                for field, expected in derived.items():
                    if frozen[field] != expected:
                        raise ValueError(
                            f"frozen ID schema {field} does not match "
                            "train+valid/Q schema"
                        )
                self.stu_ids = frozen["stu_ids"]
                self.exer_ids = frozen["exer_ids"]
                self.cpt_ids = frozen["cpt_ids"]
            else:
                self.stu_ids = derived_stu_ids
                self.exer_ids = derived_exer_ids
                self.cpt_ids = derived_cpt_ids

        self.stu2idx = {stu_id: idx for idx, stu_id in enumerate(self.stu_ids)}
        self.exer2idx = {exer_id: idx for idx, exer_id in enumerate(self.exer_ids)}
        self.cpt2idx = {cpt_id: idx for idx, cpt_id in enumerate(self.cpt_ids)}

        self.num_students = len(self.stu_ids)
        self.num_exercises = len(self.exer_ids)
        self.num_concepts = len(self.cpt_ids)

        self.q_matrix = torch.zeros(self.num_exercises, self.num_concepts, dtype=torch.float32)
        for row in q_rows:
            exer = self._lookup(self.exer2idx, row["exer_id"], "exercise")
            for c in self._parse_cpts(row["cpt_seq"]):
                concept = self._lookup(self.cpt2idx, c, "concept")
                self.q_matrix[exer, concept] = 1.0

        if split_mode != "all":
            for rows in (self.train_rows, self.valid_rows, self.test_rows):
                self._validate_q_coverage(rows, q_rows)

        self.train_triplets = self._rows_to_triplets(self.train_rows)
        self.valid_triplets = self._rows_to_triplets(self.valid_rows)
        self.test_triplets = self._rows_to_triplets(self.test_rows)
        self._validate_concepts(self.test_rows)

        self.student_interaction_counts = Counter(stu for stu, _, _ in self.train_triplets)
        self.correct_adj = self._build_semantic_adj(1)
        self.wrong_adj = self._build_semantic_adj(0)

        self.logger.info(
            f"Stats: Stu={self.num_students}, Exer={self.num_exercises}, Cpt={self.num_concepts}"
        )

    def _read_csv(self, filename):
        path = os.path.join(self.data_dir, filename)
        with open(path, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def _read_q_matrix(self):
        path = Path(self.q_matrix_path) if self.q_matrix_path is not None else None
        if self.q_matrix_bytes is not None:
            handle = io.StringIO(self.q_matrix_bytes.decode("utf-8"), newline="")
        else:
            try:
                handle = path.open("r", encoding="utf-8", newline="")
            except OSError as exc:
                raise ValueError(f"cannot read Q-matrix: {path}") from exc
        with handle:
            reader = csv.DictReader(handle)
            missing = {"exer_id", "cpt_seq"} - set(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"Q-matrix missing required columns: {', '.join(sorted(missing))}"
                )
            rows = list(reader)
        if not rows:
            raise ValueError("Q-matrix must not be empty")
        return rows

    @staticmethod
    def _token(value):
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return str(value)

    @classmethod
    def _sorted_tokens(cls, values):
        tokens = {cls._token(value) for value in values}

        def key(token):
            try:
                return 0, int(token), token
            except ValueError:
                return 1, token, token

        return sorted(tokens, key=key)

    def _load_frozen_ids(self):
        if self.id_maps_payload is not None:
            payload = self.id_maps_payload
        else:
            try:
                with open(self.id_maps_path, encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"cannot read frozen ID schema: {self.id_maps_path}"
                ) from exc
        if not isinstance(payload, dict):
            raise ValueError("frozen ID schema must be an object")
        ids = {}
        for field in ("stu_ids", "exer_ids", "cpt_ids"):
            values = payload.get(field)
            if not isinstance(values, list) or not values:
                raise ValueError(f"frozen ID schema requires {field}")
            normalized = [self._token(value) for value in values]
            if len(normalized) != len(set(normalized)):
                raise ValueError(f"frozen ID schema has duplicate {field}")
            ids[field] = normalized
        return ids

    def _lookup(self, mapping, value, kind):
        token = self._token(value)
        try:
            return mapping[token]
        except KeyError as exc:
            raise ValueError(
                f"unknown {kind} ID {token!r} in {self.split_mode} split"
            ) from exc

    @staticmethod
    def _parse_cpts(cpt_seq):
        return [int(x) for x in str(cpt_seq).split(",")]

    def _rows_to_triplets(self, rows):
        triplets = []
        for r in rows:
            triplets.append(
                (
                    self._lookup(self.stu2idx, r["stu_id"], "student"),
                    self._lookup(self.exer2idx, r["exer_id"], "exercise"),
                    int(r["label"]),
                )
            )
        return triplets

    def _validate_q_coverage(self, rows, q_rows):
        q_concepts = {}
        for row in q_rows:
            exercise = self._token(row["exer_id"])
            q_concepts.setdefault(exercise, set()).update(
                self._token(concept)
                for concept in self._parse_cpts(row["cpt_seq"])
            )
        for row in rows:
            exercise = self._token(row["exer_id"])
            self._lookup(self.exer2idx, exercise, "exercise")
            for concept in self._parse_cpts(row["cpt_seq"]):
                concept_token = self._token(concept)
                self._lookup(self.cpt2idx, concept_token, "concept")
                if concept_token not in q_concepts[exercise]:
                    raise ValueError(
                        f"concept ID {concept_token!r} is not in Q for "
                        f"exercise ID {exercise!r}"
                    )

    def _validate_concepts(self, rows):
        for row in rows:
            for concept in self._parse_cpts(row["cpt_seq"]):
                self._lookup(self.cpt2idx, concept, "concept")

    def _build_semantic_adj(self, interaction_label):
        num_nodes = self.num_students + self.num_exercises
        deg = torch.zeros(num_nodes, dtype=torch.float32)
        edges = []

        for stu, exer, label in self.train_triplets:
            if label != interaction_label:
                continue
            e_node = self.num_students + exer
            edges.append((stu, e_node))
            edges.append((e_node, stu))
            deg[stu] += 1
            deg[e_node] += 1

        rows, cols, vals = [], [], []
        for src, dst in edges:
            src_deg = deg[src].item() + 1.0
            dst_deg = deg[dst].item() + 1.0
            vals.append(float((src_deg * dst_deg) ** -0.5))
            rows.append(src)
            cols.append(dst)

        if not rows:
            indices = torch.zeros((2, 0), dtype=torch.long)
            values = torch.zeros((0,), dtype=torch.float32)
        else:
            indices = torch.tensor([rows, cols], dtype=torch.long)
            values = torch.tensor(vals, dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()

    def get_loaders(self):
        train_loader = DataLoader(
            ResponseDataset(self.train_triplets, self.q_matrix),
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=ResponseDataset.collate_fn,
        )
        valid_loader = DataLoader(
            ResponseDataset(self.valid_triplets, self.q_matrix),
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=ResponseDataset.collate_fn,
        )
        test_loader = DataLoader(
            ResponseDataset(self.test_triplets, self.q_matrix),
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=ResponseDataset.collate_fn,
        )
        return train_loader, valid_loader, test_loader
