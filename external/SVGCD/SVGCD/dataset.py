import csv
import json
import os
from collections import Counter

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
    def __init__(self, args, logger, *, split_mode="all", id_maps_path=None):
        if split_mode not in {"all", "train", "valid", "test"}:
            raise ValueError("split_mode must be all, train, valid, or test")
        if split_mode == "test" and id_maps_path is None:
            raise ValueError("test mode requires frozen id_maps_path")
        self.args = args
        self.logger = logger
        self.data_dir = args.data_dir
        self.split_mode = split_mode
        self.id_maps_path = id_maps_path

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

        if id_maps_path is not None:
            frozen = self._load_frozen_ids()
            self.stu_ids = frozen["stu_ids"]
            self.exer_ids = frozen["exer_ids"]
            self.cpt_ids = frozen["cpt_ids"]
        else:
            self.stu_ids = [str(value) for value in sorted({int(r["stu_id"]) for r in mapping_rows})]
            self.exer_ids = [str(value) for value in sorted({int(r["exer_id"]) for r in mapping_rows})]
            all_concepts = set()
            for row in mapping_rows:
                all_concepts.update(self._parse_cpts(row["cpt_seq"]))
            self.cpt_ids = [str(value) for value in sorted(all_concepts)]

        self.stu2idx = {stu_id: idx for idx, stu_id in enumerate(self.stu_ids)}
        self.exer2idx = {exer_id: idx for idx, exer_id in enumerate(self.exer_ids)}
        self.cpt2idx = {cpt_id: idx for idx, cpt_id in enumerate(self.cpt_ids)}

        self.num_students = len(self.stu_ids)
        self.num_exercises = len(self.exer_ids)
        self.num_concepts = len(self.cpt_ids)

        self.q_matrix = torch.zeros(self.num_exercises, self.num_concepts, dtype=torch.float32)
        for row in mapping_rows:
            exer = self._lookup(self.exer2idx, row["exer_id"], "exercise")
            for c in self._parse_cpts(row["cpt_seq"]):
                concept = self._lookup(self.cpt2idx, c, "concept")
                self.q_matrix[exer, concept] = 1.0

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

    @staticmethod
    def _token(value):
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return str(value)

    def _load_frozen_ids(self):
        try:
            with open(self.id_maps_path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read frozen ID schema: {self.id_maps_path}") from exc
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
