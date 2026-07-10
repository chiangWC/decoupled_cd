import csv
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
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.data_dir = args.data_dir

        self.logger.info(">>> Processing Data...")
        self.train_rows = self._read_csv(args.train_file)
        self.valid_rows = self._read_csv(args.valid_file)
        self.test_rows = self._read_csv(args.test_file)

        id_rows = self.train_rows + self.valid_rows + self.test_rows
        self.stu_ids = sorted({int(r["stu_id"]) for r in id_rows})
        self.exer_ids = sorted({int(r["exer_id"]) for r in id_rows})
        all_concepts = set()
        for row in id_rows:
            all_concepts.update(self._parse_cpts(row["cpt_seq"]))
        self.cpt_ids = sorted(all_concepts)

        self.stu2idx = {stu_id: idx for idx, stu_id in enumerate(self.stu_ids)}
        self.exer2idx = {exer_id: idx for idx, exer_id in enumerate(self.exer_ids)}
        self.cpt2idx = {cpt_id: idx for idx, cpt_id in enumerate(self.cpt_ids)}

        self.num_students = len(self.stu_ids)
        self.num_exercises = len(self.exer_ids)
        self.num_concepts = len(self.cpt_ids)

        self.q_matrix = torch.zeros(self.num_exercises, self.num_concepts, dtype=torch.float32)
        for row in id_rows:
            exer = self.exer2idx[int(row["exer_id"])]
            for c in self._parse_cpts(row["cpt_seq"]):
                self.q_matrix[exer, self.cpt2idx[c]] = 1.0

        self.train_triplets = self._rows_to_triplets(self.train_rows)
        self.valid_triplets = self._rows_to_triplets(self.valid_rows)
        self.test_triplets = self._rows_to_triplets(self.test_rows)

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
    def _parse_cpts(cpt_seq):
        return [int(x) for x in str(cpt_seq).split(",")]

    def _rows_to_triplets(self, rows):
        triplets = []
        for r in rows:
            triplets.append(
                (
                    self.stu2idx[int(r["stu_id"])],
                    self.exer2idx[int(r["exer_id"])],
                    int(r["label"]),
                )
            )
        return triplets

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
