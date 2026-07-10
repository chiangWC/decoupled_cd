import json
import io
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset

from .utils import get_device

device = get_device()


class CognitiveDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]


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
        self._load_and_process()

    def _read_csv(self, filename):
        return pd.read_csv(os.path.join(self.data_dir, filename))

    def _read_q_matrix(self):
        try:
            source = (
                io.BytesIO(self.q_matrix_bytes)
                if self.q_matrix_bytes is not None
                else self.q_matrix_path
            )
            frame = pd.read_csv(source)
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot read Q-matrix: {self.q_matrix_path}") from exc
        missing = {"exer_id", "cpt_seq"} - set(frame.columns)
        if missing:
            raise ValueError(
                f"Q-matrix missing required columns: {', '.join(sorted(missing))}"
            )
        if frame.empty:
            raise ValueError("Q-matrix must not be empty")
        return frame

    @staticmethod
    def _token(value):
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return str(int(value))
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

    def _parse_concepts(self, cpt_seq):
        if pd.isna(cpt_seq):
            return []
        return [int(x) for x in str(cpt_seq).strip('"').split(",") if str(x).strip()]

    def _load_and_process(self):
        self.train_data = self._read_csv(self.args.train_file)
        self.valid_data = self._read_csv(self.args.valid_file)
        schema_data = pd.concat([self.train_data, self.valid_data], ignore_index=True)
        self.test_data = (
            self._read_csv(self.args.test_file)
            if self.split_mode in {"all", "test"}
            else schema_data.iloc[0:0].copy()
        )
        mapping_data = (
            pd.concat([schema_data, self.test_data], ignore_index=True)
            if self.split_mode == "all"
            else schema_data
        )
        if self.split_mode == "all":
            q_data = mapping_data
            if self.id_maps_path is not None or self.id_maps_payload is not None:
                frozen = self._load_frozen_ids()
                self.stu_ids = frozen["stu_ids"]
                self.exer_ids = frozen["exer_ids"]
                self.cpt_ids = frozen["cpt_ids"]
            else:
                self.stu_ids = [
                    self._token(value)
                    for value in sorted(mapping_data["stu_id"].unique())
                ]
                self.exer_ids = [
                    self._token(value)
                    for value in sorted(mapping_data["exer_id"].unique())
                ]
                all_concepts = set()
                for cpt_seq in mapping_data["cpt_seq"]:
                    all_concepts.update(self._parse_concepts(cpt_seq))
                self.cpt_ids = [
                    self._token(value) for value in sorted(all_concepts)
                ]
        else:
            q_data = self._read_q_matrix()
            derived_stu_ids = self._sorted_tokens(schema_data["stu_id"].unique())
            derived_exer_ids = self._sorted_tokens(q_data["exer_id"].unique())
            all_concepts = set()
            for cpt_seq in q_data["cpt_seq"]:
                all_concepts.update(self._parse_concepts(cpt_seq))
            derived_cpt_ids = self._sorted_tokens(all_concepts)
            derived = {
                "stu_ids": derived_stu_ids,
                "exer_ids": derived_exer_ids,
                "cpt_ids": derived_cpt_ids,
            }
            if self.id_maps_path is not None or self.id_maps_payload is not None:
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

        self.stu2idx = {x: i for i, x in enumerate(self.stu_ids)}
        self.exer2idx = {x: i for i, x in enumerate(self.exer_ids)}
        self.cpt2idx = {x: i for i, x in enumerate(self.cpt_ids)}

        self.num_students = len(self.stu_ids)
        self.num_exercises = len(self.exer_ids)
        self.num_concepts = len(self.cpt_ids)
        self.logger.info(
            f"Stats: Stu={self.num_students}, Exer={self.num_exercises}, Cpt={self.num_concepts}"
        )

        self.q_matrix = self._build_q_matrix(q_data).to(device)
        if self.split_mode != "all":
            for data in (self.train_data, self.valid_data, self.test_data):
                self._validate_q_coverage(data, q_data)
        self.train_triplets = self._process_triplets(self.train_data)
        self.valid_triplets = self._process_triplets(self.valid_data)
        self.test_triplets = self._process_triplets(self.test_data)
        self._validate_concepts(self.test_data)
        self._build_sample_conflict(
            mapping_data,
            use_interaction_concepts=self.split_mode != "all",
        )
        self.train_response = self._process_response_array(self.train_data)
        self.graph_dict = self._build_graph_dict()

    def _build_q_matrix(self, all_data):
        q_matrix = torch.zeros(self.num_exercises, self.num_concepts, dtype=torch.float64)
        for row in all_data.itertuples(index=False):
            exer_idx = self._lookup(self.exer2idx, row.exer_id, "exercise")
            for concept in self._parse_concepts(row.cpt_seq):
                concept_idx = self._lookup(self.cpt2idx, concept, "concept")
                q_matrix[exer_idx, concept_idx] = 1.0
        return q_matrix

    def _validate_q_coverage(self, data, q_data):
        q_concepts = defaultdict(set)
        for row in q_data.itertuples(index=False):
            exercise = self._token(row.exer_id)
            q_concepts[exercise].update(
                self._token(concept)
                for concept in self._parse_concepts(row.cpt_seq)
            )
        for row in data.itertuples(index=False):
            exercise = self._token(row.exer_id)
            self._lookup(self.exer2idx, exercise, "exercise")
            for concept in self._parse_concepts(row.cpt_seq):
                concept_token = self._token(concept)
                self._lookup(self.cpt2idx, concept_token, "concept")
                if concept_token not in q_concepts[exercise]:
                    raise ValueError(
                        f"concept ID {concept_token!r} is not in Q for "
                        f"exercise ID {exercise!r}"
                    )

    def _validate_concepts(self, data):
        for row in data.itertuples(index=False):
            for concept in self._parse_concepts(row.cpt_seq):
                self._lookup(self.cpt2idx, concept, "concept")

    def _process_triplets(self, data):
        triplets = []
        for row in data.itertuples(index=False):
            triplets.append(
                (
                    self._lookup(self.stu2idx, row.stu_id, "student"),
                    self._lookup(self.exer2idx, row.exer_id, "exercise"),
                    int(row.label),
                )
            )
        return triplets

    def _process_response_array(self, data):
        rows = []
        for row in data.itertuples(index=False):
            rows.append(
                [
                    self._lookup(self.stu2idx, row.stu_id, "student"),
                    self._lookup(self.exer2idx, row.exer_id, "exercise"),
                    int(row.label),
                ]
            )
        return np.asarray(rows, dtype=np.int64)

    def _build_sample_conflict(
        self,
        all_data,
        *,
        use_interaction_concepts=False,
    ):
        if not use_interaction_concepts:
            exercise_concepts = {}
            for row in all_data.drop_duplicates(subset=["exer_id"]).itertuples(
                index=False
            ):
                exercise_concepts[
                    self._lookup(self.exer2idx, row.exer_id, "exercise")
                ] = [
                    self._lookup(self.cpt2idx, concept, "concept")
                    for concept in self._parse_concepts(row.cpt_seq)
                ]

        student_kc_correct = defaultdict(float)
        student_kc_wrong = defaultdict(float)
        if use_interaction_concepts:
            train_evidence = (
                (
                    self._lookup(self.stu2idx, row.stu_id, "student"),
                    [
                        self._lookup(self.cpt2idx, concept, "concept")
                        for concept in self._parse_concepts(row.cpt_seq)
                    ],
                    int(row.label),
                )
                for row in self.train_data.itertuples(index=False)
            )
        else:
            train_evidence = (
                (stu_idx, exercise_concepts.get(exer_idx, []), label)
                for stu_idx, exer_idx, label in self.train_triplets
            )
        for stu_idx, concepts, label in train_evidence:
            if not concepts:
                continue
            weight = 1.0 / len(concepts)
            for concept_idx in concepts:
                key = (stu_idx, concept_idx)
                if label == 1:
                    student_kc_correct[key] += weight
                else:
                    student_kc_wrong[key] += weight

        def kc_conflict(stu_idx, concept_idx):
            key = (stu_idx, concept_idx)
            correct = student_kc_correct[key]
            wrong = student_kc_wrong[key]
            support = correct + wrong
            if support <= 0:
                return None, 0.0
            return 2.0 * min(correct, wrong) / support, support

        if use_interaction_concepts:
            test_evidence = (
                (
                    self._lookup(self.stu2idx, row.stu_id, "student"),
                    [
                        self._lookup(self.cpt2idx, concept, "concept")
                        for concept in self._parse_concepts(row.cpt_seq)
                    ],
                )
                for row in self.test_data.itertuples(index=False)
            )
        else:
            test_evidence = (
                (stu_idx, exercise_concepts.get(exer_idx, []))
                for stu_idx, exer_idx, _ in self.test_triplets
            )
        self.test_sample_conflict = []
        for stu_idx, concepts in test_evidence:
            concept_scores = []
            concept_supports = []
            for concept_idx in concepts:
                score, support = kc_conflict(stu_idx, concept_idx)
                if score is None:
                    continue
                concept_scores.append(score)
                concept_supports.append(support)

            if concept_scores:
                self.test_sample_conflict.append(
                    {
                        "group": "Eligible",
                        "score": float(sum(concept_scores) / len(concept_scores)),
                        "support": float(sum(concept_supports) / len(concept_supports)),
                        "concept_count": len(concept_scores),
                    }
                )
            else:
                self.test_sample_conflict.append(
                    {
                        "group": "Unknown",
                        "score": 1.0,
                        "support": 0.0,
                        "concept_count": 0,
                    }
                )

        known = sum(1 for meta in self.test_sample_conflict if meta["group"] != "Unknown")
        self.logger.info(
            f"SampleConflict metadata: eligible_test_samples={known} | "
            f"unknown_test_samples={len(self.test_sample_conflict) - known}"
        )

    @staticmethod
    def _get_csr(rows, cols, shape):
        values = np.ones_like(rows, dtype=np.float64)
        return sp.csr_matrix((values, (rows, cols)), shape=shape)

    @staticmethod
    def _sp_mat_to_sp_tensor(sp_mat):
        coo = sp_mat.tocoo().astype(np.float64)
        indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
        return torch.sparse_coo_tensor(indices, coo.data, coo.shape, dtype=torch.float64).coalesce()

    def _create_adj_se(self, np_response, is_subgraph=False):
        if is_subgraph:
            if self.args.mode == "R":
                empty = np.zeros(shape=(self.num_students, self.num_exercises))
                return empty, empty

            train_stu_right = np_response[np_response[:, 2] == 1, 0]
            train_exer_right = np_response[np_response[:, 2] == 1, 1]
            train_stu_wrong = np_response[np_response[:, 2] == 0, 0]
            train_exer_wrong = np_response[np_response[:, 2] == 0, 1]

            adj_se_right = self._get_csr(
                train_stu_right,
                train_exer_right,
                shape=(self.num_students, self.num_exercises),
            )
            adj_se_wrong = self._get_csr(
                train_stu_wrong,
                train_exer_wrong,
                shape=(self.num_students, self.num_exercises),
            )
            return adj_se_right.toarray(), adj_se_wrong.toarray()

        if self.args.mode == "R":
            return np.zeros(shape=(self.num_students, self.num_exercises))

        response_stu = np_response[:, 0]
        response_exer = np_response[:, 1]
        adj_se = self._get_csr(
            response_stu, response_exer, shape=(self.num_students, self.num_exercises)
        )
        return adj_se.toarray()

    def _final_graph(self, se, ek):
        sek_num = self.num_students + self.num_exercises + self.num_concepts
        se_num = self.num_students + self.num_exercises
        tmp = np.zeros(shape=(sek_num, sek_num), dtype=np.float64)
        tmp[: self.num_students, self.num_students : se_num] = se
        tmp[self.num_students : se_num, se_num:sek_num] = ek
        graph = tmp + tmp.T + np.identity(sek_num, dtype=np.float64)
        graph = sp.csr_matrix(graph)

        rowsum = np.array(graph.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(graph)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return self._sp_mat_to_sp_tensor(adj_matrix).to(device)

    def _build_graph_dict(self):
        if self.args.mode == "Q":
            ek_graph = np.zeros(shape=self.q_matrix.shape, dtype=np.float64)
        else:
            ek_graph = self.q_matrix.cpu().numpy().astype(np.float64)

        se_graph_right, se_graph_wrong = self._create_adj_se(self.train_response, is_subgraph=True)
        se_graph_all = self._create_adj_se(self.train_response, is_subgraph=False)

        return {
            "right": self._final_graph(se_graph_right, ek_graph),
            "wrong": self._final_graph(se_graph_wrong, ek_graph),
            "all": self._final_graph(se_graph_all, ek_graph),
            "response": self.train_response.copy(),
            "Q_Matrix": ek_graph.copy(),
            "flip_ratio": self.args.flip_ratio,
        }

    def get_loaders(self):
        q_matrix = self.q_matrix

        def collate(batch):
            stu = torch.tensor([x[0] for x in batch], dtype=torch.long, device=device)
            exer = torch.tensor([x[1] for x in batch], dtype=torch.long, device=device)
            label = torch.tensor([x[2] for x in batch], dtype=torch.float64, device=device)
            kn_emb = q_matrix[exer]
            return stu, exer, kn_emb, label

        kwargs = {"batch_size": self.args.batch_size, "collate_fn": collate}
        return (
            DataLoader(CognitiveDataset(self.train_triplets), shuffle=True, **kwargs),
            DataLoader(CognitiveDataset(self.valid_triplets), shuffle=False, **kwargs),
            DataLoader(CognitiveDataset(self.test_triplets), shuffle=False, **kwargs),
        )

    def evaluation_rows(self, split):
        if split not in {"valid", "test"}:
            raise ValueError("evaluation rows split must be valid or test")
        frame = self.valid_data if split == "valid" else self.test_data
        fields = ["stu_id", "exer_id", "cpt_seq", "label"]
        return frame.loc[:, fields].to_dict(orient="records")
