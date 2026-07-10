"""
Decoupling plugin for ORCDF: ports our two core contributions onto ORCDF's
response-graph backbone, to test whether they help on a strong encoder.

Contribution A — supervised TKC/UKC decoupling of mastery: ORCDF's K-dim
student_ts (the mastery vector) is produced by unconstrained diffusion over the
response graph, so an untested concept's mastery is contaminated by behaviour on
adjacent concepts. We route the untested (UKC) entries through an explicit,
zero-init concept-graph propagation gate instead — accountable structure signal.

Contribution B — mastery auxiliary BCE: predict each response from the target
concepts' monotone mastery alone, giving the mastery vector direct per-concept
supervision (our biggest DOA lever).

Both are flag-gated and zero-initialised, so with the flags off the model is
bit-identical to baseline ORCDF.
"""
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ORCDFNet


def build_tkc_mask(data_proc, dtype):
    """(S, K) mask: 1 where the student answered an exercise touching concept k in TRAIN."""
    S, K = data_proc.num_students, data_proc.num_concepts
    exercise_concepts = {}
    for row in data_proc.train_data.drop_duplicates(subset=["exer_id"]).itertuples(index=False):
        exercise_concepts[data_proc.exer2idx[row.exer_id]] = [
            data_proc.cpt2idx[c] for c in data_proc._parse_concepts(row.cpt_seq)
        ]
    mask = torch.zeros(S, K, dtype=dtype)
    for stu_idx, exer_idx, _label in data_proc.train_triplets:
        for c in exercise_concepts.get(exer_idx, []):
            mask[stu_idx, c] = 1.0
    return mask


def build_concept_graph(data_proc, dtype):
    """Row-normalized (K, K) concept co-occurrence graph from TRAIN exercises only."""
    K = data_proc.num_concepts
    adj = torch.zeros(K, K, dtype=dtype)
    for row in data_proc.train_data.drop_duplicates(subset=["exer_id"]).itertuples(index=False):
        concepts = [data_proc.cpt2idx[c] for c in data_proc._parse_concepts(row.cpt_seq)]
        for i in concepts:
            for j in concepts:
                if i != j:
                    adj[i, j] += 1.0
    adj = adj / adj.sum(dim=1, keepdim=True).clamp_min(1.0)
    return adj


class DecoupledORCDF(ORCDFNet):
    def __init__(self, *args, decouple=False, aux_weight=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.decouple = decouple
        self.aux_weight = aux_weight
        self.tkc_mask = None
        self.concept_graph = None
        dim = self.extractor.knowledge_num
        dtype = self.extractor.dtype
        device = self.extractor.device
        if decouple:
            # Zero-init gate on the UKC structure-routing delta: starts as a no-op.
            self.ukc_gate = nn.Parameter(torch.zeros(1, dtype=dtype, device=device))
        if aux_weight > 0.0:
            self.aux_scale = nn.Parameter(torch.tensor(2.0, dtype=dtype, device=device))

    def set_decouple_tensors(self, tkc_mask, concept_graph):
        self.tkc_mask = tkc_mask.to(self.extractor.device)
        self.concept_graph = concept_graph.to(self.extractor.device)

    def forward(self, stu_id, exer_id, kn_emb, label=None):
        student_ts, diff_ts, disc_ts, knowledge_ts, extras = self.extractor.extract(stu_id, exer_id, kn_emb)

        if self.decouple:
            tkc = self.tkc_mask[stu_id]                      # (B, K)
            ukc_prop = student_ts @ self.concept_graph.t()   # structure propagation
            delta = (1.0 - tkc) * self.ukc_gate * (ukc_prop - student_ts)
            student_ts = student_ts + delta

        pred = self.inter_func.compute(
            student_ts=student_ts,
            diff_ts=diff_ts,
            disc_ts=disc_ts,
            q_mask=kn_emb,
            knowledge_ts=knowledge_ts,
            other=extras,
        )
        extra_loss = extras["extra_loss"]

        if self.aux_weight > 0.0 and label is not None:
            mastery = torch.sigmoid(student_ts)
            diff = torch.sigmoid(diff_ts)
            per = kn_emb * (mastery - diff)
            aux_logit = F.softplus(self.aux_scale) * per.sum(dim=1) / kn_emb.sum(dim=1).clamp_min(1.0)
            aux_bce = F.binary_cross_entropy_with_logits(aux_logit, label)
            extra_loss = extra_loss + self.aux_weight * aux_bce

        return pred, extra_loss

    def mastery_matrix(self):
        """(S, K) decoupled mastery for DOA."""
        stu_ids = torch.arange(self.extractor.student_num, device=self.extractor.device)
        exer_ids = torch.zeros_like(stu_ids)
        q = torch.ones(len(stu_ids), self.extractor.knowledge_num, dtype=self.extractor.dtype, device=self.extractor.device)
        with torch.no_grad():
            student_ts, _, _, _, _ = self.extractor.extract(stu_ids, exer_ids, q)
            if self.decouple:
                tkc = self.tkc_mask
                ukc_prop = student_ts @ self.concept_graph.t()
                student_ts = student_ts + (1.0 - tkc) * self.ukc_gate * (ukc_prop - student_ts)
            return torch.sigmoid(student_ts).float().cpu().numpy()
