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
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.mastery_auxiliary import MasteryAuxiliaryObjective

from .model import ORCDFNet


def build_tkc_mask(data_proc, dtype):
    """(S, K) mask: 1 where the student answered an exercise touching concept k in TRAIN."""
    S, K = data_proc.num_students, data_proc.num_concepts
    q_matrix = data_proc.q_matrix.detach().cpu()
    exercise_concepts = {
        exercise: torch.nonzero(q_matrix[exercise], as_tuple=False)
        .flatten()
        .tolist()
        for exercise in {exer for _, exer, _ in data_proc.train_triplets}
    }
    mask = torch.zeros(S, K, dtype=dtype)
    for student, exercise, _ in data_proc.train_triplets:
        for concept in exercise_concepts[exercise]:
            mask[student, concept] = 1.0
    return mask


def build_concept_graph(data_proc, dtype):
    """Row-normalized (K, K) concept co-occurrence graph from TRAIN exercises only."""
    K = data_proc.num_concepts
    adj = torch.zeros(K, K, dtype=dtype)
    q_matrix = data_proc.q_matrix.detach().cpu()
    train_exercises = {exercise for _, exercise, _ in data_proc.train_triplets}
    for exercise in train_exercises:
        concepts = torch.nonzero(q_matrix[exercise], as_tuple=False).flatten().tolist()
        for i in concepts:
            for j in concepts:
                if i != j:
                    adj[i, j] += 1.0
    adj = adj / adj.sum(dim=1, keepdim=True).clamp_min(1.0)
    return adj


def add_plugin_parameters_to_optimizer(optimizer, model, *, lr, weight_decay):
    """Append direct plugin parameters without modifying ORCDF's two base groups."""
    existing = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    plugin_parameters = [
        parameter
        for parameter in model.parameters(recurse=False)
        if parameter.requires_grad and id(parameter) not in existing
    ]
    if plugin_parameters:
        optimizer.add_param_group(
            {
                "params": plugin_parameters,
                "lr": lr,
                "weight_decay": weight_decay,
            }
        )
    return plugin_parameters


class DecoupledORCDF(ORCDFNet):
    def __init__(
        self,
        *args,
        decouple=False,
        aux_weight=0.0,
        aux_detach_item_difficulty=False,
        aux_warmup_fraction=0.0,
        **kwargs,
    ):
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
        self.mastery_auxiliary = (
            MasteryAuxiliaryObjective(
                aux_weight=aux_weight,
                detach_item_difficulty=aux_detach_item_difficulty,
                warmup_fraction=aux_warmup_fraction,
            )
            if aux_weight > 0.0
            else None
        )

    def set_decouple_tensors(self, tkc_mask, concept_graph):
        self.tkc_mask = tkc_mask.to(self.extractor.device)
        self.concept_graph = concept_graph.to(self.extractor.device)

    def forward(
        self, stu_id, exer_id, kn_emb, label=None, *, epoch=1, total_epochs=1
    ):
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

        if self.mastery_auxiliary is not None and label is not None:
            extra_loss = extra_loss + self.mastery_auxiliary(
                student_ts,
                diff_ts,
                kn_emb,
                label,
                epoch=epoch,
                total_epochs=total_epochs,
            )

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
