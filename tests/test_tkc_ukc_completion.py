from __future__ import annotations

import torch

from models import TKCUKCCompletionCDM
from trainers.engine import _build_context_target_batch


def _inputs() -> dict[str, torch.Tensor]:
    q_matrix = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    mask = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    responses = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    evidence = torch.zeros(4, 3, 6)
    for student_id in range(4):
        attempts = mask[student_id] @ q_matrix
        correct = (mask[student_id] * responses[student_id]) @ q_matrix
        evidence[student_id, :, 0] = attempts
        evidence[student_id, :, 1] = correct
        evidence[student_id, :, 2] = attempts - correct
        evidence[student_id, :, 3] = correct / attempts.clamp_min(1.0)
        evidence[student_id, :, 4] = torch.log1p(attempts)
        evidence[student_id, :, 5] = (attempts > 0.0).float()
    exercise_evidence = torch.zeros(5, 6)
    exercise_evidence[:, 0] = mask.sum(dim=0)
    exercise_evidence[:, 1] = (mask * responses).sum(dim=0)
    exercise_evidence[:, 2] = exercise_evidence[:, 0] - exercise_evidence[:, 1]
    exercise_evidence[:, 3] = (
        exercise_evidence[:, 1] / exercise_evidence[:, 0].clamp_min(1.0)
    )
    exercise_evidence[:, 4] = torch.log1p(exercise_evidence[:, 0])
    exercise_evidence[:, 5] = (exercise_evidence[:, 0] > 0.0).float()
    return {
        "q_matrix": q_matrix,
        "concept_graph": torch.zeros(3, 3),
        "student_exercise_mask": mask,
        "response_matrix": responses,
        "student_tkc_mask": evidence[..., 5],
        "student_ukc_mask": 1.0 - evidence[..., 5],
        "student_concept_evidence": evidence,
        "exercise_evidence": exercise_evidence,
        "target_student_ids": torch.tensor([2, 2, 0]),
        "target_exercise_ids": torch.tensor([1, 2, 4]),
        "use_student_subset": True,
    }


def _model(evidence_mode: str, completion_mode: str) -> TKCUKCCompletionCDM:
    torch.manual_seed(42)
    return TKCUKCCompletionCDM(
        num_students=4,
        num_exercises=5,
        num_concepts=3,
        concept_dim=16,
        evidence_mode=evidence_mode,
        completion_mode=completion_mode,
        attention_heads=4,
        query_chunk_size=2,
    )


def _count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def test_all_variants_share_initialization_and_contract() -> None:
    variants = [
        _model("relational", "personalized_attention"),
        _model("raw_statistics", "personalized_attention"),
        _model("relational", "global_control"),
        _model("raw_statistics", "global_control"),
    ]
    assert len({_count(model) for model in variants}) == 1
    assert len({model.initialization_hash() for model in variants}) == 1
    assert len({model.architecture_fingerprint for model in variants}) == 1
    assert not any("student_embedding" in name for name, _ in variants[0].named_parameters())
    for model in variants:
        output = model(**_inputs())
        assert output.probs.shape == (3,)
        assert output.framework_state.shape == (2, 3, 16)
        assert output.mastery.shape == (2, 3)
        assert torch.isfinite(output.probs).all()


def test_module_gradients_are_isolated() -> None:
    model = _model("relational", "personalized_attention")
    model(**_inputs()).probs.mean().backward()
    assert model.tkc_evidence.correct_relation[0].weight.grad is not None
    assert model.tkc_evidence.raw_statistics_control[0].weight.grad is None
    assert model.ukc_completion.query_projection.weight.grad is not None
    assert model.ukc_completion.global_control[0].weight.grad is None

    model = _model("raw_statistics", "global_control")
    model(**_inputs()).probs.mean().backward()
    assert model.tkc_evidence.correct_relation[0].weight.grad is None
    assert model.tkc_evidence.raw_statistics_control[0].weight.grad is not None
    assert model.ukc_completion.query_projection.weight.grad is None
    assert model.ukc_completion.global_control[0].weight.grad is not None


def test_active_controls_are_capacity_matched() -> None:
    model = _model("relational", "personalized_attention")
    for counts in [
        model.tkc_evidence.active_parameter_counts(),
        model.ukc_completion.active_parameter_counts(),
    ]:
        values = list(counts.values())
        assert abs(values[0] - values[1]) / values[0] <= 0.10


def test_context_target_mask_has_full_evidence_schema() -> None:
    inputs = _inputs()
    tensors: dict[str, torch.Tensor | None] = {
        "q_matrix": inputs["q_matrix"],
        "student_exercise_mask": inputs["student_exercise_mask"],
        "response_matrix": inputs["response_matrix"],
        "student_tkc_mask": inputs["student_tkc_mask"],
        "student_ukc_mask": inputs["student_ukc_mask"],
        "student_concept_evidence": inputs["student_concept_evidence"],
        "interaction_student_ids": inputs["target_student_ids"],
        "interaction_exercise_ids": inputs["target_exercise_ids"],
        "interaction_labels": torch.tensor([0.0, 1.0, 0.0]),
    }
    torch.manual_seed(42)
    forward, supervised = _build_context_target_batch(
        tensors=tensors,
        student_ids=torch.tensor([0, 2]),
        batch_indices=torch.arange(3),
        context_target_frac=0.5,
    )
    assert forward["student_concept_evidence"].shape[-1] == 6
    for row in supervised.tolist():
        student = int(tensors["interaction_student_ids"][row])
        exercise = int(tensors["interaction_exercise_ids"][row])
        assert forward["student_exercise_mask"][student, exercise] == 0.0
