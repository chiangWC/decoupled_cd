from __future__ import annotations

import torch

from models import NeuralProcessCDM
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
            [1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    responses = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    concept_evidence = torch.zeros(4, 3, 6)
    for student_id in range(4):
        attempts = mask[student_id] @ q_matrix
        correct = (mask[student_id] * responses[student_id]) @ q_matrix
        concept_evidence[student_id, :, 0] = attempts
        concept_evidence[student_id, :, 1] = correct
        concept_evidence[student_id, :, 2] = attempts - correct
        concept_evidence[student_id, :, 3] = correct / attempts.clamp_min(1.0)
        concept_evidence[student_id, :, 4] = torch.log1p(attempts)
        concept_evidence[student_id, :, 5] = (attempts > 0.0).float()

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
        "student_tkc_mask": concept_evidence[..., 5],
        "student_ukc_mask": 1.0 - concept_evidence[..., 5],
        "student_concept_evidence": concept_evidence,
        "exercise_evidence": exercise_evidence,
        "target_student_ids": torch.tensor([2, 2, 0]),
        "target_exercise_ids": torch.tensor([1, 2, 4]),
        "use_student_subset": True,
    }


def _model(evidence_mode: str, query_mode: str) -> NeuralProcessCDM:
    torch.manual_seed(42)
    return NeuralProcessCDM(
        num_students=4,
        num_exercises=5,
        num_concepts=3,
        concept_dim=16,
        evidence_mode=evidence_mode,
        query_mode=query_mode,
        memory_slots=4,
        attention_heads=4,
    )


def _parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def test_variants_share_topology_initialization_and_output_contract() -> None:
    variants = [
        _model("induced_posterior", "cross_attention"),
        _model("summary_control", "cross_attention"),
        _model("induced_posterior", "latent_control"),
        _model("summary_control", "latent_control"),
    ]
    assert len({_parameter_count(model) for model in variants}) == 1
    assert len({model.initialization_hash() for model in variants}) == 1
    assert len({model.architecture_fingerprint for model in variants}) == 1
    assert not any("student_embedding" in name for name, _ in variants[0].named_parameters())

    for model in variants:
        output = model(**_inputs())
        assert output.probs.shape == (3,)
        assert output.framework_state.shape == (2, 3, 16)
        assert output.mastery.shape == (2, 3)
        assert output.state_reliability.shape == (2, 3)
        assert torch.isfinite(output.probs).all()


def test_each_mode_activates_only_its_declared_module_path() -> None:
    model = _model("induced_posterior", "cross_attention")
    model(**_inputs()).probs.mean().backward()
    assert model.evidence_posterior.induced_attention.in_proj_weight.grad is not None
    assert model.evidence_posterior.summary_control[0].weight.grad is None
    assert model.concept_completion.query_attention.in_proj_weight.grad is not None
    assert model.concept_completion.latent_control[0].weight.grad is None

    model = _model("summary_control", "latent_control")
    model(**_inputs()).probs.mean().backward()
    assert model.evidence_posterior.induced_attention.in_proj_weight.grad is None
    assert model.evidence_posterior.summary_control[0].weight.grad is not None
    assert model.concept_completion.query_attention.in_proj_weight.grad is None
    assert model.concept_completion.latent_control[0].weight.grad is not None


def test_active_capacity_controls_are_within_ten_percent() -> None:
    model = _model("induced_posterior", "cross_attention")
    evidence = model.evidence_posterior
    full_evidence = (
        evidence.inducing_slots.numel()
        + _parameter_count(evidence.induced_attention)
        + _parameter_count(evidence.induced_norm)
        + _parameter_count(evidence.induced_ff)
        + _parameter_count(evidence.induced_ff_norm)
    )
    control_evidence = (
        evidence.control_slot_offsets.numel()
        + _parameter_count(evidence.summary_control)
        + _parameter_count(evidence.control_norm)
    )
    completion = model.concept_completion
    full_completion = (
        _parameter_count(completion.query_attention)
        + _parameter_count(completion.query_norm)
        + _parameter_count(completion.query_ff)
        + _parameter_count(completion.query_ff_norm)
    )
    control_completion = (
        _parameter_count(completion.latent_control)
        + _parameter_count(completion.control_norm)
    )
    assert abs(full_evidence - control_evidence) / full_evidence <= 0.10
    assert abs(full_completion - control_completion) / full_completion <= 0.10


def test_framework_state_is_consumed_by_the_only_prediction_head() -> None:
    model = _model("induced_posterior", "cross_attention")
    model.eval()
    inputs = _inputs()
    output = model(**inputs)
    row_ids = torch.tensor([1, 1, 0])
    baseline, _, _ = model.diagnose(
        framework_state_for_rows=output.framework_state[row_ids],
        mastery_for_rows=output.mastery[row_ids],
        q_matrix=inputs["q_matrix"],
        target_exercise_ids=inputs["target_exercise_ids"],
        exercise_embeddings=output.exercise_embeddings,
    )
    perturbed_state = output.framework_state[row_ids].clone()
    perturbed_state[:, :, 0] += 1.0
    changed, _, _ = model.diagnose(
        framework_state_for_rows=perturbed_state,
        mastery_for_rows=output.mastery[row_ids],
        q_matrix=inputs["q_matrix"],
        target_exercise_ids=inputs["target_exercise_ids"],
        exercise_embeddings=output.exercise_embeddings,
    )
    assert torch.allclose(baseline, output.probs)
    assert not torch.allclose(changed, baseline)


def test_context_target_training_removes_supervised_responses_from_history() -> None:
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
    forward_tensors, supervised_indices = _build_context_target_batch(
        tensors=tensors,
        student_ids=torch.tensor([0, 2]),
        batch_indices=torch.arange(3),
        context_target_frac=0.5,
    )
    assert supervised_indices.numel() >= 1
    assert forward_tensors["student_concept_evidence"].shape[-1] == 6
    for row_index in supervised_indices.tolist():
        student_id = int(tensors["interaction_student_ids"][row_index])
        exercise_id = int(tensors["interaction_exercise_ids"][row_index])
        assert forward_tensors["student_exercise_mask"][student_id, exercise_id] == 0.0
    assert forward_tensors["student_exercise_mask"][0].sum() >= 1.0
    assert forward_tensors["student_exercise_mask"][2].sum() >= 1.0
