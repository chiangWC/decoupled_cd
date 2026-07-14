from __future__ import annotations

import torch

from models import TwoStageTKCUKCCDM
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
    exercise_evidence[:, 2] = (
        exercise_evidence[:, 0] - exercise_evidence[:, 1]
    )
    exercise_evidence[:, 3] = (
        exercise_evidence[:, 1] / exercise_evidence[:, 0].clamp_min(1.0)
    )
    exercise_evidence[:, 4] = torch.log1p(exercise_evidence[:, 0])
    exercise_evidence[:, 5] = (
        exercise_evidence[:, 0] > 0.0
    ).float()
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


def _model(
    evidence_mode: str,
    completion_mode: str,
    diagnosis_mode: str = "target_conditioned",
    concept_prior_mode: str = "population_q",
) -> TwoStageTKCUKCCDM:
    torch.manual_seed(42)
    return TwoStageTKCUKCCDM(
        num_students=4,
        num_exercises=5,
        num_concepts=3,
        concept_dim=16,
        evidence_mode=evidence_mode,
        concept_prior_mode=concept_prior_mode,
        completion_mode=completion_mode,
        diagnosis_mode=diagnosis_mode,
    )


def _count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def test_all_variants_share_initialization_topology_and_contract() -> None:
    variants = [
        _model("calibrated_history", "personalized_interaction"),
        _model(
            "calibrated_history",
            "personalized_interaction",
            concept_prior_mode="semantic_q_control",
        ),
        _model(
            "calibrated_history",
            "personalized_interaction",
            concept_prior_mode="global_population_control",
        ),
        _model("identity_raw_control", "personalized_interaction"),
        _model("calibrated_summary_control", "personalized_interaction"),
        _model("calibrated_history", "additive_personalized_control"),
        _model("raw_summary_control", "personalized_interaction"),
        _model("calibrated_history", "direct_prior_control"),
        _model("raw_summary_control", "direct_prior_control"),
        _model(
            "calibrated_history",
            "personalized_interaction",
            "monotonic_control",
        ),
    ]
    assert len({_count(model) for model in variants}) == 1
    assert len({model.initialization_hash() for model in variants}) == 1
    assert len({model.architecture_fingerprint for model in variants}) == 1
    assert not any(
        "student_embedding" in name
        for name, _ in variants[0].named_parameters()
    )
    for model in variants:
        output = model(**_inputs())
        assert output.probs.shape == (3,)
        assert output.framework_state.shape == (2, 3, 16)
        assert output.mastery.shape == (2, 3)
        assert torch.isfinite(output.probs).all()


def test_module_gradients_are_isolated() -> None:
    model = _model("calibrated_history", "personalized_interaction")
    model(**_inputs()).probs.mean().backward()
    assert model.evidence_representation.calibrated_encoder[0].weight.grad is not None
    assert model.evidence_representation.raw_control_encoder[0].weight.grad is None
    assert model.evidence_representation.identity_raw_encoder[0].weight.grad is None
    assert model.evidence_representation.calibrated_summary_encoder[0].weight.grad is None
    assert model.concept_prior.population_q_encoder[0].weight.grad is not None
    assert model.concept_prior.semantic_q_encoder[0].weight.grad is None
    assert model.concept_prior.global_population_encoder[0].weight.grad is None
    assert model.state_completion.personalized_decoder[0].weight.grad is not None
    assert model.state_completion.additive_control_decoder[0].weight.grad is None
    assert model.state_completion.direct_control_decoder[0].weight.grad is None
    assert model.cognitive_match[0].weight.grad is not None
    assert model.control_cognitive_match[0].weight.grad is None

    model = _model(
        "calibrated_history",
        "personalized_interaction",
        concept_prior_mode="semantic_q_control",
    )
    model(**_inputs()).probs.mean().backward()
    assert model.concept_prior.population_q_encoder[0].weight.grad is None
    assert model.concept_prior.semantic_q_encoder[0].weight.grad is not None
    assert model.concept_prior.global_population_encoder[0].weight.grad is None
    model = _model("raw_summary_control", "direct_prior_control")
    model(**_inputs()).probs.mean().backward()
    assert model.evidence_representation.calibrated_encoder[0].weight.grad is None
    # Direct-prior completion deliberately cuts the student-evidence output.
    assert model.evidence_representation.raw_control_encoder[0].weight.grad is None
    assert model.state_completion.personalized_decoder[0].weight.grad is None
    assert model.state_completion.direct_control_decoder[0].weight.grad is not None

    model = _model("calibrated_history", "additive_personalized_control")
    model(**_inputs()).probs.mean().backward()
    assert model.evidence_representation.calibrated_encoder[0].weight.grad is not None
    assert model.state_completion.personalized_decoder[0].weight.grad is None
    assert model.state_completion.additive_control_decoder[0].weight.grad is not None
    assert model.state_completion.direct_control_decoder[0].weight.grad is None

    model = _model(
        "calibrated_history",
        "personalized_interaction",
        "monotonic_control",
    )
    model(**_inputs()).probs.mean().backward()
    assert model.cognitive_match[0].weight.grad is None
    assert model.control_cognitive_match[0].weight.grad is not None


def test_active_controls_are_capacity_matched() -> None:
    model = _model("calibrated_history", "personalized_interaction")
    for name, counts in model.active_module_parameter_counts().items():
        values = list(counts.values())
        relative_gap = (max(values) - min(values)) / max(values)
        assert relative_gap <= 0.10, name


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
