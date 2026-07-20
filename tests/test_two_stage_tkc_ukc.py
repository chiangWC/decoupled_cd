from __future__ import annotations

import hashlib
from types import SimpleNamespace

import torch

from models import TwoStageTKCUKCCDM
from models.two_stage_tkc_ukc import (
    ExerciseSpecificRequirementQuery,
    QSemanticNodeAlignment,
)
from scripts.analyze_prediction_slices import load_model
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
    semantic_node_mode: str = "bidirectional_q",
    evidence_refinement_mode: str = "identity_passthrough",
    target_requirement_mode: str = "exercise_specific",
) -> TwoStageTKCUKCCDM:
    torch.manual_seed(42)
    return TwoStageTKCUKCCDM(
        num_students=4,
        num_exercises=5,
        num_concepts=3,
        concept_dim=16,
        semantic_node_mode=semantic_node_mode,
        evidence_mode=evidence_mode,
        evidence_refinement_mode=evidence_refinement_mode,
        target_requirement_mode=target_requirement_mode,
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
            semantic_node_mode="raw_identity_control",
        ),
        _model(
            "calibrated_history",
            "personalized_interaction",
            semantic_node_mode="global_context_control",
        ),
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
        _model("calibrated_history", "query_attentive_field"),
        _model("calibrated_history", "global_attentive_control"),
        _model("raw_summary_control", "personalized_interaction"),
        _model("calibrated_history", "direct_prior_control"),
        _model("raw_summary_control", "direct_prior_control"),
        _model(
            "calibrated_history",
            "personalized_interaction",
            "monotonic_control",
        ),
        _model(
            "calibrated_history",
            "personalized_interaction",
            "item_hypernetwork",
        ),
        _model(
            "calibrated_history",
            "personalized_interaction",
            evidence_refinement_mode="outcome_multiset",
        ),
        _model(
            "calibrated_history",
            "personalized_interaction",
            evidence_refinement_mode="unconditioned_set_control",
        ),
        _model(
            "calibrated_history",
            "personalized_interaction",
            evidence_refinement_mode="base_capacity_control",
        ),
        _model(
            "calibrated_history",
            "personalized_interaction",
            target_requirement_mode="concept_prototype_control",
        ),
        _model(
            "calibrated_history",
            "personalized_interaction",
            target_requirement_mode="factorized_item_control",
        ),
        _model(
            "calibrated_history",
            "personalized_interaction",
            target_requirement_mode="q_only_control",
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
    assert (
        model.item_conditioned_hyper_diagnosis.item_to_weights.weight.grad
        is None
    )
    assert (
        model.outcome_evidence_refinement.outcome_encoder[0].weight.grad
        is None
    )
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
    assert (
        model.item_conditioned_hyper_diagnosis.item_to_weights.weight.grad
        is None
    )

    model = _model(
        "calibrated_history",
        "personalized_interaction",
        "item_hypernetwork",
    )
    model(**_inputs()).probs.mean().backward()
    assert model.cognitive_match[0].weight.grad is None
    assert model.control_cognitive_match[0].weight.grad is None
    assert (
        model.item_conditioned_hyper_diagnosis.item_to_weights.weight.grad
        is not None
    )
    assert (
        model.item_conditioned_hyper_diagnosis.cognitive_head[0].weight.grad
        is not None
    )
    assert (
        model.item_conditioned_hyper_diagnosis.guess_head[0].weight.grad
        is not None
    )
    assert (
        model.item_conditioned_hyper_diagnosis.slip_head[0].weight.grad
        is not None
    )

    model = _model(
        "calibrated_history",
        "personalized_interaction",
        evidence_refinement_mode="outcome_multiset",
    )
    model(**_inputs()).probs.mean().backward()
    assert (
        model.outcome_evidence_refinement.outcome_encoder[0].weight.grad
        is not None
    )
    assert (
        model.outcome_evidence_refinement
        .unconditioned_control_encoder[0].weight.grad
        is None
    )
    assert (
        model.outcome_evidence_refinement.base_control_encoder[0].weight.grad
        is None
    )

    model = _model(
        "calibrated_history",
        "personalized_interaction",
        evidence_refinement_mode="unconditioned_set_control",
    )
    model(**_inputs()).probs.mean().backward()
    assert (
        model.outcome_evidence_refinement.outcome_encoder[0].weight.grad
        is None
    )
    assert (
        model.outcome_evidence_refinement
        .unconditioned_control_encoder[0].weight.grad
        is not None
    )
    assert (
        model.outcome_evidence_refinement.base_control_encoder[0].weight.grad
        is None
    )

    model = _model(
        "calibrated_history",
        "personalized_interaction",
        evidence_refinement_mode="base_capacity_control",
    )
    model(**_inputs()).probs.mean().backward()
    assert (
        model.outcome_evidence_refinement.outcome_encoder[0].weight.grad
        is None
    )
    assert (
        model.outcome_evidence_refinement
        .unconditioned_control_encoder[0].weight.grad
        is None
    )
    assert (
        model.outcome_evidence_refinement.base_control_encoder[0].weight.grad
        is not None
    )

    model = _model("calibrated_history", "query_attentive_field")
    model(**_inputs()).probs.mean().backward()
    assert (
        model.observed_anchor_state_field.query_branch
        .query_projection.weight.grad
        is not None
    )
    assert (
        model.observed_anchor_state_field.global_control_branch
        .query_projection.weight.grad
        is None
    )
    assert model.state_completion.personalized_decoder[0].weight.grad is None

    model = _model("calibrated_history", "global_attentive_control")
    model(**_inputs()).probs.mean().backward()
    assert (
        model.observed_anchor_state_field.query_branch
        .query_projection.weight.grad
        is None
    )
    assert (
        model.observed_anchor_state_field.global_control_branch
        .query_projection.weight.grad
        is not None
    )


def test_semantic_alignment_controls_remove_q_specific_routing() -> None:
    module = QSemanticNodeAlignment()
    inputs = _inputs()
    concept_embeddings = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    exercise_embeddings = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    q_matrix = inputs["q_matrix"]
    permuted_q = q_matrix.roll(shifts=1, dims=1)
    full = module(
        concept_embeddings=concept_embeddings,
        exercise_embeddings=exercise_embeddings,
        q_matrix=q_matrix,
        mode="bidirectional_q",
    )
    full_permuted = module(
        concept_embeddings=concept_embeddings,
        exercise_embeddings=exercise_embeddings,
        q_matrix=permuted_q,
        mode="bidirectional_q",
    )
    assert not torch.equal(full.concept_nodes, full_permuted.concept_nodes)
    for mode in ["raw_identity_control", "global_context_control"]:
        output = module(
            concept_embeddings=concept_embeddings,
            exercise_embeddings=exercise_embeddings,
            q_matrix=q_matrix,
            mode=mode,
        )
        output_permuted = module(
            concept_embeddings=concept_embeddings,
            exercise_embeddings=exercise_embeddings,
            q_matrix=permuted_q,
            mode=mode,
        )
        assert torch.equal(output.concept_nodes, output_permuted.concept_nodes)
        assert torch.equal(output.exercise_nodes, output_permuted.exercise_nodes)


def test_legacy_target_requirement_controls_remove_exercise_identity() -> None:
    module = ExerciseSpecificRequirementQuery(dim=3)
    q_matrix = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    concept_nodes = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    exercise_nodes = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    common = {
        "q_matrix": q_matrix,
        "concept_nodes": concept_nodes,
        "exercise_nodes": exercise_nodes,
        "target_exercise_ids": torch.tensor([0, 1]),
        "projection": torch.nn.Identity(),
    }
    full = module(**common, mode="exercise_specific")
    prototype = module(**common, mode="concept_prototype_control")
    q_only = module(**common, mode="q_only_control")
    assert not torch.equal(full.q_repr[0], full.q_repr[1])
    assert torch.equal(prototype.q_repr[0], prototype.q_repr[1])
    assert torch.equal(q_only.q_repr[0], q_only.q_repr[1])


def test_factorized_item_control_preserves_q_and_target_item_information() -> None:
    torch.manual_seed(42)
    module = ExerciseSpecificRequirementQuery(dim=3)
    with torch.no_grad():
        module.factorized_q_projection.weight.copy_(torch.eye(3))
        module.factorized_q_projection.bias.zero_()
        module.factorized_item_projection.weight.copy_(torch.eye(3))
        module.factorized_norm.weight.fill_(1.0)
        module.factorized_norm.bias.zero_()
    q_matrix = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    concept_nodes = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    exercise_nodes = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [0.0, 1.0, 0.5],
        ]
    )
    output = module(
        q_matrix=q_matrix,
        concept_nodes=concept_nodes,
        exercise_nodes=exercise_nodes,
        target_exercise_ids=torch.tensor([0, 1]),
        projection=torch.nn.Identity(),
        mode="factorized_item_control",
    )
    # Same Q, different exact target items remain distinguishable.
    assert not torch.equal(output.q_repr[0], output.q_repr[1])

    changed_q = q_matrix.clone()
    changed_q[0] = torch.tensor([0.0, 1.0])
    changed = module(
        q_matrix=changed_q,
        concept_nodes=concept_nodes,
        exercise_nodes=exercise_nodes,
        target_exercise_ids=torch.tensor([0]),
        projection=torch.nn.Identity(),
        mode="factorized_item_control",
    )
    # Holding the exact item vector fixed, changing Q changes the output.
    assert not torch.equal(output.q_repr[0], changed.q_repr[0])


def test_factorized_item_control_is_exactly_capacity_matched() -> None:
    model = _model("calibrated_history", "personalized_interaction")
    counts = model.active_module_parameter_counts()["target_requirement"]
    assert counts["factorized_item_control"] == counts["exercise_specific"]
    control = _model(
        "calibrated_history",
        "personalized_interaction",
        target_requirement_mode="factorized_item_control",
    )
    assert (
        control.ablation_variant_fingerprint
        != control.architecture_fingerprint
    )


def test_frozen_full_fingerprint_and_prediction_hash_are_unchanged() -> None:
    torch.manual_seed(42)
    model = TwoStageTKCUKCCDM(
        num_students=4,
        num_exercises=5,
        num_concepts=3,
        concept_dim=64,
    )
    output = model(**_inputs())
    probabilities = output.probs.detach().cpu().contiguous()
    prediction_hash = hashlib.sha256(
        probabilities.numpy().tobytes()
    ).hexdigest()
    assert model.architecture_fingerprint == "099906acdba8c3b4"
    assert model.ablation_variant_fingerprint == "099906acdba8c3b4"
    assert prediction_hash == (
        "e0936b1175c34ccb4138cc24d026b4ec"
        "814492d1f670f0ba8b5d0fad88484afa"
    )
    assert torch.equal(
        probabilities,
        torch.tensor(
            [
                0.4899579882621765,
                0.4760880172252655,
                0.4507520794868469,
            ]
        ),
    )


def test_factorized_item_control_initialization_is_isolated() -> None:
    torch.manual_seed(1234)
    rng_before = torch.random.get_rng_state()
    ExerciseSpecificRequirementQuery(dim=16)
    rng_after = torch.random.get_rng_state()
    assert torch.equal(rng_before, rng_after)

    # Dormant control parameters must not perturb pre-existing Full weights.
    model = _model("calibrated_history", "personalized_interaction")
    expected_hashes = {
        "cognitive_match.0.weight": (
            "8bfe2ec88aa5f24b4e872fe713ed47ff"
            "517f178d31ecc8cf50e0d187efed496a"
        ),
        "guess_head.0.weight": (
            "234c9f7f4891eb14c749a0cb31df7b8"
            "96302687e150c004755ef4befa96dc93a"
        ),
        "slip_head.0.weight": (
            "78780caa66bc921e9a90d6266cc93d7"
            "4501f0989c82eb2c39f252da863ba15a0"
        ),
    }
    for name, expected in expected_hashes.items():
        value = model.state_dict()[name]
        actual = hashlib.sha256(
            value.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        assert actual == expected


def test_factorized_item_control_has_one_active_requirement_data_path() -> None:
    full = _model(
        "calibrated_history",
        "personalized_interaction",
        target_requirement_mode="exercise_specific",
    )
    full(**_inputs()).probs.mean().backward()
    assert full.q_projection[0].weight.grad is not None
    assert full.target_requirement.factorized_q_projection.weight.grad is None
    assert (
        full.target_requirement.factorized_item_projection.weight.grad is None
    )

    control = _model(
        "calibrated_history",
        "personalized_interaction",
        target_requirement_mode="factorized_item_control",
    )
    with torch.no_grad():
        control.exercise_difficulty.weight[:, 0] = torch.arange(
            control.exercise_difficulty.num_embeddings,
            dtype=control.exercise_difficulty.weight.dtype,
        )
    output = control(**_inputs())
    output.probs.mean().backward()
    assert control.q_projection[0].weight.grad is None
    assert (
        control.target_requirement.factorized_q_projection.weight.grad
        is not None
    )
    assert (
        control.target_requirement.factorized_item_projection.weight.grad
        is not None
    )
    expected_difficulty = control.exercise_difficulty(
        _inputs()["target_exercise_ids"]
    ).squeeze(-1)
    assert torch.equal(output.difficulty, expected_difficulty)
    interaction = output.module_diagnostics[
        "diagnosis_item_conditioned_interaction_norm"
    ]
    assert interaction.shape == output.probs.shape
    assert torch.isfinite(interaction).all()


def test_checkpoint_loader_only_allows_dormant_factorized_parameters_missing(
    tmp_path,
) -> None:
    model = _model("calibrated_history", "personalized_interaction")
    legacy_state = {
        name: value
        for name, value in model.state_dict().items()
        if not name.startswith("target_requirement.factorized_")
    }
    checkpoint = tmp_path / "legacy_full.pt"
    torch.save(legacy_state, checkpoint)
    bundles = {
        "train": SimpleNamespace(
            num_students=4,
            num_exercises=5,
            num_concepts=3,
        )
    }
    summary = {
        "model": "two_stage_tkc_ukc",
        "target_requirement_mode": "exercise_specific",
    }
    loaded = load_model(
        summary=summary,
        checkpoint_path=str(checkpoint),
        bundles=bundles,
        concept_dim=16,
        device="cpu",
    )
    assert isinstance(loaded, TwoStageTKCUKCCDM)
    assert loaded.architecture_fingerprint == model.architecture_fingerprint
    with torch.no_grad():
        expected = model(**_inputs()).probs
        actual = loaded(**_inputs()).probs
    assert torch.equal(actual, expected)
    assert hashlib.sha256(
        actual.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest() == hashlib.sha256(
        expected.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()

    summary["target_requirement_mode"] = "factorized_item_control"
    try:
        load_model(
            summary=summary,
            checkpoint_path=str(checkpoint),
            bundles=bundles,
            concept_dim=16,
            device="cpu",
        )
    except RuntimeError as error:
        assert "Checkpoint state mismatch" in str(error)
    else:
        raise AssertionError(
            "An active factorized control must reject a legacy checkpoint."
        )


def test_attentive_field_has_query_specific_state_contract() -> None:
    full = _model("calibrated_history", "query_attentive_field")
    control = _model("calibrated_history", "global_attentive_control")
    full_output = full(**_inputs())
    control_output = control(**_inputs())
    assert full_output.framework_state.shape == (2, 3, 16)
    assert control_output.framework_state.shape == (2, 3, 16)
    assert not torch.equal(
        full_output.framework_state,
        control_output.framework_state,
    )
    assert torch.isfinite(full_output.framework_state).all()
    assert torch.isfinite(control_output.framework_state).all()


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
