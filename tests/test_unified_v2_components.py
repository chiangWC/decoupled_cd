import inspect
import unittest

import torch

from models.unified_decoupled_cdm import UnifiedDecoupledCDM
from models.unified_v2_components import (
    MonotonicDiagnosisDecoder,
    TestedKnowledgeEvidenceEncoder,
    UntestedKnowledgeInferenceNetwork,
)
from models.unified_v2_spec import UnifiedArchitectureSpec


class UnifiedComponentTests(unittest.TestCase):
    def test_b0_always_emits_student_concept_mastery(self):
        model = UnifiedDecoupledCDM(
            num_students=3,
            num_exercises=2,
            num_concepts=3,
            dim=4,
            architecture=UnifiedArchitectureSpec(inference="prior", composer="mask"),
        )
        q = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
        history = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        tkc = (history @ q > 0).float()
        tensors = {
            "q_matrix": q,
            "concept_graph": torch.eye(3),
            "student_exercise_mask": history,
            "response_matrix": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
            "student_tkc_mask": tkc,
            "student_ukc_mask": 1.0 - tkc,
            "student_concept_evidence": None,
            "target_student_ids": torch.tensor([0, 1, 2]),
            "target_exercise_ids": torch.tensor([0, 1, 1]),
        }
        output = model(**tensors)
        self.assertEqual(tuple(output.mastery.shape), (3, 3))
        self.assertTrue(torch.isfinite(output.mastery).all())

    def test_graph_inference_wires_m2_into_state_map(self):
        model = UnifiedDecoupledCDM(
            num_students=2,
            num_exercises=1,
            num_concepts=2,
            dim=4,
            architecture=UnifiedArchitectureSpec(
                inference="graph",
                composer="mask",
            ),
        )
        with torch.no_grad():
            first_encoder = model.evidence_encoder.encoder[0]
            second_encoder = model.evidence_encoder.encoder[2]
            first_encoder.weight.zero_()
            first_encoder.bias.zero_()
            first_encoder.weight[0, 2] = 1.0
            second_encoder.weight.copy_(torch.eye(4))
            second_encoder.bias.zero_()
            for layer in model.inference_network.layers:
                layer.weight.copy_(torch.eye(4))

        output = model(
            q_matrix=torch.tensor([[1.0, 0.0]]),
            concept_graph=torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
            student_exercise_mask=torch.ones(2, 1),
            response_matrix=torch.tensor([[0.0], [1.0]]),
            student_tkc_mask=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            student_ukc_mask=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
            student_concept_evidence=None,
            target_student_ids=torch.tensor([0, 1]),
            target_exercise_ids=torch.tensor([0, 0]),
        )
        self.assertFalse(
            torch.equal(output.ukc_states[0, 1], output.ukc_states[1, 1])
        )

    def test_decoder_is_monotone_in_target_mastery(self):
        decoder = MonotonicDiagnosisDecoder(num_exercises=1, num_concepts=1, dim=4)
        low = decoder.decode_from_mastery(torch.tensor([[0.2]]), torch.tensor([[1.0]]), torch.tensor([0]))
        high = decoder.decode_from_mastery(torch.tensor([[0.8]]), torch.tensor([[1.0]]), torch.tensor([0]))
        self.assertGreaterEqual(float(high), float(low))

    def test_m1_changes_with_student_responses(self):
        encoder = TestedKnowledgeEvidenceEncoder(dim=4, evidence_cap=20.0)
        q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        history = torch.ones(1, 2)
        common = {
            "q_matrix": q,
            "student_exercise_mask": history,
            "student_tkc_mask": torch.ones(1, 2),
        }
        left = encoder(response_matrix=torch.zeros(1, 2), **common)
        right = encoder(response_matrix=torch.ones(1, 2), **common)
        self.assertFalse(torch.equal(left.tkc_states, right.tkc_states))

    def test_m2_is_student_conditioned(self):
        module = UntestedKnowledgeInferenceNetwork(
            num_concepts=3,
            dim=4,
            layers=2,
        )
        graph = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        tkc_mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        ukc_mask = 1.0 - tkc_mask
        states = torch.zeros(2, 3, 4)
        states[0, 0] = 1.0
        states[1, 0] = -1.0
        out = module(
            tkc_states=states,
            tkc_mask=tkc_mask,
            ukc_mask=ukc_mask,
            concept_graph=graph,
            direct_reliability=tkc_mask,
        )
        self.assertFalse(torch.equal(out.ukc_states[0, 1], out.ukc_states[1, 1]))

    def test_m2_unreachable_ukc_uses_prior_not_static_broadcast(self):
        module = UntestedKnowledgeInferenceNetwork(
            num_concepts=3,
            dim=4,
            layers=1,
        )
        graph = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        tkc_mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        ukc_mask = 1.0 - tkc_mask
        out = module(
            tkc_states=torch.randn(2, 3, 4),
            tkc_mask=tkc_mask,
            ukc_mask=ukc_mask,
            concept_graph=graph,
            direct_reliability=tkc_mask,
        )
        self.assertTrue(
            torch.equal(
                out.ukc_states[:, 2],
                module.concept_prior[2].expand(2, -1),
            )
        )


if __name__ == "__main__":
    unittest.main()
