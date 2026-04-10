import unittest

import torch

from models.hetero_propagation import _build_exercise_component


class ExerciseComponentAggregationTest(unittest.TestCase):
    def test_unrelated_history_does_not_shrink_concept_behavior_signal(self) -> None:
        exercise_messages = torch.tensor(
            [
                [1.0],
                [10.0],
                [100.0],
                [1000.0],
            ],
            dtype=torch.float32,
        )
        q_matrix = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        concept_0_only_history = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        concept_0_with_extra_unrelated_history = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)

        only_history_output = _build_exercise_component(
            weighted_exercises=concept_0_only_history,
            q_matrix=q_matrix,
            exercise_messages=exercise_messages,
        )
        extra_history_output = _build_exercise_component(
            weighted_exercises=concept_0_with_extra_unrelated_history,
            q_matrix=q_matrix,
            exercise_messages=exercise_messages,
        )

        self.assertAlmostEqual(only_history_output[0, 0, 0].item(), 5.5, places=6)
        self.assertAlmostEqual(extra_history_output[0, 0, 0].item(), 5.5, places=6)
        self.assertAlmostEqual(extra_history_output[0, 1, 0].item(), 550.0, places=6)


if __name__ == "__main__":
    unittest.main()
