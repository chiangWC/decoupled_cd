import unittest

from models.unified_v2_spec import UnifiedArchitectureSpec


class UnifiedArchitectureSpecTests(unittest.TestCase):
    def test_numeric_training_hyperparameters_do_not_change_fingerprint(self):
        left = UnifiedArchitectureSpec(inference="graph", composer="coverage")
        right = UnifiedArchitectureSpec(inference="graph", composer="coverage")
        self.assertEqual(left.fingerprint(), right.fingerprint())

    def test_module_change_changes_fingerprint(self):
        base = UnifiedArchitectureSpec(inference="prior", composer="mask")
        graph = UnifiedArchitectureSpec(inference="graph", composer="mask")
        self.assertNotEqual(base.fingerprint(), graph.fingerprint())

    def test_invalid_combination_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "coverage composer requires graph inference"):
            UnifiedArchitectureSpec(inference="prior", composer="coverage")
