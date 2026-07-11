import unittest

from models.unified_v2_spec import UnifiedArchitectureSpec


class UnifiedArchitectureSpecTests(unittest.TestCase):
    def test_neuralcdm_decoder_has_new_canonical_identity(self):
        manifest = UnifiedArchitectureSpec(
            inference="graph", composer="coverage"
        ).manifest()
        self.assertEqual(manifest["decoder"], "neuralcdm-monotonic")
        self.assertEqual(manifest["version"], 2)
        self.assertEqual(manifest["modules"], "m1-m2-m3-m4-neuralcdm")

    def test_old_linear_m4_manifest_is_rejected(self):
        old = {
            "inference": "graph",
            "composer": "mask",
            "decoder": "monotonic",
            "mastery_output": "student-concept",
            "version": 1,
            "modules": "m1-m2-m4",
        }
        with self.assertRaisesRegex(ValueError, "decoder|version"):
            UnifiedArchitectureSpec.from_manifest(old)

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
