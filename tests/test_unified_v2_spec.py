import unittest

from models.unified_v2_spec import UnifiedArchitectureSpec


class UnifiedArchitectureSpecTests(unittest.TestCase):
    def test_v4_manifest_and_v3_rejection(self):
        prior = UnifiedArchitectureSpec(completion="prior")
        graph = UnifiedArchitectureSpec(
            completion="evidence-relational-graph"
        )
        self.assertEqual(prior.version, 4)
        self.assertEqual(
            prior.behavior_model,
            "conditional-simplex-floor-2m20",
        )
        self.assertEqual(
            graph.manifest()["modules"],
            "m1-evidence-relational-graph-m3-m4",
        )
        old = dict(
            prior.manifest(),
            version=3,
            behavior_model="conditional-simplex",
        )
        with self.assertRaisesRegex(ValueError, "version 4"):
            UnifiedArchitectureSpec.from_manifest(old)

    def test_prior_and_graph_have_canonical_distinct_version_4_fingerprints(
        self,
    ):
        a0 = UnifiedArchitectureSpec(completion="prior")
        a1 = UnifiedArchitectureSpec(
            completion="evidence-relational-graph"
        )
        self.assertEqual(a0.manifest()["modules"], "m1-prior-m3-m4")
        self.assertEqual(
            a1.manifest()["modules"],
            "m1-evidence-relational-graph-m3-m4",
        )
        self.assertEqual(a0.manifest()["version"], 4)
        self.assertNotEqual(a0.fingerprint(), a1.fingerprint())

    def test_version_2_manifest_is_rejected(self):
        old = {
            "inference": "prior", "composer": "mask",
            "decoder": "neuralcdm-monotonic",
            "mastery_output": "student-concept", "version": 2,
            "modules": "m1-m4-neuralcdm",
        }
        with self.assertRaisesRegex(ValueError, "version 4"):
            UnifiedArchitectureSpec.from_manifest(old)

    def test_numeric_training_hyperparameters_do_not_change_fingerprint(self):
        left = UnifiedArchitectureSpec(
            completion="evidence-relational-graph"
        )
        right = UnifiedArchitectureSpec(
            completion="evidence-relational-graph"
        )
        self.assertEqual(left.fingerprint(), right.fingerprint())
        self.assertNotIn("rank", left.manifest())
        self.assertNotIn("learning_rate", left.manifest())

    def test_module_change_changes_fingerprint(self):
        prior = UnifiedArchitectureSpec(completion="prior")
        graph = UnifiedArchitectureSpec(
            completion="evidence-relational-graph"
        )
        self.assertNotEqual(prior.fingerprint(), graph.fingerprint())

    def test_invalid_combination_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "completion"):
            UnifiedArchitectureSpec(completion="lowrank")

    def test_manifest_fingerprint_mismatch_is_rejected(self):
        architecture = UnifiedArchitectureSpec(completion="prior")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            UnifiedArchitectureSpec.from_manifest(
                architecture.manifest(),
                architecture_fingerprint="0" * 64,
            )

    def test_manifest_schema_and_modules_must_be_canonical(self):
        architecture = UnifiedArchitectureSpec(completion="prior")
        extra = {**architecture.manifest(), "rank": 32}
        with self.assertRaisesRegex(ValueError, "canonical schema"):
            UnifiedArchitectureSpec.from_manifest(extra)
        tampered = {
            **architecture.manifest(),
            "modules": "m1-graph-m3-m4",
        }
        with self.assertRaisesRegex(ValueError, "modules"):
            UnifiedArchitectureSpec.from_manifest(tampered)
