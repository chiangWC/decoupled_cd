from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from models.unified_v2_spec import UnifiedArchitectureSpec
from scripts import stable_graph_validation_controller as controller


class StableGraphValidationControllerTests(unittest.TestCase):
    def test_identity_recipes_and_test_closed(self) -> None:
        self.assertEqual(controller.CAMPAIGN_ID, "unified-ergc-20260712")
        self.assertEqual(
            controller.FROZEN_RECIPES,
            {"MOOCRadar": 2, "ASSIST17": 1, "XES3G5M": 0},
        )
        self.assertEqual(
            controller.FROZEN_COHORT_SHA256,
            "6342dc8a5f73a4e03a1645780597b625c"
            "1480ba7a6513668b6766089cdd5b8a5",
        )
        self.assertEqual(
            controller.FROZEN_COMPARATOR_AUDIT_SHA256,
            "071b5df25d8641fdc249a9a561759610"
            "25b3deeba36a4a471640563ef5d0b181",
        )
        with self.assertRaisesRegex(ValueError, "test remains closed"):
            controller.issue_attempt(
                architecture="a2", dataset="ASSIST17", split="test"
            )

    def test_comparator_audit_requires_frozen_canonical_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "comparator-audit.json"
            path.write_text(
                json.dumps(
                    {
                        "audit_sha256": "0" * 64,
                        "strongest_comparators": {
                            dataset: {} for dataset in controller.FROZEN_RECIPES
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "canonical SHA-256 mismatch"):
                controller.load_verified_comparator_audit(path)

    def test_relative_gate(self) -> None:
        deltas = {
            "ASSIST17": {"standard": 0.0, "holdout": 0.0, "zero": 0.001},
            "MOOCRadar": {
                "standard": 0.0001,
                "holdout": 0.0001,
                "zero": 0.0001,
            },
            "XES3G5M": {"standard": 0.0, "holdout": 0.0, "zero": -0.0001},
        }
        self.assertTrue(controller.relative_gate(deltas)["passed"])
        deltas["XES3G5M"]["holdout"] = -1e-12
        self.assertFalse(controller.relative_gate(deltas)["passed"])

    def test_proof_binds_all_security_fields_and_artifact_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "validation-summary.json"
            artifact.write_text('{"overall_auc":0.7}\n', encoding="utf-8")
            route_commit = "a" * 40
            proof = controller.build_validation_proof(
                architecture="a2",
                dataset="ASSIST17",
                route_commit=route_commit,
                counter=1,
                artifact_path=artifact,
                nonce="b" * 64,
            )
            controller.verify_validation_proof(
                proof,
                architecture="a2",
                dataset="ASSIST17",
                route_commit=route_commit,
                counter=1,
                artifact_path=artifact,
            )

            mutations = {
                "cohort_sha256": "c" * 64,
                "architecture_fingerprint": "d" * 64,
                "route_commit": "e" * 40,
                "counter": 2,
            }
            for field, value in mutations.items():
                with self.subTest(field=field):
                    changed = json.loads(json.dumps(proof))
                    changed[field] = value
                    with self.assertRaisesRegex(ValueError, "mismatch"):
                        controller.verify_validation_proof(
                            changed,
                            architecture="a2",
                            dataset="ASSIST17",
                            route_commit=route_commit,
                            counter=1,
                            artifact_path=artifact,
                        )

            changed = json.loads(json.dumps(proof))
            changed["artifact"]["sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "mismatch"):
                controller.verify_validation_proof(
                    changed,
                    architecture="a2",
                    dataset="ASSIST17",
                    route_commit=route_commit,
                    counter=1,
                    artifact_path=artifact,
                )

            artifact.write_text('{"overall_auc":0.8}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact.*mismatch"):
                controller.verify_validation_proof(
                    proof,
                    architecture="a2",
                    dataset="ASSIST17",
                    route_commit=route_commit,
                    counter=1,
                    artifact_path=artifact,
                )

    def test_relative_failure_closes_external_and_test(self) -> None:
        deltas = {
            dataset: {"standard": -0.001, "holdout": 0.0, "zero": 0.001}
            for dataset in controller.FROZEN_RECIPES
        }
        relative = controller.issue_relative_gate(deltas)
        with self.assertRaisesRegex(ValueError, "relative gate has not passed"):
            controller.external_gate(relative, {})
        with self.assertRaisesRegex(ValueError, "test remains closed"):
            controller.authorize_test_once(relative, {})

    def test_external_nonce_is_bound_and_test_nonce_is_consumed_once(self) -> None:
        relative_deltas = {
            dataset: {"standard": 0.0, "holdout": 0.0, "zero": 0.001}
            for dataset in controller.FROZEN_RECIPES
        }
        relative = controller.issue_relative_gate(
            relative_deltas, nonce="1" * 64
        )
        comparator_deltas = {
            dataset: {"standard": 0.0, "holdout": 0.0, "zero": 0.001}
            for dataset in controller.FROZEN_RECIPES
        }
        external = controller.external_gate(
            relative, comparator_deltas, nonce="2" * 64
        )
        self.assertTrue(external["passed"])
        self.assertEqual(external["relative_nonce"], relative["nonce"])

        with tempfile.TemporaryDirectory() as temporary:
            campaign_root = Path(temporary)
            controller.register_gate_proof(
                campaign_root, "relative-gate.json", relative
            )
            controller.register_gate_proof(
                campaign_root, "external-gate.json", external
            )
            authorization = controller.authorize_test_once(
                relative,
                external,
                campaign_root=campaign_root,
                nonce="3" * 64,
            )
            self.assertEqual(authorization["nonce"], "3" * 64)
            with self.assertRaisesRegex(ValueError, "already been issued"):
                controller.authorize_test_once(
                    relative,
                    external,
                    campaign_root=campaign_root,
                    nonce="4" * 64,
                )

    def test_self_authored_gate_proof_has_no_test_authority(self) -> None:
        relative_deltas = {
            dataset: {"standard": 0.0, "holdout": 0.0, "zero": 0.001}
            for dataset in controller.FROZEN_RECIPES
        }
        relative = controller.issue_relative_gate(
            relative_deltas, nonce="1" * 64
        )
        external = controller.external_gate(
            relative, relative_deltas, nonce="2" * 64
        )
        with tempfile.TemporaryDirectory() as temporary:
            campaign_root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "registered"):
                controller.authorize_test_once(
                    relative,
                    external,
                    campaign_root=campaign_root,
                    nonce="3" * 64,
                )

    def test_fingerprints_are_v4_and_cli_exposes_required_commands(self) -> None:
        self.assertEqual(
            controller.architecture_fingerprint("a0v4"),
            UnifiedArchitectureSpec(completion="prior").fingerprint(),
        )
        self.assertEqual(
            controller.architecture_fingerprint("a2"),
            UnifiedArchitectureSpec(
                completion="evidence-relational-graph"
            ).fingerprint(),
        )
        parser = controller.build_parser()
        commands = next(
            action.choices
            for action in parser._actions
            if getattr(action, "choices", None)
        )
        self.assertEqual(
            set(commands),
            {
                "smoke",
                "run-validation",
                "replay",
                "freeze-a0v4",
                "relative-gate",
                "external-gate",
                "run-test-once",
                "status",
            },
        )


if __name__ == "__main__":
    unittest.main()
