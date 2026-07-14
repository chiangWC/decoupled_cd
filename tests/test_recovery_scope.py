from __future__ import annotations

import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RecoveryScopeTest(unittest.TestCase):
    def test_registry_uses_only_external_performance_opponents(self) -> None:
        registry = json.loads(
            (PROJECT_ROOT / "configs" / "external_benchmark_registry.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(registry["comparison_policy"]["performance_opponents"], "external_only")
        self.assertEqual(registry["required_external_wins"], 3)
        self.assertNotIn("anchor", json.dumps(registry).lower())
        self.assertEqual(len(registry["active_pool"]), 7)
        for dataset in registry["active_pool"]:
            self.assertEqual(set(registry["datasets"][dataset]["external_axes"]), {"S", "H", "T"})

    def test_rejected_model_families_are_not_present(self) -> None:
        rejected = (
            PROJECT_ROOT / "models" / "r28_completion.py",
            PROJECT_ROOT / "models" / "r29_completion.py",
        )
        self.assertFalse(any(path.exists() for path in rejected))


if __name__ == "__main__":
    unittest.main()
