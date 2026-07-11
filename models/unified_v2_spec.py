from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Literal


@dataclass(frozen=True)
class UnifiedArchitectureSpec:
    inference: Literal["prior", "graph"] = "prior"
    composer: Literal["mask", "coverage"] = "mask"
    decoder: Literal["monotonic"] = "monotonic"
    mastery_output: Literal["student-concept"] = "student-concept"
    version: int = 1

    def __post_init__(self) -> None:
        if self.composer == "coverage" and self.inference != "graph":
            raise ValueError("coverage composer requires graph inference")

    def manifest(self) -> dict[str, str | int]:
        payload = asdict(self)
        payload["modules"] = {
            ("prior", "mask"): "m1-m4",
            ("graph", "mask"): "m1-m2-m4",
            ("graph", "coverage"): "m1-m2-m3-m4",
        }[(self.inference, self.composer)]
        return payload

    def fingerprint(self) -> str:
        payload = json.dumps(self.manifest(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
