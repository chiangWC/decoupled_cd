from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Literal


_FINGERPRINT_NOT_PROVIDED = object()


@dataclass(frozen=True)
class UnifiedArchitectureSpec:
    mastery_estimator: Literal["evidence-parameter"] = "evidence-parameter"
    completion: Literal["prior", "lowrank"] = "prior"
    cognitive_decoder: Literal["neuralcdm-monotonic"] = "neuralcdm-monotonic"
    behavior_model: Literal["conditional-simplex"] = "conditional-simplex"
    mastery_output: Literal["student-concept"] = "student-concept"
    version: int = 3

    def __post_init__(self) -> None:
        expected = {
            "mastery_estimator": (
                self.mastery_estimator,
                "evidence-parameter",
            ),
            "cognitive_decoder": (
                self.cognitive_decoder,
                "neuralcdm-monotonic",
            ),
            "behavior_model": (
                self.behavior_model,
                "conditional-simplex",
            ),
            "mastery_output": (self.mastery_output, "student-concept"),
        }
        for name, (actual, required) in expected.items():
            if type(actual) is not str or actual != required:
                raise ValueError(f"{name} must be {required!r}")
        if type(self.completion) is not str or self.completion not in {
            "prior",
            "lowrank",
        }:
            raise ValueError("completion must be 'prior' or 'lowrank'")
        if type(self.version) is not int or self.version != 3:
            raise ValueError("version must be integer 3")

    @classmethod
    def from_manifest(
        cls,
        manifest: object,
        *,
        architecture_fingerprint: object = _FINGERPRINT_NOT_PROVIDED,
    ) -> UnifiedArchitectureSpec:
        if type(manifest) is not dict:
            raise ValueError("architecture_manifest must be a JSON object")
        if manifest.get("version") != 3:
            raise ValueError(
                "invalid architecture_manifest: version 3 is required"
            )
        expected_keys = {
            "mastery_estimator",
            "completion",
            "cognitive_decoder",
            "behavior_model",
            "mastery_output",
            "version",
            "modules",
        }
        actual_keys = set(manifest)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ValueError(
                "architecture_manifest must have exactly the canonical schema; "
                f"missing={missing}, extra={extra}"
            )
        for key in (
            "mastery_estimator",
            "completion",
            "cognitive_decoder",
            "behavior_model",
            "mastery_output",
            "modules",
        ):
            if type(manifest[key]) is not str:
                raise ValueError(
                    f"architecture_manifest field {key!r} must be a string"
                )
        if type(manifest["version"]) is not int:
            raise ValueError(
                "architecture_manifest field 'version' must be an integer"
            )
        try:
            architecture = cls(
                mastery_estimator=manifest["mastery_estimator"],
                completion=manifest["completion"],
                cognitive_decoder=manifest["cognitive_decoder"],
                behavior_model=manifest["behavior_model"],
                mastery_output=manifest["mastery_output"],
                version=manifest["version"],
            )
        except ValueError as exc:
            raise ValueError(f"invalid architecture_manifest: {exc}") from exc
        if manifest != architecture.manifest():
            raise ValueError(
                "architecture_manifest modules do not match the selected "
                "completion"
            )
        if architecture_fingerprint is not _FINGERPRINT_NOT_PROVIDED:
            if type(architecture_fingerprint) is not str:
                raise ValueError("architecture_fingerprint must be a string")
            if architecture_fingerprint != architecture.fingerprint():
                raise ValueError(
                    "architecture_fingerprint does not match the canonical "
                    "architecture_manifest"
                )
        return architecture

    def manifest(self) -> dict[str, str | int]:
        payload = asdict(self)
        payload["modules"] = {
            "prior": "m1-prior-m3-m4",
            "lowrank": "m1-lowrank-m3-m4",
        }[self.completion]
        return payload

    def fingerprint(self) -> str:
        payload = json.dumps(self.manifest(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
