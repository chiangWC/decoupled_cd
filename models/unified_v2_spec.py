from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Literal


_FINGERPRINT_NOT_PROVIDED = object()


@dataclass(frozen=True)
class UnifiedArchitectureSpec:
    inference: Literal["prior", "graph"] = "prior"
    composer: Literal["mask", "coverage"] = "mask"
    decoder: Literal["neuralcdm-monotonic"] = "neuralcdm-monotonic"
    mastery_output: Literal["student-concept"] = "student-concept"
    version: int = 2

    def __post_init__(self) -> None:
        if type(self.inference) is not str or self.inference not in {
            "prior",
            "graph",
        }:
            raise ValueError("inference must be 'prior' or 'graph'")
        if type(self.composer) is not str or self.composer not in {
            "mask",
            "coverage",
        }:
            raise ValueError("composer must be 'mask' or 'coverage'")
        if (
            type(self.decoder) is not str
            or self.decoder != "neuralcdm-monotonic"
        ):
            raise ValueError("decoder must be 'neuralcdm-monotonic'")
        if (
            type(self.mastery_output) is not str
            or self.mastery_output != "student-concept"
        ):
            raise ValueError("mastery_output must be 'student-concept'")
        if type(self.version) is not int or self.version != 2:
            raise ValueError("version must be integer 2")
        if self.composer == "coverage" and self.inference != "graph":
            raise ValueError("coverage composer requires graph inference")

    @classmethod
    def from_manifest(
        cls,
        manifest: object,
        *,
        architecture_fingerprint: object = _FINGERPRINT_NOT_PROVIDED,
    ) -> UnifiedArchitectureSpec:
        if type(manifest) is not dict:
            raise ValueError("architecture_manifest must be a JSON object")
        expected_keys = {
            "inference",
            "composer",
            "decoder",
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
            "inference",
            "composer",
            "decoder",
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
                inference=manifest["inference"],
                composer=manifest["composer"],
                decoder=manifest["decoder"],
                mastery_output=manifest["mastery_output"],
                version=manifest["version"],
            )
        except ValueError as exc:
            raise ValueError(f"invalid architecture_manifest: {exc}") from exc
        if manifest != architecture.manifest():
            raise ValueError(
                "architecture_manifest modules do not match the selected "
                "inference and composer"
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
            ("prior", "mask"): "m1-m4-neuralcdm",
            ("graph", "mask"): "m1-m2-m4-neuralcdm",
            ("graph", "coverage"): "m1-m2-m3-m4-neuralcdm",
        }[(self.inference, self.composer)]
        return payload

    def fingerprint(self) -> str:
        payload = json.dumps(self.manifest(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
