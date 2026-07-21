from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DATASETS = ("assist17", "moocradar", "xes3g5m", "junyi")
EXPECTED_MODES = {
    "evaluation_stage": "validation",
    "evidence_representation_mode": "calibrated_history",
    "target_requirement_mode": "factorized_item_control",
    "seed": 42,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing History anchor artifact: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def audit_anchor(artifact_root: Path, dataset: str) -> dict[str, Any]:
    stem = artifact_root / f"{dataset}_holdout_wo_requirement"
    summary_path = stem.with_suffix(".json")
    files = {
        "summary": checked_artifact(summary_path),
        "checkpoint": checked_artifact(Path(f"{stem}_best.pt")),
        "predictions": checked_artifact(Path(f"{stem}_predictions.csv")),
    }
    summary = json.loads(summary_path.read_text())
    for field, expected in EXPECTED_MODES.items():
        actual = summary.get(field)
        if actual != expected:
            raise ValueError(
                f"{dataset}: {field} must be {expected!r}, got {actual!r}."
            )
    if summary.get("test_metrics") is not None:
        raise ValueError(f"{dataset}: validation anchor contains test metrics.")
    valid_value = summary.get("valid_interactions")
    test_value = summary.get("test_interactions")
    if not valid_value or not test_value:
        raise ValueError(f"{dataset}: missing validation interaction paths.")
    valid_path = Path(str(valid_value))
    test_placeholder = Path(str(test_value))
    if valid_path.resolve() != test_placeholder.resolve():
        raise ValueError(
            f"{dataset}: test placeholder must be exactly valid_interactions."
        )
    expected_checkpoint = Path(f"{stem}_best.pt").name
    declared_checkpoint = Path(str(summary.get("best_checkpoint_path", ""))).name
    if declared_checkpoint != expected_checkpoint:
        raise ValueError(
            f"{dataset}: checkpoint declaration {declared_checkpoint!r} does not "
            f"match {expected_checkpoint!r}."
        )
    for field in (
        "architecture_fingerprint",
        "ablation_variant_fingerprint",
        "initialization_hash",
    ):
        if not summary.get(field):
            raise ValueError(f"{dataset}: missing {field}.")
    return {
        "dataset": dataset,
        "modes": dict(EXPECTED_MODES),
        "architecture_fingerprint": summary["architecture_fingerprint"],
        "ablation_variant_fingerprint": summary["ablation_variant_fingerprint"],
        "initialization_hash": summary["initialization_hash"],
        "valid_interactions": str(valid_path.resolve()),
        "files": files,
    }


def audit_history_gate_anchors(artifact_root: Path) -> dict[str, Any]:
    anchors = [audit_anchor(artifact_root, dataset) for dataset in DATASETS]
    architecture_fingerprints = {
        anchor["architecture_fingerprint"] for anchor in anchors
    }
    ablation_fingerprints = {
        anchor["ablation_variant_fingerprint"] for anchor in anchors
    }
    if len(architecture_fingerprints) != 1:
        raise ValueError("History anchors do not share one architecture fingerprint.")
    if len(ablation_fingerprints) != 1:
        raise ValueError("History anchors do not share one ablation fingerprint.")
    return {
        "status": "PASS",
        "purpose": "history_gate_full_anchors",
        "artifact_root": str(artifact_root.resolve()),
        "architecture_fingerprint": next(iter(architecture_fingerprints)),
        "ablation_variant_fingerprint": next(iter(ablation_fingerprints)),
        "anchors": anchors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lock the four validation-only calibrated-History anchors used by "
            "the clean History gate; no dataset split is opened."
        )
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_history_gate_anchors(args.artifact_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        "history_anchor_audit=PASS "
        f"anchors={len(result['anchors'])} "
        f"fingerprint={result['architecture_fingerprint']}"
    )


if __name__ == "__main__":
    main()
