from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify locked validation-only Full/History-control artifacts "
            "before the factorized Requirement experiment."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_equal(
    checks: list[dict[str, Any]],
    *,
    name: str,
    actual: Any,
    expected: Any,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": actual == expected,
            "actual": actual,
            "expected": expected,
        }
    )


def audit_entry(
    *,
    name: str,
    entry: dict[str, Any],
    artifact_root: Path,
    base_fingerprint: str,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    resolved: dict[str, Path] = {}
    for artifact_name in ["summary", "checkpoint", "predictions"]:
        record = entry[artifact_name]
        path = artifact_root / record["path"]
        resolved[artifact_name] = path
        check_equal(
            checks,
            name=f"{artifact_name}.exists",
            actual=path.is_file(),
            expected=True,
        )
        if path.is_file():
            check_equal(
                checks,
                name=f"{artifact_name}.sha256",
                actual=sha256_file(path),
                expected=record["sha256"],
            )

    summary_path = resolved["summary"]
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for key, expected in entry["recipe"].items():
            check_equal(
                checks,
                name=f"recipe.{key}",
                actual=summary.get(key),
                expected=expected,
            )
        check_equal(
            checks,
            name="validation_only",
            actual=summary.get("evaluation_stage"),
            expected="validation",
        )
        check_equal(
            checks,
            name="test_metrics_absent",
            actual=summary.get("test_metrics"),
            expected=None,
        )
        check_equal(
            checks,
            name="base_architecture_fingerprint",
            actual=summary.get("architecture_fingerprint"),
            expected=base_fingerprint,
        )
        for data_name, expected in entry["data"].items():
            data_path_value = summary.get(data_name)
            data_path = (
                Path(data_path_value)
                if isinstance(data_path_value, str)
                else Path("__missing__")
            )
            check_equal(
                checks,
                name=f"data.{data_name}.exists",
                actual=data_path.is_file(),
                expected=True,
            )
            if data_path.is_file():
                check_equal(
                    checks,
                    name=f"data.{data_name}.sha256",
                    actual=sha256_file(data_path),
                    expected=expected["sha256"],
                )

    return {
        "entry": name,
        "dataset": entry["dataset"],
        "split": entry["split"],
        "role": entry["role"],
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    artifact_root = Path(args.artifact_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("test_artifacts_in_scope") is not False:
        raise ValueError("The artifact lock must explicitly exclude test artifacts.")
    reports = [
        audit_entry(
            name=name,
            entry=entry,
            artifact_root=artifact_root,
            base_fingerprint=manifest["base_architecture_fingerprint"],
        )
        for name, entry in manifest["entries"].items()
    ]
    payload = {
        "manifest": str(manifest_path),
        "artifact_root": str(artifact_root),
        "entry_count": len(reports),
        "all_passed": all(report["passed"] for report in reports),
        "reports": reports,
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
