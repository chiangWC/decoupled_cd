from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.pool_protocol import sha256_file
from scripts.audit_static_metadata_signal import (
    RelationEdge,
    _load_protocols,
    rewire_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize provenance-locked RCPK relation graphs."
    )
    parser.add_argument("--dataset", choices=["junyi", "xes3g5m"], required=True)
    parser.add_argument(
        "--variant",
        choices=["full", "rewire_0", "rewire_1", "rewire_2"],
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--xes-admission-audit")
    parser.add_argument("--static-admission-audit")
    parser.add_argument("--assist09-protocol")
    parser.add_argument("--assist09-raw")
    parser.add_argument("--nips-protocol")
    parser.add_argument("--nips-question-metadata")
    parser.add_argument("--nips-subject-metadata")
    parser.add_argument("--junyi-protocol")
    parser.add_argument("--junyi-official-log")
    parser.add_argument("--junyi-directed")
    parser.add_argument("--junyi-undirected")
    parser.add_argument("--junyi-source-repo")
    return parser.parse_args()


def _require(args: argparse.Namespace, names: tuple[str, ...]) -> None:
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        raise ValueError("Missing source arguments: " + ", ".join(missing))


def _edge_payload(edge: RelationEdge) -> dict[str, Any]:
    return {
        "relation": edge.relation,
        "source": edge.source,
        "target": edge.target,
        "directed": edge.directed,
    }


def _edge_sha256(edges: list[RelationEdge]) -> str:
    digest = hashlib.sha256()
    for edge in sorted(edges, key=RelationEdge.canonical):
        digest.update(
            json.dumps(
                _edge_payload(edge),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _load_edges(
    args: argparse.Namespace,
) -> tuple[list[RelationEdge], dict[str, Any]]:
    if args.dataset == "xes3g5m":
        _require(args, ("xes_admission_audit",))
        path = Path(args.xes_admission_audit)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("admitted") is not True:
            raise RuntimeError("XES static-relation source is not admitted.")
        edges = [
            RelationEdge(
                relation=str(edge["relation"]),
                source=str(edge["source"]),
                target=str(edge["target"]),
                directed=bool(edge["directed"]),
            )
            for edge in payload["metadata_edges"]
        ]
        return edges, {
            "admission_path": str(path.resolve()),
            "admission_sha256": sha256_file(path),
        }

    _require(
        args,
        (
            "static_admission_audit",
            "assist09_protocol",
            "assist09_raw",
            "nips_protocol",
            "nips_question_metadata",
            "nips_subject_metadata",
            "junyi_protocol",
            "junyi_official_log",
            "junyi_directed",
            "junyi_undirected",
            "junyi_source_repo",
        ),
    )
    protocol = _load_protocols(args)["junyi"]
    return list(protocol.metadata_edges), protocol.audit


def main() -> None:
    args = parse_args()
    edges, source_audit = _load_edges(args)
    rewire_audit = None
    if args.variant != "full":
        replicate = int(args.variant.rsplit("_", 1)[1])
        edges, rewire_audit = rewire_metadata(
            edges,
            dataset=args.dataset,
            replicate=replicate,
        )
    ordered = sorted(edges, key=RelationEdge.canonical)
    payload = {
        "schema_version": 1,
        "dataset": args.dataset,
        "variant": args.variant,
        "source_audit": source_audit,
        "rewire_audit": rewire_audit,
        "metadata_edge_sha256": _edge_sha256(ordered),
        "metadata_edges": [_edge_payload(edge) for edge in ordered],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "sha256": sha256_file(output),
                "metadata_edges": len(ordered),
                "variant": args.variant,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
