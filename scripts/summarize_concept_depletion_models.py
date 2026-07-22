from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


GRAPH_MODELS = ("ORCDF-NCD", "SVGCD")
NON_GRAPH_MODEL = "KaNCD"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Gate C cross-model support consistency."
    )
    parser.add_argument(
        "--model-summary",
        action="append",
        nargs=2,
        required=True,
        metavar=("MODEL", "GATE_SUMMARY_JSON"),
    )
    parser.add_argument(
        "--official-summary",
        type=Path,
        help=("Optional EduCDM KaNCD-GMF sensitivity-analysis summary."),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_support(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("datasets")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: missing dataset results.")
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        dataset = str(row["dataset"])
        if dataset in output:
            raise ValueError(f"{path}: duplicate dataset {dataset}.")
        output[dataset] = row
    return output


def summarize(
    model_paths: dict[str, Path],
    official_path: Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    required = {*GRAPH_MODELS, NON_GRAPH_MODEL}
    if set(model_paths) != required:
        raise ValueError(
            f"Gate C requires exactly {sorted(required)}, got "
            f"{sorted(model_paths)}."
        )
    results = {
        model: load_support(path) for model, path in model_paths.items()
    }
    dataset_sets = {model: set(rows) for model, rows in results.items()}
    first = next(iter(dataset_sets.values()))
    if any(values != first for values in dataset_sets.values()):
        raise ValueError(f"Model dataset sets differ: {dataset_sets}.")
    official_results = (
        load_support(official_path) if official_path is not None else None
    )
    if official_results is not None and set(official_results) != first:
        raise ValueError(
            "Official KaNCD dataset set differs: "
            f"{set(official_results)} != {first}."
        )
    table_rows: list[dict[str, Any]] = []
    common: list[str] = []
    robust_common: list[str] = []
    for dataset in sorted(first):
        support = {
            model: bool(
                results[model][dataset][
                    "supports_concept_specific_damage"
                ]
            )
            for model in sorted(required)
        }
        graph_support = any(support[model] for model in GRAPH_MODELS)
        cross_family = support[NON_GRAPH_MODEL] and graph_support
        if cross_family:
            common.append(dataset)
        official_support = (
            bool(
                official_results[dataset][
                    "supports_concept_specific_damage"
                ]
            )
            if official_results is not None
            else None
        )
        robust_cross_family = bool(official_support and graph_support)
        if robust_cross_family:
            robust_common.append(dataset)
        table_rows.append(
            {
                "dataset": dataset,
                "orcdf_ncd_support": support["ORCDF-NCD"],
                "svgcd_support": support["SVGCD"],
                "kancd_support": support["KaNCD"],
                "graph_support": graph_support,
                "cross_family_support": cross_family,
                "official_kancd_support": official_support,
                "robust_cross_family_support": robust_cross_family,
                "orcdf_log_loss_damage": results["ORCDF-NCD"][dataset][
                    "log_loss_damage_concept_minus_random"
                ],
                "svgcd_log_loss_damage": results["SVGCD"][dataset][
                    "log_loss_damage_concept_minus_random"
                ],
                "kancd_log_loss_damage": results["KaNCD"][dataset][
                    "log_loss_damage_concept_minus_random"
                ],
                "official_kancd_log_loss_damage": (
                    official_results[dataset][
                        "log_loss_damage_concept_minus_random"
                    ]
                    if official_results is not None
                    else None
                ),
            }
        )
    payload = {
        "schema_version": 1,
        "gate": "cross_model_concept_depletion_consistency",
        "models": sorted(required),
        "required_common_datasets": 3,
        "common_cross_family_datasets": common,
        "common_cross_family_count": len(common),
        "admitted": len(common) >= 3,
        "decision": (
            "admit_tkc_ukc_problem"
            if len(common) >= 3
            else "reject_tkc_ukc_as_main_cross_dataset_problem"
        ),
        "model_summary_paths": {
            model: str(path.resolve()) for model, path in model_paths.items()
        },
    }
    if official_results is not None:
        payload["official_cross_check"] = {
            "model": "EduCDM-KaNCD-GMF",
            "summary_path": str(official_path.resolve()),
            "common_cross_family_datasets": robust_common,
            "common_cross_family_count": len(robust_common),
            "required_common_datasets": 3,
            "admitted": len(robust_common) >= 3,
            "decision": (
                "confirm_tkc_ukc_problem"
                if len(robust_common) >= 3
                else "reject_broad_problem_after_official_cross_check"
            ),
        }
    return payload, pd.DataFrame(table_rows)


def main() -> None:
    args = parse_args()
    model_paths = {
        model: Path(path) for model, path in args.model_summary
    }
    if len(model_paths) != len(args.model_summary):
        raise ValueError("Duplicate model summary name.")
    payload, table = summarize(model_paths, args.official_summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "gate_c_support_matrix.csv", index=False)
    (args.output_dir / "gate_c_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
