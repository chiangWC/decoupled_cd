from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


EPSILON = 1.0e-7
ALIGNMENT_COLUMNS = ("audit_row_id", "stu_id", "exer_id", "label")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze paired concept/random depletion predictions."
    )
    parser.add_argument(
        "--dataset-spec",
        action="append",
        nargs=4,
        required=True,
        metavar=(
            "NAME",
            "PROTOCOL_DIR",
            "CONCEPT_TEST_PREDICTIONS",
            "RANDOM_TEST_PREDICTIONS",
        ),
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--min-selected", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_prediction(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    required = {*ALIGNMENT_COLUMNS, "prob"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing columns {missing}.")
    result = frame.copy()
    for column in ("audit_row_id", "stu_id", "exer_id"):
        result[column] = result[column].astype(str)
    result["label"] = pd.to_numeric(result["label"], errors="raise").astype(int)
    result["prob"] = pd.to_numeric(result["prob"], errors="raise").astype(float)
    if not result["label"].isin([0, 1]).all():
        raise ValueError(f"{source}: labels must be binary.")
    if (
        not np.isfinite(result["prob"]).all()
        or result["prob"].lt(0).any()
        or result["prob"].gt(1).any()
    ):
        raise ValueError(f"{source}: probabilities must be finite and in [0,1].")
    if result["audit_row_id"].duplicated().any():
        raise ValueError(f"{source}: duplicate audit_row_id.")
    return result


def load_selected(protocol_dir: Path) -> pd.DataFrame:
    source = protocol_dir / "selected_targets.csv"
    selected = pd.read_csv(source)
    required = {"audit_row_id", "stu_id", "exer_id", "label"}
    missing = sorted(required.difference(selected.columns))
    if missing:
        raise ValueError(f"{source}: missing columns {missing}.")
    for column in ("audit_row_id", "stu_id", "exer_id"):
        selected[column] = selected[column].astype(str)
    selected["label"] = pd.to_numeric(
        selected["label"], errors="raise"
    ).astype(int)
    if selected["stu_id"].duplicated().any():
        raise ValueError(f"{source}: Gate B requires at most one row per student.")
    return selected


def load_aligned(
    protocol_dir: Path,
    concept_path: Path,
    random_path: Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    selected = load_selected(protocol_dir)
    concept = normalize_prediction(pd.read_csv(concept_path), concept_path)
    random = normalize_prediction(pd.read_csv(random_path), random_path)
    if len(concept) != len(random):
        raise ValueError(
            f"prediction row mismatch: concept={len(concept)}, random={len(random)}"
        )
    for column in ALIGNMENT_COLUMNS:
        if not np.array_equal(
            concept[column].to_numpy(), random[column].to_numpy()
        ):
            mismatch = np.flatnonzero(
                concept[column].to_numpy() != random[column].to_numpy()
            )
            raise ValueError(
                f"{column} arm mismatch at row "
                f"{int(mismatch[0]) if len(mismatch) else -1}."
            )
    selected_ids = set(selected["audit_row_id"])
    available_ids = set(concept["audit_row_id"])
    missing = sorted(selected_ids.difference(available_ids))
    if missing:
        raise ValueError(f"Selected audit rows absent from predictions: {missing[:5]}.")
    frame = concept.loc[concept["audit_row_id"].isin(selected_ids)].copy()
    frame = frame.rename(columns={"prob": "concept_prob"})
    frame["random_prob"] = random.loc[
        random["audit_row_id"].isin(selected_ids), "prob"
    ].to_numpy()
    selected_alignment = selected.set_index("audit_row_id")
    for column in ("stu_id", "exer_id", "label"):
        expected = frame["audit_row_id"].map(selected_alignment[column]).to_numpy()
        if not np.array_equal(frame[column].to_numpy(), expected):
            raise ValueError(f"Selected target {column} alignment failed.")
    labels = frame["label"].to_numpy(float)
    for arm in ("concept", "random"):
        prob = frame[f"{arm}_prob"].to_numpy(float)
        clipped = np.clip(prob, EPSILON, 1.0 - EPSILON)
        frame[f"{arm}_log_loss"] = -(
            labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)
        )
        frame[f"{arm}_brier"] = np.square(labels - prob)
    frame["log_loss_damage"] = (
        frame["concept_log_loss"] - frame["random_log_loss"]
    )
    frame["brier_damage"] = frame["concept_brier"] - frame["random_brier"]
    hashes = {
        "manifest.json": sha256_file(protocol_dir / "manifest.json"),
        "selected_targets.csv": sha256_file(
            protocol_dir / "selected_targets.csv"
        ),
        "concept_predictions": sha256_file(concept_path),
        "random_predictions": sha256_file(random_path),
    }
    return frame.reset_index(drop=True), hashes


def percentile_interval(values: list[float]) -> tuple[float, float]:
    finite = np.asarray([value for value in values if np.isfinite(value)])
    if not len(finite):
        return float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def clustered_bootstrap(
    frame: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    students = frame["stu_id"].drop_duplicates().to_numpy()
    rows_by_student = {
        student: frame.index[frame["stu_id"].eq(student)].to_numpy()
        for student in students
    }
    rng = np.random.default_rng(seed)
    log_loss_values: list[float] = []
    brier_values: list[float] = []
    auc_values: list[float] = []
    for _ in range(replicates):
        sampled = rng.choice(students, size=len(students), replace=True)
        indices = np.concatenate([rows_by_student[student] for student in sampled])
        boot = frame.loc[indices]
        log_loss_values.append(float(boot["log_loss_damage"].mean()))
        brier_values.append(float(boot["brier_damage"].mean()))
        if boot["label"].nunique() == 2:
            auc_values.append(
                float(
                    roc_auc_score(boot["label"], boot["random_prob"])
                    - roc_auc_score(boot["label"], boot["concept_prob"])
                )
            )
    result: dict[str, dict[str, float]] = {}
    for name, values in (
        ("log_loss_damage", log_loss_values),
        ("brier_damage", brier_values),
        ("auc_damage", auc_values),
    ):
        low, high = percentile_interval(values)
        result[name] = {"ci_low": low, "ci_high": high}
    return result


def summarize_dataset(
    *,
    name: str,
    frame: pd.DataFrame,
    hashes: dict[str, str],
    bootstrap: int,
    seed: int,
    min_selected: int,
) -> dict[str, Any]:
    labels = frame["label"].to_numpy()
    concept_prob = frame["concept_prob"].to_numpy()
    random_prob = frame["random_prob"].to_numpy()
    both_labels = len(np.unique(labels)) == 2
    intervals = clustered_bootstrap(
        frame, replicates=bootstrap, seed=seed
    )
    log_loss_damage = float(frame["log_loss_damage"].mean())
    brier_damage = float(frame["brier_damage"].mean())
    auc_damage = (
        float(
            roc_auc_score(labels, random_prob)
            - roc_auc_score(labels, concept_prob)
        )
        if both_labels
        else None
    )
    qualified = len(frame) >= min_selected and both_labels
    supports = (
        qualified
        and log_loss_damage > 0.0
        and intervals["log_loss_damage"]["ci_low"] > 0.0
    )
    return {
        "dataset": name,
        "selected_rows": len(frame),
        "selected_students": int(frame["stu_id"].nunique()),
        "labels": sorted(np.unique(labels).astype(int).tolist()),
        "positive_rate": float(np.mean(labels)),
        "concept_auc": (
            float(roc_auc_score(labels, concept_prob)) if both_labels else None
        ),
        "random_auc": (
            float(roc_auc_score(labels, random_prob)) if both_labels else None
        ),
        "auc_damage_random_minus_concept": auc_damage,
        "log_loss_damage_concept_minus_random": log_loss_damage,
        "brier_damage_concept_minus_random": brier_damage,
        "bootstrap": intervals,
        "qualified_for_gate": qualified,
        "supports_concept_specific_damage": supports,
        "input_hashes": hashes,
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed != 2024:
        raise ValueError("Gate B bootstrap is frozen at seed=2024.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for raw_name, raw_protocol, raw_concept, raw_random in args.dataset_spec:
        frame, hashes = load_aligned(
            Path(raw_protocol), Path(raw_concept), Path(raw_random)
        )
        frame.to_csv(
            args.output_dir / f"{raw_name}_paired_rows.csv", index=False
        )
        summaries.append(
            summarize_dataset(
                name=raw_name,
                frame=frame,
                hashes=hashes,
                bootstrap=args.bootstrap,
                seed=args.seed,
                min_selected=args.min_selected,
            )
        )
    qualified = sum(item["qualified_for_gate"] for item in summaries)
    supporting = sum(
        item["supports_concept_specific_damage"] for item in summaries
    )
    payload = {
        "schema_version": 1,
        "gate": "paired_concept_depletion_orcdf_screen",
        "primary_estimand": "concept_log_loss_minus_random_log_loss",
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "minimum_selected": args.min_selected,
        "qualified_datasets": qualified,
        "supporting_datasets": supporting,
        "passed": qualified >= 3 and supporting >= 3,
        "svgcd_confirmation_required": qualified >= 3 and supporting >= 3,
        "datasets": summaries,
    }
    pd.DataFrame(
        [
            {
                "dataset": item["dataset"],
                "selected_rows": item["selected_rows"],
                "positive_rate": item["positive_rate"],
                "concept_auc": item["concept_auc"],
                "random_auc": item["random_auc"],
                "auc_damage": item["auc_damage_random_minus_concept"],
                "log_loss_damage": item[
                    "log_loss_damage_concept_minus_random"
                ],
                "log_loss_ci_low": item["bootstrap"]["log_loss_damage"][
                    "ci_low"
                ],
                "log_loss_ci_high": item["bootstrap"]["log_loss_damage"][
                    "ci_high"
                ],
                "brier_damage": item["brier_damage_concept_minus_random"],
                "qualified": item["qualified_for_gate"],
                "supports": item["supports_concept_specific_damage"],
            }
            for item in summaries
        ]
    ).to_csv(args.output_dir / "gate_b_results.csv", index=False)
    (args.output_dir / "gate_b_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main() -> None:
    args = parse_args()
    payload = analyze(args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
