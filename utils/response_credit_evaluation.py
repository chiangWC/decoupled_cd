from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


VARIANTS = ("full", "direct", "capacity")
PROBABILITY_COLUMNS = {
    "full": "prob_full",
    "direct": "prob_direct",
    "capacity": "prob_capacity",
}
BOOTSTRAP_REPLICATES = 2_000
MIN_VALID_BOOTSTRAP_REPLICATES = 1_800
BOOTSTRAP_NAMESPACE = "response-credit-joint-C-student-bootstrap"
EXPECTED_DATASETS = ("MOOCRadar", "NIPS34")


def _hash_values(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def _auc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return None
    value = float(roc_auc_score(labels, probabilities))
    return value if np.isfinite(value) else None


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    indices = np.minimum(np.digitize(probabilities, boundaries[1:-1]), bins - 1)
    value = 0.0
    for index in range(bins):
        mask = indices == index
        if not bool(mask.any()):
            continue
        value += float(mask.mean()) * abs(
            float(probabilities[mask].mean()) - float(labels[mask].mean())
        )
    return value


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float | int | None]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != probabilities.shape:
        raise ValueError("Labels and probabilities must share shape.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite.")
    if not bool(np.all((probabilities >= 0.0) & (probabilities <= 1.0))):
        raise ValueError("Probabilities must lie in [0, 1].")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Labels must be binary.")
    if len(labels) == 0:
        return {
            "rows": 0,
            "positive_rows": 0,
            "negative_rows": 0,
            "auc": None,
            "acc": None,
            "rmse": None,
            "brier": None,
            "ece": None,
        }
    error = probabilities - labels
    return {
        "rows": int(len(labels)),
        "positive_rows": int(labels.sum()),
        "negative_rows": int(len(labels) - labels.sum()),
        "auc": _auc(labels, probabilities),
        "acc": float(np.mean((probabilities >= 0.5) == labels)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "brier": float(np.mean(np.square(error))),
        "ece": _ece(labels, probabilities),
    }


@dataclass(frozen=True)
class BootstrapResult:
    joint_min_delta: np.ndarray
    valid: np.ndarray
    invalid_indices: np.ndarray
    sampled_student_hashes: tuple[str, ...]
    sampling_manifest_sha256: str
    confidence_interval: tuple[float, float] | None

    def summary(self) -> dict[str, Any]:
        return {
            "requested_replicates": int(len(self.valid)),
            "valid_replicates": int(self.valid.sum()),
            "invalid_replicates": int((~self.valid).sum()),
            "invalid_indices": self.invalid_indices.tolist(),
            "sampling_manifest_sha256": self.sampling_manifest_sha256,
            "confidence_interval_95": (
                None
                if self.confidence_interval is None
                else list(self.confidence_interval)
            ),
            "minimum_valid_replicates": MIN_VALID_BOOTSTRAP_REPLICATES,
        }


def student_clustered_joint_bootstrap(
    frame: pd.DataFrame,
    *,
    stratum: str,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = 2024,
) -> BootstrapResult:
    required = {"stu_id", "label", *PROBABILITY_COLUMNS.values()}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Bootstrap frame is missing columns: {sorted(missing)}")
    if replicates < 1:
        raise ValueError("replicates must be positive.")
    students = sorted(frame["stu_id"].astype(str).unique())
    if not students:
        raise ValueError("Bootstrap frame has no students.")
    row_indices = {
        student: np.flatnonzero(frame["stu_id"].astype(str).to_numpy() == student)
        for student in students
    }
    seed_bytes = hashlib.sha256(
        f"{seed}\x1f{BOOTSTRAP_NAMESPACE}\x1f{stratum}".encode("utf-8")
    ).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(seed_bytes, "little"))
    labels = frame["label"].to_numpy(dtype=np.int64)
    probabilities = {
        name: frame[column].to_numpy(dtype=np.float64)
        for name, column in PROBABILITY_COLUMNS.items()
    }
    effects = np.full(replicates, np.nan, dtype=np.float64)
    valid = np.zeros(replicates, dtype=bool)
    hashes: list[str] = []
    for replicate in range(replicates):
        sampled = rng.choice(students, size=len(students), replace=True)
        hashes.append(_hash_values(sorted(sampled.tolist())))
        indices = np.concatenate([row_indices[student] for student in sampled])
        aucs = {
            name: _auc(labels[indices], values[indices])
            for name, values in probabilities.items()
        }
        if any(value is None for value in aucs.values()):
            continue
        effects[replicate] = min(
            float(aucs["full"]) - float(aucs["direct"]),
            float(aucs["full"]) - float(aucs["capacity"]),
        )
        valid[replicate] = True
    valid_effects = effects[valid]
    interval = (
        None
        if len(valid_effects) == 0
        else (
            float(np.quantile(valid_effects, 0.025)),
            float(np.quantile(valid_effects, 0.975)),
        )
    )
    invalid = np.flatnonzero(~valid).astype(np.int64)
    return BootstrapResult(
        joint_min_delta=effects,
        valid=valid,
        invalid_indices=invalid,
        sampled_student_hashes=tuple(hashes),
        sampling_manifest_sha256=_hash_values(
            f"{index}:{value}" for index, value in enumerate(hashes)
        ),
        confidence_interval=interval,
    )


def _scope_mask(frame: pd.DataFrame, scope: str) -> np.ndarray:
    if scope == "overall":
        return np.ones(len(frame), dtype=bool)
    column = {
        "C": "in_c",
        "C_strict": "in_c_strict",
        "T": "in_t",
    }[scope]
    return frame[column].astype(bool).to_numpy()


def evaluate_credit_predictions(
    frame: pd.DataFrame,
    *,
    dataset: str,
    split_kind: str,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = 2024,
) -> tuple[dict[str, Any], BootstrapResult]:
    if dataset not in EXPECTED_DATASETS:
        raise ValueError(f"Unexpected dataset: {dataset}.")
    if split_kind not in {"holdout", "standard"}:
        raise ValueError("split_kind must be holdout or standard.")
    required = {
        "source_row_id",
        "stu_id",
        "label",
        "in_c",
        "in_c_strict",
        "in_t",
        *PROBABILITY_COLUMNS.values(),
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Evaluation frame is missing: {sorted(missing)}")
    if frame["source_row_id"].astype(str).duplicated().any():
        raise RuntimeError("Evaluation rows contain duplicate source_row_id values.")

    metrics: dict[str, dict[str, Any]] = {}
    for scope in ("overall", "C", "C_strict", "T"):
        mask = _scope_mask(frame, scope)
        metrics[scope] = {
            variant: binary_metrics(
                frame.loc[mask, "label"].to_numpy(dtype=np.int64),
                frame.loc[mask, column].to_numpy(dtype=np.float64),
            )
            for variant, column in PROBABILITY_COLUMNS.items()
        }

    deltas: dict[str, dict[str, float | None]] = {}
    for scope, variants in metrics.items():
        full_auc = variants["full"]["auc"]
        control_aucs = (
            variants["direct"]["auc"],
            variants["capacity"]["auc"],
        )
        auc_delta = (
            None
            if full_auc is None or any(value is None for value in control_aucs)
            else float(full_auc) - max(float(value) for value in control_aucs)
        )
        brier_values = (
            variants["full"]["brier"],
            variants["direct"]["brier"],
            variants["capacity"]["brier"],
        )
        brier_delta = (
            None if any(value is None for value in brier_values)
            else float(brier_values[0])
            - min(float(value) for value in brier_values[1:])
        )
        deltas[scope] = {
            "auc_vs_control_envelope": auc_delta,
            "brier_vs_control_envelope": brier_delta,
            "full_minus_direct_auc": (
                None
                if full_auc is None or variants["direct"]["auc"] is None
                else float(full_auc) - float(variants["direct"]["auc"])
            ),
            "full_minus_capacity_auc": (
                None
                if full_auc is None or variants["capacity"]["auc"] is None
                else float(full_auc) - float(variants["capacity"]["auc"])
            ),
        }

    c_frame = frame.loc[_scope_mask(frame, "C")].copy()
    bootstrap = student_clustered_joint_bootstrap(
        c_frame,
        stratum=f"{dataset}:{split_kind}:C",
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    c_control_auc = {
        name: metrics["C"][name]["auc"] for name in ("direct", "capacity")
    }
    descriptive_control = (
        "capacity"
        if c_control_auc["capacity"] is not None
        and (
            c_control_auc["direct"] is None
            or float(c_control_auc["capacity"])
            >= float(c_control_auc["direct"]) - 1e-12
        )
        else "direct"
    )
    summary = {
        "schema_version": 1,
        "dataset": dataset,
        "split_kind": split_kind,
        "metrics": metrics,
        "control_envelope_deltas": deltas,
        "descriptive_higher_c_auc_control": descriptive_control,
        "slice_prevalence": {
            scope: {
                "rows": int(_scope_mask(frame, scope).sum()),
                "students": int(
                    frame.loc[_scope_mask(frame, scope), "stu_id"]
                    .astype(str)
                    .nunique()
                ),
            }
            for scope in ("C", "C_strict", "T")
        },
        "joint_c_bootstrap": bootstrap.summary(),
    }
    return summary, bootstrap


def compute_stage1_gate(evaluations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_dataset = {str(value["dataset"]): value for value in evaluations}
    if set(by_dataset) != set(EXPECTED_DATASETS) or len(evaluations) != 2:
        raise RuntimeError("Stage 1 requires one holdout evaluation per dataset.")
    checks: dict[str, bool] = {}
    c_deltas = {}
    t_deltas = {}
    for dataset, value in by_dataset.items():
        if value["split_kind"] != "holdout":
            raise RuntimeError("Stage 1 accepts holdout evaluations only.")
        prevalence = value["slice_prevalence"]["C"]
        metrics = value["metrics"]
        c_delta = value["control_envelope_deltas"]["C"]["auc_vs_control_envelope"]
        t_delta = value["control_envelope_deltas"]["T"]["auc_vs_control_envelope"]
        c_deltas[dataset] = c_delta
        t_deltas[dataset] = t_delta
        checks[f"{dataset}_C_identified"] = (
            prevalence["rows"] >= 500
            and prevalence["students"] >= 100
            and metrics["C"]["full"]["auc"] is not None
        )
        checks[f"{dataset}_T_defined"] = metrics["T"]["full"]["auc"] is not None
        checks[f"{dataset}_C_delta_ge_0.002"] = (
            c_delta is not None and float(c_delta) >= 0.002
        )
        checks[f"{dataset}_T_delta_ge_minus_0.001"] = (
            t_delta is not None and float(t_delta) >= -0.001
        )
        checks[f"{dataset}_holdout_overall_auc_guard"] = (
            value["control_envelope_deltas"]["overall"]["auc_vs_control_envelope"]
            is not None
            and value["control_envelope_deltas"]["overall"][
                "auc_vs_control_envelope"
            ]
            >= -0.001
        )
        for scope in ("overall", "C", "T"):
            brier_delta = value["control_envelope_deltas"][scope][
                "brier_vs_control_envelope"
            ]
            checks[f"{dataset}_{scope}_brier_guard"] = (
                brier_delta is not None and float(brier_delta) <= 0.0002
            )
        bootstrap = value["joint_c_bootstrap"]
        checks[f"{dataset}_bootstrap_valid_ge_1800"] = (
            bootstrap["requested_replicates"] == BOOTSTRAP_REPLICATES
            and bootstrap["valid_replicates"] >= MIN_VALID_BOOTSTRAP_REPLICATES
        )
    checks["both_C_delta_ge_0.002"] = all(
        value is not None and float(value) >= 0.002 for value in c_deltas.values()
    )
    checks["one_C_delta_ge_0.003"] = any(
        value is not None and float(value) >= 0.003 for value in c_deltas.values()
    )
    checks["one_T_delta_ge_0.001"] = any(
        value is not None and float(value) >= 0.001 for value in t_deltas.values()
    )
    checks["one_joint_C_ci_low_gt_zero"] = any(
        value["joint_c_bootstrap"]["confidence_interval_95"] is not None
        and value["joint_c_bootstrap"]["confidence_interval_95"][0] > 0.0
        for value in by_dataset.values()
    )
    return {
        "schema_version": 1,
        "gate": "response_credit_stage1_holdout",
        "stage1_passed": all(checks.values()),
        "checks": checks,
        "datasets": by_dataset,
        "control_rule": {
            "auc": "full minus max(direct,capacity) per metric",
            "brier": "full minus min(direct,capacity) per metric",
        },
    }


def compute_stage2_gate(
    standard_evaluations: Sequence[dict[str, Any]],
    *,
    stage1: dict[str, Any],
) -> dict[str, Any]:
    if not bool(stage1.get("stage1_passed")):
        raise RuntimeError("Stage 2 is forbidden unless Stage 1 passed.")
    by_dataset = {str(value["dataset"]): value for value in standard_evaluations}
    if set(by_dataset) != set(EXPECTED_DATASETS) or len(standard_evaluations) != 2:
        raise RuntimeError("Stage 2 requires one standard evaluation per dataset.")
    checks: dict[str, bool] = {}
    for dataset, value in by_dataset.items():
        if value["split_kind"] != "standard":
            raise RuntimeError("Stage 2 accepts standard evaluations only.")
        for scope in ("overall", "C"):
            auc_delta = value["control_envelope_deltas"][scope][
                "auc_vs_control_envelope"
            ]
            checks[f"{dataset}_{scope}_auc_guard"] = (
                auc_delta is not None and float(auc_delta) >= -0.001
            )
            brier_delta = value["control_envelope_deltas"][scope][
                "brier_vs_control_envelope"
            ]
            checks[f"{dataset}_{scope}_brier_guard"] = (
                brier_delta is not None and float(brier_delta) <= 0.0002
            )
    return {
        "schema_version": 1,
        "gate": "response_credit_stage2_standard",
        "activated": all(checks.values()),
        "stage1": stage1,
        "checks": checks,
        "standard_datasets": by_dataset,
    }
