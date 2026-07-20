from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.pool_protocol import (
    canonicalize_interactions,
    derive_q_matrix,
    sha256_file,
    stable_fraction,
)
from models.beta_ncd_baseline import (
    BETA_NCD_NAME,
    BETA_NCD_UPSTREAM_COMMIT,
    BETA_NCD_UPSTREAM_URL,
    BetaNCDConfig,
    BetaNCDBaseline,
)
from utils.metrics import compute_metrics
from utils.seed import set_global_seed


MODEL_SEED = 42
SPLIT_SEED = 2024
TASK_BATCH_SIZE = 8
META_LEARNING_RATE = 1e-4
ITEM_LEARNING_RATE = 1e-3
REQUIRED_MANIFEST_FILES = {
    "train.csv",
    "valid_support.csv",
    "valid_query.csv",
    "test_support.csv",
    "test_query.csv",
    "Q_matrix.csv",
}
LOADED_DATA_FILES = {
    "train.csv",
    "valid_support.csv",
    "valid_query.csv",
    "Q_matrix.csv",
}


@dataclass(frozen=True)
class VerifiedProtocol:
    manifest_path: Path
    manifest_sha256: str
    directory: Path
    manifest: dict[str, Any]
    train: pd.DataFrame
    valid_support: pd.DataFrame
    valid_query: pd.DataFrame
    q_matrix: pd.DataFrame


@dataclass(frozen=True)
class StudentTask:
    student_id: str
    support_item_ids: np.ndarray
    support_labels: np.ndarray
    query_item_ids: np.ndarray
    query_labels: np.ndarray
    support_frame: pd.DataFrame
    query_frame: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train BETA-CD with its NCD backbone as a standalone baseline on "
            "a verified student-disjoint validation protocol. This command has "
            "no test-split evaluation path."
        )
    )
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        required=True,
        help="schema-v1 student_disjoint_support_query manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Generated checkpoint, prediction and audit directory.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device; 'auto' selects CUDA when available.",
    )
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def _canonical_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _csv_row_count(path: Path) -> int:
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _read_verified_interactions(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        path,
        dtype={"stu_id": "string", "exer_id": "string", "cpt_seq": "string"},
    )
    required = {
        "source_row_id",
        "split_row_index",
        "stu_id",
        "exer_id",
        "cpt_seq",
        "label",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    canonical, removed = canonicalize_interactions(raw)
    if removed:
        raise ValueError(f"{path.name} contains {removed} exact duplicate rows.")
    source_ids = raw["source_row_id"].astype(str).reset_index(drop=True)
    if not source_ids.equals(canonical["source_row_id"].astype(str)):
        raise ValueError(f"{path.name} source_row_id values are not canonical.")
    split_indices = pd.to_numeric(
        raw["split_row_index"], errors="raise"
    ).astype(int).reset_index(drop=True)
    if split_indices.duplicated().any():
        raise ValueError(f"{path.name} has duplicate split_row_index values.")
    canonical.insert(1, "split_row_index", split_indices)
    return canonical


def load_verified_validation_protocol(manifest_path: Path) -> VerifiedProtocol:
    """Verify every manifest file but parse only train/validation/Q inputs."""

    manifest_path = manifest_path.expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise ValueError("BETA-CD requires protocol manifest schema_version=1.")
    if manifest.get("protocol") != "student_disjoint_support_query":
        raise ValueError("BETA-CD requires student_disjoint_support_query.")
    directory = Path(manifest.get("directory", "")).expanduser().resolve()
    if directory != manifest_path.parent:
        raise ValueError(
            "Manifest directory must equal the physical manifest parent: "
            f"{directory} != {manifest_path.parent}."
        )
    if manifest.get("audit", {}).get("seed") != SPLIT_SEED:
        raise ValueError(f"Protocol split seed must be {SPLIT_SEED}.")

    file_records = manifest.get("files")
    if not isinstance(file_records, dict):
        raise ValueError("Manifest files must be an object.")
    if set(file_records) != REQUIRED_MANIFEST_FILES:
        raise ValueError(
            "Manifest file set differs from the strict protocol: "
            f"{sorted(file_records)}."
        )
    for filename in sorted(REQUIRED_MANIFEST_FILES):
        record = file_records[filename]
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256_file(path)
        if actual_hash != record.get("sha256"):
            raise ValueError(
                f"{filename} hash mismatch: {actual_hash} != {record.get('sha256')}."
            )
        actual_rows = _csv_row_count(path)
        if actual_rows != int(record.get("rows", -1)):
            raise ValueError(
                f"{filename} row mismatch: {actual_rows} != {record.get('rows')}."
            )

    # Deliberately do not parse either test CSV. They are integrity-checked above
    # so a later confirmation run can prove it used the same frozen protocol.
    train = _read_verified_interactions(directory / "train.csv")
    valid_support = _read_verified_interactions(directory / "valid_support.csv")
    valid_query = _read_verified_interactions(directory / "valid_query.csv")
    raw_q = pd.read_csv(
        directory / "Q_matrix.csv",
        dtype={"exer_id": "string", "cpt_seq": "string"},
    )
    q_matrix, _ = derive_q_matrix(raw_q)

    train_students = set(train["stu_id"])
    valid_students = set(valid_support["stu_id"]) | set(valid_query["stu_id"])
    if train_students & valid_students:
        raise ValueError("Optimizer-train and validation students overlap.")
    if set(valid_support["stu_id"]) != set(valid_query["stu_id"]):
        raise ValueError("Validation support/query student sets differ.")
    support_groups = set(
        zip(valid_support["stu_id"], valid_support["exer_id"], strict=True)
    )
    query_groups = set(
        zip(valid_query["stu_id"], valid_query["exer_id"], strict=True)
    )
    if support_groups & query_groups:
        raise ValueError("Validation support/query student-exercise groups overlap.")
    known_items = set(train["exer_id"])
    for name, frame in (
        ("valid_support.csv", valid_support),
        ("valid_query.csv", valid_query),
    ):
        unknown = set(frame["exer_id"]) - known_items
        if unknown:
            raise ValueError(
                f"{name} contains {len(unknown)} optimizer-unseen exercises."
            )
    q_items = set(q_matrix["exer_id"])
    if known_items - q_items:
        raise ValueError("Optimizer-train exercises are missing from Q-matrix.")

    return VerifiedProtocol(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        directory=directory,
        manifest=manifest,
        train=train,
        valid_support=valid_support,
        valid_query=valid_query,
        q_matrix=q_matrix,
    )


def build_q_union(
    *, train: pd.DataFrame, q_matrix: pd.DataFrame
) -> tuple[torch.Tensor, dict[str, int], dict[str, int]]:
    q_by_item: dict[str, set[str]] = {}
    for row in q_matrix.itertuples(index=False):
        q_by_item.setdefault(str(row.exer_id), set()).update(
            token for token in str(row.cpt_seq).split(",") if token
        )
    items = sorted(set(train["exer_id"]), key=_canonical_sort_key)
    missing = [item for item in items if item not in q_by_item]
    if missing:
        raise ValueError(f"{len(missing)} train items have no Q mapping.")
    concepts = sorted(
        {concept for item in items for concept in q_by_item[item]},
        key=_canonical_sort_key,
    )
    if not concepts:
        raise ValueError("Q union contains no concepts.")
    item_to_index = {item: index for index, item in enumerate(items)}
    concept_to_index = {
        concept: index for index, concept in enumerate(concepts)
    }
    q_tensor = torch.zeros((len(items), len(concepts)), dtype=torch.float32)
    for item, item_index in item_to_index.items():
        for concept in q_by_item[item]:
            q_tensor[item_index, concept_to_index[concept]] = 1.0
    if not torch.all(q_tensor.sum(dim=1) > 0):
        raise ValueError("Every item must retain at least one concept after Q union.")
    return q_tensor, item_to_index, concept_to_index


def split_optimizer_train(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create a deterministic, group-atomic expected 80/20 task split."""

    assignments: dict[tuple[str, str], str] = {}
    excluded_students: list[str] = []
    for student, student_rows in frame.groupby("stu_id", sort=True):
        exercises = sorted(
            set(student_rows["exer_id"]), key=_canonical_sort_key
        )
        if len(exercises) < 2:
            excluded_students.append(str(student))
            continue
        scores = {
            exercise: stable_fraction(
                SPLIT_SEED,
                "beta-ncd-optimizer-query",
                student,
                exercise,
            )
            for exercise in exercises
        }
        query_items = {
            exercise for exercise, score in scores.items() if score >= 0.8
        }
        if not query_items:
            query_items = {max(exercises, key=lambda item: scores[item])}
        if len(query_items) == len(exercises):
            query_items.remove(min(exercises, key=lambda item: scores[item]))
        for exercise in exercises:
            assignments[(str(student), str(exercise))] = (
                "query" if exercise in query_items else "support"
            )

    roles = [
        assignments.get((str(row.stu_id), str(row.exer_id)), "excluded")
        for row in frame.itertuples(index=False)
    ]
    support = frame.loc[[role == "support" for role in roles]].copy()
    query = frame.loc[[role == "query" for role in roles]].copy()
    support_groups = set(zip(support["stu_id"], support["exer_id"], strict=True))
    query_groups = set(zip(query["stu_id"], query["exer_id"], strict=True))
    if support_groups & query_groups:
        raise RuntimeError("Internal support/query group split is not atomic.")
    assignment_payload = [
        [student, exercise, role]
        for (student, exercise), role in sorted(
            assignments.items(),
            key=lambda pair: (
                _canonical_sort_key(pair[0][0]),
                _canonical_sort_key(pair[0][1]),
            ),
        )
    ]
    assignment_hash = hashlib.sha256(
        json.dumps(
            assignment_payload, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    audit = {
        "seed": SPLIT_SEED,
        "method": "stable_hash_threshold_0.8_with_deterministic_nonempty_fallback",
        "atomic_key": ["stu_id", "exer_id"],
        "input_rows": int(len(frame)),
        "retained_students": int(support["stu_id"].nunique()),
        "excluded_students": len(excluded_students),
        "excluded_student_ids": excluded_students,
        "support_rows": int(len(support)),
        "query_rows": int(len(query)),
        "support_groups": len(support_groups),
        "query_groups": len(query_groups),
        "group_overlap": 0,
        "query_row_fraction": float(len(query) / (len(support) + len(query))),
        "assignment_sha256": assignment_hash,
    }
    return support, query, audit


def _support_hash(frame: pd.DataFrame) -> str:
    records = sorted(
        (
            str(row.exer_id),
            int(row.label),
            str(row.source_row_id),
        )
        for row in frame.itertuples(index=False)
    )
    return hashlib.sha256(
        json.dumps(records, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_student_tasks(
    *,
    support: pd.DataFrame,
    query: pd.DataFrame,
    item_to_index: dict[str, int],
) -> list[StudentTask]:
    if set(support["stu_id"]) != set(query["stu_id"]):
        raise ValueError("Every task must have both support and query responses.")
    tasks: list[StudentTask] = []
    for student in sorted(set(support["stu_id"]), key=_canonical_sort_key):
        support_rows = support.loc[support["stu_id"] == student].copy()
        query_rows = query.loc[query["stu_id"] == student].copy()
        support_rows["_item_index"] = support_rows["exer_id"].map(item_to_index)
        query_rows["_item_index"] = query_rows["exer_id"].map(item_to_index)
        if support_rows["_item_index"].isna().any() or query_rows[
            "_item_index"
        ].isna().any():
            raise ValueError(f"Task {student} contains an unknown exercise.")
        support_rows = support_rows.sort_values(
            ["_item_index", "label", "source_row_id"], kind="mergesort"
        )
        query_rows = query_rows.sort_values(
            ["split_row_index", "source_row_id"], kind="mergesort"
        )
        tasks.append(
            StudentTask(
                student_id=str(student),
                support_item_ids=support_rows["_item_index"].to_numpy(
                    dtype=np.int64
                ),
                support_labels=support_rows["label"].to_numpy(dtype=np.float32),
                query_item_ids=query_rows["_item_index"].to_numpy(
                    dtype=np.int64
                ),
                query_labels=query_rows["label"].to_numpy(dtype=np.float32),
                support_frame=support_rows.drop(columns=["_item_index"]),
                query_frame=query_rows.drop(columns=["_item_index"]),
            )
        )
    return tasks


def _task_tensors(
    task: StudentTask, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(task.support_item_ids, dtype=torch.long, device=device),
        torch.as_tensor(task.support_labels, dtype=torch.float32, device=device),
        torch.as_tensor(task.query_item_ids, dtype=torch.long, device=device),
        torch.as_tensor(task.query_labels, dtype=torch.float32, device=device),
    )


def evaluate_tasks(
    *,
    model: BetaNCDBaseline,
    tasks: list[StudentTask],
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    model.eval()
    prediction_frames: list[pd.DataFrame] = []
    student_ids: list[str] = []
    mastery_rows: list[np.ndarray] = []
    mastery_std_rows: list[np.ndarray] = []
    for task in tasks:
        support_items, support_labels, query_items, _ = _task_tensors(
            task, device
        )
        with torch.enable_grad():
            posterior = model.adapt(
                support_item_ids=support_items,
                support_labels=support_labels,
                create_graph=False,
                fixed_noise=True,
            )
        with torch.no_grad():
            probabilities = model.predict_query(
                posterior=posterior, query_item_ids=query_items
            )
            mastery, mastery_std = model.posterior_mastery(posterior)
        output = task.query_frame.copy()
        output["prob"] = probabilities.detach().cpu().numpy()
        output["support_sha256"] = _support_hash(task.support_frame)
        prediction_frames.append(output)
        student_ids.append(task.student_id)
        mastery_rows.append(mastery.detach().cpu().numpy())
        mastery_std_rows.append(mastery_std.detach().cpu().numpy())
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["split_row_index", "source_row_id"], kind="mergesort"
    )
    predictions = predictions.reset_index(drop=True)
    return predictions, {
        "student_ids": np.asarray(student_ids),
        "mastery": np.stack(mastery_rows),
        "mastery_std": np.stack(mastery_std_rows),
    }


def train_epoch(
    *,
    model: BetaNCDBaseline,
    tasks: list[StudentTask],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_every: int,
) -> float:
    model.train()
    order = np.random.default_rng(MODEL_SEED + epoch).permutation(len(tasks))
    total_loss = 0.0
    total_tasks = 0
    for batch_start in range(0, len(order), TASK_BATCH_SIZE):
        batch_indices = order[batch_start : batch_start + TASK_BATCH_SIZE]
        optimizer.zero_grad(set_to_none=True)
        task_losses: list[torch.Tensor] = []
        for task_index in batch_indices:
            task = tasks[int(task_index)]
            support_items, support_labels, query_items, query_labels = (
                _task_tensors(task, device)
            )
            posterior = model.adapt(
                support_item_ids=support_items,
                support_labels=support_labels,
                create_graph=True,
                fixed_noise=False,
            )
            task_losses.append(
                model.query_meta_loss(
                    posterior=posterior,
                    query_item_ids=query_items,
                    query_labels=query_labels,
                    fixed_noise=False,
                )
            )
        batch_loss = torch.stack(task_losses).mean()
        if not torch.isfinite(batch_loss):
            raise FloatingPointError(
                f"Non-finite meta loss at epoch={epoch}, batch={batch_start}."
            )
        batch_loss.backward()
        optimizer.step()
        total_loss += float(batch_loss.detach()) * len(task_losses)
        total_tasks += len(task_losses)
        batch_number = batch_start // TASK_BATCH_SIZE + 1
        if log_every > 0 and batch_number % log_every == 0:
            logging.info(
                "epoch=%d tasks=%d/%d meta_loss=%.6f",
                epoch,
                total_tasks,
                len(tasks),
                total_loss / total_tasks,
            )
    return total_loss / max(total_tasks, 1)


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def _json_safe_metrics(
    metrics: dict[str, float | list[dict[str, object]]]
) -> dict[str, object]:
    return {
        key: value
        for key, value in metrics.items()
        if key != "calibration_bins"
    }


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive.")
    if args.log_every < 0:
        raise ValueError("--log-every cannot be negative.")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    set_global_seed(MODEL_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    protocol = load_verified_validation_protocol(args.protocol_manifest)
    q_tensor, item_to_index, concept_to_index = build_q_union(
        train=protocol.train, q_matrix=protocol.q_matrix
    )
    train_support, train_query, train_split_audit = split_optimizer_train(
        protocol.train
    )
    train_tasks = build_student_tasks(
        support=train_support,
        query=train_query,
        item_to_index=item_to_index,
    )
    valid_tasks = build_student_tasks(
        support=protocol.valid_support,
        query=protocol.valid_query,
        item_to_index=item_to_index,
    )
    if not train_tasks or not valid_tasks:
        raise ValueError("Training and validation both require non-empty tasks.")

    device = _resolve_device(args.device)
    config = BetaNCDConfig(
        num_items=len(item_to_index),
        num_concepts=len(concept_to_index),
    )
    model = BetaNCDBaseline(config=config, q_matrix=q_tensor).to(device)
    optimizer = torch.optim.Adam(
        [
            {
                "params": model.meta_parameters(),
                "lr": META_LEARNING_RATE,
            },
            {
                "params": model.item_and_response_parameters(),
                "lr": ITEM_LEARNING_RATE,
            },
        ]
    )
    logging.info(
        "%s device=%s train_tasks=%d valid_tasks=%d items=%d concepts=%d",
        BETA_NCD_NAME,
        device,
        len(train_tasks),
        len(valid_tasks),
        config.num_items,
        config.num_concepts,
    )

    history: list[dict[str, object]] = []
    best_auc = float("-inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    started_at = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model=model,
            tasks=train_tasks,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            log_every=args.log_every,
        )
        valid_predictions, _ = evaluate_tasks(
            model=model, tasks=valid_tasks, device=device
        )
        valid_metrics = compute_metrics(
            valid_predictions["label"].to_numpy(),
            valid_predictions["prob"].to_numpy(),
        )
        history.append(
            {
                "epoch": epoch,
                "train_meta_loss": train_loss,
                **_json_safe_metrics(valid_metrics),
            }
        )
        logging.info(
            "epoch=%d train_meta_loss=%.6f valid_auc=%.6f",
            epoch,
            train_loss,
            valid_metrics["auc"],
        )
        if float(valid_metrics["auc"]) > best_auc:
            best_auc = float(valid_metrics["auc"])
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
    assert best_state is not None
    model.load_state_dict(best_state)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    checkpoint_payload = {
        "model_name": BETA_NCD_NAME,
        "model_state_dict": best_state,
        "config": asdict(config),
        "architecture_fingerprint": model.architecture_fingerprint,
        "best_epoch": best_epoch,
        "model_seed": MODEL_SEED,
        "protocol_split_seed": SPLIT_SEED,
        "manifest_path": str(protocol.manifest_path),
        "manifest_sha256": protocol.manifest_sha256,
        "manifest_files": protocol.manifest["files"],
        "train_internal_split": train_split_audit,
        "item_to_index": item_to_index,
        "concept_to_index": concept_to_index,
        "upstream": {
            "url": BETA_NCD_UPSTREAM_URL,
            "commit": BETA_NCD_UPSTREAM_COMMIT,
            "license": "MIT",
            "adaptation": "paper-aligned support/query outer-loss correction",
        },
        "heldout_student_parameters": False,
    }
    torch.save(checkpoint_payload, checkpoint_path)
    checkpoint_hash = sha256_file(checkpoint_path)

    final_predictions, mastery = evaluate_tasks(
        model=model, tasks=valid_tasks, device=device
    )
    final_predictions["model"] = BETA_NCD_NAME
    final_predictions["architecture_fingerprint"] = (
        model.architecture_fingerprint
    )
    final_predictions["checkpoint_sha256"] = checkpoint_hash
    prediction_path = output_dir / "valid_predictions.csv"
    final_predictions.to_csv(prediction_path, index=False)
    prediction_hash = sha256_file(prediction_path)
    mastery_path = output_dir / "valid_mastery.npz"
    np.savez_compressed(mastery_path, **mastery)
    mastery_hash = sha256_file(mastery_path)

    history_path = output_dir / "history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    final_metrics = compute_metrics(
        final_predictions["label"].to_numpy(),
        final_predictions["prob"].to_numpy(),
    )
    summary = {
        "status": "validation_only",
        "model": BETA_NCD_NAME,
        "architecture_fingerprint": model.architecture_fingerprint,
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "metrics": final_metrics,
        "elapsed_seconds": time.time() - started_at,
        "device": str(device),
        "fixed_recipe": {
            **config.architecture_payload(),
            "task_batch_size": TASK_BATCH_SIZE,
            "meta_learning_rate": META_LEARNING_RATE,
            "item_learning_rate": ITEM_LEARNING_RATE,
            "optimizer_train_partition": "atomic expected 80/20",
        },
        "protocol": {
            "manifest_path": str(protocol.manifest_path),
            "manifest_sha256": protocol.manifest_sha256,
            "split_seed": SPLIT_SEED,
            "verified_files": sorted(REQUIRED_MANIFEST_FILES),
            "parsed_files": sorted(LOADED_DATA_FILES),
            "test_files_integrity_checked_but_not_parsed": [
                "test_support.csv",
                "test_query.csv",
            ],
        },
        "train_internal_split": train_split_audit,
        "counts": {
            "train_tasks": len(train_tasks),
            "valid_tasks": len(valid_tasks),
            "valid_query_rows": len(final_predictions),
            "items": config.num_items,
            "concepts": config.num_concepts,
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
            "valid_predictions": str(prediction_path),
            "valid_predictions_sha256": prediction_hash,
            "valid_mastery": str(mastery_path),
            "valid_mastery_sha256": mastery_hash,
            "history": str(history_path),
            "history_sha256": sha256_file(history_path),
        },
        "provenance": {
            "upstream_url": BETA_NCD_UPSTREAM_URL,
            "upstream_commit": BETA_NCD_UPSTREAM_COMMIT,
            "license": "MIT",
            "notice": "docs/third_party/beta_ncd.md",
        },
        "heldout_student_parameters": False,
        "outer_loss_uses_query_only": True,
        "q_mapping": "per-exercise concept union",
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "auc": final_metrics["auc"],
                "best_epoch": best_epoch,
                "checkpoint_sha256": checkpoint_hash,
                "valid_predictions_sha256": prediction_hash,
                "summary": str(summary_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
