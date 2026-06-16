from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PYEDMINE_ROOT = Path("/home/xph/jwc/pyedmine")
DEFAULT_SETTING_NAME = "paper_cd_baselines"
DEFAULT_MANIFEST_NAME = "manifest.json"
DEFAULT_SEED = 2024
DEFAULT_MAX_EPOCH = 50

MODEL_SCRIPTS = {
    "DINA": "dina.py",
    "IRT": "irt.py",
    "MIRT": "mirt.py",
    "NCD": "ncd.py",
    "RCD": "rcd.py",
    "HierCDF": "hier_cdf.py",
    "HyperCD": "hyper_cd.py",
}

DATASET_SPECS = {
    "assist09_standard": {
        "dataset": "assist09",
        "split": "standard",
        "source_dir": "/home/xph/jwc/research/decoupled_cd/data/assist_09_ordered",
    },
    "assist09_holdout": {
        "dataset": "assist09",
        "split": "holdout",
        "source_dir": "/tmp/assist09_holdout_seed2024",
    },
    "assist17_standard": {
        "dataset": "assist17",
        "split": "standard",
        "source_dir": "/home/xph/jwc/research/local_data/cross_dataset_exp116_117/assist_17_source",
    },
    "assist17_holdout": {
        "dataset": "assist17",
        "split": "holdout",
        "source_dir": "/home/xph/jwc/research/local_data/cross_dataset_exp116_117/assist_17_holdout_seed2024",
    },
    "nips34_standard": {
        "dataset": "nips34",
        "split": "standard",
        "source_dir": "/home/xph/jwc/research/local_data/cross_dataset_exp116_117/nips34_source",
    },
    "nips34_holdout": {
        "dataset": "nips34",
        "split": "holdout",
        "source_dir": "/home/xph/jwc/research/local_data/cross_dataset_exp116_117/nips34_holdout_seed2024",
    },
    "junyi_standard": {
        "dataset": "junyi",
        "split": "standard",
        "source_dir": "/home/xph/jwc/research/local_data/cross_dataset_exp116_117/junyi_source",
    },
    "junyi_holdout": {
        "dataset": "junyi",
        "split": "holdout",
        "source_dir": "/home/xph/jwc/research/local_data/cross_dataset_exp116_117/junyi_holdout_seed2024",
    },
    "junyi_long_standard": {
        "dataset": "junyi_long",
        "split": "standard",
        "source_dir": "/home/xph/jwc/research/local_data/cross_dataset_exp116_117/junyi_long_source",
    },
    "junyi_long_holdout": {
        "dataset": "junyi_long",
        "split": "holdout",
        "source_dir": "/home/xph/jwc/research/local_data/cross_dataset_exp116_117/junyi_long_holdout_seed2024",
    },
    "junyi_sample_standard": {
        "dataset": "junyi_sample",
        "split": "standard",
        "source_dir": "/home/xph/jwc/research/local_data/cross_dataset_exp116_117/junyi_sample_source",
    },
    "junyi_sample_holdout": {
        "dataset": "junyi_sample",
        "split": "holdout",
        "source_dir": "/home/xph/jwc/research/local_data/cross_dataset_exp116_117/junyi_sample_holdout_seed2024",
    },
}


@dataclass(frozen=True)
class PreparedDataset:
    key: str
    dataset: str
    split: str
    source_dir: str
    pyedmine_dataset_name: str
    setting_name: str
    train_file_name: str
    valid_file_name: str
    test_file_name: str
    num_users: int
    num_questions: int
    num_concepts: int
    rows: dict[str, int]


@dataclass
class JobState:
    key: str
    model: str
    dataset_key: str
    dataset: str
    split: str
    gpu: int | None
    status: str
    started_at: float | None = None
    finished_at: float | None = None
    return_code: int | None = None
    log_path: str | None = None
    model_dir_name: str | None = None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and run PyEdmine CD baselines for paper splits.")
    parser.add_argument("--pyedmine-root", type=Path, default=DEFAULT_PYEDMINE_ROOT)
    parser.add_argument("--setting-name", default=DEFAULT_SETTING_NAME)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/home/xph/jwc/research/local_data/pyedmine_cd_baselines"),
        help="Directory for manifests, logs, and scheduler status.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Convert all selected datasets into PyEdmine artifacts.")
    add_selection_args(prepare)
    prepare.add_argument("--overwrite", action="store_true")
    prepare.add_argument(
        "--dataset-prefix",
        default="paper",
        help="Prefix for generated PyEdmine dataset names. Use a new prefix to avoid overwriting existing preprocessed data.",
    )
    prepare.add_argument(
        "--kt-source",
        choices=("full", "train"),
        default="full",
        help="Rows used for the KT-format data.txt consumed by RCD dependency graph construction.",
    )

    run = subparsers.add_parser("run", help="Run selected PyEdmine CD baseline jobs.")
    add_selection_args(run)
    run.add_argument("--models", default="all", help="Comma-separated model names or 'all'.")
    run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run.add_argument("--max-epoch", type=int, default=DEFAULT_MAX_EPOCH)
    run.add_argument("--max-parallel", type=int, default=2)
    run.add_argument("--min-free-gb", type=float, default=4.0)
    run.add_argument("--poll-seconds", type=float, default=15.0)
    run.add_argument("--force", action="store_true")
    run.add_argument("--cpu", action="store_true", help="Run training/evaluation in CPU mode.")
    run.add_argument("--train-batch-size", type=int)
    run.add_argument("--evaluate-batch-size", type=int)
    run.add_argument("--dry-run", action="store_true")

    status = subparsers.add_parser("status", help="Print scheduler status summary.")
    status.add_argument("--json", action="store_true")
    return parser.parse_args()


def add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated dataset keys or 'all'. Keys: " + ",".join(DATASET_SPECS),
    )


def selected_dataset_keys(value: str) -> list[str]:
    if value == "all":
        return list(DATASET_SPECS)
    keys = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(keys) - set(DATASET_SPECS))
    if unknown:
        raise ValueError(f"Unknown dataset keys: {', '.join(unknown)}")
    return keys


def selected_models(value: str) -> list[str]:
    if value == "all":
        return list(MODEL_SCRIPTS)
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(names) - set(MODEL_SCRIPTS))
    if unknown:
        raise ValueError(f"Unknown models: {', '.join(unknown)}")
    return names


def parse_concepts(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return [token.strip() for token in str(value).split(",") if token.strip()]


def load_frame(path: Path, required_columns: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    return frame


def build_maps(frames: list[pd.DataFrame], q_matrix: pd.DataFrame) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    interactions = pd.concat(frames, axis=0, ignore_index=True)
    user_ids = sorted(str(value) for value in interactions["stu_id"].unique())
    question_ids = {str(value) for value in interactions["exer_id"].unique()}
    question_ids.update(str(value) for value in q_matrix["exer_id"].unique())

    concept_ids: set[str] = set()
    for raw_value in q_matrix["cpt_seq"].tolist():
        concept_ids.update(parse_concepts(raw_value))
    for raw_value in interactions["cpt_seq"].tolist() if "cpt_seq" in interactions.columns else []:
        concept_ids.update(parse_concepts(raw_value))

    return (
        {raw_id: index for index, raw_id in enumerate(user_ids)},
        {raw_id: index for index, raw_id in enumerate(sorted(question_ids))},
        {raw_id: index for index, raw_id in enumerate(sorted(concept_ids))},
    )


def write_cd_file(frame: pd.DataFrame, path: Path, user_map: dict[str, int], question_map: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "question_id", "correctness"])
        for row in frame.itertuples(index=False):
            writer.writerow([user_map[str(row.stu_id)], question_map[str(row.exer_id)], int(row.label)])


def write_kt_file(data_frame: pd.DataFrame, path: Path, user_map: dict[str, int], question_map: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("user_id,seq_len;question_seq,correctness_seq\n")
        for raw_user_id, user_frame in data_frame.groupby("stu_id", sort=False):
            user_id = user_map[str(raw_user_id)]
            question_seq = [str(question_map[str(value)]) for value in user_frame["exer_id"].tolist()]
            correctness_seq = [str(int(value)) for value in user_frame["label"].tolist()]
            handle.write(f"{user_id}\n")
            handle.write(f"{len(question_seq)}\n")
            handle.write(",".join(question_seq) + "\n")
            handle.write(",".join(correctness_seq) + "\n")


def build_q_table(
    q_matrix: pd.DataFrame,
    question_map: dict[str, int],
    concept_map: dict[str, int],
) -> np.ndarray:
    table = np.zeros((len(question_map), len(concept_map)), dtype=np.int64)
    for row in q_matrix.itertuples(index=False):
        question_id = str(row.exer_id)
        if question_id not in question_map:
            continue
        question_index = question_map[question_id]
        for concept_id in parse_concepts(row.cpt_seq):
            concept_index = concept_map.get(concept_id)
            if concept_index is not None:
                table[question_index, concept_index] = 1
    return table


def pyedmine_meta_root(pyedmine_root: Path) -> Path:
    return pyedmine_root / "meta_data"


def setting_dir(pyedmine_root: Path, setting_name: str) -> Path:
    return pyedmine_meta_root(pyedmine_root) / "dataset" / "settings" / setting_name


def preprocessed_dir(pyedmine_root: Path, dataset_name: str) -> Path:
    return pyedmine_meta_root(pyedmine_root) / "dataset" / "dataset_preprocessed" / dataset_name


def prepare_datasets(args: argparse.Namespace) -> None:
    pyedmine_root = args.pyedmine_root
    target_setting_dir = setting_dir(pyedmine_root, args.setting_name)
    target_setting_dir.mkdir(parents=True, exist_ok=True)
    setting_path = target_setting_dir / "setting.json"
    if args.overwrite or not setting_path.exists():
        write_json(setting_path, {"name": args.setting_name, "source": "decoupled_cd paper CD baselines"})

    prepared: list[PreparedDataset] = []
    for key in selected_dataset_keys(args.datasets):
        prepared.append(prepare_one_dataset(args, key))

    args.work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.work_dir / DEFAULT_MANIFEST_NAME
    existing = load_manifest(manifest_path)
    merged = {item["key"]: item for item in existing}
    for item in prepared:
        merged[item.key] = asdict(item)
    write_json(manifest_path, {"datasets": list(merged.values())})
    print(f"Wrote manifest: {manifest_path}")


def prepare_one_dataset(args: argparse.Namespace, key: str) -> PreparedDataset:
    spec = DATASET_SPECS[key]
    source_dir = Path(spec["source_dir"])
    if not source_dir.exists():
        raise FileNotFoundError(f"{source_dir} does not exist")

    frames = {
        split: load_frame(source_dir / f"{split}.csv", ("stu_id", "exer_id", "label"))
        for split in ("train", "valid", "test")
    }
    full_data_frame = (
        load_frame(source_dir / "data.csv", ("stu_id", "exer_id", "label"))
        if (source_dir / "data.csv").exists()
        else pd.concat(frames.values(), axis=0, ignore_index=True)
    )
    kt_frame = frames["train"] if args.kt_source == "train" else full_data_frame
    q_matrix = load_frame(source_dir / "Q_matrix.csv", ("exer_id", "cpt_seq"))
    user_map, question_map, concept_map = build_maps([full_data_frame], q_matrix)

    dataset_name = f"{args.dataset_prefix}_{key}"
    train_file_name = f"{dataset_name}_train.txt"
    valid_file_name = f"{dataset_name}_valid.txt"
    test_file_name = f"{dataset_name}_test.txt"
    target_setting_dir = setting_dir(args.pyedmine_root, args.setting_name)
    target_preprocessed_dir = preprocessed_dir(args.pyedmine_root, dataset_name)

    if args.overwrite and target_preprocessed_dir.exists():
        shutil.rmtree(target_preprocessed_dir)
    target_preprocessed_dir.mkdir(parents=True, exist_ok=True)

    for split, frame in frames.items():
        file_name = {"train": train_file_name, "valid": valid_file_name, "test": test_file_name}[split]
        write_cd_file(frame, target_setting_dir / file_name, user_map, question_map)

    write_kt_file(kt_frame, target_preprocessed_dir / "data.txt", user_map, question_map)
    np.save(target_preprocessed_dir / "Q_table.npy", build_q_table(q_matrix, question_map, concept_map))
    write_json(
        target_preprocessed_dir / "statics_preprocessed.json",
        {
            "num_user": len(user_map),
            "num_question": len(question_map),
            "num_concept": len(concept_map),
        },
    )
    write_id_map(target_preprocessed_dir / "user_id_map.csv", "raw_user_id", user_map)
    write_id_map(target_preprocessed_dir / "question_id_map.csv", "raw_question_id", question_map)
    write_id_map(target_preprocessed_dir / "concept_id_map.csv", "raw_concept_id", concept_map)

    with (target_setting_dir / f"{dataset_name}_statics.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"num of user: {len(user_map)}\n")

    prepared = PreparedDataset(
        key=key,
        dataset=str(spec["dataset"]),
        split=str(spec["split"]),
        source_dir=str(source_dir),
        pyedmine_dataset_name=dataset_name,
        setting_name=args.setting_name,
        train_file_name=train_file_name,
        valid_file_name=valid_file_name,
        test_file_name=test_file_name,
        num_users=len(user_map),
        num_questions=len(question_map),
        num_concepts=len(concept_map),
        rows={split: int(len(frame)) for split, frame in frames.items()},
    )
    print(
        f"Prepared {key}: users={prepared.num_users}, questions={prepared.num_questions}, "
        f"concepts={prepared.num_concepts}, rows={prepared.rows}"
    )
    return prepared


def write_id_map(path: Path, raw_column: str, mapping: dict[str, int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([raw_column, "pyedmine_id"])
        for raw_id, mapped_id in sorted(mapping.items(), key=lambda item: item[1]):
            writer.writerow([raw_id, mapped_id])


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return list(payload.get("datasets", []))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_jobs(args: argparse.Namespace) -> None:
    manifest_path = args.work_dir / DEFAULT_MANIFEST_NAME
    datasets = {
        item["key"]: item
        for item in load_manifest(manifest_path)
        if item["key"] in selected_dataset_keys(args.datasets)
    }
    missing = sorted(set(selected_dataset_keys(args.datasets)) - set(datasets))
    if missing:
        raise ValueError(f"Datasets not prepared yet: {', '.join(missing)}. Run prepare first.")

    models = selected_models(args.models)
    jobs = [
        make_job_state(model=model, dataset=dataset)
        for dataset in datasets.values()
        for model in models
    ]

    status_path = args.work_dir / "status.jsonl"
    completed_keys = completed_job_keys(status_path) if not args.force else set()
    jobs = [job for job in jobs if job.key not in completed_keys]
    print(f"Pending jobs: {len(jobs)}")
    if args.dry_run:
        for job in jobs:
            print(job.key)
        return

    args.work_dir.mkdir(parents=True, exist_ok=True)
    run_scheduler(args, jobs, status_path)


def make_job_state(model: str, dataset: dict[str, Any]) -> JobState:
    key = f"{dataset['key']}__{model}"
    return JobState(
        key=key,
        model=model,
        dataset_key=dataset["key"],
        dataset=dataset["dataset"],
        split=dataset["split"],
        gpu=None,
        status="pending",
    )


def completed_job_keys(status_path: Path) -> set[str]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(status_path):
        latest[row["key"]] = row
    return {
        key
        for key, row in latest.items()
        if row.get("status") == "completed"
    }


def run_scheduler(args: argparse.Namespace, jobs: list[JobState], status_path: Path) -> None:
    pending = jobs[:]
    running: dict[subprocess.Popen[str], JobState] = {}

    while pending or running:
        for process, job in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            job.return_code = return_code
            job.finished_at = time.time()
            job.status = "completed" if return_code == 0 else "failed"
            append_jsonl(status_path, asdict(job))
            print(f"{job.status}: {job.key} rc={return_code}")
            del running[process]

        while pending and len(running) < args.max_parallel:
            gpu = None if args.cpu else choose_gpu(args.min_free_gb, used_gpus={job.gpu for job in running.values()})
            if gpu is None and not args.cpu:
                break
            job = pending.pop(0)
            job.gpu = gpu
            job.started_at = time.time()
            job.status = "running"
            process = launch_job(args, job)
            running[process] = job
            append_jsonl(status_path, asdict(job))
            print(f"started: {job.key} gpu={gpu} log={job.log_path}")

        if pending or running:
            time.sleep(args.poll_seconds)


def choose_gpu(min_free_gb: float, used_gpus: set[int | None]) -> int | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    candidates: list[tuple[float, int]] = []
    for line in output.strip().splitlines():
        index_text, used_text, total_text = [item.strip() for item in line.split(",")]
        index = int(index_text)
        if index in used_gpus:
            continue
        free_gb = (float(total_text) - float(used_text)) / 1024.0
        if free_gb >= min_free_gb:
            candidates.append((free_gb, index))
    if not candidates:
        return None
    return sorted(candidates, reverse=True)[0][1]


def launch_job(args: argparse.Namespace, job: JobState) -> subprocess.Popen[str]:
    logs_dir = args.work_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{job.key}.log"
    job.log_path = str(log_path)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_run_one",
        "--pyedmine-root",
        str(args.pyedmine_root),
        "--work-dir",
        str(args.work_dir),
        "--setting-name",
        args.setting_name,
        "--dataset-key",
        job.dataset_key,
        "--model",
        job.model,
        "--seed",
        str(args.seed),
        "--max-epoch",
        str(args.max_epoch),
    ]
    if args.train_batch_size is not None:
        command.extend(["--train-batch-size", str(args.train_batch_size)])
    if args.evaluate_batch_size is not None:
        command.extend(["--evaluate-batch-size", str(args.evaluate_batch_size)])
    if args.cpu:
        command.append("--cpu")

    env = os.environ.copy()
    if job.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("$ " + " ".join(command) + "\n")
        log_handle.flush()
        return subprocess.Popen(
            command,
            cwd=args.pyedmine_root,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )


def run_one_from_args(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyedmine-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--setting-name", required=True)
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SCRIPTS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-epoch", type=int, required=True)
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--evaluate-batch-size", type=int)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)

    manifest = {item["key"]: item for item in load_manifest(args.work_dir / DEFAULT_MANIFEST_NAME)}
    dataset = manifest[args.dataset_key]
    run_one(args, dataset)
    return 0


def run_one(args: argparse.Namespace, dataset: dict[str, Any]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.pyedmine_root)
    run_preprocessing(args, dataset, env)

    script = args.pyedmine_root / "examples" / "cognitive_diagnosis" / "train" / MODEL_SCRIPTS[args.model]
    train_stem = Path(dataset["train_file_name"]).stem
    model_prefix = f"{args.model}@@{args.setting_name}@@{train_stem}@@seed_{args.seed}@@"
    model_root = args.pyedmine_root / "saved_models"
    before = matching_model_dirs(model_root, model_prefix)
    train_cmd = [
        python_for_current_env(),
        str(script),
        "--setting_name",
        args.setting_name,
        "--dataset_name",
        dataset["pyedmine_dataset_name"],
        "--train_file_name",
        dataset["train_file_name"],
        "--valid_file_name",
        dataset["valid_file_name"],
        "--max_epoch",
        str(args.max_epoch),
        "--seed",
        str(args.seed),
        "--save_model",
        "True",
        "--use_wandb",
        "False",
    ]
    if args.cpu:
        train_cmd.extend(["--use_cpu", "True"])
    if args.train_batch_size is not None:
        train_cmd.extend(["--train_batch_size", str(args.train_batch_size)])
    if args.evaluate_batch_size is not None:
        train_cmd.extend(["--evaluate_batch_size", str(args.evaluate_batch_size)])
    run_command(train_cmd, args.pyedmine_root, env)

    after = matching_model_dirs(model_root, model_prefix)
    new_dirs = [path for path in after if path not in before]
    model_dir = sorted(new_dirs or after, key=lambda path: path.stat().st_mtime)[-1]
    eval_cmd = [
        python_for_current_env(),
        str(args.pyedmine_root / "examples" / "cognitive_diagnosis" / "evaluate" / "dlcd.py"),
        "--model_dir_name",
        model_dir.name,
        "--dataset_name",
        dataset["pyedmine_dataset_name"],
        "--test_file_name",
        dataset["test_file_name"],
        "--user_cold_start",
        "-1",
        "--question_cold_start",
        "-1",
        "--save_log",
        "True",
    ]
    if args.evaluate_batch_size is not None:
        eval_cmd.extend(["--evaluate_batch_size", str(args.evaluate_batch_size)])
    eval_output = run_command_capture(eval_cmd, args.pyedmine_root, env)
    write_json(
        args.work_dir / "job_outputs" / f"{dataset['key']}__{args.model}.json",
        {
            "dataset_key": dataset["key"],
            "model": args.model,
            "model_dir_name": model_dir.name,
            "model_dir": str(model_dir),
            "setting_name": args.setting_name,
            "train_file_name": dataset["train_file_name"],
            "valid_file_name": dataset["valid_file_name"],
            "test_file_name": dataset["test_file_name"],
            "train_batch_size": args.train_batch_size,
            "evaluate_batch_size": args.evaluate_batch_size,
            "test_metrics": parse_eval_metrics(eval_output),
        },
    )


def python_for_current_env() -> str:
    return sys.executable


def matching_model_dirs(model_root: Path, prefix: str) -> list[Path]:
    if not model_root.exists():
        return []
    return [path for path in model_root.iterdir() if path.is_dir() and path.name.startswith(prefix)]


def run_preprocessing(args: argparse.Namespace, dataset: dict[str, Any], env: dict[str, str]) -> None:
    if args.model in {"RCD", "HierCDF"}:
        lock_path = args.work_dir / "locks" / f"{dataset['key']}.rcd_graph.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("w", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle, fcntl.LOCK_EX)
            run_rcd_preprocessing(args, dataset, env)
            fcntl.flock(lock_handle, fcntl.LOCK_UN)
    if args.model == "HierCDF":
        run_command(
            [
                python_for_current_env(),
                "examples/cognitive_diagnosis/hier_cdf/construct_graph_from_rcd.py",
                "--setting_name",
                args.setting_name,
                "--dataset_name",
                dataset["pyedmine_dataset_name"],
            ],
            args.pyedmine_root,
            env,
        )
    if args.model == "HyperCD":
        run_command(
            [
                python_for_current_env(),
                "examples/cognitive_diagnosis/hyper_cd/construct_hyper_graph.py",
                "--setting_name",
                args.setting_name,
                "--dataset_name",
                dataset["pyedmine_dataset_name"],
                "--train_file_name",
                dataset["train_file_name"],
            ],
            args.pyedmine_root,
            env,
        )


def run_rcd_preprocessing(args: argparse.Namespace, dataset: dict[str, Any], env: dict[str, str]) -> None:
    run_command(
        [
            python_for_current_env(),
            "examples/cognitive_diagnosis/rcd/process_edge.py",
            "--setting_name",
            args.setting_name,
            "--dataset_name",
            dataset["pyedmine_dataset_name"],
        ],
        args.pyedmine_root,
        env,
    )
    run_command(
        [
            python_for_current_env(),
            "examples/cognitive_diagnosis/rcd/build_k_e_graph.py",
            "--setting_name",
            args.setting_name,
            "--dataset_name",
            dataset["pyedmine_dataset_name"],
            "--train_file_name",
            dataset["train_file_name"],
        ],
        args.pyedmine_root,
        env,
    )
    run_command(
        [
            python_for_current_env(),
            "examples/cognitive_diagnosis/rcd/build_u_e_graph.py",
            "--setting_name",
            args.setting_name,
            "--dataset_name",
            dataset["pyedmine_dataset_name"],
            "--train_file_name",
            dataset["train_file_name"],
        ],
        args.pyedmine_root,
        env,
    )


def run_command(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def run_command_capture(command: list[str], cwd: Path, env: dict[str, str]) -> str:
    print("$ " + " ".join(command), flush=True)
    process = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(process.stdout, end="", flush=True)
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command, output=process.stdout)
    return process.stdout


def parse_eval_metrics(output: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, value in re.findall(r"\b(AUC|ACC|RMSE|MAE):\s*([0-9.]+)", output):
        metrics[name.lower()] = float(value)
    return metrics


def print_status(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.work_dir / "status.jsonl")
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        latest[row["key"]] = row
    counts = defaultdict(int)
    for row in latest.values():
        counts[row["status"]] += 1
    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")
    failed = [row for row in latest.values() if row["status"] == "failed"]
    if failed:
        print("\nFailed jobs:")
        for row in failed:
            print(f"  {row['key']} log={row.get('log_path')}")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "_run_one":
        return run_one_from_args(sys.argv[2:])

    args = parse_args()
    if args.command == "prepare":
        prepare_datasets(args)
        return 0
    if args.command == "run":
        run_jobs(args)
        return 0
    if args.command == "status":
        print_status(args)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
