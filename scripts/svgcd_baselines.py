from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SVGCD_ROOT = Path("/home/xph/jwc/SVGCD")
DEFAULT_WORK_DIR = Path("/home/xph/jwc/research/local_data/svgcd_baselines")
DEFAULT_PYTHON = Path("/home/xph/anaconda3/envs/hyperbolic_cd/bin/python")
DEFAULT_SEED = 2024

DATASET_SPECS = {
    "assist09_standard": "/home/xph/jwc/research/decoupled_cd/data/assist_09_ordered",
    "assist09_holdout": "/tmp/assist09_holdout_seed2024",
    "assist17_standard": "/home/xph/jwc/research/local_data/cross_dataset_exp116_117/assist_17_source",
    "assist17_holdout": "/home/xph/jwc/research/local_data/cross_dataset_exp116_117/assist_17_holdout_seed2024",
    "nips34_standard": "/home/xph/jwc/research/local_data/cross_dataset_exp116_117/nips34_source",
    "nips34_holdout": "/home/xph/jwc/research/local_data/cross_dataset_exp116_117/nips34_holdout_seed2024",
    "junyi_standard": "/home/xph/jwc/research/local_data/cross_dataset_exp116_117/junyi_source",
    "junyi_holdout": "/home/xph/jwc/research/local_data/cross_dataset_exp116_117/junyi_holdout_seed2024",
    "junyi_long_standard": "/home/xph/jwc/research/local_data/cross_dataset_exp116_117/junyi_long_source",
    "junyi_long_holdout": "/home/xph/jwc/research/local_data/cross_dataset_exp116_117/junyi_long_holdout_seed2024",
    "junyi_sample_standard": "/home/xph/jwc/research/local_data/cross_dataset_exp116_117/junyi_sample_source",
    "junyi_sample_holdout": "/home/xph/jwc/research/local_data/cross_dataset_exp116_117/junyi_sample_holdout_seed2024",
}


@dataclass(frozen=True)
class PreparedDataset:
    key: str
    source_dir: str
    dataset_dir: str
    graph_dir: str
    rows: dict[str, int]
    num_users: int
    num_exercises: int
    num_concepts: int


@dataclass
class JobState:
    key: str
    gpu: int | None
    status: str
    started_at: float | None = None
    finished_at: float | None = None
    return_code: int | None = None
    log_path: str | None = None
    output_path: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SVGCD baselines on paper CD splits.")
    parser.add_argument("--svgcd-root", type=Path, default=DEFAULT_SVGCD_ROOT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    add_dataset_selection(prepare)
    prepare.add_argument("--overwrite", action="store_true")

    run = subparsers.add_parser("run")
    add_dataset_selection(run)
    run.add_argument("--gpus", default="0,1")
    run.add_argument("--max-parallel", type=int, default=2)
    run.add_argument("--min-free-gb", type=float, default=4.0)
    run.add_argument("--poll-seconds", type=float, default=20.0)
    run.add_argument("--epoch-num", type=int, default=100)
    run.add_argument("--batch-size", type=int, default=1024)
    run.add_argument("--eval-batch-size", type=int, default=512)
    run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run.add_argument("--force", action="store_true")
    run.add_argument("--dry-run", action="store_true")

    status = subparsers.add_parser("status")
    status.add_argument("--json", action="store_true")
    return parser.parse_args()


def add_dataset_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--datasets", default="all", help="Comma-separated dataset keys or 'all'.")


def selected_dataset_keys(value: str) -> list[str]:
    if value == "all":
        return list(DATASET_SPECS)
    keys = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(keys) - set(DATASET_SPECS))
    if unknown:
        raise ValueError(f"Unknown dataset keys: {', '.join(unknown)}")
    return keys


def split_concepts(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def load_frame(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    return frame


def build_maps(frames: dict[str, pd.DataFrame], q_matrix: pd.DataFrame) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    interactions = pd.concat(frames.values(), axis=0, ignore_index=True)
    users = sorted(str(value) for value in interactions["stu_id"].unique())
    exercises = {str(value) for value in interactions["exer_id"].unique()}
    exercises.update(str(value) for value in q_matrix["exer_id"].unique())
    concepts: set[str] = set()
    for value in q_matrix["cpt_seq"].tolist():
        concepts.update(split_concepts(value))
    for value in interactions["cpt_seq"].tolist():
        concepts.update(split_concepts(value))
    return (
        {raw_id: index for index, raw_id in enumerate(users)},
        {raw_id: index for index, raw_id in enumerate(sorted(exercises))},
        {raw_id: index for index, raw_id in enumerate(sorted(concepts))},
    )


def remap_cpt_seq(value: Any, concept_map: dict[str, int]) -> str:
    return ",".join(str(concept_map[item]) for item in split_concepts(value))


def prepare_dataset(args: argparse.Namespace, key: str) -> PreparedDataset:
    source_dir = Path(DATASET_SPECS[key])
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    frames = {
        split: load_frame(source_dir / f"{split}.csv", ("stu_id", "exer_id", "cpt_seq", "label"))
        for split in ("train", "valid", "test")
    }
    q_matrix = load_frame(source_dir / "Q_matrix.csv", ("exer_id", "cpt_seq"))
    user_map, exercise_map, concept_map = build_maps(frames, q_matrix)

    dataset_dir = args.work_dir / "datasets" / key
    graph_dir = args.work_dir / "graphs" / key
    if args.overwrite and dataset_dir.exists():
        for path in dataset_dir.glob("*.csv"):
            path.unlink()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    rows: dict[str, int] = {}
    for split, frame in frames.items():
        output = pd.DataFrame(
            {
                "stu_id": frame["stu_id"].astype(str).map(user_map),
                "exer_id": frame["exer_id"].astype(str).map(exercise_map),
                "cpt_seq": frame["cpt_seq"].map(lambda value: remap_cpt_seq(value, concept_map)),
                "label": frame["label"].astype(int),
            }
        )
        if output[["stu_id", "exer_id", "cpt_seq"]].isna().any().any():
            raise ValueError(f"{key}/{split} contains ids missing from the remap")
        output.to_csv(dataset_dir / f"{split}.csv", index=False)
        rows[split] = int(len(output))

    remapped_q = pd.DataFrame(
        {
            "exer_id": q_matrix["exer_id"].astype(str).map(exercise_map),
            "cpt_seq": q_matrix["cpt_seq"].map(lambda value: remap_cpt_seq(value, concept_map)),
        }
    ).drop_duplicates(subset=["exer_id"], keep="first")
    if remapped_q[["exer_id", "cpt_seq"]].isna().any().any():
        raise ValueError(f"{key}/Q_matrix.csv contains ids missing from the remap")
    remapped_q.sort_values("exer_id").to_csv(dataset_dir / "Q_matrix.csv", index=False)

    return PreparedDataset(
        key=key,
        source_dir=str(source_dir),
        dataset_dir=str(dataset_dir),
        graph_dir=str(graph_dir),
        rows=rows,
        num_users=len(user_map),
        num_exercises=len(exercise_map),
        num_concepts=len(concept_map),
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["datasets"]


def prepare(args: argparse.Namespace) -> None:
    args.work_dir.mkdir(parents=True, exist_ok=True)
    export_svgcd_head(args)
    prepared = [prepare_dataset(args, key) for key in selected_dataset_keys(args.datasets)]
    existing = {item["key"]: item for item in load_manifest(args.work_dir / "manifest.json")}
    for item in prepared:
        existing[item.key] = asdict(item)
        print(f"Prepared {item.key}: users={item.num_users}, exercises={item.num_exercises}, concepts={item.num_concepts}, rows={item.rows}")
    write_json(args.work_dir / "manifest.json", {"datasets": list(existing.values())})


def export_svgcd_head(args: argparse.Namespace) -> Path:
    target = args.work_dir / "svgcd_head_main.py"
    args.work_dir.mkdir(parents=True, exist_ok=True)
    content = subprocess.check_output(["git", "show", "HEAD:svgcd_only_main.py"], cwd=args.svgcd_root, text=True)
    content = content.replace(
        "    stu_num = torch.unique(torch.cat([train_df['stu_id'], test_df['stu_id'], val_df['stu_id']])).size(0)\n"
        "    exer_num = torch.unique(torch.cat([train_df['exer_id'], test_df['exer_id'], val_df['exer_id']])).size(0)\n"
        "    cpt_num = len(set(list(chain.from_iterable(Q_df['cpt_seq'].to_list()))))\n",
        "    stu_num = int(torch.max(torch.cat([train_df['stu_id'], test_df['stu_id'], val_df['stu_id']])).item()) + 1\n"
        "    exer_num = int(torch.max(torch.cat([train_df['exer_id'], test_df['exer_id'], val_df['exer_id']])).item()) + 1\n"
        "    cpt_num = max(chain.from_iterable(Q_df['cpt_seq'].to_list())) + 1\n",
    )
    target.write_text(content, encoding="utf-8")
    return target


def parse_gpus(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def gpu_free_gb() -> dict[int, float]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
        text=True,
    )
    free: dict[int, float] = {}
    for line in output.splitlines():
        index_raw, used_raw, total_raw = [item.strip() for item in line.split(",")]
        free[int(index_raw)] = (int(total_raw) - int(used_raw)) / 1024
    return free


def parse_metrics(log_text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, value in re.findall(r"\bTEST_(AUC|ACC|Precision|Recall|F1|RMSE):\s*([0-9.]+)", log_text):
        metrics[name.lower()] = float(value)
    return metrics


def run(args: argparse.Namespace) -> None:
    manifest = {item["key"]: item for item in load_manifest(args.work_dir / "manifest.json")}
    if not manifest:
        raise ValueError(f"No manifest found in {args.work_dir}; run prepare first")
    script = export_svgcd_head(args)
    args.work_dir.joinpath("logs").mkdir(parents=True, exist_ok=True)
    args.work_dir.joinpath("job_outputs").mkdir(parents=True, exist_ok=True)

    queue = [key for key in selected_dataset_keys(args.datasets)]
    active: dict[str, tuple[subprocess.Popen[str], JobState]] = {}
    states: dict[str, JobState] = {}
    gpus = parse_gpus(args.gpus)

    while queue or active:
        finished: list[str] = []
        for key, (process, state) in active.items():
            if process.poll() is None:
                continue
            state.finished_at = time.time()
            state.return_code = process.returncode
            state.status = "completed" if process.returncode == 0 else "failed"
            log_text = Path(state.log_path or "").read_text(encoding="utf-8", errors="replace")
            output = {
                "dataset_key": key,
                "status": state.status,
                "return_code": state.return_code,
                "log_path": state.log_path,
                "metrics": parse_metrics(log_text),
                "epoch_num": args.epoch_num,
                "batch_size": args.batch_size,
                "eval_batch_size": args.eval_batch_size,
                "seed": args.seed,
            }
            output_path = args.work_dir / "job_outputs" / f"{key}__SVGCD.json"
            write_json(output_path, output)
            state.output_path = str(output_path)
            states[key] = state
            finished.append(key)
            print(f"{state.status}: {key} -> {output_path}")
        for key in finished:
            del active[key]

        free = gpu_free_gb()
        used_gpus = {state.gpu for _, state in active.values()}
        available = [gpu for gpu in gpus if gpu not in used_gpus and free.get(gpu, 0.0) >= args.min_free_gb]
        while queue and available and len(active) < args.max_parallel:
            key = queue.pop(0)
            output_path = args.work_dir / "job_outputs" / f"{key}__SVGCD.json"
            if output_path.exists() and not args.force:
                print(f"skip existing: {key}")
                continue
            gpu = available.pop(0)
            state = launch_job(args, script, manifest[key], gpu)
            states[key] = state
            if args.dry_run:
                print("dry-run:", state.log_path)
            else:
                active[key] = (state_processes.pop(state.key), state)
        if args.dry_run:
            break
        write_status(args.work_dir, states)
        if queue or active:
            time.sleep(args.poll_seconds)
    write_status(args.work_dir, states)


state_processes: dict[str, subprocess.Popen[str]] = {}


def launch_job(args: argparse.Namespace, script: Path, dataset: dict[str, Any], gpu: int) -> JobState:
    key = dataset["key"]
    log_path = args.work_dir / "logs" / f"{key}__SVGCD.log"
    command = [
        str(args.python),
        str(script),
        "--dataset",
        key,
        "--train_path",
        str(Path(dataset["dataset_dir"]) / "train.csv"),
        "--val_path",
        str(Path(dataset["dataset_dir"]) / "valid.csv"),
        "--test_path",
        str(Path(dataset["dataset_dir"]) / "test.csv"),
        "--Q_matrix_path",
        str(Path(dataset["dataset_dir"]) / "Q_matrix.csv"),
        "--graph_path",
        dataset["graph_dir"],
        "--device",
        f"cuda:{gpu}",
        "--epoch_num",
        str(args.epoch_num),
        "--batch_size",
        str(args.batch_size),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--seed",
        str(args.seed),
    ]
    state = JobState(key=key, gpu=gpu, status="running", started_at=time.time(), log_path=str(log_path))
    if args.dry_run:
        state.status = "dry_run"
        state.log_path = " ".join(command)
        return state
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("$ " + " ".join(command) + "\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=args.svgcd_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    state_processes[key] = process
    print(f"started: {key} on cuda:{gpu} pid={process.pid}")
    return state


def write_status(work_dir: Path, states: dict[str, JobState]) -> None:
    write_json(work_dir / "status.json", {"jobs": [asdict(state) for state in states.values()]})


def print_status(args: argparse.Namespace) -> None:
    path = args.work_dir / "status.json"
    if not path.exists():
        print("No status.json")
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    counts: dict[str, int] = {}
    for job in payload["jobs"]:
        counts[job["status"]] = counts.get(job["status"], 0) + 1
    for status_name, count in sorted(counts.items()):
        print(f"{status_name}: {count}")


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        run(args)
    elif args.command == "status":
        print_status(args)
    else:
        raise ValueError(args.command)
    return 0


if __name__ == "__main__":
    sys.exit(main())
