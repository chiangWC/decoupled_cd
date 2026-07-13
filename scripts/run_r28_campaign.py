from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import parse_gpu_ids, write_json


PEAK_MEMORY_MB = {
    # Full validation materializes target-conditioned concept states.  The
    # observed MOO peak is substantially higher than the training-only smoke
    # peak, so these reservations intentionally permit one full evaluator per
    # 24 GiB card while still allowing unrelated small processes to coexist.
    "relational": 22000,
    "capacity_mlp": 22000,
    "direct_prior": 17000,
    "partial_vae": 22000,
    "difficulty_set": 22000,
    "difficulty_capacity": 22000,
    "poe_ability": 22000,
    "poe_capacity": 22000,
    "hierarchical_bayes": 22000,
    "hierarchical_capacity": 22000,
    "cohort_conditioned": 22000,
    "cohort_capacity": 22000,
    "bipolar_prototype": 22000,
    "bipolar_capacity": 22000,
}
DEFAULT_COMPLETERS = ("relational", "direct_prior", "capacity_mlp")


@dataclass
class CampaignTask:
    dataset: str
    dataset_variant: str
    state_completer: str
    completion_objective: str
    output_dir: str
    status: str = "pending"
    gpu: int | None = None
    returncode: int | None = None
    log_path: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resource-aware r28 validation campaign.")
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--dataset-variant", action="append", choices=["standard", "holdout"])
    parser.add_argument(
        "--state-completer",
        action="append",
        choices=sorted(PEAK_MEMORY_MB),
    )
    parser.add_argument("--completion-objective", default="none", choices=["none", "masked_reconstruction"])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", default="results/r28/campaign")
    parser.add_argument("--gpus", default=None, help="Physical GPU ids; default is every visible nvidia-smi GPU.")
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def gpu_stats() -> dict[int, dict[str, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True)
    except (OSError, subprocess.CalledProcessError):
        return {}
    stats: dict[int, dict[str, int]] = {}
    for line in output.strip().splitlines():
        index, total, used, free, utilization = [int(part.strip()) for part in line.split(",")]
        stats[index] = {
            "memory_total_mb": total,
            "memory_used_mb": used,
            "memory_free_mb": free,
            "utilization_percent": utilization,
        }
    return stats


def build_tasks(args: argparse.Namespace) -> list[CampaignTask]:
    dataset_variants = args.dataset_variant or ["standard", "holdout"]
    completers = args.state_completer or list(DEFAULT_COMPLETERS)
    tasks: list[CampaignTask] = []
    for dataset in args.dataset:
        for completer in completers:
            for dataset_variant in dataset_variants:
                name = f"{dataset}__{dataset_variant}__{completer}__{args.completion_objective}"
                tasks.append(
                    CampaignTask(
                        dataset=dataset,
                        dataset_variant=dataset_variant,
                        state_completer=completer,
                        completion_objective=args.completion_objective,
                        output_dir=str(Path(args.output_root) / name),
                    )
                )
    return tasks


def choose_gpu(
    *,
    candidates: list[int],
    expected_peak_mb: int,
    reservations: dict[int, int],
) -> int | None:
    eligible: list[tuple[int, int, int]] = []
    for gpu, stat in gpu_stats().items():
        if gpu not in candidates:
            continue
        effective_free = stat["memory_free_mb"] - reservations.get(gpu, 0)
        if effective_free < expected_peak_mb:
            continue
        score = effective_free - 24 * stat["utilization_percent"]
        eligible.append((score, effective_free, gpu))
    return max(eligible)[2] if eligible else None


def launch(
    task: CampaignTask,
    *,
    args: argparse.Namespace,
    gpu: int,
) -> tuple[subprocess.Popen[str], Any]:
    output_dir = Path(task.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "runner.log"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train_r28.py"),
        "--dataset",
        task.dataset,
        "--dataset-variant",
        task.dataset_variant,
        "--state-completer",
        task.state_completer,
        "--completion-objective",
        task.completion_objective,
        "--evaluation-stage",
        "validation",
        "--data-root",
        args.data_root,
        "--output-dir",
        task.output_dir,
        "--device",
        "cuda:0",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    handle = log_path.open("w", encoding="utf-8")
    handle.write("$ " + " ".join(command) + "\n")
    handle.flush()
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    task.status = "running"
    task.gpu = gpu
    task.log_path = str(log_path)
    return process, handle


def main() -> None:
    args = parse_args()
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be positive.")
    if not 1.0 <= args.poll_seconds <= 60.0:
        raise ValueError("--poll-seconds must be in [1, 60].")
    available = sorted(gpu_stats())
    candidates = parse_gpu_ids(args.gpus) if args.gpus is not None else available
    if not candidates:
        raise RuntimeError("No physical GPUs were discovered or selected.")

    tasks = build_tasks(args)
    for task in tasks:
        if (Path(task.output_dir) / "summary.json").exists():
            task.status = "completed"
    manifest_path = Path(args.output_root) / "campaign_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        write_json({"tasks": [asdict(task) for task in tasks]}, manifest_path)
        print(json.dumps([asdict(task) for task in tasks], indent=2))
        return

    pending = [task for task in tasks if task.status == "pending"]
    running: dict[int, tuple[CampaignTask, subprocess.Popen[str], Any, int]] = {}
    while pending or running:
        for process_id, (task, process, handle, _) in list(running.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            task.returncode = returncode
            task.status = "completed" if returncode == 0 else "failed"
            del running[process_id]
            print(f"finished {task.output_dir}: {task.status}", flush=True)

        launched = False
        reservations: dict[int, int] = {}
        for running_task, _, _, expected_peak in running.values():
            if running_task.gpu is not None:
                reservations[running_task.gpu] = (
                    reservations.get(running_task.gpu, 0) + expected_peak
                )
        while pending and len(running) < args.max_parallel:
            selected: tuple[int, CampaignTask, int, int] | None = None
            for pending_index, candidate_task in enumerate(pending):
                candidate_peak = PEAK_MEMORY_MB[candidate_task.state_completer]
                candidate_gpu = choose_gpu(
                    candidates=candidates,
                    expected_peak_mb=candidate_peak,
                    reservations=reservations,
                )
                if candidate_gpu is not None:
                    selected = (
                        pending_index,
                        candidate_task,
                        candidate_peak,
                        candidate_gpu,
                    )
                    break
            if selected is None:
                break
            pending_index, task, expected_peak, gpu = selected
            pending.pop(pending_index)
            process, handle = launch(task, args=args, gpu=gpu)
            reservations[gpu] = reservations.get(gpu, 0) + expected_peak
            running[process.pid] = (task, process, handle, expected_peak)
            launched = True
            print(f"launched {task.output_dir} on physical GPU {gpu}", flush=True)

        write_json(
            {
                "schema_version": 1,
                "gpu_policy": "free-memory/utilization aware; occupied GPUs remain eligible",
                "oom_policy": "fail without mutating the registered recipe",
                "tasks": [asdict(task) for task in tasks],
            },
            manifest_path,
        )
        if pending or running:
            if not launched and not running:
                print("No GPU currently has the declared headroom; CPU audits can continue.", flush=True)
            time.sleep(args.poll_seconds)

    if any(task.status == "failed" for task in tasks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
