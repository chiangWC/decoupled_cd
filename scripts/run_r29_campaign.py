from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_r28_campaign as scheduler


R29_PEAK_MEMORY_MB = {
    "meta_implicit": 22000,
    "capacity_control": 22000,
    "direct_prior": 17000,
    "marginal_anchor": 22000,
}


def launch(
    task: scheduler.CampaignTask,
    *,
    args,
    gpu: int,
) -> tuple[subprocess.Popen[str], Any]:
    output_dir = Path(task.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "runner.log"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train_r29.py"),
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
    scheduler.PEAK_MEMORY_MB = R29_PEAK_MEMORY_MB
    scheduler.DEFAULT_COMPLETERS = ("meta_implicit", "direct_prior", "capacity_control")
    scheduler.launch = launch
    scheduler.main()


if __name__ == "__main__":
    main()
