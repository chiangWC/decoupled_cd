from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ATTEMPT_PATTERN = re.compile(r"^attempt-(\d+)$")
DEFAULT_CAPTURE_ENV = ("CONDA_DEFAULT_ENV", "CUDA_VISIBLE_DEVICES", "PYTHONPATH")


class CampaignError(RuntimeError):
    """Raised when campaign preflight checks fail."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one reproducible command in a new immutable campaign attempt directory."
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="Git worktree to verify and record (default: --cwd or current directory).",
    )
    parser.add_argument("--cwd", type=Path, help="Command working directory (default: current directory).")
    parser.add_argument(
        "--dataset-file",
        action="append",
        type=Path,
        default=[],
        help="Dataset file to hash; repeat for every campaign input.",
    )
    parser.add_argument(
        "--output-file",
        action="append",
        default=[],
        help="Output to hash after the command; relative paths are resolved in the attempt directory.",
    )
    parser.add_argument(
        "--capture-env",
        action="append",
        default=[],
        metavar="NAME",
        help="Additional environment variable to capture; repeat as needed.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--doa-seed", type=int, default=42)
    parser.add_argument("--min-responses", type=int, default=3)
    parser.add_argument("--split-seed", type=int, default=2024)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command argv after --.")
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    args.cwd = (args.cwd or Path.cwd()).expanduser().resolve()
    args.repo_root = (args.repo_root or args.cwd).expanduser().resolve()
    args.artifact_root = args.artifact_root.expanduser().resolve()
    args.dataset_file = [resolve_from(path, args.cwd) for path in args.dataset_file]
    return args


def resolve_from(path: Path, base: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = base / expanded
    return expanded.resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CampaignError(f"Dataset file does not exist or is not a regular file: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def fingerprint_output(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    if not path.is_file():
        return {"path": str(path), "exists": True, "type": "not_regular_file"}
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def run_git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise CampaignError(f"Git preflight failed in {repo_root}: {detail}")
    return completed.stdout.strip()


def collect_git_metadata(repo_root: Path) -> dict[str, Any]:
    head = run_git(repo_root, "rev-parse", "HEAD")
    status = run_git(repo_root, "status", "--porcelain", "--untracked-files=all")
    return {
        "repo_root": str(repo_root),
        "head": head,
        "clean": not bool(status),
        "status_porcelain": status,
    }


def collect_runtime_metadata() -> dict[str, Any]:
    python_metadata = {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "platform": platform.platform(),
    }
    torch_metadata: dict[str, Any]
    cuda_metadata: dict[str, Any]
    try:
        import torch

        torch_metadata = {"available": True, "version": torch.__version__}
        cuda_metadata = {
            "build_version": torch.version.cuda,
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }
    except Exception as exc:  # pragma: no cover - depends on the remote environment
        torch_metadata = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
        cuda_metadata = {"available": False, "error": "Torch CUDA metadata unavailable"}
    return {
        "python": python_metadata,
        "torch": torch_metadata,
        "cuda": cuda_metadata,
        "gpu": collect_gpu_metadata(),
    }


def collect_gpu_metadata() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    if completed.returncode != 0:
        return {
            "available": False,
            "error": completed.stderr.strip() or f"nvidia-smi exited {completed.returncode}",
        }
    devices = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", maxsplit=4)]
        if len(fields) == 5:
            devices.append(
                dict(zip(("index", "name", "uuid", "driver_version", "memory_total_mib"), fields))
            )
    return {"available": True, "devices": devices}


def reserve_attempt(artifact_root: Path) -> Path:
    artifact_root.mkdir(parents=True, exist_ok=True)
    lock_path = artifact_root / ".campaign.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        numbers = [
            int(match.group(1))
            for path in artifact_root.iterdir()
            if (match := ATTEMPT_PATTERN.fullmatch(path.name))
        ]
        next_number = max(numbers, default=0) + 1
        while True:
            attempt_dir = artifact_root / f"attempt-{next_number:03d}"
            try:
                attempt_dir.mkdir()
                break
            except FileExistsError:  # Defensive even while the advisory lock is held.
                next_number += 1
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
    return attempt_dir


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def selected_environment(additional_names: Sequence[str]) -> dict[str, str | None]:
    names = dict.fromkeys((*DEFAULT_CAPTURE_ENV, *additional_names))
    return {name: os.environ.get(name) for name in names}


def output_paths(attempt_dir: Path, declared: Sequence[str]) -> dict[str, Path]:
    paths = {"command.log": attempt_dir / "command.log"}
    for raw_path in declared:
        path = Path(raw_path).expanduser()
        paths[raw_path] = path.resolve() if path.is_absolute() else (attempt_dir / path).resolve()
    return paths


def execute(args: argparse.Namespace, runner_argv: Sequence[str]) -> int:
    if not args.cwd.is_dir():
        raise CampaignError(f"Command working directory does not exist: {args.cwd}")
    if not args.repo_root.is_dir():
        raise CampaignError(f"Git worktree does not exist: {args.repo_root}")
    git_metadata = collect_git_metadata(args.repo_root)
    if not git_metadata["clean"]:
        raise CampaignError(
            "Refusing to create an attempt from a dirty Git tree:\n"
            + str(git_metadata["status_porcelain"])
        )
    datasets = [fingerprint_file(path) for path in args.dataset_file]
    runtime = collect_runtime_metadata()
    attempt_dir = reserve_attempt(args.artifact_root)
    status_path = attempt_dir / "status.json"
    outputs = output_paths(attempt_dir, args.output_file)
    parameters = {
        "seed": args.seed,
        "doa_seed": args.doa_seed,
        "min_responses": args.min_responses,
        "split_seed": args.split_seed,
    }
    status: dict[str, Any] = {
        "schema_version": 1,
        "attempt": attempt_dir.name,
        "status": "running",
        "dry_run": bool(args.dry_run),
        "started_at_utc": utc_now(),
        "ended_at_utc": None,
        "exit_code": None,
        "invocation": {
            "argv": list(runner_argv),
            "command": list(args.command),
            "cwd": str(args.cwd),
            "environment": selected_environment(args.capture_env),
        },
        "parameters": parameters,
        "git": git_metadata,
        "datasets": datasets,
        "runtime": runtime,
        "output_hashes": {},
    }
    atomic_write_json(status_path, status)

    child_env = os.environ.copy()
    child_env.update(
        {
            "CAMPAIGN_ATTEMPT_DIR": str(attempt_dir),
            "CAMPAIGN_SEED": str(args.seed),
            "CAMPAIGN_DOA_SEED": str(args.doa_seed),
            "CAMPAIGN_MIN_RESPONSES": str(args.min_responses),
            "CAMPAIGN_SPLIT_SEED": str(args.split_seed),
        }
    )
    exit_code: int | None = None
    error: str | None = None
    try:
        with outputs["command.log"].open("x", encoding="utf-8") as log_handle:
            log_handle.write("$ " + shlex.join(args.command) + "\n")
            log_handle.flush()
            if args.dry_run:
                log_handle.write("[dry-run] command not executed\n")
            else:
                completed = subprocess.run(
                    args.command,
                    cwd=args.cwd,
                    env=child_env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                exit_code = completed.returncode
    except FileNotFoundError as exc:
        exit_code = 127
        error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # Ensure an allocated attempt always receives a terminal status.
        exit_code = 1
        error = f"{type(exc).__name__}: {exc}"

    if args.dry_run:
        status["status"] = "dry_run"
    elif exit_code == 0:
        status["status"] = "completed"
    else:
        status["status"] = "failed"
    status["exit_code"] = exit_code
    status["ended_at_utc"] = utc_now()
    if error is not None:
        status["error"] = error
    status["output_hashes"] = {
        name: fingerprint_output(path) for name, path in outputs.items()
    }
    atomic_write_json(status_path, status)
    print(attempt_dir)
    return 0 if args.dry_run else int(exit_code or 0)


def main(argv: Sequence[str] | None = None) -> int:
    invocation = list(sys.argv if argv is None else [sys.argv[0], *argv])
    args = parse_args(argv)
    try:
        return execute(args, invocation)
    except CampaignError as exc:
        print(f"campaign error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
