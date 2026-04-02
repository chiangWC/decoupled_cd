from __future__ import annotations

import os
import subprocess
from typing import Optional

import torch


def parse_gpu_ids(text: str | None) -> list[int]:
    if text is None:
        return []
    out: list[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    return out


def get_gpu_memory_map() -> dict[int, int]:
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,nounits,noheader"],
            encoding="utf-8",
        )
    except Exception:
        return {}

    memory_map: dict[int, int] = {}
    for line in result.strip().splitlines():
        idx_text, free_text = [part.strip() for part in line.split(",")]
        memory_map[int(idx_text)] = int(free_text)
    return memory_map


def select_best_gpu(candidates: list[int] | None = None) -> Optional[int]:
    memory_map = get_gpu_memory_map()
    if not memory_map:
        return None
    target = candidates if candidates else list(memory_map.keys())
    available = [(gid, memory_map.get(gid, -1)) for gid in target if gid in memory_map]
    if not available:
        return None
    available.sort(key=lambda item: item[1], reverse=True)
    return available[0][0]


def resolve_device(device: str = "auto", gpu_candidates: str | None = None) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if not torch.cuda.is_available():
        return torch.device("cpu")

    candidates = parse_gpu_ids(gpu_candidates)
    gpu_id = select_best_gpu(candidates)
    if gpu_id is None:
        return torch.device("cuda:0")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return torch.device("cuda:0")
