from __future__ import annotations

import logging
from datetime import datetime, UTC
from pathlib import Path


def setup_logging(log_dir: str | Path = "logs", name: str = "train") -> tuple[logging.Logger, str]:
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_path = log_dir_path / f"{name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.log"

    logger_name = f"decoupled_cd.{name}.{log_path.stem}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_path)
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger, str(log_path)
