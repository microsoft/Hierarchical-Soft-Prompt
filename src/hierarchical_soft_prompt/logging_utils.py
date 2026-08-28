import logging
import json
import os
from datetime import datetime, timezone


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def log_metrics(metrics: dict, step: int, logger: logging.Logger, prefix: str = ""):
    label = f"[{prefix}] " if prefix else ""
    logger.info(f"{label}step={step} | " + " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                                        for k, v in metrics.items()))


def save_results(results: dict, output_path: str):
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    saved = dict(results)
    saved["saved_at"] = datetime.now(timezone.utc).isoformat()
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(saved, handle, indent=2)
