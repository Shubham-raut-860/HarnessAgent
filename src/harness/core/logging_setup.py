"""Local structured logging for Codex Harness.

All logs stay on disk only. Nothing is sent anywhere.
Files written:
  logs/harness.jsonl          -- rotating master log (all agents, all runs)
  logs/runs/{run_id}.jsonl    -- per-run log (one file per agent execution)

These paths are in .gitignore and will never be pushed to GitHub.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class _JSONFormatter(logging.Formatter):
    """Formats every log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        # Attach any extra fields passed via logger.info(..., extra={...})
        for key, val in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            ):
                try:
                    json.dumps(val)
                    entry[key] = val
                except (TypeError, ValueError):
                    entry[key] = str(val)
        return json.dumps(entry, ensure_ascii=False)


def setup_logging(
    log_dir: str | Path = "logs",
    level: str = "INFO",
    max_bytes: int = 50 * 1024 * 1024,   # 50 MB per file
    backup_count: int = 10,
) -> Path:
    """Configure rotating JSON file logging for the whole harness.

    Call once at startup (api/main.py or worker entry point).
    Returns the log directory path.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    (log_path / "runs").mkdir(exist_ok=True)

    root = logging.getLogger("harness")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Rotating master log
    master_file = log_path / "harness.jsonl"
    handler = RotatingFileHandler(
        master_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(_JSONFormatter())
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        root.addHandler(handler)

    return log_path


def get_run_logger(run_id: str, log_dir: str | Path = "logs") -> logging.Logger:
    """Return a logger that writes ONLY to logs/runs/{run_id}.jsonl.

    Creates the file on first call. Isolated per run so you can tail a
    single agent's activity without grepping the master log.
    """
    log_path = Path(log_dir) / "runs"
    log_path.mkdir(parents=True, exist_ok=True)

    logger_name = f"harness.run.{run_id}"
    run_logger = logging.getLogger(logger_name)

    # Only add handler once
    if run_logger.handlers:
        return run_logger

    run_logger.setLevel(logging.DEBUG)
    run_logger.propagate = True   # also flows to master log

    run_file = log_path / f"{run_id}.jsonl"
    fh = logging.FileHandler(run_file, encoding="utf-8")
    fh.setFormatter(_JSONFormatter())
    fh.setLevel(logging.DEBUG)
    run_logger.addHandler(fh)

    return run_logger


def close_run_logger(run_id: str) -> None:
    """Flush and close the per-run log file handler."""
    logger_name = f"harness.run.{run_id}"
    run_logger = logging.getLogger(logger_name)
    for handler in list(run_logger.handlers):
        handler.flush()
        handler.close()
        run_logger.removeHandler(handler)
