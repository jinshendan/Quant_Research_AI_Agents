from __future__ import annotations

import logging
import sys
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any, TextIO

LOG_FORMAT = "%(asctime)s | %(agent)s | %(action)s | %(status)s | %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"

_DEFAULT_CONTEXT = {
    "agent": "system",
    "action": "log",
    "status": "success",
}


class ContextDefaultsFilter(logging.Filter):
    """Ensure structured log fields are always present."""

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in _DEFAULT_CONTEXT.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


class AgentLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that preserves agent context while allowing per-call overrides."""

    def process(
        self,
        msg: Any,
        kwargs: MutableMapping[str, Any],
    ) -> tuple[Any, MutableMapping[str, Any]]:
        base_extra: dict[str, Any] = dict(self.extra or {})
        call_extra = kwargs.get("extra")
        if isinstance(call_extra, Mapping):
            base_extra.update(call_extra)
        kwargs["extra"] = base_extra
        return msg, kwargs


def configure_logging(
    level: int | str = logging.INFO,
    *,
    stream: TextIO | None = None,
    log_file: str | Path | None = None,
    logger_name: str = "quant_agent",
) -> logging.Logger:
    """Configure structured logging for all agents.

    The function is intentionally idempotent so tests, notebooks, and Streamlit reruns
    do not duplicate handlers.
    """

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    context_filter = ContextDefaultsFilter()

    stream_handler = logging.StreamHandler(stream or sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(context_filter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(context_filter)
        logger.addHandler(file_handler)

    return logger


def get_agent_logger(
    agent_name: str,
    *,
    logger_name: str = "quant_agent",
) -> AgentLoggerAdapter:
    """Return a structured logger adapter for an agent."""

    return AgentLoggerAdapter(
        logging.getLogger(logger_name),
        {"agent": agent_name, "action": "idle", "status": "running"},
    )
