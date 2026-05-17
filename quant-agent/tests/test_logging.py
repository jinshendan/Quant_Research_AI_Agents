from __future__ import annotations

from io import StringIO

from core.logging import configure_logging, get_agent_logger


def test_configure_logging_writes_default_context() -> None:
    stream = StringIO()
    logger = configure_logging(stream=stream)

    logger.info("ready")

    output = stream.getvalue()
    assert "system | log | success | ready" in output


def test_agent_logger_includes_agent_context_and_overrides() -> None:
    stream = StringIO()
    configure_logging(stream=stream)
    logger = get_agent_logger("DataAgent")

    logger.info("downloaded rows", extra={"action": "download", "status": "success"})

    output = stream.getvalue()
    assert "DataAgent | download | success | downloaded rows" in output


def test_configure_logging_is_idempotent() -> None:
    old_stream = StringIO()
    new_stream = StringIO()

    logger = configure_logging(stream=old_stream)
    logger = configure_logging(stream=new_stream)
    logger.info("single handler")

    assert len(logger.handlers) == 1
    assert old_stream.getvalue() == ""
    assert "single handler" in new_stream.getvalue()


def test_configure_logging_can_write_to_file(tmp_path) -> None:
    stream = StringIO()
    log_file = tmp_path / "quant-agent.log"
    logger = configure_logging(stream=stream, log_file=log_file)

    logger.error(
        "failed request",
        extra={"agent": "DataAgent", "action": "fetch", "status": "error"},
    )

    output = log_file.read_text(encoding="utf-8")
    assert "DataAgent | fetch | error | failed request" in output

