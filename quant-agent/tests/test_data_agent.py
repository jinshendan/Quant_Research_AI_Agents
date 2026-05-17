from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest

from agents.data_agent import DataAgent, MarketDataSpec
from core.config import AppConfig
from core.logging import configure_logging, get_agent_logger
from core.models import AgentRequest


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig.from_env(project_root=tmp_path, environ={})


def test_market_data_spec_normalizes_valid_payload() -> None:
    spec = MarketDataSpec.from_payload(
        {
            "universe": " CSI500 ",
            "start_date": "2020-01-01",
            "end_date": "2025-12-31",
            "frequency": "Daily",
            "provider": "AkShare",
        }
    )

    assert spec.to_dict() == {
        "universe": "CSI500",
        "start_date": "2020-01-01",
        "end_date": "2025-12-31",
        "frequency": "daily",
        "provider": "akshare",
    }


def test_market_data_spec_rejects_invalid_date_range() -> None:
    with pytest.raises(ValueError, match="end_date"):
        MarketDataSpec.from_payload(
            {
                "universe": "CSI500",
                "start_date": "2025-01-01",
                "end_date": "2020-01-01",
            }
        )


def test_data_agent_run_validates_request_and_creates_storage(tmp_path: Path) -> None:
    stream = StringIO()
    configure_logging(stream=stream)
    agent = DataAgent(config=_config(tmp_path), logger=get_agent_logger("DataAgent"))
    request = AgentRequest.create(
        {
            "universe": "CSI500",
            "start_date": "2020-01-01",
            "end_date": "2025-12-31",
        },
        task_id="data-task-1",
    )

    response = agent.run(request)

    assert response.status == "success"
    assert response.output["state"] == "validated"
    assert response.output["market_data"] is None
    assert response.output["request"]["provider"] == "akshare"
    assert response.metadata["agent"] == "DataAgent"
    assert response.metadata["task_id"] == "data-task-1"
    assert (tmp_path / "data" / "raw").is_dir()
    assert "DataAgent | validate_request | success" in stream.getvalue()


def test_data_agent_run_returns_error_for_bad_payload(tmp_path: Path) -> None:
    stream = StringIO()
    configure_logging(stream=stream)
    agent = DataAgent(config=_config(tmp_path), logger=get_agent_logger("DataAgent"))
    request = AgentRequest.create({"universe": "", "start_date": "2020-01-01"})

    response = agent.run(request)

    assert response.status == "error"
    assert response.error == "payload.universe must be a non-empty string."
    assert response.output == {}
    assert "DataAgent | validate_request | error" in stream.getvalue()


def test_data_agent_download_boundary_is_explicit(tmp_path: Path) -> None:
    agent = DataAgent(config=_config(tmp_path))
    spec = MarketDataSpec.from_payload(
        {
            "universe": "CSI500",
            "start_date": "2020-01-01",
            "end_date": "2025-12-31",
        }
    )

    with pytest.raises(NotImplementedError, match="Day 3"):
        agent.download_ohlcv(spec)

