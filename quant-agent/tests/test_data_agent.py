from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from agents.data_agent import DataAgent, MarketDataSpec
from core.config import AppConfig
from core.logging import configure_logging, get_agent_logger
from core.models import AgentRequest


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig.from_env(project_root=tmp_path, environ={})


class FakeMarketDataProvider:
    name = "akshare"

    def resolve_symbols(self, universe: str) -> list[str]:
        if universe == "CSI500":
            return ["000001", "000002"]
        return [universe]

    def download_symbol_ohlcv(
        self,
        *,
        symbol: str,
        start_date,
        end_date,
        frequency: str,
        adjust: str,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "date": pd.Timestamp(start_date),
                    "symbol": symbol,
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.5,
                    "close": 10.5,
                    "volume": 1000,
                    "amount": 10500.0,
                    "amplitude": 5.0,
                    "pct_change": 1.2,
                    "price_change": 0.1,
                    "turnover_rate": 0.8,
                }
            ]
        )


def test_market_data_spec_normalizes_valid_payload() -> None:
    spec = MarketDataSpec.from_payload(
        {
            "universe": " CSI500 ",
            "start_date": "2020-01-01",
            "end_date": "2025-12-31",
            "frequency": "Daily",
            "provider": "AkShare",
            "symbols": ["000001", "000001", "000002"],
            "adjust": "QFQ",
        }
    )

    assert spec.to_dict() == {
        "universe": "CSI500",
        "start_date": "2020-01-01",
        "end_date": "2025-12-31",
        "frequency": "daily",
        "provider": "akshare",
        "symbols": ["000001", "000002"],
        "adjust": "qfq",
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


def test_market_data_spec_rejects_invalid_symbols() -> None:
    with pytest.raises(ValueError, match="Invalid stock symbol"):
        MarketDataSpec.from_payload(
            {
                "universe": "CSI500",
                "start_date": "2020-01-01",
                "end_date": "2025-12-31",
                "symbols": ["BAD"],
            }
        )


def test_data_agent_run_downloads_and_stores_raw_data(tmp_path: Path) -> None:
    stream = StringIO()
    configure_logging(stream=stream)
    agent = DataAgent(
        config=_config(tmp_path),
        logger=get_agent_logger("DataAgent"),
        providers={"akshare": FakeMarketDataProvider()},
    )
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
    assert response.output["state"] == "processed"
    assert response.output["raw_rows"] == 2
    assert response.output["processed_rows"] == 2
    assert response.output["symbols"] == ["000001", "000002"]
    assert response.output["request"]["provider"] == "akshare"
    assert Path(response.output["raw_data_path"]).is_file()
    assert Path(response.output["processed_data_path"]).is_file()
    assert response.output["cleaning_stats"]["suspended_rows"] == 0
    assert response.metadata["agent"] == "DataAgent"
    assert response.metadata["task_id"] == "data-task-1"
    assert response.metadata["rows"] == 2
    assert response.metadata["cleaning_stats"]["output_rows"] == 2
    assert (tmp_path / "data" / "raw").is_dir()
    assert (tmp_path / "data" / "processed").is_dir()
    assert "DataAgent | prepare_ohlcv | success" in stream.getvalue()

    stored = pd.read_csv(response.output["raw_data_path"])
    assert list(stored["symbol"].astype(str).str.zfill(6)) == ["000001", "000002"]
    processed = pd.read_csv(response.output["processed_data_path"])
    assert list(processed["symbol"].astype(str).str.zfill(6)) == ["000001", "000002"]


def test_data_agent_run_returns_error_for_bad_payload(tmp_path: Path) -> None:
    stream = StringIO()
    configure_logging(stream=stream)
    agent = DataAgent(
        config=_config(tmp_path),
        logger=get_agent_logger("DataAgent"),
        providers={"akshare": FakeMarketDataProvider()},
    )
    request = AgentRequest.create({"universe": "", "start_date": "2020-01-01"})

    response = agent.run(request)

    assert response.status == "error"
    assert response.error == "payload.universe must be a non-empty string."
    assert response.output == {}
    assert "DataAgent | validate_request | error" in stream.getvalue()


def test_data_agent_download_ohlcv_uses_explicit_symbols(tmp_path: Path) -> None:
    agent = DataAgent(
        config=_config(tmp_path),
        providers={"akshare": FakeMarketDataProvider()},
    )
    spec = MarketDataSpec.from_payload(
        {
            "universe": "CSI500",
            "start_date": "2020-01-01",
            "end_date": "2025-12-31",
            "symbols": ["000001"],
        }
    )

    market_data = agent.download_ohlcv(spec)

    assert len(market_data) == 1
    assert market_data.iloc[0]["symbol"] == "000001"
