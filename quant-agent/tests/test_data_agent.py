from __future__ import annotations

import json
from datetime import date
from io import StringIO
from pathlib import Path

import duckdb  # type: ignore[import-untyped]
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

    def __init__(self) -> None:
        self.resolve_calls = 0
        self.download_calls = 0

    def resolve_symbols(self, universe: str) -> list[str]:
        self.resolve_calls += 1
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
        self.download_calls += 1
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


class FakeTradingCalendarProvider:
    name = "akshare"

    def __init__(self) -> None:
        self.calls = 0

    def get_trading_days(self, *, start_date: date, end_date: date) -> list[date]:
        self.calls += 1
        return [
            pd.Timestamp(start_date).date(),
            pd.Timestamp("2020-01-02").date(),
            pd.Timestamp(end_date).date(),
        ]


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
        "use_cache": True,
        "force_refresh": False,
        "max_retries": 2,
        "retry_backoff_sec": 0.5,
        "continue_on_symbol_error": True,
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
        calendar_providers={"akshare": FakeTradingCalendarProvider()},
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
    assert response.output["state"] == "stored"
    assert response.output["raw_rows"] == 2
    assert response.output["processed_rows"] == 2
    assert response.output["aligned_rows"] == 6
    assert response.output["symbols"] == ["000001", "000002"]
    assert response.output["successful_symbols"] == ["000001", "000002"]
    assert response.output["failed_symbols"] == []
    assert response.output["request"]["provider"] == "akshare"
    assert Path(response.output["raw_data_path"]).is_file()
    assert Path(response.output["processed_data_path"]).is_file()
    assert Path(response.output["aligned_data_path"]).is_file()
    assert response.output["cleaning_stats"]["suspended_rows"] == 0
    assert response.output["calendar_stats"]["missing_or_suspended_rows"] == 4
    assert response.output["download_stats"]["failed_symbol_count"] == 0
    assert response.output["failure_manifest_path"] is None
    assert response.output["storage_stats"]["rows_written"] == 6
    assert response.output["cache_stats"]["status"] == "refreshed"
    assert Path(response.output["storage_stats"]["database_path"]).is_file()
    assert response.metadata["agent"] == "DataAgent"
    assert response.metadata["task_id"] == "data-task-1"
    assert response.metadata["rows"] == 6
    assert response.metadata["cleaning_stats"]["output_rows"] == 2
    assert response.metadata["calendar_stats"]["output_rows"] == 6
    assert response.metadata["download_stats"]["successful_symbol_count"] == 2
    assert response.metadata["storage_stats"]["rows_written"] == 6
    assert response.metadata["cache_stats"]["status"] == "refreshed"
    assert (tmp_path / "data" / "raw").is_dir()
    assert (tmp_path / "data" / "processed").is_dir()
    assert "DataAgent | prepare_ohlcv | success" in stream.getvalue()

    stored = pd.read_csv(response.output["raw_data_path"])
    assert list(stored["symbol"].astype(str).str.zfill(6)) == ["000001", "000002"]
    processed = pd.read_csv(response.output["processed_data_path"])
    assert list(processed["symbol"].astype(str).str.zfill(6)) == ["000001", "000002"]
    aligned = pd.read_csv(response.output["aligned_data_path"])
    assert len(aligned) == 6
    assert aligned["is_suspended_or_missing"].sum() == 4

    with duckdb.connect(response.output["storage_stats"]["database_path"]) as connection:
        stored_rows = connection.execute("SELECT count(*) FROM market_ohlcv_aligned").fetchone()
        run_rows = connection.execute("SELECT count(*) FROM market_data_runs").fetchone()
    assert stored_rows == (6,)
    assert run_rows == (1,)


def test_data_agent_run_returns_cached_result_without_provider_calls(tmp_path: Path) -> None:
    request_payload = {
        "universe": "CSI500",
        "start_date": "2020-01-01",
        "end_date": "2025-12-31",
    }
    first_provider = FakeMarketDataProvider()
    first_calendar = FakeTradingCalendarProvider()
    first_agent = DataAgent(
        config=_config(tmp_path),
        providers={"akshare": first_provider},
        calendar_providers={"akshare": first_calendar},
    )

    first_response = first_agent.run(AgentRequest.create(request_payload, task_id="data-task-1"))

    assert first_response.status == "success"
    assert first_response.output["cache_stats"]["status"] == "refreshed"
    assert first_provider.resolve_calls == 1
    assert first_provider.download_calls == 2
    assert first_calendar.calls == 1

    second_provider = ExplodingMarketDataProvider()
    second_calendar = ExplodingTradingCalendarProvider()
    second_agent = DataAgent(
        config=_config(tmp_path),
        providers={"akshare": second_provider},
        calendar_providers={"akshare": second_calendar},
    )

    second_response = second_agent.run(AgentRequest.create(request_payload, task_id="data-task-2"))

    assert second_response.status == "success"
    assert second_response.output["state"] == "cached"
    assert second_response.output["raw_rows"] == 2
    assert second_response.output["aligned_rows"] == 6
    assert second_response.output["cache_stats"]["status"] == "hit"
    assert second_response.metadata["task_id"] == "data-task-2"
    assert second_response.metadata["cache_stats"]["status"] == "hit"


def test_data_agent_force_refresh_bypasses_existing_cache(tmp_path: Path) -> None:
    request_payload = {
        "universe": "CSI500",
        "start_date": "2020-01-01",
        "end_date": "2025-12-31",
    }
    first_agent = DataAgent(
        config=_config(tmp_path),
        providers={"akshare": FakeMarketDataProvider()},
        calendar_providers={"akshare": FakeTradingCalendarProvider()},
    )
    first_agent.run(AgentRequest.create(request_payload, task_id="data-task-1"))

    second_provider = FakeMarketDataProvider()
    second_calendar = FakeTradingCalendarProvider()
    second_agent = DataAgent(
        config=_config(tmp_path),
        providers={"akshare": second_provider},
        calendar_providers={"akshare": second_calendar},
    )
    refreshed_payload = {**request_payload, "force_refresh": True}

    response = second_agent.run(AgentRequest.create(refreshed_payload, task_id="data-task-2"))

    assert response.status == "success"
    assert response.output["state"] == "stored"
    assert response.output["cache_stats"]["status"] == "refreshed"
    assert second_provider.resolve_calls == 1
    assert second_provider.download_calls == 2
    assert second_calendar.calls == 1


def test_data_agent_run_returns_error_for_bad_payload(tmp_path: Path) -> None:
    stream = StringIO()
    configure_logging(stream=stream)
    agent = DataAgent(
        config=_config(tmp_path),
        logger=get_agent_logger("DataAgent"),
        providers={"akshare": FakeMarketDataProvider()},
        calendar_providers={"akshare": FakeTradingCalendarProvider()},
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
        calendar_providers={"akshare": FakeTradingCalendarProvider()},
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


def test_data_agent_retries_transient_symbol_failure(tmp_path: Path) -> None:
    stream = StringIO()
    configure_logging(stream=stream)
    provider = FlakyMarketDataProvider(failures_before_success={"000001": 1})
    agent = DataAgent(
        config=_config(tmp_path),
        logger=get_agent_logger("DataAgent"),
        providers={"akshare": provider},
        calendar_providers={"akshare": FakeTradingCalendarProvider()},
    )

    response = agent.run(
        AgentRequest.create(
            {
                "universe": "CSI500",
                "start_date": "2020-01-01",
                "end_date": "2025-12-31",
                "max_retries": 1,
                "retry_backoff_sec": 0.0,
            },
            task_id="data-retry-1",
        )
    )

    assert response.status == "success"
    assert provider.download_attempts["000001"] == 2
    assert provider.download_attempts["000002"] == 1
    assert response.output["download_stats"]["retry_attempts"] == 1
    assert response.output["download_stats"]["failed_symbol_count"] == 0
    assert response.output["failure_manifest_path"] is None
    assert "DataAgent | download_symbol_ohlcv | retry" in stream.getvalue()
    assert "DataAgent | download_symbol_ohlcv | recovered" in stream.getvalue()


def test_data_agent_isolates_failed_symbols_and_writes_manifest(
    tmp_path: Path,
) -> None:
    provider = FlakyMarketDataProvider(always_fail={"000002"})
    agent = DataAgent(
        config=_config(tmp_path),
        providers={"akshare": provider},
        calendar_providers={"akshare": FakeTradingCalendarProvider()},
    )

    response = agent.run(
        AgentRequest.create(
            {
                "universe": "CSI500",
                "start_date": "2020-01-01",
                "end_date": "2025-12-31",
                "max_retries": 1,
                "retry_backoff_sec": 0.0,
            },
            task_id="data-partial-1",
        )
    )

    assert response.status == "success"
    assert response.output["symbols"] == ["000001", "000002"]
    assert response.output["successful_symbols"] == ["000001"]
    assert response.output["failed_symbols"] == ["000002"]
    assert response.output["raw_rows"] == 1
    assert response.output["aligned_rows"] == 3
    assert response.output["download_stats"]["requested_symbol_count"] == 2
    assert response.output["download_stats"]["successful_symbol_count"] == 1
    assert response.output["download_stats"]["failed_symbol_count"] == 1
    assert response.output["download_stats"]["retry_attempts"] == 1
    assert response.output["cache_stats"]["status"] == "skipped"
    assert response.output["cache_stats"]["reason"] == "partial_download"
    assert not Path(response.output["cache_stats"]["cache_path"]).exists()
    assert response.metadata["failure_manifest_path"] == response.output[
        "failure_manifest_path"
    ]

    manifest_path = Path(response.output["failure_manifest_path"])
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["task_id"] == "data-partial-1"
    assert manifest["download_stats"]["failed_symbols"] == ["000002"]
    assert manifest["download_stats"]["failures"][0]["attempts"] == 2
    assert manifest["download_stats"]["failures"][0]["error_type"] == "RuntimeError"


def test_data_agent_returns_error_when_all_symbols_fail(tmp_path: Path) -> None:
    agent = DataAgent(
        config=_config(tmp_path),
        providers={"akshare": FlakyMarketDataProvider(always_fail={"000001", "000002"})},
        calendar_providers={"akshare": FakeTradingCalendarProvider()},
    )

    response = agent.run(
        AgentRequest.create(
            {
                "universe": "CSI500",
                "start_date": "2020-01-01",
                "end_date": "2025-12-31",
                "max_retries": 1,
                "retry_backoff_sec": 0.0,
            },
            task_id="data-all-fail",
        )
    )

    assert response.status == "error"
    assert "No OHLCV data downloaded" in str(response.error)
    assert "000001" in str(response.error)
    assert "000002" in str(response.error)
    assert "RuntimeError: transient provider failure for 000001" in str(response.error)


class FlakyMarketDataProvider(FakeMarketDataProvider):
    def __init__(
        self,
        *,
        failures_before_success: dict[str, int] | None = None,
        always_fail: set[str] | None = None,
    ) -> None:
        super().__init__()
        self.failures_before_success = dict(failures_before_success or {})
        self.always_fail = set(always_fail or set())
        self.download_attempts: dict[str, int] = {}

    def download_symbol_ohlcv(
        self,
        *,
        symbol: str,
        start_date,
        end_date,
        frequency: str,
        adjust: str,
    ) -> pd.DataFrame:
        self.download_attempts[symbol] = self.download_attempts.get(symbol, 0) + 1
        if symbol in self.always_fail:
            raise RuntimeError(f"transient provider failure for {symbol}")
        failures_remaining = self.failures_before_success.get(symbol, 0)
        if failures_remaining > 0:
            self.failures_before_success[symbol] = failures_remaining - 1
            raise RuntimeError(f"transient provider failure for {symbol}")
        return super().download_symbol_ohlcv(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            adjust=adjust,
        )


class ExplodingMarketDataProvider:
    name = "akshare"

    def resolve_symbols(self, universe: str) -> list[str]:
        raise AssertionError("resolve_symbols should not be called on cache hit.")

    def download_symbol_ohlcv(
        self,
        *,
        symbol: str,
        start_date,
        end_date,
        frequency: str,
        adjust: str,
    ) -> pd.DataFrame:
        raise AssertionError("download_symbol_ohlcv should not be called on cache hit.")


class ExplodingTradingCalendarProvider:
    name = "akshare"

    def get_trading_days(self, *, start_date: date, end_date: date) -> list[date]:
        raise AssertionError("get_trading_days should not be called on cache hit.")
