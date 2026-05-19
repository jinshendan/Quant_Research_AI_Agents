from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from agents.akshare_smoke import AkShareSmokeSpec, run_akshare_smoke
from agents.market_data_provider import OHLCV_COLUMNS
from core.config import AppConfig


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig.from_env(project_root=tmp_path, environ={})


def test_akshare_smoke_reports_success_with_injected_providers(tmp_path: Path) -> None:
    report = run_akshare_smoke(
        _config(tmp_path),
        AkShareSmokeSpec(
            symbols=("000001",),
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 3),
            retry_backoff_sec=0.0,
        ),
        providers={"akshare": SmokeMarketDataProvider()},
        calendar_providers={"akshare": SmokeCalendarProvider()},
    )

    payload = report.to_dict()
    assert report.status == "success"
    assert report.success is True
    assert report.exit_code == 0
    assert payload["diagnostics"][1]["name"] == "akshare_import"
    assert payload["diagnostics"][1]["status"] == "skipped"
    assert payload["output"]["raw_rows"] == 1
    assert payload["output"]["aligned_rows"] == 2
    assert Path(payload["output"]["raw_data_path"]).is_file()
    assert Path(payload["output"]["aligned_data_path"]).is_file()


def test_akshare_smoke_reports_partial_success_and_manifest(tmp_path: Path) -> None:
    report = run_akshare_smoke(
        _config(tmp_path),
        AkShareSmokeSpec(
            symbols=("000001", "000002"),
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 3),
            max_retries=1,
            retry_backoff_sec=0.0,
        ),
        providers={"akshare": SmokeMarketDataProvider(always_fail={"000002"})},
        calendar_providers={"akshare": SmokeCalendarProvider()},
    )

    payload = report.to_dict()
    assert report.status == "partial_success"
    assert report.success is False
    assert report.exit_code == 2
    assert payload["output"]["failed_symbols"] == ["000002"]
    assert payload["output"]["cache_stats"]["status"] == "disabled"
    assert Path(payload["output"]["failure_manifest_path"]).is_file()
    assert "Open the failure_manifest_path" in payload["suggested_actions"][0]


def test_akshare_smoke_reports_error_when_all_symbols_fail(tmp_path: Path) -> None:
    report = run_akshare_smoke(
        _config(tmp_path),
        AkShareSmokeSpec(
            symbols=("000001",),
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 3),
            max_retries=0,
            retry_backoff_sec=0.0,
        ),
        providers={"akshare": SmokeMarketDataProvider(always_fail={"000001"})},
        calendar_providers={"akshare": SmokeCalendarProvider()},
    )

    payload = report.to_dict()
    assert report.status == "error"
    assert report.exit_code == 1
    assert "No OHLCV data downloaded" in str(payload["error"])
    assert "RuntimeError: provider failure for 000001" in str(payload["error"])
    assert payload["diagnostics"][-1]["name"] == "data_agent_run"
    assert payload["diagnostics"][-1]["status"] == "error"
    assert any("Verify the symbol" in action for action in payload["suggested_actions"])


class SmokeMarketDataProvider:
    name = "akshare"

    def __init__(self, *, always_fail: set[str] | None = None) -> None:
        self.always_fail = set(always_fail or set())

    def resolve_symbols(self, universe: str) -> list[str]:
        return [universe]

    def download_symbol_ohlcv(
        self,
        *,
        symbol: str,
        start_date: date,
        end_date: date,
        frequency: str,
        adjust: str,
    ) -> pd.DataFrame:
        if symbol in self.always_fail:
            raise RuntimeError(f"provider failure for {symbol}")
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
                    "amount": 10500,
                    "amplitude": 1.5,
                    "pct_change": 0.5,
                    "price_change": 0.05,
                    "turnover_rate": 0.8,
                }
            ],
            columns=OHLCV_COLUMNS,
        )


class SmokeCalendarProvider:
    name = "akshare"

    def get_trading_days(self, *, start_date: date, end_date: date) -> list[date]:
        return [start_date, end_date]
