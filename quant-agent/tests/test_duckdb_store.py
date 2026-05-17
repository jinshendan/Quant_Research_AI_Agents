from __future__ import annotations

import json
from pathlib import Path

import duckdb  # type: ignore[import-untyped]
import pandas as pd
import pytest

from agents.duckdb_store import DuckDBMarketDataStore, MarketDataStorageContext
from agents.trading_calendar import CALENDAR_COLUMNS


def _aligned_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "symbol": "000001",
                "open": 10.0,
                "high": 10.5,
                "low": 9.8,
                "close": 10.2,
                "volume": 1000,
                "amount": 10200.0,
                "amplitude": 1.0,
                "pct_change": 0.2,
                "price_change": 0.02,
                "turnover_rate": 0.5,
                "is_expected_trading_day": True,
                "is_suspended_or_missing": False,
            },
            {
                "date": "2024-01-03",
                "symbol": "000001",
                "open": None,
                "high": None,
                "low": None,
                "close": None,
                "volume": None,
                "amount": None,
                "amplitude": None,
                "pct_change": None,
                "price_change": None,
                "turnover_rate": None,
                "is_expected_trading_day": True,
                "is_suspended_or_missing": True,
            },
        ],
        columns=CALENDAR_COLUMNS,
    )


def _context(run_id: str = "run-1") -> MarketDataStorageContext:
    return MarketDataStorageContext(
        run_id=run_id,
        task_id=run_id,
        universe="000001",
        provider="akshare",
        frequency="daily",
        adjust="",
        start_date="2024-01-02",
        end_date="2024-01-03",
        raw_data_path="/tmp/raw.csv",
        processed_data_path="/tmp/processed.csv",
        aligned_data_path="/tmp/aligned.csv",
        raw_rows=1,
        processed_rows=1,
        aligned_rows=2,
        cleaning_stats={"output_rows": 1},
        calendar_stats={"output_rows": 2, "missing_or_suspended_rows": 1},
    )


def test_duckdb_store_writes_aligned_rows_and_run_metadata(tmp_path: Path) -> None:
    database_path = tmp_path / "data" / "quant_agent.duckdb"
    store = DuckDBMarketDataStore(database_path)

    result = store.store_aligned_ohlcv(_aligned_rows(), context=_context())

    assert result.database_path == database_path
    assert result.rows_written == 2
    assert database_path.is_file()

    with duckdb.connect(str(database_path)) as connection:
        ohlcv_rows = connection.execute("SELECT count(*) FROM market_ohlcv_aligned").fetchone()
        missing_rows = connection.execute(
            "SELECT count(*) FROM market_ohlcv_aligned WHERE is_suspended_or_missing"
        ).fetchone()
        run = connection.execute(
            "SELECT run_id, rows_written, cleaning_stats_json FROM market_data_runs"
        ).fetchone()

    assert ohlcv_rows == (2,)
    assert missing_rows == (1,)
    assert run is not None
    assert run[0] == "run-1"
    assert run[1] == 2
    assert json.loads(run[2]) == {"output_rows": 1}


def test_duckdb_store_replaces_existing_symbol_range(tmp_path: Path) -> None:
    database_path = tmp_path / "quant_agent.duckdb"
    store = DuckDBMarketDataStore(database_path)

    store.store_aligned_ohlcv(_aligned_rows(), context=_context("run-1"))
    store.store_aligned_ohlcv(_aligned_rows(), context=_context("run-2"))

    with duckdb.connect(str(database_path)) as connection:
        ohlcv_rows = connection.execute("SELECT count(*) FROM market_ohlcv_aligned").fetchone()
        runs = connection.execute("SELECT count(*) FROM market_data_runs").fetchone()
        latest_run = connection.execute(
            "SELECT DISTINCT run_id FROM market_ohlcv_aligned ORDER BY run_id"
        ).fetchall()

    assert ohlcv_rows == (2,)
    assert runs == (2,)
    assert latest_run == [("run-2",)]


def test_duckdb_store_rejects_missing_aligned_columns(tmp_path: Path) -> None:
    store = DuckDBMarketDataStore(tmp_path / "quant_agent.duckdb")

    with pytest.raises(ValueError, match="missing required columns"):
        store.store_aligned_ohlcv(
            pd.DataFrame([{"date": "2024-01-02", "symbol": "000001"}]),
            context=_context(),
        )

