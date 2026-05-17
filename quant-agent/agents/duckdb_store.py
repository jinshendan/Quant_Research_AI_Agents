from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb  # type: ignore[import-untyped]
import pandas as pd

from agents.trading_calendar import CALENDAR_COLUMNS

OHLCV_TABLE = "market_ohlcv_aligned"
RUNS_TABLE = "market_data_runs"


@dataclass(frozen=True, slots=True)
class MarketDataStorageContext:
    """Metadata needed to persist one market-data run."""

    run_id: str
    task_id: str
    universe: str
    provider: str
    frequency: str
    adjust: str
    start_date: str
    end_date: str
    raw_data_path: str
    processed_data_path: str
    aligned_data_path: str
    raw_rows: int
    processed_rows: int
    aligned_rows: int
    cleaning_stats: dict[str, int]
    calendar_stats: dict[str, int]


@dataclass(frozen=True, slots=True)
class MarketDataStorageResult:
    """DuckDB persistence result for one market-data run."""

    database_path: Path
    ohlcv_table: str
    runs_table: str
    run_id: str
    rows_written: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "database_path": str(self.database_path),
            "ohlcv_table": self.ohlcv_table,
            "runs_table": self.runs_table,
            "run_id": self.run_id,
            "rows_written": self.rows_written,
        }


class DuckDBMarketDataStore:
    """Persist aligned OHLCV data and run metadata into DuckDB."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

    def store_aligned_ohlcv(
        self,
        aligned_data: pd.DataFrame,
        *,
        context: MarketDataStorageContext,
    ) -> MarketDataStorageResult:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        with duckdb.connect(str(self.database_path)) as connection:
            self._ensure_schema(connection)
            rows_written = self._replace_aligned_rows(connection, aligned_data, context)
            self._record_run(connection, context, rows_written)

        return MarketDataStorageResult(
            database_path=self.database_path,
            ohlcv_table=OHLCV_TABLE,
            runs_table=RUNS_TABLE,
            run_id=context.run_id,
            rows_written=rows_written,
        )

    def _ensure_schema(self, connection: duckdb.DuckDBPyConnection) -> None:
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {OHLCV_TABLE} (
                run_id VARCHAR,
                universe VARCHAR,
                provider VARCHAR,
                frequency VARCHAR,
                adjust VARCHAR,
                date DATE,
                symbol VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                amount DOUBLE,
                amplitude DOUBLE,
                pct_change DOUBLE,
                price_change DOUBLE,
                turnover_rate DOUBLE,
                is_expected_trading_day BOOLEAN,
                is_suspended_or_missing BOOLEAN,
                updated_at TIMESTAMP
            )
            """
        )
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {RUNS_TABLE} (
                run_id VARCHAR PRIMARY KEY,
                task_id VARCHAR,
                universe VARCHAR,
                provider VARCHAR,
                frequency VARCHAR,
                adjust VARCHAR,
                start_date DATE,
                end_date DATE,
                raw_data_path VARCHAR,
                processed_data_path VARCHAR,
                aligned_data_path VARCHAR,
                raw_rows BIGINT,
                processed_rows BIGINT,
                aligned_rows BIGINT,
                rows_written BIGINT,
                cleaning_stats_json VARCHAR,
                calendar_stats_json VARCHAR,
                created_at TIMESTAMP
            )
            """
        )

    def _replace_aligned_rows(
        self,
        connection: duckdb.DuckDBPyConnection,
        aligned_data: pd.DataFrame,
        context: MarketDataStorageContext,
    ) -> int:
        if aligned_data.empty:
            return 0

        frame = _prepare_aligned_frame(aligned_data, context)
        symbols = pd.DataFrame({"symbol": sorted(frame["symbol"].drop_duplicates().tolist())})
        start_date = frame["date"].min().date()
        end_date = frame["date"].max().date()

        connection.register("incoming_symbols", symbols)
        connection.execute(
            f"""
            DELETE FROM {OHLCV_TABLE}
            USING incoming_symbols
            WHERE {OHLCV_TABLE}.symbol = incoming_symbols.symbol
              AND {OHLCV_TABLE}.provider = ?
              AND {OHLCV_TABLE}.universe = ?
              AND {OHLCV_TABLE}.frequency = ?
              AND {OHLCV_TABLE}.adjust = ?
              AND {OHLCV_TABLE}.date BETWEEN ? AND ?
            """,
            [
                context.provider,
                context.universe,
                context.frequency,
                context.adjust,
                start_date,
                end_date,
            ],
        )
        connection.unregister("incoming_symbols")

        connection.register("aligned_rows", frame)
        connection.execute(
            f"""
            INSERT INTO {OHLCV_TABLE}
            SELECT
                run_id,
                universe,
                provider,
                frequency,
                adjust,
                date,
                symbol,
                open,
                high,
                low,
                close,
                volume,
                amount,
                amplitude,
                pct_change,
                price_change,
                turnover_rate,
                is_expected_trading_day,
                is_suspended_or_missing,
                updated_at
            FROM aligned_rows
            """
        )
        connection.unregister("aligned_rows")
        return len(frame)

    def _record_run(
        self,
        connection: duckdb.DuckDBPyConnection,
        context: MarketDataStorageContext,
        rows_written: int,
    ) -> None:
        now = datetime.now(UTC).replace(tzinfo=None)
        connection.execute(f"DELETE FROM {RUNS_TABLE} WHERE run_id = ?", [context.run_id])
        connection.execute(
            f"""
            INSERT INTO {RUNS_TABLE} (
                run_id,
                task_id,
                universe,
                provider,
                frequency,
                adjust,
                start_date,
                end_date,
                raw_data_path,
                processed_data_path,
                aligned_data_path,
                raw_rows,
                processed_rows,
                aligned_rows,
                rows_written,
                cleaning_stats_json,
                calendar_stats_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                context.run_id,
                context.task_id,
                context.universe,
                context.provider,
                context.frequency,
                context.adjust,
                context.start_date,
                context.end_date,
                context.raw_data_path,
                context.processed_data_path,
                context.aligned_data_path,
                context.raw_rows,
                context.processed_rows,
                context.aligned_rows,
                rows_written,
                json.dumps(context.cleaning_stats, sort_keys=True),
                json.dumps(context.calendar_stats, sort_keys=True),
                now,
            ],
        )


def _prepare_aligned_frame(
    aligned_data: pd.DataFrame,
    context: MarketDataStorageContext,
) -> pd.DataFrame:
    missing_columns = [column for column in CALENDAR_COLUMNS if column not in aligned_data.columns]
    if missing_columns:
        msg = f"Aligned OHLCV data missing required columns: {missing_columns}."
        raise ValueError(msg)

    frame = aligned_data[CALENDAR_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["symbol"] = frame["symbol"].astype("string").str.strip().str.zfill(6)
    frame["is_expected_trading_day"] = frame["is_expected_trading_day"].astype(bool)
    frame["is_suspended_or_missing"] = frame["is_suspended_or_missing"].astype(bool)
    frame.insert(0, "adjust", context.adjust)
    frame.insert(0, "frequency", context.frequency)
    frame.insert(0, "provider", context.provider)
    frame.insert(0, "universe", context.universe)
    frame.insert(0, "run_id", context.run_id)
    frame["updated_at"] = datetime.now(UTC).replace(tzinfo=None)
    return frame

