from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

import pandas as pd

from agents.market_data_provider import OHLCV_COLUMNS

CALENDAR_COLUMNS = [
    *OHLCV_COLUMNS,
    "is_expected_trading_day",
    "is_suspended_or_missing",
]


class TradingCalendarProvider(Protocol):
    """Trading calendar provider boundary used by DataAgent."""

    name: str

    def get_trading_days(self, *, start_date: date, end_date: date) -> list[date]:
        """Return exchange trading days in the inclusive date range."""
        ...


@dataclass(slots=True)
class AkShareTradingCalendarProvider:
    """AkShare-backed A-share trading calendar provider."""

    calendar_func: Callable[..., pd.DataFrame] | None = None
    name: str = "akshare"

    def __post_init__(self) -> None:
        if self.calendar_func is None:
            import akshare as ak  # type: ignore[import-untyped]

            self.calendar_func = ak.tool_trade_date_hist_sina

    def get_trading_days(self, *, start_date: date, end_date: date) -> list[date]:
        if self.calendar_func is None:
            msg = "AkShare trading calendar function is not configured."
            raise RuntimeError(msg)

        raw_calendar = self.calendar_func()
        if "trade_date" not in raw_calendar.columns:
            msg = "AkShare trading calendar is missing column: trade_date."
            raise ValueError(msg)

        calendar = pd.to_datetime(raw_calendar["trade_date"], errors="coerce").dropna()
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        filtered = calendar[(calendar >= start) & (calendar <= end)].drop_duplicates().sort_values()
        return [item.date() for item in filtered]


@dataclass(frozen=True, slots=True)
class CalendarAlignmentResult:
    """OHLCV data aligned to an expected symbol/date trading grid."""

    data: pd.DataFrame
    stats: dict[str, int]


def align_to_trading_calendar(
    processed: pd.DataFrame,
    *,
    symbols: Sequence[str],
    trading_days: Sequence[date],
) -> CalendarAlignmentResult:
    """Align processed OHLCV data to symbol/date trading days.

    Missing rows are retained with null OHLCV fields and marked as
    suspended/missing. Price imputation is intentionally avoided.
    """

    frame = _with_required_columns(processed)
    input_rows = len(frame)

    normalized_symbols = _normalize_symbols(symbols)
    normalized_days = _normalize_trading_days(trading_days)
    if not normalized_symbols:
        msg = "symbols must contain at least one symbol."
        raise ValueError(msg)
    if not normalized_days:
        msg = "trading_days must contain at least one date."
        raise ValueError(msg)

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["symbol"] = frame["symbol"].astype("string").str.strip().str.zfill(6)
    frame = frame.dropna(subset=["date", "symbol"]).copy()
    frame = frame.drop_duplicates(["symbol", "date"], keep="last")

    trading_index = pd.MultiIndex.from_product(
        [normalized_symbols, pd.to_datetime(normalized_days)],
        names=["symbol", "date"],
    )
    aligned = (
        frame.set_index(["symbol", "date"])
        .reindex(trading_index)
        .reset_index()
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )
    aligned["is_expected_trading_day"] = True
    aligned["is_suspended_or_missing"] = aligned["close"].isna()

    aligned = aligned[CALENDAR_COLUMNS]
    missing_rows = int(aligned["is_suspended_or_missing"].sum())
    stats = {
        "input_rows": input_rows,
        "output_rows": len(aligned),
        "symbols_count": len(normalized_symbols),
        "trading_days_count": len(normalized_days),
        "missing_or_suspended_rows": missing_rows,
        "observed_rows": len(aligned) - missing_rows,
    }
    return CalendarAlignmentResult(data=aligned, stats=stats)


def _with_required_columns(processed: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in OHLCV_COLUMNS if column not in processed.columns]
    if missing_columns:
        msg = f"Processed OHLCV data missing required columns: {missing_columns}."
        raise ValueError(msg)
    return processed[OHLCV_COLUMNS].copy()


def _normalize_symbols(symbols: Sequence[str]) -> list[str]:
    normalized = []
    for symbol in symbols:
        value = str(symbol).strip().upper().zfill(6)
        if len(value) == 6 and value.isdigit() and value not in normalized:
            normalized.append(value)
    return normalized


def _normalize_trading_days(trading_days: Sequence[date]) -> list[date]:
    normalized = pd.to_datetime(list(trading_days), errors="coerce").dropna().drop_duplicates()
    return [item.date() for item in normalized.sort_values()]

