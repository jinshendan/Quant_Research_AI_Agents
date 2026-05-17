from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from agents.market_data_provider import OHLCV_COLUMNS
from agents.trading_calendar import (
    AkShareTradingCalendarProvider,
    CALENDAR_COLUMNS,
    align_to_trading_calendar,
)


def _row(symbol: str, day: str, close: float) -> dict[str, object]:
    return {
        "date": day,
        "symbol": symbol,
        "open": close - 0.1,
        "high": close + 0.2,
        "low": close - 0.2,
        "close": close,
        "volume": 1000,
        "amount": 10000.0,
        "amplitude": 1.0,
        "pct_change": 0.1,
        "price_change": 0.01,
        "turnover_rate": 0.5,
    }


def test_align_to_trading_calendar_adds_missing_symbol_date_rows() -> None:
    processed = pd.DataFrame(
        [
            _row("1", "2024-01-02", 10.0),
            _row("000002", "2024-01-03", 20.0),
        ]
    )

    result = align_to_trading_calendar(
        processed,
        symbols=["000001", "000002"],
        trading_days=[date(2024, 1, 2), date(2024, 1, 3)],
    )

    assert list(result.data.columns) == CALENDAR_COLUMNS
    assert len(result.data) == 4
    assert result.stats == {
        "input_rows": 2,
        "output_rows": 4,
        "symbols_count": 2,
        "trading_days_count": 2,
        "missing_or_suspended_rows": 2,
        "observed_rows": 2,
    }

    missing = result.data[result.data["is_suspended_or_missing"]]
    assert set(missing["symbol"]) == {"000001", "000002"}
    observed = result.data[~result.data["is_suspended_or_missing"]]
    assert observed["close"].notna().all()


def test_align_to_trading_calendar_rejects_missing_required_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        align_to_trading_calendar(
            pd.DataFrame([{"date": "2024-01-02", "symbol": "000001"}]),
            symbols=["000001"],
            trading_days=[date(2024, 1, 2)],
        )


def test_align_to_trading_calendar_rejects_empty_calendar() -> None:
    processed = pd.DataFrame(columns=OHLCV_COLUMNS)

    with pytest.raises(ValueError, match="trading_days"):
        align_to_trading_calendar(processed, symbols=["000001"], trading_days=[])


def test_akshare_calendar_provider_filters_and_sorts_dates() -> None:
    def fake_calendar_func() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "trade_date": [
                    "2024-01-03",
                    "2024-01-02",
                    "2024-01-02",
                    "2023-12-29",
                    "bad-date",
                ]
            }
        )

    provider = AkShareTradingCalendarProvider(calendar_func=fake_calendar_func)

    assert provider.get_trading_days(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
    ) == [date(2024, 1, 2), date(2024, 1, 3)]

