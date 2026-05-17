from __future__ import annotations

import pandas as pd
import pytest

from agents.market_data_provider import OHLCV_COLUMNS
from agents.ohlcv_cleaner import clean_ohlcv


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "date": "2024-01-02",
        "symbol": "1",
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
    row.update(overrides)
    return row


def test_clean_ohlcv_filters_missing_invalid_duplicate_and_suspended_rows() -> None:
    raw = pd.DataFrame(
        [
            _row(symbol="1", date="2024-01-02", close=10.4),
            _row(symbol="1", date="2024-01-02", close=10.5),
            _row(symbol="2", date="not-a-date"),
            _row(symbol="3", close=None),
            _row(symbol="4", high=9.0),
            _row(symbol="5", volume=0),
            _row(symbol="6", amount=0),
            _row(symbol="7", amplitude=None, pct_change=None),
        ]
    )

    result = clean_ohlcv(raw)

    assert list(result.data["symbol"]) == ["000001", "000007"]
    assert list(result.data.columns) == OHLCV_COLUMNS
    assert result.data.loc[result.data["symbol"] == "000007", "amplitude"].iloc[0] == 0.0
    assert result.data.loc[result.data["symbol"] == "000007", "pct_change"].iloc[0] == 0.0
    assert result.stats == {
        "input_rows": 8,
        "output_rows": 2,
        "dropped_rows": 6,
        "invalid_identity_rows": 1,
        "duplicate_rows": 1,
        "missing_essential_rows": 1,
        "invalid_price_rows": 1,
        "suspended_rows": 2,
        "filled_optional_missing_cells": 2,
    }


def test_clean_ohlcv_rejects_missing_required_columns() -> None:
    raw = pd.DataFrame([{"date": "2024-01-02", "symbol": "000001"}])

    with pytest.raises(ValueError, match="missing required columns"):
        clean_ohlcv(raw)


def test_clean_ohlcv_keeps_empty_schema_for_empty_input() -> None:
    raw = pd.DataFrame(columns=OHLCV_COLUMNS)

    result = clean_ohlcv(raw)

    assert result.data.empty
    assert list(result.data.columns) == OHLCV_COLUMNS
    assert result.stats["input_rows"] == 0
    assert result.stats["output_rows"] == 0

