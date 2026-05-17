from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from agents.market_data_provider import OHLCV_COLUMNS

ESSENTIAL_PRICE_COLUMNS = ["open", "high", "low", "close"]
ESSENTIAL_NUMERIC_COLUMNS = [*ESSENTIAL_PRICE_COLUMNS, "volume", "amount"]
OPTIONAL_NUMERIC_COLUMNS = ["amplitude", "pct_change", "price_change", "turnover_rate"]
NUMERIC_COLUMNS = [*ESSENTIAL_NUMERIC_COLUMNS, *OPTIONAL_NUMERIC_COLUMNS]


@dataclass(frozen=True, slots=True)
class OhlcvCleanResult:
    """Cleaned OHLCV data with row-level quality statistics."""

    data: pd.DataFrame
    stats: dict[str, int]


def clean_ohlcv(raw: pd.DataFrame) -> OhlcvCleanResult:
    """Clean raw OHLCV rows before persistence to processed storage.

    Day 4 deliberately handles only row-level data quality:
    missing values, duplicate symbol/date rows, invalid prices, and no-trade
    rows that represent suspended or effectively suspended sessions.
    Trading-calendar alignment is a separate Day 5 concern.
    """

    frame = _with_required_columns(raw)
    input_rows = len(frame)

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["symbol"] = frame["symbol"].astype("string").str.strip().str.zfill(6)
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    invalid_identity_mask = frame["date"].isna() | frame["symbol"].isna() | (frame["symbol"] == "")
    invalid_identity_rows = int(invalid_identity_mask.sum())
    frame = frame.loc[~invalid_identity_mask].copy()

    duplicate_rows = int(frame.duplicated(["symbol", "date"], keep="last").sum())
    frame = frame.drop_duplicates(["symbol", "date"], keep="last")

    essential_missing_mask = frame[ESSENTIAL_NUMERIC_COLUMNS].isna().any(axis=1)
    missing_essential_rows = int(essential_missing_mask.sum())
    frame = frame.loc[~essential_missing_mask].copy()

    invalid_price_mask = (
        (frame[ESSENTIAL_PRICE_COLUMNS] <= 0).any(axis=1)
        | (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
        | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
    )
    invalid_price_rows = int(invalid_price_mask.sum())
    frame = frame.loc[~invalid_price_mask].copy()

    suspended_mask = (frame["volume"] <= 0) | (frame["amount"] <= 0)
    suspended_rows = int(suspended_mask.sum())
    frame = frame.loc[~suspended_mask].copy()

    optional_missing_cells = int(frame[OPTIONAL_NUMERIC_COLUMNS].isna().sum().sum())
    frame.loc[:, OPTIONAL_NUMERIC_COLUMNS] = frame[OPTIONAL_NUMERIC_COLUMNS].fillna(0.0)

    cleaned = frame[OHLCV_COLUMNS].sort_values(["symbol", "date"]).reset_index(drop=True)
    stats = {
        "input_rows": input_rows,
        "output_rows": len(cleaned),
        "dropped_rows": input_rows - len(cleaned),
        "invalid_identity_rows": invalid_identity_rows,
        "duplicate_rows": duplicate_rows,
        "missing_essential_rows": missing_essential_rows,
        "invalid_price_rows": invalid_price_rows,
        "suspended_rows": suspended_rows,
        "filled_optional_missing_cells": optional_missing_cells,
    }
    return OhlcvCleanResult(data=cleaned, stats=stats)


def _with_required_columns(raw: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in OHLCV_COLUMNS if column not in raw.columns]
    if missing_columns:
        msg = f"OHLCV data missing required columns: {missing_columns}."
        raise ValueError(msg)
    return raw[OHLCV_COLUMNS].copy()

