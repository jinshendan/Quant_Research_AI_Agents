from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from agents.market_data_provider import (
    AkShareMarketDataProvider,
    OHLCV_COLUMNS,
    normalize_akshare_daily_ohlcv,
    normalize_akshare_ohlcv,
)


def test_normalize_akshare_ohlcv_maps_columns_and_types() -> None:
    raw = pd.DataFrame(
        [
            {
                "日期": "2020-01-02",
                "股票代码": "1",
                "开盘": "10.1",
                "收盘": "10.5",
                "最高": "10.8",
                "最低": "9.9",
                "成交量": "1000",
                "成交额": "10500",
                "振幅": "5.0",
                "涨跌幅": "1.2",
                "涨跌额": "0.1",
                "换手率": "0.8",
            }
        ]
    )

    normalized = normalize_akshare_ohlcv(raw, symbol="000001")

    assert list(normalized.columns) == OHLCV_COLUMNS
    assert normalized.iloc[0]["symbol"] == "000001"
    assert normalized.iloc[0]["date"] == pd.Timestamp("2020-01-02")
    assert normalized.iloc[0]["open"] == 10.1
    assert normalized.iloc[0]["close"] == 10.5


def test_normalize_akshare_ohlcv_rejects_missing_columns() -> None:
    raw = pd.DataFrame([{"日期": "2020-01-02"}])

    with pytest.raises(ValueError, match="missing columns"):
        normalize_akshare_ohlcv(raw, symbol="000001")


def test_normalize_akshare_daily_ohlcv_computes_missing_optional_fields() -> None:
    raw = pd.DataFrame(
        [
            {
                "date": "2020-01-02",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 1000,
                "amount": 10500,
                "turnover": 0.01,
            },
            {
                "date": "2020-01-03",
                "open": 10.5,
                "high": 12.0,
                "low": 10.0,
                "close": 11.55,
                "volume": 1200,
                "amount": 13860,
                "turnover": 0.02,
            },
        ]
    )

    normalized = normalize_akshare_daily_ohlcv(raw, symbol="000001")

    assert list(normalized.columns) == OHLCV_COLUMNS
    assert normalized.iloc[0]["symbol"] == "000001"
    assert normalized.iloc[0]["turnover_rate"] == 1.0
    assert pd.isna(normalized.iloc[0]["pct_change"])
    assert normalized.iloc[1]["price_change"] == pytest.approx(1.05)
    assert normalized.iloc[1]["pct_change"] == pytest.approx(10.0)
    assert normalized.iloc[1]["amplitude"] == pytest.approx(19.047619047619047)


def test_akshare_provider_resolves_stock_symbol_directly() -> None:
    provider = AkShareMarketDataProvider(
        stock_hist_func=lambda **_: pd.DataFrame(),
        index_cons_func=lambda **_: pd.DataFrame(),
    )

    assert provider.resolve_symbols("000001") == ["000001"]


def test_akshare_provider_resolves_known_index_alias() -> None:
    def fake_index_cons_func(**_: object) -> pd.DataFrame:
        return pd.DataFrame({"成分券代码": ["1", "000002", "000002"]})

    provider = AkShareMarketDataProvider(
        stock_hist_func=lambda **_: pd.DataFrame(),
        index_cons_func=fake_index_cons_func,
    )

    assert provider.resolve_symbols("CSI500") == ["000001", "000002"]


def test_akshare_provider_downloads_and_normalizes_symbol_history() -> None:
    calls: list[dict[str, object]] = []

    def fake_stock_hist_func(**kwargs: object) -> pd.DataFrame:
        calls.append(kwargs)
        return pd.DataFrame(
            [
                {
                    "日期": "2020-01-02",
                    "股票代码": "000001",
                    "开盘": 10.1,
                    "收盘": 10.5,
                    "最高": 10.8,
                    "最低": 9.9,
                    "成交量": 1000,
                    "成交额": 10500,
                    "振幅": 5.0,
                    "涨跌幅": 1.2,
                    "涨跌额": 0.1,
                    "换手率": 0.8,
                }
            ]
        )

    provider = AkShareMarketDataProvider(
        timeout=3.0,
        stock_hist_func=fake_stock_hist_func,
        index_cons_func=lambda **_: pd.DataFrame(),
    )

    data = provider.download_symbol_ohlcv(
        symbol="000001",
        start_date=date(2020, 1, 1),
        end_date=date(2020, 1, 31),
        frequency="daily",
        adjust="qfq",
    )

    assert calls == [
        {
            "symbol": "000001",
            "period": "daily",
            "start_date": "20200101",
            "end_date": "20200131",
            "adjust": "qfq",
            "timeout": 3.0,
        }
    ]
    assert list(data.columns) == OHLCV_COLUMNS
    assert len(data) == 1


def test_akshare_provider_falls_back_to_daily_history() -> None:
    calls: list[dict[str, object]] = []

    def broken_stock_hist_func(**_: object) -> pd.DataFrame:
        raise ConnectionError("remote closed")

    def fake_stock_daily_func(**kwargs: object) -> pd.DataFrame:
        calls.append(kwargs)
        return pd.DataFrame(
            [
                {
                    "date": "2020-01-02",
                    "open": 10.1,
                    "high": 10.8,
                    "low": 9.9,
                    "close": 10.5,
                    "volume": 1000,
                    "amount": 10500,
                    "turnover": 0.008,
                }
            ]
        )

    provider = AkShareMarketDataProvider(
        timeout=3.0,
        stock_hist_func=broken_stock_hist_func,
        stock_daily_func=fake_stock_daily_func,
        index_cons_func=lambda **_: pd.DataFrame(),
    )

    data = provider.download_symbol_ohlcv(
        symbol="002126",
        start_date=date(2020, 1, 1),
        end_date=date(2020, 1, 31),
        frequency="daily",
        adjust="",
    )

    assert calls == [
        {
            "symbol": "sz002126",
            "start_date": "20200101",
            "end_date": "20200131",
            "adjust": "",
        }
    ]
    assert list(data.columns) == OHLCV_COLUMNS
    assert len(data) == 1
