from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

import pandas as pd

OHLCV_COLUMNS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "amplitude",
    "pct_change",
    "price_change",
    "turnover_rate",
]

AKSHARE_COLUMN_MAP = {
    "日期": "date",
    "股票代码": "symbol",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "price_change",
    "换手率": "turnover_rate",
}

INDEX_ALIASES = {
    "CSI300": "000300",
    "CSI500": "000905",
    "CSI1000": "000852",
    "SSE50": "000016",
}


class MarketDataProvider(Protocol):
    """External market data provider boundary used by DataAgent."""

    name: str

    def resolve_symbols(self, universe: str) -> list[str]:
        """Resolve a universe name into stock symbols."""
        ...

    def download_symbol_ohlcv(
        self,
        *,
        symbol: str,
        start_date: date,
        end_date: date,
        frequency: str,
        adjust: str,
    ) -> pd.DataFrame:
        """Download one symbol's OHLCV data."""
        ...


@dataclass(slots=True)
class AkShareMarketDataProvider:
    """AkShare-backed A-share daily OHLCV provider."""

    timeout: float = 15.0
    stock_hist_func: Callable[..., pd.DataFrame] | None = None
    index_cons_func: Callable[..., pd.DataFrame] | None = None

    name: str = "akshare"

    def __post_init__(self) -> None:
        if self.stock_hist_func is None or self.index_cons_func is None:
            import akshare as ak  # type: ignore[import-untyped]

            if self.stock_hist_func is None:
                self.stock_hist_func = ak.stock_zh_a_hist
            if self.index_cons_func is None:
                self.index_cons_func = ak.index_stock_cons_csindex

    def resolve_symbols(self, universe: str) -> list[str]:
        normalized = universe.strip().upper()
        if _is_stock_symbol(normalized):
            return [normalized]

        index_symbol = INDEX_ALIASES.get(normalized)
        if index_symbol is None:
            msg = (
                f"Unsupported universe: {universe}. "
                "Provide payload.symbols or use one of CSI300, CSI500, CSI1000, SSE50."
            )
            raise ValueError(msg)

        if self.index_cons_func is None:
            msg = "AkShare index constituent function is not configured."
            raise RuntimeError(msg)

        constituents = self.index_cons_func(symbol=index_symbol)
        if "成分券代码" not in constituents.columns:
            msg = "AkShare index constituents are missing column: 成分券代码."
            raise ValueError(msg)

        symbols = (
            constituents["成分券代码"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.zfill(6)
            .drop_duplicates()
            .tolist()
        )
        if not symbols:
            msg = f"No symbols resolved for universe: {universe}."
            raise ValueError(msg)
        return symbols

    def download_symbol_ohlcv(
        self,
        *,
        symbol: str,
        start_date: date,
        end_date: date,
        frequency: str,
        adjust: str,
    ) -> pd.DataFrame:
        if self.stock_hist_func is None:
            msg = "AkShare stock history function is not configured."
            raise RuntimeError(msg)

        raw = self.stock_hist_func(
            symbol=symbol,
            period=frequency,
            start_date=_akshare_date(start_date),
            end_date=_akshare_date(end_date),
            adjust=adjust,
            timeout=self.timeout,
        )
        return normalize_akshare_ohlcv(raw, symbol=symbol)


def normalize_akshare_ohlcv(raw: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    """Normalize AkShare's Chinese OHLCV columns into the project schema."""

    if raw.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    frame = raw.rename(columns=AKSHARE_COLUMN_MAP).copy()
    missing_columns = [column for column in OHLCV_COLUMNS if column not in frame.columns]
    if missing_columns:
        msg = f"AkShare OHLCV data missing columns: {missing_columns}."
        raise ValueError(msg)

    frame = frame[OHLCV_COLUMNS]
    frame["symbol"] = frame["symbol"].fillna(symbol).astype(str).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

    numeric_columns = [column for column in OHLCV_COLUMNS if column not in {"date", "symbol"}]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame.sort_values(["symbol", "date"]).reset_index(drop=True)


def combine_ohlcv_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    return pd.concat(non_empty, ignore_index=True).sort_values(["symbol", "date"])


def _akshare_date(value: date) -> str:
    return value.strftime("%Y%m%d")


def _is_stock_symbol(value: str) -> bool:
    return len(value) == 6 and value.isdigit()
