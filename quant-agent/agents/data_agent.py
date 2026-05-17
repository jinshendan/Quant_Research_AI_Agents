from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from core.config import AppConfig
from core.logging import AgentLoggerAdapter, get_agent_logger
from core.models import AgentRequest, AgentResponse

from agents.market_data_provider import (
    AkShareMarketDataProvider,
    MarketDataProvider,
    combine_ohlcv_frames,
)

SUPPORTED_FREQUENCIES = {"daily"}
SUPPORTED_PROVIDERS = {"akshare"}
SUPPORTED_ADJUSTMENTS = {"", "qfq", "hfq"}

_SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class MarketDataSpec:
    """Validated market data request for DataAgent."""

    universe: str
    start_date: date
    end_date: date
    frequency: str = "daily"
    provider: str = "akshare"
    symbols: tuple[str, ...] = ()
    adjust: str = ""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> MarketDataSpec:
        universe = _required_str(payload, "universe")
        start_date = _parse_date(payload.get("start_date"), "start_date")
        end_date = _parse_date(payload.get("end_date"), "end_date")
        frequency = _optional_str(payload, "frequency", "daily")
        provider = _optional_str(payload, "provider", "akshare")
        symbols = _optional_symbols(payload)
        adjust = _optional_adjust(payload)

        if end_date < start_date:
            msg = "end_date must be greater than or equal to start_date."
            raise ValueError(msg)
        if frequency not in SUPPORTED_FREQUENCIES:
            msg = f"Unsupported frequency: {frequency}."
            raise ValueError(msg)
        if provider not in SUPPORTED_PROVIDERS:
            msg = f"Unsupported provider: {provider}."
            raise ValueError(msg)
        if adjust not in SUPPORTED_ADJUSTMENTS:
            msg = f"Unsupported adjust: {adjust}."
            raise ValueError(msg)

        return cls(
            universe=universe,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            provider=provider,
            symbols=symbols,
            adjust=adjust,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "universe": self.universe,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "frequency": self.frequency,
            "provider": self.provider,
            "symbols": list(self.symbols),
            "adjust": self.adjust,
        }


class DataAgent:
    """Download raw market data through a provider boundary."""

    name = "DataAgent"

    def __init__(
        self,
        *,
        config: AppConfig | None = None,
        logger: AgentLoggerAdapter | None = None,
        providers: Mapping[str, MarketDataProvider] | None = None,
    ) -> None:
        self.config = config or AppConfig.from_env()
        self.logger = logger or get_agent_logger(self.name)
        self.providers = dict(providers) if providers is not None else {
            "akshare": AkShareMarketDataProvider()
        }

    def run(self, request: AgentRequest) -> AgentResponse:
        started_at = perf_counter()
        self.logger.info(
            "Received market data request.",
            extra={"action": "validate_request", "status": "running"},
        )

        try:
            spec = MarketDataSpec.from_payload(request.payload)
            self.config.ensure_directories()
            provider = self._provider_for(spec.provider)
            symbols = list(spec.symbols) or provider.resolve_symbols(spec.universe)
        except (RuntimeError, ValueError) as exc:
            elapsed = perf_counter() - started_at
            self.logger.warning(
                "Market data request validation failed.",
                extra={"action": "validate_request", "status": "error"},
            )
            return AgentResponse.failure(
                str(exc),
                metadata=self._metadata(request, elapsed),
            )

        self.logger.info(
            "Resolved market data symbols.",
            extra={"action": "resolve_symbols", "status": "success"},
        )

        try:
            market_data = self.download_ohlcv(spec, symbols=symbols, provider=provider)
            raw_data_path = self.save_raw_ohlcv(market_data, spec)
        except Exception as exc:
            elapsed = perf_counter() - started_at
            self.logger.exception(
                "Market data download failed.",
                extra={"action": "download_ohlcv", "status": "error"},
            )
            return AgentResponse.failure(
                str(exc),
                metadata=self._metadata(
                    request,
                    elapsed,
                    provider=spec.provider,
                    frequency=spec.frequency,
                    symbols_count=len(symbols),
                ),
            )

        elapsed = perf_counter() - started_at
        self.logger.info(
            "Downloaded and stored raw OHLCV data.",
            extra={"action": "download_ohlcv", "status": "success"},
        )
        return AgentResponse.success(
            output={
                "state": "downloaded",
                "request": spec.to_dict(),
                "symbols": symbols,
                "rows": len(market_data),
                "columns": list(market_data.columns),
                "raw_data_path": str(raw_data_path),
                "next_action": "Clean missing values and handle suspended stocks in Day 4.",
            },
            metadata=self._metadata(
                request,
                elapsed,
                provider=spec.provider,
                frequency=spec.frequency,
                symbols_count=len(symbols),
                rows=len(market_data),
                raw_data_path=raw_data_path,
            ),
        )

    def download_ohlcv(
        self,
        spec: MarketDataSpec,
        *,
        symbols: Sequence[str] | None = None,
        provider: MarketDataProvider | None = None,
    ) -> pd.DataFrame:
        selected_provider = provider or self._provider_for(spec.provider)
        selected_symbols = list(symbols) if symbols is not None else list(spec.symbols)
        if not selected_symbols:
            selected_symbols = selected_provider.resolve_symbols(spec.universe)

        frames = []
        for symbol in selected_symbols:
            self.logger.info(
                f"Downloading OHLCV for {symbol}.",
                extra={"action": "download_symbol_ohlcv", "status": "running"},
            )
            frames.append(
                selected_provider.download_symbol_ohlcv(
                    symbol=symbol,
                    start_date=spec.start_date,
                    end_date=spec.end_date,
                    frequency=spec.frequency,
                    adjust=spec.adjust,
                )
            )

        return combine_ohlcv_frames(frames).reset_index(drop=True)

    def save_raw_ohlcv(self, market_data: pd.DataFrame, spec: MarketDataSpec) -> Path:
        output_path = self._raw_ohlcv_path(spec)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        market_data.to_csv(output_path, index=False, date_format="%Y-%m-%d")
        return output_path

    def _metadata(
        self,
        request: AgentRequest,
        elapsed: float,
        *,
        provider: str | None = None,
        frequency: str | None = None,
        symbols_count: int | None = None,
        rows: int | None = None,
        raw_data_path: Path | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "agent": self.name,
            "task_id": request.task_id,
            "execution_time_sec": round(elapsed, 6),
            "raw_data_dir": str(self.config.raw_data_dir),
            "processed_data_dir": str(self.config.processed_data_dir),
            "cache_dir": str(self.config.cache_dir),
        }
        if provider is not None:
            metadata["provider"] = provider
        if frequency is not None:
            metadata["frequency"] = frequency
        if symbols_count is not None:
            metadata["symbols_count"] = symbols_count
        if rows is not None:
            metadata["rows"] = rows
        if raw_data_path is not None:
            metadata["raw_data_path"] = str(raw_data_path)
        return metadata

    def _provider_for(self, provider_name: str) -> MarketDataProvider:
        provider = self.providers.get(provider_name)
        if provider is None:
            msg = f"No market data provider configured for: {provider_name}."
            raise RuntimeError(msg)
        return provider

    def _raw_ohlcv_path(self, spec: MarketDataSpec) -> Path:
        safe_universe = _safe_filename(spec.universe)
        adjustment = spec.adjust or "none"
        filename = (
            f"ohlcv_{spec.provider}_{safe_universe}_{spec.frequency}_"
            f"{adjustment}_{spec.start_date:%Y%m%d}_{spec.end_date:%Y%m%d}.csv"
        )
        return self.config.raw_data_dir / filename


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        msg = f"payload.{key} must be a non-empty string."
        raise ValueError(msg)
    return value.strip()


def _optional_str(payload: Mapping[str, Any], key: str, default: str) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str) or not value.strip():
        msg = f"payload.{key} must be a non-empty string."
        raise ValueError(msg)
    return value.strip().lower()


def _optional_adjust(payload: Mapping[str, Any]) -> str:
    value = payload.get("adjust", "")
    if not isinstance(value, str):
        msg = "payload.adjust must be a string."
        raise ValueError(msg)
    return value.strip().lower()


def _optional_symbols(payload: Mapping[str, Any]) -> tuple[str, ...]:
    value = payload.get("symbols")
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        msg = "payload.symbols must be a sequence of stock-code strings."
        raise ValueError(msg)

    symbols = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            msg = "payload.symbols must contain only non-empty strings."
            raise ValueError(msg)
        symbol = item.strip().upper()
        if not (len(symbol) == 6 and symbol.isdigit()):
            msg = f"Invalid stock symbol: {item}."
            raise ValueError(msg)
        symbols.append(symbol)

    if not symbols:
        msg = "payload.symbols must not be empty when provided."
        raise ValueError(msg)
    return tuple(dict.fromkeys(symbols))


def _parse_date(value: Any, key: str) -> date:
    if not isinstance(value, str):
        msg = f"payload.{key} must be an ISO date string."
        raise ValueError(msg)
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        msg = f"payload.{key} must be an ISO date string."
        raise ValueError(msg) from exc


def _safe_filename(value: str) -> str:
    cleaned = _SAFE_FILENAME_PATTERN.sub("_", value.strip())
    return cleaned.strip("._") or "universe"
