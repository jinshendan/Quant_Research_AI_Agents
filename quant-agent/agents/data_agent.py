from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

import pandas as pd

from core.config import AppConfig
from core.logging import AgentLoggerAdapter, get_agent_logger
from core.models import AgentRequest, AgentResponse

from agents.duckdb_store import (
    DuckDBMarketDataStore,
    MarketDataStorageContext,
    MarketDataStorageResult,
)
from agents.market_data_cache import (
    MarketDataCache,
    MarketDataCacheIdentity,
    MarketDataCacheLookup,
)
from agents.market_data_provider import (
    AkShareMarketDataProvider,
    MarketDataProvider,
    combine_ohlcv_frames,
)
from agents.ohlcv_cleaner import clean_ohlcv
from agents.trading_calendar import (
    AkShareTradingCalendarProvider,
    TradingCalendarProvider,
    align_to_trading_calendar,
)

SUPPORTED_FREQUENCIES = {"daily"}
SUPPORTED_PROVIDERS = {"akshare"}
SUPPORTED_ADJUSTMENTS = {"", "qfq", "hfq"}
DEFAULT_DOWNLOAD_MAX_RETRIES = 2
DEFAULT_DOWNLOAD_RETRY_BACKOFF_SEC = 0.5
MAX_DOWNLOAD_RETRIES = 5
MAX_DOWNLOAD_RETRY_BACKOFF_SEC = 60.0
DOWNLOAD_FAILURE_MANIFEST_SCHEMA_VERSION = 1

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
    use_cache: bool = True
    force_refresh: bool = False
    max_retries: int = DEFAULT_DOWNLOAD_MAX_RETRIES
    retry_backoff_sec: float = DEFAULT_DOWNLOAD_RETRY_BACKOFF_SEC
    continue_on_symbol_error: bool = True

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> MarketDataSpec:
        universe = _required_str(payload, "universe")
        start_date = _parse_date(payload.get("start_date"), "start_date")
        end_date = _parse_date(payload.get("end_date"), "end_date")
        frequency = _optional_str(payload, "frequency", "daily")
        provider = _optional_str(payload, "provider", "akshare")
        symbols = _optional_symbols(payload)
        adjust = _optional_adjust(payload)
        use_cache = _optional_bool(payload, "use_cache", True)
        force_refresh = _optional_bool_alias(payload, ("force_refresh", "refresh_cache"), False)
        max_retries = _optional_int(
            payload,
            "max_retries",
            DEFAULT_DOWNLOAD_MAX_RETRIES,
            minimum=0,
            maximum=MAX_DOWNLOAD_RETRIES,
        )
        retry_backoff_sec = _optional_float(
            payload,
            "retry_backoff_sec",
            DEFAULT_DOWNLOAD_RETRY_BACKOFF_SEC,
            minimum=0.0,
            maximum=MAX_DOWNLOAD_RETRY_BACKOFF_SEC,
        )
        continue_on_symbol_error = _optional_bool(
            payload,
            "continue_on_symbol_error",
            True,
        )

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
            use_cache=use_cache,
            force_refresh=force_refresh,
            max_retries=max_retries,
            retry_backoff_sec=retry_backoff_sec,
            continue_on_symbol_error=continue_on_symbol_error,
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
            "use_cache": self.use_cache,
            "force_refresh": self.force_refresh,
            "max_retries": self.max_retries,
            "retry_backoff_sec": self.retry_backoff_sec,
            "continue_on_symbol_error": self.continue_on_symbol_error,
        }


@dataclass(frozen=True, slots=True)
class SymbolDownloadFailure:
    """One failed symbol download after all retry attempts."""

    symbol: str
    attempts: int
    error_type: str
    error_message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "attempts": self.attempts,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclass(frozen=True, slots=True)
class OhlcvDownloadResult:
    """Downloaded OHLCV rows plus symbol-level reliability details."""

    data: pd.DataFrame
    requested_symbols: tuple[str, ...]
    successful_symbols: tuple[str, ...]
    failures: tuple[SymbolDownloadFailure, ...]
    retry_attempts: int = 0

    @property
    def failed_symbols(self) -> tuple[str, ...]:
        return tuple(failure.symbol for failure in self.failures)

    def stats(self) -> dict[str, Any]:
        return {
            "requested_symbol_count": len(self.requested_symbols),
            "successful_symbol_count": len(self.successful_symbols),
            "failed_symbol_count": len(self.failures),
            "requested_symbols": list(self.requested_symbols),
            "successful_symbols": list(self.successful_symbols),
            "failed_symbols": list(self.failed_symbols),
            "retry_attempts": self.retry_attempts,
            "failures": [failure.to_dict() for failure in self.failures],
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
        calendar_providers: Mapping[str, TradingCalendarProvider] | None = None,
        market_data_store: DuckDBMarketDataStore | None = None,
        market_data_cache: MarketDataCache | None = None,
    ) -> None:
        self.config = config or AppConfig.from_env()
        self.logger = logger or get_agent_logger(self.name)
        self.providers = dict(providers) if providers is not None else {
            "akshare": AkShareMarketDataProvider()
        }
        self.calendar_providers = (
            dict(calendar_providers)
            if calendar_providers is not None
            else {"akshare": AkShareTradingCalendarProvider()}
        )
        self.market_data_store = market_data_store or DuckDBMarketDataStore(self.config.duckdb_path)
        self.market_data_cache = market_data_cache or MarketDataCache(self.config.cache_dir)

    def run(self, request: AgentRequest) -> AgentResponse:
        started_at = perf_counter()
        self.logger.info(
            "Received market data request.",
            extra={"action": "validate_request", "status": "running"},
        )

        try:
            spec = MarketDataSpec.from_payload(request.payload)
            self.config.ensure_directories()
            cache_identity = self._cache_identity(spec)
            cache_lookup = self._lookup_cache(spec, cache_identity)
            if cache_lookup is not None and cache_lookup.hit:
                elapsed = perf_counter() - started_at
                self.logger.info(
                    "Market data cache hit.",
                    extra={"action": "cache_lookup", "status": "hit"},
                )
                output = self._cached_output(cache_lookup)
                return AgentResponse.success(
                    output=output,
                    metadata=self._metadata(
                        request,
                        elapsed,
                        provider=spec.provider,
                        frequency=spec.frequency,
                        symbols_count=len(output.get("symbols", [])),
                        rows=output.get("aligned_rows"),
                        raw_data_path=_optional_path(output.get("raw_data_path")),
                        processed_data_path=_optional_path(output.get("processed_data_path")),
                        aligned_data_path=_optional_path(output.get("aligned_data_path")),
                        cleaning_stats=output.get("cleaning_stats"),
                        calendar_stats=output.get("calendar_stats"),
                        storage_stats=output.get("storage_stats"),
                        cache_stats=output.get("cache_stats"),
                        download_stats=output.get("download_stats"),
                        failure_manifest_path=_optional_path(
                            output.get("failure_manifest_path")
                        ),
                    ),
                )

            provider = self._provider_for(spec.provider)
            calendar_provider = self._calendar_provider_for(spec.provider)
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
            download_result = self.download_ohlcv_with_reliability(
                spec,
                symbols=symbols,
                provider=provider,
            )
            market_data = download_result.data
            failure_manifest_path = self.save_download_failure_manifest(
                download_result,
                spec,
                task_id=request.task_id,
            )
            raw_data_path = self.save_raw_ohlcv(market_data, spec)
            clean_result = clean_ohlcv(market_data)
            processed_data_path = self.save_processed_ohlcv(clean_result.data, spec)
            trading_days = calendar_provider.get_trading_days(
                start_date=spec.start_date,
                end_date=spec.end_date,
            )
            calendar_result = align_to_trading_calendar(
                clean_result.data,
                symbols=download_result.successful_symbols,
                trading_days=trading_days,
            )
            aligned_data_path = self.save_aligned_ohlcv(calendar_result.data, spec)
            storage_result = self.market_data_store.store_aligned_ohlcv(
                calendar_result.data,
                context=MarketDataStorageContext(
                    run_id=request.task_id,
                    task_id=request.task_id,
                    universe=spec.universe,
                    provider=spec.provider,
                    frequency=spec.frequency,
                    adjust=spec.adjust,
                    start_date=spec.start_date.isoformat(),
                    end_date=spec.end_date.isoformat(),
                    raw_data_path=str(raw_data_path),
                    processed_data_path=str(processed_data_path),
                    aligned_data_path=str(aligned_data_path),
                    raw_rows=len(market_data),
                    processed_rows=len(clean_result.data),
                    aligned_rows=len(calendar_result.data),
                    cleaning_stats=clean_result.stats,
                    calendar_stats=calendar_result.stats,
                ),
            )
            cache_stats = self.market_data_cache.disabled_stats(cache_identity)
        except Exception as exc:
            elapsed = perf_counter() - started_at
            self.logger.exception(
                "Market data preparation or storage failed.",
                extra={"action": "prepare_ohlcv", "status": "error"},
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
            "Downloaded, cleaned, aligned, and persisted OHLCV data.",
            extra={"action": "prepare_ohlcv", "status": "success"},
        )
        output = {
            "state": "stored",
            "request": spec.to_dict(),
            "symbols": symbols,
            "successful_symbols": list(download_result.successful_symbols),
            "failed_symbols": list(download_result.failed_symbols),
            "raw_rows": len(market_data),
            "processed_rows": len(clean_result.data),
            "aligned_rows": len(calendar_result.data),
            "columns": list(calendar_result.data.columns),
            "raw_data_path": str(raw_data_path),
            "processed_data_path": str(processed_data_path),
            "aligned_data_path": str(aligned_data_path),
            "cleaning_stats": clean_result.stats,
            "calendar_stats": calendar_result.stats,
            "download_stats": download_result.stats(),
            "failure_manifest_path": (
                str(failure_manifest_path) if failure_manifest_path else None
            ),
            "storage_stats": storage_result.to_dict(),
            "next_action": "Build HypothesisAgent in Day 8.",
        }
        if spec.use_cache and download_result.failures:
            output["cache_stats"] = self.market_data_cache.skipped_stats(
                cache_identity,
                reason="partial_download",
            )
            self.logger.info(
                "Market data cache skipped for partial download.",
                extra={"action": "cache_write", "status": "skipped"},
            )
        elif spec.use_cache:
            try:
                cache_entry = self.market_data_cache.store(cache_identity, output)
                output = cache_entry.output
            except Exception as exc:
                elapsed = perf_counter() - started_at
                self.logger.exception(
                    "Market data cache write failed.",
                    extra={"action": "cache_write", "status": "error"},
                )
                return AgentResponse.failure(
                    str(exc),
                    metadata=self._metadata(
                        request,
                        elapsed,
                        provider=spec.provider,
                        frequency=spec.frequency,
                        symbols_count=len(symbols),
                        rows=len(calendar_result.data),
                        raw_data_path=raw_data_path,
                        processed_data_path=processed_data_path,
                        aligned_data_path=aligned_data_path,
                        cleaning_stats=clean_result.stats,
                        calendar_stats=calendar_result.stats,
                        download_stats=download_result.stats(),
                        failure_manifest_path=failure_manifest_path,
                        storage_result=storage_result,
                    ),
                )
            self.logger.info(
                "Market data cache refreshed.",
                extra={"action": "cache_write", "status": "success"},
            )
        else:
            output["cache_stats"] = cache_stats

        return AgentResponse.success(
            output=output,
            metadata=self._metadata(
                request,
                elapsed,
                provider=spec.provider,
                frequency=spec.frequency,
                symbols_count=len(symbols),
                rows=len(calendar_result.data),
                raw_data_path=raw_data_path,
                processed_data_path=processed_data_path,
                aligned_data_path=aligned_data_path,
                cleaning_stats=clean_result.stats,
                calendar_stats=calendar_result.stats,
                download_stats=download_result.stats(),
                failure_manifest_path=failure_manifest_path,
                storage_result=storage_result,
                cache_stats=output.get("cache_stats"),
            ),
        )

    def download_ohlcv(
        self,
        spec: MarketDataSpec,
        *,
        symbols: Sequence[str] | None = None,
        provider: MarketDataProvider | None = None,
    ) -> pd.DataFrame:
        return self.download_ohlcv_with_reliability(
            spec,
            symbols=symbols,
            provider=provider,
        ).data

    def download_ohlcv_with_reliability(
        self,
        spec: MarketDataSpec,
        *,
        symbols: Sequence[str] | None = None,
        provider: MarketDataProvider | None = None,
    ) -> OhlcvDownloadResult:
        selected_provider = provider or self._provider_for(spec.provider)
        selected_symbols = tuple(symbols if symbols is not None else spec.symbols)
        if not selected_symbols:
            selected_symbols = tuple(selected_provider.resolve_symbols(spec.universe))

        frames: list[pd.DataFrame] = []
        successful_symbols: list[str] = []
        failures: list[SymbolDownloadFailure] = []
        retry_attempts = 0
        for symbol in selected_symbols:
            frame, failure, attempts = self._download_symbol_with_retries(
                spec,
                symbol=symbol,
                provider=selected_provider,
            )
            retry_attempts += max(0, attempts - 1)
            if failure is not None:
                failures.append(failure)
                if not spec.continue_on_symbol_error:
                    msg = (
                        f"OHLCV download failed for {symbol} after "
                        f"{failure.attempts} attempts: {failure.error_message}"
                    )
                    raise RuntimeError(msg)
                continue
            if frame is not None:
                frames.append(frame)
                successful_symbols.append(symbol)

        if failures:
            self.logger.warning(
                "Some OHLCV symbol downloads failed.",
                extra={"action": "download_symbol_ohlcv", "status": "partial"},
            )
        if not frames:
            failed_symbols = ", ".join(failure.symbol for failure in failures) or "none"
            msg = f"No OHLCV data downloaded. Failed symbols: {failed_symbols}."
            raise RuntimeError(msg)

        return OhlcvDownloadResult(
            data=combine_ohlcv_frames(frames).reset_index(drop=True),
            requested_symbols=selected_symbols,
            successful_symbols=tuple(successful_symbols),
            failures=tuple(failures),
            retry_attempts=retry_attempts,
        )

    def save_download_failure_manifest(
        self,
        result: OhlcvDownloadResult,
        spec: MarketDataSpec,
        *,
        task_id: str,
    ) -> Path | None:
        if not result.failures:
            return None

        output_path = self._download_failure_manifest_path(spec, task_id=task_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "schema_version": DOWNLOAD_FAILURE_MANIFEST_SCHEMA_VERSION,
            "task_id": task_id,
            "request": spec.to_dict(),
            "download_stats": result.stats(),
        }
        temp_path = Path(f"{output_path}.tmp")
        temp_path.write_text(
            json.dumps(
                document,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        temp_path.replace(output_path)
        return output_path

    def _download_symbol_with_retries(
        self,
        spec: MarketDataSpec,
        *,
        symbol: str,
        provider: MarketDataProvider,
    ) -> tuple[pd.DataFrame | None, SymbolDownloadFailure | None, int]:
        max_attempts = spec.max_retries + 1
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            self.logger.info(
                f"Downloading OHLCV for {symbol}.",
                extra={"action": "download_symbol_ohlcv", "status": "running"},
            )
            try:
                frame = provider.download_symbol_ohlcv(
                    symbol=symbol,
                    start_date=spec.start_date,
                    end_date=spec.end_date,
                    frequency=spec.frequency,
                    adjust=spec.adjust,
                )
            except Exception as exc:  # noqa: BLE001 - provider boundary isolation.
                last_exc = exc
                if attempt < max_attempts:
                    self.logger.warning(
                        f"Retrying OHLCV download for {symbol}.",
                        extra={"action": "download_symbol_ohlcv", "status": "retry"},
                    )
                    backoff = spec.retry_backoff_sec * (2 ** (attempt - 1))
                    if backoff > 0:
                        sleep(backoff)
                    continue
                break
            else:
                if attempt > 1:
                    self.logger.info(
                        f"OHLCV download for {symbol} recovered after retry.",
                        extra={
                            "action": "download_symbol_ohlcv",
                            "status": "recovered",
                        },
                    )
                return frame, None, attempt

        if last_exc is None:
            last_exc = RuntimeError("Unknown provider failure.")
        failure = SymbolDownloadFailure(
            symbol=symbol,
            attempts=max_attempts,
            error_type=type(last_exc).__name__,
            error_message=str(last_exc),
        )
        self.logger.warning(
            f"OHLCV download failed for {symbol}.",
            extra={"action": "download_symbol_ohlcv", "status": "failed"},
        )
        return None, failure, max_attempts

    def save_raw_ohlcv(self, market_data: pd.DataFrame, spec: MarketDataSpec) -> Path:
        output_path = self._raw_ohlcv_path(spec)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        market_data.to_csv(output_path, index=False, date_format="%Y-%m-%d")
        return output_path

    def save_processed_ohlcv(self, clean_data: pd.DataFrame, spec: MarketDataSpec) -> Path:
        output_path = self._processed_ohlcv_path(spec)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        clean_data.to_csv(output_path, index=False, date_format="%Y-%m-%d")
        return output_path

    def save_aligned_ohlcv(self, aligned_data: pd.DataFrame, spec: MarketDataSpec) -> Path:
        output_path = self._aligned_ohlcv_path(spec)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        aligned_data.to_csv(output_path, index=False, date_format="%Y-%m-%d")
        return output_path

    def _lookup_cache(
        self,
        spec: MarketDataSpec,
        identity: MarketDataCacheIdentity,
    ) -> MarketDataCacheLookup | None:
        if not spec.use_cache:
            self.logger.info(
                "Market data cache disabled.",
                extra={"action": "cache_lookup", "status": "disabled"},
            )
            return None
        if spec.force_refresh:
            self.logger.info(
                "Market data cache bypassed by force_refresh.",
                extra={"action": "cache_lookup", "status": "bypassed"},
            )
            return None

        lookup = self.market_data_cache.lookup(identity)
        if not lookup.hit:
            self.logger.info(
                f"Market data cache {lookup.status}: {lookup.reason}.",
                extra={"action": "cache_lookup", "status": lookup.status},
            )
        return lookup

    def _cache_identity(self, spec: MarketDataSpec) -> MarketDataCacheIdentity:
        return MarketDataCacheIdentity(
            universe=spec.universe,
            provider=spec.provider,
            frequency=spec.frequency,
            adjust=spec.adjust,
            start_date=spec.start_date.isoformat(),
            end_date=spec.end_date.isoformat(),
            symbols=spec.symbols,
        )

    def _cached_output(self, lookup: MarketDataCacheLookup) -> dict[str, Any]:
        if lookup.entry is None:
            msg = "Cannot build cached output from a cache miss."
            raise RuntimeError(msg)

        output = copy.deepcopy(lookup.entry.output)
        output["state"] = "cached"
        output["cache_stats"] = lookup.stats()
        output["next_action"] = "Build HypothesisAgent in Day 8."
        return output

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
        processed_data_path: Path | None = None,
        aligned_data_path: Path | None = None,
        cleaning_stats: dict[str, int] | None = None,
        calendar_stats: dict[str, int] | None = None,
        storage_result: MarketDataStorageResult | None = None,
        storage_stats: Mapping[str, Any] | None = None,
        cache_stats: Mapping[str, Any] | None = None,
        download_stats: Mapping[str, Any] | None = None,
        failure_manifest_path: Path | None = None,
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
        if processed_data_path is not None:
            metadata["processed_data_path"] = str(processed_data_path)
        if aligned_data_path is not None:
            metadata["aligned_data_path"] = str(aligned_data_path)
        if cleaning_stats is not None:
            metadata["cleaning_stats"] = cleaning_stats
        if calendar_stats is not None:
            metadata["calendar_stats"] = calendar_stats
        if storage_result is not None:
            metadata["storage_stats"] = storage_result.to_dict()
        if storage_stats is not None:
            metadata["storage_stats"] = dict(storage_stats)
        if cache_stats is not None:
            metadata["cache_stats"] = dict(cache_stats)
        if download_stats is not None:
            metadata["download_stats"] = dict(download_stats)
        if failure_manifest_path is not None:
            metadata["failure_manifest_path"] = str(failure_manifest_path)
        return metadata

    def _provider_for(self, provider_name: str) -> MarketDataProvider:
        provider = self.providers.get(provider_name)
        if provider is None:
            msg = f"No market data provider configured for: {provider_name}."
            raise RuntimeError(msg)
        return provider

    def _calendar_provider_for(self, provider_name: str) -> TradingCalendarProvider:
        provider = self.calendar_providers.get(provider_name)
        if provider is None:
            msg = f"No trading calendar provider configured for: {provider_name}."
            raise RuntimeError(msg)
        return provider

    def _raw_ohlcv_path(self, spec: MarketDataSpec) -> Path:
        return self.config.raw_data_dir / self._ohlcv_filename(spec)

    def _processed_ohlcv_path(self, spec: MarketDataSpec) -> Path:
        return self.config.processed_data_dir / self._ohlcv_filename(spec)

    def _aligned_ohlcv_path(self, spec: MarketDataSpec) -> Path:
        return self.config.processed_data_dir / f"aligned_{self._ohlcv_filename(spec)}"

    def _download_failure_manifest_path(self, spec: MarketDataSpec, *, task_id: str) -> Path:
        safe_task_id = _safe_filename(task_id)
        ohlcv_stem = self._ohlcv_filename(spec).removesuffix(".csv")
        return self.config.data_dir / "failures" / (
            f"download_failures_{safe_task_id}_{ohlcv_stem}.json"
        )

    def _ohlcv_filename(self, spec: MarketDataSpec) -> str:
        safe_universe = _safe_filename(spec.universe)
        adjustment = spec.adjust or "none"
        return (
            f"ohlcv_{spec.provider}_{safe_universe}_{spec.frequency}_"
            f"{adjustment}_{spec.start_date:%Y%m%d}_{spec.end_date:%Y%m%d}.csv"
        )


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


def _optional_int(
    payload: Mapping[str, Any],
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"payload.{key} must be an integer."
        raise ValueError(msg)
    if value < minimum or value > maximum:
        msg = f"payload.{key} must be between {minimum} and {maximum}."
        raise ValueError(msg)
    return value


def _optional_float(
    payload: Mapping[str, Any],
    key: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"payload.{key} must be a finite number."
        raise ValueError(msg)
    normalized = float(value)
    if not math.isfinite(normalized):
        msg = f"payload.{key} must be a finite number."
        raise ValueError(msg)
    if normalized < minimum or normalized > maximum:
        msg = f"payload.{key} must be between {minimum} and {maximum}."
        raise ValueError(msg)
    return normalized


def _optional_bool(payload: Mapping[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        msg = f"payload.{key} must be a boolean."
        raise ValueError(msg)
    return value


def _optional_bool_alias(
    payload: Mapping[str, Any],
    keys: Sequence[str],
    default: bool,
) -> bool:
    for key in keys:
        if key in payload:
            return _optional_bool(payload, key, default)
    return default


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


def _optional_path(value: Any) -> Path | None:
    if isinstance(value, str) and value:
        return Path(value)
    return None
