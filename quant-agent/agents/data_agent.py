from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from time import perf_counter
from typing import Any

from core.config import AppConfig
from core.logging import AgentLoggerAdapter, get_agent_logger
from core.models import AgentRequest, AgentResponse

SUPPORTED_FREQUENCIES = {"daily"}
SUPPORTED_PROVIDERS = {"akshare", "tushare"}


@dataclass(frozen=True, slots=True)
class MarketDataSpec:
    """Validated market data request for DataAgent."""

    universe: str
    start_date: date
    end_date: date
    frequency: str = "daily"
    provider: str = "akshare"

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> MarketDataSpec:
        universe = _required_str(payload, "universe")
        start_date = _parse_date(payload.get("start_date"), "start_date")
        end_date = _parse_date(payload.get("end_date"), "end_date")
        frequency = _optional_str(payload, "frequency", "daily")
        provider = _optional_str(payload, "provider", "akshare")

        if end_date < start_date:
            msg = "end_date must be greater than or equal to start_date."
            raise ValueError(msg)
        if frequency not in SUPPORTED_FREQUENCIES:
            msg = f"Unsupported frequency: {frequency}."
            raise ValueError(msg)
        if provider not in SUPPORTED_PROVIDERS:
            msg = f"Unsupported provider: {provider}."
            raise ValueError(msg)

        return cls(
            universe=universe,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            provider=provider,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "universe": self.universe,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "frequency": self.frequency,
            "provider": self.provider,
        }


class DataAgent:
    """Validate and prepare market-data work.

    Provider integration starts on Day 3. This skeleton deliberately keeps the
    external data boundary explicit so real downloads can be added behind a
    tested interface.
    """

    name = "DataAgent"

    def __init__(
        self,
        *,
        config: AppConfig | None = None,
        logger: AgentLoggerAdapter | None = None,
    ) -> None:
        self.config = config or AppConfig.from_env()
        self.logger = logger or get_agent_logger(self.name)

    def run(self, request: AgentRequest) -> AgentResponse:
        started_at = perf_counter()
        self.logger.info(
            "Received market data request.",
            extra={"action": "validate_request", "status": "running"},
        )

        try:
            spec = MarketDataSpec.from_payload(request.payload)
            self.config.ensure_directories()
        except ValueError as exc:
            elapsed = perf_counter() - started_at
            self.logger.warning(
                "Market data request validation failed.",
                extra={"action": "validate_request", "status": "error"},
            )
            return AgentResponse.failure(
                str(exc),
                metadata=self._metadata(request, elapsed),
            )

        elapsed = perf_counter() - started_at
        self.logger.info(
            "Market data request validated.",
            extra={"action": "validate_request", "status": "success"},
        )
        return AgentResponse.success(
            output={
                "state": "validated",
                "market_data": None,
                "request": spec.to_dict(),
                "next_action": "Integrate OHLCV provider in Day 3.",
            },
            metadata=self._metadata(
                request,
                elapsed,
                provider=spec.provider,
                frequency=spec.frequency,
            ),
        )

    def download_ohlcv(self, spec: MarketDataSpec) -> Any:
        raise NotImplementedError("OHLCV download is scheduled for Day 3.")

    def _metadata(
        self,
        request: AgentRequest,
        elapsed: float,
        *,
        provider: str | None = None,
        frequency: str | None = None,
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
        return metadata


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


def _parse_date(value: Any, key: str) -> date:
    if not isinstance(value, str):
        msg = f"payload.{key} must be an ISO date string."
        raise ValueError(msg)
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        msg = f"payload.{key} must be an ISO date string."
        raise ValueError(msg) from exc

