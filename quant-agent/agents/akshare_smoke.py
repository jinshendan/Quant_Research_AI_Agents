from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from agents.data_agent import DataAgent
from agents.market_data_provider import AkShareMarketDataProvider, MarketDataProvider
from agents.trading_calendar import AkShareTradingCalendarProvider, TradingCalendarProvider
from core.config import AppConfig
from core.models import AgentRequest

DEFAULT_SMOKE_START_DATE = date(2024, 1, 2)
DEFAULT_SMOKE_END_DATE = date(2024, 1, 3)
DEFAULT_SMOKE_SYMBOLS = ("000001",)


@dataclass(frozen=True, slots=True)
class AkShareSmokeSpec:
    """Inputs for one real AkShare ingestion smoke run."""

    universe: str = "akshare_smoke"
    symbols: tuple[str, ...] = DEFAULT_SMOKE_SYMBOLS
    start_date: date = DEFAULT_SMOKE_START_DATE
    end_date: date = DEFAULT_SMOKE_END_DATE
    frequency: str = "daily"
    adjust: str = ""
    max_retries: int = 2
    retry_backoff_sec: float = 0.5
    symbol_sleep_sec: float = 0.0
    timeout_sec: float = 15.0
    task_id: str = "akshare-smoke"

    def payload(self) -> dict[str, Any]:
        return {
            "universe": self.universe,
            "symbols": list(self.symbols),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "frequency": self.frequency,
            "provider": "akshare",
            "adjust": self.adjust,
            "use_cache": False,
            "force_refresh": True,
            "max_retries": self.max_retries,
            "retry_backoff_sec": self.retry_backoff_sec,
            "symbol_sleep_sec": self.symbol_sleep_sec,
            "continue_on_symbol_error": True,
        }


@dataclass(frozen=True, slots=True)
class SmokeDiagnostic:
    name: str
    status: str
    message: str
    elapsed_sec: float
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "elapsed_sec": round(self.elapsed_sec, 6),
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class AkShareSmokeReport:
    status: str
    started_at: str
    elapsed_sec: float
    request: Mapping[str, Any]
    diagnostics: tuple[SmokeDiagnostic, ...]
    output: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None
    suggested_actions: tuple[str, ...] = ()

    @property
    def success(self) -> bool:
        return self.status == "success"

    @property
    def exit_code(self) -> int:
        if self.status == "success":
            return 0
        if self.status == "partial_success":
            return 2
        return 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "success": self.success,
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "elapsed_sec": round(self.elapsed_sec, 6),
            "request": dict(self.request),
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
            "output": dict(self.output),
            "error": self.error,
            "suggested_actions": list(self.suggested_actions),
        }


def run_akshare_smoke(
    config: AppConfig,
    spec: AkShareSmokeSpec,
    *,
    providers: Mapping[str, MarketDataProvider] | None = None,
    calendar_providers: Mapping[str, TradingCalendarProvider] | None = None,
) -> AkShareSmokeReport:
    """Run a live-data smoke test and return machine-readable diagnostics.

    Tests can inject fake providers. When providers are not injected, this
    function imports AkShare and runs through the real AkShare-backed provider.
    """

    started_at = datetime.now(UTC).isoformat()
    started_clock = perf_counter()
    diagnostics: list[SmokeDiagnostic] = [
        _configuration_diagnostic(config=config, spec=spec),
    ]

    if providers is None or calendar_providers is None:
        import_diagnostic, import_ok = _akshare_import_diagnostic()
        diagnostics.append(import_diagnostic)
        if not import_ok:
            error = import_diagnostic.message
            return _build_report(
                status="error",
                started_at=started_at,
                started_clock=started_clock,
                spec=spec,
                diagnostics=diagnostics,
                error=error,
            )
    else:
        diagnostics.append(
            SmokeDiagnostic(
                name="akshare_import",
                status="skipped",
                message="Skipped because test providers were injected.",
                elapsed_sec=0.0,
            )
        )

    try:
        active_providers = (
            dict(providers)
            if providers is not None
            else {"akshare": AkShareMarketDataProvider(timeout=spec.timeout_sec)}
        )
        active_calendar_providers = (
            dict(calendar_providers)
            if calendar_providers is not None
            else {"akshare": AkShareTradingCalendarProvider()}
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic boundary.
        diagnostic = _error_diagnostic(
            name="provider_setup",
            message="Failed to initialize AkShare providers.",
            exc=exc,
        )
        diagnostics.append(diagnostic)
        return _build_report(
            status="error",
            started_at=started_at,
            started_clock=started_clock,
            spec=spec,
            diagnostics=diagnostics,
            error=diagnostic.message,
        )

    run_started = perf_counter()
    response = DataAgent(
        config=config,
        providers=active_providers,
        calendar_providers=active_calendar_providers,
    ).run(AgentRequest.create(spec.payload(), task_id=spec.task_id))
    run_elapsed = perf_counter() - run_started

    if response.status != "success":
        diagnostics.append(
            SmokeDiagnostic(
                name="data_agent_run",
                status="error",
                message="DataAgent returned an error.",
                elapsed_sec=run_elapsed,
                details={
                    "error": response.error,
                    "metadata": response.metadata,
                },
            )
        )
        return _build_report(
            status="error",
            started_at=started_at,
            started_clock=started_clock,
            spec=spec,
            diagnostics=diagnostics,
            output=response.output,
            error=str(response.error),
        )

    output = dict(response.output)
    failed_symbols = _string_list(output.get("failed_symbols"))
    raw_rows = _int_value(output.get("raw_rows"))
    aligned_rows = _int_value(output.get("aligned_rows"))
    if failed_symbols:
        status = "partial_success"
        quality_status = "warning"
        quality_message = "AkShare smoke run completed with failed symbols."
    elif raw_rows <= 0 or aligned_rows <= 0:
        status = "error"
        quality_status = "error"
        quality_message = "AkShare smoke run produced no usable rows."
    else:
        status = "success"
        quality_status = "success"
        quality_message = "AkShare smoke run downloaded, cleaned, and aligned data."

    diagnostics.append(
        SmokeDiagnostic(
            name="data_agent_run",
            status="success",
            message="DataAgent completed.",
            elapsed_sec=run_elapsed,
            details={
                "raw_rows": raw_rows,
                "aligned_rows": aligned_rows,
                "failed_symbols": failed_symbols,
                "failure_manifest_path": output.get("failure_manifest_path"),
            },
        )
    )
    diagnostics.append(
        SmokeDiagnostic(
            name="download_quality",
            status=quality_status,
            message=quality_message,
            elapsed_sec=0.0,
            details=output.get("download_stats", {}),
        )
    )
    diagnostics.append(_artifact_diagnostic(output))

    return _build_report(
        status=status,
        started_at=started_at,
        started_clock=started_clock,
        spec=spec,
        diagnostics=diagnostics,
        output=output,
        error=None if status != "error" else quality_message,
    )


def _configuration_diagnostic(
    *,
    config: AppConfig,
    spec: AkShareSmokeSpec,
) -> SmokeDiagnostic:
    return SmokeDiagnostic(
        name="configuration",
        status="success",
        message="Smoke configuration parsed.",
        elapsed_sec=0.0,
        details={
            "project_root": str(config.project_root),
            "data_dir": str(config.data_dir),
            "symbols": list(spec.symbols),
            "start_date": spec.start_date.isoformat(),
            "end_date": spec.end_date.isoformat(),
            "max_retries": spec.max_retries,
            "retry_backoff_sec": spec.retry_backoff_sec,
            "symbol_sleep_sec": spec.symbol_sleep_sec,
            "timeout_sec": spec.timeout_sec,
        },
    )


def _akshare_import_diagnostic() -> tuple[SmokeDiagnostic, bool]:
    started = perf_counter()
    try:
        akshare = importlib.import_module("akshare")
    except Exception as exc:  # noqa: BLE001 - diagnostic boundary.
        return (
            _error_diagnostic(
                name="akshare_import",
                message="AkShare import failed.",
                exc=exc,
                elapsed_sec=perf_counter() - started,
            ),
            False,
        )

    return (
        SmokeDiagnostic(
            name="akshare_import",
            status="success",
            message="AkShare import succeeded.",
            elapsed_sec=perf_counter() - started,
            details={"version": getattr(akshare, "__version__", "unknown")},
        ),
        True,
    )


def _artifact_diagnostic(output: Mapping[str, Any]) -> SmokeDiagnostic:
    required_fields = (
        "raw_data_path",
        "processed_data_path",
        "aligned_data_path",
    )
    checked: dict[str, str] = {}
    missing: list[str] = []
    for field_name in required_fields:
        raw_path = output.get(field_name)
        if not isinstance(raw_path, str) or not raw_path:
            missing.append(field_name)
            continue
        checked[field_name] = raw_path
        if not Path(raw_path).is_file():
            missing.append(field_name)

    failure_manifest = output.get("failure_manifest_path")
    if isinstance(failure_manifest, str) and failure_manifest:
        checked["failure_manifest_path"] = failure_manifest
        if not Path(failure_manifest).is_file():
            missing.append("failure_manifest_path")

    status = "success" if not missing else "error"
    message = "Smoke artifacts exist." if not missing else "Smoke artifacts are missing."
    return SmokeDiagnostic(
        name="artifact_check",
        status=status,
        message=message,
        elapsed_sec=0.0,
        details={"checked": checked, "missing": missing},
    )


def _build_report(
    *,
    status: str,
    started_at: str,
    started_clock: float,
    spec: AkShareSmokeSpec,
    diagnostics: list[SmokeDiagnostic],
    output: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> AkShareSmokeReport:
    elapsed = perf_counter() - started_clock
    safe_output = dict(output or {})
    return AkShareSmokeReport(
        status=status,
        started_at=started_at,
        elapsed_sec=elapsed,
        request=spec.payload(),
        diagnostics=tuple(diagnostics),
        output=safe_output,
        error=error,
        suggested_actions=_suggest_actions(
            status=status,
            error=error,
            output=safe_output,
        ),
    )


def _suggest_actions(
    *,
    status: str,
    error: str | None,
    output: Mapping[str, Any],
) -> tuple[str, ...]:
    if status == "success":
        return ("No action required for the smoke run.",)

    actions: list[str] = []
    if status == "partial_success":
        actions.extend(
            [
                "Open the failure_manifest_path and review failed symbols.",
                "Retry the failed symbols with a smaller batch before using the dataset.",
            ]
        )

    error_text = (error or "").lower()
    if "no module named" in error_text or "import failed" in error_text:
        actions.append("Install dependencies with: python -m pip install -r requirements-dev.txt")
    if "timeout" in error_text or "connection" in error_text or "network" in error_text:
        actions.append("Check network access to AkShare/Eastmoney and rerun with a larger timeout.")
    if "missing columns" in error_text:
        actions.append("AkShare response schema may have changed; inspect provider normalization.")
    if "no ohlcv data downloaded" in error_text or "no usable rows" in error_text:
        actions.append("Verify the symbol, date range, adjustment flag, and AkShare availability.")

    failure_manifest_path = output.get("failure_manifest_path")
    if isinstance(failure_manifest_path, str) and failure_manifest_path:
        actions.append(f"Inspect failed-symbol manifest: {failure_manifest_path}")

    if not actions:
        actions.append("Review diagnostics and rerun the smoke test with one known liquid symbol.")
    return tuple(dict.fromkeys(actions))


def _error_diagnostic(
    *,
    name: str,
    message: str,
    exc: Exception,
    elapsed_sec: float = 0.0,
) -> SmokeDiagnostic:
    return SmokeDiagnostic(
        name=name,
        status="error",
        message=f"{message} {exc}",
        elapsed_sec=elapsed_sec,
        details={
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        },
    )


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _int_value(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value
