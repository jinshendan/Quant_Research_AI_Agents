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
from core.i18n import (
    DEFAULT_OUTPUT_LANGUAGE,
    LocalizedText,
    OutputLanguage,
    normalize_output_language,
)
from core.models import AgentRequest

DEFAULT_SMOKE_START_DATE = date(2024, 1, 2)
DEFAULT_SMOKE_END_DATE = date(2024, 1, 3)
DEFAULT_SMOKE_SYMBOLS = ("000001",)
SMOKE_MESSAGES = {
    "configuration_parsed": LocalizedText(
        en="Smoke configuration parsed.",
        zh="Smoke test 配置已解析。",
    ),
    "injected_providers": LocalizedText(
        en="Skipped because test providers were injected.",
        zh="已跳过，因为注入了测试 provider。",
    ),
    "provider_setup_failed": LocalizedText(
        en="Failed to initialize AkShare providers.",
        zh="AkShare provider 初始化失败。",
    ),
    "akshare_import_failed": LocalizedText(
        en="AkShare import failed.",
        zh="AkShare 导入失败。",
    ),
    "akshare_import_succeeded": LocalizedText(
        en="AkShare import succeeded.",
        zh="AkShare 导入成功。",
    ),
    "data_agent_error": LocalizedText(
        en="DataAgent returned an error.",
        zh="DataAgent 返回错误。",
    ),
    "partial_quality": LocalizedText(
        en="AkShare smoke run completed with failed symbols.",
        zh="AkShare smoke test 已完成，但存在失败股票代码。",
    ),
    "no_rows": LocalizedText(
        en="AkShare smoke run produced no usable rows.",
        zh="AkShare smoke test 未产出可用数据行。",
    ),
    "quality_success": LocalizedText(
        en="AkShare smoke run downloaded, cleaned, and aligned data.",
        zh="AkShare smoke test 已完成下载、清洗和交易日对齐。",
    ),
    "data_agent_completed": LocalizedText(
        en="DataAgent completed.",
        zh="DataAgent 已完成。",
    ),
    "artifacts_exist": LocalizedText(
        en="Smoke artifacts exist.",
        zh="Smoke test 产物已存在。",
    ),
    "artifacts_missing": LocalizedText(
        en="Smoke artifacts are missing.",
        zh="Smoke test 产物缺失。",
    ),
}
SMOKE_ACTIONS = {
    "success": LocalizedText(
        en="No action required for the smoke run.",
        zh="本次 smoke test 无需额外处理。",
    ),
    "open_failure_manifest": LocalizedText(
        en="Open the failure_manifest_path and review failed symbols.",
        zh="打开 failure_manifest_path 并检查失败股票代码。",
    ),
    "retry_failed_symbols": LocalizedText(
        en="Retry the failed symbols with a smaller batch before using the dataset.",
        zh="使用数据集前，先用更小批次重试失败股票代码。",
    ),
    "install_dependencies": LocalizedText(
        en="Install dependencies with: python -m pip install -r requirements-dev.txt",
        zh="安装依赖：python -m pip install -r requirements-dev.txt",
    ),
    "check_network": LocalizedText(
        en="Check network access to AkShare/Eastmoney and rerun with a larger timeout.",
        zh="检查 AkShare/东方财富网络访问，并用更长 timeout 重跑。",
    ),
    "schema_changed": LocalizedText(
        en="AkShare response schema may have changed; inspect provider normalization.",
        zh="AkShare 返回结构可能已变化；请检查 provider 归一化逻辑。",
    ),
    "verify_symbol": LocalizedText(
        en="Verify the symbol, date range, adjustment flag, and AkShare availability.",
        zh="检查股票代码、日期范围、复权参数和 AkShare 可用性。",
    ),
    "review_diagnostics": LocalizedText(
        en="Review diagnostics and rerun the smoke test with one known liquid symbol.",
        zh="查看 diagnostics，并用一个已知流动性较好的股票代码重跑 smoke test。",
    ),
}


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
    output_language: OutputLanguage = DEFAULT_OUTPUT_LANGUAGE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "output_language",
            normalize_output_language(self.output_language),
        )

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
            "output_language": self.output_language,
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
        import_diagnostic, import_ok = _akshare_import_diagnostic(spec.output_language)
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
                message=_smoke_message("injected_providers", spec.output_language),
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
            message=_smoke_message("provider_setup_failed", spec.output_language),
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
                message=_smoke_message("data_agent_error", spec.output_language),
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
        quality_message = _smoke_message("partial_quality", spec.output_language)
    elif raw_rows <= 0 or aligned_rows <= 0:
        status = "error"
        quality_status = "error"
        quality_message = _smoke_message("no_rows", spec.output_language)
    else:
        status = "success"
        quality_status = "success"
        quality_message = _smoke_message("quality_success", spec.output_language)

    diagnostics.append(
        SmokeDiagnostic(
            name="data_agent_run",
            status="success",
            message=_smoke_message("data_agent_completed", spec.output_language),
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
    diagnostics.append(_artifact_diagnostic(output, output_language=spec.output_language))

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
        message=_smoke_message("configuration_parsed", spec.output_language),
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
            "output_language": spec.output_language,
        },
    )


def _akshare_import_diagnostic(
    output_language: OutputLanguage = DEFAULT_OUTPUT_LANGUAGE,
) -> tuple[SmokeDiagnostic, bool]:
    started = perf_counter()
    try:
        akshare = importlib.import_module("akshare")
    except Exception as exc:  # noqa: BLE001 - diagnostic boundary.
        return (
            _error_diagnostic(
                name="akshare_import",
                message=_smoke_message("akshare_import_failed", output_language),
                exc=exc,
                elapsed_sec=perf_counter() - started,
            ),
            False,
        )

    return (
        SmokeDiagnostic(
            name="akshare_import",
            status="success",
            message=_smoke_message("akshare_import_succeeded", output_language),
            elapsed_sec=perf_counter() - started,
            details={"version": getattr(akshare, "__version__", "unknown")},
        ),
        True,
    )


def _artifact_diagnostic(
    output: Mapping[str, Any],
    *,
    output_language: OutputLanguage,
) -> SmokeDiagnostic:
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
    message = _smoke_message(
        "artifacts_exist" if not missing else "artifacts_missing",
        output_language,
    )
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
            output_language=spec.output_language,
        ),
    )


def _suggest_actions(
    *,
    status: str,
    error: str | None,
    output: Mapping[str, Any],
    output_language: OutputLanguage,
) -> tuple[str, ...]:
    if status == "success":
        return (_smoke_action("success", output_language),)

    actions: list[str] = []
    if status == "partial_success":
        actions.extend(
            [
                _smoke_action("open_failure_manifest", output_language),
                _smoke_action("retry_failed_symbols", output_language),
            ]
        )

    error_text = (error or "").lower()
    if "no module named" in error_text or "import failed" in error_text:
        actions.append(_smoke_action("install_dependencies", output_language))
    if "timeout" in error_text or "connection" in error_text or "network" in error_text:
        actions.append(_smoke_action("check_network", output_language))
    if "missing columns" in error_text:
        actions.append(_smoke_action("schema_changed", output_language))
    if "no ohlcv data downloaded" in error_text or "no usable rows" in error_text:
        actions.append(_smoke_action("verify_symbol", output_language))

    failure_manifest_path = output.get("failure_manifest_path")
    if isinstance(failure_manifest_path, str) and failure_manifest_path:
        actions.append(
            _smoke_action_with_path(
                failure_manifest_path,
                output_language=output_language,
            )
        )

    if not actions:
        actions.append(_smoke_action("review_diagnostics", output_language))
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


def _smoke_message(key: str, output_language: OutputLanguage) -> str:
    return SMOKE_MESSAGES[key].render(output_language)


def _smoke_action(key: str, output_language: OutputLanguage) -> str:
    return SMOKE_ACTIONS[key].render(output_language)


def _smoke_action_with_path(
    failure_manifest_path: str,
    *,
    output_language: OutputLanguage,
) -> str:
    if output_language == "en":
        return f"Inspect failed-symbol manifest: {failure_manifest_path}"
    if output_language == "zh":
        return f"检查失败股票清单：{failure_manifest_path}"
    return (
        f"检查失败股票清单：{failure_manifest_path} / "
        f"Inspect failed-symbol manifest: {failure_manifest_path}"
    )


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _int_value(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value
