from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from agents.ashare_trading_constraints import AshareTradingConstraintSpec
from agents.backtest_agent import BacktestAgent, default_benchmark_thresholds
from agents.data_agent import DataAgent
from agents.daily_ranking import (
    DEFAULT_RANKING_TOP_N,
    DailyRankingAgent,
)
from agents.feature_agent import FeatureAgent
from agents.market_data_provider import MarketDataProvider
from agents.memory_agent import MemoryAgent
from agents.report_agent import ReportAgent
from agents.transaction_costs import TransactionCostSpec
from agents.trading_calendar import TradingCalendarProvider
from core.config import AppConfig
from core.i18n import (
    DEFAULT_OUTPUT_LANGUAGE,
    LocalizedText,
    OutputLanguage,
    normalize_output_language,
)
from core.models import AgentRequest, AgentResponse

DAILY_RESEARCH_MANIFEST_SCHEMA_VERSION = 1
DEFAULT_DAILY_OUTPUT_DIR = "daily_runs"
DEFAULT_DAILY_TEMPLATE_IDS = ("close_to_open_return",)
DEFAULT_DAILY_FACTOR_DIRECTION = "positive"
DEFAULT_DAILY_FORWARD_RETURN_DAYS = 1
DEFAULT_DAILY_QUANTILE_COUNT = 5
DEFAULT_DAILY_ANNUALIZATION_FACTOR = 252
DEFAULT_DAILY_PREVIEW_ROWS = 5
_SAFE_RUN_ID_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
SUMMARY_LABELS = {
    "status": LocalizedText(en="status", zh="状态"),
    "run_id": LocalizedText(en="run_id", zh="运行 ID"),
    "manifest": LocalizedText(en="manifest", zh="清单文件"),
    "error": LocalizedText(en="error", zh="错误"),
    "symbols": LocalizedText(en="symbols", zh="股票代码"),
    "failed_symbols": LocalizedText(en="failed_symbols", zh="失败股票代码"),
    "factor_column": LocalizedText(en="factor_column", zh="因子列"),
    "benchmark_status": LocalizedText(en="benchmark_status", zh="基准状态"),
    "failed_benchmark_tests": LocalizedText(
        en="failed_benchmark_tests",
        zh="失败基准项",
    ),
    "memory_id": LocalizedText(en="memory_id", zh="记忆 ID"),
    "top_ranked_symbols": LocalizedText(en="top_ranked_symbols", zh="排名靠前股票"),
    "ranking_path": LocalizedText(en="ranking_path", zh="排名 CSV 路径"),
    "ranking_markdown_path": LocalizedText(
        en="ranking_markdown_path",
        zh="排名 Markdown 路径",
    ),
    "report_path": LocalizedText(en="report_path", zh="报告路径"),
}


@dataclass(frozen=True, slots=True)
class DailyResearchSpec:
    """Configuration for one DataAgent -> ReportAgent daily research run."""

    universe: str
    start_date: date
    end_date: date
    run_id: str = ""
    symbols: tuple[str, ...] = ()
    output_dir: Path = Path(DEFAULT_DAILY_OUTPUT_DIR)
    provider: str = "akshare"
    frequency: str = "daily"
    adjust: str = ""
    use_cache: bool = True
    force_refresh: bool = False
    max_retries: int = 2
    retry_backoff_sec: float = 0.5
    symbol_sleep_sec: float = 0.0
    template_ids: tuple[str, ...] = DEFAULT_DAILY_TEMPLATE_IDS
    rolling_features: tuple[str, ...] = ()
    rolling_windows: tuple[int, ...] = ()
    rank_transforms: tuple[str, ...] = ()
    factor_set_name: str = "daily_research"
    factor_column: str | None = None
    factor_direction: str = DEFAULT_DAILY_FACTOR_DIRECTION
    forward_return_days: int = DEFAULT_DAILY_FORWARD_RETURN_DAYS
    quantile_count: int = DEFAULT_DAILY_QUANTILE_COUNT
    annualization_factor: int = DEFAULT_DAILY_ANNUALIZATION_FACTOR
    benchmark_thresholds: dict[str, int | float | None] = field(
        default_factory=default_benchmark_thresholds
    )
    factor_metadata: dict[str, Any] = field(default_factory=dict)
    preview_rows: int = DEFAULT_DAILY_PREVIEW_ROWS
    ranking_top_n: int = DEFAULT_RANKING_TOP_N
    transaction_costs: TransactionCostSpec = field(
        default_factory=TransactionCostSpec,
    )
    trading_constraints: AshareTradingConstraintSpec = field(
        default_factory=AshareTradingConstraintSpec,
    )
    output_language: OutputLanguage = DEFAULT_OUTPUT_LANGUAGE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "output_language",
            normalize_output_language(self.output_language),
        )
        if self.end_date < self.start_date:
            msg = "config.end_date must be greater than or equal to config.start_date."
            raise ValueError(msg)
        if not self.template_ids:
            msg = "config.template_ids must contain at least one template id."
            raise ValueError(msg)
        if self.factor_direction not in {"positive", "negative"}:
            msg = "config.factor_direction must be positive or negative."
            raise ValueError(msg)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> DailyResearchSpec:
        raw = _payload_section(payload)
        return cls(
            universe=_required_str(raw, "universe"),
            start_date=_parse_date(raw.get("start_date"), "start_date"),
            end_date=_parse_date(raw.get("end_date"), "end_date"),
            run_id=_optional_str(raw, "run_id", ""),
            symbols=_optional_str_sequence(raw, "symbols"),
            output_dir=_optional_path(raw, "output_dir", DEFAULT_DAILY_OUTPUT_DIR),
            provider=_optional_str(raw, "provider", "akshare"),
            frequency=_optional_str(raw, "frequency", "daily"),
            adjust=_optional_str(raw, "adjust", ""),
            use_cache=_optional_bool(raw, "use_cache", True),
            force_refresh=_optional_bool(raw, "force_refresh", False),
            max_retries=_optional_int(raw, "max_retries", 2, minimum=0, maximum=5),
            retry_backoff_sec=_optional_float(
                raw,
                "retry_backoff_sec",
                0.5,
                minimum=0.0,
                maximum=60.0,
            ),
            symbol_sleep_sec=_optional_float(
                raw,
                "symbol_sleep_sec",
                0.0,
                minimum=0.0,
                maximum=60.0,
            ),
            template_ids=_optional_str_sequence(
                raw,
                "template_ids",
                default=DEFAULT_DAILY_TEMPLATE_IDS,
            ),
            rolling_features=_optional_str_sequence(raw, "rolling_features"),
            rolling_windows=_optional_int_sequence(raw, "rolling_windows"),
            rank_transforms=_optional_str_sequence(raw, "rank_transforms"),
            factor_set_name=_optional_str(raw, "factor_set_name", "daily_research"),
            factor_column=_optional_nullable_str(raw, "factor_column"),
            factor_direction=_optional_str(
                raw,
                "factor_direction",
                DEFAULT_DAILY_FACTOR_DIRECTION,
            ),
            forward_return_days=_optional_int(
                raw,
                "forward_return_days",
                DEFAULT_DAILY_FORWARD_RETURN_DAYS,
                minimum=1,
                maximum=60,
            ),
            quantile_count=_optional_int(
                raw,
                "quantile_count",
                DEFAULT_DAILY_QUANTILE_COUNT,
                minimum=2,
                maximum=20,
            ),
            annualization_factor=_optional_int(
                raw,
                "annualization_factor",
                DEFAULT_DAILY_ANNUALIZATION_FACTOR,
                minimum=1,
                maximum=366,
            ),
            benchmark_thresholds=_optional_benchmark_thresholds(raw),
            factor_metadata=_optional_mapping(raw, "factor_metadata"),
            preview_rows=_optional_int(
                raw,
                "preview_rows",
                DEFAULT_DAILY_PREVIEW_ROWS,
                minimum=0,
                maximum=50,
            ),
            ranking_top_n=_optional_int(
                raw,
                "ranking_top_n",
                DEFAULT_RANKING_TOP_N,
                minimum=1,
                maximum=200,
            ),
            transaction_costs=TransactionCostSpec.from_mapping(
                _optional_mapping_alias(raw, ("transaction_costs", "cost_profile")),
            ),
            trading_constraints=AshareTradingConstraintSpec.from_mapping(
                _optional_mapping(raw, "trading_constraints"),
            ),
            output_language=_optional_output_language(raw),
        )

    @property
    def effective_run_id(self) -> str:
        if self.run_id.strip():
            return _safe_run_id(self.run_id)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"daily-research-{timestamp}"

    def run_dir(self, project_root: Path, run_id: str) -> Path:
        output_dir = self.output_dir
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
        return output_dir / _safe_run_id(run_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "universe": self.universe,
            "symbols": list(self.symbols),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "run_id": self.run_id,
            "output_dir": str(self.output_dir),
            "provider": self.provider,
            "frequency": self.frequency,
            "adjust": self.adjust,
            "use_cache": self.use_cache,
            "force_refresh": self.force_refresh,
            "max_retries": self.max_retries,
            "retry_backoff_sec": self.retry_backoff_sec,
            "symbol_sleep_sec": self.symbol_sleep_sec,
            "template_ids": list(self.template_ids),
            "rolling_features": list(self.rolling_features),
            "rolling_windows": list(self.rolling_windows),
            "rank_transforms": list(self.rank_transforms),
            "factor_set_name": self.factor_set_name,
            "factor_column": self.factor_column,
            "factor_direction": self.factor_direction,
            "forward_return_days": self.forward_return_days,
            "quantile_count": self.quantile_count,
            "annualization_factor": self.annualization_factor,
            "benchmark_thresholds": dict(self.benchmark_thresholds),
            "factor_metadata": dict(self.factor_metadata),
            "preview_rows": self.preview_rows,
            "ranking_top_n": self.ranking_top_n,
            "transaction_costs": self.transaction_costs.to_dict(),
            "trading_constraints": self.trading_constraints.to_dict(),
            "output_language": self.output_language,
        }


@dataclass(frozen=True, slots=True)
class DailyResearchRunResult:
    status: str
    run_id: str
    manifest_path: Path
    summary: Mapping[str, Any]
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.status == "success"

    @property
    def exit_code(self) -> int:
        return 0 if self.success else 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "success": self.success,
            "exit_code": self.exit_code,
            "run_id": self.run_id,
            "manifest_path": str(self.manifest_path),
            "summary": dict(self.summary),
            "error": self.error,
        }


def load_daily_research_config(path: str | Path) -> DailyResearchSpec:
    config_path = Path(path)
    if not config_path.is_file():
        msg = f"Daily research config file not found: {config_path}."
        raise OSError(msg)
    if config_path.suffix.lower() == ".json":
        document = json.loads(config_path.read_text(encoding="utf-8"))
    elif config_path.suffix.lower() == ".toml":
        document = tomllib.loads(config_path.read_text(encoding="utf-8"))
    else:
        msg = "Daily research config must be a .json or .toml file."
        raise ValueError(msg)
    if not isinstance(document, Mapping):
        msg = "Daily research config must contain an object."
        raise ValueError(msg)
    return DailyResearchSpec.from_mapping(document)


def run_daily_research(
    config: AppConfig,
    spec: DailyResearchSpec,
    *,
    providers: Mapping[str, MarketDataProvider] | None = None,
    calendar_providers: Mapping[str, TradingCalendarProvider] | None = None,
) -> DailyResearchRunResult:
    started_at = datetime.now(UTC).isoformat()
    started_clock = perf_counter()
    run_id = spec.effective_run_id
    run_dir = spec.run_dir(config.project_root, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "daily_research_manifest.json"
    stages: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, str | None] = {}

    try:
        data_response = DataAgent(
            config=config,
            providers=providers,
            calendar_providers=calendar_providers,
        ).run(AgentRequest.create(_data_payload(spec), task_id=f"{run_id}-data"))
        _require_success(data_response, "data")
        stages["data"] = _data_stage(data_response)
        artifacts.update(stages["data"]["artifacts"])

        aligned_data_path = _required_output_path(
            data_response,
            "aligned_data_path",
            stage="data",
        )
        feature_response = FeatureAgent(config=config).run(
            AgentRequest.create(
                _feature_payload(spec, aligned_data_path),
                task_id=f"{run_id}-feature",
            )
        )
        _require_success(feature_response, "feature")
        stages["feature"] = _feature_stage(feature_response)
        artifacts.update(stages["feature"]["artifacts"])

        factor_manifest_path = _required_nested_output_path(
            feature_response,
            "storage_stats",
            "manifest_path",
            stage="feature",
        )
        factor_column = _selected_factor_column(spec, feature_response)
        result_json_path = run_dir / "backtest_result.json"
        backtest_response = BacktestAgent().run(
            AgentRequest.create(
                _backtest_payload(
                    spec,
                    factor_manifest_path=factor_manifest_path,
                    factor_column=factor_column,
                    result_json_path=result_json_path,
                ),
                task_id=f"{run_id}-backtest",
            )
        )
        _require_success(backtest_response, "backtest")
        stages["backtest"] = _backtest_stage(backtest_response)
        artifacts.update(stages["backtest"]["artifacts"])

        factor_matrix_path = _required_nested_output_path(
            feature_response,
            "storage_stats",
            "matrix_path",
            stage="feature",
        )
        ranking_path = run_dir / "daily_stock_ranking.csv"
        ranking_markdown_path = run_dir / "daily_stock_ranking.md"
        ranking_response = DailyRankingAgent(config=config).run(
            AgentRequest.create(
                {
                    "factor_matrix_path": str(factor_matrix_path),
                    "aligned_data_path": str(aligned_data_path),
                    "factor_column": factor_column,
                    "factor_direction": spec.factor_direction,
                    "top_n": spec.ranking_top_n,
                    "ranking_path": str(ranking_path),
                    "ranking_markdown_path": str(ranking_markdown_path),
                    "trading_constraints": spec.trading_constraints.to_dict(),
                    "output_language": spec.output_language,
                },
                task_id=f"{run_id}-ranking",
            )
        )
        _require_success(ranking_response, "ranking")
        stages["ranking"] = _ranking_stage(ranking_response)
        artifacts.update(stages["ranking"]["artifacts"])

        result_json_path = _required_output_path(
            backtest_response,
            "result_json_path",
            stage="backtest",
        )
        factor_metadata = _factor_metadata(spec, factor_column)
        memory_response = MemoryAgent(config=config).run(
            AgentRequest.create(
                {
                    "result_json_path": str(result_json_path),
                    "factor_metadata": factor_metadata,
                    "output_language": spec.output_language,
                },
                task_id=f"{run_id}-memory",
            )
        )
        _require_success(memory_response, "memory")
        stages["memory"] = _memory_stage(memory_response)
        artifacts.update(stages["memory"]["artifacts"])

        memory_path = _required_output_path(memory_response, "memory_path", stage="memory")
        wiki_path = _required_output_path(
            memory_response,
            "factor_wiki_path",
            stage="memory",
        )
        report_path = run_dir / "research_report.md"
        report_response = ReportAgent(config=config).run(
            AgentRequest.create(
                {
                    "memory_path": str(memory_path),
                    "factor_name": str(factor_metadata["name"]),
                    "factor_wiki_path": str(wiki_path),
                    "report_path": str(report_path),
                    "output_language": spec.output_language,
                },
                task_id=f"{run_id}-report",
            )
        )
        _require_success(report_response, "report")
        stages["report"] = _report_stage(report_response)
        artifacts.update(stages["report"]["artifacts"])

        elapsed = perf_counter() - started_clock
        summary = _summary(stages, output_language=spec.output_language)
        manifest = _manifest(
            status="success",
            started_at=started_at,
            elapsed=elapsed,
            run_id=run_id,
            spec=spec,
            run_dir=run_dir,
            stages=stages,
            artifacts=artifacts,
            summary=summary,
            error=None,
        )
        _write_manifest(manifest_path, manifest)
        return DailyResearchRunResult(
            status="success",
            run_id=run_id,
            manifest_path=manifest_path,
            summary=summary,
        )
    except Exception as exc:  # noqa: BLE001 - pipeline boundary.
        elapsed = perf_counter() - started_clock
        error = str(exc)
        summary = _summary(stages, output_language=spec.output_language)
        manifest = _manifest(
            status="error",
            started_at=started_at,
            elapsed=elapsed,
            run_id=run_id,
            spec=spec,
            run_dir=run_dir,
            stages=stages,
            artifacts=artifacts,
            summary=summary,
            error=error,
        )
        _write_manifest(manifest_path, manifest)
        return DailyResearchRunResult(
            status="error",
            run_id=run_id,
            manifest_path=manifest_path,
            summary=summary,
            error=error,
        )


def format_daily_research_summary(
    result: DailyResearchRunResult,
    *,
    output_language: str | None = None,
) -> str:
    summary = dict(result.summary)
    language = normalize_output_language(
        output_language if output_language is not None else _summary_language(summary),
    )
    lines = [
        f"{_summary_label('status', language)}: {result.status}",
        f"{_summary_label('run_id', language)}: {result.run_id}",
        f"{_summary_label('manifest', language)}: {result.manifest_path}",
    ]
    if result.error:
        lines.append(f"{_summary_label('error', language)}: {result.error}")
    for key in (
        "symbols",
        "failed_symbols",
        "factor_column",
        "benchmark_status",
        "failed_benchmark_tests",
        "memory_id",
        "top_ranked_symbols",
        "ranking_path",
        "ranking_markdown_path",
        "report_path",
    ):
        value = summary.get(key)
        if value not in (None, "", []):
            lines.append(f"{_summary_label(key, language)}: {value}")
    return "\n".join(lines)


def _data_payload(spec: DailyResearchSpec) -> dict[str, Any]:
    payload = {
        "universe": spec.universe,
        "start_date": spec.start_date.isoformat(),
        "end_date": spec.end_date.isoformat(),
        "provider": spec.provider,
        "frequency": spec.frequency,
        "adjust": spec.adjust,
        "use_cache": spec.use_cache,
        "force_refresh": spec.force_refresh,
        "max_retries": spec.max_retries,
        "retry_backoff_sec": spec.retry_backoff_sec,
        "symbol_sleep_sec": spec.symbol_sleep_sec,
        "continue_on_symbol_error": True,
    }
    if spec.symbols:
        payload["symbols"] = list(spec.symbols)
    return payload


def _feature_payload(spec: DailyResearchSpec, aligned_data_path: Path) -> dict[str, Any]:
    return {
        "aligned_data_path": str(aligned_data_path),
        "template_ids": list(spec.template_ids),
        "rolling_features": list(spec.rolling_features),
        "rolling_windows": list(spec.rolling_windows),
        "rank_transforms": list(spec.rank_transforms),
        "quantile_count": spec.quantile_count,
        "preview_rows": spec.preview_rows,
        "save_factors": True,
        "factor_set_name": spec.factor_set_name,
    }


def _backtest_payload(
    spec: DailyResearchSpec,
    *,
    factor_manifest_path: Path,
    factor_column: str,
    result_json_path: Path,
) -> dict[str, Any]:
    return {
        "factor_manifest_path": str(factor_manifest_path),
        "factor_column": factor_column,
        "factor_direction": spec.factor_direction,
        "forward_return_days": spec.forward_return_days,
        "quantile_count": spec.quantile_count,
        "annualization_factor": spec.annualization_factor,
        "result_json_path": str(result_json_path),
        "benchmark_thresholds": dict(spec.benchmark_thresholds),
        "transaction_costs": spec.transaction_costs.to_dict(),
        "preview_rows": spec.preview_rows,
    }


def _selected_factor_column(spec: DailyResearchSpec, response: AgentResponse) -> str:
    if spec.factor_column:
        return spec.factor_column
    factor_columns = response.output.get("factor_columns")
    if not isinstance(factor_columns, list) or not factor_columns:
        msg = "FeatureAgent output has no factor_columns."
        raise ValueError(msg)
    first = factor_columns[0]
    if not isinstance(first, str) or not first:
        msg = "FeatureAgent factor_columns must contain strings."
        raise ValueError(msg)
    return first


def _factor_metadata(spec: DailyResearchSpec, factor_column: str) -> dict[str, Any]:
    metadata = dict(spec.factor_metadata)
    factor_name = metadata.get("name")
    if not isinstance(factor_name, str) or not factor_name.strip():
        metadata["name"] = factor_column.removeprefix("factor__")
    metadata.setdefault("formula", factor_column)
    metadata.setdefault(
        "hypothesis",
        f"Daily research run for {factor_column}.",
    )
    metadata.setdefault("universe", spec.universe)
    return metadata


def _require_success(response: AgentResponse, stage: str) -> None:
    if response.status != "success":
        msg = f"{stage} stage failed: {response.error}"
        raise RuntimeError(msg)


def _data_stage(response: AgentResponse) -> dict[str, Any]:
    output = response.output
    return {
        "status": response.status,
        "task_id": response.metadata.get("task_id"),
        "artifacts": {
            "raw_data_path": _optional_output_path(output, "raw_data_path"),
            "processed_data_path": _optional_output_path(output, "processed_data_path"),
            "aligned_data_path": _optional_output_path(output, "aligned_data_path"),
            "failure_manifest_path": _optional_output_path(output, "failure_manifest_path"),
            "duckdb_path": _nested_optional_str(output, "storage_stats", "database_path"),
        },
        "summary": {
            "symbols": output.get("symbols"),
            "successful_symbols": output.get("successful_symbols"),
            "failed_symbols": output.get("failed_symbols"),
            "raw_rows": output.get("raw_rows"),
            "processed_rows": output.get("processed_rows"),
            "aligned_rows": output.get("aligned_rows"),
            "download_stats": output.get("download_stats"),
        },
    }


def _feature_stage(response: AgentResponse) -> dict[str, Any]:
    output = response.output
    return {
        "status": response.status,
        "task_id": response.metadata.get("task_id"),
        "artifacts": {
            "factor_matrix_path": _nested_optional_str(
                output,
                "storage_stats",
                "matrix_path",
            ),
            "factor_manifest_path": _nested_optional_str(
                output,
                "storage_stats",
                "manifest_path",
            ),
        },
        "summary": {
            "factor_columns": output.get("factor_columns"),
            "row_count": output.get("row_count"),
            "factor_count": output.get("factor_count"),
            "feature_stats": output.get("feature_stats"),
        },
    }


def _backtest_stage(response: AgentResponse) -> dict[str, Any]:
    output = response.output
    return {
        "status": response.status,
        "task_id": response.metadata.get("task_id"),
        "artifacts": {
            "result_json_path": _optional_output_path(output, "result_json_path"),
            "factor_matrix_path": _optional_output_path(output, "factor_matrix_path"),
            "aligned_data_path": _optional_output_path(output, "aligned_data_path"),
        },
        "summary": {
            "factor_column": output.get("factor_column"),
            "benchmark_status": output.get("benchmark_status"),
            "failed_benchmark_tests": _failed_benchmark_tests(
                output.get("benchmark_tests"),
            ),
            "usable_row_count": output.get("usable_row_count"),
            "portfolio_date_count": output.get("portfolio_date_count"),
            "ic_stats": output.get("ic_stats"),
            "rank_ic_stats": output.get("rank_ic_stats"),
            "sharpe_stats": output.get("sharpe_stats"),
            "gross_sharpe_stats": output.get("gross_sharpe_stats"),
            "drawdown_stats": output.get("drawdown_stats"),
            "gross_drawdown_stats": output.get("gross_drawdown_stats"),
            "transaction_costs": output.get("transaction_costs"),
            "cost_stats": output.get("cost_stats"),
        },
    }


def _memory_stage(response: AgentResponse) -> dict[str, Any]:
    output = response.output
    return {
        "status": response.status,
        "task_id": response.metadata.get("task_id"),
        "artifacts": {
            "memory_path": _optional_output_path(output, "memory_path"),
            "vector_index_path": _optional_output_path(output, "vector_index_path"),
            "vector_metadata_path": _optional_output_path(
                output,
                "vector_metadata_path",
            ),
            "factor_wiki_path": _optional_output_path(output, "factor_wiki_path"),
        },
        "summary": {
            "memory_id": output.get("memory_id"),
            "factor_name": response.metadata.get("factor_name"),
            "benchmark_status": response.metadata.get("benchmark_status"),
        },
    }


def _ranking_stage(response: AgentResponse) -> dict[str, Any]:
    output = response.output
    return {
        "status": response.status,
        "task_id": response.metadata.get("task_id"),
        "artifacts": {
            "ranking_path": _optional_output_path(output, "ranking_path"),
            "ranking_markdown_path": _optional_output_path(
                output,
                "ranking_markdown_path",
            ),
        },
        "summary": {
            "ranking_date": output.get("ranking_date"),
            "factor_column": output.get("factor_column"),
            "row_count": output.get("row_count"),
            "top_symbols": output.get("top_symbols"),
            "ranking_stats": output.get("ranking_stats"),
        },
    }


def _report_stage(response: AgentResponse) -> dict[str, Any]:
    output = response.output
    return {
        "status": response.status,
        "task_id": response.metadata.get("task_id"),
        "artifacts": {
            "report_path": _optional_output_path(output, "report_path"),
        },
        "summary": {
            "report_title": output.get("report_title"),
            "section_count": output.get("section_count"),
        },
    }


def _summary(
    stages: Mapping[str, Mapping[str, Any]],
    *,
    output_language: OutputLanguage,
) -> dict[str, Any]:
    data = _stage_summary(stages, "data")
    feature = _stage_summary(stages, "feature")
    backtest = _stage_summary(stages, "backtest")
    ranking = _stage_summary(stages, "ranking")
    ranking_artifacts = _stage_artifacts(stages, "ranking")
    memory = _stage_summary(stages, "memory")
    report_artifacts = _stage_artifacts(stages, "report")
    return {
        "symbols": data.get("symbols"),
        "failed_symbols": data.get("failed_symbols"),
        "factor_columns": feature.get("factor_columns"),
        "factor_column": backtest.get("factor_column"),
        "benchmark_status": backtest.get("benchmark_status"),
        "failed_benchmark_tests": backtest.get("failed_benchmark_tests"),
        "memory_id": memory.get("memory_id"),
        "top_ranked_symbols": ranking.get("top_symbols"),
        "ranking_path": ranking_artifacts.get("ranking_path"),
        "ranking_markdown_path": ranking_artifacts.get("ranking_markdown_path"),
        "report_path": report_artifacts.get("report_path"),
        "output_language": output_language,
    }


def _manifest(
    *,
    status: str,
    started_at: str,
    elapsed: float,
    run_id: str,
    spec: DailyResearchSpec,
    run_dir: Path,
    stages: Mapping[str, Mapping[str, Any]],
    artifacts: Mapping[str, str | None],
    summary: Mapping[str, Any],
    error: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": DAILY_RESEARCH_MANIFEST_SCHEMA_VERSION,
        "status": status,
        "run_id": run_id,
        "started_at": started_at,
        "elapsed_sec": round(elapsed, 6),
        "run_dir": str(run_dir),
        "output_language": spec.output_language,
        "request": spec.to_dict(),
        "stages": dict(stages),
        "artifacts": dict(artifacts),
        "summary": dict(summary),
        "error": error,
    }


def _write_manifest(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(f"{path}.tmp")
    temp_path.write_text(
        json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _required_output_path(response: AgentResponse, key: str, *, stage: str) -> Path:
    value = response.output.get(key)
    if not isinstance(value, str) or not value:
        msg = f"{stage} stage output is missing {key}."
        raise ValueError(msg)
    return Path(value)


def _required_nested_output_path(
    response: AgentResponse,
    parent: str,
    key: str,
    *,
    stage: str,
) -> Path:
    value = _nested_optional_str(response.output, parent, key)
    if not value:
        msg = f"{stage} stage output is missing {parent}.{key}."
        raise ValueError(msg)
    return Path(value)


def _optional_output_path(output: Mapping[str, Any], key: str) -> str | None:
    value = output.get(key)
    return value if isinstance(value, str) and value else None


def _nested_optional_str(
    output: Mapping[str, Any],
    parent: str,
    key: str,
) -> str | None:
    parent_value = output.get(parent)
    if not isinstance(parent_value, Mapping):
        return None
    value = parent_value.get(key)
    return value if isinstance(value, str) and value else None


def _stage_summary(
    stages: Mapping[str, Mapping[str, Any]],
    stage_name: str,
) -> Mapping[str, Any]:
    stage = stages.get(stage_name)
    if not isinstance(stage, Mapping):
        return {}
    summary = stage.get("summary")
    return summary if isinstance(summary, Mapping) else {}


def _stage_artifacts(
    stages: Mapping[str, Mapping[str, Any]],
    stage_name: str,
) -> Mapping[str, Any]:
    stage = stages.get(stage_name)
    if not isinstance(stage, Mapping):
        return {}
    artifacts = stage.get("artifacts")
    return artifacts if isinstance(artifacts, Mapping) else {}


def _failed_benchmark_tests(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return []
    raw_failed_tests = value.get("failed_tests")
    if isinstance(raw_failed_tests, list):
        return [str(item) for item in raw_failed_tests if str(item).strip()]

    raw_tests = value.get("tests")
    if not isinstance(raw_tests, list):
        return []
    failed = []
    for item in raw_tests:
        if isinstance(item, Mapping) and item.get("passed") is False:
            name = item.get("name")
            if name is not None:
                failed.append(str(name))
    return failed


def _payload_section(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = payload.get("daily_research")
    if isinstance(nested, Mapping):
        return nested
    return payload


def _summary_language(summary: Mapping[str, Any]) -> str | None:
    value = summary.get("output_language")
    return value if isinstance(value, str) else None


def _summary_label(key: str, output_language: OutputLanguage) -> str:
    label = SUMMARY_LABELS.get(key)
    if label is None:
        return key
    return label.render(output_language)


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        msg = f"config.{key} must be a non-empty string."
        raise ValueError(msg)
    return value.strip()


def _optional_str(payload: Mapping[str, Any], key: str, default: str) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str):
        msg = f"config.{key} must be a string."
        raise ValueError(msg)
    return value.strip()


def _optional_nullable_str(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"config.{key} must be a string."
        raise ValueError(msg)
    stripped = value.strip()
    return stripped or None


def _optional_output_language(payload: Mapping[str, Any]) -> OutputLanguage:
    value = payload.get("output_language")
    if value is None:
        return DEFAULT_OUTPUT_LANGUAGE
    if not isinstance(value, str):
        msg = "config.output_language must be a string."
        raise ValueError(msg)
    return normalize_output_language(value)


def _optional_bool(payload: Mapping[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        msg = f"config.{key} must be a boolean."
        raise ValueError(msg)
    return value


def _optional_path(payload: Mapping[str, Any], key: str, default: str) -> Path:
    value = payload.get(key, default)
    if not isinstance(value, str) or not value.strip():
        msg = f"config.{key} must be a non-empty string."
        raise ValueError(msg)
    return Path(value).expanduser()


def _optional_str_sequence(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: Sequence[str] = (),
) -> tuple[str, ...]:
    value = payload.get(key, list(default))
    if not isinstance(value, Sequence) or isinstance(value, str):
        msg = f"config.{key} must be a list of strings."
        raise ValueError(msg)
    normalized = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            msg = f"config.{key} must be a list of non-empty strings."
            raise ValueError(msg)
        normalized.append(item.strip())
    return tuple(dict.fromkeys(normalized))


def _optional_int_sequence(payload: Mapping[str, Any], key: str) -> tuple[int, ...]:
    value = payload.get(key, [])
    if not isinstance(value, Sequence) or isinstance(value, str):
        msg = f"config.{key} must be a list of integers."
        raise ValueError(msg)
    normalized = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            msg = f"config.{key} must be a list of integers."
            raise ValueError(msg)
        normalized.append(item)
    return tuple(dict.fromkeys(normalized))


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
        msg = f"config.{key} must be an integer."
        raise ValueError(msg)
    if value < minimum or value > maximum:
        msg = f"config.{key} must be between {minimum} and {maximum}."
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
        msg = f"config.{key} must be a finite number."
        raise ValueError(msg)
    normalized = float(value)
    if not (minimum <= normalized <= maximum):
        msg = f"config.{key} must be between {minimum} and {maximum}."
        raise ValueError(msg)
    return normalized


def _optional_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    if not isinstance(value, Mapping):
        msg = f"config.{key} must be an object."
        raise ValueError(msg)
    return dict(value)


def _optional_mapping_alias(
    payload: Mapping[str, Any],
    keys: Sequence[str],
) -> dict[str, Any] | None:
    found_key: str | None = None
    found_value: Any = None
    for key in keys:
        if key in payload:
            if found_key is not None:
                msg = f"Provide only one of config.{found_key} or config.{key}."
                raise ValueError(msg)
            found_key = key
            found_value = payload[key]
    if found_key is None or found_value is None:
        return None
    if not isinstance(found_value, Mapping):
        msg = f"config.{found_key} must be an object."
        raise ValueError(msg)
    return dict(found_value)


def _optional_benchmark_thresholds(
    payload: Mapping[str, Any],
) -> dict[str, int | float | None]:
    thresholds = default_benchmark_thresholds()
    value = payload.get("benchmark_thresholds")
    if value is None:
        return thresholds
    if not isinstance(value, Mapping):
        msg = "config.benchmark_thresholds must be an object."
        raise ValueError(msg)
    for key, item in value.items():
        if not isinstance(key, str):
            msg = "config.benchmark_thresholds keys must be strings."
            raise ValueError(msg)
        if item is not None and (
            isinstance(item, bool) or not isinstance(item, int | float)
        ):
            msg = f"config.benchmark_thresholds.{key} must be a number or null."
            raise ValueError(msg)
        thresholds[key] = item
    return thresholds


def _parse_date(value: Any, key: str) -> date:
    if not isinstance(value, str) or not value.strip():
        msg = f"config.{key} must be a YYYY-MM-DD string."
        raise ValueError(msg)
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        msg = f"config.{key} must be a valid YYYY-MM-DD date."
        raise ValueError(msg) from exc


def _safe_run_id(value: str) -> str:
    cleaned = _SAFE_RUN_ID_PATTERN.sub("_", value.strip())
    return cleaned.strip("._") or "daily-research"
