from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from core.config import AppConfig
from core.i18n import (
    DEFAULT_OUTPUT_LANGUAGE,
    LocalizedText,
    OutputLanguage,
    normalize_output_language,
    render_label,
)
from core.logging import AgentLoggerAdapter, get_agent_logger
from core.models import AgentRequest, AgentResponse
from agents.memory_agent import FactorMemoryStore

REPORT_SCHEMA_VERSION = 1
REPORT_DRAFT_FORMAT = "structured_json"
REPORT_FORMAT = "markdown"
DEFAULT_REPORTS_DIRNAME = "research_logs"
REPORT_SECTION_ORDER = (
    ("hypothesis", "Hypothesis"),
    ("factor_formula", "Factor Formula"),
    ("backtest_results", "Backtest Results"),
    ("out_of_sample_validation", "Out-of-sample Validation"),
    ("risk_analysis", "Risk Analysis"),
    ("conclusion", "Conclusion"),
)
REPORT_SECTION_TITLES = {
    "hypothesis": LocalizedText(en="Hypothesis", zh="研究假设"),
    "factor_formula": LocalizedText(en="Factor Formula", zh="因子公式"),
    "backtest_results": LocalizedText(en="Backtest Results", zh="回测结果"),
    "out_of_sample_validation": LocalizedText(
        en="Out-of-sample Validation",
        zh="样本外验证",
    ),
    "risk_analysis": LocalizedText(en="Risk Analysis", zh="风险分析"),
    "conclusion": LocalizedText(en="Conclusion", zh="结论"),
}
REPORT_LABELS = {
    "metadata": LocalizedText(en="Metadata", zh="元数据"),
    "memory_id": LocalizedText(en="Memory ID", zh="记忆 ID"),
    "source_task": LocalizedText(en="Source task", zh="来源任务"),
    "source_agent": LocalizedText(en="Source agent", zh="来源 Agent"),
    "generated_at": LocalizedText(en="Generated at", zh="生成时间"),
    "factor": LocalizedText(en="Factor", zh="因子"),
    "name": LocalizedText(en="Name", zh="名称"),
    "factor_column": LocalizedText(en="Factor column", zh="因子列"),
    "source_type": LocalizedText(en="Source type", zh="因子来源类型"),
    "category": LocalizedText(en="Category", zh="类别"),
    "formula": LocalizedText(en="Formula", zh="公式"),
    "hypothesis": LocalizedText(en="Hypothesis", zh="假设"),
    "direction": LocalizedText(en="Direction", zh="方向"),
    "lookback_days": LocalizedText(en="Lookback days", zh="回看天数"),
    "data_lag_days": LocalizedText(en="Data lag days", zh="数据滞后天数"),
    "universe": LocalizedText(en="Universe", zh="股票池"),
    "forward_return_days": LocalizedText(en="Forward return days", zh="前瞻收益天数"),
    "related_factors": LocalizedText(en="Related factors", zh="相关因子"),
    "paper_reference": LocalizedText(en="Paper reference", zh="文献/资料引用"),
    "ic": LocalizedText(en="IC", zh="IC"),
    "rank_ic": LocalizedText(en="RankIC", zh="RankIC"),
    "sharpe": LocalizedText(en="Sharpe", zh="夏普"),
    "gross_sharpe": LocalizedText(en="Gross Sharpe", zh="未扣成本夏普"),
    "net_sharpe": LocalizedText(en="Net Sharpe", zh="扣成本后夏普"),
    "max_drawdown": LocalizedText(en="Max drawdown", zh="最大回撤"),
    "max_drawdown_abs": LocalizedText(en="Max drawdown abs", zh="最大回撤绝对值"),
    "total_return": LocalizedText(en="Total return", zh="总收益"),
    "gross_total_return": LocalizedText(en="Gross total return", zh="未扣成本总收益"),
    "net_total_return": LocalizedText(en="Net total return", zh="扣成本后总收益"),
    "turnover": LocalizedText(en="Turnover", zh="换手率"),
    "average_turnover": LocalizedText(en="Average turnover", zh="平均换手率"),
    "average_transaction_cost": LocalizedText(
        en="Average transaction cost",
        zh="平均交易成本",
    ),
    "total_transaction_cost": LocalizedText(
        en="Total transaction cost",
        zh="累计交易成本",
    ),
    "transaction_costs": LocalizedText(en="Transaction costs", zh="交易成本假设"),
    "transaction_cost_stats": LocalizedText(
        en="Transaction cost stats",
        zh="交易成本统计",
    ),
    "profile_name": LocalizedText(en="Profile name", zh="成本配置名称"),
    "commission_rate": LocalizedText(en="Commission rate", zh="佣金率"),
    "stamp_duty_rate": LocalizedText(en="Stamp duty rate", zh="印花税率"),
    "transfer_fee_rate": LocalizedText(en="Transfer fee rate", zh="过户费率"),
    "slippage_rate": LocalizedText(en="Slippage rate", zh="滑点率"),
    "buy_cost_rate": LocalizedText(en="Buy cost rate", zh="买入成本率"),
    "sell_cost_rate": LocalizedText(en="Sell cost rate", zh="卖出成本率"),
    "enabled": LocalizedText(en="Enabled", zh="是否启用"),
    "total_turnover": LocalizedText(en="Total turnover", zh="累计换手率"),
    "max_transaction_cost": LocalizedText(en="Max transaction cost", zh="最大单期交易成本"),
    "benchmark_status": LocalizedText(en="Benchmark status", zh="基准状态"),
    "out_of_sample_status": LocalizedText(
        en="Out-of-sample status",
        zh="样本外状态",
    ),
    "out_of_sample_passed": LocalizedText(
        en="Out-of-sample passed",
        zh="样本外是否通过",
    ),
    "validation_id": LocalizedText(en="Validation ID", zh="验证 ID"),
    "validation_method": LocalizedText(en="Validation method", zh="验证方法"),
    "split_count": LocalizedText(en="Split count", zh="切分数量"),
    "basic_oos_status": LocalizedText(en="Basic OOS status", zh="基础样本外状态"),
    "walk_forward_status": LocalizedText(
        en="Walk-forward status",
        zh="滚动验证状态",
    ),
    "metric_comparison_status": LocalizedText(
        en="Metric comparison status",
        zh="指标对比状态",
    ),
    "in_sample_split_names": LocalizedText(
        en="In-sample splits",
        zh="样本内切分",
    ),
    "out_of_sample_split_names": LocalizedText(
        en="Out-of-sample splits",
        zh="样本外切分",
    ),
    "out_of_sample_mean_rank_ic": LocalizedText(
        en="Out-of-sample mean RankIC",
        zh="样本外平均 RankIC",
    ),
    "out_of_sample_net_sharpe": LocalizedText(
        en="Out-of-sample net Sharpe",
        zh="样本外扣成本后夏普",
    ),
    "out_of_sample_net_total_return": LocalizedText(
        en="Out-of-sample net total return",
        zh="样本外扣成本后总收益",
    ),
    "passed_count": LocalizedText(en="Passed count", zh="通过数量"),
    "failed_count": LocalizedText(en="Failed count", zh="失败数量"),
    "failed_tests": LocalizedText(en="Failed tests", zh="失败测试"),
    "failure_reason": LocalizedText(en="Failure reason", zh="失败原因"),
    "market_condition": LocalizedText(en="Market condition", zh="市场条件"),
    "result_json_path": LocalizedText(en="Result JSON path", zh="结果 JSON 路径"),
    "factor_matrix_path": LocalizedText(en="Factor matrix path", zh="因子矩阵路径"),
    "verdict": LocalizedText(en="Verdict", zh="结论判断"),
    "key_metric": LocalizedText(en="Key metric", zh="关键指标"),
}
REPORT_VALUE_LABELS = {
    "candidate_for_follow_up": LocalizedText(
        en="candidate_for_follow_up",
        zh="可继续跟踪候选",
    ),
    "needs_review": LocalizedText(en="needs_review", zh="需要复核"),
    "passed": LocalizedText(en="passed", zh="通过"),
    "failed": LocalizedText(en="failed", zh="未通过"),
    "not_provided": LocalizedText(en="not_provided", zh="未提供"),
    "not_applicable": LocalizedText(en="not_applicable", zh="不适用"),
    "not_enough_data": LocalizedText(en="not_enough_data", zh="数据不足"),
    "available": LocalizedText(en="available", zh="可用"),
    "partial": LocalizedText(en="partial", zh="部分完成"),
    "unknown": LocalizedText(en="unknown", zh="未知"),
}
_SAFE_REPORT_STEM_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class ReportSpec:
    """Validated request for building a structured research report draft."""

    memory_record: dict[str, Any] | None = None
    memory_path: Path | None = None
    memory_id: str | None = None
    factor_name: str | None = None
    factor_wiki_path: Path | None = None
    out_of_sample_result_path: Path | None = None
    report_path: Path | None = None
    report_title: str | None = None
    output_language: OutputLanguage | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ReportSpec:
        memory_record = _optional_mapping(payload, "memory_record")
        memory_path = _optional_path(payload, "memory_path")
        if memory_record is None and memory_path is None:
            msg = "payload.memory_record or payload.memory_path is required."
            raise ValueError(msg)
        if memory_record is not None and memory_path is not None:
            msg = "Provide only one of payload.memory_record or payload.memory_path."
            raise ValueError(msg)

        return cls(
            memory_record=memory_record,
            memory_path=memory_path,
            memory_id=_optional_str(payload, "memory_id"),
            factor_name=_optional_str(payload, "factor_name"),
            factor_wiki_path=_optional_path_alias(
                payload,
                ("factor_wiki_path", "wiki_path"),
            ),
            out_of_sample_result_path=_optional_path_alias(
                payload,
                (
                    "out_of_sample_result_path",
                    "oos_result_path",
                    "validation_result_path",
                ),
            ),
            report_path=_optional_path(payload, "report_path"),
            report_title=_optional_str(payload, "report_title"),
            output_language=_optional_output_language(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_memory_record": self.memory_record is not None,
            "memory_path": str(self.memory_path) if self.memory_path else None,
            "memory_id": self.memory_id,
            "factor_name": self.factor_name,
            "factor_wiki_path": (
                str(self.factor_wiki_path) if self.factor_wiki_path else None
            ),
            "out_of_sample_result_path": (
                str(self.out_of_sample_result_path)
                if self.out_of_sample_result_path
                else None
            ),
            "report_path": str(self.report_path) if self.report_path else None,
            "report_title": self.report_title,
            "output_language": self.output_language,
        }


@dataclass(frozen=True, slots=True)
class ReportDraft:
    """Structured report draft before Markdown rendering."""

    document: dict[str, Any]

    @property
    def title(self) -> str:
        return str(self.document["title"])


@dataclass(frozen=True, slots=True)
class MarkdownReportResult:
    """Rendered Markdown report and persistence metadata."""

    report_path: Path
    bytes_written: int
    report_format: str = REPORT_FORMAT

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_path": str(self.report_path),
            "bytes_written": self.bytes_written,
            "report_format": self.report_format,
        }


class ReportAgent:
    """Build structured research report drafts and render Markdown reports."""

    name = "ReportAgent"

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
            "Received report request.",
            extra={"action": "validate_request", "status": "running"},
        )

        try:
            spec = ReportSpec.from_payload(request.payload)
        except ValueError as exc:
            elapsed = perf_counter() - started_at
            self.logger.warning(
                "Report request validation failed.",
                extra={"action": "validate_request", "status": "error"},
            )
            return AgentResponse.failure(
                str(exc),
                metadata=self._metadata(request, elapsed),
            )

        self.logger.info(
            "Loading report context.",
            extra={"action": "load_report_context", "status": "running"},
        )
        try:
            memory_record = self.load_memory_record(spec)
            factor_wiki_text = self.load_factor_wiki(spec.factor_wiki_path)
            out_of_sample_result = self.load_out_of_sample_result(
                spec.out_of_sample_result_path,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            elapsed = perf_counter() - started_at
            self.logger.warning(
                "Report context loading failed.",
                extra={"action": "load_report_context", "status": "error"},
            )
            return AgentResponse.failure(
                str(exc),
                metadata=self._metadata(request, elapsed),
            )
        self.logger.info(
            "Loaded report context.",
            extra={"action": "load_report_context", "status": "success"},
        )
        output_language = spec.output_language or self.config.output_language

        self.logger.info(
            "Building structured report draft.",
            extra={"action": "build_report_draft", "status": "running"},
        )
        try:
            report_draft = build_report_draft(
                memory_record,
                report_title=spec.report_title,
                factor_wiki_path=spec.factor_wiki_path,
                factor_wiki_text=factor_wiki_text,
                out_of_sample_result=out_of_sample_result,
                out_of_sample_result_path=spec.out_of_sample_result_path,
                output_language=output_language,
            )
        except (TypeError, ValueError) as exc:
            elapsed = perf_counter() - started_at
            self.logger.warning(
                "Structured report draft construction failed.",
                extra={"action": "build_report_draft", "status": "error"},
            )
            return AgentResponse.failure(
                str(exc),
                metadata=self._metadata(request, elapsed),
            )

        elapsed = perf_counter() - started_at
        self.logger.info(
            "Built structured report draft.",
            extra={"action": "build_report_draft", "status": "success"},
        )

        self.logger.info(
            "Generating markdown report.",
            extra={"action": "generate_markdown_report", "status": "running"},
        )
        try:
            markdown_report = render_report_markdown(report_draft.document)
            markdown_result = save_markdown_report(
                markdown_report,
                self.resolve_report_path(spec, report_draft),
            )
        except (OSError, TypeError, ValueError) as exc:
            elapsed = perf_counter() - started_at
            self.logger.warning(
                "Markdown report generation failed.",
                extra={"action": "generate_markdown_report", "status": "error"},
            )
            return AgentResponse.failure(
                str(exc),
                metadata=self._metadata(
                    request,
                    elapsed,
                    memory_id=report_draft.document["source"]["memory_id"],
                    factor_name=report_draft.document["factor"]["name"],
                    section_count=len(report_draft.document["sections"]),
                ),
            )

        elapsed = perf_counter() - started_at
        self.logger.info(
            "Generated markdown report.",
            extra={"action": "generate_markdown_report", "status": "success"},
        )
        return AgentResponse.success(
            output={
                "state": "markdown_report_generated",
                "request": spec.to_dict(),
                "report_draft": report_draft.document,
                "report_title": report_draft.title,
                "output_language": output_language,
                "section_count": len(report_draft.document["sections"]),
                "report_draft_format": REPORT_DRAFT_FORMAT,
                "report_format": REPORT_FORMAT,
                "report_markdown": markdown_report,
                "report_file": markdown_result.to_dict(),
                "report_path": str(markdown_result.report_path),
                "next_action": "Build Streamlit dashboard in Day 27.",
            },
            metadata=self._metadata(
                request,
                elapsed,
                memory_id=report_draft.document["source"]["memory_id"],
                factor_name=report_draft.document["factor"]["name"],
                section_count=len(report_draft.document["sections"]),
                report_path=markdown_result.report_path,
            ),
        )

    def load_memory_record(self, spec: ReportSpec) -> dict[str, Any]:
        if spec.memory_record is not None:
            return dict(spec.memory_record)
        if spec.memory_path is None:
            msg = "memory_path is required when memory_record is not provided."
            raise ValueError(msg)
        records = FactorMemoryStore(spec.memory_path).load_all()
        return select_memory_record(
            records,
            memory_id=spec.memory_id,
            factor_name=spec.factor_name,
        )

    def load_factor_wiki(self, factor_wiki_path: Path | None) -> str | None:
        if factor_wiki_path is None:
            return None
        if not factor_wiki_path.is_file():
            msg = f"Factor wiki file not found: {factor_wiki_path}."
            raise OSError(msg)
        return factor_wiki_path.read_text(encoding="utf-8")

    def load_out_of_sample_result(
        self,
        out_of_sample_result_path: Path | None,
    ) -> dict[str, Any] | None:
        if out_of_sample_result_path is None:
            return None
        if not out_of_sample_result_path.is_file():
            msg = f"Out-of-sample result file not found: {out_of_sample_result_path}."
            raise OSError(msg)
        value = json.loads(out_of_sample_result_path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            msg = "out_of_sample_result_path must contain a JSON object."
            raise ValueError(msg)
        return dict(value)

    def resolve_report_path(self, spec: ReportSpec, report_draft: ReportDraft) -> Path:
        if spec.report_path is not None:
            return spec.report_path
        memory_id = str(report_draft.document["source"]["memory_id"])
        factor_name = str(report_draft.document["factor"]["name"])
        stem = _safe_report_stem(f"{factor_name}_{memory_id}")
        return self.config.project_root / DEFAULT_REPORTS_DIRNAME / f"{stem}.md"

    def _metadata(
        self,
        request: AgentRequest,
        elapsed: float,
        *,
        memory_id: str | None = None,
        factor_name: str | None = None,
        section_count: int | None = None,
        report_path: Path | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "agent": self.name,
            "task_id": request.task_id,
            "execution_time_sec": round(elapsed, 6),
        }
        if memory_id is not None:
            metadata["memory_id"] = memory_id
        if factor_name is not None:
            metadata["factor_name"] = factor_name
        if section_count is not None:
            metadata["section_count"] = section_count
        if report_path is not None:
            metadata["report_path"] = str(report_path)
        return metadata


def select_memory_record(
    records: Sequence[Mapping[str, Any]],
    *,
    memory_id: str | None = None,
    factor_name: str | None = None,
) -> dict[str, Any]:
    """Select one memory record for report drafting."""

    if not records:
        msg = "No memory records are available for report drafting."
        raise ValueError(msg)
    if memory_id is not None:
        matches = [record for record in records if record.get("memory_id") == memory_id]
        if not matches:
            msg = f"No memory record found for memory_id: {memory_id}."
            raise ValueError(msg)
        return dict(matches[0])
    if factor_name is not None:
        matches = [
            record for record in records if _factor_name(record).lower() == factor_name.lower()
        ]
        if not matches:
            msg = f"No memory record found for factor_name: {factor_name}."
            raise ValueError(msg)
        return dict(sorted(matches, key=_record_sort_key)[-1])
    if len(records) == 1:
        return dict(records[0])
    msg = "payload.memory_id or payload.factor_name is required for multiple records."
    raise ValueError(msg)


def build_report_draft(
    memory_record: Mapping[str, Any],
    *,
    report_title: str | None = None,
    factor_wiki_path: Path | None = None,
    factor_wiki_text: str | None = None,
    out_of_sample_result: Mapping[str, Any] | None = None,
    out_of_sample_result_path: Path | None = None,
    output_language: str | None = None,
) -> ReportDraft:
    """Build a structured report draft without rendering Markdown."""

    language = _normalize_report_language(output_language)
    factor = _required_mapping(memory_record, "factor")
    performance = _required_mapping(memory_record, "performance")
    benchmark = _required_mapping(memory_record, "benchmark")
    diagnostics = _required_mapping(memory_record, "diagnostics")
    artifacts = _required_mapping(memory_record, "artifacts")
    source = _required_mapping(memory_record, "source")
    memory_id = _required_str(memory_record, "memory_id")
    factor_name = _required_str(factor, "name")

    document = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_format": REPORT_DRAFT_FORMAT,
        "output_language": language,
        "title": report_title or _default_report_title(factor_name, language),
        "source": {
            "memory_id": memory_id,
            "source_task_id": source.get("task_id"),
            "source_agent": source.get("agent"),
            "generated_at": source.get("generated_at"),
        },
        "factor": {
            "name": factor_name,
            "factor_column": factor.get("factor_column"),
            "source_type": factor.get("source_type"),
            "category": factor.get("category"),
            "formula": factor.get("formula"),
            "hypothesis": factor.get("hypothesis"),
            "direction": factor.get("direction"),
            "lookback_days": factor.get("lookback_days"),
            "data_lag_days": factor.get("data_lag_days"),
            "forward_return_days": factor.get("forward_return_days"),
            "universe": factor.get("universe"),
        },
        "context": {
            "factor_wiki_path": str(factor_wiki_path) if factor_wiki_path else None,
            "factor_wiki_line_count": (
                len(factor_wiki_text.splitlines()) if factor_wiki_text else 0
            ),
            "out_of_sample_result_path": (
                str(out_of_sample_result_path) if out_of_sample_result_path else None
            ),
        },
        "sections": [
            _hypothesis_section(factor, output_language=language),
            _factor_formula_section(factor, diagnostics, output_language=language),
            _backtest_results_section(performance, benchmark, output_language=language),
            _out_of_sample_validation_section(
                out_of_sample_result,
                out_of_sample_result_path=out_of_sample_result_path,
                output_language=language,
            ),
            _risk_analysis_section(
                performance,
                benchmark,
                diagnostics,
                artifacts,
                output_language=language,
            ),
            _conclusion_section(
                performance,
                benchmark,
                diagnostics,
                out_of_sample_result,
                output_language=language,
            ),
        ],
        "next_action": "Build Streamlit dashboard in Day 27.",
    }
    json.dumps(document, ensure_ascii=True, allow_nan=False)
    return ReportDraft(document=document)


def render_report_markdown(report_draft: Mapping[str, Any]) -> str:
    """Render a structured report draft to Markdown."""

    title = _required_str(report_draft, "title")
    output_language = _normalize_report_language(report_draft.get("output_language"))
    source = _required_mapping(report_draft, "source")
    factor = _required_mapping(report_draft, "factor")
    sections = _required_sequence(report_draft, "sections")
    lines = [
        f"# {title}",
        "",
        f"## {_report_label('metadata', output_language)}",
        "",
        f"- {_report_label('memory_id', output_language)}: "
        f"`{_string_or_na(source.get('memory_id'))}`",
        f"- {_report_label('source_task', output_language)}: "
        f"`{_string_or_na(source.get('source_task_id'))}`",
        f"- {_report_label('source_agent', output_language)}: "
        f"{_string_or_na(source.get('source_agent'))}",
        f"- {_report_label('generated_at', output_language)}: "
        f"{_string_or_na(source.get('generated_at'))}",
        "",
        f"## {_report_label('factor', output_language)}",
        "",
        f"- {_report_label('name', output_language)}: {_string_or_na(factor.get('name'))}",
        f"- {_report_label('factor_column', output_language)}: "
        f"{_string_or_na(factor.get('factor_column'))}",
        f"- {_report_label('source_type', output_language)}: "
        f"{_format_markdown_value(factor.get('source_type'), output_language)}",
        f"- {_report_label('category', output_language)}: "
        f"{_string_or_na(factor.get('category'))}",
        f"- {_report_label('formula', output_language)}: "
        f"{_string_or_na(factor.get('formula'))}",
        f"- {_report_label('direction', output_language)}: "
        f"{_format_markdown_value(factor.get('direction'), output_language)}",
        f"- {_report_label('lookback_days', output_language)}: "
        f"{_string_or_na(factor.get('lookback_days'))}",
        f"- {_report_label('data_lag_days', output_language)}: "
        f"{_string_or_na(factor.get('data_lag_days'))}",
        f"- {_report_label('universe', output_language)}: "
        f"{_string_or_na(factor.get('universe'))}",
        "",
    ]

    for section in sections:
        if not isinstance(section, Mapping):
            msg = "report_draft.sections must contain objects."
            raise ValueError(msg)
        lines.extend(_render_section(section, output_language=output_language))

    return "\n".join(lines).rstrip() + "\n"


def save_markdown_report(markdown_report: str, report_path: Path) -> MarkdownReportResult:
    """Persist a Markdown report atomically."""

    if not markdown_report.strip():
        msg = "markdown_report must not be empty."
        raise ValueError(msg)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(f"{report_path}.tmp")
    temp_path.write_text(markdown_report, encoding="utf-8")
    temp_path.replace(report_path)
    return MarkdownReportResult(
        report_path=report_path,
        bytes_written=len(markdown_report.encode("utf-8")),
    )


def _hypothesis_section(
    factor: Mapping[str, Any],
    *,
    output_language: OutputLanguage,
) -> dict[str, Any]:
    return _section(
        "hypothesis",
        {
            "hypothesis": factor.get("hypothesis"),
            "direction": factor.get("direction"),
            "universe": factor.get("universe"),
            "forward_return_days": factor.get("forward_return_days"),
        },
        output_language=output_language,
    )


def _factor_formula_section(
    factor: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    *,
    output_language: OutputLanguage,
) -> dict[str, Any]:
    return _section(
        "factor_formula",
        {
            "formula": factor.get("formula"),
            "related_factors": list(_string_sequence(diagnostics.get("related_factors"))),
            "paper_reference": diagnostics.get("paper_reference"),
        },
        output_language=output_language,
    )


def _backtest_results_section(
    performance: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    *,
    output_language: OutputLanguage,
) -> dict[str, Any]:
    return _section(
        "backtest_results",
        {
            "ic": performance.get("ic"),
            "rank_ic": performance.get("rank_ic"),
            "sharpe": performance.get("sharpe"),
            "net_sharpe": performance.get("net_sharpe"),
            "gross_sharpe": performance.get("gross_sharpe"),
            "max_drawdown": performance.get("max_drawdown"),
            "max_drawdown_abs": performance.get("max_drawdown_abs"),
            "total_return": performance.get("total_return"),
            "net_total_return": performance.get("net_total_return"),
            "gross_total_return": performance.get("gross_total_return"),
            "turnover": performance.get("turnover"),
            "average_transaction_cost": performance.get("average_transaction_cost"),
            "total_transaction_cost": performance.get("total_transaction_cost"),
            "benchmark_status": benchmark.get("status"),
            "passed_count": benchmark.get("passed_count"),
            "failed_count": benchmark.get("failed_count"),
        },
        output_language=output_language,
    )


def _out_of_sample_validation_section(
    out_of_sample_result: Mapping[str, Any] | None,
    *,
    out_of_sample_result_path: Path | None,
    output_language: OutputLanguage,
) -> dict[str, Any]:
    summary = _out_of_sample_report_summary(
        out_of_sample_result,
        out_of_sample_result_path=out_of_sample_result_path,
    )
    return _section(
        "out_of_sample_validation",
        {
            "out_of_sample_status": summary["out_of_sample_status"],
            "out_of_sample_passed": summary["out_of_sample_passed"],
            "validation_id": summary["validation_id"],
            "validation_method": summary["validation_method"],
            "split_count": summary["split_count"],
            "basic_oos_status": summary["basic_oos_status"],
            "walk_forward_status": summary["walk_forward_status"],
            "metric_comparison_status": summary["metric_comparison_status"],
            "in_sample_split_names": summary["in_sample_split_names"],
            "out_of_sample_split_names": summary["out_of_sample_split_names"],
            "out_of_sample_mean_rank_ic": summary["out_of_sample_mean_rank_ic"],
            "out_of_sample_net_sharpe": summary["out_of_sample_net_sharpe"],
            "out_of_sample_net_total_return": summary[
                "out_of_sample_net_total_return"
            ],
            "failure_reason": summary["failure_reason"],
            "result_json_path": summary["result_json_path"],
        },
        output_language=output_language,
    )


def _risk_analysis_section(
    performance: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    *,
    output_language: OutputLanguage,
) -> dict[str, Any]:
    return _section(
        "risk_analysis",
        {
            "max_drawdown_abs": performance.get("max_drawdown_abs"),
            "average_turnover": performance.get("average_turnover"),
            "transaction_costs": diagnostics.get("transaction_costs"),
            "transaction_cost_stats": diagnostics.get("transaction_cost_stats"),
            "failed_tests": list(_string_sequence(benchmark.get("failed_tests"))),
            "failure_reason": diagnostics.get("failure_reason"),
            "market_condition": diagnostics.get("market_condition"),
            "result_json_path": artifacts.get("result_json_path"),
            "factor_matrix_path": artifacts.get("factor_matrix_path"),
        },
        output_language=output_language,
    )


def _conclusion_section(
    performance: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    out_of_sample_result: Mapping[str, Any] | None,
    *,
    output_language: OutputLanguage,
) -> dict[str, Any]:
    benchmark_status = str(benchmark.get("status") or "unknown")
    oos_summary = _out_of_sample_report_summary(
        out_of_sample_result,
        out_of_sample_result_path=None,
    )
    verdict = "candidate_for_follow_up" if benchmark_status == "passed" else "needs_review"
    if oos_summary["out_of_sample_status"] == "failed":
        verdict = "needs_review"
    return _section(
        "conclusion",
        {
            "verdict": verdict,
            "benchmark_status": benchmark_status,
            "out_of_sample_status": oos_summary["out_of_sample_status"],
            "out_of_sample_passed": oos_summary["out_of_sample_passed"],
            "key_metric": {
                "rank_ic": performance.get("rank_ic"),
                "net_sharpe": performance.get("net_sharpe"),
                "gross_sharpe": performance.get("gross_sharpe"),
            },
            "failure_reason": diagnostics.get("failure_reason"),
        },
        output_language=output_language,
    )


def _out_of_sample_report_summary(
    out_of_sample_result: Mapping[str, Any] | None,
    *,
    out_of_sample_result_path: Path | None,
) -> dict[str, Any]:
    if out_of_sample_result is None:
        return {
            "out_of_sample_status": "not_provided",
            "out_of_sample_passed": False,
            "validation_id": None,
            "validation_method": None,
            "split_count": None,
            "basic_oos_status": None,
            "walk_forward_status": None,
            "metric_comparison_status": None,
            "in_sample_split_names": [],
            "out_of_sample_split_names": [],
            "out_of_sample_mean_rank_ic": None,
            "out_of_sample_net_sharpe": None,
            "out_of_sample_net_total_return": None,
            "failure_reason": "out_of_sample_result_not_provided",
            "result_json_path": None,
        }

    summary = _required_mapping(out_of_sample_result, "summary")
    request = _mapping_or_empty(out_of_sample_result.get("request"))
    basic_oos_check = _mapping_or_empty(summary.get("basic_oos_check"))
    walk_forward_check = _mapping_or_empty(summary.get("walk_forward_check"))
    metric_comparison = _mapping_or_empty(summary.get("metric_comparison"))
    metrics = _mapping_or_empty(metric_comparison.get("metrics"))

    validation_status = str(summary.get("status") or "unknown")
    basic_oos_status = str(basic_oos_check.get("status") or "unknown")
    walk_forward_status = str(
        walk_forward_check.get("status")
        or ("not_applicable" if not walk_forward_check else "unknown")
    )
    metric_comparison_status = str(metric_comparison.get("status") or "unknown")
    passed = (
        validation_status == "success"
        and basic_oos_status == "passed"
        and walk_forward_status in {"passed", "not_applicable"}
    )
    out_of_sample_status = "passed" if passed else "failed"
    failed_split_names = list(_string_sequence(summary.get("failed_split_names")))
    failure_reason = (
        "out_of_sample_validation_passed"
        if passed
        else _out_of_sample_failure_reason(
            validation_status=validation_status,
            basic_oos_status=basic_oos_status,
            walk_forward_status=walk_forward_status,
            failed_split_names=failed_split_names,
        )
    )

    return {
        "out_of_sample_status": out_of_sample_status,
        "out_of_sample_passed": passed,
        "validation_id": out_of_sample_result.get("validation_id"),
        "validation_method": request.get("validation_method"),
        "split_count": summary.get("split_count"),
        "basic_oos_status": basic_oos_status,
        "walk_forward_status": walk_forward_status,
        "metric_comparison_status": metric_comparison_status,
        "in_sample_split_names": list(
            _string_sequence(metric_comparison.get("in_sample_split_names"))
        ),
        "out_of_sample_split_names": list(
            _string_sequence(metric_comparison.get("out_of_sample_split_names"))
        )
        or list(_string_sequence(basic_oos_check.get("out_of_sample_split_names"))),
        "out_of_sample_mean_rank_ic": _comparison_metric_value(
            metrics,
            "mean_rank_ic",
            "out_of_sample_mean",
        ),
        "out_of_sample_net_sharpe": _comparison_metric_value(
            metrics,
            "net_sharpe",
            "out_of_sample_mean",
        ),
        "out_of_sample_net_total_return": _comparison_metric_value(
            metrics,
            "net_total_return",
            "out_of_sample_mean",
        ),
        "failure_reason": failure_reason,
        "result_json_path": str(out_of_sample_result_path)
        if out_of_sample_result_path is not None
        else None,
    }


def _out_of_sample_failure_reason(
    *,
    validation_status: str,
    basic_oos_status: str,
    walk_forward_status: str,
    failed_split_names: Sequence[str],
) -> str:
    reasons = [
        f"validation_status={validation_status}",
        f"basic_oos_status={basic_oos_status}",
        f"walk_forward_status={walk_forward_status}",
    ]
    if failed_split_names:
        reasons.append("failed_splits=" + ",".join(failed_split_names))
    return "; ".join(reasons)


def _comparison_metric_value(
    metrics: Mapping[str, Any],
    metric_key: str,
    value_key: str,
) -> Any:
    metric = metrics.get(metric_key)
    if not isinstance(metric, Mapping):
        return None
    return metric.get(value_key)


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _section(
    section_id: str,
    content: Mapping[str, Any],
    *,
    output_language: OutputLanguage,
) -> dict[str, Any]:
    return {
        "id": section_id,
        "title": _section_title(section_id, output_language),
        "content": dict(content),
    }


def _record_sort_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        _factor_name(record).lower(),
        str(record.get("created_at") or ""),
        str(record.get("memory_id") or ""),
    )


def _factor_name(record: Mapping[str, Any]) -> str:
    factor = _required_mapping(record, "factor")
    return _required_str(factor, "name")


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        msg = f"{key} must be an object."
        raise ValueError(msg)
    return value


def _required_sequence(payload: Mapping[str, Any], key: str) -> Sequence[Any]:
    value = payload.get(key)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = f"{key} must be a sequence."
        raise ValueError(msg)
    return value


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        msg = f"{key} must be a non-empty string."
        raise ValueError(msg)
    return value.strip()


def _optional_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any] | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        msg = f"payload.{key} must be an object when provided."
        raise ValueError(msg)
    return dict(value)


def _optional_path(payload: Mapping[str, Any], key: str) -> Path | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        msg = f"payload.{key} must be a non-empty path string when provided."
        raise ValueError(msg)
    return Path(value.strip()).expanduser().resolve()


def _optional_path_alias(
    payload: Mapping[str, Any],
    keys: Sequence[str],
) -> Path | None:
    for key in keys:
        if key in payload:
            return _optional_path(payload, key)
    return None


def _optional_str(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        msg = f"payload.{key} must be a non-empty string when provided."
        raise ValueError(msg)
    return value.strip()


def _optional_output_language(payload: Mapping[str, Any]) -> OutputLanguage | None:
    value = payload.get("output_language")
    if value is None:
        return None
    if not isinstance(value, str):
        msg = "payload.output_language must be a string when provided."
        raise ValueError(msg)
    return normalize_output_language(value)


def _string_sequence(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _render_section(
    section: Mapping[str, Any],
    *,
    output_language: OutputLanguage,
) -> list[str]:
    title = _required_str(section, "title")
    content = _required_mapping(section, "content")
    lines = [f"## {title}", ""]
    for key, value in content.items():
        lines.append(
            f"- {_report_label(str(key), output_language)}: "
            f"{_format_markdown_value(value, output_language)}"
        )
    lines.append("")
    return lines


def _format_markdown_value(value: Any, output_language: OutputLanguage) -> str:
    if isinstance(value, Mapping):
        rendered_items = [
            f"{_report_label(str(key), output_language)}="
            f"{_format_markdown_value(item, output_language)}"
            for key, item in value.items()
        ]
        return ", ".join(rendered_items) if rendered_items else "N/A"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rendered_values = [_format_markdown_value(item, output_language) for item in value]
        return ", ".join(rendered_values) if rendered_values else "N/A"
    if isinstance(value, str):
        localized = REPORT_VALUE_LABELS.get(value.strip().lower())
        if localized is not None:
            return localized.render(output_language)
    return _string_or_na(value)


def _string_or_na(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else "N/A"
    return str(value)


def _default_report_title(factor_name: str, output_language: OutputLanguage) -> str:
    return render_label(
        f"Research Report: {factor_name}",
        f"研究报告：{factor_name}",
        output_language,
    )


def _section_title(section_id: str, output_language: OutputLanguage) -> str:
    title = REPORT_SECTION_TITLES.get(section_id)
    if title is None:
        return _humanize_key(section_id)
    return title.render(output_language)


def _report_label(key: str, output_language: OutputLanguage) -> str:
    label = REPORT_LABELS.get(key)
    if label is not None:
        return label.render(output_language)
    return _humanize_key(key)


def _normalize_report_language(value: Any) -> OutputLanguage:
    return normalize_output_language(
        value if isinstance(value, str) else None,
        default=DEFAULT_OUTPUT_LANGUAGE,
    )


def _humanize_key(key: str) -> str:
    return key.replace("_", " ").strip().capitalize()


def _safe_report_stem(value: str) -> str:
    stem = _SAFE_REPORT_STEM_PATTERN.sub("_", value.strip()).strip("._")
    return stem or "factor_report"
