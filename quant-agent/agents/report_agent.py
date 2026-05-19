from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from core.config import AppConfig
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
    ("risk_analysis", "Risk Analysis"),
    ("conclusion", "Conclusion"),
)
_SAFE_REPORT_STEM_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class ReportSpec:
    """Validated request for building a structured research report draft."""

    memory_record: dict[str, Any] | None = None
    memory_path: Path | None = None
    memory_id: str | None = None
    factor_name: str | None = None
    factor_wiki_path: Path | None = None
    report_path: Path | None = None
    report_title: str | None = None

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
            report_path=_optional_path(payload, "report_path"),
            report_title=_optional_str(payload, "report_title"),
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
            "report_path": str(self.report_path) if self.report_path else None,
            "report_title": self.report_title,
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
) -> ReportDraft:
    """Build a structured report draft without rendering Markdown."""

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
        "title": report_title or f"Research Report: {factor_name}",
        "source": {
            "memory_id": memory_id,
            "source_task_id": source.get("task_id"),
            "source_agent": source.get("agent"),
            "generated_at": source.get("generated_at"),
        },
        "factor": {
            "name": factor_name,
            "formula": factor.get("formula"),
            "hypothesis": factor.get("hypothesis"),
            "direction": factor.get("direction"),
            "forward_return_days": factor.get("forward_return_days"),
            "universe": factor.get("universe"),
        },
        "context": {
            "factor_wiki_path": str(factor_wiki_path) if factor_wiki_path else None,
            "factor_wiki_line_count": (
                len(factor_wiki_text.splitlines()) if factor_wiki_text else 0
            ),
        },
        "sections": [
            _hypothesis_section(factor),
            _factor_formula_section(factor, diagnostics),
            _backtest_results_section(performance, benchmark),
            _risk_analysis_section(performance, benchmark, diagnostics, artifacts),
            _conclusion_section(performance, benchmark, diagnostics),
        ],
        "next_action": "Build Streamlit dashboard in Day 27.",
    }
    json.dumps(document, ensure_ascii=True, allow_nan=False)
    return ReportDraft(document=document)


def render_report_markdown(report_draft: Mapping[str, Any]) -> str:
    """Render a structured report draft to Markdown."""

    title = _required_str(report_draft, "title")
    source = _required_mapping(report_draft, "source")
    factor = _required_mapping(report_draft, "factor")
    sections = _required_sequence(report_draft, "sections")
    lines = [
        f"# {title}",
        "",
        "## Metadata",
        "",
        f"- Memory ID: `{_string_or_na(source.get('memory_id'))}`",
        f"- Source task: `{_string_or_na(source.get('source_task_id'))}`",
        f"- Source agent: {_string_or_na(source.get('source_agent'))}",
        f"- Generated at: {_string_or_na(source.get('generated_at'))}",
        "",
        "## Factor",
        "",
        f"- Name: {_string_or_na(factor.get('name'))}",
        f"- Formula: {_string_or_na(factor.get('formula'))}",
        f"- Direction: {_string_or_na(factor.get('direction'))}",
        f"- Universe: {_string_or_na(factor.get('universe'))}",
        "",
    ]

    for section in sections:
        if not isinstance(section, Mapping):
            msg = "report_draft.sections must contain objects."
            raise ValueError(msg)
        lines.extend(_render_section(section))

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


def _hypothesis_section(factor: Mapping[str, Any]) -> dict[str, Any]:
    return _section(
        "hypothesis",
        {
            "hypothesis": factor.get("hypothesis"),
            "direction": factor.get("direction"),
            "universe": factor.get("universe"),
            "forward_return_days": factor.get("forward_return_days"),
        },
    )


def _factor_formula_section(
    factor: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    return _section(
        "factor_formula",
        {
            "formula": factor.get("formula"),
            "related_factors": list(_string_sequence(diagnostics.get("related_factors"))),
            "paper_reference": diagnostics.get("paper_reference"),
        },
    )


def _backtest_results_section(
    performance: Mapping[str, Any],
    benchmark: Mapping[str, Any],
) -> dict[str, Any]:
    return _section(
        "backtest_results",
        {
            "ic": performance.get("ic"),
            "rank_ic": performance.get("rank_ic"),
            "sharpe": performance.get("sharpe"),
            "max_drawdown": performance.get("max_drawdown"),
            "max_drawdown_abs": performance.get("max_drawdown_abs"),
            "total_return": performance.get("total_return"),
            "turnover": performance.get("turnover"),
            "benchmark_status": benchmark.get("status"),
            "passed_count": benchmark.get("passed_count"),
            "failed_count": benchmark.get("failed_count"),
        },
    )


def _risk_analysis_section(
    performance: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    return _section(
        "risk_analysis",
        {
            "max_drawdown_abs": performance.get("max_drawdown_abs"),
            "failed_tests": list(_string_sequence(benchmark.get("failed_tests"))),
            "failure_reason": diagnostics.get("failure_reason"),
            "market_condition": diagnostics.get("market_condition"),
            "result_json_path": artifacts.get("result_json_path"),
            "factor_matrix_path": artifacts.get("factor_matrix_path"),
        },
    )


def _conclusion_section(
    performance: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    benchmark_status = str(benchmark.get("status") or "unknown")
    verdict = "candidate_for_follow_up" if benchmark_status == "passed" else "needs_review"
    return _section(
        "conclusion",
        {
            "verdict": verdict,
            "benchmark_status": benchmark_status,
            "key_metric": {
                "rank_ic": performance.get("rank_ic"),
                "sharpe": performance.get("sharpe"),
            },
            "failure_reason": diagnostics.get("failure_reason"),
        },
    )


def _section(section_id: str, content: Mapping[str, Any]) -> dict[str, Any]:
    titles = dict(REPORT_SECTION_ORDER)
    return {
        "id": section_id,
        "title": titles[section_id],
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


def _string_sequence(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _render_section(section: Mapping[str, Any]) -> list[str]:
    title = _required_str(section, "title")
    content = _required_mapping(section, "content")
    lines = [f"## {title}", ""]
    for key, value in content.items():
        lines.append(f"- {_humanize_key(str(key))}: {_format_markdown_value(value)}")
    lines.append("")
    return lines


def _format_markdown_value(value: Any) -> str:
    if isinstance(value, Mapping):
        rendered_items = [
            f"{_humanize_key(str(key))}={_string_or_na(item)}"
            for key, item in value.items()
        ]
        return ", ".join(rendered_items) if rendered_items else "N/A"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rendered_values = [_string_or_na(item) for item in value]
        return ", ".join(rendered_values) if rendered_values else "N/A"
    return _string_or_na(value)


def _string_or_na(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else "N/A"
    return str(value)


def _humanize_key(key: str) -> str:
    return key.replace("_", " ").strip().capitalize()


def _safe_report_stem(value: str) -> str:
    stem = _SAFE_REPORT_STEM_PATTERN.sub("_", value.strip()).strip("._")
    return stem or "factor_report"
