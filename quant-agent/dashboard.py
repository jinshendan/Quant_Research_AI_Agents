from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents.factor_wiki import DEFAULT_FACTOR_WIKI_FILENAME, summarize_factor_wiki_records
from agents.memory_index import (
    DEFAULT_MEMORY_INDEX_FILENAME,
    DEFAULT_MEMORY_INDEX_METADATA_FILENAME,
    FactorMemoryVectorIndex,
    MemorySearchResult,
)
from agents.memory_agent import DEFAULT_MEMORY_FILENAME, FactorMemoryStore
from agents.report_agent import DEFAULT_REPORTS_DIRNAME
from core.config import AppConfig
from core.i18n import OutputLanguage, render_label
from core.logging import AgentLoggerAdapter, get_agent_logger

DASHBOARD_TITLE = "Quant Research Dashboard"
DEFAULT_HISTOGRAM_BINS = 10
FACTOR_RANKING_COLUMNS = [
    "rank",
    "factor_name",
    "memory_id",
    "benchmark_status",
    "rank_ic",
    "ic",
    "sharpe",
    "max_drawdown",
    "total_return",
    "turnover",
    "failure_reason",
    "created_at",
]
EXPLORER_PERFORMANCE_FIELDS = (
    "ic",
    "rank_ic",
    "sharpe",
    "max_drawdown",
    "max_drawdown_abs",
    "total_return",
    "turnover",
)


@dataclass(frozen=True, slots=True)
class DashboardPaths:
    """Artifact paths consumed by the Streamlit dashboard."""

    memory_path: Path
    memory_index_path: Path
    memory_index_metadata_path: Path
    wiki_path: Path
    report_dir: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "memory_path": str(self.memory_path),
            "memory_index_path": str(self.memory_index_path),
            "memory_index_metadata_path": str(self.memory_index_metadata_path),
            "wiki_path": str(self.wiki_path),
            "report_dir": str(self.report_dir),
        }


@dataclass(frozen=True, slots=True)
class MarkdownReportSummary:
    """Lightweight metadata for a generated Markdown report."""

    path: Path
    title: str
    byte_count: int
    modified_time: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "title": self.title,
            "byte_count": self.byte_count,
            "modified_time": self.modified_time,
        }


@dataclass(frozen=True, slots=True)
class FactorExplorerOption:
    """Selectable factor memory record in the dashboard explorer."""

    label: str
    factor_name: str
    memory_id: str
    created_at: str

    def to_dict(self) -> dict[str, str]:
        return {
            "label": self.label,
            "factor_name": self.factor_name,
            "memory_id": self.memory_id,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class FactorExplorerView:
    """Detailed factor view for a selected memory record."""

    factor_name: str
    memory_id: str
    title: str
    overview: dict[str, str]
    performance: dict[str, float | None]
    benchmark: dict[str, Any]
    diagnostics: dict[str, Any]
    artifacts: dict[str, str]
    report_summary: MarkdownReportSummary | None
    raw_record: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_name": self.factor_name,
            "memory_id": self.memory_id,
            "title": self.title,
            "overview": dict(self.overview),
            "performance": dict(self.performance),
            "benchmark": dict(self.benchmark),
            "diagnostics": dict(self.diagnostics),
            "artifacts": dict(self.artifacts),
            "report_summary": (
                self.report_summary.to_dict() if self.report_summary is not None else None
            ),
            "raw_record": dict(self.raw_record),
        }


@dataclass(frozen=True, slots=True)
class SemanticSearchMatchView:
    """Dashboard-ready semantic search match."""

    rank: int
    score: float
    memory_id: str
    factor_name: str
    benchmark_status: str
    report_summary: MarkdownReportSummary | None
    record: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": self.score,
            "memory_id": self.memory_id,
            "factor_name": self.factor_name,
            "benchmark_status": self.benchmark_status,
            "report_summary": (
                self.report_summary.to_dict() if self.report_summary is not None else None
            ),
            "record": dict(self.record),
        }


@dataclass(frozen=True, slots=True)
class SemanticSearchView:
    """Dashboard semantic search result state."""

    query: str
    top_k: int
    matches: tuple[SemanticSearchMatchView, ...]
    index_path: Path
    metadata_path: Path
    error: str | None = None

    @property
    def status(self) -> str:
        return "error" if self.error else "success"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "top_k": self.top_k,
            "match_count": len(self.matches),
            "matches": [match.to_dict() for match in self.matches],
            "index_path": str(self.index_path),
            "metadata_path": str(self.metadata_path),
            "status": self.status,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class DashboardData:
    """Loaded dashboard artifacts before Streamlit rendering."""

    records: list[dict[str, Any]]
    factor_wiki_markdown: str | None
    report_summaries: list[MarkdownReportSummary]


def default_dashboard_paths(config: AppConfig) -> DashboardPaths:
    """Resolve the dashboard's default local artifact paths."""

    return DashboardPaths(
        memory_path=config.memory_dir / DEFAULT_MEMORY_FILENAME,
        memory_index_path=config.memory_dir / DEFAULT_MEMORY_INDEX_FILENAME,
        memory_index_metadata_path=(
            config.memory_dir / DEFAULT_MEMORY_INDEX_METADATA_FILENAME
        ),
        wiki_path=config.memory_dir / DEFAULT_FACTOR_WIKI_FILENAME,
        report_dir=config.project_root / DEFAULT_REPORTS_DIRNAME,
    )


def load_dashboard_data(
    paths: DashboardPaths,
    *,
    logger: AgentLoggerAdapter | None = None,
) -> DashboardData:
    """Load persisted research artifacts for the dashboard."""

    dashboard_logger = logger or get_agent_logger("Dashboard")
    dashboard_logger.info(
        "Loading dashboard artifacts.",
        extra={"action": "load_dashboard_data", "status": "running"},
    )
    records = FactorMemoryStore(paths.memory_path).load_all()
    factor_wiki_markdown = (
        paths.wiki_path.read_text(encoding="utf-8") if paths.wiki_path.is_file() else None
    )
    report_summaries = load_markdown_report_summaries(paths.report_dir)
    dashboard_logger.info(
        "Loaded dashboard artifacts.",
        extra={"action": "load_dashboard_data", "status": "success"},
    )
    return DashboardData(
        records=records,
        factor_wiki_markdown=factor_wiki_markdown,
        report_summaries=report_summaries,
    )


def load_markdown_report_summaries(report_dir: Path) -> list[MarkdownReportSummary]:
    """Load metadata for generated Markdown reports."""

    if not report_dir.is_dir():
        return []

    report_summaries: list[MarkdownReportSummary] = []
    for path in sorted(report_dir.glob("*.md")):
        markdown = path.read_text(encoding="utf-8")
        stat = path.stat()
        report_summaries.append(
            MarkdownReportSummary(
                path=path,
                title=_extract_markdown_title(markdown) or path.stem,
                byte_count=stat.st_size,
                modified_time=stat.st_mtime,
            )
        )
    return sorted(report_summaries, key=lambda report: report.modified_time, reverse=True)


def build_dashboard_summary(
    records: Sequence[Mapping[str, Any]],
    report_summaries: Sequence[MarkdownReportSummary],
) -> dict[str, int]:
    """Build top-level dashboard counters."""

    summary = summarize_factor_wiki_records(records)
    return {
        **summary,
        "report_count": len(report_summaries),
    }


def build_factor_ranking_frame(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Build a factor ranking table sorted by RankIC and Sharpe."""

    rows = [_factor_ranking_row(record) for record in records]
    if not rows:
        return pd.DataFrame(columns=FACTOR_RANKING_COLUMNS)

    frame = pd.DataFrame(rows)
    frame["_rank_ic_sort"] = pd.to_numeric(frame["rank_ic"], errors="coerce").fillna(
        float("-inf")
    )
    frame["_sharpe_sort"] = pd.to_numeric(frame["sharpe"], errors="coerce").fillna(
        float("-inf")
    )
    frame = frame.sort_values(
        by=["_rank_ic_sort", "_sharpe_sort", "factor_name", "memory_id"],
        ascending=[False, False, True, True],
    ).drop(columns=["_rank_ic_sort", "_sharpe_sort"])
    frame = frame.reset_index(drop=True)
    frame.insert(0, "rank", range(1, len(frame) + 1))
    return frame[FACTOR_RANKING_COLUMNS]


def build_metric_distribution_frame(
    records: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    bins: int = DEFAULT_HISTOGRAM_BINS,
) -> pd.DataFrame:
    """Build histogram counts for a numeric factor metric."""

    values = [_metric_value(record, metric) for record in records]
    numeric_values = [value for value in values if value is not None]
    if not numeric_values:
        return pd.DataFrame(columns=["bucket", "count"])

    if len(set(numeric_values)) == 1:
        return pd.DataFrame(
            [{"bucket": _format_float(numeric_values[0]), "count": len(numeric_values)}],
            columns=["bucket", "count"],
        )

    bucket_count = min(max(1, bins), len(set(numeric_values)))
    counts, edges = np.histogram(numeric_values, bins=bucket_count)
    rows = [
        {
            "bucket": f"{_format_float(edges[index])} to {_format_float(edges[index + 1])}",
            "count": int(count),
        }
        for index, count in enumerate(counts)
    ]
    return pd.DataFrame(rows, columns=["bucket", "count"])


def build_factor_explorer_options(
    records: Sequence[Mapping[str, Any]],
) -> list[FactorExplorerOption]:
    """Build deterministic factor explorer choices from memory records."""

    options = []
    for record in records:
        factor_name = _factor_name(record)
        memory_id = _text_or_na(record.get("memory_id"))
        created_at = _text_or_na(record.get("created_at"))
        options.append(
            FactorExplorerOption(
                label=f"{factor_name} | {memory_id} | {created_at}",
                factor_name=factor_name,
                memory_id=memory_id,
                created_at=created_at,
            )
        )
    return sorted(
        options,
        key=lambda option: (option.factor_name.lower(), option.created_at, option.memory_id),
    )


def select_factor_record(
    records: Sequence[Mapping[str, Any]],
    *,
    memory_id: str | None = None,
    factor_name: str | None = None,
) -> dict[str, Any] | None:
    """Select a factor memory record for the explorer."""

    if memory_id is not None:
        for record in records:
            if _text_or_na(record.get("memory_id")) == memory_id:
                return dict(record)
        return None

    if factor_name is not None:
        matches = [
            record
            for record in records
            if _factor_name(record).lower() == factor_name.lower()
        ]
        if not matches:
            return None
        return dict(sorted(matches, key=_record_sort_key)[-1])

    return dict(sorted(records, key=_record_sort_key)[-1]) if records else None


def build_factor_explorer_view(
    record: Mapping[str, Any],
    *,
    report_summaries: Sequence[MarkdownReportSummary] = (),
) -> FactorExplorerView:
    """Build detail sections for one selected factor memory record."""

    factor = _mapping(record.get("factor"))
    performance = _mapping(record.get("performance"))
    benchmark = _mapping(record.get("benchmark"))
    diagnostics = _mapping(record.get("diagnostics"))
    artifacts = _mapping(record.get("artifacts"))
    source = _mapping(record.get("source"))
    factor_name = _factor_name(record)
    memory_id = _text_or_na(record.get("memory_id"))
    return FactorExplorerView(
        factor_name=factor_name,
        memory_id=memory_id,
        title=f"{factor_name} ({memory_id})",
        overview={
            "formula": _text_or_na(factor.get("formula")),
            "hypothesis": _text_or_na(factor.get("hypothesis")),
            "direction": _text_or_na(factor.get("direction")),
            "universe": _text_or_na(factor.get("universe")),
            "forward_return_days": _text_or_na(factor.get("forward_return_days")),
            "source_agent": _text_or_na(source.get("agent")),
            "source_task_id": _text_or_na(source.get("task_id")),
            "created_at": _text_or_na(record.get("created_at")),
        },
        performance={
            field: _to_float(performance.get(field))
            for field in EXPLORER_PERFORMANCE_FIELDS
        },
        benchmark=dict(benchmark),
        diagnostics=dict(diagnostics),
        artifacts={
            key: _text_or_na(value)
            for key, value in artifacts.items()
        },
        report_summary=match_report_summary(record, report_summaries),
        raw_record=dict(record),
    )


def match_report_summary(
    record: Mapping[str, Any],
    report_summaries: Sequence[MarkdownReportSummary],
) -> MarkdownReportSummary | None:
    """Find the most relevant generated report for a factor memory record."""

    if not report_summaries:
        return None

    memory_id = _text_or_na(record.get("memory_id")).lower()
    factor_name = _factor_name(record).lower()
    scored_reports = [
        (_report_match_score(report, factor_name=factor_name, memory_id=memory_id), report)
        for report in report_summaries
    ]
    scored_reports = [(score, report) for score, report in scored_reports if score > 0]
    if not scored_reports:
        return None
    return sorted(
        scored_reports,
        key=lambda item: (item[0], item[1].modified_time),
        reverse=True,
    )[0][1]


def run_semantic_memory_search(
    paths: DashboardPaths,
    query: str,
    *,
    top_k: int = 5,
    report_summaries: Sequence[MarkdownReportSummary] = (),
) -> SemanticSearchView:
    """Run semantic search over the saved FAISS factor memory index."""

    normalized_query = query.strip()
    if not normalized_query:
        return SemanticSearchView(
            query=normalized_query,
            top_k=top_k,
            matches=(),
            index_path=paths.memory_index_path,
            metadata_path=paths.memory_index_metadata_path,
            error="Search query must not be empty.",
        )

    try:
        search_result = FactorMemoryVectorIndex(
            index_path=paths.memory_index_path,
            metadata_path=paths.memory_index_metadata_path,
        ).search(normalized_query, top_k=top_k)
    except (ImportError, OSError, ValueError) as exc:
        return SemanticSearchView(
            query=normalized_query,
            top_k=top_k,
            matches=(),
            index_path=paths.memory_index_path,
            metadata_path=paths.memory_index_metadata_path,
            error=str(exc),
        )

    return build_semantic_search_view(
        search_result,
        top_k=top_k,
        report_summaries=report_summaries,
    )


def build_semantic_search_view(
    search_result: MemorySearchResult,
    *,
    top_k: int,
    report_summaries: Sequence[MarkdownReportSummary] = (),
) -> SemanticSearchView:
    """Convert a memory search result into dashboard-ready rows."""

    matches = tuple(
        SemanticSearchMatchView(
            rank=match.rank,
            score=match.score,
            memory_id=match.memory_id,
            factor_name=_text_or_na(match.factor_name),
            benchmark_status=_text_or_na(match.benchmark_status),
            report_summary=match_report_summary(match.record, report_summaries),
            record=dict(match.record),
        )
        for match in search_result.matches
    )
    return SemanticSearchView(
        query=search_result.query,
        top_k=top_k,
        matches=matches,
        index_path=search_result.index_path,
        metadata_path=search_result.metadata_path,
    )


def render_streamlit_dashboard(
    *,
    config: AppConfig | None = None,
    logger: AgentLoggerAdapter | None = None,
) -> None:
    """Render the Streamlit dashboard from local research artifacts."""

    import streamlit as st

    dashboard_config = config or AppConfig.from_env()
    output_language = dashboard_config.output_language
    dashboard_logger = logger or get_agent_logger("Dashboard")
    paths = default_dashboard_paths(dashboard_config)
    data = load_dashboard_data(paths, logger=dashboard_logger)
    summary = build_dashboard_summary(data.records, data.report_summaries)
    ranking_frame = build_factor_ranking_frame(data.records)
    ic_distribution = build_metric_distribution_frame(data.records, "ic")
    sharpe_distribution = build_metric_distribution_frame(data.records, "sharpe")
    explorer_options = build_factor_explorer_options(data.records)

    dashboard_logger.info(
        "Rendering dashboard.",
        extra={"action": "render_dashboard", "status": "running"},
    )
    dashboard_title = _ui_label(DASHBOARD_TITLE, "量化研究 Dashboard", output_language)
    st.set_page_config(page_title=dashboard_title, layout="wide")
    st.title(dashboard_title)

    with st.sidebar:
        st.header(_ui_label("Artifacts", "产物", output_language))
        st.caption(_ui_label("Memory", "记忆", output_language))
        st.code(str(paths.memory_path))
        st.caption(_ui_label("Memory Index", "记忆索引", output_language))
        st.code(str(paths.memory_index_path))
        st.caption(_ui_label("Wiki", "知识库", output_language))
        st.code(str(paths.wiki_path))
        st.caption(_ui_label("Reports", "报告", output_language))
        st.code(str(paths.report_dir))

    dashboard_tab, explorer_tab, search_tab = st.tabs(
        [
            _ui_label("Dashboard", "仪表盘", output_language),
            _ui_label("Factor Explorer", "因子浏览", output_language),
            _ui_label("Semantic Search", "语义搜索", output_language),
        ]
    )

    with dashboard_tab:
        metric_columns = st.columns(5)
        metric_columns[0].metric(_ui_label("Records", "记录", output_language), summary["record_count"])
        metric_columns[1].metric(_ui_label("Factors", "因子", output_language), summary["factor_count"])
        metric_columns[2].metric(_ui_label("Passed", "通过", output_language), summary["passed_count"])
        metric_columns[3].metric(_ui_label("Failed", "未通过", output_language), summary["failed_count"])
        metric_columns[4].metric(_ui_label("Reports", "报告", output_language), summary["report_count"])

        st.subheader(_ui_label("Factor Ranking", "因子排名", output_language))
        if ranking_frame.empty:
            st.info(_ui_label("No factor memory records found.", "未找到因子记忆记录。", output_language))
        else:
            st.dataframe(ranking_frame, use_container_width=True, hide_index=True)

        chart_columns = st.columns(2)
        with chart_columns[0]:
            st.subheader(_ui_label("IC Distribution", "IC 分布", output_language))
            _render_distribution_chart(st, ic_distribution, output_language=output_language)
        with chart_columns[1]:
            st.subheader(_ui_label("Sharpe Distribution", "夏普分布", output_language))
            _render_distribution_chart(st, sharpe_distribution, output_language=output_language)

        st.subheader(_ui_label("Generated Reports", "已生成报告", output_language))
        report_frame = pd.DataFrame(
            [report.to_dict() for report in data.report_summaries],
            columns=["title", "path", "byte_count", "modified_time"],
        )
        if report_frame.empty:
            st.info(_ui_label("No markdown reports found.", "未找到 Markdown 报告。", output_language))
        else:
            st.dataframe(report_frame, use_container_width=True, hide_index=True)

    with explorer_tab:
        st.subheader(_ui_label("Factor Explorer", "因子浏览", output_language))
        if not explorer_options:
            st.info(_ui_label("No factor memory records found.", "未找到因子记忆记录。", output_language))
        else:
            selected_label = st.selectbox(
                _ui_label("Factor memory record", "因子记忆记录", output_language),
                [option.label for option in explorer_options],
            )
            selected_option = next(
                option for option in explorer_options if option.label == selected_label
            )
            selected_record = select_factor_record(
                data.records,
                memory_id=selected_option.memory_id,
            )
            if selected_record is None:
                st.warning(
                    _ui_label(
                        "Selected factor memory record is no longer available.",
                        "所选因子记忆记录已不可用。",
                        output_language,
                    )
                )
            else:
                explorer_view = build_factor_explorer_view(
                    selected_record,
                    report_summaries=data.report_summaries,
                )
                _render_factor_explorer(
                    st,
                    explorer_view,
                    output_language=output_language,
                )

    with search_tab:
        st.subheader(_ui_label("Semantic Search", "语义搜索", output_language))
        search_query = st.text_input(
            _ui_label("Search factor memory", "搜索因子记忆", output_language),
            placeholder="momentum rank_ic sharpe drawdown",
        )
        top_k = st.slider(_ui_label("Matches", "匹配数量", output_language), min_value=1, max_value=10, value=5)
        if st.button(_ui_label("Search", "搜索", output_language), type="primary"):
            search_view = run_semantic_memory_search(
                paths,
                search_query,
                top_k=top_k,
                report_summaries=data.report_summaries,
            )
            _render_semantic_search(st, search_view, output_language=output_language)

    dashboard_logger.info(
        "Rendered dashboard.",
        extra={"action": "render_dashboard", "status": "success"},
    )


def main() -> None:
    render_streamlit_dashboard()


def _render_distribution_chart(
    st: Any,
    distribution_frame: pd.DataFrame,
    *,
    output_language: OutputLanguage,
) -> None:
    if distribution_frame.empty:
        st.info(
            _ui_label(
                "No numeric metric values found.",
                "未找到数值型指标。",
                output_language,
            )
        )
        return
    st.bar_chart(distribution_frame.set_index("bucket")["count"])


def _render_factor_explorer(
    st: Any,
    view: FactorExplorerView,
    *,
    output_language: OutputLanguage,
) -> None:
    st.markdown(f"### {view.title}")
    metric_columns = st.columns(4)
    metric_columns[0].metric("RankIC", _format_optional_float(view.performance["rank_ic"]))
    metric_columns[1].metric("IC", _format_optional_float(view.performance["ic"]))
    metric_columns[2].metric("Sharpe", _format_optional_float(view.performance["sharpe"]))
    metric_columns[3].metric(
        "Max Drawdown",
        _format_optional_float(view.performance["max_drawdown"]),
    )

    st.markdown(f"#### {_ui_label('Overview', '概览', output_language)}")
    st.table(_section_frame(view.overview, output_language=output_language))

    detail_columns = st.columns(2)
    with detail_columns[0]:
        st.markdown(f"#### {_ui_label('Performance', '表现', output_language)}")
        st.table(_section_frame(view.performance, output_language=output_language))
        st.markdown(f"#### {_ui_label('Benchmark', '基准', output_language)}")
        st.table(_section_frame(view.benchmark, output_language=output_language))
    with detail_columns[1]:
        st.markdown(f"#### {_ui_label('Diagnostics', '诊断', output_language)}")
        st.table(_section_frame(view.diagnostics, output_language=output_language))
        st.markdown(f"#### {_ui_label('Artifacts', '产物', output_language)}")
        st.table(_section_frame(view.artifacts, output_language=output_language))

    st.markdown(f"#### {_ui_label('Report', '报告', output_language)}")
    if view.report_summary is None:
        st.info(
            _ui_label(
                "No generated report matched this factor.",
                "没有匹配该因子的已生成报告。",
                output_language,
            )
        )
    else:
        st.table(
            _section_frame(
                view.report_summary.to_dict(),
                output_language=output_language,
            )
        )

    with st.expander(_ui_label("Raw memory record", "原始记忆记录", output_language)):
        st.json(view.raw_record)


def _render_semantic_search(
    st: Any,
    view: SemanticSearchView,
    *,
    output_language: OutputLanguage,
) -> None:
    if view.error is not None:
        st.warning(view.error)
        return

    if not view.matches:
        st.info(_ui_label("No semantic matches found.", "未找到语义匹配。", output_language))
        return

    result_frame = pd.DataFrame(
        [
            {
                "rank": match.rank,
                "score": match.score,
                "factor_name": match.factor_name,
                "memory_id": match.memory_id,
                "benchmark_status": match.benchmark_status,
                "report": (
                    match.report_summary.title
                    if match.report_summary is not None
                    else "N/A"
                ),
            }
            for match in view.matches
        ]
    )
    st.dataframe(result_frame, use_container_width=True, hide_index=True)
    for match in view.matches:
        with st.expander(
            f"#{match.rank} {match.factor_name} | score {_format_float(match.score)}"
        ):
            explorer_view = build_factor_explorer_view(
                match.record,
                report_summaries=[
                    match.report_summary
                ] if match.report_summary is not None else [],
            )
            _render_factor_explorer(
                st,
                explorer_view,
                output_language=output_language,
            )


def _section_frame(
    section: Mapping[str, Any],
    *,
    output_language: OutputLanguage,
) -> pd.DataFrame:
    rows = [
        {
            _ui_label("field", "字段", output_language): str(key),
            _ui_label("value", "值", output_language): _format_display_value(value),
        }
        for key, value in section.items()
    ]
    return pd.DataFrame(
        rows,
        columns=[
            _ui_label("field", "字段", output_language),
            _ui_label("value", "值", output_language),
        ],
    )


def _ui_label(en: str, zh: str, output_language: OutputLanguage) -> str:
    return render_label(en, zh, output_language)


def _factor_ranking_row(record: Mapping[str, Any]) -> dict[str, Any]:
    factor = _mapping(record.get("factor"))
    performance = _mapping(record.get("performance"))
    benchmark = _mapping(record.get("benchmark"))
    diagnostics = _mapping(record.get("diagnostics"))
    return {
        "factor_name": _text_or_na(factor.get("name")),
        "memory_id": _text_or_na(record.get("memory_id")),
        "benchmark_status": _text_or_na(benchmark.get("status")),
        "rank_ic": _to_float(performance.get("rank_ic")),
        "ic": _to_float(performance.get("ic")),
        "sharpe": _to_float(performance.get("sharpe")),
        "max_drawdown": _to_float(performance.get("max_drawdown")),
        "total_return": _to_float(performance.get("total_return")),
        "turnover": _to_float(performance.get("turnover")),
        "failure_reason": _text_or_na(diagnostics.get("failure_reason")),
        "created_at": _text_or_na(record.get("created_at")),
    }


def _record_sort_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        _factor_name(record).lower(),
        _text_or_na(record.get("created_at")),
        _text_or_na(record.get("memory_id")),
    )


def _factor_name(record: Mapping[str, Any]) -> str:
    factor = _mapping(record.get("factor"))
    return _text_or_na(factor.get("name"))


def _metric_value(record: Mapping[str, Any], metric: str) -> float | None:
    performance = _mapping(record.get("performance"))
    return _to_float(performance.get(metric))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text_or_na(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else "N/A"
    return str(value)


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _format_float(value: float) -> str:
    return f"{value:.4g}"


def _format_optional_float(value: float | None) -> str:
    return _format_float(value) if value is not None else "N/A"


def _format_display_value(value: Any) -> str:
    if isinstance(value, float):
        return _format_float(value)
    if isinstance(value, Mapping):
        return ", ".join(
            f"{key}={_format_display_value(item)}" for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rendered = [_format_display_value(item) for item in value]
        return ", ".join(rendered) if rendered else "N/A"
    return _text_or_na(value)


def _report_match_score(
    report: MarkdownReportSummary,
    *,
    factor_name: str,
    memory_id: str,
) -> int:
    haystack = f"{report.title} {report.path.stem} {report.path}".lower()
    score = 0
    if memory_id != "n/a" and memory_id in haystack:
        score += 2
    if factor_name != "n/a" and factor_name in haystack:
        score += 1
    return score


def _extract_markdown_title(markdown: str) -> str | None:
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            return title or None
    return None


if __name__ == "__main__":
    main()
