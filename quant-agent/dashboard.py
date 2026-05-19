from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents.factor_wiki import DEFAULT_FACTOR_WIKI_FILENAME, summarize_factor_wiki_records
from agents.memory_agent import DEFAULT_MEMORY_FILENAME, FactorMemoryStore
from agents.report_agent import DEFAULT_REPORTS_DIRNAME
from core.config import AppConfig
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


@dataclass(frozen=True, slots=True)
class DashboardPaths:
    """Artifact paths consumed by the Streamlit dashboard."""

    memory_path: Path
    wiki_path: Path
    report_dir: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "memory_path": str(self.memory_path),
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
class DashboardData:
    """Loaded dashboard artifacts before Streamlit rendering."""

    records: list[dict[str, Any]]
    factor_wiki_markdown: str | None
    report_summaries: list[MarkdownReportSummary]


def default_dashboard_paths(config: AppConfig) -> DashboardPaths:
    """Resolve the dashboard's default local artifact paths."""

    return DashboardPaths(
        memory_path=config.memory_dir / DEFAULT_MEMORY_FILENAME,
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


def render_streamlit_dashboard(
    *,
    config: AppConfig | None = None,
    logger: AgentLoggerAdapter | None = None,
) -> None:
    """Render the Streamlit dashboard from local research artifacts."""

    import streamlit as st

    dashboard_config = config or AppConfig.from_env()
    dashboard_logger = logger or get_agent_logger("Dashboard")
    paths = default_dashboard_paths(dashboard_config)
    data = load_dashboard_data(paths, logger=dashboard_logger)
    summary = build_dashboard_summary(data.records, data.report_summaries)
    ranking_frame = build_factor_ranking_frame(data.records)
    ic_distribution = build_metric_distribution_frame(data.records, "ic")
    sharpe_distribution = build_metric_distribution_frame(data.records, "sharpe")

    dashboard_logger.info(
        "Rendering dashboard.",
        extra={"action": "render_dashboard", "status": "running"},
    )
    st.set_page_config(page_title=DASHBOARD_TITLE, layout="wide")
    st.title(DASHBOARD_TITLE)

    with st.sidebar:
        st.header("Artifacts")
        st.caption("Memory")
        st.code(str(paths.memory_path))
        st.caption("Wiki")
        st.code(str(paths.wiki_path))
        st.caption("Reports")
        st.code(str(paths.report_dir))

    metric_columns = st.columns(5)
    metric_columns[0].metric("Records", summary["record_count"])
    metric_columns[1].metric("Factors", summary["factor_count"])
    metric_columns[2].metric("Passed", summary["passed_count"])
    metric_columns[3].metric("Failed", summary["failed_count"])
    metric_columns[4].metric("Reports", summary["report_count"])

    st.subheader("Factor Ranking")
    if ranking_frame.empty:
        st.info("No factor memory records found.")
    else:
        st.dataframe(ranking_frame, use_container_width=True, hide_index=True)

    chart_columns = st.columns(2)
    with chart_columns[0]:
        st.subheader("IC Distribution")
        _render_distribution_chart(st, ic_distribution)
    with chart_columns[1]:
        st.subheader("Sharpe Distribution")
        _render_distribution_chart(st, sharpe_distribution)

    st.subheader("Generated Reports")
    report_frame = pd.DataFrame(
        [report.to_dict() for report in data.report_summaries],
        columns=["title", "path", "byte_count", "modified_time"],
    )
    if report_frame.empty:
        st.info("No markdown reports found.")
    else:
        st.dataframe(report_frame, use_container_width=True, hide_index=True)

    dashboard_logger.info(
        "Rendered dashboard.",
        extra={"action": "render_dashboard", "status": "success"},
    )


def main() -> None:
    render_streamlit_dashboard()


def _render_distribution_chart(st: Any, distribution_frame: pd.DataFrame) -> None:
    if distribution_frame.empty:
        st.info("No numeric metric values found.")
        return
    st.bar_chart(distribution_frame.set_index("bucket")["count"])


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


def _extract_markdown_title(markdown: str) -> str | None:
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            return title or None
    return None


if __name__ == "__main__":
    main()
