from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_FACTOR_WIKI_FILENAME = "factor_wiki.md"
FACTOR_WIKI_FORMAT = "markdown"


@dataclass(frozen=True, slots=True)
class FactorWikiBuildResult:
    """Paths and counts for a saved factor wiki."""

    wiki_path: Path
    record_count: int
    factor_count: int
    passed_count: int
    failed_count: int
    wiki_format: str = FACTOR_WIKI_FORMAT

    def to_dict(self) -> dict[str, Any]:
        return {
            "wiki_path": str(self.wiki_path),
            "record_count": self.record_count,
            "factor_count": self.factor_count,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "wiki_format": self.wiki_format,
        }


class FactorWikiStore:
    """File-backed Markdown wiki for factor research memory records."""

    def __init__(self, wiki_path: str | Path) -> None:
        self.wiki_path = Path(wiki_path)

    def save(self, records: Sequence[Mapping[str, Any]]) -> FactorWikiBuildResult:
        markdown = build_factor_wiki_markdown(records)
        self.wiki_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = Path(f"{self.wiki_path}.tmp")
        temp_path.write_text(markdown, encoding="utf-8")
        temp_path.replace(self.wiki_path)

        summary = summarize_factor_wiki_records(records)
        return FactorWikiBuildResult(
            wiki_path=self.wiki_path,
            record_count=summary["record_count"],
            factor_count=summary["factor_count"],
            passed_count=summary["passed_count"],
            failed_count=summary["failed_count"],
        )


def build_factor_wiki_markdown(records: Sequence[Mapping[str, Any]]) -> str:
    """Render factor memory records into a deterministic Markdown wiki."""

    sorted_records = sorted(records, key=_record_sort_key)
    summary = summarize_factor_wiki_records(sorted_records)
    lines = [
        "# Quant Factor Wiki",
        "",
        f"Generated from {summary['record_count']} factor memory records.",
        "",
        "## Summary",
        "",
        f"- Factors: {summary['factor_count']}",
        f"- Passed benchmarks: {summary['passed_count']}",
        f"- Failed benchmarks: {summary['failed_count']}",
        "",
        "| Factor | Benchmark | IC | RankIC | Sharpe | Max Drawdown | Total Return | Failure Reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for record in sorted_records:
        lines.append(_summary_row(record))

    lines.extend(["", "## Factors", ""])
    if not sorted_records:
        lines.append("No factor memory records saved yet.")
    for record in sorted_records:
        lines.extend(_factor_section(record))

    return "\n".join(lines).rstrip() + "\n"


def summarize_factor_wiki_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Compute high-level counts for a factor wiki."""

    factor_names = [_factor_name(record) for record in records]
    benchmark_statuses = Counter(_benchmark_status(record) for record in records)
    return {
        "record_count": len(records),
        "factor_count": len(set(factor_names)),
        "passed_count": benchmark_statuses["passed"],
        "failed_count": benchmark_statuses["failed"],
    }


def _summary_row(record: Mapping[str, Any]) -> str:
    factor = _required_mapping(record, "factor")
    performance = _required_mapping(record, "performance")
    benchmark = _required_mapping(record, "benchmark")
    diagnostics = _required_mapping(record, "diagnostics")
    return (
        "| "
        + " | ".join(
            [
                _markdown_cell(factor.get("name")),
                _markdown_cell(benchmark.get("status")),
                _markdown_cell(_format_number(performance.get("ic"))),
                _markdown_cell(_format_number(performance.get("rank_ic"))),
                _markdown_cell(_format_number(performance.get("sharpe"))),
                _markdown_cell(_format_number(performance.get("max_drawdown"))),
                _markdown_cell(_format_number(performance.get("total_return"))),
                _markdown_cell(diagnostics.get("failure_reason")),
            ]
        )
        + " |"
    )


def _factor_section(record: Mapping[str, Any]) -> list[str]:
    factor = _required_mapping(record, "factor")
    performance = _required_mapping(record, "performance")
    benchmark = _required_mapping(record, "benchmark")
    diagnostics = _required_mapping(record, "diagnostics")
    artifacts = _required_mapping(record, "artifacts")
    source = _required_mapping(record, "source")

    factor_name = _text(factor.get("name"))
    lines = [
        f"### {factor_name}",
        "",
        f"- Memory ID: `{_text(record.get('memory_id'))}`",
        f"- Source task: `{_text(source.get('task_id'))}`",
        f"- Formula: {_text_or_na(factor.get('formula'))}",
        f"- Hypothesis: {_text_or_na(factor.get('hypothesis'))}",
        f"- Direction: {_text_or_na(factor.get('direction'))}",
        f"- Forward return days: {_text_or_na(factor.get('forward_return_days'))}",
        f"- Universe: {_text_or_na(factor.get('universe'))}",
        f"- IC: {_format_number(performance.get('ic'))}",
        f"- RankIC: {_format_number(performance.get('rank_ic'))}",
        f"- Sharpe: {_format_number(performance.get('sharpe'))}",
        f"- Max drawdown: {_format_number(performance.get('max_drawdown'))}",
        f"- Total return: {_format_number(performance.get('total_return'))}",
        f"- Turnover: {_format_number(performance.get('turnover'))}",
        f"- Benchmark status: {_text_or_na(benchmark.get('status'))}",
        f"- Failed tests: {_format_sequence(benchmark.get('failed_tests'))}",
        f"- Failure reason: {_text_or_na(diagnostics.get('failure_reason'))}",
        f"- Market condition: {_text_or_na(diagnostics.get('market_condition'))}",
        f"- Related factors: {_format_sequence(diagnostics.get('related_factors'))}",
        f"- Paper reference: {_text_or_na(diagnostics.get('paper_reference'))}",
        f"- Result JSON: {_text_or_na(artifacts.get('result_json_path'))}",
        f"- Factor matrix: {_text_or_na(artifacts.get('factor_matrix_path'))}",
        f"- Aligned data: {_text_or_na(artifacts.get('aligned_data_path'))}",
        "",
    ]
    return lines


def _record_sort_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        _factor_name(record).lower(),
        _text(record.get("created_at")),
        _text(record.get("memory_id")),
    )


def _factor_name(record: Mapping[str, Any]) -> str:
    factor = _required_mapping(record, "factor")
    value = factor.get("name")
    if not isinstance(value, str) or not value.strip():
        msg = "factor.name must be a non-empty string."
        raise ValueError(msg)
    return value.strip()


def _benchmark_status(record: Mapping[str, Any]) -> str:
    benchmark = _required_mapping(record, "benchmark")
    value = benchmark.get("status")
    return str(value).strip().lower() if value is not None else "unknown"


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        msg = f"{key} must be an object."
        raise ValueError(msg)
    return value


def _markdown_cell(value: Any) -> str:
    return _text_or_na(value).replace("|", "\\|").replace("\n", " ")


def _format_number(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool) or not isinstance(value, int | float):
        return str(value)
    return f"{value:.6g}"


def _format_sequence(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value if value.strip() else "N/A"
    if isinstance(value, Sequence):
        items = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(items) if items else "N/A"
    return str(value)


def _text_or_na(value: Any) -> str:
    text = _text(value)
    return text if text else "N/A"


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
