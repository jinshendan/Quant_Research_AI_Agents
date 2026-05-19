from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.i18n import (
    DEFAULT_OUTPUT_LANGUAGE,
    LocalizedText,
    OutputLanguage,
    normalize_output_language,
    render_label,
    render_paragraph,
)

DEFAULT_FACTOR_WIKI_FILENAME = "factor_wiki.md"
FACTOR_WIKI_FORMAT = "markdown"

WIKI_LABELS = {
    "title": LocalizedText(en="Quant Factor Wiki", zh="量化因子知识库"),
    "summary": LocalizedText(en="Summary", zh="摘要"),
    "factors": LocalizedText(en="Factors", zh="因子"),
    "factor": LocalizedText(en="Factor", zh="因子"),
    "benchmark": LocalizedText(en="Benchmark", zh="基准"),
    "ic": LocalizedText(en="IC", zh="IC"),
    "rank_ic": LocalizedText(en="RankIC", zh="RankIC"),
    "sharpe": LocalizedText(en="Sharpe", zh="夏普"),
    "max_drawdown": LocalizedText(en="Max Drawdown", zh="最大回撤"),
    "total_return": LocalizedText(en="Total Return", zh="总收益"),
    "failure_reason": LocalizedText(en="Failure Reason", zh="失败原因"),
    "passed_benchmarks": LocalizedText(en="Passed benchmarks", zh="通过基准"),
    "failed_benchmarks": LocalizedText(en="Failed benchmarks", zh="未通过基准"),
    "memory_id": LocalizedText(en="Memory ID", zh="记忆 ID"),
    "source_task": LocalizedText(en="Source task", zh="来源任务"),
    "formula": LocalizedText(en="Formula", zh="公式"),
    "hypothesis": LocalizedText(en="Hypothesis", zh="假设"),
    "direction": LocalizedText(en="Direction", zh="方向"),
    "forward_return_days": LocalizedText(en="Forward return days", zh="前瞻收益天数"),
    "universe": LocalizedText(en="Universe", zh="股票池"),
    "turnover": LocalizedText(en="Turnover", zh="换手率"),
    "benchmark_status": LocalizedText(en="Benchmark status", zh="基准状态"),
    "failed_tests": LocalizedText(en="Failed tests", zh="失败测试"),
    "market_condition": LocalizedText(en="Market condition", zh="市场条件"),
    "related_factors": LocalizedText(en="Related factors", zh="相关因子"),
    "paper_reference": LocalizedText(en="Paper reference", zh="文献/资料引用"),
    "result_json": LocalizedText(en="Result JSON", zh="结果 JSON"),
    "factor_matrix": LocalizedText(en="Factor matrix", zh="因子矩阵"),
    "aligned_data": LocalizedText(en="Aligned data", zh="对齐行情数据"),
}


@dataclass(frozen=True, slots=True)
class FactorWikiBuildResult:
    """Paths and counts for a saved factor wiki."""

    wiki_path: Path
    record_count: int
    factor_count: int
    passed_count: int
    failed_count: int
    output_language: OutputLanguage = DEFAULT_OUTPUT_LANGUAGE
    wiki_format: str = FACTOR_WIKI_FORMAT

    def to_dict(self) -> dict[str, Any]:
        return {
            "wiki_path": str(self.wiki_path),
            "record_count": self.record_count,
            "factor_count": self.factor_count,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "output_language": self.output_language,
            "wiki_format": self.wiki_format,
        }


class FactorWikiStore:
    """File-backed Markdown wiki for factor research memory records."""

    def __init__(
        self,
        wiki_path: str | Path,
        *,
        output_language: str | None = None,
    ) -> None:
        self.wiki_path = Path(wiki_path)
        self.output_language = normalize_output_language(output_language)

    def save(self, records: Sequence[Mapping[str, Any]]) -> FactorWikiBuildResult:
        markdown = build_factor_wiki_markdown(
            records,
            output_language=self.output_language,
        )
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
            output_language=self.output_language,
        )


def build_factor_wiki_markdown(
    records: Sequence[Mapping[str, Any]],
    *,
    output_language: str | None = None,
) -> str:
    """Render factor memory records into a deterministic Markdown wiki."""

    language = normalize_output_language(output_language)
    sorted_records = sorted(records, key=_record_sort_key)
    summary = summarize_factor_wiki_records(sorted_records)
    generated_sentence = render_paragraph(
        en=f"Generated from {summary['record_count']} factor memory records.",
        zh=f"由 {summary['record_count']} 条因子记忆记录生成。",
        language=language,
    )
    lines = [
        f"# {_wiki_label('title', language)}",
        "",
        generated_sentence,
        "",
        f"## {_wiki_label('summary', language)}",
        "",
        f"- {_wiki_label('factors', language)}: {summary['factor_count']}",
        f"- {_wiki_label('passed_benchmarks', language)}: {summary['passed_count']}",
        f"- {_wiki_label('failed_benchmarks', language)}: {summary['failed_count']}",
        "",
        "| "
        + " | ".join(
            [
                _wiki_label("factor", language),
                _wiki_label("benchmark", language),
                _wiki_label("ic", language),
                _wiki_label("rank_ic", language),
                _wiki_label("sharpe", language),
                _wiki_label("max_drawdown", language),
                _wiki_label("total_return", language),
                _wiki_label("failure_reason", language),
            ]
        )
        + " |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for record in sorted_records:
        lines.append(_summary_row(record))

    lines.extend(["", f"## {_wiki_label('factors', language)}", ""])
    if not sorted_records:
        lines.append(
            render_label(
                "No factor memory records saved yet.",
                "尚未保存因子记忆记录。",
                language,
            )
        )
    for record in sorted_records:
        lines.extend(_factor_section(record, output_language=language))

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


def _factor_section(
    record: Mapping[str, Any],
    *,
    output_language: OutputLanguage,
) -> list[str]:
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
        f"- {_wiki_label('memory_id', output_language)}: `{_text(record.get('memory_id'))}`",
        f"- {_wiki_label('source_task', output_language)}: `{_text(source.get('task_id'))}`",
        f"- {_wiki_label('formula', output_language)}: {_text_or_na(factor.get('formula'))}",
        f"- {_wiki_label('hypothesis', output_language)}: "
        f"{_text_or_na(factor.get('hypothesis'))}",
        f"- {_wiki_label('direction', output_language)}: "
        f"{_text_or_na(factor.get('direction'))}",
        f"- {_wiki_label('forward_return_days', output_language)}: "
        f"{_text_or_na(factor.get('forward_return_days'))}",
        f"- {_wiki_label('universe', output_language)}: "
        f"{_text_or_na(factor.get('universe'))}",
        f"- {_wiki_label('ic', output_language)}: {_format_number(performance.get('ic'))}",
        f"- {_wiki_label('rank_ic', output_language)}: "
        f"{_format_number(performance.get('rank_ic'))}",
        f"- {_wiki_label('sharpe', output_language)}: "
        f"{_format_number(performance.get('sharpe'))}",
        f"- {_wiki_label('max_drawdown', output_language)}: "
        f"{_format_number(performance.get('max_drawdown'))}",
        f"- {_wiki_label('total_return', output_language)}: "
        f"{_format_number(performance.get('total_return'))}",
        f"- {_wiki_label('turnover', output_language)}: "
        f"{_format_number(performance.get('turnover'))}",
        f"- {_wiki_label('benchmark_status', output_language)}: "
        f"{_text_or_na(benchmark.get('status'))}",
        f"- {_wiki_label('failed_tests', output_language)}: "
        f"{_format_sequence(benchmark.get('failed_tests'))}",
        f"- {_wiki_label('failure_reason', output_language)}: "
        f"{_text_or_na(diagnostics.get('failure_reason'))}",
        f"- {_wiki_label('market_condition', output_language)}: "
        f"{_text_or_na(diagnostics.get('market_condition'))}",
        f"- {_wiki_label('related_factors', output_language)}: "
        f"{_format_sequence(diagnostics.get('related_factors'))}",
        f"- {_wiki_label('paper_reference', output_language)}: "
        f"{_text_or_na(diagnostics.get('paper_reference'))}",
        f"- {_wiki_label('result_json', output_language)}: "
        f"{_text_or_na(artifacts.get('result_json_path'))}",
        f"- {_wiki_label('factor_matrix', output_language)}: "
        f"{_text_or_na(artifacts.get('factor_matrix_path'))}",
        f"- {_wiki_label('aligned_data', output_language)}: "
        f"{_text_or_na(artifacts.get('aligned_data_path'))}",
        "",
    ]
    return lines


def _wiki_label(key: str, output_language: OutputLanguage) -> str:
    label = WIKI_LABELS.get(key)
    if label is None:
        return key.replace("_", " ").strip().capitalize()
    return label.render(output_language)


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
