from __future__ import annotations

from pathlib import Path

import pytest

from agents.factor_wiki import (
    FactorWikiStore,
    build_factor_wiki_markdown,
    summarize_factor_wiki_records,
)


def _memory_record(
    memory_id: str,
    *,
    name: str,
    benchmark_status: str = "passed",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "memory_id": memory_id,
        "created_at": "2026-05-18T10:00:00+00:00",
        "source": {"task_id": f"task-{memory_id}"},
        "factor": {
            "name": name,
            "formula": "rank(return_5d)",
            "hypothesis": "Short-term momentum continues.",
            "direction": "positive",
            "forward_return_days": 1,
            "universe": "CSI500",
        },
        "performance": {
            "ic": 0.04,
            "rank_ic": 0.05,
            "sharpe": 1.2,
            "max_drawdown": -0.1,
            "total_return": 0.2,
            "turnover": 0.18,
        },
        "benchmark": {
            "status": benchmark_status,
            "failed_tests": [] if benchmark_status == "passed" else ["sharpe"],
        },
        "diagnostics": {
            "failure_reason": (
                None
                if benchmark_status == "passed"
                else "Failed benchmark tests: sharpe"
            ),
            "market_condition": "short_horizon_cross_section",
            "related_factors": ["momentum", "return_5d"],
            "paper_reference": "internal-note",
        },
        "artifacts": {
            "result_json_path": "/tmp/result.json",
            "factor_matrix_path": "/tmp/factors.csv",
            "aligned_data_path": "/tmp/aligned.csv",
        },
    }


def test_build_factor_wiki_markdown_summarizes_and_lists_records() -> None:
    records = [
        _memory_record("memory-2", name="alpha_reversal", benchmark_status="failed"),
        _memory_record("memory-1", name="alpha_momentum"),
    ]

    markdown = build_factor_wiki_markdown(records, output_language="en")

    assert markdown.startswith("# Quant Factor Wiki")
    assert "- Factors: 2" in markdown
    assert "- Passed benchmarks: 1" in markdown
    assert "- Failed benchmarks: 1" in markdown
    assert "| alpha_momentum | passed | 0.04 | 0.05 | 1.2 | -0.1 | 0.2 | N/A |" in markdown
    assert (
        "| alpha_reversal | failed | 0.04 | 0.05 | 1.2 | -0.1 | 0.2 | "
        "Failed benchmark tests: sharpe |"
    ) in markdown
    assert markdown.index("### alpha_momentum") < markdown.index("### alpha_reversal")
    assert "- Formula: rank(return_5d)" in markdown
    assert "- Related factors: momentum, return_5d" in markdown


def test_build_factor_wiki_markdown_supports_bilingual_output() -> None:
    markdown = build_factor_wiki_markdown(
        [_memory_record("memory-1", name="alpha_momentum")],
        output_language="bilingual",
    )

    assert markdown.startswith("# 量化因子知识库 / Quant Factor Wiki")
    assert "- 因子 / Factors: 1" in markdown
    assert "- 公式 / Formula: rank(return_5d)" in markdown


def test_factor_wiki_store_writes_markdown_atomically(tmp_path: Path) -> None:
    records = [_memory_record("memory-1", name="alpha_momentum")]
    wiki_path = tmp_path / "memory" / "factor_wiki.md"

    result = FactorWikiStore(wiki_path).save(records)

    assert result.wiki_path == wiki_path
    assert result.record_count == 1
    assert result.factor_count == 1
    assert result.passed_count == 1
    assert result.failed_count == 0
    assert wiki_path.is_file()
    assert "### alpha_momentum" in wiki_path.read_text(encoding="utf-8")


def test_summarize_factor_wiki_records_counts_unique_factors() -> None:
    summary = summarize_factor_wiki_records(
        [
            _memory_record("memory-1", name="alpha_momentum"),
            _memory_record("memory-2", name="alpha_momentum"),
            _memory_record("memory-3", name="alpha_reversal", benchmark_status="failed"),
        ]
    )

    assert summary == {
        "record_count": 3,
        "factor_count": 2,
        "passed_count": 2,
        "failed_count": 1,
    }


def test_build_factor_wiki_markdown_rejects_missing_factor_name() -> None:
    record = _memory_record("memory-1", name="alpha_momentum")
    record["factor"] = {}

    with pytest.raises(ValueError, match="factor.name"):
        build_factor_wiki_markdown([record], output_language="en")
