from __future__ import annotations

from pathlib import Path

from agents.memory_agent import FactorMemoryStore
from core.config import AppConfig
from dashboard import (
    build_dashboard_summary,
    build_factor_ranking_frame,
    build_metric_distribution_frame,
    default_dashboard_paths,
    load_dashboard_data,
    load_markdown_report_summaries,
)


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig.from_env(project_root=tmp_path, environ={})


def _memory_record(
    memory_id: str,
    *,
    name: str,
    ic: float,
    rank_ic: float,
    sharpe: float,
    benchmark_status: str = "passed",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "memory_id": memory_id,
        "created_at": f"2026-05-19T10:0{memory_id[-1]}:00+00:00",
        "source": {
            "agent": "BacktestAgent",
            "task_id": f"task-{memory_id}",
            "state": "backtest_benchmark_tested",
            "generated_at": "2026-05-19T10:00:00+00:00",
        },
        "factor": {
            "name": name,
            "formula": "rank(close)",
            "hypothesis": "Higher ranked close predicts returns.",
            "direction": "positive",
            "forward_return_days": 1,
            "universe": "CSI500",
        },
        "performance": {
            "ic": ic,
            "rank_ic": rank_ic,
            "sharpe": sharpe,
            "max_drawdown": -0.1,
            "max_drawdown_abs": 0.1,
            "total_return": 0.2,
            "turnover": 0.18,
        },
        "benchmark": {
            "status": benchmark_status,
            "test_count": 2,
            "passed_count": 2 if benchmark_status == "passed" else 1,
            "failed_count": 0 if benchmark_status == "passed" else 1,
            "failed_tests": [] if benchmark_status == "passed" else ["sharpe"],
        },
        "diagnostics": {
            "failure_reason": (
                None
                if benchmark_status == "passed"
                else "Failed benchmark tests: sharpe"
            ),
            "market_condition": "unit-test",
            "related_factors": ["momentum"],
            "paper_reference": "internal-note",
        },
        "artifacts": {
            "result_json_path": "/tmp/result.json",
            "factor_matrix_path": "/tmp/factors.csv",
            "aligned_data_path": "/tmp/aligned.csv",
        },
    }


def test_default_dashboard_paths_use_project_artifacts(tmp_path: Path) -> None:
    paths = default_dashboard_paths(_config(tmp_path))

    assert paths.memory_path == tmp_path / "memory" / "factor_memory.jsonl"
    assert paths.wiki_path == tmp_path / "memory" / "factor_wiki.md"
    assert paths.report_dir == tmp_path / "research_logs"
    assert paths.to_dict()["report_dir"] == str(tmp_path / "research_logs")


def test_load_dashboard_data_reads_memory_wiki_and_reports(tmp_path: Path) -> None:
    paths = default_dashboard_paths(_config(tmp_path))
    FactorMemoryStore(paths.memory_path).append(
        _memory_record(
            "memory-1",
            name="alpha_momentum",
            ic=0.03,
            rank_ic=0.05,
            sharpe=1.2,
        )
    )
    paths.wiki_path.write_text("# Quant Factor Wiki\n", encoding="utf-8")
    paths.report_dir.mkdir()
    report_path = paths.report_dir / "alpha_momentum.md"
    report_path.write_text("# Research Report: alpha_momentum\n", encoding="utf-8")

    data = load_dashboard_data(paths)

    assert len(data.records) == 1
    assert data.factor_wiki_markdown == "# Quant Factor Wiki\n"
    assert len(data.report_summaries) == 1
    assert data.report_summaries[0].path == report_path
    assert data.report_summaries[0].title == "Research Report: alpha_momentum"


def test_load_markdown_report_summaries_returns_latest_first(tmp_path: Path) -> None:
    first_report = tmp_path / "first.md"
    second_report = tmp_path / "second.md"
    first_report.write_text("# First\n", encoding="utf-8")
    second_report.write_text("# Second\n", encoding="utf-8")

    summaries = load_markdown_report_summaries(tmp_path)

    assert {summary.title for summary in summaries} == {"First", "Second"}
    assert summaries[0].modified_time >= summaries[-1].modified_time


def test_build_factor_ranking_frame_sorts_by_rank_ic_then_sharpe() -> None:
    records = [
        _memory_record("memory-1", name="alpha_low", ic=0.01, rank_ic=0.02, sharpe=2.0),
        _memory_record(
            "memory-2",
            name="alpha_high",
            ic=0.03,
            rank_ic=0.08,
            sharpe=0.5,
        ),
        _memory_record("memory-3", name="alpha_tie", ic=0.04, rank_ic=0.08, sharpe=1.5),
    ]

    frame = build_factor_ranking_frame(records)

    assert frame["rank"].tolist() == [1, 2, 3]
    assert frame["factor_name"].tolist() == ["alpha_tie", "alpha_high", "alpha_low"]
    assert frame["rank_ic"].tolist() == [0.08, 0.08, 0.02]
    assert frame["sharpe"].tolist() == [1.5, 0.5, 2.0]


def test_build_metric_distribution_frame_handles_empty_and_constant_values() -> None:
    empty_frame = build_metric_distribution_frame([], "ic")
    constant_frame = build_metric_distribution_frame(
        [
            _memory_record("memory-1", name="alpha_a", ic=0.03, rank_ic=0.05, sharpe=1.0),
            _memory_record("memory-2", name="alpha_b", ic=0.03, rank_ic=0.06, sharpe=1.2),
        ],
        "ic",
    )

    assert empty_frame.empty
    assert constant_frame.to_dict("records") == [{"bucket": "0.03", "count": 2}]


def test_build_metric_distribution_frame_counts_numeric_values() -> None:
    records = [
        _memory_record("memory-1", name="alpha_a", ic=0.01, rank_ic=0.02, sharpe=1.0),
        _memory_record("memory-2", name="alpha_b", ic=0.02, rank_ic=0.03, sharpe=1.5),
        _memory_record("memory-3", name="alpha_c", ic=0.03, rank_ic=0.04, sharpe=2.0),
    ]

    frame = build_metric_distribution_frame(records, "sharpe", bins=2)

    assert frame["count"].sum() == 3
    assert len(frame) == 2


def test_build_dashboard_summary_counts_records_and_reports(tmp_path: Path) -> None:
    records = [
        _memory_record("memory-1", name="alpha_a", ic=0.01, rank_ic=0.02, sharpe=1.0),
        _memory_record(
            "memory-2",
            name="alpha_b",
            ic=0.02,
            rank_ic=0.03,
            sharpe=0.2,
            benchmark_status="failed",
        ),
    ]
    report = load_markdown_report_summaries(tmp_path)

    summary = build_dashboard_summary(records, report)

    assert summary == {
        "record_count": 2,
        "factor_count": 2,
        "passed_count": 1,
        "failed_count": 1,
        "report_count": 0,
    }
