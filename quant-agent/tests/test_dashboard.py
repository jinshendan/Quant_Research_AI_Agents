from __future__ import annotations

from pathlib import Path

import pytest

import dashboard
from agents.daily_research import DailyResearchRunResult
from agents.memory_index import FactorMemoryVectorIndex
from agents.memory_agent import FactorMemoryStore
from core.config import AppConfig
from dashboard import (
    MarkdownReportSummary,
    build_dashboard_summary,
    build_semantic_search_view,
    build_factor_explorer_options,
    build_factor_explorer_view,
    build_factor_ranking_frame,
    build_metric_distribution_frame,
    default_daily_research_config_path,
    default_dashboard_paths,
    load_dashboard_data,
    load_markdown_report_summaries,
    match_report_summary,
    run_daily_research_from_config_file,
    run_semantic_memory_search,
    select_factor_record,
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
    assert paths.memory_index_path == tmp_path / "memory" / "factor_memory.faiss"
    assert (
        paths.memory_index_metadata_path
        == tmp_path / "memory" / "factor_memory.faiss.metadata.json"
    )
    assert paths.wiki_path == tmp_path / "memory" / "factor_wiki.md"
    assert paths.report_dir == tmp_path / "research_logs"
    assert paths.to_dict()["memory_index_path"] == str(
        tmp_path / "memory" / "factor_memory.faiss"
    )
    assert paths.to_dict()["report_dir"] == str(tmp_path / "research_logs")


def test_default_daily_research_config_path_prefers_tmp_config(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs"
    tmp_dir = tmp_path / "tmp"
    configs_dir.mkdir()
    tmp_dir.mkdir()
    example_config = configs_dir / "yinlun_daily.example.json"
    tmp_config = tmp_dir / "yinlun_daily.json"
    example_config.write_text("{}", encoding="utf-8")
    tmp_config.write_text("{}", encoding="utf-8")

    assert default_daily_research_config_path(tmp_path) == tmp_config


def test_default_daily_research_config_path_falls_back_to_example(
    tmp_path: Path,
) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    example_config = configs_dir / "yinlun_daily.example.json"
    example_config.write_text("{}", encoding="utf-8")

    assert default_daily_research_config_path(tmp_path) == example_config


def test_run_daily_research_from_config_file_returns_dashboard_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config_path = tmp_path / "configs" / "daily.json"
    config_path.parent.mkdir()
    config_path.write_text("{}", encoding="utf-8")
    manifest_path = tmp_path / "daily_runs" / "run-1" / "daily_research_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        """
        {
          "summary": {"benchmark_status": "passed"},
          "artifacts": {"report_path": "daily_runs/run-1/research_report.md"}
        }
        """,
        encoding="utf-8",
    )
    loaded_spec = object()

    def fake_load_daily_research_config(path: Path) -> object:
        assert path == config_path
        return loaded_spec

    def fake_run_daily_research(
        received_config: AppConfig,
        received_spec: object,
    ) -> DailyResearchRunResult:
        assert received_config == config
        assert received_spec is loaded_spec
        return DailyResearchRunResult(
            status="success",
            run_id="run-1",
            manifest_path=manifest_path,
            summary={"benchmark_status": "passed"},
        )

    monkeypatch.setattr(
        dashboard,
        "load_daily_research_config",
        fake_load_daily_research_config,
    )
    monkeypatch.setattr(dashboard, "run_daily_research", fake_run_daily_research)

    view = run_daily_research_from_config_file(config, config_path)

    assert view.success is True
    assert view.status == "success"
    assert view.run_id == "run-1"
    assert view.manifest_path == manifest_path
    assert view.summary == {"benchmark_status": "passed"}
    assert view.artifacts == {"report_path": "daily_runs/run-1/research_report.md"}
    assert "benchmark_status: passed" in view.terminal_summary


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


def test_build_factor_explorer_options_are_deterministic() -> None:
    records = [
        _memory_record("memory-2", name="alpha_b", ic=0.02, rank_ic=0.03, sharpe=1.2),
        _memory_record("memory-1", name="alpha_a", ic=0.01, rank_ic=0.02, sharpe=1.0),
    ]

    options = build_factor_explorer_options(records)

    assert [option.factor_name for option in options] == ["alpha_a", "alpha_b"]
    assert options[0].to_dict() == {
        "label": "alpha_a | memory-1 | 2026-05-19T10:01:00+00:00",
        "factor_name": "alpha_a",
        "memory_id": "memory-1",
        "created_at": "2026-05-19T10:01:00+00:00",
    }


def test_select_factor_record_supports_memory_id_and_latest_factor_name() -> None:
    records = [
        _memory_record("memory-1", name="alpha_a", ic=0.01, rank_ic=0.02, sharpe=1.0),
        _memory_record("memory-2", name="alpha_a", ic=0.03, rank_ic=0.06, sharpe=1.5),
        _memory_record("memory-3", name="alpha_b", ic=0.02, rank_ic=0.04, sharpe=1.1),
    ]

    selected_by_id = select_factor_record(records, memory_id="memory-1")
    selected_latest = select_factor_record(records, factor_name="ALPHA_A")
    missing = select_factor_record(records, memory_id="missing")

    assert selected_by_id is not None
    assert selected_by_id["memory_id"] == "memory-1"
    assert selected_latest is not None
    assert selected_latest["memory_id"] == "memory-2"
    assert missing is None


def test_build_factor_explorer_view_includes_sections_and_matched_report(
    tmp_path: Path,
) -> None:
    record = _memory_record(
        "memory-1",
        name="alpha_momentum",
        ic=0.03,
        rank_ic=0.05,
        sharpe=1.2,
    )
    report = MarkdownReportSummary(
        path=tmp_path / "alpha_momentum_memory-1.md",
        title="Research Report: alpha_momentum",
        byte_count=128,
        modified_time=1.0,
    )

    view = build_factor_explorer_view(record, report_summaries=[report])

    assert view.factor_name == "alpha_momentum"
    assert view.memory_id == "memory-1"
    assert view.overview["formula"] == "rank(close)"
    assert view.performance["rank_ic"] == 0.05
    assert view.benchmark["status"] == "passed"
    assert view.diagnostics["market_condition"] == "unit-test"
    assert view.artifacts["result_json_path"] == "/tmp/result.json"
    assert view.report_summary == report
    assert view.to_dict()["report_summary"]["title"] == "Research Report: alpha_momentum"


def test_match_report_summary_prefers_memory_id_match(tmp_path: Path) -> None:
    record = _memory_record(
        "memory-2",
        name="alpha_momentum",
        ic=0.03,
        rank_ic=0.05,
        sharpe=1.2,
    )
    factor_only_report = MarkdownReportSummary(
        path=tmp_path / "alpha_momentum.md",
        title="Research Report: alpha_momentum",
        byte_count=100,
        modified_time=2.0,
    )
    memory_report = MarkdownReportSummary(
        path=tmp_path / "alpha_momentum_memory-2.md",
        title="Research Report: alpha_momentum",
        byte_count=100,
        modified_time=1.0,
    )

    matched = match_report_summary(record, [factor_only_report, memory_report])

    assert matched == memory_report


def test_run_semantic_memory_search_returns_error_for_blank_query(
    tmp_path: Path,
) -> None:
    paths = default_dashboard_paths(_config(tmp_path))

    view = run_semantic_memory_search(paths, "   ")

    assert view.status == "error"
    assert view.error == "Search query must not be empty."
    assert view.matches == ()


def test_run_semantic_memory_search_returns_error_for_missing_index(
    tmp_path: Path,
) -> None:
    paths = default_dashboard_paths(_config(tmp_path))

    view = run_semantic_memory_search(paths, "momentum rank_ic")

    assert view.status == "error"
    assert "Memory FAISS index file not found" in str(view.error)
    assert view.index_path == paths.memory_index_path
    assert view.metadata_path == paths.memory_index_metadata_path


def test_run_semantic_memory_search_uses_faiss_index_and_matches_reports(
    tmp_path: Path,
) -> None:
    paths = default_dashboard_paths(_config(tmp_path))
    records = [
        _memory_record("memory-1", name="alpha_momentum", ic=0.03, rank_ic=0.05, sharpe=1.2),
        _memory_record("memory-2", name="alpha_reversal", ic=0.01, rank_ic=0.02, sharpe=0.4),
    ]
    FactorMemoryVectorIndex(
        index_path=paths.memory_index_path,
        metadata_path=paths.memory_index_metadata_path,
    ).build(records)
    report = MarkdownReportSummary(
        path=tmp_path / "alpha_momentum_memory-1.md",
        title="Research Report: alpha_momentum",
        byte_count=128,
        modified_time=1.0,
    )

    view = run_semantic_memory_search(
        paths,
        "alpha_momentum momentum rank_ic",
        top_k=2,
        report_summaries=[report],
    )

    assert view.status == "success"
    assert view.query == "alpha_momentum momentum rank_ic"
    assert len(view.matches) == 2
    assert view.matches[0].memory_id == "memory-1"
    assert view.matches[0].factor_name == "alpha_momentum"
    assert view.matches[0].report_summary == report
    assert view.to_dict()["match_count"] == 2


def test_build_semantic_search_view_preserves_match_metadata(tmp_path: Path) -> None:
    paths = default_dashboard_paths(_config(tmp_path))
    record = _memory_record(
        "memory-1",
        name="alpha_momentum",
        ic=0.03,
        rank_ic=0.05,
        sharpe=1.2,
    )
    search_result = FactorMemoryVectorIndex(
        index_path=paths.memory_index_path,
        metadata_path=paths.memory_index_metadata_path,
    ).build([record])
    raw_result = FactorMemoryVectorIndex(
        index_path=search_result.index_path,
        metadata_path=search_result.metadata_path,
    ).search("alpha_momentum", top_k=1)

    view = build_semantic_search_view(raw_result, top_k=1)

    assert view.status == "success"
    assert view.matches[0].rank == 1
    assert view.matches[0].benchmark_status == "passed"
    assert view.matches[0].record["memory_id"] == "memory-1"


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
