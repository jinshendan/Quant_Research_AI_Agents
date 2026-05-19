from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest

from agents.memory_agent import FactorMemoryStore
from agents.report_agent import (
    ReportAgent,
    ReportSpec,
    build_report_draft,
    render_report_markdown,
    save_markdown_report,
    select_memory_record,
)
from core.config import AppConfig
from core.logging import configure_logging, get_agent_logger
from core.models import AgentRequest


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig.from_env(project_root=tmp_path, environ={})


def _memory_record(
    memory_id: str,
    *,
    name: str = "alpha_momentum",
    created_at: str = "2026-05-18T10:00:00+00:00",
    benchmark_status: str = "passed",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "memory_id": memory_id,
        "created_at": created_at,
        "source": {
            "agent": "BacktestAgent",
            "task_id": f"task-{memory_id}",
            "state": "backtest_benchmark_tested",
            "generated_at": "2026-05-18T09:59:00+00:00",
        },
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
            "sharpe": 1.2 if benchmark_status == "passed" else 0.2,
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


def test_report_spec_accepts_memory_path_and_wiki_alias(tmp_path: Path) -> None:
    spec = ReportSpec.from_payload(
        {
            "memory_path": str(tmp_path / "memory" / "factor_memory.jsonl"),
            "factor_name": "alpha_momentum",
            "wiki_path": str(tmp_path / "memory" / "factor_wiki.md"),
            "report_path": str(tmp_path / "reports" / "momentum.md"),
            "report_title": "Momentum Factor Report",
        }
    )

    assert spec.memory_record is None
    assert spec.memory_path == tmp_path / "memory" / "factor_memory.jsonl"
    assert spec.factor_name == "alpha_momentum"
    assert spec.factor_wiki_path == tmp_path / "memory" / "factor_wiki.md"
    assert spec.report_path == tmp_path / "reports" / "momentum.md"
    assert spec.report_title == "Momentum Factor Report"


def test_report_spec_rejects_missing_memory_source() -> None:
    with pytest.raises(ValueError, match="memory_record"):
        ReportSpec.from_payload({})


def test_report_agent_generates_markdown_report_from_memory_path(
    tmp_path: Path,
) -> None:
    memory_path = tmp_path / "memory" / "factor_memory.jsonl"
    wiki_path = tmp_path / "memory" / "factor_wiki.md"
    FactorMemoryStore(memory_path).append(_memory_record("memory-1"))
    wiki_path.write_text("# Quant Factor Wiki\n\n### alpha_momentum\n", encoding="utf-8")

    stream = StringIO()
    configure_logging(stream=stream)
    agent = ReportAgent(
        config=_config(tmp_path),
        logger=get_agent_logger("ReportAgent"),
    )

    response = agent.run(
        AgentRequest.create(
            {
                "memory_path": str(memory_path),
                "factor_name": "alpha_momentum",
                "factor_wiki_path": str(wiki_path),
            },
            task_id="report-task-1",
        )
    )

    assert response.status == "success"
    assert response.output["state"] == "markdown_report_generated"
    assert response.output["report_title"] == "Research Report: alpha_momentum"
    assert response.output["section_count"] == 5
    assert response.output["report_draft_format"] == "structured_json"
    assert response.output["report_format"] == "markdown"
    assert response.output["next_action"] == "Build Streamlit dashboard in Day 27."
    assert response.metadata["agent"] == "ReportAgent"
    assert response.metadata["task_id"] == "report-task-1"
    assert response.metadata["memory_id"] == "memory-1"
    assert response.metadata["factor_name"] == "alpha_momentum"

    draft = response.output["report_draft"]
    assert draft["schema_version"] == 1
    assert draft["factor"]["formula"] == "rank(return_5d)"
    assert draft["context"]["factor_wiki_path"] == str(wiki_path.resolve())
    assert draft["context"]["factor_wiki_line_count"] == 3
    assert [section["id"] for section in draft["sections"]] == [
        "hypothesis",
        "factor_formula",
        "backtest_results",
        "risk_analysis",
        "conclusion",
    ]
    assert draft["sections"][-1]["content"]["verdict"] == "candidate_for_follow_up"
    assert draft["next_action"] == "Build Streamlit dashboard in Day 27."
    report_path = tmp_path / "research_logs" / "alpha_momentum_memory-1.md"
    markdown_report = response.output["report_markdown"]
    assert response.output["report_path"] == str(report_path)
    assert response.metadata["report_path"] == str(report_path)
    assert response.output["report_file"]["report_path"] == str(report_path)
    assert response.output["report_file"]["report_format"] == "markdown"
    assert response.output["report_file"]["bytes_written"] == len(
        markdown_report.encode("utf-8")
    )
    assert report_path.read_text(encoding="utf-8") == markdown_report
    assert markdown_report.startswith("# Research Report: alpha_momentum\n")
    assert "## Backtest Results" in markdown_report
    assert "- Benchmark status: passed" in markdown_report
    assert "ReportAgent | build_report_draft | success" in stream.getvalue()
    assert "ReportAgent | generate_markdown_report | success" in stream.getvalue()


def test_report_agent_writes_custom_markdown_report_path(tmp_path: Path) -> None:
    report_path = tmp_path / "reports" / "custom.md"
    agent = ReportAgent(config=_config(tmp_path))

    response = agent.run(
        AgentRequest.create(
            {
                "memory_record": _memory_record("memory-custom"),
                "report_path": str(report_path),
            }
        )
    )

    assert response.status == "success"
    assert response.output["report_path"] == str(report_path)
    assert report_path.is_file()


def test_report_agent_requires_selector_for_multiple_records(tmp_path: Path) -> None:
    memory_path = tmp_path / "memory" / "factor_memory.jsonl"
    store = FactorMemoryStore(memory_path)
    store.append(_memory_record("memory-1", name="alpha_momentum"))
    store.append(_memory_record("memory-2", name="alpha_reversal"))
    agent = ReportAgent(config=_config(tmp_path))

    response = agent.run(AgentRequest.create({"memory_path": str(memory_path)}))

    assert response.status == "error"
    assert "memory_id or payload.factor_name" in str(response.error)


def test_select_memory_record_uses_latest_record_for_factor_name() -> None:
    records = [
        _memory_record(
            "memory-old",
            name="alpha_momentum",
            created_at="2026-05-18T10:00:00+00:00",
        ),
        _memory_record(
            "memory-new",
            name="alpha_momentum",
            created_at="2026-05-18T11:00:00+00:00",
        ),
    ]

    selected = select_memory_record(records, factor_name="ALPHA_MOMENTUM")

    assert selected["memory_id"] == "memory-new"


def test_build_report_draft_marks_failed_benchmark_as_needs_review() -> None:
    draft = build_report_draft(
        _memory_record(
            "memory-failed",
            benchmark_status="failed",
        )
    )

    conclusion = draft.document["sections"][-1]
    risk = draft.document["sections"][3]

    assert draft.title == "Research Report: alpha_momentum"
    assert conclusion["content"]["verdict"] == "needs_review"
    assert conclusion["content"]["failure_reason"] == "Failed benchmark tests: sharpe"
    assert risk["content"]["failed_tests"] == ["sharpe"]


def test_render_report_markdown_outputs_expected_sections() -> None:
    draft = build_report_draft(_memory_record("memory-1"))

    markdown = render_report_markdown(draft.document)

    assert markdown.startswith("# Research Report: alpha_momentum\n")
    assert "## Metadata" in markdown
    assert "- Memory ID: `memory-1`" in markdown
    assert "## Hypothesis" in markdown
    assert "## Conclusion" in markdown
    assert "- Verdict: candidate_for_follow_up" in markdown


def test_render_report_markdown_rejects_missing_sections() -> None:
    draft = build_report_draft(_memory_record("memory-1")).document
    draft.pop("sections")

    with pytest.raises(ValueError, match="sections"):
        render_report_markdown(draft)


def test_save_markdown_report_writes_file_atomically(tmp_path: Path) -> None:
    report_path = tmp_path / "nested" / "report.md"

    result = save_markdown_report("# Report\n", report_path)

    assert result.report_path == report_path
    assert result.report_format == "markdown"
    assert result.bytes_written == len("# Report\n".encode("utf-8"))
    assert report_path.read_text(encoding="utf-8") == "# Report\n"
    assert not Path(f"{report_path}.tmp").exists()


def test_build_report_draft_rejects_missing_factor_name() -> None:
    record = _memory_record("memory-1")
    record["factor"] = {}

    with pytest.raises(ValueError, match="name"):
        build_report_draft(record)
