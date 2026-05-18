from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest

from agents.memory_agent import (
    FactorMemoryStore,
    MemoryAgent,
    MemorySpec,
    build_factor_memory_record,
)
from agents.memory_index import FactorMemoryVectorIndex
from core.config import AppConfig
from core.logging import configure_logging, get_agent_logger
from core.models import AgentRequest


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig.from_env(project_root=tmp_path, environ={})


def _result_json(*, benchmark_status: str = "passed") -> dict[str, object]:
    tests = [
        {
            "name": "usable_row_count",
            "threshold_key": "min_usable_rows",
            "metric": "summary.usable_row_count",
            "operator": ">=",
            "threshold": 10,
            "actual": 12,
            "passed": True,
        },
        {
            "name": "sharpe",
            "threshold_key": "min_sharpe",
            "metric": "summary.sharpe",
            "operator": ">=",
            "threshold": 1.0,
            "actual": 0.4 if benchmark_status == "failed" else 1.2,
            "passed": benchmark_status != "failed",
        },
    ]
    failed_count = sum(1 for item in tests if not item["passed"])
    return {
        "schema_version": 1,
        "state": "backtest_benchmark_tested",
        "generated_at": "2026-05-18T10:00:00+00:00",
        "agent": "BacktestAgent",
        "task_id": "backtest-task-1",
        "inputs": {
            "factor_matrix_path": "/tmp/factors.csv",
            "aligned_data_path": "/tmp/aligned.csv",
            "factor_column": "factor__alpha",
            "factor_direction": "positive",
            "forward_return_days": 1,
        },
        "summary": {
            "usable_row_count": 12,
            "portfolio_date_count": 2,
            "ic_date_count": 2,
            "rank_ic_date_count": 2,
            "mean_ic": 0.06,
            "mean_rank_ic": 0.07,
            "sharpe": 0.4 if benchmark_status == "failed" else 1.2,
            "max_drawdown": -0.08,
            "total_return": 0.12,
        },
        "metrics": {
            "drawdown": {
                "max_drawdown": -0.08,
                "max_drawdown_abs": 0.08,
            }
        },
        "benchmark_tests": {
            "schema_version": 1,
            "status": benchmark_status,
            "test_count": len(tests),
            "passed_count": len(tests) - failed_count,
            "failed_count": failed_count,
            "thresholds": {
                "min_usable_rows": 10,
                "min_sharpe": 1.0,
            },
            "tests": tests,
        },
    }


def test_memory_spec_accepts_inline_result_json_and_metadata() -> None:
    result_json = _result_json()
    spec = MemorySpec.from_payload(
        {
            "result_json": result_json,
            "memory_path": "memory/custom.jsonl",
            "vector_index_path": "memory/custom.faiss",
            "vector_metadata_path": "memory/custom.faiss.metadata.json",
            "factor_metadata": {
                "name": "alpha_001",
                "formula": "rank(close)",
                "related_factors": ["alpha_000", "alpha_000"],
            },
        }
    )

    assert spec.result_json == result_json
    assert spec.result_json_path is None
    assert spec.memory_path == Path("memory/custom.jsonl").resolve()
    assert spec.vector_index_path == Path("memory/custom.faiss").resolve()
    assert spec.vector_metadata_path == Path("memory/custom.faiss.metadata.json").resolve()
    assert spec.factor_metadata == {
        "name": "alpha_001",
        "formula": "rank(close)",
        "related_factors": ["alpha_000", "alpha_000"],
    }


def test_memory_spec_rejects_missing_result_json_source() -> None:
    with pytest.raises(ValueError, match="result_json"):
        MemorySpec.from_payload({})


def test_memory_agent_saves_factor_memory_record_from_result_json_path(
    tmp_path: Path,
) -> None:
    result_json = _result_json()
    result_json_path = tmp_path / "results" / "backtest.json"
    result_json_path.parent.mkdir(parents=True)
    result_json_path.write_text(json.dumps(result_json), encoding="utf-8")

    stream = StringIO()
    configure_logging(stream=stream)
    agent = MemoryAgent(
        config=_config(tmp_path),
        logger=get_agent_logger("MemoryAgent"),
    )
    request = AgentRequest.create(
        {
            "result_json_path": str(result_json_path),
            "factor_metadata": {
                "name": "alpha_001",
                "formula": "rank(close)",
                "hypothesis": "Higher rank(close) predicts returns.",
                "turnover": 0.18,
                "market_condition": "unit-test",
                "related_factors": ["alpha_000", "alpha_000"],
                "paper_reference": "internal-note",
            },
        },
        task_id="memory-task-1",
    )

    response = agent.run(request)

    assert response.status == "success"
    assert response.output["state"] == "memory_record_saved"
    assert response.output["next_action"] == "Save factor wiki in Day 24."
    assert response.metadata["agent"] == "MemoryAgent"
    assert response.metadata["task_id"] == "memory-task-1"
    assert response.metadata["factor_name"] == "alpha_001"
    assert response.metadata["benchmark_status"] == "passed"
    assert response.metadata["total_records"] == 1
    assert response.metadata["vector_index_records"] == 1

    memory_path = Path(response.output["memory_path"])
    assert memory_path == tmp_path / "memory" / "factor_memory.jsonl"
    assert memory_path.is_file()
    records = FactorMemoryStore(memory_path).load_all()
    assert len(records) == 1
    assert records[0] == response.output["memory_record"]

    record = response.output["memory_record"]
    assert record["schema_version"] == 1
    assert record["memory_id"] == response.output["memory_id"]
    assert record["source"]["task_id"] == "backtest-task-1"
    assert record["factor"] == {
        "name": "alpha_001",
        "formula": "rank(close)",
        "hypothesis": "Higher rank(close) predicts returns.",
        "direction": "positive",
        "forward_return_days": 1,
        "universe": None,
    }
    assert record["performance"]["ic"] == 0.06
    assert record["performance"]["rank_ic"] == 0.07
    assert record["performance"]["sharpe"] == 1.2
    assert record["performance"]["max_drawdown_abs"] == 0.08
    assert record["performance"]["turnover"] == 0.18
    assert record["benchmark"]["status"] == "passed"
    assert record["benchmark"]["failed_tests"] == []
    assert record["diagnostics"]["failure_reason"] is None
    assert record["diagnostics"]["related_factors"] == ["alpha_000"]
    assert record["artifacts"]["result_json_path"] == str(result_json_path.resolve())
    vector_index_path = Path(response.output["vector_index_path"])
    vector_metadata_path = Path(response.output["vector_metadata_path"])
    assert vector_index_path == tmp_path / "memory" / "factor_memory.faiss"
    assert vector_index_path.is_file()
    assert vector_metadata_path.is_file()
    assert response.output["vector_index"]["record_count"] == 1
    search_result = FactorMemoryVectorIndex(
        index_path=vector_index_path,
        metadata_path=vector_metadata_path,
    ).search("rank close alpha", top_k=1)
    assert len(search_result.matches) == 1
    assert search_result.matches[0].memory_id == response.output["memory_id"]
    assert search_result.matches[0].factor_name == "alpha_001"
    assert "MemoryAgent | save_memory_record | success" in stream.getvalue()
    assert "MemoryAgent | build_vector_index | success" in stream.getvalue()


def test_memory_agent_derives_failure_reason_for_failed_benchmark(
    tmp_path: Path,
) -> None:
    agent = MemoryAgent(config=_config(tmp_path))

    response = agent.run(
        AgentRequest.create(
            {
                "result_json": _result_json(benchmark_status="failed"),
            }
        )
    )

    assert response.status == "success"
    record = response.output["memory_record"]
    assert record["factor"]["name"] == "factor__alpha"
    assert record["benchmark"]["status"] == "failed"
    assert record["benchmark"]["failed_tests"] == ["sharpe"]
    assert record["diagnostics"]["failure_reason"] == "Failed benchmark tests: sharpe"


def test_memory_agent_rejects_pre_benchmark_result_json(tmp_path: Path) -> None:
    agent = MemoryAgent(config=_config(tmp_path))
    result_json = _result_json()
    result_json["state"] = "backtest_result_generated"

    response = agent.run(AgentRequest.create({"result_json": result_json}))

    assert response.status == "error"
    assert response.error == "result_json.state must be backtest_benchmark_tested."


def test_build_factor_memory_record_requires_drawdown_metrics() -> None:
    result_json = _result_json()
    result_json["metrics"] = {}

    with pytest.raises(ValueError, match="drawdown"):
        build_factor_memory_record(
            result_json=result_json,
            request=AgentRequest.create({}),
        )
