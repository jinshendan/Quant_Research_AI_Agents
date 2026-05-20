from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest

from agents.critic_agent import CriticAgent, CriticSpec, build_factor_critique
from core.logging import configure_logging, get_agent_logger
from core.models import AgentRequest


def _result_json(*, status: str = "failed") -> dict[str, object]:
    tests = [
        _benchmark_test("usable_row_count", 240, 252, passed=True),
        _benchmark_test("average_leg_count", 2.0, 3, passed=status == "passed"),
        _benchmark_test("mean_rank_ic", -0.04, 0.02, passed=status == "passed"),
        _benchmark_test("sharpe", -1.8, 0.5, passed=status == "passed"),
        _benchmark_test("total_return", -0.93, 0.0, passed=status == "passed"),
        _benchmark_test("max_drawdown_abs", 0.94, 0.35, operator="<=", passed=status == "passed"),
    ]
    failed_tests = [test["name"] for test in tests if not test["passed"]]
    return {
        "schema_version": 1,
        "state": "backtest_benchmark_tested",
        "generated_at": "2026-05-20T10:00:00+00:00",
        "agent": "BacktestAgent",
        "task_id": "backtest-task-1",
        "inputs": {
            "factor_column": "factor__close_to_open_return",
            "factor_direction": "positive",
            "forward_return_days": 1,
        },
        "summary": {
            "usable_row_count": 240,
            "portfolio_date_count": 80,
            "ic_date_count": 80,
            "rank_ic_date_count": 80,
            "average_leg_count": 2.0,
            "mean_ic": -0.03,
            "mean_rank_ic": -0.04,
            "sharpe": -1.8,
            "net_sharpe": -1.8,
            "total_return": -0.93,
            "net_total_return": -0.93,
            "total_transaction_cost": 2.3,
        },
        "metrics": {
            "drawdown": {
                "max_drawdown_abs": 0.94,
            }
        },
        "benchmark_tests": {
            "schema_version": 1,
            "status": status,
            "test_count": len(tests),
            "passed_count": len(tests) - len(failed_tests),
            "failed_count": len(failed_tests),
            "failed_tests": failed_tests,
            "tests": tests,
        },
    }


def _benchmark_test(
    name: str,
    actual: float,
    threshold: float,
    *,
    operator: str = ">=",
    passed: bool,
) -> dict[str, object]:
    return {
        "name": name,
        "threshold_key": f"min_{name}",
        "metric": f"summary.{name}",
        "operator": operator,
        "threshold": threshold,
        "actual": actual,
        "passed": passed,
    }


def test_critic_spec_accepts_result_json_path(tmp_path: Path) -> None:
    result_path = tmp_path / "backtest.json"
    spec = CriticSpec.from_payload(
        {
            "result_json_path": str(result_path),
            "output_language": "zh",
        }
    )

    assert spec.result_json_path == result_path.resolve()
    assert spec.result_json is None
    assert spec.output_language == "zh"


def test_build_factor_critique_rejects_failed_factor() -> None:
    critique = build_factor_critique(_result_json(), output_language="zh").document

    assert critique["state"] == "factor_critique_built"
    assert critique["factor_column"] == "factor__close_to_open_return"
    assert critique["verdict"] == "reject_for_now"
    assert critique["severity"] == "high"
    assert critique["failed_tests"] == [
        "average_leg_count",
        "mean_rank_ic",
        "sharpe",
        "total_return",
        "max_drawdown_abs",
    ]
    assert "暂时拒绝该因子" in critique["summary_text"]
    assert critique["metrics_snapshot"]["net_total_return"] == -0.93
    assert any(reason["code"] == "average_leg_count" for reason in critique["reasons"])
    assert any("不要用该因子做入场时机依据" in item for item in critique["action_items"])


def test_build_factor_critique_tracks_passed_factor() -> None:
    critique = build_factor_critique(_result_json(status="passed"), output_language="en")

    assert critique.verdict == "track"
    assert critique.document["severity"] == "low"
    assert critique.document["failed_tests"] == []
    assert "passes the current quality gates" in critique.document["summary_text"]


def test_critic_agent_loads_result_json_path_and_logs(tmp_path: Path) -> None:
    result_json_path = tmp_path / "backtest.json"
    result_json_path.write_text(json.dumps(_result_json()), encoding="utf-8")
    stream = StringIO()
    configure_logging(stream=stream)
    agent = CriticAgent(logger=get_agent_logger("CriticAgent"))

    response = agent.run(
        AgentRequest.create(
            {
                "result_json_path": str(result_json_path),
                "output_language": "bilingual",
            },
            task_id="critic-task-1",
        )
    )

    assert response.status == "success"
    assert response.output["state"] == "factor_critique_built"
    assert response.output["verdict"] == "reject_for_now"
    assert response.output["severity"] == "high"
    assert response.metadata["agent"] == "CriticAgent"
    assert response.metadata["task_id"] == "critic-task-1"
    assert response.metadata["benchmark_status"] == "failed"
    assert "CriticAgent | build_critique | success" in stream.getvalue()


def test_critic_agent_rejects_pre_benchmark_result() -> None:
    result_json = _result_json()
    result_json["state"] = "backtest_result_generated"

    response = CriticAgent().run(
        AgentRequest.create(
            {
                "result_json": result_json,
            }
        )
    )

    assert response.status == "error"
    assert response.error == "result_json.state must be backtest_benchmark_tested."


def test_critic_spec_rejects_missing_result_source() -> None:
    with pytest.raises(ValueError, match="result_json"):
        CriticSpec.from_payload({})
