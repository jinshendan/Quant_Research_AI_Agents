from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from agents.backtest_agent import (
    BacktestAgent,
    BacktestSpec,
    compute_drawdown,
    compute_information_coefficient,
    compute_rank_information_coefficient,
    compute_sharpe_ratio,
    run_benchmark_tests,
    save_backtest_result_json,
    normalize_aligned_prices,
    normalize_factor_matrix,
)
from core.logging import configure_logging, get_agent_logger
from core.models import AgentRequest


def _symbols() -> list[str]:
    return [f"{number:06d}" for number in range(1, 7)]


def _write_aligned_data(tmp_path: Path) -> Path:
    day_1_returns = [-0.06, -0.04, -0.02, 0.02, 0.04, 0.06]
    day_2_returns = [-0.03, -0.02, -0.01, 0.01, 0.02, 0.03]
    rows = []
    for symbol, first_return, second_return in zip(
        _symbols(),
        day_1_returns,
        day_2_returns,
        strict=True,
    ):
        first_close = 100.0
        second_close = first_close * (1.0 + first_return)
        third_close = second_close * (1.0 + second_return)
        for date, close in (
            ("2024-01-01", first_close),
            ("2024-01-02", second_close),
            ("2024-01-03", third_close),
        ):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "close": close,
                }
            )
    path = tmp_path / "aligned.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_factor_matrix(tmp_path: Path, *, include_second_factor: bool = False) -> Path:
    rows = []
    for date in ("2024-01-01", "2024-01-02"):
        for value, symbol in enumerate(_symbols(), start=1):
            row = {
                "date": date,
                "symbol": symbol,
                "factor__alpha": float(value),
            }
            if include_second_factor:
                row["factor__beta"] = float(7 - value)
            rows.append(row)
    path = tmp_path / "factor_matrix.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_manifest(
    tmp_path: Path,
    *,
    factor_matrix_path: Path,
    aligned_data_path: Path,
    factor_columns: list[str] | None = None,
) -> Path:
    path = tmp_path / "factor_matrix.manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "storage": {
                    "matrix_path": str(factor_matrix_path),
                    "factor_columns": factor_columns or ["factor__alpha"],
                },
                "context": {
                    "source_aligned_data_path": str(aligned_data_path),
                    "template_ids": ["alpha"],
                },
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    return path


def test_backtest_spec_accepts_manifest_only(tmp_path: Path) -> None:
    manifest_path = tmp_path / "factor_matrix.manifest.json"
    spec = BacktestSpec.from_payload(
        {
            "factor_manifest_path": str(manifest_path),
            "factor_column": "factor__alpha",
            "forward_return_days": 2,
            "quantile_count": 3,
            "benchmark_thresholds": {
                "min_usable_rows": 10,
                "min_sharpe": 1.0,
                "max_drawdown_abs": 0.2,
            },
            "preview_rows": 0,
        }
    )

    assert spec.factor_manifest_path == manifest_path.resolve()
    assert spec.factor_matrix_path is None
    assert spec.factor_column == "factor__alpha"
    assert spec.forward_return_days == 2
    assert spec.quantile_count == 3
    assert spec.annualization_factor == 252
    assert spec.benchmark_thresholds["min_usable_rows"] == 10
    assert spec.benchmark_thresholds["min_sharpe"] == 1.0
    assert spec.benchmark_thresholds["max_drawdown_abs"] == 0.2
    assert spec.benchmark_thresholds["min_mean_ic"] is None
    assert spec.preview_rows == 0


def test_backtest_spec_rejects_missing_factor_source() -> None:
    with pytest.raises(ValueError, match="factor_matrix_path"):
        BacktestSpec.from_payload({"aligned_data_path": "aligned.csv"})


def test_backtest_spec_rejects_unknown_benchmark_threshold(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported keys"):
        BacktestSpec.from_payload(
            {
                "factor_manifest_path": str(tmp_path / "factor_matrix.manifest.json"),
                "benchmark_thresholds": {"unknown": 1},
            }
        )


def test_normalize_factor_matrix_rejects_missing_identity_columns() -> None:
    with pytest.raises(ValueError, match="date"):
        normalize_factor_matrix(
            pd.DataFrame({"symbol": ["000001"], "factor__alpha": [1.0]})
        )


def test_normalize_aligned_prices_rejects_missing_close() -> None:
    with pytest.raises(ValueError, match="close"):
        normalize_aligned_prices(pd.DataFrame({"date": ["2024-01-01"], "symbol": ["1"]}))


def test_backtest_agent_builds_long_short_returns_from_manifest(tmp_path: Path) -> None:
    aligned_path = _write_aligned_data(tmp_path)
    factor_matrix_path = _write_factor_matrix(tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        factor_matrix_path=factor_matrix_path,
        aligned_data_path=aligned_path,
    )
    stream = StringIO()
    configure_logging(stream=stream)
    agent = BacktestAgent(logger=get_agent_logger("BacktestAgent"))

    response = agent.run(
        AgentRequest.create(
            {
                "factor_manifest_path": str(manifest_path),
                "factor_column": "factor__alpha",
                "quantile_count": 3,
                "result_json_path": str(tmp_path / "results" / "backtest.json"),
                "benchmark_thresholds": {
                    "min_usable_rows": 12,
                    "min_portfolio_dates": 2,
                    "min_ic_dates": 2,
                    "min_rank_ic_dates": 2,
                    "min_mean_ic": 0.9,
                    "min_mean_rank_ic": 0.9,
                    "min_sharpe": 1.0,
                    "min_total_return": 0.01,
                    "max_drawdown_abs": 0.05,
                },
                "preview_rows": 5,
            },
            task_id="backtest-task-1",
        )
    )

    assert response.status == "success"
    assert response.output["state"] == "backtest_benchmark_tested"
    assert response.output["factor_matrix_path"] == str(factor_matrix_path.resolve())
    assert response.output["aligned_data_path"] == str(aligned_path.resolve())
    assert response.output["factor_column"] == "factor__alpha"
    assert response.output["annualization_factor"] == 252
    assert response.output["request"]["result_json_path"] == str(
        tmp_path / "results" / "backtest.json"
    )
    assert response.output["portfolio_date_count"] == 2
    assert response.output["usable_row_count"] == 12
    assert response.output["ic_date_count"] == 2
    assert response.output["rank_ic_date_count"] == 2
    assert response.output["drawdown_curve_columns"] == [
        "date",
        "equity_curve",
        "cumulative_peak",
        "drawdown",
    ]
    assert response.metadata["agent"] == "BacktestAgent"
    assert response.metadata["task_id"] == "backtest-task-1"
    assert response.metadata["portfolio_date_count"] == 2
    assert response.metadata["ic_date_count"] == 2
    assert response.metadata["rank_ic_date_count"] == 2
    assert response.metadata["mean_ic"] > 0.99
    assert response.metadata["mean_rank_ic"] == pytest.approx(1.0)
    assert response.metadata["sharpe"] == pytest.approx(33.67491648096547)
    assert response.metadata["max_drawdown"] == pytest.approx(0.0)
    assert response.metadata["benchmark_status"] == "passed"
    assert response.metadata["result_json_path"] == str(
        tmp_path / "results" / "backtest.json"
    )

    preview = response.output["preview"]
    assert preview[0]["date"] == "2024-01-01"
    assert preview[0]["long_return"] == pytest.approx(0.05)
    assert preview[0]["short_return"] == pytest.approx(-0.05)
    assert preview[0]["long_short_return"] == pytest.approx(0.10)
    assert preview[0]["long_count"] == 2
    assert preview[0]["short_count"] == 2
    assert preview[1]["long_short_return"] == pytest.approx(0.05)

    stats = response.output["backtest_stats"]
    assert stats["input_row_count"] == 12
    assert stats["valid_factor_row_count"] == 12
    assert stats["valid_forward_return_row_count"] == 12
    assert stats["skipped_date_count"] == 0
    ic_preview = response.output["ic_series_preview"]
    assert ic_preview[0]["date"] == "2024-01-01"
    assert ic_preview[0]["ic"] > 0.99
    assert ic_preview[0]["raw_ic"] > 0.99
    assert ic_preview[0]["pair_count"] == 6
    assert response.output["ic_stats"]["method"] == "pearson"
    assert response.output["ic_stats"]["ic_date_count"] == 2
    assert response.output["ic_stats"]["mean_ic"] > 0.99
    assert response.output["ic_stats"]["positive_ic_ratio"] == 1.0
    rank_ic_preview = response.output["rank_ic_series_preview"]
    assert rank_ic_preview[0]["date"] == "2024-01-01"
    assert rank_ic_preview[0]["rank_ic"] == pytest.approx(1.0)
    assert rank_ic_preview[0]["raw_rank_ic"] == pytest.approx(1.0)
    assert rank_ic_preview[0]["pair_count"] == 6
    assert response.output["rank_ic_stats"]["method"] == "spearman"
    assert response.output["rank_ic_stats"]["rank_ic_date_count"] == 2
    assert response.output["rank_ic_stats"]["mean_rank_ic"] == pytest.approx(1.0)
    assert response.output["rank_ic_stats"]["positive_rank_ic_ratio"] == 1.0
    assert response.output["sharpe_stats"]["method"] == "mean_std"
    assert response.output["sharpe_stats"]["return_column"] == "long_short_return"
    assert response.output["sharpe_stats"]["return_count"] == 2
    assert response.output["sharpe_stats"]["mean_period_return"] == pytest.approx(0.075)
    assert response.output["sharpe_stats"]["std_period_return"] == pytest.approx(
        0.03535533905932738
    )
    assert response.output["sharpe_stats"]["annualized_mean_return"] == pytest.approx(18.9)
    assert response.output["sharpe_stats"]["sharpe"] == pytest.approx(
        33.67491648096547
    )
    assert response.output["sharpe_stats"]["positive_return_ratio"] == 1.0
    drawdown_preview = response.output["drawdown_curve_preview"]
    assert drawdown_preview[0]["date"] == "2024-01-01"
    assert drawdown_preview[0]["equity_curve"] == pytest.approx(1.1)
    assert drawdown_preview[0]["drawdown"] == pytest.approx(0.0)
    assert response.output["drawdown_stats"]["method"] == "cumulative_return"
    assert response.output["drawdown_stats"]["return_column"] == "long_short_return"
    assert response.output["drawdown_stats"]["return_count"] == 2
    assert response.output["drawdown_stats"]["end_equity"] == pytest.approx(1.155)
    assert response.output["drawdown_stats"]["total_return"] == pytest.approx(0.155)
    assert response.output["drawdown_stats"]["max_drawdown"] == pytest.approx(0.0)
    assert response.output["drawdown_stats"]["drawdown_period_count"] == 0
    benchmark_tests = response.output["benchmark_tests"]
    assert response.output["benchmark_status"] == "passed"
    assert benchmark_tests["status"] == "passed"
    assert benchmark_tests["test_count"] == 9
    assert benchmark_tests["passed_count"] == 9
    assert benchmark_tests["failed_count"] == 0
    result_json = response.output["result_json"]
    assert result_json["schema_version"] == 1
    assert result_json["state"] == "backtest_benchmark_tested"
    assert result_json["task_id"] == "backtest-task-1"
    assert result_json["inputs"]["factor_column"] == "factor__alpha"
    assert result_json["summary"]["mean_rank_ic"] == pytest.approx(1.0)
    assert result_json["summary"]["sharpe"] == pytest.approx(33.67491648096547)
    assert result_json["summary"]["max_drawdown"] == pytest.approx(0.0)
    assert result_json["metrics"]["drawdown"] == response.output["drawdown_stats"]
    assert result_json["previews"]["portfolio_returns"] == response.output["preview"]
    assert result_json["benchmark_tests"] == benchmark_tests
    assert result_json["next_action"] == "Build MemoryAgent in Day 22."
    result_json_path = Path(response.output["result_json_path"])
    assert result_json_path.is_file()
    assert json.loads(result_json_path.read_text(encoding="utf-8")) == result_json
    assert "BacktestAgent | build_backtest | success" in stream.getvalue()
    assert "BacktestAgent | compute_ic | success" in stream.getvalue()
    assert "BacktestAgent | compute_rank_ic | success" in stream.getvalue()
    assert "BacktestAgent | compute_sharpe | success" in stream.getvalue()
    assert "BacktestAgent | compute_drawdown | success" in stream.getvalue()
    assert "BacktestAgent | generate_result_json | success" in stream.getvalue()
    assert "BacktestAgent | run_benchmark_tests | success" in stream.getvalue()


def test_backtest_agent_uses_single_factor_without_explicit_column(
    tmp_path: Path,
) -> None:
    aligned_path = _write_aligned_data(tmp_path)
    factor_matrix_path = _write_factor_matrix(tmp_path)
    agent = BacktestAgent()

    response = agent.run(
        AgentRequest.create(
            {
                "factor_matrix_path": str(factor_matrix_path),
                "aligned_data_path": str(aligned_path),
                "quantile_count": 3,
                "preview_rows": 0,
            }
        )
    )

    assert response.status == "success"
    assert response.output["factor_column"] == "factor__alpha"
    assert response.output["benchmark_status"] == "passed"
    assert response.output["result_json_path"] is None
    assert response.output["preview"] == []


def test_backtest_agent_requires_factor_column_for_multi_factor_matrix(
    tmp_path: Path,
) -> None:
    aligned_path = _write_aligned_data(tmp_path)
    factor_matrix_path = _write_factor_matrix(tmp_path, include_second_factor=True)
    manifest_path = _write_manifest(
        tmp_path,
        factor_matrix_path=factor_matrix_path,
        aligned_data_path=aligned_path,
        factor_columns=["factor__alpha", "factor__beta"],
    )
    agent = BacktestAgent()

    response = agent.run(
        AgentRequest.create(
            {
                "factor_manifest_path": str(manifest_path),
                "quantile_count": 3,
            }
        )
    )

    assert response.status == "error"
    assert "factor_column is required" in str(response.error)


def test_backtest_agent_requires_aligned_data_path_without_manifest(
    tmp_path: Path,
) -> None:
    factor_matrix_path = _write_factor_matrix(tmp_path)
    agent = BacktestAgent()

    response = agent.run(
        AgentRequest.create({"factor_matrix_path": str(factor_matrix_path)})
    )

    assert response.status == "error"
    assert "aligned_data_path is required" in str(response.error)


def test_compute_information_coefficient_adjusts_negative_direction() -> None:
    panel = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-01"],
            "factor__negative": [3.0, 2.0, 1.0],
            "forward_return": [0.01, 0.02, 0.03],
        }
    )

    result = compute_information_coefficient(
        panel,
        factor_column="factor__negative",
        factor_direction="negative",
    )

    assert result.data["raw_ic"].iloc[0] == pytest.approx(-1.0)
    assert result.data["ic"].iloc[0] == pytest.approx(1.0)
    assert result.stats["mean_ic"] == pytest.approx(1.0)
    assert result.stats["positive_ic_ratio"] == 1.0


def test_compute_information_coefficient_skips_undefined_dates() -> None:
    panel = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "factor__constant": [1.0, 1.0, 2.0],
            "forward_return": [0.01, 0.02, 0.03],
        }
    )

    result = compute_information_coefficient(
        panel,
        factor_column="factor__constant",
    )

    assert result.data.empty
    assert result.stats == {
        "method": "pearson",
        "ic_date_count": 0,
        "skipped_date_count": 2,
        "mean_ic": None,
        "std_ic": None,
        "positive_ic_ratio": None,
        "average_pair_count": 0.0,
    }


def test_compute_rank_information_coefficient_handles_ties() -> None:
    panel = pd.DataFrame(
        {
            "date": ["2024-01-01"] * 5,
            "factor__alpha": [1.0, 1.0, 2.0, 3.0, 3.0],
            "forward_return": [0.01, 0.02, 0.03, 0.05, 0.04],
        }
    )

    result = compute_rank_information_coefficient(
        panel,
        factor_column="factor__alpha",
    )

    assert result.data["rank_ic"].iloc[0] == pytest.approx(0.9486832980505138)
    assert result.data["raw_rank_ic"].iloc[0] == pytest.approx(0.9486832980505138)
    assert result.data["pair_count"].iloc[0] == 5
    assert result.stats["method"] == "spearman"
    assert result.stats["mean_rank_ic"] == pytest.approx(0.9486832980505138)


def test_compute_rank_information_coefficient_adjusts_negative_direction() -> None:
    panel = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-01"],
            "factor__negative": [3.0, 2.0, 1.0],
            "forward_return": [0.01, 0.02, 0.03],
        }
    )

    result = compute_rank_information_coefficient(
        panel,
        factor_column="factor__negative",
        factor_direction="negative",
    )

    assert result.data["raw_rank_ic"].iloc[0] == pytest.approx(-1.0)
    assert result.data["rank_ic"].iloc[0] == pytest.approx(1.0)
    assert result.stats["mean_rank_ic"] == pytest.approx(1.0)
    assert result.stats["positive_rank_ic_ratio"] == 1.0


def test_compute_rank_information_coefficient_skips_undefined_dates() -> None:
    panel = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "factor__constant": [1.0, 1.0, 2.0],
            "forward_return": [0.01, 0.02, 0.03],
        }
    )

    result = compute_rank_information_coefficient(
        panel,
        factor_column="factor__constant",
    )

    assert result.data.empty
    assert result.stats == {
        "method": "spearman",
        "rank_ic_date_count": 0,
        "skipped_date_count": 2,
        "mean_rank_ic": None,
        "std_rank_ic": None,
        "positive_rank_ic_ratio": None,
        "average_pair_count": 0.0,
    }


def test_compute_sharpe_ratio_annualizes_long_short_returns() -> None:
    returns = pd.Series([0.01, 0.02, -0.01])
    result = compute_sharpe_ratio(
        pd.DataFrame({"long_short_return": returns}),
        annualization_factor=12,
    )

    expected_sharpe = returns.mean() / returns.std() * (12**0.5)
    assert result.stats["method"] == "mean_std"
    assert result.stats["annualization_factor"] == 12
    assert result.stats["return_count"] == 3
    assert result.stats["mean_period_return"] == pytest.approx(returns.mean())
    assert result.stats["std_period_return"] == pytest.approx(returns.std())
    assert result.stats["annualized_mean_return"] == pytest.approx(returns.mean() * 12)
    assert result.stats["sharpe"] == pytest.approx(expected_sharpe)
    assert result.stats["positive_return_ratio"] == pytest.approx(2 / 3)


def test_compute_sharpe_ratio_returns_none_for_zero_volatility() -> None:
    result = compute_sharpe_ratio(
        pd.DataFrame({"long_short_return": [0.01, 0.01, 0.01]})
    )

    assert result.stats["return_count"] == 3
    assert result.stats["mean_period_return"] == pytest.approx(0.01)
    assert result.stats["std_period_return"] == pytest.approx(0.0)
    assert result.stats["sharpe"] is None


def test_compute_sharpe_ratio_rejects_missing_return_column() -> None:
    with pytest.raises(ValueError, match="long_short_return"):
        compute_sharpe_ratio(pd.DataFrame({"other_return": [0.01]}))


def test_compute_drawdown_tracks_peak_trough_and_recovery() -> None:
    result = compute_drawdown(
        pd.DataFrame(
            {
                "date": [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                ],
                "long_short_return": [0.10, -0.20, 0.05, 0.25],
            }
        )
    )

    assert result.data["equity_curve"].tolist() == pytest.approx(
        [1.1, 0.88, 0.924, 1.155]
    )
    assert result.data["drawdown"].tolist() == pytest.approx(
        [0.0, -0.2, -0.16, 0.0]
    )
    assert result.stats["method"] == "cumulative_return"
    assert result.stats["return_count"] == 4
    assert result.stats["end_equity"] == pytest.approx(1.155)
    assert result.stats["total_return"] == pytest.approx(0.155)
    assert result.stats["max_drawdown"] == pytest.approx(-0.2)
    assert result.stats["max_drawdown_abs"] == pytest.approx(0.2)
    assert result.stats["peak_date"] == "2024-01-01"
    assert result.stats["trough_date"] == "2024-01-02"
    assert result.stats["recovery_date"] == "2024-01-04"
    assert result.stats["drawdown_period_count"] == 2
    assert result.stats["average_drawdown"] == pytest.approx(-0.18)


def test_compute_drawdown_returns_empty_stats_without_valid_returns() -> None:
    result = compute_drawdown(
        pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "long_short_return": [None],
            }
        )
    )

    assert result.data.empty
    assert result.stats == {
        "method": "cumulative_return",
        "return_column": "long_short_return",
        "return_count": 0,
        "start_equity": 1.0,
        "end_equity": None,
        "total_return": None,
        "max_drawdown": None,
        "max_drawdown_abs": None,
        "peak_date": None,
        "trough_date": None,
        "recovery_date": None,
        "drawdown_period_count": 0,
        "average_drawdown": None,
    }


def test_compute_drawdown_rejects_missing_return_column() -> None:
    with pytest.raises(ValueError, match="long_short_return"):
        compute_drawdown(pd.DataFrame({"date": ["2024-01-01"], "other": [0.01]}))


def test_save_backtest_result_json_is_optional(tmp_path: Path) -> None:
    result_json = {"schema_version": 1, "state": "backtest_result_generated"}

    assert save_backtest_result_json(result_json, None) is None

    path = tmp_path / "nested" / "result.json"
    saved_path = save_backtest_result_json(result_json, path)

    assert saved_path == path
    assert json.loads(path.read_text(encoding="utf-8")) == result_json


def test_run_benchmark_tests_reports_failures() -> None:
    result_json = {
        "summary": {
            "usable_row_count": 8,
            "portfolio_date_count": 2,
            "ic_date_count": 2,
            "rank_ic_date_count": 2,
            "mean_ic": 0.04,
            "mean_rank_ic": 0.03,
            "sharpe": 0.8,
            "total_return": -0.01,
        },
        "metrics": {"drawdown": {"max_drawdown_abs": 0.25}},
    }

    benchmark_tests = run_benchmark_tests(
        result_json,
        {
            "min_usable_rows": 10,
            "min_portfolio_dates": 2,
            "min_ic_dates": 2,
            "min_rank_ic_dates": 2,
            "min_mean_ic": 0.05,
            "min_mean_rank_ic": 0.05,
            "min_sharpe": 1.0,
            "min_total_return": 0.0,
            "max_drawdown_abs": 0.20,
        },
    )

    assert benchmark_tests["status"] == "failed"
    assert benchmark_tests["test_count"] == 9
    assert benchmark_tests["passed_count"] == 3
    assert benchmark_tests["failed_count"] == 6
    failed_names = {
        test["name"] for test in benchmark_tests["tests"] if not test["passed"]
    }
    assert failed_names == {
        "usable_row_count",
        "mean_ic",
        "mean_rank_ic",
        "sharpe",
        "total_return",
        "max_drawdown_abs",
    }
