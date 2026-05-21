from __future__ import annotations

import csv
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from agents.experiment_agent import ExperimentAgent, ExperimentSpec
from core.config import AppConfig
from core.models import AgentRequest


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig.from_env(project_root=tmp_path, environ={})


def test_experiment_spec_normalizes_valid_payload(tmp_path: Path) -> None:
    manifest_path = tmp_path / "factor.manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    spec = ExperimentSpec.from_payload(
        {
            "factor_manifest_path": str(manifest_path),
            "experiment_id": "demo exp",
            "output_dir": "exp_runs",
            "factor_columns": ["factor__alpha", "factor__alpha", "factor__beta"],
            "factor_direction": "negative",
            "factor_directions": {"factor__beta": "positive"},
            "continue_on_error": False,
            "preview_rows": 2,
            "output_language": "zh",
        }
    )

    assert spec.factor_manifest_path == manifest_path.resolve()
    assert spec.effective_experiment_id == "demo_exp"
    assert spec.output_dir == Path("exp_runs")
    assert spec.factor_columns == ("factor__alpha", "factor__beta")
    assert spec.factor_direction == "negative"
    assert spec.factor_directions == {"factor__beta": "positive"}
    assert spec.continue_on_error is False
    assert spec.preview_rows == 2
    assert spec.output_language == "zh"


def test_experiment_agent_runs_factor_batch_and_writes_artifacts(tmp_path: Path) -> None:
    manifest_path = _write_experiment_fixture(tmp_path)
    agent = ExperimentAgent(config=_config(tmp_path))

    response = agent.run(
        AgentRequest.create(
            {
                "factor_manifest_path": str(manifest_path),
                "experiment_id": "batch-demo",
                "output_dir": "experiments",
                "factor_columns": ["factor__good", "factor__bad"],
                "quantile_count": 3,
                "benchmark_thresholds": {
                    "min_usable_rows": 18,
                    "min_portfolio_dates": 3,
                    "min_ic_dates": 3,
                    "min_rank_ic_dates": 3,
                    "min_average_leg_count": 2,
                    "min_mean_ic": 0.5,
                    "min_mean_rank_ic": 0.5,
                    "min_sharpe": 0.1,
                    "min_total_return": 0.0,
                    "max_drawdown_abs": 0.2,
                },
                "preview_rows": 0,
            },
            task_id="experiment-task-1",
        )
    )

    assert response.status == "success"
    assert response.output["state"] == "experiment_completed"
    assert response.output["experiment_id"] == "batch-demo"
    assert response.output["factor_count"] == 2
    assert response.output["successful_factor_count"] == 2
    assert response.output["failed_factor_count"] == 0
    assert response.output["summary"]["status"] == "success"
    assert response.output["summary"]["track_factor_columns"] == ["factor__good"]
    assert response.output["summary"]["rejected_factor_columns"] == ["factor__bad"]

    records = response.output["records"]
    assert records[0]["factor_column"] == "factor__good"
    assert records[0]["benchmark_status"] == "passed"
    assert records[0]["critic_verdict"] == "track"
    assert records[1]["factor_column"] == "factor__bad"
    assert records[1]["benchmark_status"] == "failed"
    assert records[1]["critic_verdict"] == "reject_for_now"

    storage = response.output["storage_stats"]
    result_path = Path(storage["result_path"])
    summary_path = Path(storage["summary_path"])
    index_path = Path(storage["index_path"])
    assert result_path.is_file()
    assert summary_path.is_file()
    assert index_path.is_file()
    saved = json.loads(result_path.read_text(encoding="utf-8"))
    assert saved["selected_factor_columns"] == ["factor__good", "factor__bad"]
    assert saved["factor_definitions"][0]["factor_column"] == "factor__good"

    summary_rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    assert [row["factor_column"] for row in summary_rows] == [
        "factor__good",
        "factor__bad",
    ]
    assert Path(records[0]["result_json_path"]).is_file()
    assert Path(records[1]["result_json_path"]).is_file()


def test_experiment_agent_rejects_unknown_factor_column(tmp_path: Path) -> None:
    manifest_path = _write_experiment_fixture(tmp_path)
    response = ExperimentAgent(config=_config(tmp_path)).run(
        AgentRequest.create(
            {
                "factor_manifest_path": str(manifest_path),
                "factor_columns": ["factor__missing"],
            }
        )
    )

    assert response.status == "error"
    assert "factor__missing" in str(response.error)


def _write_experiment_fixture(tmp_path: Path) -> Path:
    symbols = tuple(f"{number:06d}" for number in range(1, 7))
    start = date(2024, 1, 1)
    aligned_rows = []
    factor_rows = []
    close_by_symbol = {symbol: 100.0 for symbol in symbols}
    for day_offset in range(5):
        current = start + timedelta(days=day_offset)
        for symbol in symbols:
            signal = _symbol_signal(symbol)
            close = close_by_symbol[symbol]
            aligned_rows.append(
                {
                    "date": current,
                    "symbol": symbol,
                    "open": close / (1.0 + signal),
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1000.0,
                    "amount": close * 1000.0,
                    "turnover_rate": 1.0,
                    "is_expected_trading_day": True,
                    "is_suspended_or_missing": False,
                }
            )
            factor_rows.append(
                {
                    "date": current,
                    "symbol": symbol,
                    "factor__good": signal,
                    "factor__bad": -signal,
                }
            )
            close_by_symbol[symbol] *= 1.0 + signal

    aligned_path = tmp_path / "aligned.csv"
    matrix_path = tmp_path / "factors.csv"
    manifest_path = tmp_path / "factors.manifest.json"
    pd.DataFrame(aligned_rows).to_csv(aligned_path, index=False)
    pd.DataFrame(factor_rows).to_csv(matrix_path, index=False)
    manifest = {
        "schema_version": 1,
        "storage": {
            "matrix_path": str(matrix_path),
            "factor_columns": ["factor__good", "factor__bad"],
        },
        "context": {
            "source_aligned_data_path": str(aligned_path),
            "factor_definitions": [
                {
                    "factor_id": "good",
                    "factor_column": "factor__good",
                    "name": "good",
                    "source_type": "manual",
                    "formula": "signal",
                    "hypothesis": "Higher signal predicts return.",
                    "category": "unit_test",
                    "direction": "positive",
                    "lookback_days": 1,
                    "data_lag_days": 0,
                },
                {
                    "factor_id": "bad",
                    "factor_column": "factor__bad",
                    "name": "bad",
                    "source_type": "manual",
                    "formula": "-signal",
                    "hypothesis": "Wrong-way signal should fail.",
                    "category": "unit_test",
                    "direction": "positive",
                    "lookback_days": 1,
                    "data_lag_days": 0,
                },
            ],
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _symbol_signal(symbol: str) -> float:
    signals = {
        "000001": -0.04,
        "000002": -0.02,
        "000003": -0.01,
        "000004": 0.01,
        "000005": 0.02,
        "000006": 0.04,
    }
    return signals[symbol]
