from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from agents.out_of_sample_agent import (
    OutOfSampleAgent,
    OutOfSampleSpec,
    ValidationSplitSpec,
    WalkForwardSpec,
)
from core.config import AppConfig
from core.models import AgentRequest


def _symbols() -> list[str]:
    return [f"{number:06d}" for number in range(1, 7)]


def _dates() -> list[str]:
    return [f"2024-01-{day:02d}" for day in range(1, 8)]


def _write_aligned_data(tmp_path: Path) -> Path:
    rows = []
    daily_returns = [-0.06, -0.04, -0.02, 0.02, 0.04, 0.06]
    closes = {symbol: 100.0 for symbol in _symbols()}
    for index, trade_date in enumerate(_dates()):
        for symbol in _symbols():
            rows.append({"date": trade_date, "symbol": symbol, "close": closes[symbol]})
        if index < len(_dates()) - 1:
            for symbol, daily_return in zip(_symbols(), daily_returns, strict=True):
                closes[symbol] = closes[symbol] * (1.0 + daily_return)

    path = tmp_path / "aligned.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_factor_matrix(tmp_path: Path) -> Path:
    rows = []
    for trade_date in _dates()[:-1]:
        for value, symbol in enumerate(_symbols(), start=1):
            rows.append(
                {
                    "date": trade_date,
                    "symbol": symbol,
                    "factor__alpha": float(value),
                }
            )
    path = tmp_path / "factor_matrix.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_manifest(
    tmp_path: Path,
    *,
    factor_matrix_path: Path,
    aligned_data_path: Path,
) -> Path:
    path = tmp_path / "factor_matrix.manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "storage": {
                    "matrix_path": str(factor_matrix_path),
                    "factor_columns": ["factor__alpha"],
                },
                "context": {
                    "source_aligned_data_path": str(aligned_data_path),
                    "factor_definitions": [
                        {
                            "factor_column": "factor__alpha",
                            "name": "alpha",
                            "direction": "positive",
                            "category": "test",
                        }
                    ],
                },
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    return path


def _loose_thresholds() -> dict[str, float | int | None]:
    return {
        "min_usable_rows": 12,
        "min_portfolio_dates": 2,
        "min_ic_dates": 2,
        "min_rank_ic_dates": 2,
        "min_average_leg_count": 2,
        "min_mean_ic": 0.9,
        "min_mean_rank_ic": 0.9,
        "min_sharpe": None,
        "min_total_return": None,
        "max_drawdown_abs": None,
    }


def test_out_of_sample_spec_accepts_train_validation_test(tmp_path: Path) -> None:
    manifest_path = tmp_path / "factor_matrix.manifest.json"
    spec = OutOfSampleSpec.from_payload(
        {
            "factor_manifest_path": str(manifest_path),
            "factor_column": "factor__alpha",
            "validation_id": "alpha-oos",
            "splits": [
                {"name": "train", "start_date": "2024-01-01", "end_date": "2024-01-02"},
                {
                    "name": "validation",
                    "start_date": "2024-01-03",
                    "end_date": "2024-01-04",
                },
                {"name": "test", "start_date": "2024-01-05", "end_date": "2024-01-06"},
            ],
            "benchmark_thresholds": _loose_thresholds(),
        }
    )

    assert spec.factor_manifest_path == manifest_path.resolve()
    assert spec.factor_column == "factor__alpha"
    assert spec.validation_id == "alpha-oos"
    assert [split.name for split in spec.splits] == ["train", "validation", "test"]
    assert [split.role for split in spec.splits] == ["train", "validation", "test"]
    assert spec.splits[0].start_date == "2024-01-01"
    assert spec.validation_method == "explicit_splits"
    assert spec.walk_forward is None
    assert spec.benchmark_thresholds["min_mean_rank_ic"] == 0.9
    assert spec.transaction_costs.enabled is True


def test_walk_forward_spec_generates_train_test_splits() -> None:
    spec = WalkForwardSpec.from_mapping(
        {
            "start_date": "2024-01-01",
            "end_date": "2024-01-06",
            "train_window_days": 2,
            "test_window_days": 2,
            "step_days": 2,
        }
    )

    assert [
        (split.name, split.role, split.fold_index, split.start_date, split.end_date)
        for split in spec.to_splits()
    ] == [
        ("walk_001_train", "train", 1, "2024-01-01", "2024-01-02"),
        ("walk_001_test", "test", 1, "2024-01-03", "2024-01-04"),
        ("walk_002_train", "train", 2, "2024-01-03", "2024-01-04"),
        ("walk_002_test", "test", 2, "2024-01-05", "2024-01-06"),
    ]


def test_out_of_sample_spec_accepts_walk_forward(tmp_path: Path) -> None:
    manifest_path = tmp_path / "factor_matrix.manifest.json"
    spec = OutOfSampleSpec.from_payload(
        {
            "factor_manifest_path": str(manifest_path),
            "factor_column": "factor__alpha",
            "walk_forward": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-06",
                "train_window_days": 2,
                "test_window_days": 2,
                "step_days": 2,
            },
        }
    )

    assert spec.validation_method == "walk_forward"
    assert spec.walk_forward is not None
    assert len(spec.splits) == 4
    assert [split.role for split in spec.splits] == ["train", "test", "train", "test"]
    assert [split.fold_index for split in spec.splits] == [1, 1, 2, 2]


def test_validation_split_rejects_invalid_date_range() -> None:
    with pytest.raises(ValueError, match="start_date"):
        ValidationSplitSpec.from_mapping(
            {
                "name": "train",
                "start_date": "2024-02-01",
                "end_date": "2024-01-01",
            }
        )


def test_out_of_sample_spec_rejects_splits_and_walk_forward_together(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="only one"):
        OutOfSampleSpec.from_payload(
            {
                "factor_manifest_path": str(tmp_path / "factor_matrix.manifest.json"),
                "factor_column": "factor__alpha",
                "splits": [
                    {
                        "name": "train",
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-02",
                    },
                ],
                "walk_forward": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-06",
                    "train_window_days": 2,
                    "test_window_days": 2,
                    "step_days": 2,
                },
            }
        )


def test_out_of_sample_spec_rejects_duplicate_split_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unique"):
        OutOfSampleSpec.from_payload(
            {
                "factor_manifest_path": str(tmp_path / "factor_matrix.manifest.json"),
                "factor_column": "factor__alpha",
                "splits": [
                    {
                        "name": "train",
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-02",
                    },
                    {
                        "name": "train",
                        "start_date": "2024-01-03",
                        "end_date": "2024-01-04",
                    },
                ],
            }
        )


def test_out_of_sample_agent_runs_split_backtests_and_saves_artifacts(
    tmp_path: Path,
) -> None:
    aligned_path = _write_aligned_data(tmp_path)
    factor_matrix_path = _write_factor_matrix(tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        factor_matrix_path=factor_matrix_path,
        aligned_data_path=aligned_path,
    )
    agent = OutOfSampleAgent(
        config=AppConfig.from_env(project_root=tmp_path),
    )

    response = agent.run(
        AgentRequest.create(
            {
                "factor_manifest_path": str(manifest_path),
                "factor_column": "factor__alpha",
                "validation_id": "alpha-oos",
                "output_dir": "validations",
                "splits": [
                    {
                        "name": "train",
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-02",
                    },
                    {
                        "name": "validation",
                        "start_date": "2024-01-03",
                        "end_date": "2024-01-04",
                    },
                    {
                        "name": "test",
                        "start_date": "2024-01-05",
                        "end_date": "2024-01-06",
                    },
                ],
                "quantile_count": 3,
                "benchmark_thresholds": _loose_thresholds(),
                "preview_rows": 1,
            },
            task_id="oos-task-1",
        )
    )

    assert response.status == "success"
    assert response.output["state"] == "out_of_sample_validated"
    assert response.output["validation_id"] == "alpha-oos"
    assert response.output["validation_status"] == "success"
    assert response.output["factor_column"] == "factor__alpha"
    assert response.output["summary"]["split_count"] == 3
    assert response.output["summary"]["successful_split_count"] == 3
    assert response.output["summary"]["failed_split_count"] == 0
    assert response.output["summary"]["benchmark_status_counts"] == {"passed": 3}
    assert response.output["summary"]["basic_oos_check"]["status"] == "passed"
    assert response.output["summary"]["basic_oos_check"][
        "rank_ic_direction_consistent"
    ] is True
    assert response.metadata["agent"] == "OutOfSampleAgent"
    assert response.metadata["validation_id"] == "alpha-oos"

    records = response.output["records"]
    assert [record["split_name"] for record in records] == [
        "train",
        "validation",
        "test",
    ]
    for record in records:
        assert record["status"] == "success"
        assert record["benchmark_status"] == "passed"
        assert record["metrics_snapshot"]["usable_row_count"] == 12
        assert record["metrics_snapshot"]["portfolio_date_count"] == 2
        assert record["metrics_snapshot"]["mean_rank_ic"] == pytest.approx(1.0)
        assert Path(record["result_json_path"]).is_file()

    storage_stats = response.output["storage_stats"]
    result_path = Path(storage_stats["result_path"])
    summary_path = Path(storage_stats["summary_path"])
    assert result_path.is_file()
    assert summary_path.is_file()
    saved_result = json.loads(result_path.read_text(encoding="utf-8"))
    assert saved_result["summary"] == response.output["summary"]
    saved_summary = pd.read_csv(summary_path)
    assert list(saved_summary["split_name"]) == ["train", "validation", "test"]
    assert saved_summary["benchmark_status"].tolist() == ["passed", "passed", "passed"]


def test_out_of_sample_agent_runs_walk_forward_and_saves_artifacts(
    tmp_path: Path,
) -> None:
    aligned_path = _write_aligned_data(tmp_path)
    factor_matrix_path = _write_factor_matrix(tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        factor_matrix_path=factor_matrix_path,
        aligned_data_path=aligned_path,
    )
    agent = OutOfSampleAgent(
        config=AppConfig.from_env(project_root=tmp_path),
    )

    response = agent.run(
        AgentRequest.create(
            {
                "factor_manifest_path": str(manifest_path),
                "factor_column": "factor__alpha",
                "validation_id": "alpha-walk-forward",
                "output_dir": "validations",
                "walk_forward": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-06",
                    "train_window_days": 2,
                    "test_window_days": 2,
                    "step_days": 2,
                },
                "quantile_count": 3,
                "benchmark_thresholds": _loose_thresholds(),
                "preview_rows": 1,
            },
            task_id="oos-task-2",
        )
    )

    assert response.status == "success"
    assert response.output["request"]["validation_method"] == "walk_forward"
    assert response.output["summary"]["split_count"] == 4
    assert response.output["summary"]["successful_split_count"] == 4
    assert response.output["summary"]["benchmark_status_counts"] == {"passed": 4}
    assert response.output["summary"]["basic_oos_check"]["status"] == "passed"
    walk_check = response.output["summary"]["walk_forward_check"]
    assert walk_check["status"] == "passed"
    assert walk_check["fold_count"] == 2
    assert walk_check["passed_fold_count"] == 2
    assert walk_check["failed_fold_count"] == 0
    assert walk_check["rank_ic_direction_consistent_count"] == 2
    assert walk_check["test_benchmark_passed_count"] == 2

    records = response.output["records"]
    assert [record["split_name"] for record in records] == [
        "walk_001_train",
        "walk_001_test",
        "walk_002_train",
        "walk_002_test",
    ]
    assert [record["split_role"] for record in records] == [
        "train",
        "test",
        "train",
        "test",
    ]
    assert [record["fold_index"] for record in records] == [1, 1, 2, 2]
    for record in records:
        assert record["status"] == "success"
        assert record["benchmark_status"] == "passed"
        assert record["metrics_snapshot"]["usable_row_count"] == 12
        assert Path(record["result_json_path"]).is_file()

    summary_path = Path(response.output["storage_stats"]["summary_path"])
    saved_summary = pd.read_csv(summary_path)
    assert list(saved_summary["split_role"]) == ["train", "test", "train", "test"]
    assert saved_summary["fold_index"].tolist() == [1, 1, 2, 2]
