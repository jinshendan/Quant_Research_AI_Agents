from __future__ import annotations

import csv
import json
from pathlib import Path

from agents.experiment_store import ExperimentStore


def test_experiment_store_writes_result_and_summary(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "experiments")
    document = {
        "experiment_id": "demo experiment",
        "created_at": "2026-05-21T10:00:00+00:00",
        "task_id": "task-1",
        "storage_schema_version": 1,
        "factor_manifest_path": "/tmp/factors.manifest.json",
        "request": {
            "output_dir": "experiments",
            "forward_return_days": 1,
            "quantile_count": 5,
        },
        "summary": {"status": "success", "factor_count": 1},
        "factor_definitions": [
            {
                "factor_id": "alpha",
                "factor_column": "factor__alpha",
                "name": "Alpha",
                "source_type": "manual",
                "category": "momentum",
                "formula": "return_5d",
                "hypothesis": "Momentum persists.",
            }
        ],
        "records": [
            {
                "factor_column": "factor__alpha",
                "factor_direction": "positive",
                "status": "success",
                "stage": "completed",
                "benchmark_status": "passed",
                "critic_verdict": "track",
                "critic_severity": "low",
                "metrics_snapshot": {
                    "mean_ic": 0.1,
                    "mean_rank_ic": 0.2,
                    "net_sharpe": 1.5,
                    "net_total_return": 0.3,
                    "max_drawdown_abs": 0.05,
                },
                "failed_benchmark_tests": [],
                "error": None,
                "result_json_path": "/tmp/result.json",
            }
        ],
    }

    result = store.store(document)

    assert result.experiment_id == "demo experiment"
    assert result.records_written == 1
    assert result.index_records_written == 1
    assert result.run_dir == tmp_path / "experiments" / "demo_experiment"
    assert json.loads(result.result_path.read_text(encoding="utf-8")) == document

    rows = list(csv.DictReader(result.summary_path.open(encoding="utf-8")))
    assert rows == [
        {
            "experiment_id": "demo experiment",
            "factor_column": "factor__alpha",
            "factor_direction": "positive",
            "status": "success",
            "stage": "completed",
            "benchmark_status": "passed",
            "critic_verdict": "track",
            "critic_severity": "low",
            "mean_ic": "0.1",
            "mean_rank_ic": "0.2",
            "net_sharpe": "1.5",
            "net_total_return": "0.3",
            "max_drawdown_abs": "0.05",
            "failed_benchmark_tests": "[]",
            "error": "",
            "result_json_path": "/tmp/result.json",
        }
    ]

    index_rows = [
        json.loads(line)
        for line in result.index_path.read_text(encoding="utf-8").splitlines()
    ]
    assert index_rows == [
        {
            "index_schema_version": 1,
            "storage_schema_version": 1,
            "experiment_id": "demo experiment",
            "created_at": "2026-05-21T10:00:00+00:00",
            "task_id": "task-1",
            "experiment_status": "success",
            "factor_count": 1,
            "factor_manifest_path": "/tmp/factors.manifest.json",
            "output_dir": "experiments",
            "forward_return_days": 1,
            "quantile_count": 5,
            "factor_column": "factor__alpha",
            "factor_direction": "positive",
            "factor_id": "alpha",
            "factor_name": "Alpha",
            "factor_source_type": "manual",
            "factor_category": "momentum",
            "factor_formula": "return_5d",
            "factor_hypothesis": "Momentum persists.",
            "status": "success",
            "stage": "completed",
            "benchmark_status": "passed",
            "critic_verdict": "track",
            "critic_severity": "low",
            "mean_ic": 0.1,
            "mean_rank_ic": 0.2,
            "net_sharpe": 1.5,
            "net_total_return": 0.3,
            "max_drawdown_abs": 0.05,
            "average_turnover": None,
            "failed_benchmark_tests": [],
            "error": "",
            "result_json_path": "/tmp/result.json",
            "experiment_result_path": str(result.result_path),
            "experiment_summary_path": str(result.summary_path),
        }
    ]


def test_experiment_store_replaces_index_rows_for_same_experiment(
    tmp_path: Path,
) -> None:
    store = ExperimentStore(tmp_path / "experiments")

    first = _minimal_document(
        experiment_id="same-experiment",
        factor_column="factor__old",
        verdict="reject_for_now",
    )
    second = _minimal_document(
        experiment_id="same-experiment",
        factor_column="factor__new",
        verdict="track",
    )

    store.store(first)
    result = store.store(second)

    index_rows = [
        json.loads(line)
        for line in result.index_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["factor_column"] for row in index_rows] == ["factor__new"]
    assert index_rows[0]["critic_verdict"] == "track"


def test_experiment_store_appends_index_rows_for_different_experiments(
    tmp_path: Path,
) -> None:
    store = ExperimentStore(tmp_path / "experiments")

    first = _minimal_document(
        experiment_id="experiment-a",
        factor_column="factor__a",
        verdict="track",
    )
    second = _minimal_document(
        experiment_id="experiment-b",
        factor_column="factor__b",
        verdict="reject_for_now",
    )

    store.store(first)
    result = store.store(second)

    index_rows = [
        json.loads(line)
        for line in result.index_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["experiment_id"] for row in index_rows] == [
        "experiment-a",
        "experiment-b",
    ]
    assert [row["factor_column"] for row in index_rows] == [
        "factor__a",
        "factor__b",
    ]


def _minimal_document(
    *,
    experiment_id: str,
    factor_column: str,
    verdict: str,
) -> dict[str, object]:
    return {
        "experiment_id": experiment_id,
        "summary": {"status": "success", "factor_count": 1},
        "factor_definitions": [
            {
                "factor_id": factor_column.removeprefix("factor__"),
                "factor_column": factor_column,
                "name": factor_column,
                "category": "unit_test",
            }
        ],
        "records": [
            {
                "factor_column": factor_column,
                "factor_direction": "positive",
                "status": "success",
                "stage": "completed",
                "benchmark_status": "passed",
                "critic_verdict": verdict,
                "critic_severity": "low",
                "metrics_snapshot": {},
                "failed_benchmark_tests": [],
                "error": None,
                "result_json_path": f"/tmp/{factor_column}.json",
            }
        ],
    }
