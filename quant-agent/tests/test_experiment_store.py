from __future__ import annotations

import csv
import json
from pathlib import Path

from agents.experiment_store import ExperimentStore


def test_experiment_store_writes_result_and_summary(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "experiments")
    document = {
        "experiment_id": "demo experiment",
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
