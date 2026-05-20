from __future__ import annotations

import csv
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPERIMENT_STORAGE_SCHEMA_VERSION = 1
EXPERIMENT_RESULT_FILENAME = "experiment_result.json"
EXPERIMENT_SUMMARY_FILENAME = "experiment_summary.csv"
_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class ExperimentStorageResult:
    """Paths and counts for a persisted experiment run."""

    experiment_id: str
    run_dir: Path
    result_path: Path
    summary_path: Path
    records_written: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "run_dir": str(self.run_dir),
            "result_path": str(self.result_path),
            "summary_path": str(self.summary_path),
            "records_written": self.records_written,
        }


class ExperimentStore:
    """File-backed storage for batch factor experiment runs."""

    def __init__(self, experiments_dir: str | Path) -> None:
        self.experiments_dir = Path(experiments_dir)

    def run_dir(self, experiment_id: str) -> Path:
        return self.experiments_dir / _safe_name(experiment_id)

    def store(self, document: Mapping[str, Any]) -> ExperimentStorageResult:
        experiment_id = _required_str(document, "experiment_id")
        records = _required_records(document)
        run_dir = self.run_dir(experiment_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        result_path = run_dir / EXPERIMENT_RESULT_FILENAME
        summary_path = run_dir / EXPERIMENT_SUMMARY_FILENAME

        temp_result_path = Path(f"{result_path}.tmp")
        temp_result_path.write_text(
            json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_result_path.replace(result_path)

        _write_summary_csv(summary_path, experiment_id, records)
        return ExperimentStorageResult(
            experiment_id=experiment_id,
            run_dir=run_dir,
            result_path=result_path,
            summary_path=summary_path,
            records_written=len(records),
        )


def _write_summary_csv(
    summary_path: Path,
    experiment_id: str,
    records: Sequence[Mapping[str, Any]],
) -> None:
    fieldnames = (
        "experiment_id",
        "factor_column",
        "factor_direction",
        "status",
        "stage",
        "benchmark_status",
        "critic_verdict",
        "critic_severity",
        "mean_ic",
        "mean_rank_ic",
        "net_sharpe",
        "net_total_return",
        "max_drawdown_abs",
        "failed_benchmark_tests",
        "error",
        "result_json_path",
    )
    temp_summary_path = Path(f"{summary_path}.tmp")
    with temp_summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            metrics = _mapping(record.get("metrics_snapshot"))
            row = {
                "experiment_id": experiment_id,
                "factor_column": _text(record.get("factor_column")),
                "factor_direction": _text(record.get("factor_direction")),
                "status": _text(record.get("status")),
                "stage": _text(record.get("stage")),
                "benchmark_status": _text(record.get("benchmark_status")),
                "critic_verdict": _text(record.get("critic_verdict")),
                "critic_severity": _text(record.get("critic_severity")),
                "mean_ic": _text(metrics.get("mean_ic")),
                "mean_rank_ic": _text(metrics.get("mean_rank_ic")),
                "net_sharpe": _text(metrics.get("net_sharpe")),
                "net_total_return": _text(metrics.get("net_total_return")),
                "max_drawdown_abs": _text(metrics.get("max_drawdown_abs")),
                "failed_benchmark_tests": json.dumps(
                    record.get("failed_benchmark_tests") or [],
                    ensure_ascii=True,
                ),
                "error": _text(record.get("error")),
                "result_json_path": _text(record.get("result_json_path")),
            }
            writer.writerow(row)
    temp_summary_path.replace(summary_path)


def _required_records(document: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    records = document.get("records")
    if isinstance(records, str) or not isinstance(records, Sequence):
        msg = "experiment document records must be a sequence."
        raise ValueError(msg)
    normalized = []
    for record in records:
        if not isinstance(record, Mapping):
            msg = "experiment document records must contain objects."
            raise ValueError(msg)
        normalized.append(record)
    return tuple(normalized)


def _required_str(document: Mapping[str, Any], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value.strip():
        msg = f"experiment document {key} must be a non-empty string."
        raise ValueError(msg)
    return value.strip()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _safe_name(value: str) -> str:
    cleaned = _SAFE_NAME_PATTERN.sub("_", value.strip())
    return cleaned.strip("._") or "experiment"
