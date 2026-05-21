from __future__ import annotations

import csv
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

EXPERIMENT_STORAGE_SCHEMA_VERSION = 1
EXPERIMENT_INDEX_SCHEMA_VERSION = 1
EXPERIMENT_RESULT_FILENAME = "experiment_result.json"
EXPERIMENT_SUMMARY_FILENAME = "experiment_summary.csv"
EXPERIMENT_INDEX_FILENAME = "experiment_index.jsonl"
_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class ExperimentStorageResult:
    """Paths and counts for a persisted experiment run."""

    experiment_id: str
    run_dir: Path
    result_path: Path
    summary_path: Path
    index_path: Path
    records_written: int
    index_records_written: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "run_dir": str(self.run_dir),
            "result_path": str(self.result_path),
            "summary_path": str(self.summary_path),
            "index_path": str(self.index_path),
            "records_written": self.records_written,
            "index_records_written": self.index_records_written,
        }


@dataclass(frozen=True, slots=True)
class ExperimentQuerySpec:
    """Filters for reading the experiment history index."""

    experiment_ids: tuple[str, ...] = ()
    factor_columns: tuple[str, ...] = ()
    factor_categories: tuple[str, ...] = ()
    factor_source_types: tuple[str, ...] = ()
    benchmark_statuses: tuple[str, ...] = ()
    critic_verdicts: tuple[str, ...] = ()
    experiment_statuses: tuple[str, ...] = ()
    statuses: tuple[str, ...] = ()
    factor_set_names: tuple[str, ...] = ()
    universes: tuple[str, ...] = ()
    config_hashes: tuple[str, ...] = ()
    data_versions: tuple[str, ...] = ()
    factor_manifest_hashes: tuple[str, ...] = ()
    factor_manifest_paths: tuple[str, ...] = ()
    source_aligned_data_paths: tuple[str, ...] = ()
    created_at_start: str | date | datetime | None = None
    created_at_end: str | date | datetime | None = None
    limit: int | None = None
    sort_desc: bool = True

    def __post_init__(self) -> None:
        if self.limit is not None and self.limit <= 0:
            msg = "ExperimentQuerySpec.limit must be positive when provided."
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_ids": list(self.experiment_ids),
            "factor_columns": list(self.factor_columns),
            "factor_categories": list(self.factor_categories),
            "factor_source_types": list(self.factor_source_types),
            "benchmark_statuses": list(self.benchmark_statuses),
            "critic_verdicts": list(self.critic_verdicts),
            "experiment_statuses": list(self.experiment_statuses),
            "statuses": list(self.statuses),
            "factor_set_names": list(self.factor_set_names),
            "universes": list(self.universes),
            "config_hashes": list(self.config_hashes),
            "data_versions": list(self.data_versions),
            "factor_manifest_hashes": list(self.factor_manifest_hashes),
            "factor_manifest_paths": list(self.factor_manifest_paths),
            "source_aligned_data_paths": list(self.source_aligned_data_paths),
            "created_at_start": _date_bound_to_text(self.created_at_start),
            "created_at_end": _date_bound_to_text(self.created_at_end),
            "limit": self.limit,
            "sort_desc": self.sort_desc,
        }


@dataclass(frozen=True, slots=True)
class ExperimentQueryResult:
    """Records returned from an experiment history query."""

    index_path: Path
    spec: ExperimentQuerySpec
    total_records: int
    matched_records: int
    records: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_path": str(self.index_path),
            "query": self.spec.to_dict(),
            "total_records": self.total_records,
            "matched_records": self.matched_records,
            "records": list(self.records),
        }


class ExperimentStore:
    """File-backed storage for batch factor experiment runs."""

    def __init__(self, experiments_dir: str | Path) -> None:
        self.experiments_dir = Path(experiments_dir)

    def run_dir(self, experiment_id: str) -> Path:
        return self.experiments_dir / _safe_name(experiment_id)

    @property
    def index_path(self) -> Path:
        return self.experiments_dir / EXPERIMENT_INDEX_FILENAME

    def store(self, document: Mapping[str, Any]) -> ExperimentStorageResult:
        experiment_id = _required_str(document, "experiment_id")
        records = _required_records(document)
        run_dir = self.run_dir(experiment_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        result_path = run_dir / EXPERIMENT_RESULT_FILENAME
        summary_path = run_dir / EXPERIMENT_SUMMARY_FILENAME
        index_path = self.index_path

        temp_result_path = Path(f"{result_path}.tmp")
        temp_result_path.write_text(
            json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_result_path.replace(result_path)

        _write_summary_csv(summary_path, experiment_id, records)
        index_records = _build_index_records(
            document=document,
            experiment_id=experiment_id,
            records=records,
            result_path=result_path,
            summary_path=summary_path,
        )
        _replace_index_records(index_path, experiment_id, index_records)
        return ExperimentStorageResult(
            experiment_id=experiment_id,
            run_dir=run_dir,
            result_path=result_path,
            summary_path=summary_path,
            index_path=index_path,
            records_written=len(records),
            index_records_written=len(index_records),
        )

    def query(
        self,
        spec: ExperimentQuerySpec | None = None,
    ) -> ExperimentQueryResult:
        query_spec = spec or ExperimentQuerySpec()
        records = _read_index_records(self.index_path)
        matched = tuple(_filter_index_records(records, query_spec))
        return ExperimentQueryResult(
            index_path=self.index_path,
            spec=query_spec,
            total_records=len(records),
            matched_records=len(matched),
            records=matched,
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


def _build_index_records(
    *,
    document: Mapping[str, Any],
    experiment_id: str,
    records: Sequence[Mapping[str, Any]],
    result_path: Path,
    summary_path: Path,
) -> tuple[dict[str, Any], ...]:
    definitions = _factor_definitions_by_column(document)
    request = _mapping(document.get("request"))
    summary = _mapping(document.get("summary"))
    lineage = _mapping(document.get("lineage"))
    data_version_inputs = _mapping(lineage.get("data_version_inputs"))
    factor_matrix = _mapping(data_version_inputs.get("factor_matrix"))
    source_aligned_data = _mapping(data_version_inputs.get("source_aligned_data"))
    index_records = []
    for record in records:
        factor_column = _text(record.get("factor_column"))
        definition = definitions.get(factor_column, {})
        metrics = _mapping(record.get("metrics_snapshot"))
        index_records.append(
            {
                "index_schema_version": EXPERIMENT_INDEX_SCHEMA_VERSION,
                "storage_schema_version": document.get("storage_schema_version"),
                "experiment_id": experiment_id,
                "created_at": _text(document.get("created_at")),
                "task_id": _text(document.get("task_id")),
                "experiment_status": _text(summary.get("status")),
                "factor_count": summary.get("factor_count"),
                "factor_manifest_path": _text(document.get("factor_manifest_path")),
                "git_commit": _text(lineage.get("git_commit")),
                "git_is_dirty": lineage.get("git_is_dirty"),
                "config_hash": _text(lineage.get("config_hash")),
                "factor_manifest_hash": _text(lineage.get("factor_manifest_hash")),
                "data_version": _text(lineage.get("data_version")),
                "factor_set_name": _text(data_version_inputs.get("factor_set_name")),
                "universe": _text(data_version_inputs.get("universe")),
                "factor_matrix_path": _text(factor_matrix.get("path")),
                "source_aligned_data_path": _text(source_aligned_data.get("path")),
                "output_dir": _text(request.get("output_dir")),
                "forward_return_days": request.get("forward_return_days"),
                "quantile_count": request.get("quantile_count"),
                "factor_column": factor_column,
                "factor_direction": _text(record.get("factor_direction")),
                "factor_id": _text(definition.get("factor_id")),
                "factor_name": _text(definition.get("name")),
                "factor_source_type": _text(definition.get("source_type")),
                "factor_category": _text(definition.get("category")),
                "factor_formula": _text(definition.get("formula")),
                "factor_hypothesis": _text(definition.get("hypothesis")),
                "status": _text(record.get("status")),
                "stage": _text(record.get("stage")),
                "benchmark_status": _text(record.get("benchmark_status")),
                "critic_verdict": _text(record.get("critic_verdict")),
                "critic_severity": _text(record.get("critic_severity")),
                "mean_ic": metrics.get("mean_ic"),
                "mean_rank_ic": metrics.get("mean_rank_ic"),
                "net_sharpe": metrics.get("net_sharpe"),
                "net_total_return": metrics.get("net_total_return"),
                "max_drawdown_abs": metrics.get("max_drawdown_abs"),
                "average_turnover": metrics.get("average_turnover"),
                "failed_benchmark_tests": _string_list(
                    record.get("failed_benchmark_tests")
                ),
                "error": _text(record.get("error")),
                "result_json_path": _text(record.get("result_json_path")),
                "experiment_result_path": str(result_path),
                "experiment_summary_path": str(summary_path),
            }
        )
    return tuple(index_records)


def _filter_index_records(
    records: Sequence[Mapping[str, Any]],
    spec: ExperimentQuerySpec,
) -> list[dict[str, Any]]:
    start = _parse_date_bound(spec.created_at_start, is_end=False)
    end = _parse_date_bound(spec.created_at_end, is_end=True)
    matched = [
        dict(record)
        for record in records
        if _matches_query(record, spec, created_at_start=start, created_at_end=end)
    ]
    matched.sort(
        key=lambda record: (
            _text(record.get("created_at")),
            _text(record.get("experiment_id")),
            _text(record.get("factor_column")),
        ),
        reverse=spec.sort_desc,
    )
    if spec.limit is not None:
        return matched[: spec.limit]
    return matched


def _matches_query(
    record: Mapping[str, Any],
    spec: ExperimentQuerySpec,
    *,
    created_at_start: datetime | None,
    created_at_end: datetime | None,
) -> bool:
    field_filters = (
        ("experiment_id", spec.experiment_ids),
        ("factor_column", spec.factor_columns),
        ("factor_category", spec.factor_categories),
        ("factor_source_type", spec.factor_source_types),
        ("benchmark_status", spec.benchmark_statuses),
        ("critic_verdict", spec.critic_verdicts),
        ("experiment_status", spec.experiment_statuses),
        ("status", spec.statuses),
        ("factor_set_name", spec.factor_set_names),
        ("universe", spec.universes),
        ("config_hash", spec.config_hashes),
        ("data_version", spec.data_versions),
        ("factor_manifest_hash", spec.factor_manifest_hashes),
        ("factor_manifest_path", spec.factor_manifest_paths),
        ("source_aligned_data_path", spec.source_aligned_data_paths),
    )
    for field_name, allowed_values in field_filters:
        if allowed_values and _text(record.get(field_name)) not in set(allowed_values):
            return False
    return _created_at_in_range(
        record.get("created_at"),
        start=created_at_start,
        end=created_at_end,
    )


def _created_at_in_range(
    value: Any,
    *,
    start: datetime | None,
    end: datetime | None,
) -> bool:
    if start is None and end is None:
        return True
    created_at = _parse_record_datetime(value)
    if created_at is None:
        return False
    if start is not None and created_at < start:
        return False
    if end is not None and created_at > end:
        return False
    return True


def _parse_date_bound(
    value: str | date | datetime | None,
    *,
    is_end: bool,
) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _to_naive_utc(value)
    if isinstance(value, date):
        boundary_time = time.max if is_end else time.min
        return datetime.combine(value, boundary_time)
    if not isinstance(value, str) or not value.strip():
        msg = "Experiment query date bounds must be non-empty strings, dates, or datetimes."
        raise ValueError(msg)
    raw = value.strip()
    if len(raw) == 10:
        parsed_date = date.fromisoformat(raw)
        boundary_time = time.max if is_end else time.min
        return datetime.combine(parsed_date, boundary_time)
    return _to_naive_utc(datetime.fromisoformat(raw.replace("Z", "+00:00")))


def _parse_record_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return _to_naive_utc(datetime.fromisoformat(value.strip().replace("Z", "+00:00")))
    except ValueError:
        return None


def _to_naive_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def _date_bound_to_text(value: str | date | datetime | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return value.isoformat()


def _replace_index_records(
    index_path: Path,
    experiment_id: str,
    index_records: Sequence[Mapping[str, Any]],
) -> None:
    existing = _read_index_records(index_path)
    retained = [
        record
        for record in existing
        if _text(record.get("experiment_id")) != experiment_id
    ]
    temp_index_path = Path(f"{index_path}.tmp")
    with temp_index_path.open("w", encoding="utf-8") as handle:
        for record in (*retained, *index_records):
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            handle.write("\n")
    temp_index_path.replace(index_path)


def _read_index_records(index_path: Path) -> tuple[dict[str, Any], ...]:
    if not index_path.is_file():
        return ()
    records = []
    for line_number, line in enumerate(
        index_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            msg = f"Invalid experiment index JSONL at line {line_number}: {exc}."
            raise ValueError(msg) from exc
        if not isinstance(record, Mapping):
            msg = f"Invalid experiment index record at line {line_number}."
            raise ValueError(msg)
        records.append(dict(record))
    return tuple(records)


def _factor_definitions_by_column(
    document: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    definitions = document.get("factor_definitions")
    if isinstance(definitions, str) or not isinstance(definitions, Sequence):
        return {}
    by_column: dict[str, Mapping[str, Any]] = {}
    for definition in definitions:
        if not isinstance(definition, Mapping):
            continue
        factor_column = definition.get("factor_column")
        if isinstance(factor_column, str) and factor_column.strip():
            by_column[factor_column.strip()] = definition
    return by_column


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


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [str(item) for item in value if str(item).strip()]


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _safe_name(value: str) -> str:
    cleaned = _SAFE_NAME_PATTERN.sub("_", value.strip())
    return cleaned.strip("._") or "experiment"
