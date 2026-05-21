from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from agents.backtest_agent import (
    BenchmarkThresholdValue,
    BacktestAgent,
    default_benchmark_thresholds,
)
from agents.factor_transforms import DEFAULT_QUANTILE_COUNT, validate_quantile_count
from agents.transaction_costs import TransactionCostSpec
from core.config import AppConfig
from core.logging import AgentLoggerAdapter, get_agent_logger
from core.models import AgentRequest, AgentResponse

OUT_OF_SAMPLE_SCHEMA_VERSION = 1
DEFAULT_VALIDATION_OUTPUT_DIR = "validations"
DEFAULT_FORWARD_RETURN_DAYS = 1
DEFAULT_ANNUALIZATION_FACTOR = 252
DEFAULT_PREVIEW_ROWS = 0
MAX_PREVIEW_ROWS = 50
MAX_FORWARD_RETURN_DAYS = 60
MIN_ANNUALIZATION_FACTOR = 1
MAX_ANNUALIZATION_FACTOR = 366
SUPPORTED_FACTOR_DIRECTIONS = {"positive", "negative"}
FactorDirection = Literal["positive", "negative"]
_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class ValidationSplitSpec:
    """One signal-date window for out-of-sample validation."""

    name: str
    start_date: str
    end_date: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ValidationSplitSpec:
        name = _required_str(value, "name")
        start_date = _required_date(value, "start_date")
        end_date = _required_date(value, "end_date")
        if date.fromisoformat(start_date) > date.fromisoformat(end_date):
            msg = "split.start_date must be on or before split.end_date."
            raise ValueError(msg)
        return cls(name=name, start_date=start_date, end_date=end_date)

    @property
    def safe_name(self) -> str:
        return _safe_name(self.name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start_date": self.start_date,
            "end_date": self.end_date,
        }


@dataclass(frozen=True, slots=True)
class OutOfSampleSpec:
    """Validated request for train/validation/test factor validation."""

    factor_manifest_path: Path
    factor_column: str
    validation_id: str = ""
    output_dir: Path = Path(DEFAULT_VALIDATION_OUTPUT_DIR)
    splits: tuple[ValidationSplitSpec, ...] = ()
    factor_direction: FactorDirection = "positive"
    forward_return_days: int = DEFAULT_FORWARD_RETURN_DAYS
    quantile_count: int = DEFAULT_QUANTILE_COUNT
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR
    benchmark_thresholds: dict[str, BenchmarkThresholdValue] = field(
        default_factory=default_benchmark_thresholds,
    )
    transaction_costs: TransactionCostSpec = field(
        default_factory=TransactionCostSpec,
    )
    continue_on_error: bool = True
    preview_rows: int = DEFAULT_PREVIEW_ROWS

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> OutOfSampleSpec:
        splits = _required_splits(payload)
        _validate_unique_splits(splits)
        return cls(
            factor_manifest_path=_required_path(payload, "factor_manifest_path"),
            factor_column=_required_str(payload, "factor_column"),
            validation_id=_optional_str(payload, "validation_id", ""),
            output_dir=_optional_path(
                payload,
                "output_dir",
                DEFAULT_VALIDATION_OUTPUT_DIR,
            ),
            splits=splits,
            factor_direction=_optional_factor_direction(payload),
            forward_return_days=_optional_int(
                payload,
                "forward_return_days",
                DEFAULT_FORWARD_RETURN_DAYS,
                minimum=1,
                maximum=MAX_FORWARD_RETURN_DAYS,
            ),
            quantile_count=validate_quantile_count(
                _optional_int(
                    payload,
                    "quantile_count",
                    DEFAULT_QUANTILE_COUNT,
                    minimum=2,
                    maximum=20,
                )
            ),
            annualization_factor=_optional_int(
                payload,
                "annualization_factor",
                DEFAULT_ANNUALIZATION_FACTOR,
                minimum=MIN_ANNUALIZATION_FACTOR,
                maximum=MAX_ANNUALIZATION_FACTOR,
            ),
            benchmark_thresholds=_optional_benchmark_thresholds(payload),
            transaction_costs=TransactionCostSpec.from_mapping(
                _optional_mapping_alias(payload, ("transaction_costs", "cost_profile")),
            ),
            continue_on_error=_optional_bool(payload, "continue_on_error", True),
            preview_rows=_optional_int(
                payload,
                "preview_rows",
                DEFAULT_PREVIEW_ROWS,
                minimum=0,
                maximum=MAX_PREVIEW_ROWS,
            ),
        )

    @property
    def effective_validation_id(self) -> str:
        if self.validation_id.strip():
            return _safe_name(self.validation_id)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"validation-{timestamp}"

    def resolved_output_dir(self, project_root: Path) -> Path:
        if self.output_dir.is_absolute():
            return self.output_dir
        return project_root / self.output_dir

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_manifest_path": str(self.factor_manifest_path),
            "factor_column": self.factor_column,
            "validation_id": self.validation_id,
            "output_dir": str(self.output_dir),
            "splits": [split.to_dict() for split in self.splits],
            "factor_direction": self.factor_direction,
            "forward_return_days": self.forward_return_days,
            "quantile_count": self.quantile_count,
            "annualization_factor": self.annualization_factor,
            "benchmark_thresholds": dict(self.benchmark_thresholds),
            "transaction_costs": self.transaction_costs.to_dict(),
            "continue_on_error": self.continue_on_error,
            "preview_rows": self.preview_rows,
        }


@dataclass(frozen=True, slots=True)
class OutOfSampleStorageResult:
    """Paths written by an out-of-sample validation run."""

    result_path: Path
    summary_path: Path
    backtest_dir: Path
    records_written: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_path": str(self.result_path),
            "summary_path": str(self.summary_path),
            "backtest_dir": str(self.backtest_dir),
            "records_written": self.records_written,
        }


class OutOfSampleAgent:
    """Run one factor through explicit train/validation/test windows."""

    name = "OutOfSampleAgent"

    def __init__(
        self,
        *,
        config: AppConfig | None = None,
        logger: AgentLoggerAdapter | None = None,
        backtest_agent: BacktestAgent | None = None,
    ) -> None:
        self.config = config or AppConfig.from_env()
        self.logger = logger or get_agent_logger(self.name)
        self.backtest_agent = backtest_agent or BacktestAgent()

    def run(self, request: AgentRequest) -> AgentResponse:
        started_clock = perf_counter()
        started_at = request.timestamp.isoformat()
        self.logger.info(
            "Received out-of-sample validation request.",
            extra={"action": "validate_request", "status": "running"},
        )

        try:
            spec = OutOfSampleSpec.from_payload(request.payload)
            validation_id = spec.effective_validation_id
            run_dir = spec.resolved_output_dir(self.config.project_root) / validation_id
            backtest_dir = run_dir / "backtests"
            backtest_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError) as exc:
            elapsed = perf_counter() - started_clock
            self.logger.warning(
                "Out-of-sample request validation failed.",
                extra={"action": "validate_request", "status": "error"},
            )
            return AgentResponse.failure(
                str(exc),
                metadata=self._metadata(request, elapsed),
            )

        records: list[dict[str, Any]] = []
        self.logger.info(
            "Running split backtests.",
            extra={"action": "run_split_backtests", "status": "running"},
        )
        for split in spec.splits:
            result_json_path = backtest_dir / f"{split.safe_name}.json"
            record = self._run_one_split(
                request=request,
                spec=spec,
                split=split,
                result_json_path=result_json_path,
            )
            records.append(record)
            if record["status"] == "error" and not spec.continue_on_error:
                break

        elapsed = perf_counter() - started_clock
        summary = _validation_summary(records)
        document = {
            "schema_version": OUT_OF_SAMPLE_SCHEMA_VERSION,
            "validation_id": validation_id,
            "created_at": started_at,
            "agent": self.name,
            "task_id": request.task_id,
            "request": spec.to_dict(),
            "factor_manifest_path": str(spec.factor_manifest_path),
            "factor_column": spec.factor_column,
            "factor_direction": spec.factor_direction,
            "records": records,
            "summary": summary,
            "elapsed_sec": round(elapsed, 6),
        }

        try:
            storage_result = save_out_of_sample_result(
                document,
                run_dir=run_dir,
                records=records,
            )
        except (OSError, TypeError, ValueError) as exc:
            elapsed = perf_counter() - started_clock
            self.logger.warning(
                "Out-of-sample storage failed.",
                extra={"action": "save_validation", "status": "error"},
            )
            return AgentResponse.failure(
                str(exc),
                metadata=self._metadata(
                    request,
                    elapsed,
                    validation_id=validation_id,
                    split_count=len(records),
                ),
            )

        self.logger.info(
            "Completed out-of-sample validation.",
            extra={"action": "run_split_backtests", "status": "success"},
        )
        return AgentResponse.success(
            output={
                "state": "out_of_sample_validated",
                "request": spec.to_dict(),
                "validation_id": validation_id,
                "validation_status": summary["status"],
                "factor_column": spec.factor_column,
                "factor_direction": spec.factor_direction,
                "records": records,
                "summary": summary,
                "storage_stats": storage_result.to_dict(),
                "next_action": "Add walk-forward validation and factor decay tests.",
            },
            metadata=self._metadata(
                request,
                perf_counter() - started_clock,
                validation_id=validation_id,
                split_count=summary["split_count"],
                successful_split_count=summary["successful_split_count"],
                failed_split_count=summary["failed_split_count"],
                storage_result=storage_result,
            ),
        )

    def _run_one_split(
        self,
        *,
        request: AgentRequest,
        spec: OutOfSampleSpec,
        split: ValidationSplitSpec,
        result_json_path: Path,
    ) -> dict[str, Any]:
        backtest_response = self.backtest_agent.run(
            AgentRequest.create(
                {
                    "factor_manifest_path": str(spec.factor_manifest_path),
                    "factor_column": spec.factor_column,
                    "factor_direction": spec.factor_direction,
                    "start_date": split.start_date,
                    "end_date": split.end_date,
                    "forward_return_days": spec.forward_return_days,
                    "quantile_count": spec.quantile_count,
                    "annualization_factor": spec.annualization_factor,
                    "benchmark_thresholds": dict(spec.benchmark_thresholds),
                    "transaction_costs": spec.transaction_costs.to_dict(),
                    "preview_rows": spec.preview_rows,
                    "result_json_path": str(result_json_path),
                },
                task_id=(
                    f"{request.task_id}-{_safe_name(spec.factor_column)}-"
                    f"{split.safe_name}-backtest"
                ),
            )
        )
        if backtest_response.status != "success":
            return _error_record(
                split=split,
                factor_column=spec.factor_column,
                factor_direction=spec.factor_direction,
                error=str(backtest_response.error),
            )
        return _success_record(
            split=split,
            factor_column=spec.factor_column,
            factor_direction=spec.factor_direction,
            backtest_output=backtest_response.output,
        )

    def _metadata(
        self,
        request: AgentRequest,
        elapsed: float,
        *,
        validation_id: str | None = None,
        split_count: int | None = None,
        successful_split_count: int | None = None,
        failed_split_count: int | None = None,
        storage_result: OutOfSampleStorageResult | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "agent": self.name,
            "task_id": request.task_id,
            "execution_time_sec": round(elapsed, 6),
        }
        if validation_id is not None:
            metadata["validation_id"] = validation_id
        if split_count is not None:
            metadata["split_count"] = split_count
        if successful_split_count is not None:
            metadata["successful_split_count"] = successful_split_count
        if failed_split_count is not None:
            metadata["failed_split_count"] = failed_split_count
        if storage_result is not None:
            metadata["storage_stats"] = storage_result.to_dict()
        return metadata


def save_out_of_sample_result(
    document: Mapping[str, Any],
    *,
    run_dir: Path,
    records: Sequence[Mapping[str, Any]],
) -> OutOfSampleStorageResult:
    """Persist out-of-sample validation JSON and CSV summary artifacts."""

    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "out_of_sample_result.json"
    summary_path = run_dir / "out_of_sample_summary.csv"
    _atomic_json_write(result_path, document)
    pd.DataFrame(_summary_rows(records)).to_csv(summary_path, index=False)
    return OutOfSampleStorageResult(
        result_path=result_path,
        summary_path=summary_path,
        backtest_dir=run_dir / "backtests",
        records_written=len(records),
    )


def _success_record(
    *,
    split: ValidationSplitSpec,
    factor_column: str,
    factor_direction: FactorDirection,
    backtest_output: Mapping[str, Any],
) -> dict[str, Any]:
    result_json = _mapping(backtest_output.get("result_json"))
    summary = _mapping(result_json.get("summary"))
    metrics = _mapping(result_json.get("metrics"))
    drawdown = _mapping(metrics.get("drawdown"))
    benchmark_tests = _mapping(backtest_output.get("benchmark_tests"))
    return {
        "split_name": split.name,
        "start_date": split.start_date,
        "end_date": split.end_date,
        "factor_column": factor_column,
        "factor_direction": factor_direction,
        "status": "success",
        "stage": "completed",
        "benchmark_status": _text(backtest_output.get("benchmark_status")),
        "failed_benchmark_tests": _string_list(benchmark_tests.get("failed_tests")),
        "metrics_snapshot": {
            "row_count": summary.get("row_count"),
            "usable_row_count": summary.get("usable_row_count"),
            "portfolio_date_count": summary.get("portfolio_date_count"),
            "ic_date_count": summary.get("ic_date_count"),
            "rank_ic_date_count": summary.get("rank_ic_date_count"),
            "mean_ic": summary.get("mean_ic"),
            "mean_rank_ic": summary.get("mean_rank_ic"),
            "net_sharpe": summary.get("net_sharpe"),
            "gross_sharpe": summary.get("gross_sharpe"),
            "net_total_return": summary.get("net_total_return"),
            "gross_total_return": summary.get("gross_total_return"),
            "average_turnover": summary.get("average_turnover"),
            "total_transaction_cost": summary.get("total_transaction_cost"),
            "max_drawdown_abs": drawdown.get("max_drawdown_abs"),
        },
        "result_json_path": _text(backtest_output.get("result_json_path")),
        "error": None,
    }


def _error_record(
    *,
    split: ValidationSplitSpec,
    factor_column: str,
    factor_direction: FactorDirection,
    error: str,
) -> dict[str, Any]:
    return {
        "split_name": split.name,
        "start_date": split.start_date,
        "end_date": split.end_date,
        "factor_column": factor_column,
        "factor_direction": factor_direction,
        "status": "error",
        "stage": "backtest",
        "benchmark_status": None,
        "failed_benchmark_tests": [],
        "metrics_snapshot": {},
        "result_json_path": None,
        "error": error,
    }


def _validation_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    split_count = len(records)
    successful = [record for record in records if record.get("status") == "success"]
    failed = [record for record in records if record.get("status") != "success"]
    benchmark_counts = Counter(_text(record.get("benchmark_status")) for record in records)
    return {
        "status": (
            "success"
            if not failed
            else "failed"
            if not successful
            else "partial"
        ),
        "split_count": split_count,
        "successful_split_count": len(successful),
        "failed_split_count": len(failed),
        "failed_split_names": [
            str(record["split_name"])
            for record in failed
            if isinstance(record.get("split_name"), str)
        ],
        "benchmark_status_counts": dict(benchmark_counts),
        "basic_oos_check": _basic_oos_check(successful, failed),
    }


def _basic_oos_check(
    successful: Sequence[Mapping[str, Any]],
    failed: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if failed:
        return {
            "status": "failed",
            "reason": "one_or_more_splits_failed",
            "train_split_name": None,
            "out_of_sample_split_names": [],
            "train_mean_rank_ic": None,
            "out_of_sample_mean_rank_ic_min": None,
            "rank_ic_direction_consistent": None,
            "out_of_sample_benchmarks_passed": False,
        }
    train_record = _train_record(successful)
    if train_record is None:
        return _not_enough_oos_data("missing_train_split")
    oos_records = [record for record in successful if record is not train_record]
    if not oos_records:
        return _not_enough_oos_data("missing_out_of_sample_splits")

    train_mean_rank_ic = _metric(train_record, "mean_rank_ic")
    oos_rank_ics = [_metric(record, "mean_rank_ic") for record in oos_records]
    if train_mean_rank_ic is None or any(value is None for value in oos_rank_ics):
        return _not_enough_oos_data("missing_rank_ic")

    train_sign = _sign(train_mean_rank_ic)
    oos_signs = [_sign(cast(float, value)) for value in oos_rank_ics]
    direction_consistent = train_sign != 0 and all(sign == train_sign for sign in oos_signs)
    oos_benchmarks_passed = all(
        record.get("benchmark_status") == "passed" for record in oos_records
    )
    status = "passed" if direction_consistent and oos_benchmarks_passed else "failed"
    return {
        "status": status,
        "reason": (
            "rank_ic_direction_and_benchmarks_passed"
            if status == "passed"
            else "rank_ic_direction_or_benchmark_failed"
        ),
        "train_split_name": train_record.get("split_name"),
        "out_of_sample_split_names": [record.get("split_name") for record in oos_records],
        "train_mean_rank_ic": train_mean_rank_ic,
        "out_of_sample_mean_rank_ic_min": min(cast(list[float], oos_rank_ics)),
        "rank_ic_direction_consistent": direction_consistent,
        "out_of_sample_benchmarks_passed": oos_benchmarks_passed,
    }


def _not_enough_oos_data(reason: str) -> dict[str, Any]:
    return {
        "status": "not_enough_data",
        "reason": reason,
        "train_split_name": None,
        "out_of_sample_split_names": [],
        "train_mean_rank_ic": None,
        "out_of_sample_mean_rank_ic_min": None,
        "rank_ic_direction_consistent": None,
        "out_of_sample_benchmarks_passed": False,
    }


def _train_record(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for record in records:
        split_name = record.get("split_name")
        if isinstance(split_name, str) and split_name.lower() == "train":
            return record
    return records[0] if records else None


def _metric(record: Mapping[str, Any], key: str) -> float | None:
    metrics = _mapping(record.get("metrics_snapshot"))
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if not np.isfinite(value):
        return None
    return float(value)


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _summary_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        metrics = _mapping(record.get("metrics_snapshot"))
        rows.append(
            {
                "split_name": record.get("split_name"),
                "start_date": record.get("start_date"),
                "end_date": record.get("end_date"),
                "factor_column": record.get("factor_column"),
                "factor_direction": record.get("factor_direction"),
                "status": record.get("status"),
                "benchmark_status": record.get("benchmark_status"),
                "failed_benchmark_tests": ",".join(
                    _string_list(record.get("failed_benchmark_tests"))
                ),
                "row_count": metrics.get("row_count"),
                "usable_row_count": metrics.get("usable_row_count"),
                "portfolio_date_count": metrics.get("portfolio_date_count"),
                "ic_date_count": metrics.get("ic_date_count"),
                "rank_ic_date_count": metrics.get("rank_ic_date_count"),
                "mean_ic": metrics.get("mean_ic"),
                "mean_rank_ic": metrics.get("mean_rank_ic"),
                "net_sharpe": metrics.get("net_sharpe"),
                "net_total_return": metrics.get("net_total_return"),
                "average_turnover": metrics.get("average_turnover"),
                "total_transaction_cost": metrics.get("total_transaction_cost"),
                "max_drawdown_abs": metrics.get("max_drawdown_abs"),
                "result_json_path": record.get("result_json_path"),
                "error": record.get("error"),
            }
        )
    return rows


def _atomic_json_write(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(f"{path}.tmp")
    temp_path.write_text(
        json.dumps(
            document,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _required_path(payload: Mapping[str, Any], key: str) -> Path:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        msg = f"payload.{key} must be a non-empty path string."
        raise ValueError(msg)
    return Path(value.strip()).expanduser().resolve()


def _optional_path(payload: Mapping[str, Any], key: str, default: str) -> Path:
    value = payload.get(key, default)
    if not isinstance(value, str) or not value.strip():
        msg = f"payload.{key} must be a non-empty path string."
        raise ValueError(msg)
    return Path(value.strip()).expanduser()


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        msg = f"payload.{key} must be a non-empty string."
        raise ValueError(msg)
    return value.strip()


def _optional_str(payload: Mapping[str, Any], key: str, default: str) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str):
        msg = f"payload.{key} must be a string."
        raise ValueError(msg)
    return value.strip()


def _required_date(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        msg = f"payload.{key} must be a non-empty ISO date string."
        raise ValueError(msg)
    try:
        return date.fromisoformat(value.strip()).isoformat()
    except ValueError as exc:
        msg = f"payload.{key} must be a valid ISO date string."
        raise ValueError(msg) from exc


def _optional_int(
    payload: Mapping[str, Any],
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"payload.{key} must be an integer."
        raise ValueError(msg)
    if value < minimum or value > maximum:
        msg = f"payload.{key} must be between {minimum} and {maximum}."
        raise ValueError(msg)
    return value


def _optional_bool(payload: Mapping[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        msg = f"payload.{key} must be a boolean."
        raise ValueError(msg)
    return value


def _optional_factor_direction(payload: Mapping[str, Any]) -> FactorDirection:
    value = payload.get("factor_direction", "positive")
    if not isinstance(value, str) or not value.strip():
        msg = "payload.factor_direction must be a non-empty string."
        raise ValueError(msg)
    normalized = value.strip().lower()
    if normalized not in SUPPORTED_FACTOR_DIRECTIONS:
        msg = "payload.factor_direction must be either positive or negative."
        raise ValueError(msg)
    return cast(FactorDirection, normalized)


def _optional_benchmark_thresholds(
    payload: Mapping[str, Any],
) -> dict[str, BenchmarkThresholdValue]:
    thresholds = default_benchmark_thresholds()
    value = payload.get("benchmark_thresholds")
    if value is None:
        return thresholds
    if not isinstance(value, Mapping):
        msg = "payload.benchmark_thresholds must be an object."
        raise ValueError(msg)
    unknown_keys = sorted(set(value) - set(thresholds))
    if unknown_keys:
        msg = "payload.benchmark_thresholds has unsupported keys: "
        msg += ", ".join(str(key) for key in unknown_keys)
        raise ValueError(msg)
    for key, item in value.items():
        if item is not None and (
            isinstance(item, bool) or not isinstance(item, int | float)
        ):
            msg = f"payload.benchmark_thresholds.{key} must be a number or null."
            raise ValueError(msg)
        if item is not None and not np.isfinite(item):
            msg = f"payload.benchmark_thresholds.{key} must be finite or null."
            raise ValueError(msg)
        thresholds[str(key)] = item
    return thresholds


def _optional_mapping_alias(
    payload: Mapping[str, Any],
    keys: Sequence[str],
) -> Mapping[str, Any] | None:
    found_key: str | None = None
    found_value: Any = None
    for key in keys:
        if key in payload:
            if found_key is not None:
                msg = f"Provide only one of payload.{found_key} or payload.{key}."
                raise ValueError(msg)
            found_key = key
            found_value = payload[key]
    if found_key is None or found_value is None:
        return None
    if not isinstance(found_value, Mapping):
        msg = f"payload.{found_key} must be an object when provided."
        raise ValueError(msg)
    return found_value


def _required_splits(payload: Mapping[str, Any]) -> tuple[ValidationSplitSpec, ...]:
    value = payload.get("splits")
    if isinstance(value, str) or not isinstance(value, Sequence) or not value:
        msg = "payload.splits must be a non-empty sequence of split objects."
        raise ValueError(msg)
    splits = []
    for item in value:
        if not isinstance(item, Mapping):
            msg = "payload.splits must contain split objects."
            raise ValueError(msg)
        splits.append(ValidationSplitSpec.from_mapping(item))
    return tuple(splits)


def _validate_unique_splits(splits: Sequence[ValidationSplitSpec]) -> None:
    names = [split.name for split in splits]
    if len(set(names)) != len(names):
        msg = "payload.splits names must be unique."
        raise ValueError(msg)
    safe_names = [split.safe_name for split in splits]
    if len(set(safe_names)) != len(safe_names):
        msg = "payload.splits safe file names must be unique."
        raise ValueError(msg)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [str(item) for item in value if str(item).strip()]


def _text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _safe_name(value: str) -> str:
    cleaned = _SAFE_NAME_PATTERN.sub("_", value.strip())
    return cleaned.strip("._") or "validation"
