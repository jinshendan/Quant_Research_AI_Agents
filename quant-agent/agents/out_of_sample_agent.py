from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
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
SplitRole = Literal["train", "validation", "test", "custom"]
ValidationMethod = Literal["explicit_splits", "walk_forward"]
SUPPORTED_SPLIT_ROLES = {"train", "validation", "test", "custom"}
_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
COMPARISON_METRICS: tuple[tuple[str, str, bool], ...] = (
    ("mean_ic", "IC", True),
    ("mean_rank_ic", "RankIC", True),
    ("net_sharpe", "Net Sharpe", True),
    ("max_drawdown_abs", "Max Drawdown Abs", False),
    ("average_turnover", "Average Turnover", False),
    ("net_total_return", "Net Total Return", True),
)


@dataclass(frozen=True, slots=True)
class ValidationSplitSpec:
    """One signal-date window for out-of-sample validation."""

    name: str
    start_date: str
    end_date: str
    role: SplitRole = "custom"
    fold_index: int | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ValidationSplitSpec:
        name = _required_str(value, "name")
        start_date = _required_date(value, "start_date")
        end_date = _required_date(value, "end_date")
        role = _optional_split_role(value, default=_infer_split_role(name))
        fold_index = _optional_positive_int(value, "fold_index")
        if date.fromisoformat(start_date) > date.fromisoformat(end_date):
            msg = "split.start_date must be on or before split.end_date."
            raise ValueError(msg)
        return cls(
            name=name,
            start_date=start_date,
            end_date=end_date,
            role=role,
            fold_index=fold_index,
        )

    @property
    def safe_name(self) -> str:
        return _safe_name(self.name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "role": self.role,
            "fold_index": self.fold_index,
        }


@dataclass(frozen=True, slots=True)
class WalkForwardSpec:
    """Generate rolling train/test signal-date windows."""

    start_date: str
    end_date: str
    train_window_days: int
    test_window_days: int
    step_days: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> WalkForwardSpec:
        start_date = _required_date(value, "start_date")
        end_date = _required_date(value, "end_date")
        spec = cls(
            start_date=start_date,
            end_date=end_date,
            train_window_days=_required_positive_int(value, "train_window_days"),
            test_window_days=_required_positive_int(value, "test_window_days"),
            step_days=_required_positive_int(value, "step_days"),
        )
        if date.fromisoformat(spec.start_date) > date.fromisoformat(spec.end_date):
            msg = "walk_forward.start_date must be on or before walk_forward.end_date."
            raise ValueError(msg)
        if not spec.to_splits():
            msg = "walk_forward does not produce any train/test folds."
            raise ValueError(msg)
        return spec

    def to_splits(self) -> tuple[ValidationSplitSpec, ...]:
        splits: list[ValidationSplitSpec] = []
        start = date.fromisoformat(self.start_date)
        final = date.fromisoformat(self.end_date)
        cursor = start
        fold_index = 1
        while True:
            train_start = cursor
            train_end = train_start + timedelta(days=self.train_window_days - 1)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=self.test_window_days - 1)
            if test_end > final:
                break
            splits.extend(
                [
                    ValidationSplitSpec(
                        name=f"walk_{fold_index:03d}_train",
                        start_date=train_start.isoformat(),
                        end_date=train_end.isoformat(),
                        role="train",
                        fold_index=fold_index,
                    ),
                    ValidationSplitSpec(
                        name=f"walk_{fold_index:03d}_test",
                        start_date=test_start.isoformat(),
                        end_date=test_end.isoformat(),
                        role="test",
                        fold_index=fold_index,
                    ),
                ]
            )
            cursor = cursor + timedelta(days=self.step_days)
            fold_index += 1
        return tuple(splits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "train_window_days": self.train_window_days,
            "test_window_days": self.test_window_days,
            "step_days": self.step_days,
        }


@dataclass(frozen=True, slots=True)
class OutOfSampleSpec:
    """Validated request for train/validation/test factor validation."""

    factor_manifest_path: Path
    factor_column: str
    validation_id: str = ""
    output_dir: Path = Path(DEFAULT_VALIDATION_OUTPUT_DIR)
    splits: tuple[ValidationSplitSpec, ...] = ()
    validation_method: ValidationMethod = "explicit_splits"
    walk_forward: WalkForwardSpec | None = None
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
        explicit_splits = _optional_splits(payload)
        walk_forward = _optional_walk_forward(payload)
        if explicit_splits and walk_forward is not None:
            msg = "Provide only one of payload.splits or payload.walk_forward."
            raise ValueError(msg)
        if not explicit_splits and walk_forward is None:
            msg = "payload.splits or payload.walk_forward is required."
            raise ValueError(msg)
        splits = explicit_splits or cast(WalkForwardSpec, walk_forward).to_splits()
        _validate_unique_splits(splits)
        validation_method: ValidationMethod = (
            "walk_forward" if walk_forward is not None else "explicit_splits"
        )
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
            validation_method=validation_method,
            walk_forward=walk_forward,
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
            "validation_method": self.validation_method,
            "walk_forward": (
                self.walk_forward.to_dict() if self.walk_forward is not None else None
            ),
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
                "next_action": (
                    "Add out-of-sample pass/fail report markers and factor decay tests."
                ),
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
        "split_role": split.role,
        "fold_index": split.fold_index,
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
        "split_role": split.role,
        "fold_index": split.fold_index,
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
        "walk_forward_check": _walk_forward_check(records),
        "metric_comparison": _metric_comparison(successful, failed),
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
    oos_records = _out_of_sample_records(successful, train_record)
    if not oos_records:
        return _not_enough_oos_data("missing_out_of_sample_splits")

    train_mean_rank_ic = _metric(train_record, "mean_rank_ic")
    oos_rank_ics = [_metric(record, "mean_rank_ic") for record in oos_records]
    if train_mean_rank_ic is None or any(value is None for value in oos_rank_ics):
        return _not_enough_oos_data("missing_rank_ic")

    oos_rank_ic_values = [cast(float, value) for value in oos_rank_ics]
    train_sign = _sign(train_mean_rank_ic)
    oos_signs = [_sign(value) for value in oos_rank_ic_values]
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
        "out_of_sample_mean_rank_ic_min": min(oos_rank_ic_values),
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
        if record.get("split_role") == "train":
            return record
    for record in records:
        split_name = record.get("split_name")
        if isinstance(split_name, str) and split_name.lower() == "train":
            return record
    return records[0] if records else None


def _training_records(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    role_based = [record for record in records if record.get("split_role") == "train"]
    if role_based:
        return role_based
    train_record = _train_record(records)
    return [train_record] if train_record is not None else []


def _out_of_sample_records(
    records: Sequence[Mapping[str, Any]],
    train_record: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    role_based = [
        record
        for record in records
        if record is not train_record and record.get("split_role") != "train"
    ]
    if role_based:
        return role_based
    return [record for record in records if record is not train_record]


def _comparison_out_of_sample_records(
    records: Sequence[Mapping[str, Any]],
    training_records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    role_based = [
        record
        for record in records
        if record.get("split_role") != "train"
        and not _contains_same_record(training_records, record)
    ]
    if role_based:
        return role_based
    return [
        record
        for record in records
        if not _contains_same_record(training_records, record)
    ]


def _contains_same_record(
    records: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
) -> bool:
    return any(record is candidate for record in records)


def _metric_comparison(
    successful: Sequence[Mapping[str, Any]],
    failed: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    training_records = _training_records(successful)
    out_of_sample_records = _comparison_out_of_sample_records(
        successful,
        training_records,
    )
    if not training_records or not out_of_sample_records:
        return {
            "status": "not_enough_data",
            "reason": (
                "missing_train_split"
                if not training_records
                else "missing_out_of_sample_splits"
            ),
            "in_sample_split_names": _split_names(training_records),
            "out_of_sample_split_names": _split_names(out_of_sample_records),
            "failed_split_names": _split_names(failed),
            "metrics": {},
        }

    metrics = {
        metric_key: _compare_metric(
            metric_key,
            label=label,
            higher_is_better=higher_is_better,
            in_sample_records=training_records,
            out_of_sample_records=out_of_sample_records,
        )
        for metric_key, label, higher_is_better in COMPARISON_METRICS
    }
    available_count = sum(1 for metric in metrics.values() if metric["status"] == "available")
    if available_count == 0:
        status = "not_enough_data"
    elif failed:
        status = "partial"
    else:
        status = "available"
    return {
        "status": status,
        "reason": (
            "metric_comparison_available"
            if status == "available"
            else "one_or_more_splits_failed"
            if status == "partial"
            else "missing_comparable_metrics"
        ),
        "in_sample_split_names": _split_names(training_records),
        "out_of_sample_split_names": _split_names(out_of_sample_records),
        "failed_split_names": _split_names(failed),
        "metric_count": len(metrics),
        "available_metric_count": available_count,
        "metrics": metrics,
    }


def _compare_metric(
    metric_key: str,
    *,
    label: str,
    higher_is_better: bool,
    in_sample_records: Sequence[Mapping[str, Any]],
    out_of_sample_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    in_sample_values = _metric_values(in_sample_records, metric_key)
    out_of_sample_values = _metric_values(out_of_sample_records, metric_key)
    if not in_sample_values or not out_of_sample_values:
        return {
            "status": "missing",
            "label": label,
            "higher_is_better": higher_is_better,
            "in_sample_mean": None,
            "out_of_sample_mean": None,
            "delta": None,
            "retention_ratio": None,
            "out_of_sample_better_or_equal": None,
            "in_sample_values": in_sample_values,
            "out_of_sample_values": out_of_sample_values,
        }

    in_sample_mean = _mean(in_sample_values)
    out_of_sample_mean = _mean(out_of_sample_values)
    delta = out_of_sample_mean - in_sample_mean
    retention_ratio = (
        out_of_sample_mean / in_sample_mean
        if abs(in_sample_mean) > 1e-12
        else None
    )
    better_or_equal = (
        out_of_sample_mean >= in_sample_mean
        if higher_is_better
        else out_of_sample_mean <= in_sample_mean
    )
    return {
        "status": "available",
        "label": label,
        "higher_is_better": higher_is_better,
        "in_sample_mean": in_sample_mean,
        "out_of_sample_mean": out_of_sample_mean,
        "delta": delta,
        "retention_ratio": retention_ratio,
        "out_of_sample_better_or_equal": better_or_equal,
        "in_sample_values": in_sample_values,
        "out_of_sample_values": out_of_sample_values,
    }


def _metric_values(
    records: Sequence[Mapping[str, Any]],
    metric_key: str,
) -> list[float]:
    values = []
    for record in records:
        value = _metric(record, metric_key)
        if value is not None:
            values.append(value)
    return values


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values))


def _split_names(records: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        str(record["split_name"])
        for record in records
        if isinstance(record.get("split_name"), str)
    ]


def _walk_forward_check(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for record in records:
        fold_index = record.get("fold_index")
        if isinstance(fold_index, bool) or not isinstance(fold_index, int):
            continue
        grouped.setdefault(fold_index, []).append(record)

    if not grouped:
        return {
            "status": "not_applicable",
            "fold_count": 0,
            "passed_fold_count": 0,
            "failed_fold_count": 0,
            "rank_ic_direction_consistent_count": 0,
            "test_benchmark_passed_count": 0,
            "test_mean_rank_ic_min": None,
            "folds": [],
        }

    folds = [
        _walk_forward_fold_summary(fold_index, grouped[fold_index])
        for fold_index in sorted(grouped)
    ]
    passed_fold_count = sum(1 for fold in folds if fold["status"] == "passed")
    test_rank_ics = [
        fold["test_mean_rank_ic"]
        for fold in folds
        if isinstance(fold.get("test_mean_rank_ic"), int | float)
    ]
    return {
        "status": "passed" if passed_fold_count == len(folds) else "failed",
        "fold_count": len(folds),
        "passed_fold_count": passed_fold_count,
        "failed_fold_count": len(folds) - passed_fold_count,
        "rank_ic_direction_consistent_count": sum(
            1 for fold in folds if fold["rank_ic_direction_consistent"] is True
        ),
        "test_benchmark_passed_count": sum(
            1 for fold in folds if fold["test_benchmark_passed"] is True
        ),
        "test_mean_rank_ic_min": min(test_rank_ics) if test_rank_ics else None,
        "folds": folds,
    }


def _walk_forward_fold_summary(
    fold_index: int,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    train_record = _role_record(records, "train")
    test_record = _role_record(records, "test")
    base: dict[str, Any] = {
        "fold_index": fold_index,
        "train_split_name": train_record.get("split_name") if train_record else None,
        "test_split_name": test_record.get("split_name") if test_record else None,
        "train_status": train_record.get("status") if train_record else None,
        "test_status": test_record.get("status") if test_record else None,
        "train_benchmark_status": (
            train_record.get("benchmark_status") if train_record else None
        ),
        "test_benchmark_status": test_record.get("benchmark_status") if test_record else None,
        "train_mean_rank_ic": _metric(train_record, "mean_rank_ic")
        if train_record
        else None,
        "test_mean_rank_ic": _metric(test_record, "mean_rank_ic") if test_record else None,
        "train_net_sharpe": _metric(train_record, "net_sharpe") if train_record else None,
        "test_net_sharpe": _metric(test_record, "net_sharpe") if test_record else None,
        "train_net_total_return": _metric(train_record, "net_total_return")
        if train_record
        else None,
        "test_net_total_return": _metric(test_record, "net_total_return")
        if test_record
        else None,
        "test_max_drawdown_abs": _metric(test_record, "max_drawdown_abs")
        if test_record
        else None,
        "rank_ic_direction_consistent": None,
        "test_benchmark_passed": False,
    }
    if train_record is None or test_record is None:
        return {
            **base,
            "status": "failed",
            "reason": "missing_train_or_test_split",
        }
    if train_record.get("status") != "success" or test_record.get("status") != "success":
        return {
            **base,
            "status": "failed",
            "reason": "train_or_test_split_failed",
        }

    train_mean_rank_ic = cast(float | None, base["train_mean_rank_ic"])
    test_mean_rank_ic = cast(float | None, base["test_mean_rank_ic"])
    if train_mean_rank_ic is None or test_mean_rank_ic is None:
        return {
            **base,
            "status": "not_enough_data",
            "reason": "missing_rank_ic",
        }

    train_sign = _sign(train_mean_rank_ic)
    test_sign = _sign(test_mean_rank_ic)
    direction_consistent = train_sign != 0 and test_sign == train_sign
    test_benchmark_passed = test_record.get("benchmark_status") == "passed"
    status = "passed" if direction_consistent and test_benchmark_passed else "failed"
    return {
        **base,
        "status": status,
        "reason": (
            "rank_ic_direction_and_test_benchmark_passed"
            if status == "passed"
            else "rank_ic_direction_or_test_benchmark_failed"
        ),
        "rank_ic_direction_consistent": direction_consistent,
        "test_benchmark_passed": test_benchmark_passed,
    }


def _role_record(
    records: Sequence[Mapping[str, Any]],
    role: SplitRole,
) -> Mapping[str, Any] | None:
    for record in records:
        if record.get("split_role") == role:
            return record
    return None


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
                "split_role": record.get("split_role"),
                "fold_index": record.get("fold_index"),
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


def _required_positive_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"payload.{key} must be a positive integer."
        raise ValueError(msg)
    if value <= 0:
        msg = f"payload.{key} must be greater than zero."
        raise ValueError(msg)
    return value


def _optional_positive_int(payload: Mapping[str, Any], key: str) -> int | None:
    if key not in payload or payload[key] is None:
        return None
    return _required_positive_int(payload, key)


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


def _optional_split_role(payload: Mapping[str, Any], *, default: SplitRole) -> SplitRole:
    value = payload.get("role", default)
    if not isinstance(value, str) or not value.strip():
        msg = "payload.role must be a non-empty string."
        raise ValueError(msg)
    normalized = value.strip().lower()
    if normalized not in SUPPORTED_SPLIT_ROLES:
        msg = "payload.role must be one of train, validation, test, or custom."
        raise ValueError(msg)
    return cast(SplitRole, normalized)


def _infer_split_role(name: str) -> SplitRole:
    normalized = name.strip().lower()
    for role in ("train", "validation", "test"):
        if normalized == role:
            return cast(SplitRole, role)
        if normalized.endswith(f"_{role}") or normalized.endswith(f"-{role}"):
            return cast(SplitRole, role)
        if normalized.endswith(f".{role}"):
            return cast(SplitRole, role)
    return "custom"


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


def _optional_splits(payload: Mapping[str, Any]) -> tuple[ValidationSplitSpec, ...]:
    if "splits" not in payload or payload["splits"] is None:
        return ()
    value = payload["splits"]
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


def _optional_walk_forward(payload: Mapping[str, Any]) -> WalkForwardSpec | None:
    if "walk_forward" not in payload or payload["walk_forward"] is None:
        return None
    value = payload["walk_forward"]
    if not isinstance(value, Mapping):
        msg = "payload.walk_forward must be an object."
        raise ValueError(msg)
    return WalkForwardSpec.from_mapping(value)


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
