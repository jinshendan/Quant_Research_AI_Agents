from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

from agents.backtest_agent import (
    BenchmarkThresholdValue,
    BacktestAgent,
    default_benchmark_thresholds,
)
from agents.critic_agent import CriticAgent
from agents.experiment_store import (
    EXPERIMENT_STORAGE_SCHEMA_VERSION,
    ExperimentStorageResult,
    ExperimentStore,
)
from agents.factor_transforms import DEFAULT_QUANTILE_COUNT, validate_quantile_count
from agents.transaction_costs import TransactionCostSpec
from core.config import AppConfig
from core.i18n import (
    DEFAULT_OUTPUT_LANGUAGE,
    OutputLanguage,
    normalize_output_language,
)
from core.logging import AgentLoggerAdapter, get_agent_logger
from core.models import AgentRequest, AgentResponse

EXPERIMENT_SCHEMA_VERSION = 1
EXPERIMENT_LINEAGE_SCHEMA_VERSION = 1
DEFAULT_EXPERIMENT_OUTPUT_DIR = "experiments"
FactorDirection = Literal["positive", "negative"]
DEFAULT_EXPERIMENT_FACTOR_DIRECTION: FactorDirection = "positive"
DEFAULT_FORWARD_RETURN_DAYS = 1
DEFAULT_ANNUALIZATION_FACTOR = 252
DEFAULT_PREVIEW_ROWS = 0
MAX_PREVIEW_ROWS = 50
MAX_FORWARD_RETURN_DAYS = 60
MIN_ANNUALIZATION_FACTOR = 1
MAX_ANNUALIZATION_FACTOR = 366
SUPPORTED_FACTOR_DIRECTIONS = {"positive", "negative"}
_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """Validated request for one batch factor experiment run."""

    factor_manifest_path: Path
    experiment_id: str = ""
    output_dir: Path = Path(DEFAULT_EXPERIMENT_OUTPUT_DIR)
    factor_columns: tuple[str, ...] = ()
    factor_direction: FactorDirection = DEFAULT_EXPERIMENT_FACTOR_DIRECTION
    factor_directions: Mapping[str, FactorDirection] = field(default_factory=dict)
    forward_return_days: int = DEFAULT_FORWARD_RETURN_DAYS
    quantile_count: int = DEFAULT_QUANTILE_COUNT
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR
    benchmark_thresholds: dict[str, BenchmarkThresholdValue] = field(
        default_factory=default_benchmark_thresholds
    )
    transaction_costs: TransactionCostSpec = field(
        default_factory=TransactionCostSpec,
    )
    continue_on_error: bool = True
    preview_rows: int = DEFAULT_PREVIEW_ROWS
    output_language: OutputLanguage = DEFAULT_OUTPUT_LANGUAGE

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ExperimentSpec:
        return cls(
            factor_manifest_path=_required_path(payload, "factor_manifest_path"),
            experiment_id=_optional_str(payload, "experiment_id", ""),
            output_dir=_optional_path(
                payload,
                "output_dir",
                DEFAULT_EXPERIMENT_OUTPUT_DIR,
            ),
            factor_columns=_optional_str_sequence(payload, "factor_columns"),
            factor_direction=_optional_factor_direction(
                payload,
                "factor_direction",
                DEFAULT_EXPERIMENT_FACTOR_DIRECTION,
            ),
            factor_directions=_optional_factor_directions(payload),
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
            output_language=_optional_output_language(payload),
        )

    @property
    def effective_experiment_id(self) -> str:
        if self.experiment_id.strip():
            return _safe_name(self.experiment_id)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"experiment-{timestamp}"

    def resolved_output_dir(self, project_root: Path) -> Path:
        if self.output_dir.is_absolute():
            return self.output_dir
        return project_root / self.output_dir

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_manifest_path": str(self.factor_manifest_path),
            "experiment_id": self.experiment_id,
            "output_dir": str(self.output_dir),
            "factor_columns": list(self.factor_columns),
            "factor_direction": self.factor_direction,
            "factor_directions": dict(self.factor_directions),
            "forward_return_days": self.forward_return_days,
            "quantile_count": self.quantile_count,
            "annualization_factor": self.annualization_factor,
            "benchmark_thresholds": dict(self.benchmark_thresholds),
            "transaction_costs": self.transaction_costs.to_dict(),
            "continue_on_error": self.continue_on_error,
            "preview_rows": self.preview_rows,
            "output_language": self.output_language,
        }


class ExperimentAgent:
    """Batch backtest factor columns and persist experiment summaries."""

    name = "ExperimentAgent"

    def __init__(
        self,
        *,
        config: AppConfig | None = None,
        logger: AgentLoggerAdapter | None = None,
        store: ExperimentStore | None = None,
        backtest_agent: BacktestAgent | None = None,
        critic_agent: CriticAgent | None = None,
    ) -> None:
        self.config = config or AppConfig.from_env()
        self.logger = logger or get_agent_logger(self.name)
        self.store = store
        self.backtest_agent = backtest_agent or BacktestAgent()
        self.critic_agent = critic_agent or CriticAgent()

    def run(self, request: AgentRequest) -> AgentResponse:
        started_clock = perf_counter()
        started_at = request.timestamp.isoformat()
        self.logger.info(
            "Received experiment request.",
            extra={"action": "validate_request", "status": "running"},
        )

        try:
            spec = ExperimentSpec.from_payload(request.payload)
            manifest = _load_manifest(spec.factor_manifest_path)
            available_factor_columns = _manifest_factor_columns(manifest)
            selected_factor_columns = _selected_factor_columns(
                spec,
                available_factor_columns,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            elapsed = perf_counter() - started_clock
            self.logger.warning(
                "Experiment request validation failed.",
                extra={"action": "validate_request", "status": "error"},
            )
            return AgentResponse.failure(
                str(exc),
                metadata=self._metadata(request, elapsed),
            )

        experiment_id = spec.effective_experiment_id
        store = self.store or ExperimentStore(
            spec.resolved_output_dir(self.config.project_root)
        )
        run_dir = store.run_dir(experiment_id)
        backtest_dir = run_dir / "backtests"
        backtest_dir.mkdir(parents=True, exist_ok=True)

        definitions = _manifest_factor_definitions(manifest)
        records: list[dict[str, Any]] = []
        self.logger.info(
            "Running batch factor experiment.",
            extra={"action": "run_experiment", "status": "running"},
        )
        for factor_column in selected_factor_columns:
            factor_direction = _factor_direction_for(
                factor_column,
                spec,
                definitions,
            )
            result_json_path = backtest_dir / f"{_safe_name(factor_column)}.json"
            record = self._run_one_factor(
                request=request,
                spec=spec,
                factor_column=factor_column,
                factor_direction=factor_direction,
                result_json_path=result_json_path,
            )
            records.append(record)
            if record["status"] == "error" and not spec.continue_on_error:
                break

        elapsed = perf_counter() - started_clock
        summary = _experiment_summary(records)
        request_config = spec.to_dict()
        lineage = _build_experiment_lineage(
            project_root=self.config.project_root,
            request_config=request_config,
            factor_manifest_path=spec.factor_manifest_path,
            manifest=manifest,
        )
        document = {
            "schema_version": EXPERIMENT_SCHEMA_VERSION,
            "storage_schema_version": EXPERIMENT_STORAGE_SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "created_at": started_at,
            "agent": self.name,
            "task_id": request.task_id,
            "request": request_config,
            "lineage": lineage,
            "factor_manifest_path": str(spec.factor_manifest_path),
            "available_factor_columns": list(available_factor_columns),
            "selected_factor_columns": list(selected_factor_columns),
            "factor_definitions": [
                definitions[column]
                for column in selected_factor_columns
                if column in definitions
            ],
            "records": records,
            "summary": summary,
            "elapsed_sec": round(elapsed, 6),
        }
        try:
            storage_result = store.store(document)
        except (OSError, ValueError, TypeError) as exc:
            elapsed = perf_counter() - started_clock
            self.logger.warning(
                "Experiment storage failed.",
                extra={"action": "save_experiment", "status": "error"},
            )
            return AgentResponse.failure(
                str(exc),
                metadata=self._metadata(
                    request,
                    elapsed,
                    experiment_id=experiment_id,
                    factor_count=len(selected_factor_columns),
                ),
            )

        self.logger.info(
            "Completed batch factor experiment.",
            extra={"action": "run_experiment", "status": "success"},
        )
        return AgentResponse.success(
            output={
                "state": "experiment_completed",
                "request": spec.to_dict(),
                "experiment_id": experiment_id,
                "experiment_status": summary["status"],
                "factor_count": summary["factor_count"],
                "successful_factor_count": summary["successful_factor_count"],
                "failed_factor_count": summary["failed_factor_count"],
                "records": records,
                "summary": summary,
                "storage_stats": storage_result.to_dict(),
                "next_action": "Add out-of-sample validation before promoting factors.",
            },
            metadata=self._metadata(
                request,
                perf_counter() - started_clock,
                experiment_id=experiment_id,
                factor_count=summary["factor_count"],
                successful_factor_count=summary["successful_factor_count"],
                failed_factor_count=summary["failed_factor_count"],
                storage_result=storage_result,
            ),
        )

    def _run_one_factor(
        self,
        *,
        request: AgentRequest,
        spec: ExperimentSpec,
        factor_column: str,
        factor_direction: FactorDirection,
        result_json_path: Path,
    ) -> dict[str, Any]:
        backtest_response = self.backtest_agent.run(
            AgentRequest.create(
                {
                    "factor_manifest_path": str(spec.factor_manifest_path),
                    "factor_column": factor_column,
                    "factor_direction": factor_direction,
                    "forward_return_days": spec.forward_return_days,
                    "quantile_count": spec.quantile_count,
                    "annualization_factor": spec.annualization_factor,
                    "benchmark_thresholds": dict(spec.benchmark_thresholds),
                    "transaction_costs": spec.transaction_costs.to_dict(),
                    "preview_rows": spec.preview_rows,
                    "result_json_path": str(result_json_path),
                },
                task_id=f"{request.task_id}-{_safe_name(factor_column)}-backtest",
            )
        )
        if backtest_response.status != "success":
            return _error_record(
                factor_column=factor_column,
                factor_direction=factor_direction,
                stage="backtest",
                error=str(backtest_response.error),
            )

        result_json_path_value = backtest_response.output.get("result_json_path")
        if not isinstance(result_json_path_value, str) or not result_json_path_value:
            return _error_record(
                factor_column=factor_column,
                factor_direction=factor_direction,
                stage="backtest",
                error="BacktestAgent did not return result_json_path.",
            )

        critic_response = self.critic_agent.run(
            AgentRequest.create(
                {
                    "result_json_path": result_json_path_value,
                    "output_language": spec.output_language,
                },
                task_id=f"{request.task_id}-{_safe_name(factor_column)}-critic",
            )
        )
        if critic_response.status != "success":
            return _error_record(
                factor_column=factor_column,
                factor_direction=factor_direction,
                stage="critic",
                error=str(critic_response.error),
                result_json_path=result_json_path_value,
                backtest_output=backtest_response.output,
            )

        return _success_record(
            factor_column=factor_column,
            factor_direction=factor_direction,
            result_json_path=result_json_path_value,
            backtest_output=backtest_response.output,
            critic_output=critic_response.output,
        )

    def _metadata(
        self,
        request: AgentRequest,
        elapsed: float,
        *,
        experiment_id: str | None = None,
        factor_count: int | None = None,
        successful_factor_count: int | None = None,
        failed_factor_count: int | None = None,
        storage_result: ExperimentStorageResult | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "agent": self.name,
            "task_id": request.task_id,
            "execution_time_sec": round(elapsed, 6),
        }
        if experiment_id is not None:
            metadata["experiment_id"] = experiment_id
        if factor_count is not None:
            metadata["factor_count"] = factor_count
        if successful_factor_count is not None:
            metadata["successful_factor_count"] = successful_factor_count
        if failed_factor_count is not None:
            metadata["failed_factor_count"] = failed_factor_count
        if storage_result is not None:
            metadata["storage_stats"] = storage_result.to_dict()
        return metadata


def _success_record(
    *,
    factor_column: str,
    factor_direction: FactorDirection,
    result_json_path: str,
    backtest_output: Mapping[str, Any],
    critic_output: Mapping[str, Any],
) -> dict[str, Any]:
    result_json = _mapping(backtest_output.get("result_json"))
    result_summary = _mapping(result_json.get("summary"))
    drawdown = _mapping(_mapping(result_json.get("metrics")).get("drawdown"))
    benchmark_tests = _mapping(backtest_output.get("benchmark_tests"))
    return {
        "factor_column": factor_column,
        "factor_direction": factor_direction,
        "status": "success",
        "stage": "completed",
        "benchmark_status": _text(backtest_output.get("benchmark_status")),
        "failed_benchmark_tests": _string_list(benchmark_tests.get("failed_tests")),
        "critic_verdict": _text(critic_output.get("verdict")),
        "critic_severity": _text(critic_output.get("severity")),
        "critic_summary": _text(critic_output.get("summary_text")),
        "metrics_snapshot": {
            "usable_row_count": result_summary.get("usable_row_count"),
            "portfolio_date_count": result_summary.get("portfolio_date_count"),
            "mean_ic": result_summary.get("mean_ic"),
            "mean_rank_ic": result_summary.get("mean_rank_ic"),
            "net_sharpe": result_summary.get("net_sharpe"),
            "net_total_return": result_summary.get("net_total_return"),
            "average_turnover": result_summary.get("average_turnover"),
            "max_drawdown_abs": drawdown.get("max_drawdown_abs"),
        },
        "result_json_path": result_json_path,
        "error": None,
    }


def _error_record(
    *,
    factor_column: str,
    factor_direction: FactorDirection,
    stage: str,
    error: str,
    result_json_path: str | None = None,
    backtest_output: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    benchmark_status = None
    failed_tests: list[str] = []
    metrics_snapshot: dict[str, Any] = {}
    if backtest_output is not None:
        benchmark_status = backtest_output.get("benchmark_status")
        benchmark_tests = _mapping(backtest_output.get("benchmark_tests"))
        failed_tests = _string_list(benchmark_tests.get("failed_tests"))
        result_json = _mapping(backtest_output.get("result_json"))
        metrics_snapshot = dict(_mapping(result_json.get("summary")))
    return {
        "factor_column": factor_column,
        "factor_direction": factor_direction,
        "status": "error",
        "stage": stage,
        "benchmark_status": _text(benchmark_status),
        "failed_benchmark_tests": failed_tests,
        "critic_verdict": None,
        "critic_severity": None,
        "critic_summary": None,
        "metrics_snapshot": metrics_snapshot,
        "result_json_path": result_json_path,
        "error": error,
    }


def _experiment_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    factor_count = len(records)
    successful = [record for record in records if record.get("status") == "success"]
    failed = [record for record in records if record.get("status") != "success"]
    verdict_counts = Counter(_text(record.get("critic_verdict")) for record in successful)
    benchmark_counts = Counter(_text(record.get("benchmark_status")) for record in records)
    return {
        "status": (
            "success"
            if not failed
            else "failed"
            if not successful
            else "partial"
        ),
        "factor_count": factor_count,
        "successful_factor_count": len(successful),
        "failed_factor_count": len(failed),
        "benchmark_status_counts": dict(benchmark_counts),
        "critic_verdict_counts": dict(verdict_counts),
        "track_factor_columns": [
            str(record["factor_column"])
            for record in successful
            if record.get("critic_verdict") == "track"
        ],
        "rejected_factor_columns": [
            str(record["factor_column"])
            for record in successful
            if record.get("critic_verdict") == "reject_for_now"
        ],
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        msg = f"Factor manifest file not found: {path}."
        raise OSError(msg)
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        msg = "Factor manifest must contain an object."
        raise ValueError(msg)
    return dict(document)


def _build_experiment_lineage(
    *,
    project_root: Path,
    request_config: Mapping[str, Any],
    factor_manifest_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    storage = _mapping(manifest.get("storage"))
    context = _mapping(manifest.get("context"))
    factor_manifest_hash = _file_sha256(factor_manifest_path)
    data_version_inputs = {
        "factor_manifest_hash": factor_manifest_hash,
        "factor_matrix": _file_identity(storage.get("matrix_path")),
        "source_aligned_data": _file_identity(context.get("source_aligned_data_path")),
        "factor_columns": _string_list(storage.get("factor_columns")),
        "factor_count": storage.get("factor_count"),
        "rows_written": storage.get("rows_written"),
        "manifest_created_at": manifest.get("created_at"),
        "manifest_schema_version": manifest.get("schema_version"),
    }
    return {
        "schema_version": EXPERIMENT_LINEAGE_SCHEMA_VERSION,
        "git_commit": _git_output(project_root, ("rev-parse", "HEAD")),
        "git_is_dirty": _git_is_dirty(project_root),
        "config_hash": _stable_json_hash(request_config),
        "factor_manifest_hash": factor_manifest_hash,
        "data_version": _stable_json_hash(data_version_inputs),
        "data_version_inputs": data_version_inputs,
    }


def _git_is_dirty(project_root: Path) -> bool | None:
    output = _git_output(project_root, ("status", "--short"))
    if output is None:
        return None
    return bool(output.strip())


def _git_output(project_root: Path, args: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(
            ("git", *args),
            cwd=project_root,
            capture_output=True,
            check=False,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    output = result.stdout.strip()
    return output or None


def _file_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        return {"path": None, "exists": False}
    path = Path(value).expanduser()
    identity: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return identity
    stat = path.stat()
    identity["size_bytes"] = stat.st_size
    identity["modified_ns"] = stat.st_mtime_ns
    return identity


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _manifest_factor_columns(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    storage = manifest.get("storage")
    if not isinstance(storage, Mapping):
        msg = "Factor manifest is missing storage."
        raise ValueError(msg)
    value = storage.get("factor_columns")
    if isinstance(value, str) or not isinstance(value, Sequence) or not value:
        msg = "Factor manifest storage.factor_columns must be a non-empty sequence."
        raise ValueError(msg)
    columns = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            msg = "Factor manifest storage.factor_columns must contain strings."
            raise ValueError(msg)
        columns.append(item.strip())
    return tuple(dict.fromkeys(columns))


def _manifest_factor_definitions(
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    context = manifest.get("context")
    if not isinstance(context, Mapping):
        return {}
    definitions = context.get("factor_definitions")
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


def _selected_factor_columns(
    spec: ExperimentSpec,
    available_factor_columns: Sequence[str],
) -> tuple[str, ...]:
    if not spec.factor_columns:
        return tuple(available_factor_columns)
    available = set(available_factor_columns)
    missing = [
        factor_column
        for factor_column in spec.factor_columns
        if factor_column not in available
    ]
    if missing:
        msg = (
            "Requested factor_columns are not in the factor manifest: "
            + ", ".join(missing)
            + "."
        )
        raise ValueError(msg)
    return spec.factor_columns


def _factor_direction_for(
    factor_column: str,
    spec: ExperimentSpec,
    definitions: Mapping[str, Mapping[str, Any]],
) -> FactorDirection:
    explicit = spec.factor_directions.get(factor_column)
    if explicit is not None:
        return explicit
    definition = definitions.get(factor_column, {})
    direction = definition.get("direction")
    if direction in SUPPORTED_FACTOR_DIRECTIONS:
        return cast(FactorDirection, direction)
    return spec.factor_direction


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


def _optional_str(payload: Mapping[str, Any], key: str, default: str) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str):
        msg = f"payload.{key} must be a string."
        raise ValueError(msg)
    return value.strip()


def _optional_str_sequence(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key, [])
    if isinstance(value, str) or not isinstance(value, Sequence):
        msg = f"payload.{key} must be a sequence of strings."
        raise ValueError(msg)
    values = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            msg = f"payload.{key} must contain only non-empty strings."
            raise ValueError(msg)
        values.append(item.strip())
    return tuple(dict.fromkeys(values))


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


def _optional_factor_direction(
    payload: Mapping[str, Any],
    key: str,
    default: FactorDirection,
) -> FactorDirection:
    value = payload.get(key, default)
    if not isinstance(value, str) or not value.strip():
        msg = f"payload.{key} must be a non-empty string."
        raise ValueError(msg)
    normalized = value.strip()
    if normalized not in SUPPORTED_FACTOR_DIRECTIONS:
        msg = f"payload.{key} must be either positive or negative."
        raise ValueError(msg)
    return cast(FactorDirection, normalized)


def _optional_factor_directions(
    payload: Mapping[str, Any],
) -> dict[str, FactorDirection]:
    value = payload.get("factor_directions", {})
    if not isinstance(value, Mapping):
        msg = "payload.factor_directions must be an object."
        raise ValueError(msg)
    normalized: dict[str, FactorDirection] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            msg = "payload.factor_directions keys must be non-empty strings."
            raise ValueError(msg)
        if not isinstance(item, str) or item not in SUPPORTED_FACTOR_DIRECTIONS:
            msg = "payload.factor_directions values must be positive or negative."
            raise ValueError(msg)
        normalized[key.strip()] = cast(FactorDirection, item)
    return normalized


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
    for key, item in value.items():
        if not isinstance(key, str):
            msg = "payload.benchmark_thresholds keys must be strings."
            raise ValueError(msg)
        if item is not None and (
            isinstance(item, bool) or not isinstance(item, int | float)
        ):
            msg = f"payload.benchmark_thresholds.{key} must be a number or null."
            raise ValueError(msg)
        thresholds[key] = item
    return thresholds


def _optional_mapping_alias(
    payload: Mapping[str, Any],
    keys: Sequence[str],
) -> dict[str, Any] | None:
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
        msg = f"payload.{found_key} must be an object."
        raise ValueError(msg)
    return dict(found_value)


def _optional_output_language(payload: Mapping[str, Any]) -> OutputLanguage:
    value = payload.get("output_language")
    if value is None:
        return DEFAULT_OUTPUT_LANGUAGE
    if not isinstance(value, str):
        msg = "payload.output_language must be a string."
        raise ValueError(msg)
    return normalize_output_language(value)


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
    return cleaned.strip("._") or "experiment"
