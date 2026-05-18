from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from agents.factor_transforms import (
    DEFAULT_QUANTILE_COUNT,
    validate_quantile_count,
)
from core.logging import AgentLoggerAdapter, get_agent_logger
from core.models import AgentRequest, AgentResponse

FactorDirection = Literal["positive", "negative"]

IDENTITY_COLUMNS = ("date", "symbol")
CLOSE_COLUMN = "close"
FORWARD_RETURN_COLUMN = "forward_return"
PORTFOLIO_COLUMNS = (
    "date",
    "long_return",
    "short_return",
    "long_short_return",
    "long_count",
    "short_count",
)
DEFAULT_FORWARD_RETURN_DAYS = 1
DEFAULT_PREVIEW_ROWS = 5
MAX_PREVIEW_ROWS = 50
MAX_FORWARD_RETURN_DAYS = 60
SUPPORTED_FACTOR_DIRECTIONS = {"positive", "negative"}


@dataclass(frozen=True, slots=True)
class BacktestSpec:
    """Validated request for building a factor backtest return series."""

    factor_matrix_path: Path | None = None
    factor_manifest_path: Path | None = None
    aligned_data_path: Path | None = None
    factor_column: str | None = None
    factor_direction: FactorDirection = "positive"
    forward_return_days: int = DEFAULT_FORWARD_RETURN_DAYS
    quantile_count: int = DEFAULT_QUANTILE_COUNT
    preview_rows: int = DEFAULT_PREVIEW_ROWS

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> BacktestSpec:
        factor_matrix_path = _optional_path(payload, "factor_matrix_path")
        factor_manifest_path = _optional_path(payload, "factor_manifest_path")
        aligned_data_path = _optional_path(payload, "aligned_data_path")
        if factor_matrix_path is None and factor_manifest_path is None:
            msg = "payload.factor_matrix_path or payload.factor_manifest_path is required."
            raise ValueError(msg)

        factor_column = _optional_str(payload, "factor_column")
        factor_direction = _optional_factor_direction(payload)
        forward_return_days = _optional_int(
            payload,
            "forward_return_days",
            DEFAULT_FORWARD_RETURN_DAYS,
            minimum=1,
            maximum=MAX_FORWARD_RETURN_DAYS,
        )
        quantile_count = validate_quantile_count(
            _optional_int(
                payload,
                "quantile_count",
                DEFAULT_QUANTILE_COUNT,
                minimum=2,
                maximum=20,
            )
        )
        preview_rows = _optional_int(
            payload,
            "preview_rows",
            DEFAULT_PREVIEW_ROWS,
            minimum=0,
            maximum=MAX_PREVIEW_ROWS,
        )
        return cls(
            factor_matrix_path=factor_matrix_path,
            factor_manifest_path=factor_manifest_path,
            aligned_data_path=aligned_data_path,
            factor_column=factor_column,
            factor_direction=factor_direction,
            forward_return_days=forward_return_days,
            quantile_count=quantile_count,
            preview_rows=preview_rows,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_matrix_path": (
                str(self.factor_matrix_path) if self.factor_matrix_path else None
            ),
            "factor_manifest_path": (
                str(self.factor_manifest_path) if self.factor_manifest_path else None
            ),
            "aligned_data_path": (
                str(self.aligned_data_path) if self.aligned_data_path else None
            ),
            "factor_column": self.factor_column,
            "factor_direction": self.factor_direction,
            "forward_return_days": self.forward_return_days,
            "quantile_count": self.quantile_count,
            "preview_rows": self.preview_rows,
        }


@dataclass(frozen=True, slots=True)
class PortfolioBuildResult:
    """Long/short return series and construction statistics."""

    data: pd.DataFrame
    stats: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BacktestBuildResult:
    """Backtest-ready panel and derived portfolio return series."""

    panel: pd.DataFrame
    portfolio_returns: pd.DataFrame
    factor_column: str
    factor_matrix_path: Path
    aligned_data_path: Path
    stats: dict[str, Any]


class BacktestAgent:
    """Build a simple factor long/short backtest return series."""

    name = "BacktestAgent"

    def __init__(self, *, logger: AgentLoggerAdapter | None = None) -> None:
        self.logger = logger or get_agent_logger(self.name)

    def run(self, request: AgentRequest) -> AgentResponse:
        started_at = perf_counter()
        self.logger.info(
            "Received backtest request.",
            extra={"action": "validate_request", "status": "running"},
        )

        try:
            spec = BacktestSpec.from_payload(request.payload)
            manifest = self.load_factor_manifest(spec.factor_manifest_path)
            factor_matrix_path = _resolve_factor_matrix_path(spec, manifest)
            aligned_data_path = _resolve_aligned_data_path(spec, manifest)
            factor_matrix = self.load_factor_matrix(factor_matrix_path)
            factor_column = _select_factor_column(factor_matrix, spec, manifest)
            aligned_data = self.load_aligned_data(aligned_data_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            elapsed = perf_counter() - started_at
            self.logger.warning(
                "Backtest request validation failed.",
                extra={"action": "validate_request", "status": "error"},
            )
            return AgentResponse.failure(
                str(exc),
                metadata=self._metadata(request, elapsed),
            )

        self.logger.info(
            "Building factor backtest series.",
            extra={"action": "build_backtest", "status": "running"},
        )
        try:
            result = self.build_backtest(
                factor_matrix=factor_matrix,
                aligned_data=aligned_data,
                factor_column=factor_column,
                factor_matrix_path=factor_matrix_path,
                aligned_data_path=aligned_data_path,
                spec=spec,
            )
        except ValueError as exc:
            elapsed = perf_counter() - started_at
            self.logger.warning(
                "Backtest construction failed.",
                extra={"action": "build_backtest", "status": "error"},
            )
            return AgentResponse.failure(
                str(exc),
                metadata=self._metadata(
                    request,
                    elapsed,
                    factor_matrix_path=factor_matrix_path,
                    aligned_data_path=aligned_data_path,
                    factor_column=factor_column,
                ),
            )

        elapsed = perf_counter() - started_at
        self.logger.info(
            "Built factor backtest series.",
            extra={"action": "build_backtest", "status": "success"},
        )
        return AgentResponse.success(
            output={
                "state": "backtest_built",
                "request": spec.to_dict(),
                "factor_matrix_path": str(result.factor_matrix_path),
                "aligned_data_path": str(result.aligned_data_path),
                "factor_column": result.factor_column,
                "factor_direction": spec.factor_direction,
                "forward_return_days": spec.forward_return_days,
                "quantile_count": spec.quantile_count,
                "portfolio_return_columns": list(PORTFOLIO_COLUMNS),
                "row_count": len(result.panel),
                "usable_row_count": result.stats["usable_row_count"],
                "portfolio_date_count": len(result.portfolio_returns),
                "preview": _preview_records(
                    result.portfolio_returns,
                    spec.preview_rows,
                ),
                "backtest_stats": result.stats,
                "next_action": "Compute IC in Day 16.",
            },
            metadata=self._metadata(
                request,
                elapsed,
                factor_matrix_path=result.factor_matrix_path,
                aligned_data_path=result.aligned_data_path,
                factor_column=result.factor_column,
                portfolio_date_count=len(result.portfolio_returns),
                usable_row_count=result.stats["usable_row_count"],
            ),
        )

    def load_factor_manifest(
        self,
        factor_manifest_path: Path | None,
    ) -> dict[str, Any]:
        if factor_manifest_path is None:
            return {}
        if not factor_manifest_path.is_file():
            msg = f"Factor manifest file not found: {factor_manifest_path}."
            raise OSError(msg)
        document = json.loads(factor_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            msg = "Factor manifest must be a JSON object."
            raise ValueError(msg)
        return document

    def load_factor_matrix(self, factor_matrix_path: Path) -> pd.DataFrame:
        if not factor_matrix_path.is_file():
            msg = f"Factor matrix file not found: {factor_matrix_path}."
            raise OSError(msg)
        return normalize_factor_matrix(pd.read_csv(factor_matrix_path, dtype={"symbol": str}))

    def load_aligned_data(self, aligned_data_path: Path) -> pd.DataFrame:
        if not aligned_data_path.is_file():
            msg = f"Aligned data file not found: {aligned_data_path}."
            raise OSError(msg)
        return normalize_aligned_prices(pd.read_csv(aligned_data_path, dtype={"symbol": str}))

    def build_backtest(
        self,
        *,
        factor_matrix: pd.DataFrame,
        aligned_data: pd.DataFrame,
        factor_column: str,
        factor_matrix_path: Path,
        aligned_data_path: Path,
        spec: BacktestSpec,
    ) -> BacktestBuildResult:
        if factor_column not in factor_matrix.columns:
            msg = f"Factor matrix is missing factor column: {factor_column}."
            raise ValueError(msg)

        returns = _forward_returns(aligned_data, spec.forward_return_days)
        panel = factor_matrix[list(IDENTITY_COLUMNS) + [factor_column]].merge(
            returns,
            on=list(IDENTITY_COLUMNS),
            how="left",
        )
        panel[factor_column] = pd.to_numeric(panel[factor_column], errors="coerce")
        panel[FORWARD_RETURN_COLUMN] = pd.to_numeric(
            panel[FORWARD_RETURN_COLUMN],
            errors="coerce",
        )

        portfolio_result = _build_portfolio_returns(panel, factor_column, spec)
        if portfolio_result.data.empty:
            msg = (
                "No portfolio return dates were produced. Check factor coverage, "
                "forward returns, and quantile_count."
            )
            raise ValueError(msg)

        stats = _backtest_stats(panel, factor_column, portfolio_result)
        return BacktestBuildResult(
            panel=panel,
            portfolio_returns=portfolio_result.data,
            factor_column=factor_column,
            factor_matrix_path=factor_matrix_path,
            aligned_data_path=aligned_data_path,
            stats=stats,
        )

    def _metadata(
        self,
        request: AgentRequest,
        elapsed: float,
        *,
        factor_matrix_path: Path | None = None,
        aligned_data_path: Path | None = None,
        factor_column: str | None = None,
        portfolio_date_count: int | None = None,
        usable_row_count: int | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "agent": self.name,
            "task_id": request.task_id,
            "execution_time_sec": round(elapsed, 6),
        }
        if factor_matrix_path is not None:
            metadata["factor_matrix_path"] = str(factor_matrix_path)
        if aligned_data_path is not None:
            metadata["aligned_data_path"] = str(aligned_data_path)
        if factor_column is not None:
            metadata["factor_column"] = factor_column
        if portfolio_date_count is not None:
            metadata["portfolio_date_count"] = portfolio_date_count
        if usable_row_count is not None:
            metadata["usable_row_count"] = usable_row_count
        return metadata


def normalize_factor_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in IDENTITY_COLUMNS if column not in frame.columns]
    if missing_columns:
        msg = f"Factor matrix is missing required columns: {', '.join(missing_columns)}."
        raise ValueError(msg)

    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["symbol"] = normalized["symbol"].astype(str).str.zfill(6)
    if normalized["date"].isna().any():
        msg = "Factor matrix contains invalid dates."
        raise ValueError(msg)

    for column in normalized.columns:
        if column not in IDENTITY_COLUMNS:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized.sort_values(["symbol", "date"]).reset_index(drop=True)


def normalize_aligned_prices(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = (*IDENTITY_COLUMNS, CLOSE_COLUMN)
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        msg = f"Aligned data is missing required columns: {', '.join(missing_columns)}."
        raise ValueError(msg)

    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["symbol"] = normalized["symbol"].astype(str).str.zfill(6)
    normalized[CLOSE_COLUMN] = pd.to_numeric(normalized[CLOSE_COLUMN], errors="coerce")
    if normalized["date"].isna().any():
        msg = "Aligned data contains invalid dates."
        raise ValueError(msg)
    return normalized.sort_values(["symbol", "date"]).reset_index(drop=True)


def _resolve_factor_matrix_path(
    spec: BacktestSpec,
    manifest: Mapping[str, Any],
) -> Path:
    if spec.factor_matrix_path is not None:
        return spec.factor_matrix_path
    path = _manifest_path(manifest, "storage", "matrix_path")
    if path is None:
        msg = "factor_matrix_path is required when the manifest has no storage.matrix_path."
        raise ValueError(msg)
    return path


def _resolve_aligned_data_path(
    spec: BacktestSpec,
    manifest: Mapping[str, Any],
) -> Path:
    if spec.aligned_data_path is not None:
        return spec.aligned_data_path
    path = _manifest_path(manifest, "context", "source_aligned_data_path")
    if path is None:
        msg = "aligned_data_path is required when the manifest has no source_aligned_data_path."
        raise ValueError(msg)
    return path


def _manifest_path(
    manifest: Mapping[str, Any],
    section: str,
    key: str,
) -> Path | None:
    raw_section = manifest.get(section)
    if not isinstance(raw_section, Mapping):
        return None
    value = raw_section.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value.strip()).expanduser().resolve()


def _select_factor_column(
    factor_matrix: pd.DataFrame,
    spec: BacktestSpec,
    manifest: Mapping[str, Any],
) -> str:
    if spec.factor_column is not None:
        if spec.factor_column not in factor_matrix.columns:
            msg = f"Factor matrix is missing factor column: {spec.factor_column}."
            raise ValueError(msg)
        return spec.factor_column

    candidates = _manifest_factor_columns(manifest)
    if not candidates:
        candidates = tuple(
            column for column in factor_matrix.columns if column not in IDENTITY_COLUMNS
        )
    existing_candidates = tuple(column for column in candidates if column in factor_matrix.columns)
    if len(existing_candidates) == 1:
        return existing_candidates[0]
    if not existing_candidates:
        msg = "No factor columns were found in the factor matrix."
        raise ValueError(msg)

    msg = "payload.factor_column is required when the factor matrix has multiple factor columns."
    raise ValueError(msg)


def _manifest_factor_columns(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    storage = manifest.get("storage")
    if not isinstance(storage, Mapping):
        return ()
    factor_columns = storage.get("factor_columns")
    if not isinstance(factor_columns, Sequence) or isinstance(factor_columns, str):
        return ()
    columns = []
    for column in factor_columns:
        if isinstance(column, str) and column.strip():
            columns.append(column.strip())
    return tuple(dict.fromkeys(columns))


def _forward_returns(aligned_data: pd.DataFrame, forward_return_days: int) -> pd.DataFrame:
    frame = aligned_data[list(IDENTITY_COLUMNS) + [CLOSE_COLUMN]].copy()
    future_close = frame.groupby("symbol", sort=False)[CLOSE_COLUMN].shift(
        -forward_return_days
    )
    base_close = frame[CLOSE_COLUMN].replace(0, np.nan)
    frame[FORWARD_RETURN_COLUMN] = (future_close / base_close) - 1.0
    frame[FORWARD_RETURN_COLUMN] = frame[FORWARD_RETURN_COLUMN].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return frame[list(IDENTITY_COLUMNS) + [FORWARD_RETURN_COLUMN]]


def _build_portfolio_returns(
    panel: pd.DataFrame,
    factor_column: str,
    spec: BacktestSpec,
) -> PortfolioBuildResult:
    rows: list[dict[str, Any]] = []
    skipped_date_count = 0
    sort_ascending = spec.factor_direction == "negative"

    for trade_date, group in panel.groupby("date", sort=True):
        usable = group.dropna(subset=[factor_column, FORWARD_RETURN_COLUMN]).copy()
        if len(usable) < spec.quantile_count:
            skipped_date_count += 1
            continue

        selection_count = max(1, len(usable) // spec.quantile_count)
        sorted_group = usable.sort_values(
            factor_column,
            ascending=sort_ascending,
            kind="mergesort",
        )
        long_leg = sorted_group.head(selection_count)
        short_leg = sorted_group.tail(selection_count)
        long_return = float(long_leg[FORWARD_RETURN_COLUMN].mean())
        short_return = float(short_leg[FORWARD_RETURN_COLUMN].mean())
        rows.append(
            {
                "date": trade_date,
                "long_return": long_return,
                "short_return": short_return,
                "long_short_return": long_return - short_return,
                "long_count": int(len(long_leg)),
                "short_count": int(len(short_leg)),
            }
        )

    portfolio_returns = pd.DataFrame(rows, columns=PORTFOLIO_COLUMNS)
    if not portfolio_returns.empty:
        portfolio_returns = portfolio_returns.sort_values("date").reset_index(drop=True)
    return PortfolioBuildResult(
        data=portfolio_returns,
        stats={"skipped_date_count": skipped_date_count},
    )


def _backtest_stats(
    panel: pd.DataFrame,
    factor_column: str,
    portfolio_result: PortfolioBuildResult,
) -> dict[str, Any]:
    valid_factor = panel[factor_column].notna()
    valid_forward_return = panel[FORWARD_RETURN_COLUMN].notna()
    usable = valid_factor & valid_forward_return

    portfolio_returns = portfolio_result.data
    if portfolio_returns.empty:
        average_long_count = 0.0
        average_short_count = 0.0
    else:
        average_long_count = float(portfolio_returns["long_count"].mean())
        average_short_count = float(portfolio_returns["short_count"].mean())

    return {
        "factor_column": factor_column,
        "input_row_count": len(panel),
        "valid_factor_row_count": int(valid_factor.sum()),
        "valid_forward_return_row_count": int(valid_forward_return.sum()),
        "usable_row_count": int(usable.sum()),
        "portfolio_date_count": len(portfolio_returns),
        "skipped_date_count": portfolio_result.stats["skipped_date_count"],
        "average_long_count": average_long_count,
        "average_short_count": average_short_count,
    }


def _preview_records(frame: pd.DataFrame, preview_rows: int) -> list[dict[str, Any]]:
    if preview_rows == 0:
        return []
    records: list[dict[str, Any]] = []
    for row in frame.head(preview_rows).to_dict(orient="records"):
        records.append({str(key): _json_safe_value(value) for key, value in row.items()})
    return records


def _json_safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _optional_path(payload: Mapping[str, Any], key: str) -> Path | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        msg = f"payload.{key} must be a non-empty path string when provided."
        raise ValueError(msg)
    return Path(value.strip()).expanduser().resolve()


def _optional_str(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        msg = f"payload.{key} must be a non-empty string when provided."
        raise ValueError(msg)
    return value.strip()


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
