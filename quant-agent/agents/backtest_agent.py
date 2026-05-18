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
IC_COLUMNS = ("date", "ic", "raw_ic", "pair_count")
RANK_IC_COLUMNS = ("date", "rank_ic", "raw_rank_ic", "pair_count")
SHARPE_RETURN_COLUMN = "long_short_return"
DRAWDOWN_RETURN_COLUMN = SHARPE_RETURN_COLUMN
DRAWDOWN_COLUMNS = ("date", "equity_curve", "cumulative_peak", "drawdown")
DEFAULT_ANNUALIZATION_FACTOR = 252
DEFAULT_FORWARD_RETURN_DAYS = 1
DEFAULT_PREVIEW_ROWS = 5
MAX_PREVIEW_ROWS = 50
MAX_FORWARD_RETURN_DAYS = 60
MIN_ANNUALIZATION_FACTOR = 1
MAX_ANNUALIZATION_FACTOR = 366
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
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR
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
        annualization_factor = _optional_int(
            payload,
            "annualization_factor",
            DEFAULT_ANNUALIZATION_FACTOR,
            minimum=MIN_ANNUALIZATION_FACTOR,
            maximum=MAX_ANNUALIZATION_FACTOR,
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
            annualization_factor=annualization_factor,
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
            "annualization_factor": self.annualization_factor,
            "preview_rows": self.preview_rows,
        }


@dataclass(frozen=True, slots=True)
class PortfolioBuildResult:
    """Long/short return series and construction statistics."""

    data: pd.DataFrame
    stats: dict[str, Any]


@dataclass(frozen=True, slots=True)
class InformationCoefficientResult:
    """Cross-sectional Pearson IC series and summary statistics."""

    data: pd.DataFrame
    stats: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RankInformationCoefficientResult:
    """Cross-sectional Spearman RankIC series and summary statistics."""

    data: pd.DataFrame
    stats: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SharpeResult:
    """Sharpe ratio and return-distribution statistics."""

    stats: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DrawdownResult:
    """Drawdown curve and summary statistics."""

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
        self.logger.info(
            "Computing information coefficient.",
            extra={"action": "compute_ic", "status": "running"},
        )
        try:
            ic_result = compute_information_coefficient(
                result.panel,
                factor_column=result.factor_column,
                factor_direction=spec.factor_direction,
            )
        except ValueError as exc:
            elapsed = perf_counter() - started_at
            self.logger.warning(
                "Information coefficient calculation failed.",
                extra={"action": "compute_ic", "status": "error"},
            )
            return AgentResponse.failure(
                str(exc),
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
        elapsed = perf_counter() - started_at
        self.logger.info(
            "Computed information coefficient.",
            extra={"action": "compute_ic", "status": "success"},
        )
        self.logger.info(
            "Computing rank information coefficient.",
            extra={"action": "compute_rank_ic", "status": "running"},
        )
        try:
            rank_ic_result = compute_rank_information_coefficient(
                result.panel,
                factor_column=result.factor_column,
                factor_direction=spec.factor_direction,
            )
        except ValueError as exc:
            elapsed = perf_counter() - started_at
            self.logger.warning(
                "Rank information coefficient calculation failed.",
                extra={"action": "compute_rank_ic", "status": "error"},
            )
            return AgentResponse.failure(
                str(exc),
                metadata=self._metadata(
                    request,
                    elapsed,
                    factor_matrix_path=result.factor_matrix_path,
                    aligned_data_path=result.aligned_data_path,
                    factor_column=result.factor_column,
                    portfolio_date_count=len(result.portfolio_returns),
                    usable_row_count=result.stats["usable_row_count"],
                    ic_date_count=ic_result.stats["ic_date_count"],
                    mean_ic=ic_result.stats["mean_ic"],
                ),
            )
        elapsed = perf_counter() - started_at
        self.logger.info(
            "Computed rank information coefficient.",
            extra={"action": "compute_rank_ic", "status": "success"},
        )
        self.logger.info(
            "Computing Sharpe ratio.",
            extra={"action": "compute_sharpe", "status": "running"},
        )
        try:
            sharpe_result = compute_sharpe_ratio(
                result.portfolio_returns,
                annualization_factor=spec.annualization_factor,
            )
        except ValueError as exc:
            elapsed = perf_counter() - started_at
            self.logger.warning(
                "Sharpe ratio calculation failed.",
                extra={"action": "compute_sharpe", "status": "error"},
            )
            return AgentResponse.failure(
                str(exc),
                metadata=self._metadata(
                    request,
                    elapsed,
                    factor_matrix_path=result.factor_matrix_path,
                    aligned_data_path=result.aligned_data_path,
                    factor_column=result.factor_column,
                    portfolio_date_count=len(result.portfolio_returns),
                    usable_row_count=result.stats["usable_row_count"],
                    ic_date_count=ic_result.stats["ic_date_count"],
                    mean_ic=ic_result.stats["mean_ic"],
                    rank_ic_date_count=rank_ic_result.stats["rank_ic_date_count"],
                    mean_rank_ic=rank_ic_result.stats["mean_rank_ic"],
                ),
            )
        elapsed = perf_counter() - started_at
        self.logger.info(
            "Computed Sharpe ratio.",
            extra={"action": "compute_sharpe", "status": "success"},
        )
        self.logger.info(
            "Computing drawdown.",
            extra={"action": "compute_drawdown", "status": "running"},
        )
        try:
            drawdown_result = compute_drawdown(result.portfolio_returns)
        except ValueError as exc:
            elapsed = perf_counter() - started_at
            self.logger.warning(
                "Drawdown calculation failed.",
                extra={"action": "compute_drawdown", "status": "error"},
            )
            return AgentResponse.failure(
                str(exc),
                metadata=self._metadata(
                    request,
                    elapsed,
                    factor_matrix_path=result.factor_matrix_path,
                    aligned_data_path=result.aligned_data_path,
                    factor_column=result.factor_column,
                    portfolio_date_count=len(result.portfolio_returns),
                    usable_row_count=result.stats["usable_row_count"],
                    ic_date_count=ic_result.stats["ic_date_count"],
                    mean_ic=ic_result.stats["mean_ic"],
                    rank_ic_date_count=rank_ic_result.stats["rank_ic_date_count"],
                    mean_rank_ic=rank_ic_result.stats["mean_rank_ic"],
                    sharpe=sharpe_result.stats["sharpe"],
                ),
            )
        elapsed = perf_counter() - started_at
        self.logger.info(
            "Computed drawdown.",
            extra={"action": "compute_drawdown", "status": "success"},
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
                "annualization_factor": spec.annualization_factor,
                "portfolio_return_columns": list(PORTFOLIO_COLUMNS),
                "ic_series_columns": list(IC_COLUMNS),
                "rank_ic_series_columns": list(RANK_IC_COLUMNS),
                "drawdown_curve_columns": list(DRAWDOWN_COLUMNS),
                "row_count": len(result.panel),
                "usable_row_count": result.stats["usable_row_count"],
                "portfolio_date_count": len(result.portfolio_returns),
                "ic_date_count": ic_result.stats["ic_date_count"],
                "rank_ic_date_count": rank_ic_result.stats["rank_ic_date_count"],
                "preview": _preview_records(
                    result.portfolio_returns,
                    spec.preview_rows,
                ),
                "ic_series_preview": _preview_records(
                    ic_result.data,
                    spec.preview_rows,
                ),
                "rank_ic_series_preview": _preview_records(
                    rank_ic_result.data,
                    spec.preview_rows,
                ),
                "drawdown_curve_preview": _preview_records(
                    drawdown_result.data,
                    spec.preview_rows,
                ),
                "backtest_stats": result.stats,
                "ic_stats": ic_result.stats,
                "rank_ic_stats": rank_ic_result.stats,
                "sharpe_stats": sharpe_result.stats,
                "drawdown_stats": drawdown_result.stats,
                "next_action": "Generate result JSON in Day 20.",
            },
            metadata=self._metadata(
                request,
                elapsed,
                factor_matrix_path=result.factor_matrix_path,
                aligned_data_path=result.aligned_data_path,
                factor_column=result.factor_column,
                portfolio_date_count=len(result.portfolio_returns),
                usable_row_count=result.stats["usable_row_count"],
                ic_date_count=ic_result.stats["ic_date_count"],
                mean_ic=ic_result.stats["mean_ic"],
                rank_ic_date_count=rank_ic_result.stats["rank_ic_date_count"],
                mean_rank_ic=rank_ic_result.stats["mean_rank_ic"],
                sharpe=sharpe_result.stats["sharpe"],
                max_drawdown=drawdown_result.stats["max_drawdown"],
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
        ic_date_count: int | None = None,
        mean_ic: float | None = None,
        rank_ic_date_count: int | None = None,
        mean_rank_ic: float | None = None,
        sharpe: float | None = None,
        max_drawdown: float | None = None,
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
        if ic_date_count is not None:
            metadata["ic_date_count"] = ic_date_count
        if mean_ic is not None:
            metadata["mean_ic"] = mean_ic
        if rank_ic_date_count is not None:
            metadata["rank_ic_date_count"] = rank_ic_date_count
        if mean_rank_ic is not None:
            metadata["mean_rank_ic"] = mean_rank_ic
        if sharpe is not None:
            metadata["sharpe"] = sharpe
        if max_drawdown is not None:
            metadata["max_drawdown"] = max_drawdown
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


def compute_information_coefficient(
    panel: pd.DataFrame,
    *,
    factor_column: str,
    factor_direction: FactorDirection = "positive",
) -> InformationCoefficientResult:
    """Compute cross-sectional Pearson IC by trading date."""

    if factor_direction not in SUPPORTED_FACTOR_DIRECTIONS:
        msg = "factor_direction must be either positive or negative."
        raise ValueError(msg)
    missing_columns = [
        column
        for column in ("date", factor_column, FORWARD_RETURN_COLUMN)
        if column not in panel.columns
    ]
    if missing_columns:
        msg = f"IC panel is missing required columns: {', '.join(missing_columns)}."
        raise ValueError(msg)

    rows: list[dict[str, Any]] = []
    skipped_date_count = 0
    direction_multiplier = -1.0 if factor_direction == "negative" else 1.0

    for trade_date, group in panel.groupby("date", sort=True):
        usable = group[[factor_column, FORWARD_RETURN_COLUMN]].dropna()
        if len(usable) < 2:
            skipped_date_count += 1
            continue
        if (
            usable[factor_column].nunique(dropna=True) < 2
            or usable[FORWARD_RETURN_COLUMN].nunique(dropna=True) < 2
        ):
            skipped_date_count += 1
            continue

        raw_ic = usable[factor_column].corr(usable[FORWARD_RETURN_COLUMN])
        if pd.isna(raw_ic):
            skipped_date_count += 1
            continue
        raw_ic_float = float(raw_ic)
        rows.append(
            {
                "date": trade_date,
                "ic": direction_multiplier * raw_ic_float,
                "raw_ic": raw_ic_float,
                "pair_count": int(len(usable)),
            }
        )

    ic_series = pd.DataFrame(rows, columns=IC_COLUMNS)
    if not ic_series.empty:
        ic_series = ic_series.sort_values("date").reset_index(drop=True)
    return InformationCoefficientResult(
        data=ic_series,
        stats=_ic_stats(ic_series, skipped_date_count),
    )


def _ic_stats(ic_series: pd.DataFrame, skipped_date_count: int) -> dict[str, Any]:
    if ic_series.empty:
        return {
            "method": "pearson",
            "ic_date_count": 0,
            "skipped_date_count": skipped_date_count,
            "mean_ic": None,
            "std_ic": None,
            "positive_ic_ratio": None,
            "average_pair_count": 0.0,
        }

    mean_ic = float(ic_series["ic"].mean())
    std_ic_raw = ic_series["ic"].std()
    std_ic = None if pd.isna(std_ic_raw) else float(std_ic_raw)
    return {
        "method": "pearson",
        "ic_date_count": len(ic_series),
        "skipped_date_count": skipped_date_count,
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "positive_ic_ratio": float((ic_series["ic"] > 0).mean()),
        "average_pair_count": float(ic_series["pair_count"].mean()),
    }


def compute_rank_information_coefficient(
    panel: pd.DataFrame,
    *,
    factor_column: str,
    factor_direction: FactorDirection = "positive",
) -> RankInformationCoefficientResult:
    """Compute cross-sectional Spearman RankIC by trading date."""

    if factor_direction not in SUPPORTED_FACTOR_DIRECTIONS:
        msg = "factor_direction must be either positive or negative."
        raise ValueError(msg)
    missing_columns = [
        column
        for column in ("date", factor_column, FORWARD_RETURN_COLUMN)
        if column not in panel.columns
    ]
    if missing_columns:
        msg = f"RankIC panel is missing required columns: {', '.join(missing_columns)}."
        raise ValueError(msg)

    rows: list[dict[str, Any]] = []
    skipped_date_count = 0
    direction_multiplier = -1.0 if factor_direction == "negative" else 1.0

    for trade_date, group in panel.groupby("date", sort=True):
        usable = group[[factor_column, FORWARD_RETURN_COLUMN]].dropna()
        if len(usable) < 2:
            skipped_date_count += 1
            continue
        if (
            usable[factor_column].nunique(dropna=True) < 2
            or usable[FORWARD_RETURN_COLUMN].nunique(dropna=True) < 2
        ):
            skipped_date_count += 1
            continue

        factor_rank = usable[factor_column].rank(method="average")
        return_rank = usable[FORWARD_RETURN_COLUMN].rank(method="average")
        raw_rank_ic = factor_rank.corr(return_rank)
        if pd.isna(raw_rank_ic):
            skipped_date_count += 1
            continue
        raw_rank_ic_float = float(raw_rank_ic)
        rows.append(
            {
                "date": trade_date,
                "rank_ic": direction_multiplier * raw_rank_ic_float,
                "raw_rank_ic": raw_rank_ic_float,
                "pair_count": int(len(usable)),
            }
        )

    rank_ic_series = pd.DataFrame(rows, columns=RANK_IC_COLUMNS)
    if not rank_ic_series.empty:
        rank_ic_series = rank_ic_series.sort_values("date").reset_index(drop=True)
    return RankInformationCoefficientResult(
        data=rank_ic_series,
        stats=_rank_ic_stats(rank_ic_series, skipped_date_count),
    )


def _rank_ic_stats(
    rank_ic_series: pd.DataFrame,
    skipped_date_count: int,
) -> dict[str, Any]:
    if rank_ic_series.empty:
        return {
            "method": "spearman",
            "rank_ic_date_count": 0,
            "skipped_date_count": skipped_date_count,
            "mean_rank_ic": None,
            "std_rank_ic": None,
            "positive_rank_ic_ratio": None,
            "average_pair_count": 0.0,
        }

    mean_rank_ic = float(rank_ic_series["rank_ic"].mean())
    std_rank_ic_raw = rank_ic_series["rank_ic"].std()
    std_rank_ic = None if pd.isna(std_rank_ic_raw) else float(std_rank_ic_raw)
    return {
        "method": "spearman",
        "rank_ic_date_count": len(rank_ic_series),
        "skipped_date_count": skipped_date_count,
        "mean_rank_ic": mean_rank_ic,
        "std_rank_ic": std_rank_ic,
        "positive_rank_ic_ratio": float((rank_ic_series["rank_ic"] > 0).mean()),
        "average_pair_count": float(rank_ic_series["pair_count"].mean()),
    }


def compute_sharpe_ratio(
    portfolio_returns: pd.DataFrame,
    *,
    return_column: str = SHARPE_RETURN_COLUMN,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
) -> SharpeResult:
    """Compute annualized Sharpe from a portfolio return series."""

    if isinstance(annualization_factor, bool) or not isinstance(
        annualization_factor,
        int,
    ):
        msg = "annualization_factor must be an integer."
        raise ValueError(msg)
    if (
        annualization_factor < MIN_ANNUALIZATION_FACTOR
        or annualization_factor > MAX_ANNUALIZATION_FACTOR
    ):
        msg = (
            "annualization_factor must be between "
            f"{MIN_ANNUALIZATION_FACTOR} and {MAX_ANNUALIZATION_FACTOR}."
        )
        raise ValueError(msg)
    if return_column not in portfolio_returns.columns:
        msg = f"Portfolio returns are missing return column: {return_column}."
        raise ValueError(msg)

    returns = pd.to_numeric(portfolio_returns[return_column], errors="coerce").dropna()
    if returns.empty:
        return SharpeResult(
            stats={
                "method": "mean_std",
                "return_column": return_column,
                "annualization_factor": annualization_factor,
                "return_count": 0,
                "mean_period_return": None,
                "std_period_return": None,
                "annualized_mean_return": None,
                "sharpe": None,
                "positive_return_ratio": None,
            }
        )

    mean_return = float(returns.mean())
    positive_return_ratio = float((returns > 0).mean())
    if len(returns) < 2:
        std_return: float | None = None
        sharpe: float | None = None
    else:
        std_return_raw = returns.std()
        std_return = None if pd.isna(std_return_raw) else float(std_return_raw)
        sharpe = (
            None
            if std_return is None or std_return == 0.0
            else mean_return / std_return * float(np.sqrt(annualization_factor))
        )

    return SharpeResult(
        stats={
            "method": "mean_std",
            "return_column": return_column,
            "annualization_factor": annualization_factor,
            "return_count": int(len(returns)),
            "mean_period_return": mean_return,
            "std_period_return": std_return,
            "annualized_mean_return": mean_return * annualization_factor,
            "sharpe": sharpe,
            "positive_return_ratio": positive_return_ratio,
        }
    )


def compute_drawdown(
    portfolio_returns: pd.DataFrame,
    *,
    return_column: str = DRAWDOWN_RETURN_COLUMN,
) -> DrawdownResult:
    """Compute an equity curve and drawdowns from portfolio returns."""

    missing_columns = [
        column for column in ("date", return_column) if column not in portfolio_returns.columns
    ]
    if missing_columns:
        msg = f"Portfolio returns are missing required columns: {', '.join(missing_columns)}."
        raise ValueError(msg)

    frame = portfolio_returns[["date", return_column]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame[return_column] = pd.to_numeric(frame[return_column], errors="coerce")
    if frame["date"].isna().any():
        msg = "Portfolio returns contain invalid dates."
        raise ValueError(msg)

    frame = frame.dropna(subset=[return_column]).sort_values("date").reset_index(drop=True)
    if frame.empty:
        return DrawdownResult(
            data=pd.DataFrame(columns=DRAWDOWN_COLUMNS),
            stats={
                "method": "cumulative_return",
                "return_column": return_column,
                "return_count": 0,
                "start_equity": 1.0,
                "end_equity": None,
                "total_return": None,
                "max_drawdown": None,
                "max_drawdown_abs": None,
                "peak_date": None,
                "trough_date": None,
                "recovery_date": None,
                "drawdown_period_count": 0,
                "average_drawdown": None,
            },
        )

    equity_curve = (1.0 + frame[return_column]).cumprod()
    cumulative_peak = equity_curve.cummax()
    drawdown = equity_curve / cumulative_peak.replace(0.0, np.nan) - 1.0
    drawdown = drawdown.fillna(0.0)
    curve = pd.DataFrame(
        {
            "date": frame["date"],
            "equity_curve": equity_curve,
            "cumulative_peak": cumulative_peak,
            "drawdown": drawdown,
        },
        columns=DRAWDOWN_COLUMNS,
    )
    return DrawdownResult(
        data=curve,
        stats=_drawdown_stats(curve, return_column),
    )


def _drawdown_stats(curve: pd.DataFrame, return_column: str) -> dict[str, Any]:
    trough_index = curve["drawdown"].idxmin()
    max_drawdown = float(cast(float, curve.loc[trough_index, "drawdown"]))
    peak_value_at_trough = cast(float, curve.loc[trough_index, "cumulative_peak"])
    peak_candidates = curve.loc[:trough_index]
    peak_index = peak_candidates[
        peak_candidates["equity_curve"] == peak_value_at_trough
    ].index[-1]

    recovery_candidates = curve.loc[trough_index:]
    recovered = recovery_candidates[
        recovery_candidates["equity_curve"] >= peak_value_at_trough
    ]
    recovery_date = None if recovered.empty else _date_string(recovered.iloc[0]["date"])
    negative_drawdowns = curve.loc[curve["drawdown"] < 0.0, "drawdown"]
    average_drawdown = (
        None
        if negative_drawdowns.empty
        else float(negative_drawdowns.mean())
    )

    end_equity = float(curve["equity_curve"].iloc[-1])
    return {
        "method": "cumulative_return",
        "return_column": return_column,
        "return_count": len(curve),
        "start_equity": 1.0,
        "end_equity": end_equity,
        "total_return": end_equity - 1.0,
        "max_drawdown": max_drawdown,
        "max_drawdown_abs": abs(max_drawdown),
        "peak_date": _date_string(curve.loc[peak_index, "date"]),
        "trough_date": _date_string(curve.loc[trough_index, "date"]),
        "recovery_date": recovery_date,
        "drawdown_period_count": int((curve["drawdown"] < 0.0).sum()),
        "average_drawdown": average_drawdown,
    }


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


def _date_string(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return str(value)


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
