from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from agents.factor_rolling import (
    RollingFeatureSpec,
    apply_rolling_features,
    normalize_rolling_windows,
)
from agents.factor_templates import FactorTemplate, FactorTemplateLibrary
from agents.factor_transforms import (
    DEFAULT_QUANTILE_COUNT,
    RankTransformSpec,
    apply_rank_transforms,
    validate_quantile_count,
)
from core.logging import AgentLoggerAdapter, get_agent_logger
from core.models import AgentRequest, AgentResponse

IDENTITY_COLUMNS = ("date", "symbol")
SUSPENSION_COLUMN = "is_suspended_or_missing"
DEFAULT_PREVIEW_ROWS = 5
MAX_PREVIEW_ROWS = 50

FactorExecutor = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Validated request for factor feature generation."""

    aligned_data_path: Path
    template_ids: tuple[str, ...] = ()
    rolling_features: tuple[str, ...] = ()
    rolling_windows: tuple[int, ...] = ()
    rank_transforms: tuple[str, ...] = ()
    quantile_count: int = DEFAULT_QUANTILE_COUNT
    preview_rows: int = DEFAULT_PREVIEW_ROWS

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> FeatureSpec:
        aligned_data_path = _required_path(payload, "aligned_data_path")
        template_ids = _optional_str_sequence(payload, "template_ids")
        rolling_features = _optional_str_sequence(payload, "rolling_features")
        raw_rolling_windows = _optional_int_sequence(payload, "rolling_windows")
        rolling_windows = (
            normalize_rolling_windows(raw_rolling_windows)
            if raw_rolling_windows
            else ()
        )
        rank_transforms = _optional_str_sequence(payload, "rank_transforms")
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
            aligned_data_path=aligned_data_path,
            template_ids=template_ids,
            rolling_features=rolling_features,
            rolling_windows=rolling_windows,
            rank_transforms=rank_transforms,
            quantile_count=quantile_count,
            preview_rows=preview_rows,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "aligned_data_path": str(self.aligned_data_path),
            "template_ids": list(self.template_ids),
            "rolling_features": list(self.rolling_features),
            "rolling_windows": list(self.rolling_windows),
            "rank_transforms": list(self.rank_transforms),
            "quantile_count": self.quantile_count,
            "preview_rows": self.preview_rows,
        }


@dataclass(frozen=True, slots=True)
class FeatureGenerationResult:
    """In-memory factor matrix and quality statistics."""

    data: pd.DataFrame
    factor_columns: tuple[str, ...]
    base_factor_columns: tuple[str, ...]
    rolling_feature_columns: tuple[str, ...]
    transformed_factor_columns: tuple[str, ...]
    stats: dict[str, Any]
    rolling_feature_stats: dict[str, Any] | None = None
    rank_transform_stats: dict[str, Any] | None = None


class FeatureAgent:
    """Compute factor values from aligned OHLCV data and factor templates."""

    name = "FeatureAgent"

    def __init__(
        self,
        *,
        logger: AgentLoggerAdapter | None = None,
        template_library: FactorTemplateLibrary | None = None,
    ) -> None:
        self.logger = logger or get_agent_logger(self.name)
        self.template_library = template_library or FactorTemplateLibrary()

    def run(self, request: AgentRequest) -> AgentResponse:
        started_at = perf_counter()
        self.logger.info(
            "Received feature generation request.",
            extra={"action": "validate_request", "status": "running"},
        )

        try:
            spec = FeatureSpec.from_payload(request.payload)
            templates = self._templates_for(spec)
            aligned_data = self.load_aligned_data(spec.aligned_data_path)
        except (OSError, KeyError, ValueError) as exc:
            elapsed = perf_counter() - started_at
            self.logger.warning(
                "Feature generation request validation failed.",
                extra={"action": "validate_request", "status": "error"},
            )
            return AgentResponse.failure(
                str(exc),
                metadata=self._metadata(request, elapsed),
            )

        self.logger.info(
            "Computing factor features.",
            extra={"action": "generate_features", "status": "running"},
        )
        try:
            result = self.generate_features(
                aligned_data,
                templates,
                rolling_feature_names=spec.rolling_features,
                rolling_windows=spec.rolling_windows,
                rank_transform_names=spec.rank_transforms,
                quantile_count=spec.quantile_count,
            )
        except ValueError as exc:
            elapsed = perf_counter() - started_at
            self.logger.warning(
                "Feature generation failed.",
                extra={"action": "generate_features", "status": "error"},
            )
            return AgentResponse.failure(
                str(exc),
                metadata=self._metadata(
                    request,
                    elapsed,
                    template_count=len(templates),
                    template_ids=[template.template_id for template in templates],
                ),
            )

        elapsed = perf_counter() - started_at
        self.logger.info(
            "Generated factor features.",
            extra={"action": "generate_features", "status": "success"},
        )
        template_ids = [template.template_id for template in templates]
        return AgentResponse.success(
            output={
                "state": "features_generated",
                "request": spec.to_dict(),
                "template_ids": template_ids,
                "factor_columns": list(result.factor_columns),
                "base_factor_columns": list(result.base_factor_columns),
                "rolling_feature_columns": list(result.rolling_feature_columns),
                "transformed_factor_columns": list(result.transformed_factor_columns),
                "rolling_features": list(spec.rolling_features),
                "rolling_windows": list(spec.rolling_windows),
                "rank_transforms": list(spec.rank_transforms),
                "row_count": len(result.data),
                "factor_count": len(result.factor_columns),
                "preview": _preview_records(result.data, spec.preview_rows),
                "feature_stats": result.stats,
                "rolling_feature_stats": result.rolling_feature_stats or {},
                "rank_transform_stats": result.rank_transform_stats or {},
                "next_action": "Save generated factors in Day 14.",
            },
            metadata=self._metadata(
                request,
                elapsed,
                aligned_data_path=spec.aligned_data_path,
                row_count=len(result.data),
                factor_count=len(result.factor_columns),
                template_count=len(templates),
                template_ids=template_ids,
            ),
        )

    def load_aligned_data(self, aligned_data_path: Path) -> pd.DataFrame:
        if not aligned_data_path.is_file():
            msg = f"Aligned data file not found: {aligned_data_path}."
            raise OSError(msg)
        frame = pd.read_csv(aligned_data_path)
        return normalize_aligned_ohlcv(frame)

    def generate_features(
        self,
        aligned_data: pd.DataFrame,
        templates: Sequence[FactorTemplate],
        *,
        rolling_feature_names: Sequence[str] = (),
        rolling_windows: Sequence[int] = (),
        rank_transform_names: Sequence[str] = (),
        quantile_count: int = DEFAULT_QUANTILE_COUNT,
    ) -> FeatureGenerationResult:
        if not templates:
            msg = "At least one factor template is required."
            raise ValueError(msg)

        frame = normalize_aligned_ohlcv(aligned_data)
        factor_columns = []
        for template in templates:
            _validate_required_columns(frame, template)
            executor = _executor_for(template.template_id)
            factor_column = f"factor__{template.template_id}"
            values = executor(frame)
            frame[factor_column] = _mask_untradable_rows(values, frame)
            factor_columns.append(factor_column)

        base_factor_columns = tuple(factor_columns)
        rolling_feature_columns: tuple[str, ...] = ()
        rolling_feature_stats: dict[str, Any] | None = None
        if rolling_feature_names or rolling_windows:
            rolling_result = apply_rolling_features(
                frame[list(IDENTITY_COLUMNS) + factor_columns],
                factor_columns=factor_columns,
                spec=RollingFeatureSpec.create(
                    rolling_feature_names or None,
                    windows=rolling_windows or None,
                ),
            )
            frame = rolling_result.data
            rolling_feature_columns = rolling_result.rolling_columns
            factor_columns.extend(rolling_feature_columns)
            rolling_feature_stats = rolling_result.stats

        transformed_factor_columns: tuple[str, ...] = ()
        rank_transform_stats: dict[str, Any] | None = None
        if rank_transform_names:
            transform_result = apply_rank_transforms(
                frame[list(IDENTITY_COLUMNS) + factor_columns],
                factor_columns=factor_columns,
                spec=RankTransformSpec.create(
                    rank_transform_names,
                    quantile_count=quantile_count,
                ),
            )
            frame = transform_result.data
            transformed_factor_columns = transform_result.transformed_columns
            factor_columns.extend(transformed_factor_columns)
            rank_transform_stats = transform_result.stats

        stats = _feature_stats(frame, factor_columns)
        return FeatureGenerationResult(
            data=frame[list(IDENTITY_COLUMNS) + factor_columns].copy(),
            factor_columns=tuple(factor_columns),
            base_factor_columns=base_factor_columns,
            rolling_feature_columns=rolling_feature_columns,
            transformed_factor_columns=transformed_factor_columns,
            stats=stats,
            rolling_feature_stats=rolling_feature_stats,
            rank_transform_stats=rank_transform_stats,
        )

    def _templates_for(self, spec: FeatureSpec) -> list[FactorTemplate]:
        if spec.template_ids:
            return [self.template_library.get(template_id) for template_id in spec.template_ids]
        return self.template_library.all()

    def _metadata(
        self,
        request: AgentRequest,
        elapsed: float,
        *,
        aligned_data_path: Path | None = None,
        row_count: int | None = None,
        factor_count: int | None = None,
        template_count: int | None = None,
        template_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "agent": self.name,
            "task_id": request.task_id,
            "execution_time_sec": round(elapsed, 6),
        }
        if aligned_data_path is not None:
            metadata["aligned_data_path"] = str(aligned_data_path)
        if row_count is not None:
            metadata["row_count"] = row_count
        if factor_count is not None:
            metadata["factor_count"] = factor_count
        if template_count is not None:
            metadata["template_count"] = template_count
        if template_ids is not None:
            metadata["template_ids"] = list(template_ids)
        return metadata


def normalize_aligned_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize aligned OHLCV rows before feature computation."""

    missing_columns = [column for column in IDENTITY_COLUMNS if column not in frame.columns]
    if missing_columns:
        msg = f"Aligned data is missing required columns: {', '.join(missing_columns)}."
        raise ValueError(msg)

    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["symbol"] = normalized["symbol"].astype(str).str.zfill(6)
    if normalized["date"].isna().any():
        msg = "Aligned data contains invalid dates."
        raise ValueError(msg)

    for column in _numeric_columns(normalized):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    if SUSPENSION_COLUMN in normalized.columns:
        normalized[SUSPENSION_COLUMN] = normalized[SUSPENSION_COLUMN].fillna(False).astype(bool)
    else:
        normalized[SUSPENSION_COLUMN] = False

    return normalized.sort_values(["symbol", "date"]).reset_index(drop=True)


def _executor_for(template_id: str) -> FactorExecutor:
    try:
        return _TEMPLATE_EXECUTORS[template_id]
    except KeyError as exc:
        msg = f"FeatureAgent does not support template execution yet: {template_id}."
        raise ValueError(msg) from exc


def _pct_change(frame: pd.DataFrame, column: str, periods: int) -> pd.Series:
    return frame.groupby("symbol", sort=False)[column].pct_change(periods=periods)


def _delay(frame: pd.DataFrame, column: str, periods: int) -> pd.Series:
    return frame.groupby("symbol", sort=False)[column].shift(periods)


def _rolling_mean(frame: pd.DataFrame, values: pd.Series, window: int) -> pd.Series:
    series = values.rename("_value")
    return series.groupby(frame["symbol"], sort=False).transform(
        lambda group: group.rolling(window, min_periods=window).mean()
    )


def _rolling_std(frame: pd.DataFrame, values: pd.Series, window: int) -> pd.Series:
    series = values.rename("_value")
    return series.groupby(frame["symbol"], sort=False).transform(
        lambda group: group.rolling(window, min_periods=window).std()
    )


def _rolling_max(frame: pd.DataFrame, values: pd.Series, window: int) -> pd.Series:
    series = values.rename("_value")
    return series.groupby(frame["symbol"], sort=False).transform(
        lambda group: group.rolling(window, min_periods=window).max()
    )


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def _zscore(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    values = frame[column]
    mean = _rolling_mean(frame, values, window)
    std = _rolling_std(frame, values, window)
    return _safe_divide(values - mean, std)


def _max_drawdown(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    def calculate(values: np.ndarray) -> float:
        cumulative_peak = np.maximum.accumulate(values)
        drawdowns = values / cumulative_peak - 1.0
        return float(np.nanmin(drawdowns))

    return frame.groupby("symbol", sort=False)[column].transform(
        lambda group: group.rolling(window, min_periods=window).apply(calculate, raw=True)
    )


def _one_day_return(frame: pd.DataFrame) -> pd.Series:
    return _pct_change(frame, "close", 1)


def _template_amount_growth_5d(frame: pd.DataFrame) -> pd.Series:
    return _pct_change(frame, "amount", 5)


def _template_turnover_rate_change_5d(frame: pd.DataFrame) -> pd.Series:
    return frame["turnover_rate"] - _delay(frame, "turnover_rate", 5)


def _template_volume_ratio_5d_20d(frame: pd.DataFrame) -> pd.Series:
    fast = _rolling_mean(frame, frame["volume"], 5)
    slow = _rolling_mean(frame, frame["volume"], 20)
    return _safe_divide(fast, slow)


def _template_return_5d(frame: pd.DataFrame) -> pd.Series:
    return _pct_change(frame, "close", 5)


def _template_return_3d(frame: pd.DataFrame) -> pd.Series:
    return _pct_change(frame, "close", 3)


def _template_volume_zscore_5d(frame: pd.DataFrame) -> pd.Series:
    return _zscore(frame, "volume", 5)


def _template_intraday_range_5d(frame: pd.DataFrame) -> pd.Series:
    intraday_range = _safe_divide(frame["high"] - frame["low"], frame["close"])
    return _rolling_mean(frame, intraday_range, 5)


def _template_realized_volatility_20d(frame: pd.DataFrame) -> pd.Series:
    return _rolling_std(frame, _one_day_return(frame), 20)


def _template_high_breakout_20d(frame: pd.DataFrame) -> pd.Series:
    prior_high = _delay(frame, "high", 1)
    rolling_prior_high = _rolling_max(frame, prior_high, 20)
    return _safe_divide(frame["close"], rolling_prior_high) - 1.0


def _template_close_position_in_range(frame: pd.DataFrame) -> pd.Series:
    return _safe_divide(frame["close"] - frame["low"], frame["high"] - frame["low"])


def _template_max_drawdown_20d(frame: pd.DataFrame) -> pd.Series:
    return _max_drawdown(frame, "close", 20)


def _template_positive_return_days_20d(frame: pd.DataFrame) -> pd.Series:
    positive_return = (_one_day_return(frame) > 0).astype(float)
    positive_return[_one_day_return(frame).isna()] = np.nan
    return _rolling_mean(frame, positive_return, 20)


def _template_turnover_rate_zscore_20d(frame: pd.DataFrame) -> pd.Series:
    return _zscore(frame, "turnover_rate", 20)


def _template_amount_zscore_20d(frame: pd.DataFrame) -> pd.Series:
    return _zscore(frame, "amount", 20)


def _template_absolute_return_1d(frame: pd.DataFrame) -> pd.Series:
    return _one_day_return(frame).abs()


def _template_close_to_open_return(frame: pd.DataFrame) -> pd.Series:
    return _safe_divide(frame["close"], frame["open"]) - 1.0


def _template_volume_ratio_1d_20d(frame: pd.DataFrame) -> pd.Series:
    slow = _rolling_mean(frame, frame["volume"], 20)
    return _safe_divide(frame["volume"], slow)


def _template_upper_shadow_ratio(frame: pd.DataFrame) -> pd.Series:
    candle_body_high = pd.concat([frame["open"], frame["close"]], axis=1).max(axis=1)
    return _safe_divide(frame["high"] - candle_body_high, frame["high"] - frame["low"])


_TEMPLATE_EXECUTORS: dict[str, FactorExecutor] = {
    "amount_growth_5d": _template_amount_growth_5d,
    "turnover_rate_change_5d": _template_turnover_rate_change_5d,
    "volume_ratio_5d_20d": _template_volume_ratio_5d_20d,
    "return_5d": _template_return_5d,
    "return_3d": _template_return_3d,
    "volume_zscore_5d": _template_volume_zscore_5d,
    "intraday_range_5d": _template_intraday_range_5d,
    "realized_volatility_20d": _template_realized_volatility_20d,
    "high_breakout_20d": _template_high_breakout_20d,
    "close_position_in_range": _template_close_position_in_range,
    "max_drawdown_20d": _template_max_drawdown_20d,
    "positive_return_days_20d": _template_positive_return_days_20d,
    "turnover_rate_zscore_20d": _template_turnover_rate_zscore_20d,
    "amount_zscore_20d": _template_amount_zscore_20d,
    "absolute_return_1d": _template_absolute_return_1d,
    "close_to_open_return": _template_close_to_open_return,
    "volume_ratio_1d_20d": _template_volume_ratio_1d_20d,
    "upper_shadow_ratio": _template_upper_shadow_ratio,
}


def _validate_required_columns(frame: pd.DataFrame, template: FactorTemplate) -> None:
    missing_columns = [
        column
        for column in template.required_columns
        if column not in frame.columns
    ]
    if missing_columns:
        msg = (
            f"Template {template.template_id} requires missing columns: "
            f"{', '.join(missing_columns)}."
        )
        raise ValueError(msg)


def _mask_untradable_rows(values: pd.Series, frame: pd.DataFrame) -> pd.Series:
    masked = values.replace([np.inf, -np.inf], np.nan)
    return masked.mask(frame[SUSPENSION_COLUMN])


def _feature_stats(frame: pd.DataFrame, factor_columns: Sequence[str]) -> dict[str, Any]:
    by_factor = {}
    for column in factor_columns:
        valid_values = int(frame[column].notna().sum())
        missing_values = int(frame[column].isna().sum())
        by_factor[column] = {
            "valid_values": valid_values,
            "missing_values": missing_values,
        }
    return {
        "row_count": len(frame),
        "symbol_count": int(frame["symbol"].nunique()),
        "factor_count": len(factor_columns),
        "by_factor": by_factor,
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


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in (*IDENTITY_COLUMNS, SUSPENSION_COLUMN)
    ]


def _required_path(payload: Mapping[str, Any], key: str) -> Path:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        msg = f"payload.{key} must be a non-empty path string."
        raise ValueError(msg)
    return Path(value.strip()).expanduser().resolve()


def _optional_str_sequence(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if value is None:
        return ()
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


def _optional_int_sequence(payload: Mapping[str, Any], key: str) -> tuple[int, ...]:
    value = payload.get(key)
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        msg = f"payload.{key} must be a sequence of integers."
        raise ValueError(msg)

    values = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            msg = f"payload.{key} must contain only integers."
            raise ValueError(msg)
        values.append(item)
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
