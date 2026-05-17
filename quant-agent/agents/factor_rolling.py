from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

RollingFeatureName = Literal["mean", "std", "min", "max", "zscore"]

IDENTITY_COLUMNS = ("date", "symbol")
SUPPORTED_ROLLING_FEATURES: set[str] = {"mean", "std", "min", "max", "zscore"}
DEFAULT_ROLLING_FEATURES: tuple[RollingFeatureName, ...] = ("mean", "std")
DEFAULT_ROLLING_WINDOWS = (5, 20)
MIN_ROLLING_WINDOW = 2
MAX_ROLLING_WINDOW = 252


@dataclass(frozen=True, slots=True)
class RollingFeatureSpec:
    """Validated per-symbol rolling-window feature request."""

    feature_names: tuple[RollingFeatureName, ...] = DEFAULT_ROLLING_FEATURES
    windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS

    @classmethod
    def create(
        cls,
        feature_names: Sequence[str] | None = None,
        *,
        windows: Sequence[int] | None = None,
    ) -> RollingFeatureSpec:
        return cls(
            feature_names=normalize_rolling_feature_names(
                feature_names or DEFAULT_ROLLING_FEATURES
            ),
            windows=normalize_rolling_windows(windows or DEFAULT_ROLLING_WINDOWS),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "windows": list(self.windows),
        }


@dataclass(frozen=True, slots=True)
class RollingFeatureResult:
    """Factor matrix with appended per-symbol rolling feature columns."""

    data: pd.DataFrame
    rolling_columns: tuple[str, ...]
    stats: dict[str, Any]


def apply_rolling_features(
    factor_data: pd.DataFrame,
    *,
    factor_columns: Sequence[str],
    spec: RollingFeatureSpec,
) -> RollingFeatureResult:
    """Append per-symbol rolling-window features for factor columns."""

    _validate_factor_data(factor_data, factor_columns)
    transformed = factor_data.copy()
    transformed["date"] = pd.to_datetime(transformed["date"], errors="coerce")
    if transformed["date"].isna().any():
        msg = "Factor data contains invalid dates."
        raise ValueError(msg)
    transformed = transformed.sort_values(["symbol", "date"]).reset_index(drop=True)

    rolling_columns: list[str] = []
    for factor_column in factor_columns:
        values = transformed[factor_column]
        for window in spec.windows:
            for feature_name in spec.feature_names:
                output_column = rolling_column_name(
                    factor_column,
                    feature_name,
                    window=window,
                )
                transformed[output_column] = _apply_one_rolling_feature(
                    transformed,
                    values,
                    feature_name,
                    window=window,
                )
                rolling_columns.append(output_column)

    return RollingFeatureResult(
        data=transformed,
        rolling_columns=tuple(rolling_columns),
        stats=_rolling_feature_stats(transformed, rolling_columns, spec),
    )


def normalize_rolling_feature_names(
    feature_names: Sequence[str],
) -> tuple[RollingFeatureName, ...]:
    if isinstance(feature_names, str):
        msg = "rolling_features must be a sequence of strings."
        raise ValueError(msg)
    if not feature_names:
        msg = "rolling_features must not be empty when provided."
        raise ValueError(msg)

    normalized: list[RollingFeatureName] = []
    for feature_name in feature_names:
        if not isinstance(feature_name, str) or not feature_name.strip():
            msg = "rolling_features must contain only non-empty strings."
            raise ValueError(msg)
        cleaned = feature_name.strip().lower()
        if cleaned not in SUPPORTED_ROLLING_FEATURES:
            msg = f"Unsupported rolling feature: {feature_name}."
            raise ValueError(msg)
        normalized.append(cast(RollingFeatureName, cleaned))
    return tuple(dict.fromkeys(normalized))


def normalize_rolling_windows(windows: Sequence[int]) -> tuple[int, ...]:
    if isinstance(windows, str):
        msg = "rolling_windows must be a sequence of integers."
        raise ValueError(msg)
    if not windows:
        msg = "rolling_windows must not be empty when provided."
        raise ValueError(msg)

    normalized = []
    for window in windows:
        if isinstance(window, bool) or not isinstance(window, int):
            msg = "rolling_windows must contain only integers."
            raise ValueError(msg)
        if window < MIN_ROLLING_WINDOW or window > MAX_ROLLING_WINDOW:
            msg = f"rolling window must be between {MIN_ROLLING_WINDOW} and {MAX_ROLLING_WINDOW}."
            raise ValueError(msg)
        normalized.append(window)
    return tuple(dict.fromkeys(normalized))


def rolling_column_name(
    factor_column: str,
    feature_name: RollingFeatureName,
    *,
    window: int,
) -> str:
    return f"{factor_column}__roll_{feature_name}_{window}"


def _apply_one_rolling_feature(
    frame: pd.DataFrame,
    values: pd.Series,
    feature_name: RollingFeatureName,
    *,
    window: int,
) -> pd.Series:
    grouped = values.groupby(frame["symbol"], sort=False)
    rolling_mean = grouped.transform(lambda group: group.rolling(window, min_periods=window).mean())

    if feature_name == "mean":
        return rolling_mean
    if feature_name == "std":
        return grouped.transform(lambda group: group.rolling(window, min_periods=window).std())
    if feature_name == "min":
        return grouped.transform(lambda group: group.rolling(window, min_periods=window).min())
    if feature_name == "max":
        return grouped.transform(lambda group: group.rolling(window, min_periods=window).max())

    rolling_std = grouped.transform(lambda group: group.rolling(window, min_periods=window).std())
    return (values - rolling_mean) / rolling_std.replace(0, np.nan)


def _validate_factor_data(factor_data: pd.DataFrame, factor_columns: Sequence[str]) -> None:
    missing_identity_columns = [
        column for column in IDENTITY_COLUMNS if column not in factor_data.columns
    ]
    if missing_identity_columns:
        msg = f"Factor data is missing required columns: {', '.join(missing_identity_columns)}."
        raise ValueError(msg)
    if not factor_columns:
        msg = "At least one factor column is required."
        raise ValueError(msg)

    missing_factor_columns = [
        column for column in factor_columns if column not in factor_data.columns
    ]
    if missing_factor_columns:
        msg = f"Factor data is missing factor columns: {', '.join(missing_factor_columns)}."
        raise ValueError(msg)


def _rolling_feature_stats(
    frame: pd.DataFrame,
    rolling_columns: Sequence[str],
    spec: RollingFeatureSpec,
) -> dict[str, Any]:
    counts_by_feature: Counter[str] = Counter()
    by_column = {}
    for column in rolling_columns:
        feature_name = _rolling_feature_name_from_column(column)
        counts_by_feature[feature_name] += 1
        by_column[column] = {
            "valid_values": int(frame[column].notna().sum()),
            "missing_values": int(frame[column].isna().sum()),
        }

    return {
        "feature_names": list(spec.feature_names),
        "windows": list(spec.windows),
        "rolling_factor_count": len(rolling_columns),
        "counts_by_feature": dict(sorted(counts_by_feature.items())),
        "by_column": by_column,
    }


def _rolling_feature_name_from_column(column: str) -> str:
    suffix = column.rsplit("__roll_", maxsplit=1)[-1]
    return suffix.rsplit("_", maxsplit=1)[0]
