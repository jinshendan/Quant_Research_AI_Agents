from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

RankTransformName = Literal["rank", "rank_pct", "demean", "zscore", "quantile"]

IDENTITY_COLUMNS = ("date", "symbol")
SUPPORTED_RANK_TRANSFORMS: set[str] = {
    "rank",
    "rank_pct",
    "demean",
    "zscore",
    "quantile",
}
DEFAULT_QUANTILE_COUNT = 5
MIN_QUANTILE_COUNT = 2
MAX_QUANTILE_COUNT = 20


@dataclass(frozen=True, slots=True)
class RankTransformSpec:
    """Validated cross-sectional ranking transform request."""

    transform_names: tuple[RankTransformName, ...] = ("rank_pct",)
    quantile_count: int = DEFAULT_QUANTILE_COUNT

    @classmethod
    def create(
        cls,
        transform_names: Sequence[str],
        *,
        quantile_count: int = DEFAULT_QUANTILE_COUNT,
    ) -> RankTransformSpec:
        return cls(
            transform_names=normalize_rank_transform_names(transform_names),
            quantile_count=validate_quantile_count(quantile_count),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform_names": list(self.transform_names),
            "quantile_count": self.quantile_count,
        }


@dataclass(frozen=True, slots=True)
class RankTransformResult:
    """Factor matrix with appended ranking transform columns."""

    data: pd.DataFrame
    transformed_columns: tuple[str, ...]
    stats: dict[str, Any]


def apply_rank_transforms(
    factor_data: pd.DataFrame,
    *,
    factor_columns: Sequence[str],
    spec: RankTransformSpec,
) -> RankTransformResult:
    """Append cross-sectional transforms for factor columns by date."""

    _validate_factor_data(factor_data, factor_columns)
    transformed = factor_data.copy()
    transformed_columns: list[str] = []

    for factor_column in factor_columns:
        for transform_name in spec.transform_names:
            output_column = transformed_column_name(
                factor_column,
                transform_name,
                quantile_count=spec.quantile_count,
            )
            transformed[output_column] = _apply_one_transform(
                transformed,
                factor_column,
                transform_name,
                quantile_count=spec.quantile_count,
            )
            transformed_columns.append(output_column)

    return RankTransformResult(
        data=transformed,
        transformed_columns=tuple(transformed_columns),
        stats=_transform_stats(transformed, transformed_columns, spec),
    )


def normalize_rank_transform_names(transform_names: Sequence[str]) -> tuple[RankTransformName, ...]:
    if isinstance(transform_names, str):
        msg = "rank_transforms must be a sequence of strings."
        raise ValueError(msg)
    if not transform_names:
        msg = "rank_transforms must not be empty when provided."
        raise ValueError(msg)

    normalized: list[RankTransformName] = []
    for transform_name in transform_names:
        if not isinstance(transform_name, str) or not transform_name.strip():
            msg = "rank_transforms must contain only non-empty strings."
            raise ValueError(msg)
        cleaned = transform_name.strip().lower()
        if cleaned not in SUPPORTED_RANK_TRANSFORMS:
            msg = f"Unsupported rank transform: {transform_name}."
            raise ValueError(msg)
        normalized.append(cast(RankTransformName, cleaned))
    return tuple(dict.fromkeys(normalized))


def validate_quantile_count(quantile_count: int) -> int:
    if isinstance(quantile_count, bool) or not isinstance(quantile_count, int):
        msg = "quantile_count must be an integer."
        raise ValueError(msg)
    if quantile_count < MIN_QUANTILE_COUNT or quantile_count > MAX_QUANTILE_COUNT:
        msg = f"quantile_count must be between {MIN_QUANTILE_COUNT} and {MAX_QUANTILE_COUNT}."
        raise ValueError(msg)
    return quantile_count


def transformed_column_name(
    factor_column: str,
    transform_name: RankTransformName,
    *,
    quantile_count: int = DEFAULT_QUANTILE_COUNT,
) -> str:
    if transform_name == "quantile":
        return f"{factor_column}__quantile_{quantile_count}"
    return f"{factor_column}__{transform_name}"


def _apply_one_transform(
    frame: pd.DataFrame,
    factor_column: str,
    transform_name: RankTransformName,
    *,
    quantile_count: int,
) -> pd.Series:
    grouped = frame.groupby("date", sort=False)[factor_column]
    if transform_name == "rank":
        return grouped.rank(method="average", na_option="keep", pct=False)
    if transform_name == "rank_pct":
        return grouped.rank(method="average", na_option="keep", pct=True)
    if transform_name == "demean":
        return frame[factor_column] - grouped.transform("mean")
    if transform_name == "zscore":
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        return (frame[factor_column] - mean) / std

    rank_pct = grouped.rank(method="average", na_option="keep", pct=True)
    quantiles = np.ceil(rank_pct * quantile_count)
    return quantiles.clip(lower=1, upper=quantile_count)


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


def _transform_stats(
    frame: pd.DataFrame,
    transformed_columns: Sequence[str],
    spec: RankTransformSpec,
) -> dict[str, Any]:
    counts_by_transform: Counter[str] = Counter()
    by_column = {}
    for column in transformed_columns:
        transform_name = _transform_name_from_column(column)
        counts_by_transform[transform_name] += 1
        by_column[column] = {
            "valid_values": int(frame[column].notna().sum()),
            "missing_values": int(frame[column].isna().sum()),
        }

    return {
        "transform_names": list(spec.transform_names),
        "quantile_count": spec.quantile_count,
        "transformed_factor_count": len(transformed_columns),
        "counts_by_transform": dict(sorted(counts_by_transform.items())),
        "by_column": by_column,
    }


def _transform_name_from_column(column: str) -> str:
    suffix = column.rsplit("__", maxsplit=1)[-1]
    if suffix.startswith("quantile_"):
        return "quantile"
    return suffix
