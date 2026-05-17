from __future__ import annotations

import pandas as pd
import pytest

from agents.factor_rolling import (
    RollingFeatureSpec,
    apply_rolling_features,
    normalize_rolling_feature_names,
    normalize_rolling_windows,
    rolling_column_name,
)


def _factor_data() -> pd.DataFrame:
    rows = []
    for symbol in ("000001", "000002"):
        base = 0.0 if symbol == "000001" else 10.0
        for day in range(1, 6):
            rows.append(
                {
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day - 1),
                    "symbol": symbol,
                    "factor__return_5d": base + day,
                }
            )
    return pd.DataFrame(rows)


def test_rolling_feature_spec_normalizes_inputs() -> None:
    spec = RollingFeatureSpec.create(
        [" Mean ", "mean", "zscore"],
        windows=[3, 3, 5],
    )

    assert spec.to_dict() == {
        "feature_names": ["mean", "zscore"],
        "windows": [3, 5],
    }


def test_apply_rolling_features_appends_per_symbol_columns() -> None:
    result = apply_rolling_features(
        _factor_data(),
        factor_columns=["factor__return_5d"],
        spec=RollingFeatureSpec.create(
            ["mean", "std", "min", "max", "zscore"],
            windows=[3],
        ),
    )

    assert result.rolling_columns == (
        "factor__return_5d__roll_mean_3",
        "factor__return_5d__roll_std_3",
        "factor__return_5d__roll_min_3",
        "factor__return_5d__roll_max_3",
        "factor__return_5d__roll_zscore_3",
    )

    row = result.data[
        (result.data["symbol"] == "000001")
        & (result.data["date"] == pd.Timestamp("2024-01-03"))
    ].iloc[0]
    assert row["factor__return_5d__roll_mean_3"] == pytest.approx(2.0)
    assert row["factor__return_5d__roll_std_3"] == pytest.approx(1.0)
    assert row["factor__return_5d__roll_min_3"] == pytest.approx(1.0)
    assert row["factor__return_5d__roll_max_3"] == pytest.approx(3.0)
    assert row["factor__return_5d__roll_zscore_3"] == pytest.approx(1.0)

    assert result.stats["rolling_factor_count"] == 5
    assert result.stats["counts_by_feature"] == {
        "max": 1,
        "mean": 1,
        "min": 1,
        "std": 1,
        "zscore": 1,
    }


def test_rolling_column_name_includes_window() -> None:
    assert (
        rolling_column_name("factor__return_5d", "mean", window=20)
        == "factor__return_5d__roll_mean_20"
    )


def test_rolling_features_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="Unsupported rolling feature"):
        normalize_rolling_feature_names(["bad"])
    with pytest.raises(ValueError, match="rolling window"):
        normalize_rolling_windows([1])
    with pytest.raises(ValueError, match="missing factor columns"):
        apply_rolling_features(
            _factor_data(),
            factor_columns=["missing_factor"],
            spec=RollingFeatureSpec.create(["mean"], windows=[3]),
        )
