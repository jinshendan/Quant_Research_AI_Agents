from __future__ import annotations

import pandas as pd
import pytest

from agents.factor_transforms import (
    RankTransformSpec,
    apply_rank_transforms,
    normalize_rank_transform_names,
    transformed_column_name,
    validate_quantile_count,
)


def _factor_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
                "2024-01-02",
            ],
            "symbol": ["000001", "000002", "000003", "000001", "000002", "000003"],
            "factor__return_5d": [0.1, 0.3, None, -0.2, 0.0, 0.2],
        }
    )


def test_rank_transform_spec_normalizes_names() -> None:
    spec = RankTransformSpec.create(
        [" Rank_Pct ", "rank_pct", "quantile"],
        quantile_count=4,
    )

    assert spec.to_dict() == {
        "transform_names": ["rank_pct", "quantile"],
        "quantile_count": 4,
    }


def test_apply_rank_transforms_appends_cross_sectional_columns() -> None:
    result = apply_rank_transforms(
        _factor_data(),
        factor_columns=["factor__return_5d"],
        spec=RankTransformSpec.create(
            ["rank", "rank_pct", "demean", "zscore", "quantile"],
            quantile_count=4,
        ),
    )

    assert result.transformed_columns == (
        "factor__return_5d__rank",
        "factor__return_5d__rank_pct",
        "factor__return_5d__demean",
        "factor__return_5d__zscore",
        "factor__return_5d__quantile_4",
    )

    first_day = result.data[result.data["date"] == "2024-01-01"].set_index("symbol")
    assert first_day.loc["000001", "factor__return_5d__rank"] == pytest.approx(1.0)
    assert first_day.loc["000002", "factor__return_5d__rank_pct"] == pytest.approx(1.0)
    assert pd.isna(first_day.loc["000003", "factor__return_5d__rank_pct"])
    assert first_day.loc["000001", "factor__return_5d__demean"] == pytest.approx(-0.1)
    assert first_day.loc["000002", "factor__return_5d__quantile_4"] == pytest.approx(4.0)

    assert result.stats["transformed_factor_count"] == 5
    assert result.stats["counts_by_transform"] == {
        "demean": 1,
        "quantile": 1,
        "rank": 1,
        "rank_pct": 1,
        "zscore": 1,
    }


def test_transformed_column_name_includes_quantile_count() -> None:
    assert (
        transformed_column_name("factor__return_5d", "quantile", quantile_count=10)
        == "factor__return_5d__quantile_10"
    )


def test_rank_transforms_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="Unsupported rank transform"):
        normalize_rank_transform_names(["bad"])
    with pytest.raises(ValueError, match="quantile_count"):
        validate_quantile_count(1)
    with pytest.raises(ValueError, match="missing factor columns"):
        apply_rank_transforms(
            _factor_data(),
            factor_columns=["missing_factor"],
            spec=RankTransformSpec.create(["rank_pct"]),
        )
