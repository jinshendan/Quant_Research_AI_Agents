from __future__ import annotations

import pandas as pd
import pytest

from agents.factor_expression import (
    GeneratedFactorExpression,
    evaluate_factor_expression,
)


def _frame() -> pd.DataFrame:
    rows = []
    for symbol in ("000001", "000002"):
        base = 10.0 if symbol == "000001" else 20.0
        for day in range(1, 7):
            close = base + day
            rows.append(
                {
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day - 1),
                    "symbol": symbol,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1000.0 + day * 100.0,
                    "amount": (1000.0 + day * 100.0) * close,
                    "turnover_rate": float(day),
                }
            )
    return pd.DataFrame(rows)


def test_evaluate_generated_factor_expression_supports_core_functions() -> None:
    frame = _frame()

    momentum = evaluate_factor_expression(frame, "close / delay(close, 1) - 1")
    assert pd.isna(momentum.iloc[0])
    assert momentum.iloc[1] == pytest.approx(1 / 11)

    volume_ratio = evaluate_factor_expression(frame, "mean(volume, 2) / mean(volume, 3)")
    assert pd.isna(volume_ratio.iloc[1])
    assert volume_ratio.iloc[2] == pytest.approx(((1200 + 1300) / 2) / ((1100 + 1200 + 1300) / 3))

    positive_days = evaluate_factor_expression(
        frame,
        "mean((close / delay(close, 1) - 1) > 0, 3)",
    )
    assert pd.isna(positive_days.iloc[2])
    assert positive_days.iloc[3] == pytest.approx(1.0)

    breakout = evaluate_factor_expression(frame, "close / max(delay(high, 1), 3) - 1")
    assert pd.isna(breakout.iloc[2])
    assert breakout.iloc[3] == pytest.approx(14 / 14 - 1)


def test_generated_factor_expression_builds_factor_definition() -> None:
    generated = GeneratedFactorExpression.from_mapping(
        {
            "factor_id": "alpha_001",
            "family_id": "momentum_return",
            "source_template_id": "return_5d",
            "name": "One-Day Momentum Return",
            "category": "momentum",
            "expression": "close / delay(close, 1) - 1",
            "direction": "positive",
            "required_columns": ["close"],
            "parameters": {"window": 1},
            "lookback_days": 1,
            "signal_tags": ["momentum_return"],
            "risk_flags": ["momentum_crowding"],
            "generation_method": "deterministic_factor_family_v1",
        }
    )

    assert generated.factor_column == "factor__alpha_001"
    definition = generated.to_factor_definition().to_dict()
    assert definition["factor_column"] == "factor__alpha_001"
    assert definition["source_type"] == "generated"
    assert definition["formula"] == "close / delay(close, 1) - 1"
    assert definition["parameters"] == {"window": 1}


def test_evaluate_generated_factor_expression_rejects_unsafe_syntax() -> None:
    with pytest.raises(ValueError, match="supported"):
        evaluate_factor_expression(_frame(), "__import__('os').system('echo bad')")


def test_generated_factor_expression_rejects_future_tokens() -> None:
    with pytest.raises(ValueError, match="future-looking"):
        GeneratedFactorExpression.from_mapping(
            {
                "factor_id": "bad_alpha",
                "name": "Bad Alpha",
                "category": "bad",
                "expression": "lead(close, 1) / close - 1",
                "direction": "positive",
                "required_columns": ["close"],
                "parameters": {"window": 1},
                "lookback_days": 1,
            }
        )
