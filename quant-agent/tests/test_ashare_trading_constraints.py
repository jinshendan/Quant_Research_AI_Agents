from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from agents.ashare_trading_constraints import (
    AshareTradingConstraintSpec,
    apply_ashare_trading_constraints,
)


def test_apply_ashare_trading_constraints_flags_limits_and_risk_names() -> None:
    result = apply_ashare_trading_constraints(
        pd.DataFrame(
            [
                _row("2024-01-01", "600000", 10.0, stock_name="normal"),
                _row("2024-01-02", "600000", 11.0, stock_name="normal"),
                _row("2024-01-01", "300001", 10.0, stock_name="normal"),
                _row("2024-01-02", "300001", 12.0, stock_name="normal"),
                _row("2024-01-01", "600001", 10.0, stock_name="ST sample"),
                _row("2024-01-02", "600001", 10.5, stock_name="ST sample"),
                _row("2024-01-01", "600002", 10.0, stock_name="normal"),
                _row("2024-01-02", "600002", 9.0, stock_name="normal"),
                _row("2024-01-02", "600003", 8.0, stock_name="delist risk"),
                _row("2024-01-02", "600004", 8.0, stock_name="退市风险"),
            ]
        )
    )

    latest = result.data.loc[result.data["date"] == pd.Timestamp("2024-01-02")]
    by_symbol = latest.set_index("symbol")

    assert bool(by_symbol.loc["600000", "is_limit_up"])
    assert by_symbol.loc["600000", "limit_threshold_pct"] == pytest.approx(0.10)
    assert bool(by_symbol.loc["300001", "is_limit_up"])
    assert by_symbol.loc["300001", "limit_threshold_pct"] == pytest.approx(0.20)
    assert bool(by_symbol.loc["600001", "is_st"])
    assert by_symbol.loc["600001", "limit_threshold_pct"] == pytest.approx(0.05)
    assert bool(by_symbol.loc["600002", "is_limit_down"])
    assert bool(by_symbol.loc["600003", "is_delisting_risk"])
    assert bool(by_symbol.loc["600004", "is_delisting_risk"])
    assert result.stats["limit_up_rows"] == 3
    assert result.stats["limit_down_rows"] == 1
    assert result.stats["st_rows"] == 2
    assert result.stats["delisting_risk_rows"] == 2


def test_apply_ashare_trading_constraints_filters_configured_hard_rules() -> None:
    result = apply_ashare_trading_constraints(
        pd.DataFrame(
            [
                _row("2024-01-01", "600000", 10.0),
                _row("2024-01-02", "600000", 11.0),
                _row("2024-01-01", "600001", 10.0, stock_name="ST sample"),
                _row("2024-01-02", "600001", 10.1, stock_name="ST sample"),
                _row(
                    "2024-01-02",
                    "600002",
                    9.0,
                    trading_days_since_listing=10,
                ),
                _row(
                    "2024-01-02",
                    "600003",
                    9.0,
                    is_suspended_or_missing=True,
                ),
            ]
        ),
        spec=AshareTradingConstraintSpec(exclude_limit_up=True),
    )

    latest = result.data.loc[result.data["date"] == pd.Timestamp("2024-01-02")]
    eligible_symbols = latest.loc[latest["is_trade_eligible"], "symbol"].tolist()

    assert eligible_symbols == []
    assert result.stats["ineligible_rows"] == 5
    assert "excluded: limit_up" in str(
        latest.set_index("symbol").loc["600000", "trade_constraint_reason"]
    )


def _row(
    current_date: str,
    symbol: str,
    close: float,
    *,
    stock_name: str = "normal",
    trading_days_since_listing: int | None = None,
    is_suspended_or_missing: bool = False,
) -> dict[str, object]:
    return {
        "date": date.fromisoformat(current_date),
        "symbol": symbol,
        "close": close,
        "turnover_rate": 1.0,
        "stock_name": stock_name,
        "trading_days_since_listing": trading_days_since_listing,
        "is_suspended_or_missing": is_suspended_or_missing,
    }
