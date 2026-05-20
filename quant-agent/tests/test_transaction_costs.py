from __future__ import annotations

import pytest

from agents.transaction_costs import (
    TransactionCostSpec,
    compute_turnover,
    equal_weight_positions,
    estimate_transaction_cost,
)


def test_transaction_cost_spec_defaults_to_a_share_retail_profile() -> None:
    spec = TransactionCostSpec()

    assert spec.enabled is True
    assert spec.profile_name == "a_share_retail_default"
    assert spec.buy_cost_rate == pytest.approx(0.00081)
    assert spec.sell_cost_rate == pytest.approx(0.00131)


def test_transaction_cost_spec_can_be_disabled() -> None:
    spec = TransactionCostSpec.from_mapping({"enabled": False})
    turnover = compute_turnover({}, {"000001": 1.0})

    assert spec.buy_cost_rate == 0.0
    assert spec.sell_cost_rate == 0.0
    assert estimate_transaction_cost(turnover, spec) == 0.0


def test_compute_turnover_splits_buy_and_sell_notional() -> None:
    previous = {"000001": 0.5, "000002": 0.5}
    current = {"000002": 0.25, "000003": 0.75}

    turnover = compute_turnover(previous, current)

    assert turnover.buy_turnover == pytest.approx(0.75)
    assert turnover.sell_turnover == pytest.approx(0.75)
    assert turnover.total_turnover == pytest.approx(1.5)


def test_estimate_transaction_cost_uses_buy_and_sell_rates() -> None:
    spec = TransactionCostSpec(
        commission_rate=0.001,
        stamp_duty_rate=0.002,
        transfer_fee_rate=0.003,
        slippage_rate=0.004,
    )
    turnover = compute_turnover(
        previous_weights={"000001": 1.0},
        current_weights={"000002": 1.0},
    )

    assert estimate_transaction_cost(turnover, spec) == pytest.approx(0.018)


def test_equal_weight_positions_normalizes_symbols() -> None:
    assert equal_weight_positions(["000001", "000002", "000001", " "]) == {
        "000001": 0.5,
        "000002": 0.5,
    }
