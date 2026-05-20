from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

TRANSACTION_COST_SCHEMA_VERSION = 1
DEFAULT_COST_PROFILE_NAME = "a_share_retail_default"
DEFAULT_COMMISSION_RATE = 0.0003
DEFAULT_STAMP_DUTY_RATE = 0.0005
DEFAULT_TRANSFER_FEE_RATE = 0.00001
DEFAULT_SLIPPAGE_RATE = 0.0005
MAX_COST_RATE = 0.05


@dataclass(frozen=True, slots=True)
class TransactionCostSpec:
    """Return-level transaction cost assumptions for factor backtests."""

    enabled: bool = True
    profile_name: str = DEFAULT_COST_PROFILE_NAME
    commission_rate: float = DEFAULT_COMMISSION_RATE
    stamp_duty_rate: float = DEFAULT_STAMP_DUTY_RATE
    transfer_fee_rate: float = DEFAULT_TRANSFER_FEE_RATE
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE

    def __post_init__(self) -> None:
        if not self.profile_name.strip():
            msg = "transaction_costs.profile_name must be non-empty."
            raise ValueError(msg)
        for field_name, value in (
            ("commission_rate", self.commission_rate),
            ("stamp_duty_rate", self.stamp_duty_rate),
            ("transfer_fee_rate", self.transfer_fee_rate),
            ("slippage_rate", self.slippage_rate),
        ):
            if value < 0.0 or value > MAX_COST_RATE:
                msg = f"transaction_costs.{field_name} must be between 0 and {MAX_COST_RATE}."
                raise ValueError(msg)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> TransactionCostSpec:
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            msg = "transaction_costs must be a mapping when provided."
            raise ValueError(msg)
        return cls(
            enabled=_optional_bool(payload, "enabled", True),
            profile_name=_optional_str(
                payload,
                "profile_name",
                DEFAULT_COST_PROFILE_NAME,
            ),
            commission_rate=_optional_float(
                payload,
                "commission_rate",
                DEFAULT_COMMISSION_RATE,
            ),
            stamp_duty_rate=_optional_float(
                payload,
                "stamp_duty_rate",
                DEFAULT_STAMP_DUTY_RATE,
            ),
            transfer_fee_rate=_optional_float(
                payload,
                "transfer_fee_rate",
                DEFAULT_TRANSFER_FEE_RATE,
            ),
            slippage_rate=_optional_float(
                payload,
                "slippage_rate",
                DEFAULT_SLIPPAGE_RATE,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TRANSACTION_COST_SCHEMA_VERSION,
            "enabled": self.enabled,
            "profile_name": self.profile_name,
            "commission_rate": self.commission_rate,
            "stamp_duty_rate": self.stamp_duty_rate,
            "transfer_fee_rate": self.transfer_fee_rate,
            "slippage_rate": self.slippage_rate,
            "buy_cost_rate": self.buy_cost_rate,
            "sell_cost_rate": self.sell_cost_rate,
        }

    @property
    def buy_cost_rate(self) -> float:
        if not self.enabled:
            return 0.0
        return self.commission_rate + self.transfer_fee_rate + self.slippage_rate

    @property
    def sell_cost_rate(self) -> float:
        if not self.enabled:
            return 0.0
        return (
            self.commission_rate
            + self.transfer_fee_rate
            + self.slippage_rate
            + self.stamp_duty_rate
        )


@dataclass(frozen=True, slots=True)
class TurnoverBreakdown:
    """Portfolio turnover implied by equal-weight basket rebalancing."""

    buy_turnover: float
    sell_turnover: float

    @property
    def total_turnover(self) -> float:
        return self.buy_turnover + self.sell_turnover


def equal_weight_positions(symbols: list[str]) -> dict[str, float]:
    normalized = [symbol.strip() for symbol in symbols if symbol.strip()]
    normalized = list(dict.fromkeys(normalized))
    if not normalized:
        return {}
    weight = 1.0 / len(normalized)
    return {symbol: weight for symbol in normalized}


def compute_turnover(
    previous_weights: Mapping[str, float],
    current_weights: Mapping[str, float],
) -> TurnoverBreakdown:
    symbols = set(previous_weights) | set(current_weights)
    buy_turnover = 0.0
    sell_turnover = 0.0
    for symbol in symbols:
        diff = float(current_weights.get(symbol, 0.0)) - float(
            previous_weights.get(symbol, 0.0)
        )
        if diff > 0.0:
            buy_turnover += diff
        elif diff < 0.0:
            sell_turnover += abs(diff)
    return TurnoverBreakdown(
        buy_turnover=buy_turnover,
        sell_turnover=sell_turnover,
    )


def estimate_transaction_cost(
    turnover: TurnoverBreakdown,
    spec: TransactionCostSpec,
) -> float:
    return (
        turnover.buy_turnover * spec.buy_cost_rate
        + turnover.sell_turnover * spec.sell_cost_rate
    )


def _optional_bool(payload: Mapping[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    msg = f"transaction_costs.{key} must be a boolean."
    raise ValueError(msg)


def _optional_str(payload: Mapping[str, Any], key: str, default: str) -> str:
    value = payload.get(key, default)
    if isinstance(value, str) and value.strip():
        return value.strip()
    msg = f"transaction_costs.{key} must be a non-empty string."
    raise ValueError(msg)


def _optional_float(payload: Mapping[str, Any], key: str, default: float) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"transaction_costs.{key} must be a number."
        raise ValueError(msg)
    return float(value)
