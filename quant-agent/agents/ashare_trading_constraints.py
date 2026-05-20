from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

A_SHARE_CONSTRAINT_SCHEMA_VERSION = 1
DEFAULT_NEW_STOCK_MIN_TRADING_DAYS = 60
DEFAULT_LIMIT_PRICE_TOLERANCE = 0.001

LIMIT_10_PCT = 0.10
LIMIT_ST_PCT = 0.05
LIMIT_STAR_CHINEXT_PCT = 0.20
LIMIT_BEIJING_PCT = 0.30

CONSTRAINT_COLUMNS = (
    "previous_close",
    "limit_threshold_pct",
    "upper_limit_price",
    "lower_limit_price",
    "is_limit_up",
    "is_limit_down",
    "is_st",
    "is_new_stock",
    "is_delisting_risk",
    "trading_days_since_listing",
    "is_t_plus_one",
    "is_trade_eligible",
    "trade_constraint_reason",
)

NAME_COLUMNS = ("name", "stock_name", "security_name", "display_name")
ST_COLUMNS = ("is_st", "st", "special_treatment")
DELISTING_COLUMNS = ("is_delisting_risk", "delisting_risk", "is_delisting")
LISTING_DATE_COLUMNS = ("listing_date", "ipo_date", "list_date")
LISTING_AGE_COLUMNS = ("trading_days_since_listing", "listed_trading_days")


@dataclass(frozen=True, slots=True)
class AshareTradingConstraintSpec:
    """Configurable A-share trading-constraint policy for daily research."""

    t_plus_one: bool = True
    exclude_suspended: bool = True
    exclude_limit_up: bool = False
    exclude_limit_down: bool = False
    exclude_st: bool = True
    exclude_new_stock: bool = True
    exclude_delisting_risk: bool = True
    new_stock_min_trading_days: int = DEFAULT_NEW_STOCK_MIN_TRADING_DAYS
    limit_price_tolerance: float = DEFAULT_LIMIT_PRICE_TOLERANCE

    def __post_init__(self) -> None:
        if self.new_stock_min_trading_days < 0 or self.new_stock_min_trading_days > 1000:
            msg = "new_stock_min_trading_days must be between 0 and 1000."
            raise ValueError(msg)
        if self.limit_price_tolerance < 0 or self.limit_price_tolerance > 0.05:
            msg = "limit_price_tolerance must be between 0 and 0.05."
            raise ValueError(msg)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any] | None,
    ) -> AshareTradingConstraintSpec:
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            msg = "trading_constraints must be a mapping when provided."
            raise ValueError(msg)
        return cls(
            t_plus_one=_optional_bool(payload, "t_plus_one", True),
            exclude_suspended=_optional_bool(payload, "exclude_suspended", True),
            exclude_limit_up=_optional_bool(payload, "exclude_limit_up", False),
            exclude_limit_down=_optional_bool(payload, "exclude_limit_down", False),
            exclude_st=_optional_bool(payload, "exclude_st", True),
            exclude_new_stock=_optional_bool(payload, "exclude_new_stock", True),
            exclude_delisting_risk=_optional_bool(
                payload,
                "exclude_delisting_risk",
                True,
            ),
            new_stock_min_trading_days=_optional_int(
                payload,
                "new_stock_min_trading_days",
                DEFAULT_NEW_STOCK_MIN_TRADING_DAYS,
            ),
            limit_price_tolerance=_optional_float(
                payload,
                "limit_price_tolerance",
                DEFAULT_LIMIT_PRICE_TOLERANCE,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": A_SHARE_CONSTRAINT_SCHEMA_VERSION,
            "t_plus_one": self.t_plus_one,
            "exclude_suspended": self.exclude_suspended,
            "exclude_limit_up": self.exclude_limit_up,
            "exclude_limit_down": self.exclude_limit_down,
            "exclude_st": self.exclude_st,
            "exclude_new_stock": self.exclude_new_stock,
            "exclude_delisting_risk": self.exclude_delisting_risk,
            "new_stock_min_trading_days": self.new_stock_min_trading_days,
            "limit_price_tolerance": self.limit_price_tolerance,
        }


@dataclass(frozen=True, slots=True)
class AshareTradingConstraintResult:
    """A-share trading-constraint flags and summary statistics."""

    data: pd.DataFrame
    stats: dict[str, Any]


def apply_ashare_trading_constraints(
    frame: pd.DataFrame,
    *,
    spec: AshareTradingConstraintSpec | None = None,
) -> AshareTradingConstraintResult:
    """Add A-share trading-constraint flags to an OHLCV-like panel."""

    policy = spec or AshareTradingConstraintSpec()
    constrained = _normalize_frame(frame)

    close = pd.to_numeric(constrained["close"], errors="coerce")
    previous_close = close.groupby(constrained["symbol"], sort=False).shift(1)
    is_suspended = _suspended_mask(constrained)
    is_st = _st_mask(constrained)
    is_delisting = _delisting_risk_mask(constrained)
    trading_days_since_listing = _trading_days_since_listing(constrained)
    is_new_stock = (
        trading_days_since_listing.notna()
        & (trading_days_since_listing < policy.new_stock_min_trading_days)
    )

    limit_threshold = _limit_thresholds(constrained["symbol"], is_st)
    upper_limit = previous_close * (1.0 + limit_threshold)
    lower_limit = previous_close * (1.0 - limit_threshold)
    tolerance = previous_close.abs() * policy.limit_price_tolerance
    valid_limit_base = previous_close.notna() & close.notna() & (previous_close > 0)
    is_limit_up = valid_limit_base & (close >= upper_limit - tolerance)
    is_limit_down = valid_limit_base & (close <= lower_limit + tolerance)

    constrained["is_suspended_or_missing"] = is_suspended.astype(bool)
    constrained["previous_close"] = previous_close
    constrained["limit_threshold_pct"] = limit_threshold
    constrained["upper_limit_price"] = upper_limit
    constrained["lower_limit_price"] = lower_limit
    constrained["is_limit_up"] = is_limit_up.astype(bool)
    constrained["is_limit_down"] = is_limit_down.astype(bool)
    constrained["is_st"] = is_st.astype(bool)
    constrained["is_new_stock"] = is_new_stock.fillna(False).astype(bool)
    constrained["is_delisting_risk"] = is_delisting.astype(bool)
    constrained["trading_days_since_listing"] = trading_days_since_listing
    constrained["is_t_plus_one"] = bool(policy.t_plus_one)

    ineligible = pd.Series(False, index=constrained.index)
    if policy.exclude_suspended:
        ineligible |= constrained["is_suspended_or_missing"]
    if policy.exclude_limit_up:
        ineligible |= constrained["is_limit_up"]
    if policy.exclude_limit_down:
        ineligible |= constrained["is_limit_down"]
    if policy.exclude_st:
        ineligible |= constrained["is_st"]
    if policy.exclude_new_stock:
        ineligible |= constrained["is_new_stock"]
    if policy.exclude_delisting_risk:
        ineligible |= constrained["is_delisting_risk"]

    constrained["is_trade_eligible"] = ~ineligible
    constrained["trade_constraint_reason"] = [
        _constraint_reason(row, policy)
        for _, row in constrained.iterrows()
    ]

    return AshareTradingConstraintResult(
        data=constrained,
        stats=_constraint_stats(constrained, policy),
    )


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in ("date", "symbol", "close") if column not in frame.columns]
    if missing_columns:
        msg = f"A-share constraint frame is missing columns: {', '.join(missing_columns)}."
        raise ValueError(msg)

    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["symbol"] = normalized["symbol"].astype(str).str.strip().str.zfill(6)
    if "is_suspended_or_missing" in normalized.columns:
        normalized["is_suspended_or_missing"] = _bool_series(
            normalized["is_suspended_or_missing"],
        )
    else:
        normalized["is_suspended_or_missing"] = pd.to_numeric(
            normalized["close"],
            errors="coerce",
        ).isna()

    numeric_columns = [
        column
        for column in (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "turnover_rate",
            *LISTING_AGE_COLUMNS,
        )
        if column in normalized.columns
    ]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    if normalized["date"].isna().any():
        msg = "A-share constraint frame contains invalid dates."
        raise ValueError(msg)
    return normalized.sort_values(["symbol", "date"]).reset_index(drop=True)


def _suspended_mask(frame: pd.DataFrame) -> pd.Series:
    close = pd.to_numeric(frame["close"], errors="coerce")
    base = frame["is_suspended_or_missing"] | close.isna()
    if "volume" in frame.columns:
        volume = pd.to_numeric(frame["volume"], errors="coerce")
        base |= volume.fillna(0) <= 0
    if "amount" in frame.columns:
        amount = pd.to_numeric(frame["amount"], errors="coerce")
        base |= amount.fillna(0) <= 0
    return base.astype(bool)


def _st_mask(frame: pd.DataFrame) -> pd.Series:
    result = _any_bool_column(frame, ST_COLUMNS)
    names = _combined_name_series(frame)
    if names is not None:
        upper_names = names.str.upper()
        result |= upper_names.str.startswith("ST", na=False)
        result |= upper_names.str.startswith("*ST", na=False)
        result |= upper_names.str.contains(" ST", na=False)
    return result.astype(bool)


def _delisting_risk_mask(frame: pd.DataFrame) -> pd.Series:
    result = _any_bool_column(frame, DELISTING_COLUMNS)
    names = _combined_name_series(frame)
    if names is not None:
        lower_names = names.str.lower()
        result |= names.str.contains("退", na=False)
        result |= lower_names.str.contains("delist", na=False)
    return result.astype(bool)


def _trading_days_since_listing(frame: pd.DataFrame) -> pd.Series:
    for column in LISTING_AGE_COLUMNS:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")

    for column in LISTING_DATE_COLUMNS:
        if column in frame.columns:
            listing_date = pd.to_datetime(frame[column], errors="coerce")
            calendar_days = (frame["date"] - listing_date).dt.days
            estimated_trading_days = np.floor(calendar_days * 5.0 / 7.0)
            return pd.Series(estimated_trading_days, index=frame.index)

    return pd.Series(np.nan, index=frame.index)


def _limit_thresholds(symbols: pd.Series, is_st: pd.Series) -> pd.Series:
    thresholds = []
    for symbol, st_flag in zip(symbols.astype(str), is_st, strict=True):
        if bool(st_flag):
            thresholds.append(LIMIT_ST_PCT)
        elif symbol.startswith(("300", "301", "688", "689")):
            thresholds.append(LIMIT_STAR_CHINEXT_PCT)
        elif symbol.startswith(("4", "8")):
            thresholds.append(LIMIT_BEIJING_PCT)
        else:
            thresholds.append(LIMIT_10_PCT)
    return pd.Series(thresholds, index=symbols.index, dtype="float64")


def _constraint_reason(row: pd.Series, policy: AshareTradingConstraintSpec) -> str:
    exclusions: list[str] = []
    flags: list[str] = []
    if bool(row["is_suspended_or_missing"]):
        (exclusions if policy.exclude_suspended else flags).append("suspended_or_missing")
    if bool(row["is_limit_up"]):
        (exclusions if policy.exclude_limit_up else flags).append("limit_up")
    if bool(row["is_limit_down"]):
        (exclusions if policy.exclude_limit_down else flags).append("limit_down")
    if bool(row["is_st"]):
        (exclusions if policy.exclude_st else flags).append("st")
    if bool(row["is_new_stock"]):
        (exclusions if policy.exclude_new_stock else flags).append("new_stock")
    if bool(row["is_delisting_risk"]):
        (
            exclusions
            if policy.exclude_delisting_risk
            else flags
        ).append("delisting_risk")

    notes = ["t_plus_one"] if policy.t_plus_one else []
    if exclusions:
        return "excluded: " + ", ".join([*exclusions, *notes])
    if flags:
        return "eligible_with_flags: " + ", ".join([*flags, *notes])
    return "eligible: " + ", ".join(notes) if notes else "eligible"


def _constraint_stats(
    frame: pd.DataFrame,
    policy: AshareTradingConstraintSpec,
) -> dict[str, Any]:
    return {
        "schema_version": A_SHARE_CONSTRAINT_SCHEMA_VERSION,
        "policy": policy.to_dict(),
        "input_rows": int(len(frame)),
        "eligible_rows": int(frame["is_trade_eligible"].sum()),
        "ineligible_rows": int((~frame["is_trade_eligible"]).sum()),
        "suspended_or_missing_rows": int(frame["is_suspended_or_missing"].sum()),
        "limit_up_rows": int(frame["is_limit_up"].sum()),
        "limit_down_rows": int(frame["is_limit_down"].sum()),
        "st_rows": int(frame["is_st"].sum()),
        "new_stock_rows": int(frame["is_new_stock"].sum()),
        "delisting_risk_rows": int(frame["is_delisting_risk"].sum()),
        "t_plus_one_assumption": policy.t_plus_one,
    }


def _combined_name_series(frame: pd.DataFrame) -> pd.Series | None:
    available = [column for column in NAME_COLUMNS if column in frame.columns]
    if not available:
        return None
    names = pd.Series("", index=frame.index, dtype="string")
    for column in available:
        values = frame[column].astype("string").fillna("")
        names = names.mask(names.str.len() == 0, values)
    return names.fillna("")


def _any_bool_column(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    result = pd.Series(False, index=frame.index)
    for column in columns:
        if column in frame.columns:
            result |= _bool_series(frame[column])
    return result.astype(bool)


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    truthy = normalized.isin({"1", "true", "t", "yes", "y"})
    numeric = pd.to_numeric(series, errors="coerce")
    return (truthy | (numeric.fillna(0) != 0)).fillna(False).astype(bool)


def _optional_bool(payload: Mapping[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    msg = f"trading_constraints.{key} must be a boolean."
    raise ValueError(msg)


def _optional_int(payload: Mapping[str, Any], key: str, default: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"trading_constraints.{key} must be an integer."
        raise ValueError(msg)
    return value


def _optional_float(payload: Mapping[str, Any], key: str, default: float) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"trading_constraints.{key} must be a number."
        raise ValueError(msg)
    return float(value)
