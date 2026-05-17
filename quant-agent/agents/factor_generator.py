from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from agents.factor_templates import FactorDirection
from core.logging import AgentLoggerAdapter, get_agent_logger
from core.models import AgentRequest, AgentResponse

DEFAULT_FACTOR_COUNT = 50
MAX_FACTOR_COUNT = 200
GENERATION_METHOD = "deterministic_factor_family_v1"
FORBIDDEN_FACTOR_TOKENS = ("future_", "lead(", "shift(-")


@dataclass(frozen=True, slots=True)
class FactorGenerationSpec:
    """Validated request for generating symbolic candidate factors."""

    target_count: int = DEFAULT_FACTOR_COUNT
    source_template_ids: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> FactorGenerationSpec:
        return cls(
            target_count=_optional_int(
                payload,
                "target_count",
                DEFAULT_FACTOR_COUNT,
                minimum=1,
                maximum=MAX_FACTOR_COUNT,
            ),
            source_template_ids=_optional_str_sequence(payload, "source_template_ids"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_count": self.target_count,
            "source_template_ids": list(self.source_template_ids),
        }


@dataclass(frozen=True, slots=True)
class FactorFamily:
    """A deterministic family that expands into related symbolic factors."""

    family_id: str
    source_template_id: str
    category: str
    direction: FactorDirection
    required_columns: tuple[str, ...]
    risk_flags: tuple[str, ...]
    name_template: str
    expression_template: str
    parameter_grid: tuple[Mapping[str, int], ...]

    def instantiate(self, parameters: Mapping[str, int]) -> GeneratedFactorSeed:
        formatted_parameters = dict(parameters)
        expression = self.expression_template.format(**formatted_parameters)
        _validate_expression(expression)
        return GeneratedFactorSeed(
            family_id=self.family_id,
            source_template_id=self.source_template_id,
            name=self.name_template.format(**formatted_parameters),
            category=self.category,
            expression=expression,
            direction=self.direction,
            required_columns=self.required_columns,
            parameters=formatted_parameters,
            lookback_days=max(formatted_parameters.values()),
            signal_tags=(self.family_id, self.source_template_id),
            risk_flags=self.risk_flags,
        )


@dataclass(frozen=True, slots=True)
class GeneratedFactorSeed:
    """Generated symbolic factor before sequential alpha id assignment."""

    family_id: str
    source_template_id: str
    name: str
    category: str
    expression: str
    direction: FactorDirection
    required_columns: tuple[str, ...]
    parameters: Mapping[str, int]
    lookback_days: int
    signal_tags: tuple[str, ...]
    risk_flags: tuple[str, ...]

    def with_factor_id(self, factor_id: str) -> GeneratedFactor:
        return GeneratedFactor(
            factor_id=factor_id,
            family_id=self.family_id,
            source_template_id=self.source_template_id,
            name=self.name,
            category=self.category,
            expression=self.expression,
            direction=self.direction,
            required_columns=self.required_columns,
            parameters=self.parameters,
            lookback_days=self.lookback_days,
            signal_tags=self.signal_tags,
            risk_flags=self.risk_flags,
            generation_method=GENERATION_METHOD,
        )


@dataclass(frozen=True, slots=True)
class GeneratedFactor:
    """Generated symbolic factor definition."""

    factor_id: str
    family_id: str
    source_template_id: str
    name: str
    category: str
    expression: str
    direction: FactorDirection
    required_columns: tuple[str, ...]
    parameters: Mapping[str, int]
    lookback_days: int
    signal_tags: tuple[str, ...]
    risk_flags: tuple[str, ...]
    generation_method: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "family_id": self.family_id,
            "source_template_id": self.source_template_id,
            "name": self.name,
            "category": self.category,
            "expression": self.expression,
            "direction": self.direction,
            "required_columns": list(self.required_columns),
            "parameters": dict(self.parameters),
            "lookback_days": self.lookback_days,
            "signal_tags": list(self.signal_tags),
            "risk_flags": list(self.risk_flags),
            "generation_method": self.generation_method,
        }


@dataclass(frozen=True, slots=True)
class FactorBatchResult:
    """Generated factor batch and summary statistics."""

    factors: tuple[GeneratedFactor, ...]
    stats: dict[str, Any]


class FactorCandidateGenerator:
    """Generate deterministic candidate factors from formula families."""

    def __init__(self, families: Sequence[FactorFamily] | None = None) -> None:
        self.families = tuple(families or DEFAULT_FACTOR_FAMILIES)
        self._validate_families()

    def generate(self, spec: FactorGenerationSpec) -> FactorBatchResult:
        families = self._selected_families(spec.source_template_ids)
        seeds = [family.instantiate(parameters) for family in families for parameters in family.parameter_grid]
        if len(seeds) < spec.target_count:
            msg = (
                f"Only {len(seeds)} candidate factors are available for the request; "
                f"target_count={spec.target_count}."
            )
            raise ValueError(msg)

        factors = tuple(
            seed.with_factor_id(f"alpha_{index:03d}")
            for index, seed in enumerate(seeds[: spec.target_count], start=1)
        )
        return FactorBatchResult(
            factors=factors,
            stats=_generation_stats(factors, available_seed_count=len(seeds)),
        )

    def available_source_template_ids(self) -> list[str]:
        return sorted({family.source_template_id for family in self.families})

    def _selected_families(self, source_template_ids: Sequence[str]) -> tuple[FactorFamily, ...]:
        if not source_template_ids:
            return self.families

        requested = tuple(dict.fromkeys(source_template_ids))
        available = set(self.available_source_template_ids())
        unknown = [source_id for source_id in requested if source_id not in available]
        if unknown:
            msg = f"Unknown source_template_ids: {', '.join(unknown)}."
            raise ValueError(msg)

        requested_set = set(requested)
        return tuple(
            family
            for family in self.families
            if family.source_template_id in requested_set
        )

    def _validate_families(self) -> None:
        if not self.families:
            msg = "At least one factor family is required."
            raise ValueError(msg)
        family_ids = [family.family_id for family in self.families]
        duplicates = [family_id for family_id, count in Counter(family_ids).items() if count > 1]
        if duplicates:
            msg = f"Duplicate factor family ids: {', '.join(duplicates)}."
            raise ValueError(msg)


class FactorGenerationAgent:
    """Generate the first deterministic batch of symbolic factors."""

    name = "FactorGenerationAgent"

    def __init__(
        self,
        *,
        logger: AgentLoggerAdapter | None = None,
        generator: FactorCandidateGenerator | None = None,
    ) -> None:
        self.logger = logger or get_agent_logger(self.name)
        self.generator = generator or FactorCandidateGenerator()

    def run(self, request: AgentRequest) -> AgentResponse:
        started_at = perf_counter()
        self.logger.info(
            "Received factor generation request.",
            extra={"action": "validate_request", "status": "running"},
        )

        try:
            spec = FactorGenerationSpec.from_payload(request.payload)
            self.logger.info(
                "Generating factor candidates.",
                extra={"action": "generate_factors", "status": "running"},
            )
            result = self.generator.generate(spec)
        except ValueError as exc:
            elapsed = perf_counter() - started_at
            self.logger.warning(
                "Factor generation failed.",
                extra={"action": "generate_factors", "status": "error"},
            )
            return AgentResponse.failure(
                str(exc),
                metadata=self._metadata(request, elapsed),
            )

        elapsed = perf_counter() - started_at
        self.logger.info(
            "Generated factor candidates.",
            extra={"action": "generate_factors", "status": "success"},
        )
        factors = [factor.to_dict() for factor in result.factors]
        return AgentResponse.success(
            output={
                "state": "factors_generated",
                "request": spec.to_dict(),
                "factor_count": len(factors),
                "generation_method": GENERATION_METHOD,
                "factors": factors,
                "generation_stats": result.stats,
                "next_action": "Add ranking transforms in Day 12.",
            },
            metadata=self._metadata(
                request,
                elapsed,
                factor_count=len(factors),
                source_template_count=len(result.stats["source_template_counts"]),
                max_lookback_days=result.stats["max_lookback_days"],
            ),
        )

    def _metadata(
        self,
        request: AgentRequest,
        elapsed: float,
        *,
        factor_count: int | None = None,
        source_template_count: int | None = None,
        max_lookback_days: int | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "agent": self.name,
            "task_id": request.task_id,
            "execution_time_sec": round(elapsed, 6),
        }
        if factor_count is not None:
            metadata["factor_count"] = factor_count
        if source_template_count is not None:
            metadata["source_template_count"] = source_template_count
        if max_lookback_days is not None:
            metadata["max_lookback_days"] = max_lookback_days
        return metadata


def _window_grid(*windows: int) -> tuple[Mapping[str, int], ...]:
    return tuple({"window": window} for window in windows)


def _fast_slow_grid(*pairs: tuple[int, int]) -> tuple[Mapping[str, int], ...]:
    return tuple(
        {"fast_window": fast_window, "slow_window": slow_window}
        for fast_window, slow_window in pairs
    )


def _validate_expression(expression: str) -> None:
    compact_expression = expression.lower().replace(" ", "")
    for token in FORBIDDEN_FACTOR_TOKENS:
        if token in compact_expression:
            msg = f"Forbidden future-looking token in expression: {token}."
            raise ValueError(msg)


DEFAULT_FACTOR_FAMILIES = (
    FactorFamily(
        family_id="momentum_return",
        source_template_id="return_5d",
        category="momentum",
        direction="positive",
        required_columns=("close",),
        risk_flags=("momentum_crowding", "reversal_risk"),
        name_template="{window}D Momentum Return",
        expression_template="close / delay(close, {window}) - 1",
        parameter_grid=_window_grid(1, 2, 3, 5, 10, 20),
    ),
    FactorFamily(
        family_id="reversal_return",
        source_template_id="return_3d",
        category="reversal",
        direction="positive",
        required_columns=("close",),
        risk_flags=("falling_knife", "event_risk"),
        name_template="{window}D Reversal Return",
        expression_template="-1 * (close / delay(close, {window}) - 1)",
        parameter_grid=_window_grid(1, 2, 3, 5, 10),
    ),
    FactorFamily(
        family_id="amount_growth",
        source_template_id="amount_growth_5d",
        category="liquidity",
        direction="positive",
        required_columns=("amount",),
        risk_flags=("liquidity_bias", "event_volume_noise"),
        name_template="{window}D Amount Growth",
        expression_template="pct_change(amount, window={window})",
        parameter_grid=_window_grid(1, 3, 5, 10, 20),
    ),
    FactorFamily(
        family_id="turnover_rate_change",
        source_template_id="turnover_rate_change_5d",
        category="liquidity",
        direction="positive",
        required_columns=("turnover_rate",),
        risk_flags=("liquidity_bias", "crowded_flow"),
        name_template="{window}D Turnover Change",
        expression_template="turnover_rate - delay(turnover_rate, {window})",
        parameter_grid=_window_grid(1, 3, 5, 20),
    ),
    FactorFamily(
        family_id="volume_ratio",
        source_template_id="volume_ratio_5d_20d",
        category="liquidity",
        direction="positive",
        required_columns=("volume",),
        risk_flags=("volume_spike_noise", "limit_up_bias"),
        name_template="{fast_window}D To {slow_window}D Volume Ratio",
        expression_template="mean(volume, {fast_window}) / mean(volume, {slow_window})",
        parameter_grid=_fast_slow_grid((1, 5), (3, 10), (5, 20), (10, 40), (20, 60)),
    ),
    FactorFamily(
        family_id="realized_volatility",
        source_template_id="realized_volatility_20d",
        category="volatility",
        direction="negative",
        required_columns=("close",),
        risk_flags=("volatility_regime_dependence", "low_volatility_trap"),
        name_template="{window}D Realized Volatility",
        expression_template="std(close / delay(close, 1) - 1, {window})",
        parameter_grid=_window_grid(5, 10, 20, 40, 60),
    ),
    FactorFamily(
        family_id="high_breakout",
        source_template_id="high_breakout_20d",
        category="breakout",
        direction="positive",
        required_columns=("close", "high"),
        risk_flags=("false_breakout", "limit_up_bias"),
        name_template="{window}D High Breakout",
        expression_template="close / max(delay(high, 1), {window}) - 1",
        parameter_grid=_window_grid(10, 20, 40, 60),
    ),
    FactorFamily(
        family_id="max_drawdown",
        source_template_id="max_drawdown_20d",
        category="risk_adjusted_momentum",
        direction="negative",
        required_columns=("close",),
        risk_flags=("momentum_crash", "drawdown_clustering"),
        name_template="{window}D Max Drawdown",
        expression_template="max_drawdown(close, window={window})",
        parameter_grid=_window_grid(10, 20, 40, 60),
    ),
    FactorFamily(
        family_id="positive_return_days",
        source_template_id="positive_return_days_20d",
        category="risk_adjusted_momentum",
        direction="positive",
        required_columns=("close",),
        risk_flags=("market_regime_dependence",),
        name_template="{window}D Positive Return Day Ratio",
        expression_template="mean((close / delay(close, 1) - 1) > 0, {window})",
        parameter_grid=_window_grid(5, 10, 20, 40),
    ),
    FactorFamily(
        family_id="intraday_range",
        source_template_id="intraday_range_5d",
        category="volatility",
        direction="negative",
        required_columns=("high", "low", "close"),
        risk_flags=("volatility_regime_dependence",),
        name_template="{window}D Intraday Range",
        expression_template="mean((high - low) / close, {window})",
        parameter_grid=_window_grid(3, 5, 10),
    ),
    FactorFamily(
        family_id="turnover_rate_zscore",
        source_template_id="turnover_rate_zscore_20d",
        category="liquidity",
        direction="neutral",
        required_columns=("turnover_rate",),
        risk_flags=("liquidity_bias", "crowded_flow"),
        name_template="{window}D Turnover Z-Score",
        expression_template="zscore(turnover_rate, window={window})",
        parameter_grid=_window_grid(10, 20, 40),
    ),
    FactorFamily(
        family_id="amount_zscore",
        source_template_id="amount_zscore_20d",
        category="liquidity",
        direction="neutral",
        required_columns=("amount",),
        risk_flags=("block_trade_noise", "liquidity_bias"),
        name_template="{window}D Amount Z-Score",
        expression_template="zscore(amount, window={window})",
        parameter_grid=_window_grid(10, 20),
    ),
)


def _generation_stats(
    factors: Sequence[GeneratedFactor],
    *,
    available_seed_count: int,
) -> dict[str, Any]:
    category_counts = Counter(factor.category for factor in factors)
    source_template_counts = Counter(factor.source_template_id for factor in factors)
    return {
        "available_seed_count": available_seed_count,
        "category_counts": dict(sorted(category_counts.items())),
        "source_template_counts": dict(sorted(source_template_counts.items())),
        "unique_expression_count": len({factor.expression for factor in factors}),
        "max_lookback_days": max(factor.lookback_days for factor in factors),
    }


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
