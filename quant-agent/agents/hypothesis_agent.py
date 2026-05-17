from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from core.logging import AgentLoggerAdapter, get_agent_logger
from core.models import AgentRequest, AgentResponse

SUPPORTED_HORIZONS = {"short_term", "medium_term", "long_term"}
DEFAULT_OBJECTIVE = "Find robust alpha hypotheses for cross-sectional equity selection."
GENERATION_METHOD = "deterministic_template_v1"

_HORIZON_ALIASES = {
    "short": "short_term",
    "short-term": "short_term",
    "short_term": "short_term",
    "1d": "short_term",
    "5d": "short_term",
    "medium": "medium_term",
    "medium-term": "medium_term",
    "medium_term": "medium_term",
    "20d": "medium_term",
    "long": "long_term",
    "long-term": "long_term",
    "long_term": "long_term",
    "60d": "long_term",
}


@dataclass(frozen=True, slots=True)
class HypothesisSpec:
    """Validated request for alpha hypothesis generation."""

    objective: str = DEFAULT_OBJECTIVE
    market: str = "a_share"
    universe: str = "CSI500"
    horizon: str = "short_term"
    max_hypotheses: int = 5
    constraints: tuple[str, ...] = ()
    data_context: dict[str, Any] | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> HypothesisSpec:
        objective = _optional_str(payload, "objective", DEFAULT_OBJECTIVE)
        market = _optional_str(payload, "market", "a_share").lower()
        universe = _optional_str(payload, "universe", "CSI500")
        horizon = _normalize_horizon(_optional_str(payload, "horizon", "short_term"))
        max_hypotheses = _optional_int(payload, "max_hypotheses", 5, minimum=1, maximum=20)
        constraints = _optional_str_sequence(payload, "constraints")
        data_context = _optional_mapping(payload, "data_context")

        if horizon not in SUPPORTED_HORIZONS:
            msg = f"Unsupported horizon: {horizon}."
            raise ValueError(msg)

        return cls(
            objective=objective,
            market=market,
            universe=universe,
            horizon=horizon,
            max_hypotheses=max_hypotheses,
            constraints=constraints,
            data_context=data_context,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "market": self.market,
            "universe": self.universe,
            "horizon": self.horizon,
            "max_hypotheses": self.max_hypotheses,
            "constraints": list(self.constraints),
            "data_context": dict(self.data_context or {}),
        }


@dataclass(frozen=True, slots=True)
class HypothesisTemplate:
    """Reusable seed for a testable alpha hypothesis."""

    template_id: str
    title: str
    description: str
    rationale: str
    candidate_signals: tuple[str, ...]
    expected_direction: str
    required_data: tuple[str, ...]
    risk_flags: tuple[str, ...]
    test_plan: tuple[str, ...]

    def instantiate(self, spec: HypothesisSpec, index: int) -> dict[str, Any]:
        return {
            "hypothesis_id": f"HYP-{index:03d}-{self.template_id}",
            "title": self.title,
            "description": self.description.format(
                market=spec.market,
                universe=spec.universe,
                horizon=spec.horizon,
            ),
            "rationale": self.rationale,
            "horizon": spec.horizon,
            "candidate_signals": list(self.candidate_signals),
            "expected_direction": self.expected_direction,
            "required_data": list(self.required_data),
            "risk_flags": list(self.risk_flags),
            "test_plan": list(self.test_plan),
            "source": GENERATION_METHOD,
        }


class HypothesisAgent:
    """Generate structured, testable alpha hypotheses."""

    name = "HypothesisAgent"

    def __init__(
        self,
        *,
        logger: AgentLoggerAdapter | None = None,
        templates: Sequence[HypothesisTemplate] | None = None,
    ) -> None:
        self.logger = logger or get_agent_logger(self.name)
        self.templates = tuple(templates or DEFAULT_HYPOTHESIS_TEMPLATES)

    def run(self, request: AgentRequest) -> AgentResponse:
        started_at = perf_counter()
        self.logger.info(
            "Received hypothesis generation request.",
            extra={"action": "validate_request", "status": "running"},
        )

        try:
            spec = HypothesisSpec.from_payload(request.payload)
        except ValueError as exc:
            elapsed = perf_counter() - started_at
            self.logger.warning(
                "Hypothesis request validation failed.",
                extra={"action": "validate_request", "status": "error"},
            )
            return AgentResponse.failure(
                str(exc),
                metadata=self._metadata(request, elapsed),
            )

        self.logger.info(
            "Generating alpha hypotheses.",
            extra={"action": "generate_hypotheses", "status": "running"},
        )
        hypotheses = self.generate_hypotheses(spec)
        elapsed = perf_counter() - started_at

        self.logger.info(
            "Generated alpha hypotheses.",
            extra={"action": "generate_hypotheses", "status": "success"},
        )
        return AgentResponse.success(
            output={
                "state": "hypotheses_generated",
                "request": spec.to_dict(),
                "hypotheses": hypotheses,
                "hypothesis_count": len(hypotheses),
                "generation_method": GENERATION_METHOD,
                "next_action": "Create factor templates in Day 9.",
            },
            metadata=self._metadata(
                request,
                elapsed,
                market=spec.market,
                universe=spec.universe,
                horizon=spec.horizon,
                hypothesis_count=len(hypotheses),
            ),
        )

    def generate_hypotheses(self, spec: HypothesisSpec) -> list[dict[str, Any]]:
        selected_templates = self.templates[: spec.max_hypotheses]
        return [
            template.instantiate(spec, index=index)
            for index, template in enumerate(selected_templates, start=1)
        ]

    def _metadata(
        self,
        request: AgentRequest,
        elapsed: float,
        *,
        market: str | None = None,
        universe: str | None = None,
        horizon: str | None = None,
        hypothesis_count: int | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "agent": self.name,
            "task_id": request.task_id,
            "execution_time_sec": round(elapsed, 6),
        }
        if market is not None:
            metadata["market"] = market
        if universe is not None:
            metadata["universe"] = universe
        if horizon is not None:
            metadata["horizon"] = horizon
        if hypothesis_count is not None:
            metadata["hypothesis_count"] = hypothesis_count
        return metadata


DEFAULT_HYPOTHESIS_TEMPLATES = (
    HypothesisTemplate(
        template_id="capital_inflow_volume_confirmation",
        title="Capital Inflow With Volume Confirmation",
        description=(
            "In {universe}, stocks with rising traded amount and expanding turnover, "
            "while price remains constructive, may earn {horizon} excess returns."
        ),
        rationale=(
            "Persistent capital participation can reveal demand before the price fully "
            "adjusts, especially when volume confirms rather than contradicts the move."
        ),
        candidate_signals=(
            "amount_growth_5d",
            "turnover_rate_change_5d",
            "volume_ratio_5d_20d",
            "return_5d",
        ),
        expected_direction="Higher inflow with controlled positive price movement implies higher forward return.",
        required_data=("OHLCV", "turnover_rate", "amount"),
        risk_flags=("liquidity_chasing", "crowded_momentum", "limit_up_bias"),
        test_plan=(
            "Create rolling amount and turnover expansion features.",
            "Evaluate next 1, 5, and 10 trading-day excess returns with IC and RankIC.",
            "Compare results across liquidity buckets.",
        ),
    ),
    HypothesisTemplate(
        template_id="short_term_reversal_after_selloff",
        title="Short-Term Reversal After Liquidity-Supported Selloff",
        description=(
            "In {universe}, stocks with sharp recent losses but stabilizing volume may "
            "mean-revert over the {horizon} horizon."
        ),
        rationale=(
            "Temporary liquidity pressure can push prices below fair value. Reversal is "
            "more plausible when selling pressure weakens instead of accelerating."
        ),
        candidate_signals=(
            "return_3d",
            "return_5d",
            "volume_zscore_5d",
            "intraday_range_5d",
        ),
        expected_direction="More negative short-run return with non-accelerating volume implies higher forward return.",
        required_data=("OHLCV", "volume"),
        risk_flags=("falling_knife", "event_risk", "suspension_bias"),
        test_plan=(
            "Rank recent losers after excluding no-trade and suspended rows.",
            "Condition reversal signal on volume not making a new rolling high.",
            "Check drawdown and hit-rate by volatility bucket.",
        ),
    ),
    HypothesisTemplate(
        template_id="breakout_after_volatility_compression",
        title="Breakout After Volatility Compression",
        description=(
            "In {universe}, volatility compression followed by price and volume expansion "
            "may predict {horizon} continuation."
        ),
        rationale=(
            "Low realized volatility can mark an information buildup phase. A confirmed "
            "breakout suggests new information is being incorporated into prices."
        ),
        candidate_signals=(
            "realized_volatility_20d",
            "high_breakout_20d",
            "volume_ratio_5d_20d",
            "close_position_in_range",
        ),
        expected_direction="Lower prior volatility plus confirmed upside breakout implies higher forward return.",
        required_data=("OHLCV", "volume"),
        risk_flags=("false_breakout", "limit_up_bias", "market_regime_dependence"),
        test_plan=(
            "Measure rolling volatility compression before breakout dates.",
            "Require close price near the upper daily range on confirmation days.",
            "Evaluate persistence under broad-market up, flat, and down regimes.",
        ),
    ),
    HypothesisTemplate(
        template_id="quality_momentum_low_drawdown",
        title="Low-Drawdown Momentum Quality",
        description=(
            "In {universe}, stocks with positive momentum and shallow recent drawdowns may "
            "deliver more reliable {horizon} excess returns than high-volatility winners."
        ),
        rationale=(
            "Stable momentum can indicate steady institutional accumulation, while noisy "
            "momentum is more likely to reverse after short-term crowding."
        ),
        candidate_signals=(
            "return_20d",
            "max_drawdown_20d",
            "realized_volatility_20d",
            "positive_return_days_20d",
        ),
        expected_direction="Higher momentum with lower drawdown and volatility implies higher forward return.",
        required_data=("OHLCV",),
        risk_flags=("momentum_crash", "sector_crowding", "size_bias"),
        test_plan=(
            "Compare raw momentum against drawdown-adjusted momentum.",
            "Neutralize by volatility buckets before evaluating RankIC.",
            "Check whether results survive market drawdown months.",
        ),
    ),
    HypothesisTemplate(
        template_id="price_underreaction_to_liquidity_shock",
        title="Price Underreaction To Liquidity Shock",
        description=(
            "In {universe}, unusually high turnover without an immediate price response "
            "may indicate delayed {horizon} repricing."
        ),
        rationale=(
            "A liquidity shock with muted price movement can reflect accumulation or "
            "distribution before the visible trend becomes obvious."
        ),
        candidate_signals=(
            "turnover_rate_zscore_20d",
            "amount_zscore_20d",
            "absolute_return_1d",
            "close_to_open_return",
        ),
        expected_direction="High turnover shock with modest positive price response implies higher forward return.",
        required_data=("OHLCV", "turnover_rate", "amount"),
        risk_flags=("block_trade_noise", "news_event_contamination", "liquidity_bias"),
        test_plan=(
            "Identify turnover shocks relative to each stock's own history.",
            "Separate positive, flat, and negative price-response buckets.",
            "Evaluate delayed returns and turnover decay after the event.",
        ),
    ),
    HypothesisTemplate(
        template_id="turnover_exhaustion_reversal",
        title="Turnover Exhaustion Reversal",
        description=(
            "In {universe}, extreme turnover after an extended rise may predict weaker "
            "{horizon} returns as short-term demand becomes exhausted."
        ),
        rationale=(
            "Late-stage crowding often appears as unusually active trading after a strong "
            "move. When marginal buyers are exhausted, forward returns can deteriorate."
        ),
        candidate_signals=(
            "return_20d",
            "turnover_rate_zscore_20d",
            "volume_ratio_1d_20d",
            "upper_shadow_ratio",
        ),
        expected_direction="High prior return plus extreme turnover implies lower forward return.",
        required_data=("OHLCV", "turnover_rate"),
        risk_flags=("momentum_interference", "sector_rotation", "limit_up_bias"),
        test_plan=(
            "Rank stocks by combined prior return and turnover extremeness.",
            "Test both long-only avoidance and short-leg contribution.",
            "Check whether the effect is concentrated in high-volatility names.",
        ),
    ),
)


def _optional_str(payload: Mapping[str, Any], key: str, default: str) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str) or not value.strip():
        msg = f"payload.{key} must be a non-empty string."
        raise ValueError(msg)
    return value.strip()


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


def _optional_str_sequence(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        msg = f"payload.{key} must be a sequence of strings."
        raise ValueError(msg)

    items = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            msg = f"payload.{key} must contain only non-empty strings."
            raise ValueError(msg)
        items.append(item.strip())
    return tuple(dict.fromkeys(items))


def _optional_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        msg = f"payload.{key} must be an object."
        raise ValueError(msg)
    return dict(value)


def _normalize_horizon(value: str) -> str:
    key = value.strip().lower().replace(" ", "_")
    return _HORIZON_ALIASES.get(key, key)
