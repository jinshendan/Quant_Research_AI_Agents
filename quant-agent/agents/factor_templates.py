from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

FactorDirection = Literal["positive", "negative", "neutral"]
FactorParameter = int | float | str

SUPPORTED_DIRECTIONS: set[str] = {"positive", "negative", "neutral"}
FORBIDDEN_EXPRESSION_TOKENS = ("future_", "lead(", "shift(-")


@dataclass(frozen=True, slots=True)
class FactorTemplate:
    """Symbolic factor definition to be implemented by FeatureAgent."""

    template_id: str
    name: str
    category: str
    description: str
    expression: str
    direction: FactorDirection
    required_columns: tuple[str, ...]
    parameters: Mapping[str, FactorParameter]
    lookback_days: int
    signal_tags: tuple[str, ...]
    risk_flags: tuple[str, ...] = ()

    def validate(self) -> None:
        _require_non_empty_str(self.template_id, "template_id")
        _require_non_empty_str(self.name, "name")
        _require_non_empty_str(self.category, "category")
        _require_non_empty_str(self.description, "description")
        _require_non_empty_str(self.expression, "expression")
        if self.direction not in SUPPORTED_DIRECTIONS:
            msg = f"Unsupported factor direction: {self.direction}."
            raise ValueError(msg)
        if self.lookback_days < 1:
            msg = "lookback_days must be greater than zero."
            raise ValueError(msg)

        _require_non_empty_tuple(self.required_columns, "required_columns")
        _require_non_empty_tuple(self.signal_tags, "signal_tags")
        _validate_parameters(self.parameters)

        expression = self.expression.lower().replace(" ", "")
        for token in FORBIDDEN_EXPRESSION_TOKENS:
            if token in expression:
                msg = f"Forbidden future-looking token in expression: {token}."
                raise ValueError(msg)

    def matches_any_signal(self, signal_tags: Sequence[str]) -> bool:
        signals = set(_normalize_string_sequence(signal_tags, "signal_tags"))
        return bool(signals.intersection(self.signal_tags))

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "expression": self.expression,
            "direction": self.direction,
            "required_columns": list(self.required_columns),
            "parameters": dict(self.parameters),
            "lookback_days": self.lookback_days,
            "signal_tags": list(self.signal_tags),
            "risk_flags": list(self.risk_flags),
        }


class FactorTemplateLibrary:
    """Read-only registry for symbolic factor templates."""

    def __init__(self, templates: Sequence[FactorTemplate] | None = None) -> None:
        self._templates = _validate_template_collection(templates or DEFAULT_FACTOR_TEMPLATES)
        self._by_id = {template.template_id: template for template in self._templates}

    @property
    def template_count(self) -> int:
        return len(self._templates)

    def all(self) -> list[FactorTemplate]:
        return list(self._templates)

    def get(self, template_id: str) -> FactorTemplate:
        try:
            return self._by_id[template_id]
        except KeyError as exc:
            msg = f"Unknown factor template: {template_id}."
            raise KeyError(msg) from exc

    def find_by_signals(
        self,
        signal_tags: Sequence[str],
        *,
        limit: int | None = None,
    ) -> list[FactorTemplate]:
        normalized_signals = _normalize_string_sequence(signal_tags, "signal_tags")
        if limit is not None and limit < 1:
            msg = "limit must be greater than zero."
            raise ValueError(msg)

        matches = [
            template
            for template in self._templates
            if template.matches_any_signal(normalized_signals)
        ]
        return matches[:limit] if limit is not None else matches

    def templates_for_hypotheses(
        self,
        hypotheses: Sequence[Mapping[str, Any]],
        *,
        limit_per_hypothesis: int = 3,
    ) -> list[dict[str, Any]]:
        if limit_per_hypothesis < 1:
            msg = "limit_per_hypothesis must be greater than zero."
            raise ValueError(msg)

        results = []
        for hypothesis in hypotheses:
            hypothesis_id = hypothesis.get("hypothesis_id")
            if not isinstance(hypothesis_id, str) or not hypothesis_id.strip():
                msg = "hypothesis.hypothesis_id must be a non-empty string."
                raise ValueError(msg)

            candidate_signals = _normalize_string_sequence(
                hypothesis.get("candidate_signals", ()),
                "candidate_signals",
            )
            matched_templates = self.find_by_signals(
                candidate_signals,
                limit=limit_per_hypothesis,
            )
            results.append(
                {
                    "hypothesis_id": hypothesis_id,
                    "candidate_signals": list(candidate_signals),
                    "template_count": len(matched_templates),
                    "templates": [template.to_dict() for template in matched_templates],
                }
            )
        return results

    def export_manifest(self) -> dict[str, Any]:
        return {
            "template_count": len(self._templates),
            "templates": [template.to_dict() for template in self._templates],
        }

    def available_signal_tags(self) -> list[str]:
        tags = {
            signal_tag
            for template in self._templates
            for signal_tag in template.signal_tags
        }
        return sorted(tags)


DEFAULT_FACTOR_TEMPLATES = (
    FactorTemplate(
        template_id="amount_growth_5d",
        name="Five-Day Amount Growth",
        category="liquidity",
        description="Rolling growth in traded amount over five sessions.",
        expression="pct_change(amount, window=5)",
        direction="positive",
        required_columns=("amount",),
        parameters={"window": 5},
        lookback_days=6,
        signal_tags=("amount_growth_5d",),
        risk_flags=("liquidity_bias", "event_volume_noise"),
    ),
    FactorTemplate(
        template_id="turnover_rate_change_5d",
        name="Five-Day Turnover Change",
        category="liquidity",
        description="Recent turnover expansion relative to five sessions ago.",
        expression="turnover_rate - delay(turnover_rate, 5)",
        direction="positive",
        required_columns=("turnover_rate",),
        parameters={"window": 5},
        lookback_days=6,
        signal_tags=("turnover_rate_change_5d",),
        risk_flags=("liquidity_bias", "crowded_flow"),
    ),
    FactorTemplate(
        template_id="volume_ratio_5d_20d",
        name="Volume Ratio 5D To 20D",
        category="liquidity",
        description="Short-window average volume relative to the medium-window average.",
        expression="mean(volume, 5) / mean(volume, 20)",
        direction="positive",
        required_columns=("volume",),
        parameters={"fast_window": 5, "slow_window": 20},
        lookback_days=20,
        signal_tags=("volume_ratio_5d_20d",),
        risk_flags=("volume_spike_noise", "limit_up_bias"),
    ),
    FactorTemplate(
        template_id="return_5d",
        name="Five-Day Return",
        category="momentum",
        description="Close-to-close return over five sessions.",
        expression="close / delay(close, 5) - 1",
        direction="positive",
        required_columns=("close",),
        parameters={"window": 5},
        lookback_days=6,
        signal_tags=("return_5d",),
        risk_flags=("momentum_crowding", "reversal_risk"),
    ),
    FactorTemplate(
        template_id="return_3d",
        name="Three-Day Return",
        category="reversal",
        description="Close-to-close return over three sessions.",
        expression="close / delay(close, 3) - 1",
        direction="negative",
        required_columns=("close",),
        parameters={"window": 3},
        lookback_days=4,
        signal_tags=("return_3d",),
        risk_flags=("falling_knife", "event_risk"),
    ),
    FactorTemplate(
        template_id="volume_zscore_5d",
        name="Five-Day Volume Z-Score",
        category="liquidity",
        description="Recent volume standardized against its five-day history.",
        expression="zscore(volume, window=5)",
        direction="neutral",
        required_columns=("volume",),
        parameters={"window": 5},
        lookback_days=5,
        signal_tags=("volume_zscore_5d",),
        risk_flags=("event_volume_noise",),
    ),
    FactorTemplate(
        template_id="intraday_range_5d",
        name="Five-Day Intraday Range",
        category="volatility",
        description="Average intraday high-low range scaled by close price.",
        expression="mean((high - low) / close, 5)",
        direction="negative",
        required_columns=("high", "low", "close"),
        parameters={"window": 5},
        lookback_days=5,
        signal_tags=("intraday_range_5d",),
        risk_flags=("volatility_regime_dependence",),
    ),
    FactorTemplate(
        template_id="realized_volatility_20d",
        name="Twenty-Day Realized Volatility",
        category="volatility",
        description="Standard deviation of one-day returns over twenty sessions.",
        expression="std(close / delay(close, 1) - 1, 20)",
        direction="negative",
        required_columns=("close",),
        parameters={"window": 20},
        lookback_days=21,
        signal_tags=("realized_volatility_20d",),
        risk_flags=("volatility_regime_dependence", "low_volatility_trap"),
    ),
    FactorTemplate(
        template_id="high_breakout_20d",
        name="Twenty-Day High Breakout",
        category="breakout",
        description="Close price position relative to the prior twenty-day high.",
        expression="close / max(delay(high, 1), 20) - 1",
        direction="positive",
        required_columns=("close", "high"),
        parameters={"window": 20},
        lookback_days=21,
        signal_tags=("high_breakout_20d",),
        risk_flags=("false_breakout", "limit_up_bias"),
    ),
    FactorTemplate(
        template_id="close_position_in_range",
        name="Close Position In Daily Range",
        category="price_action",
        description="Close location inside the daily high-low range.",
        expression="safe_divide(close - low, high - low)",
        direction="positive",
        required_columns=("close", "high", "low"),
        parameters={"window": 1},
        lookback_days=1,
        signal_tags=("close_position_in_range",),
        risk_flags=("single_day_noise",),
    ),
    FactorTemplate(
        template_id="max_drawdown_20d",
        name="Twenty-Day Max Drawdown",
        category="risk_adjusted_momentum",
        description="Maximum peak-to-trough drawdown over twenty sessions.",
        expression="max_drawdown(close, window=20)",
        direction="negative",
        required_columns=("close",),
        parameters={"window": 20},
        lookback_days=20,
        signal_tags=("max_drawdown_20d",),
        risk_flags=("momentum_crash", "drawdown_clustering"),
    ),
    FactorTemplate(
        template_id="positive_return_days_20d",
        name="Positive Return Days Ratio",
        category="risk_adjusted_momentum",
        description="Share of positive one-day returns over twenty sessions.",
        expression="mean((close / delay(close, 1) - 1) > 0, 20)",
        direction="positive",
        required_columns=("close",),
        parameters={"window": 20},
        lookback_days=21,
        signal_tags=("positive_return_days_20d",),
        risk_flags=("market_regime_dependence",),
    ),
    FactorTemplate(
        template_id="turnover_rate_zscore_20d",
        name="Twenty-Day Turnover Z-Score",
        category="liquidity",
        description="Turnover rate standardized against its twenty-day history.",
        expression="zscore(turnover_rate, window=20)",
        direction="neutral",
        required_columns=("turnover_rate",),
        parameters={"window": 20},
        lookback_days=20,
        signal_tags=("turnover_rate_zscore_20d",),
        risk_flags=("liquidity_bias", "crowded_flow"),
    ),
    FactorTemplate(
        template_id="amount_zscore_20d",
        name="Twenty-Day Amount Z-Score",
        category="liquidity",
        description="Traded amount standardized against its twenty-day history.",
        expression="zscore(amount, window=20)",
        direction="neutral",
        required_columns=("amount",),
        parameters={"window": 20},
        lookback_days=20,
        signal_tags=("amount_zscore_20d",),
        risk_flags=("block_trade_noise", "liquidity_bias"),
    ),
    FactorTemplate(
        template_id="absolute_return_1d",
        name="One-Day Absolute Return",
        category="price_action",
        description="Absolute close-to-close return from the prior session.",
        expression="abs(close / delay(close, 1) - 1)",
        direction="negative",
        required_columns=("close",),
        parameters={"window": 1},
        lookback_days=2,
        signal_tags=("absolute_return_1d",),
        risk_flags=("event_risk",),
    ),
    FactorTemplate(
        template_id="close_to_open_return",
        name="Close-To-Open Return",
        category="price_action",
        description="Intraday return from open to close.",
        expression="close / open - 1",
        direction="positive",
        required_columns=("open", "close"),
        parameters={"window": 1},
        lookback_days=1,
        signal_tags=("close_to_open_return",),
        risk_flags=("intraday_noise", "limit_up_bias"),
    ),
    FactorTemplate(
        template_id="volume_ratio_1d_20d",
        name="Volume Ratio 1D To 20D",
        category="liquidity",
        description="Current volume relative to the twenty-day average.",
        expression="volume / mean(volume, 20)",
        direction="neutral",
        required_columns=("volume",),
        parameters={"slow_window": 20},
        lookback_days=20,
        signal_tags=("volume_ratio_1d_20d",),
        risk_flags=("volume_spike_noise", "event_volume_noise"),
    ),
    FactorTemplate(
        template_id="upper_shadow_ratio",
        name="Upper Shadow Ratio",
        category="price_action",
        description="Upper candle shadow scaled by daily high-low range.",
        expression="safe_divide(high - max(open, close), high - low)",
        direction="negative",
        required_columns=("open", "high", "low", "close"),
        parameters={"window": 1},
        lookback_days=1,
        signal_tags=("upper_shadow_ratio",),
        risk_flags=("single_day_noise", "limit_up_bias"),
    ),
)


def _validate_template_collection(
    templates: Sequence[FactorTemplate],
) -> tuple[FactorTemplate, ...]:
    if not templates:
        msg = "At least one factor template is required."
        raise ValueError(msg)

    seen_ids: set[str] = set()
    validated = []
    for template in templates:
        template.validate()
        if template.template_id in seen_ids:
            msg = f"Duplicate factor template id: {template.template_id}."
            raise ValueError(msg)
        seen_ids.add(template.template_id)
        validated.append(template)
    return tuple(validated)


def _require_non_empty_str(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        msg = f"{field_name} must be a non-empty string."
        raise ValueError(msg)


def _require_non_empty_tuple(values: tuple[str, ...], field_name: str) -> None:
    if not values:
        msg = f"{field_name} must not be empty."
        raise ValueError(msg)
    for value in values:
        _require_non_empty_str(value, field_name)


def _validate_parameters(parameters: Mapping[str, FactorParameter]) -> None:
    for key, value in parameters.items():
        _require_non_empty_str(key, "parameter name")
        if isinstance(value, bool) or not isinstance(value, int | float | str):
            msg = f"Unsupported parameter value for {key}: {value!r}."
            raise ValueError(msg)


def _normalize_string_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        msg = f"{field_name} must be a sequence of strings."
        raise ValueError(msg)

    normalized = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            msg = f"{field_name} must contain only non-empty strings."
            raise ValueError(msg)
        normalized.append(item.strip())
    return tuple(dict.fromkeys(normalized))
