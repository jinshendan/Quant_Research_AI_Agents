from __future__ import annotations

import pytest

from agents.factor_templates import (
    DEFAULT_FACTOR_TEMPLATES,
    FactorTemplate,
    FactorTemplateLibrary,
)
from agents.hypothesis_agent import HypothesisAgent, HypothesisSpec


def test_default_factor_templates_are_valid_and_unique() -> None:
    library = FactorTemplateLibrary()
    template_ids = [template.template_id for template in library.all()]

    assert library.template_count == len(DEFAULT_FACTOR_TEMPLATES)
    assert library.template_count >= 18
    assert len(template_ids) == len(set(template_ids))
    assert "amount_growth_5d" in template_ids
    assert "upper_shadow_ratio" in template_ids


def test_factor_template_serializes_to_stable_dict() -> None:
    template = FactorTemplateLibrary().get("volume_ratio_5d_20d")

    assert template.to_dict() == {
        "template_id": "volume_ratio_5d_20d",
        "name": "Volume Ratio 5D To 20D",
        "category": "liquidity",
        "description": "Short-window average volume relative to the medium-window average.",
        "expression": "mean(volume, 5) / mean(volume, 20)",
        "direction": "positive",
        "required_columns": ["volume"],
        "parameters": {"fast_window": 5, "slow_window": 20},
        "lookback_days": 20,
        "signal_tags": ["volume_ratio_5d_20d"],
        "risk_flags": ["volume_spike_noise", "limit_up_bias"],
    }


def test_factor_template_library_finds_templates_by_signal_tags() -> None:
    library = FactorTemplateLibrary()

    matches = library.find_by_signals(
        ["volume_ratio_5d_20d", "return_5d"],
        limit=2,
    )

    assert [template.template_id for template in matches] == [
        "volume_ratio_5d_20d",
        "return_5d",
    ]


def test_factor_template_library_maps_hypotheses_to_templates() -> None:
    hypothesis_agent = HypothesisAgent()
    spec = HypothesisSpec.from_payload(
        {
            "universe": "CSI500",
            "horizon": "short",
            "max_hypotheses": 1,
        }
    )
    hypotheses = hypothesis_agent.generate_hypotheses(spec)
    library = FactorTemplateLibrary()

    mappings = library.templates_for_hypotheses(hypotheses, limit_per_hypothesis=3)

    assert mappings[0]["hypothesis_id"].startswith("HYP-001-")
    assert mappings[0]["template_count"] == 3
    assert [template["template_id"] for template in mappings[0]["templates"]] == [
        "amount_growth_5d",
        "turnover_rate_change_5d",
        "volume_ratio_5d_20d",
    ]


def test_factor_template_library_exports_manifest() -> None:
    manifest = FactorTemplateLibrary().export_manifest()

    assert manifest["template_count"] == len(manifest["templates"])
    assert manifest["templates"][0]["template_id"] == "amount_growth_5d"


def test_factor_template_library_rejects_duplicate_template_ids() -> None:
    template = DEFAULT_FACTOR_TEMPLATES[0]

    with pytest.raises(ValueError, match="Duplicate factor template id"):
        FactorTemplateLibrary(templates=(template, template))


def test_factor_template_rejects_future_looking_expression() -> None:
    template = FactorTemplate(
        template_id="bad_future_factor",
        name="Bad Future Factor",
        category="bad",
        description="Invalid future-looking expression.",
        expression="lead(close, 1) / close - 1",
        direction="positive",
        required_columns=("close",),
        parameters={"window": 1},
        lookback_days=1,
        signal_tags=("bad_signal",),
    )

    with pytest.raises(ValueError, match="future-looking"):
        FactorTemplateLibrary(templates=(template,))


def test_factor_template_library_rejects_invalid_hypothesis_shape() -> None:
    library = FactorTemplateLibrary()

    with pytest.raises(ValueError, match="hypothesis_id"):
        library.templates_for_hypotheses([{"candidate_signals": ["return_5d"]}])
