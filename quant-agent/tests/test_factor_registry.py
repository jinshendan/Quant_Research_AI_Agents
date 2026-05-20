from __future__ import annotations

import pytest

from agents.factor_registry import (
    CompositeFactorSpec,
    FactorDefinition,
    FactorDefinitionRegistry,
)
from agents.factor_templates import FactorTemplateLibrary


def test_factor_definition_from_template_contains_research_metadata() -> None:
    template = FactorTemplateLibrary().get("return_5d")

    definition = FactorDefinition.from_template(template)

    assert definition.to_dict() == {
        "factor_id": "return_5d",
        "factor_column": "factor__return_5d",
        "name": "Five-Day Return",
        "source_type": "template",
        "formula": "close / delay(close, 5) - 1",
        "hypothesis": "Close-to-close return over five sessions.",
        "category": "momentum",
        "direction": "positive",
        "lookback_days": 6,
        "data_lag_days": 0,
        "required_columns": ["close"],
        "parameters": {"window": 5},
        "signal_tags": ["return_5d"],
        "risk_flags": ["momentum_crowding", "reversal_risk"],
        "components": [],
    }


def test_registry_builds_composite_definition_from_components() -> None:
    library = FactorTemplateLibrary()
    composite = CompositeFactorSpec.from_mapping(
        {
            "name": "daily_blend",
            "normalize": "rank_pct",
            "components": [
                {"factor": "return_5d", "weight": 0.6},
                {"factor": "volume_ratio_5d_20d", "weight": 0.4},
            ],
        }
    )

    registry = FactorDefinitionRegistry.from_templates_and_composites(
        [library.get("return_5d"), library.get("volume_ratio_5d_20d")],
        [composite],
    )

    definition = registry.get("factor__daily_blend").to_dict()
    assert definition["source_type"] == "composite"
    assert definition["category"] == "composite"
    assert definition["direction"] == "positive"
    assert definition["lookback_days"] == 20
    assert definition["formula"] == (
        "0.6 * rank_pct(return_5d) + "
        "0.4 * rank_pct(volume_ratio_5d_20d)"
    )
    assert definition["parameters"] == {
        "method": "weighted_sum",
        "normalize": "rank_pct",
    }
    assert definition["components"][0]["factor_column"] == "factor__return_5d"


def test_registry_rejects_composite_with_missing_definition() -> None:
    composite = CompositeFactorSpec.from_mapping(
        {
            "name": "bad_blend",
            "components": [
                {"factor": "return_5d", "weight": 1.0},
                {"factor": "missing_factor", "weight": 1.0},
            ],
        }
    )

    with pytest.raises(ValueError, match="factor__missing_factor"):
        FactorDefinitionRegistry.from_templates_and_composites(
            [FactorTemplateLibrary().get("return_5d")],
            [composite],
        )
