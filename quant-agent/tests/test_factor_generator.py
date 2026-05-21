from __future__ import annotations

from io import StringIO

import pytest

from agents.factor_generator import (
    DEFAULT_FACTOR_COUNT,
    FactorCandidateGenerator,
    FactorFamily,
    FactorGenerationAgent,
    FactorGenerationSpec,
)
from core.logging import configure_logging, get_agent_logger
from core.models import AgentRequest


def test_factor_generation_spec_normalizes_valid_payload() -> None:
    spec = FactorGenerationSpec.from_payload(
        {
            "target_count": 5,
            "source_template_ids": ["return_5d", "return_5d", "amount_growth_5d"],
        }
    )

    assert spec.to_dict() == {
        "target_count": 5,
        "source_template_ids": ["return_5d", "amount_growth_5d"],
    }


def test_factor_generation_spec_rejects_invalid_count() -> None:
    with pytest.raises(ValueError, match="target_count"):
        FactorGenerationSpec.from_payload({"target_count": 0})


def test_factor_candidate_generator_creates_default_factor_batch() -> None:
    generator = FactorCandidateGenerator()
    result = generator.generate(FactorGenerationSpec())
    factors = result.factors

    assert len(factors) == DEFAULT_FACTOR_COUNT
    assert factors[0].factor_id == "alpha_001"
    assert factors[-1].factor_id == f"alpha_{DEFAULT_FACTOR_COUNT:03d}"
    assert len({factor.factor_id for factor in factors}) == DEFAULT_FACTOR_COUNT
    assert len({factor.expression for factor in factors}) == DEFAULT_FACTOR_COUNT
    assert result.stats["unique_expression_count"] == DEFAULT_FACTOR_COUNT
    assert result.stats["max_lookback_days"] == 60
    assert result.stats["source_template_counts"]["return_5d"] >= 6
    assert result.stats["category_counts"]["liquidity"] >= 15
    assert result.stats["category_counts"]["volume_price"] >= 1
    assert result.stats["category_counts"]["composite"] >= 1

    for factor in factors:
        expression = factor.expression.lower().replace(" ", "")
        assert "future_" not in expression
        assert "lead(" not in expression
        assert "shift(-" not in expression
        assert factor.required_columns
        assert factor.lookback_days >= 1


def test_factor_candidate_generator_filters_by_source_template() -> None:
    generator = FactorCandidateGenerator()
    spec = FactorGenerationSpec.from_payload(
        {
            "target_count": 3,
            "source_template_ids": ["return_5d"],
        }
    )

    result = generator.generate(spec)

    assert [factor.source_template_id for factor in result.factors] == [
        "return_5d",
        "return_5d",
        "return_5d",
    ]
    assert [factor.factor_id for factor in result.factors] == [
        "alpha_001",
        "alpha_002",
        "alpha_003",
    ]


def test_factor_candidate_generator_rejects_unknown_source_template() -> None:
    generator = FactorCandidateGenerator()
    spec = FactorGenerationSpec.from_payload(
        {
            "target_count": 1,
            "source_template_ids": ["not_a_template"],
        }
    )

    with pytest.raises(ValueError, match="Unknown source_template_ids"):
        generator.generate(spec)


def test_factor_candidate_generator_rejects_unavailable_target_count() -> None:
    generator = FactorCandidateGenerator()
    spec = FactorGenerationSpec.from_payload(
        {
            "target_count": 4,
            "source_template_ids": ["turnover_rate_zscore_20d"],
        }
    )

    with pytest.raises(ValueError, match="Only 3 candidate factors"):
        generator.generate(spec)


def test_factor_generation_agent_returns_structured_response() -> None:
    stream = StringIO()
    configure_logging(stream=stream)
    agent = FactorGenerationAgent(logger=get_agent_logger("FactorGenerationAgent"))
    request = AgentRequest.create({"target_count": 5}, task_id="factor-task-1")

    response = agent.run(request)

    assert response.status == "success"
    assert response.output["state"] == "factors_generated"
    assert response.output["factor_count"] == 5
    assert response.output["generation_method"] == "deterministic_factor_family_v1"
    assert response.output["next_action"] == "Save generated factors in Day 14."
    assert response.output["factors"][0]["factor_id"] == "alpha_001"
    assert response.output["generation_stats"]["unique_expression_count"] == 5
    assert response.metadata["agent"] == "FactorGenerationAgent"
    assert response.metadata["task_id"] == "factor-task-1"
    assert response.metadata["factor_count"] == 5
    assert "FactorGenerationAgent | generate_factors | success" in stream.getvalue()


def test_factor_generation_agent_returns_error_for_bad_request() -> None:
    agent = FactorGenerationAgent()
    response = agent.run(AgentRequest.create({"target_count": 999}))

    assert response.status == "error"
    assert "target_count" in str(response.error)
    assert response.output == {}


def test_factor_family_rejects_future_looking_expression() -> None:
    family = FactorFamily(
        family_id="bad_future",
        source_template_id="return_5d",
        category="bad",
        direction="positive",
        required_columns=("close",),
        risk_flags=(),
        name_template="Bad Future",
        expression_template="lead(close, {window}) / close - 1",
        parameter_grid=({"window": 1},),
    )
    generator = FactorCandidateGenerator(families=(family,))

    with pytest.raises(ValueError, match="future-looking"):
        generator.generate(FactorGenerationSpec.from_payload({"target_count": 1}))
