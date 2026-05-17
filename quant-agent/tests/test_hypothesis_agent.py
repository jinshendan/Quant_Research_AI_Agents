from __future__ import annotations

from io import StringIO

import pytest

from agents.hypothesis_agent import (
    HypothesisAgent,
    HypothesisSpec,
    HypothesisTemplate,
)
from core.logging import configure_logging, get_agent_logger
from core.models import AgentRequest


def test_hypothesis_spec_normalizes_valid_payload() -> None:
    spec = HypothesisSpec.from_payload(
        {
            "objective": " Find short-term alpha ",
            "market": "A_SHARE",
            "universe": " CSI500 ",
            "horizon": "5d",
            "max_hypotheses": 3,
            "constraints": ["no future leakage", "no future leakage", "liquid names"],
            "data_context": {"aligned_rows": 100},
        }
    )

    assert spec.to_dict() == {
        "objective": "Find short-term alpha",
        "market": "a_share",
        "universe": "CSI500",
        "horizon": "short_term",
        "max_hypotheses": 3,
        "constraints": ["no future leakage", "liquid names"],
        "data_context": {"aligned_rows": 100},
    }


def test_hypothesis_spec_rejects_invalid_horizon() -> None:
    with pytest.raises(ValueError, match="Unsupported horizon"):
        HypothesisSpec.from_payload({"horizon": "intraday"})


def test_hypothesis_spec_rejects_invalid_count() -> None:
    with pytest.raises(ValueError, match="max_hypotheses"):
        HypothesisSpec.from_payload({"max_hypotheses": 0})


def test_hypothesis_agent_generates_structured_hypotheses() -> None:
    stream = StringIO()
    configure_logging(stream=stream)
    agent = HypothesisAgent(logger=get_agent_logger("HypothesisAgent"))
    request = AgentRequest.create(
        {
            "objective": "Find short-term alpha opportunities",
            "universe": "CSI500",
            "horizon": "short",
            "max_hypotheses": 2,
        },
        task_id="hypothesis-task-1",
    )

    response = agent.run(request)

    assert response.status == "success"
    assert response.output["state"] == "hypotheses_generated"
    assert response.output["hypothesis_count"] == 2
    assert response.output["generation_method"] == "deterministic_template_v1"
    assert response.output["next_action"] == "Create factor templates in Day 9."
    assert response.metadata["agent"] == "HypothesisAgent"
    assert response.metadata["task_id"] == "hypothesis-task-1"
    assert response.metadata["hypothesis_count"] == 2
    assert response.metadata["horizon"] == "short_term"

    first = response.output["hypotheses"][0]
    assert first["hypothesis_id"].startswith("HYP-001-")
    assert first["horizon"] == "short_term"
    assert first["candidate_signals"]
    assert first["required_data"]
    assert first["risk_flags"]
    assert first["test_plan"]
    assert "HypothesisAgent | generate_hypotheses | success" in stream.getvalue()


def test_hypothesis_agent_returns_error_for_bad_payload() -> None:
    stream = StringIO()
    configure_logging(stream=stream)
    agent = HypothesisAgent(logger=get_agent_logger("HypothesisAgent"))
    request = AgentRequest.create({"constraints": "must be a list"})

    response = agent.run(request)

    assert response.status == "error"
    assert response.error == "payload.constraints must be a sequence of strings."
    assert response.output == {}
    assert response.metadata["agent"] == "HypothesisAgent"
    assert "HypothesisAgent | validate_request | error" in stream.getvalue()


def test_hypothesis_agent_supports_template_injection() -> None:
    template = HypothesisTemplate(
        template_id="custom_liquidity",
        title="Custom Liquidity Hypothesis",
        description="{universe} custom description for {horizon}.",
        rationale="Custom rationale.",
        candidate_signals=("custom_signal",),
        expected_direction="Higher custom_signal implies higher forward return.",
        required_data=("OHLCV",),
        risk_flags=("custom_risk",),
        test_plan=("Run a custom IC test.",),
    )
    agent = HypothesisAgent(templates=(template,))
    spec = HypothesisSpec.from_payload({"universe": "SSE50", "max_hypotheses": 5})

    hypotheses = agent.generate_hypotheses(spec)

    assert hypotheses == [
        {
            "hypothesis_id": "HYP-001-custom_liquidity",
            "title": "Custom Liquidity Hypothesis",
            "description": "SSE50 custom description for short_term.",
            "rationale": "Custom rationale.",
            "horizon": "short_term",
            "candidate_signals": ["custom_signal"],
            "expected_direction": "Higher custom_signal implies higher forward return.",
            "required_data": ["OHLCV"],
            "risk_flags": ["custom_risk"],
            "test_plan": ["Run a custom IC test."],
            "source": "deterministic_template_v1",
        }
    ]
