from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.models import AgentRequest, AgentResponse


def test_agent_request_create_generates_protocol_envelope() -> None:
    request = AgentRequest.create(
        {"universe": "CSI500"},
        task_id="task-1",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert request.to_dict() == {
        "task_id": "task-1",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "payload": {"universe": "CSI500"},
    }


def test_agent_request_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        AgentRequest(task_id="task-1", timestamp=datetime(2026, 1, 1))


def test_agent_response_success_and_failure_to_dict() -> None:
    success = AgentResponse.success(output={"rows": 10}, metadata={"agent": "DataAgent"})
    failure = AgentResponse.failure("bad request", metadata={"agent": "DataAgent"})

    assert success.to_dict() == {
        "status": "success",
        "output": {"rows": 10},
        "metadata": {"agent": "DataAgent"},
    }
    assert failure.to_dict() == {
        "status": "error",
        "output": {},
        "metadata": {"agent": "DataAgent"},
        "error": "bad request",
    }

