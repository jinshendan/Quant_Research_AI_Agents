from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

AgentStatus = Literal["success", "error"]


@dataclass(frozen=True, slots=True)
class AgentRequest:
    """Common request envelope used by every agent."""

    task_id: str
    timestamp: datetime
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            msg = "AgentRequest.timestamp must be timezone-aware."
            raise ValueError(msg)
        object.__setattr__(self, "payload", dict(self.payload))

    @classmethod
    def create(
        cls,
        payload: dict[str, Any] | None = None,
        *,
        task_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> AgentRequest:
        return cls(
            task_id=task_id or str(uuid4()),
            timestamp=timestamp or datetime.now(UTC),
            payload=payload or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "timestamp": self.timestamp.isoformat(),
            "payload": dict(self.payload),
        }


@dataclass(frozen=True, slots=True)
class AgentResponse:
    """Common response envelope used by every agent."""

    status: AgentStatus
    output: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def success(
        cls,
        *,
        output: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResponse:
        return cls(
            status="success",
            output=output or {},
            metadata=metadata or {},
        )

    @classmethod
    def failure(
        cls,
        error: str,
        *,
        output: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResponse:
        return cls(
            status="error",
            output=output or {},
            metadata=metadata or {},
            error=error,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": self.status,
            "output": dict(self.output),
            "metadata": dict(self.metadata),
        }
        if self.error is not None:
            result["error"] = self.error
        return result

