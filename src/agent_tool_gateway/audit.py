"""Audit trail. Every decision and every execution outcome emits one event."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, TextIO

from .context import ToolCallContext
from .decision import DecisionResult


@dataclass
class AuditEvent:
    ts: float
    phase: str  # "decision" | "execution" | "dry_run"
    trace_id: str
    session_id: str
    tool: str
    tool_version: str
    principal: str
    agent: str
    agent_chain: list[str]
    turn: int
    decision: str | None = None
    stage: str | None = None
    reason: str | None = None
    risk_score: float | None = None
    args_hash: str = ""
    tainted: bool = False
    duration_ms: float | None = None
    error_code: str | None = None
    error_detail: str | None = None  # full-fidelity; never shown to the model
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_context(cls, ctx: ToolCallContext, phase: str, **kw: Any) -> AuditEvent:
        return cls(
            ts=time.time(),
            phase=phase,
            trace_id=ctx.trace_id,
            session_id=ctx.session.session_id,
            tool=ctx.tool.name,
            tool_version=ctx.tool.version,
            principal=ctx.principal.id,
            agent=ctx.agent.id,
            agent_chain=ctx.agent.chain,
            turn=ctx.session.turn,
            args_hash=ctx.args_hash,
            **{"tainted": ctx.session.tainted, **kw},
        )

    @classmethod
    def from_decision(cls, ctx: ToolCallContext, result: DecisionResult, phase: str = "decision") -> AuditEvent:
        details = result.details
        return cls.from_context(
            ctx,
            phase,
            decision=result.decision.value,
            stage=result.stage,
            reason=result.reason,
            risk_score=result.risk_score,
            error_code=details.get("error"),
            error_detail=details.get("_detail"),  # full-fidelity channel; underscore keys never reach ``extra``
            extra={k: v for k, v in details.items() if not k.startswith("_")},
        )


class AuditSink(Protocol):
    async def emit(self, event: AuditEvent) -> None: ...


class InMemoryAuditSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def emit(self, event: AuditEvent) -> None:
        self.events.append(event)


class JsonlAuditSink:
    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stderr

    async def emit(self, event: AuditEvent) -> None:
        self._stream.write(json.dumps(asdict(event), default=str) + "\n")
        self._stream.flush()


class NullAuditSink:
    async def emit(self, event: AuditEvent) -> None:
        return None
