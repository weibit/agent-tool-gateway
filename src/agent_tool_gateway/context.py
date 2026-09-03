"""Runtime context that travels with every tool call.

Effective authority = delegated principal ∩ agent identity (attenuated along the
spawn chain) ∩ resource policy ∩ runtime context. Everything a stage needs to
decide should be reachable from ToolCallContext; stages must not reach outside it.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .manifest import ToolManifest


@dataclass(frozen=True)
class Principal:
    """The human or service on whose behalf the agent acts."""

    id: str
    tenant: str | None = None
    scopes: frozenset[str] = frozenset()
    attributes: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    @classmethod
    def anonymous(cls) -> Principal:
        return cls(id="anonymous")


@dataclass(frozen=True)
class AgentIdentity:
    """An agent instance and its position in the delegation chain.

    A child spawned via ``spawn`` can never hold more scopes than its parent:
    authority only attenuates as it is delegated.
    """

    id: str
    scopes: frozenset[str] = frozenset()
    parent: AgentIdentity | None = None
    kind: str = "agent"  # e.g. orchestrator / worker / coding-agent

    @property
    def depth(self) -> int:
        return 0 if self.parent is None else self.parent.depth + 1

    @property
    def chain(self) -> list[str]:
        ids = [self.id]
        p = self.parent
        while p is not None:
            ids.append(p.id)
            p = p.parent
        return list(reversed(ids))

    @property
    def effective_scopes(self) -> frozenset[str]:
        if self.parent is None:
            return self.scopes
        return self.scopes & self.parent.effective_scopes

    def spawn(self, child_id: str, scopes: frozenset[str] | None = None, kind: str = "worker") -> AgentIdentity:
        requested = self.effective_scopes if scopes is None else frozenset(scopes)
        return AgentIdentity(id=child_id, scopes=requested & self.effective_scopes, parent=self, kind=kind)


@dataclass
class SessionState:
    """Mutable per-session state the gateway reads and updates."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    turn: int = 0
    budget_limit_usd: float | None = None
    budget_used_usd: float = 0.0
    budget_reserved_usd: float = 0.0  # reserved by calls in flight; settled or released by the gateway
    tainted: bool = False
    taint_sources: list[str] = field(default_factory=list)
    approvals: set[str] = field(default_factory=set)  # approval keys ("tool:args_hash" / "tool:*") granted
    pending_approvals: dict[str, str] = field(default_factory=dict)  # approval_id -> approval key
    recent_calls: deque[tuple[str, str, float]] = field(default_factory=lambda: deque(maxlen=256))
    metadata: dict[str, Any] = field(default_factory=dict)

    def record_call(self, tool_name: str, args_hash: str) -> None:
        self.recent_calls.append((tool_name, args_hash, time.monotonic()))

    def mark_tainted(self, source: str) -> None:
        self.tainted = True
        self.taint_sources.append(source)

    def clear_taint(self) -> None:
        self.tainted = False
        self.taint_sources.clear()


def hash_args(args: dict[str, Any]) -> str:
    payload = json.dumps(args, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass
class ToolCallContext:
    tool: ToolManifest
    args: dict[str, Any]
    principal: Principal
    agent: AgentIdentity
    session: SessionState
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    tool_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def args_hash(self) -> str:
        return hash_args(self.args)

    @property
    def effective_scopes(self) -> frozenset[str]:
        return self.principal.scopes & self.agent.effective_scopes

    @property
    def approval_key(self) -> str:
        """What a human approves: this tool with exactly these arguments."""
        return f"{self.tool.name}:{self.args_hash}"


@dataclass
class ToolResult:
    content: Any
    tainted: bool = False
    truncated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
