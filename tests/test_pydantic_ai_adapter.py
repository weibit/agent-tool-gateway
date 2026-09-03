"""Pydantic AI adapter: end-to-end through Agent.run with a scripted FunctionModel."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import pytest

pydantic_ai = pytest.importorskip("pydantic_ai")

from pydantic_ai import (  # noqa: E402
    Agent,
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    ToolApproved,
    ToolDenied,
)
from pydantic_ai.messages import ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart  # noqa: E402
from pydantic_ai.models.function import AgentInfo, FunctionModel  # noqa: E402
from pydantic_ai.toolsets import FunctionToolset  # noqa: E402

from agent_tool_gateway import (  # noqa: E402
    AgentIdentity,
    Gateway,
    InMemoryAuditSink,
    Principal,
    SessionState,
    SideEffect,
    ToolRegistry,
)
from agent_tool_gateway.adapters.pydantic_ai import (  # noqa: E402
    GatedToolset,
    default_identity,
    gate_toolset,
    manifest_from_tool_def,
)
from agent_tool_gateway.stages import RulePolicy, TokenBucketLimiter, default_stages  # noqa: E402


# ------------------------------------------------------------- harness


def scripted(calls: list[tuple[str, dict[str, Any], str]]) -> FunctionModel:
    """One ToolCallPart per queued (name, args, call_id), then a final text."""
    queue = list(calls)

    def fn(messages: list, info: AgentInfo) -> ModelResponse:
        if queue:
            name, args, cid = queue.pop(0)
            return ModelResponse(parts=[ToolCallPart(tool_name=name, args=args, tool_call_id=cid)])
        return ModelResponse(parts=[TextPart("done")])

    return FunctionModel(fn)


@dataclasses.dataclass
class Identity:
    principal: Principal
    agent: AgentIdentity
    session: SessionState


def make_toolset() -> FunctionToolset[Identity]:
    ts: FunctionToolset[Identity] = FunctionToolset()

    @ts.tool
    def echo(ctx: RunContext[Identity], text: str) -> str:
        """Echo the text back."""
        return f"echo:{text}"

    @ts.tool
    def pay(ctx: RunContext[Identity], amount: int) -> str:
        """Charge an amount."""
        return f"paid:{amount}"

    @ts.tool
    def leak(ctx: RunContext[Identity]) -> str:
        """Return something that needs redaction."""
        return "ssn 123-45-6789"

    return ts


def make(**session_kw):
    audit = InMemoryAuditSink()
    ts = make_toolset()
    defs = {name: tool.tool_def for name, tool in ts.tools.items()}
    reg = ToolRegistry(
        [
            manifest_from_tool_def(defs["echo"], required_scopes=["echo"]),
            manifest_from_tool_def(defs["pay"], side_effect="write", cost_usd=0.01),
            manifest_from_tool_def(defs["leak"]),
        ]
    )
    policy = (
        RulePolicy()
        .allow("echo")
        .allow("leak")
        .require_approval("pay", reason="money")
        .deny("echo", when=lambda c: c.args.get("text") == "deny", reason="denied text", priority=10)
        .require_approval("echo", when=lambda c: c.args.get("text") == "risky", reason="risky text", priority=10)
        .rewrite(
            "echo",
            transform=lambda c: {**c.args, "text": c.args["text"].removeprefix("raw:")},
            when=lambda c: c.args.get("text", "").startswith("raw:"),
            priority=5,
        )
    )
    gw = Gateway(reg, default_stages(policy, limiter=TokenBucketLimiter(rate_per_s=100, burst=100)), audit=audit)
    scopes = frozenset({"echo"})
    ident = Identity(Principal("u", scopes=scopes), AgentIdentity("a", scopes=scopes), SessionState(**session_kw))
    return gw, audit, ident, ts


def agent_for(gw, ts, calls, *, gate=True):
    toolset = GatedToolset(ts, gw) if gate else ts
    return Agent(scripted(calls), toolsets=[toolset], deps_type=Identity, output_type=[str, DeferredToolRequests])


def returns(result) -> list:
    out = []
    for m in result.all_messages():
        for p in m.parts:
            if isinstance(p, ToolReturnPart):
                out.append(p.content)
            elif isinstance(p, RetryPromptPart):
                out.append(("retry", p.content))
    return out


# --------------------------------------------------------------- tests


async def test_harness_runs_without_gateway():
    gw, _, ident, ts = make()
    r = await agent_for(gw, ts, [("echo", {"text": "hi"}, "c0")], gate=False).run("go", deps=ident)
    assert returns(r) == ["echo:hi"] and r.output == "done"
