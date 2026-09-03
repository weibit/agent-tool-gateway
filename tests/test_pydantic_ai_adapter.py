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


def test_default_identity_reads_deps_or_raises():
    gw, _, ident, _ = make()
    assert default_identity(SimpleNamespace(deps=ident)) == (ident.principal, ident.agent, ident.session)
    with pytest.raises(RuntimeError, match="principal"):
        default_identity(SimpleNamespace(deps={"who": "bit"}))
    with pytest.raises(RuntimeError):
        default_identity(SimpleNamespace(deps=None))


def test_manifest_from_tool_def_copies_schema_and_applies_overrides():
    ts = make_toolset()
    d = ts.tools["echo"].tool_def
    m = manifest_from_tool_def(d)
    assert m.name == "echo" and m.description == "Echo the text back."
    assert m.input_schema["required"] == ["text"] and m.side_effect is SideEffect.READ
    m2 = manifest_from_tool_def(d, side_effect="write", required_scopes=["x"], cost_usd=0.5)
    assert m2.side_effect is SideEffect.WRITE and m2.required_scopes == frozenset({"x"}) and m2.cost_usd == 0.5


async def test_allow_runs_tool_and_audits():
    gw, audit, ident, ts = make()
    r = await agent_for(gw, ts, [("echo", {"text": "hi"}, "c1")]).run("go", deps=ident)
    assert returns(r) == ["echo:hi"] and r.output == "done"
    assert [e.phase for e in audit.events] == ["decision", "execution"]
    assert audit.events[0].decision == "allow" and audit.events[0].tool == "echo"
    assert len(ident.session.recent_calls) == 1 and ident.session.turn == 1


async def test_deny_returns_structured_error_and_run_continues():
    gw, _, ident, ts = make()
    r = await agent_for(gw, ts, [("echo", {"text": "deny"}, "c2")]).run("go", deps=ident)
    err = returns(r)[0]
    assert err["error"] == "authorization_denied" and err["retryable"] is False
    assert "denied text" in err["message"] and r.output == "done"


async def test_consecutive_denies_do_not_exhaust_retries():
    gw, _, ident, ts = make()
    calls = [("echo", {"text": "deny"}, "c3a"), ("echo", {"text": "deny"}, "c3b"), ("echo", {"text": "deny"}, "c3c")]
    r = await agent_for(gw, ts, calls).run("go", deps=ident)
    assert [e["error"] for e in returns(r)] == ["authorization_denied"] * 3 and r.output == "done"


async def test_schema_deny_is_retryable_invalid_arguments():
    gw, _, ident, ts = make()
    r = await agent_for(gw, ts, [("echo", {"text": 5}, "c4")]).run("go", deps=ident)
    out = returns(r)[0]
    if isinstance(out, tuple):  # Pydantic AI validated first and asked the model to retry
        pytest.skip("SDK validates before the toolset is reached; gateway schema stage not exercised")
    assert out["error"] == "invalid_arguments" and out["retryable"] is True


async def test_unregistered_tool_is_denied_not_crashed():
    gw, _, ident, ts = make()

    @ts.tool
    def mystery(ctx: RunContext[Identity]) -> str:
        """Not in the registry."""
        return "ran"

    r = await agent_for(gw, ts, [("mystery", {}, "c5")]).run("go", deps=ident)
    assert returns(r)[0]["error"] == "tool_not_registered" and r.output == "done"


async def test_transform_rewrites_arguments():
    gw, _, ident, ts = make()
    r = await agent_for(gw, ts, [("echo", {"text": "raw:hi"}, "c6")]).run("go", deps=ident)
    assert returns(r) == ["echo:hi"]


async def test_require_approval_yields_deferred_requests_without_running_or_reserving():
    gw, audit, ident, ts = make(budget_limit_usd=0.05)
    r = await agent_for(gw, ts, [("pay", {"amount": 3}, "c7")]).run("go", deps=ident)
    assert isinstance(r.output, DeferredToolRequests)
    assert [c.tool_call_id for c in r.output.approvals] == ["c7"]
    assert returns(r) == []
    assert ident.session.budget_reserved_usd == 0.0 and len(ident.session.recent_calls) == 0
    assert audit.events[-1].decision == "require_approval"
    meta = r.output.approvals[0]
    assert audit.events[-1].reason == "money" and meta.tool_name == "pay"


async def test_approve_then_resume_runs_tool_and_settles_budget():
    gw, audit, ident, ts = make(budget_limit_usd=0.05)
    agent = agent_for(gw, ts, [("pay", {"amount": 3}, "c8")])
    r = await agent.run("go", deps=ident)
    r2 = await agent.run(
        message_history=r.all_messages(),
        deferred_tool_results=DeferredToolResults(approvals={"c8": True}),
        deps=ident,
    )
    assert returns(r2)[-1] == "paid:3" and r2.output == "done"
    assert ident.session.budget_used_usd == pytest.approx(0.01) and ident.session.budget_reserved_usd == 0.0
    assert len(ident.session.recent_calls) == 1
    assert audit.events[-1].phase == "execution" and audit.events[-1].error_code is None


async def test_denied_approval_does_not_run_tool():
    gw, _, ident, ts = make()
    agent = agent_for(gw, ts, [("echo", {"text": "risky"}, "c9")])
    r = await agent.run("go", deps=ident)
    r2 = await agent.run(
        message_history=r.all_messages(),
        deferred_tool_results=DeferredToolResults(approvals={"c9": ToolDenied("human said no")}),
        deps=ident,
    )
    assert returns(r2)[-1] == "human said no" and r2.output == "done"
    assert len(ident.session.recent_calls) == 0


async def test_approval_with_override_args_is_evaluated_on_overridden_args():
    gw, _, ident, ts = make()
    agent = agent_for(gw, ts, [("echo", {"text": "risky"}, "c10")])
    r = await agent.run("go", deps=ident)
    r2 = await agent.run(
        message_history=r.all_messages(),
        deferred_tool_results=DeferredToolResults(approvals={"c10": ToolApproved(override_args={"text": "deny"})}),
        deps=ident,
    )
    assert returns(r2)[-1]["error"] == "authorization_denied"  # policy saw the overridden args
    r3 = await agent.run(
        message_history=r.all_messages(),
        deferred_tool_results=DeferredToolResults(approvals={"c10": ToolApproved(override_args={"text": "safe"})}),
        deps=ident,
    )
    assert returns(r3)[-1] == "echo:safe"


async def test_budget_exhausted_while_pending_denies_on_resume():
    gw, _, ident, ts = make(budget_limit_usd=0.05)
    agent = agent_for(gw, ts, [("pay", {"amount": 3}, "c11")])
    r = await agent.run("go", deps=ident)
    ident.session.budget_used_usd = 0.05  # spent elsewhere while waiting
    r2 = await agent.run(
        message_history=r.all_messages(),
        deferred_tool_results=DeferredToolResults(approvals={"c11": True}),
        deps=ident,
    )
    assert returns(r2)[-1]["error"] == "budget_exceeded"


async def test_output_guards_rewrite_what_the_model_sees():
    gw, _, ident, ts = make()
    r = await agent_for(gw, ts, [("leak", {}, "c12")]).run("go", deps=ident)
    assert returns(r) == ["ssn [REDACTED]"]


def test_run_sync_path():
    gw, _, ident, ts = make()
    r = agent_for(gw, ts, [("echo", {"text": "hi"}, "c13")]).run_sync("go", deps=ident)
    assert returns(r) == ["echo:hi"]


async def test_gate_outermost_sees_prefixed_names():
    gw, audit, ident, ts = make()
    gw.registry.register(manifest_from_tool_def(ts.tools["echo"].tool_def, name="x_echo", required_scopes=["echo"]))
    gw.stages[2].policy.allow("x_echo")  # PolicyStage is third in default_stages
    agent = Agent(
        scripted([("x_echo", {"text": "hi"}, "c14")]),
        toolsets=[gate_toolset(gw, ts.prefixed("x"))],
        deps_type=Identity,
        output_type=[str, DeferredToolRequests],
    )
    r = await agent.run("go", deps=ident)
    assert returns(r) == ["echo:hi"] and audit.events[0].tool == "x_echo"


async def test_tool_exception_releases_reservation_and_propagates():
    gw, _, ident, ts = make(budget_limit_usd=0.05)

    @ts.tool
    def boom(ctx: RunContext[Identity]) -> str:
        """Always fails."""
        raise RuntimeError("kaboom")

    gw.registry.register(manifest_from_tool_def(ts.tools["boom"].tool_def, cost_usd=0.01))
    gw.stages[2].policy.allow("boom")
    with pytest.raises(RuntimeError, match="kaboom"):
        await agent_for(gw, ts, [("boom", {}, "c15")]).run("go", deps=ident)
    assert ident.session.budget_reserved_usd == 0.0 and ident.session.budget_used_usd == 0.0
