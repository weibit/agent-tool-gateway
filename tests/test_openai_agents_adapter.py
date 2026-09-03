"""OpenAI Agents SDK adapter: end-to-end through Runner.run with a scripted model."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import pytest

agents = pytest.importorskip("agents")

from agents import Agent, RunConfig, Runner, WebSearchTool, function_tool  # noqa: E402
from agents.items import ModelResponse  # noqa: E402
from agents.models.interface import Model  # noqa: E402
from agents.tool_context import ToolContext  # noqa: E402
from agents.usage import Usage  # noqa: E402
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText  # noqa: E402

from agent_tool_gateway import (  # noqa: E402
    AgentIdentity,
    Gateway,
    InMemoryAuditSink,
    Principal,
    SessionState,
    SideEffect,
    ToolRegistry,
)
from agent_tool_gateway.adapters.openai_agents import (  # noqa: E402
    OpenAIAgentsAdapter,
    default_identity,
    gate_tools,
    manifest_from_function_tool,
)
from agent_tool_gateway.stages import RulePolicy, TokenBucketLimiter, default_stages  # noqa: E402

CFG = RunConfig(tracing_disabled=True)


# ------------------------------------------------------------- harness


def tool_call(name: str, args: dict, call_id: str) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        call_id=call_id, name=name, arguments=json.dumps(args), type="function_call", id=f"fc_{call_id}"
    )


def final(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id="msg_1",
        role="assistant",
        status="completed",
        type="message",
        content=[ResponseOutputText(text=text, type="output_text", annotations=[])],
    )


class ScriptedModel(Model):
    """Returns the queued outputs in order, then a final 'done' message."""

    def __init__(self, outputs: list[list]) -> None:
        self.outputs = list(outputs)

    async def get_response(self, *args, **kwargs) -> ModelResponse:
        out = self.outputs.pop(0) if self.outputs else [final("done")]
        return ModelResponse(output=out, usage=Usage(), response_id=None)

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError


@function_tool
def echo(text: str) -> str:
    """Echo the text back."""
    return f"echo:{text}"


@function_tool
def pay(amount: int) -> str:
    """Charge an amount."""
    return f"paid:{amount}"


@function_tool
def leak() -> str:
    """Return something that needs redaction."""
    return "ssn 123-45-6789"


@dataclasses.dataclass
class Identity:
    principal: Principal
    agent: AgentIdentity
    session: SessionState


def make(**session_kw):
    audit = InMemoryAuditSink()
    reg = ToolRegistry(
        [
            manifest_from_function_tool(echo, required_scopes=["echo"]),
            manifest_from_function_tool(pay, side_effect="write", cost_usd=0.01),
            manifest_from_function_tool(leak),
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
    return gw, audit, ident


def outputs(result) -> list:
    return [i.output for i in result.new_items if i.type == "tool_call_output_item"]


async def run(adapter, ident, calls, tools=(echo, pay, leak)):
    agent = Agent(name="t", model=ScriptedModel([calls]), tools=adapter.gate_tools(list(tools)))
    return agent, await Runner.run(agent, "go", context=ident, run_config=CFG)


# --------------------------------------------------------------- tests


async def test_harness_runs_without_gateway():
    agent = Agent(name="t", model=ScriptedModel([[tool_call("echo", {"text": "hi"}, "c0")]]), tools=[echo])
    r = await Runner.run(agent, "go", run_config=CFG)
    assert outputs(r) == ["echo:hi"] and r.final_output == "done"


def test_default_identity_reads_context_or_raises():
    gw, _, ident = make()
    assert default_identity(SimpleNamespace(context=ident)) == (ident.principal, ident.agent, ident.session)
    with pytest.raises(RuntimeError, match="principal"):
        default_identity(SimpleNamespace(context={"who": "bit"}))
    with pytest.raises(RuntimeError):
        default_identity(SimpleNamespace(context=None))


def test_manifest_from_function_tool_copies_schema_and_applies_overrides():
    m = manifest_from_function_tool(echo)
    assert m.name == "echo" and m.description == "Echo the text back."
    assert m.input_schema["required"] == ["text"] and m.side_effect is SideEffect.READ
    m2 = manifest_from_function_tool(echo, side_effect="write", required_scopes=["x"], cost_usd=0.5)
    assert m2.side_effect is SideEffect.WRITE and m2.required_scopes == frozenset({"x"}) and m2.cost_usd == 0.5
    assert m2.input_schema == m.input_schema


async def test_allow_runs_tool_and_audits():
    gw, audit, ident = make()
    ad = OpenAIAgentsAdapter(gw)
    _, r = await run(ad, ident, [tool_call("echo", {"text": "hi"}, "c1")])
    assert outputs(r) == ["echo:hi"] and r.final_output == "done"
    assert [e.phase for e in audit.events] == ["decision", "execution"]
    assert audit.events[0].decision == "allow" and audit.events[0].tool == "echo"
    assert len(ident.session.recent_calls) == 1 and ident.session.turn == 1
    assert not ad._inflight


async def test_deny_returns_structured_error_to_model_and_run_continues():
    gw, _, ident = make()
    ad = OpenAIAgentsAdapter(gw)
    _, r = await run(ad, ident, [tool_call("echo", {"text": "deny"}, "c2")])
    err = json.loads(outputs(r)[0])
    assert err["error"] == "authorization_denied" and err["retryable"] is False
    assert "denied text" in err["message"] and r.final_output == "done"


async def test_schema_deny_is_retryable_invalid_arguments():
    gw, _, ident = make()
    ad = OpenAIAgentsAdapter(gw)
    _, r = await run(ad, ident, [tool_call("echo", {}, "c3")])
    err = json.loads(outputs(r)[0])
    assert err["error"] == "invalid_arguments" and err["retryable"] is True


async def test_unregistered_tool_is_denied_not_crashed():
    gw, _, ident = make()
    ad = OpenAIAgentsAdapter(gw)

    @function_tool
    def mystery() -> str:
        """Not in the registry."""
        return "ran"

    _, r = await run(ad, ident, [tool_call("mystery", {}, "c3b")], tools=(mystery,))
    err = json.loads(outputs(r)[0])
    assert err["error"] == "tool_not_registered" and r.final_output == "done"


async def test_transform_rewrites_arguments():
    gw, _, ident = make()
    ad = OpenAIAgentsAdapter(gw)
    _, r = await run(ad, ident, [tool_call("echo", {"text": "raw:hi"}, "c4")])
    assert outputs(r) == ["echo:hi"]


def test_hosted_tools_pass_through_and_function_tools_are_copied():
    gw, _, _ = make()
    ad = OpenAIAgentsAdapter(gw)
    ws = WebSearchTool()
    out = ad.gate_tools([echo, ws])
    assert out[1] is ws and out[0] is not echo and out[0].name == "echo"
    assert callable(out[0].needs_approval)
    assert gate_tools(gw, [echo])[0].name == "echo"


async def test_require_approval_interrupts_without_running_or_reserving():
    gw, audit, ident = make(budget_limit_usd=0.05)
    ad = OpenAIAgentsAdapter(gw)
    _, r = await run(ad, ident, [tool_call("pay", {"amount": 3}, "c5")])
    assert len(r.interruptions) == 1 and r.interruptions[0].tool_name == "pay"
    assert outputs(r) == []
    assert ident.session.budget_reserved_usd == 0.0 and len(ident.session.recent_calls) == 0
    assert audit.events[-1].decision == "require_approval"
    assert "c5" in ad._inflight


async def test_approve_then_resume_runs_tool_and_settles_budget():
    gw, audit, ident = make(budget_limit_usd=0.05)
    ad = OpenAIAgentsAdapter(gw)
    agent, r = await run(ad, ident, [tool_call("pay", {"amount": 3}, "c6")])
    state = r.to_state()
    state.approve(r.interruptions[0])
    r2 = await Runner.run(agent, state, run_config=CFG)
    assert outputs(r2) == ["paid:3"] and r2.final_output == "done"
    assert ident.session.budget_used_usd == pytest.approx(0.01) and ident.session.budget_reserved_usd == 0.0
    assert len(ident.session.recent_calls) == 1
    assert audit.events[-1].phase == "execution" and audit.events[-1].error_code is None
    assert "c6" not in ad._inflight


async def test_budget_exhausted_while_pending_denies_on_resume():
    gw, _, ident = make(budget_limit_usd=0.05)
    ad = OpenAIAgentsAdapter(gw)
    agent, r = await run(ad, ident, [tool_call("pay", {"amount": 3}, "c6b")])
    ident.session.budget_used_usd = 0.05  # spent elsewhere while waiting
    state = r.to_state()
    state.approve(r.interruptions[0])
    r2 = await Runner.run(agent, state, run_config=CFG)
    assert json.loads(outputs(r2)[0])["error"] == "budget_exceeded"
    assert ident.session.budget_reserved_usd == 0.0 and len(ident.session.recent_calls) == 0
    assert "pay:*" not in ident.session.approvals and not ident.session.approvals


async def test_reject_then_resume_does_not_run_and_cache_is_bounded():
    gw, _, ident = make()
    ad = OpenAIAgentsAdapter(gw, max_inflight=1)
    agent, r = await run(ad, ident, [tool_call("echo", {"text": "risky"}, "c7")])
    state = r.to_state()
    state.reject(r.interruptions[0], rejection_message="human said no")
    r2 = await Runner.run(agent, state, run_config=CFG)
    assert outputs(r2) == ["human said no"]
    assert len(ident.session.recent_calls) == 0
    assert "c7" in ad._inflight  # orphaned by the rejection...
    _, r3 = await run(ad, ident, [tool_call("echo", {"text": "hi"}, "c8")])
    assert outputs(r3) == ["echo:hi"]
    assert "c7" not in ad._inflight and not ad._inflight  # ...and evicted by the bound


async def test_user_needs_approval_is_honoured_when_gateway_allows():
    gw, _, ident = make()
    ad = OpenAIAgentsAdapter(gw)
    strict_echo = dataclasses.replace(echo, needs_approval=True)
    _, r = await run(ad, ident, [tool_call("echo", {"text": "hi"}, "c9")], tools=(strict_echo,))
    assert len(r.interruptions) == 1 and outputs(r) == []

    async def user_rule(run_ctx, params, call_id):
        return params.get("text") == "hi"

    dyn_echo = dataclasses.replace(echo, needs_approval=user_rule)
    _, r = await run(ad, ident, [tool_call("echo", {"text": "hi"}, "c9b")], tools=(dyn_echo,))
    assert len(r.interruptions) == 1
    _, r = await run(ad, ident, [tool_call("echo", {"text": "ok"}, "c9c")], tools=(dyn_echo,))
    assert outputs(r) == ["echo:ok"]


async def test_output_guards_rewrite_what_the_model_sees():
    gw, _, ident = make()
    ad = OpenAIAgentsAdapter(gw)
    _, r = await run(ad, ident, [tool_call("leak", {}, "c10")])
    assert outputs(r) == ["ssn [REDACTED]"]


async def test_direct_invoke_without_planning_step():
    gw, _, ident = make()
    ad = OpenAIAgentsAdapter(gw)
    gated = ad.gate_tool(echo)

    def tctx(call_id: str, args: dict) -> ToolContext:
        return ToolContext(
            context=ident, usage=Usage(), tool_name="echo", tool_call_id=call_id, tool_arguments=json.dumps(args)
        )

    assert await gated.on_invoke_tool(tctx("d1", {"text": "hi"}), json.dumps({"text": "hi"})) == "echo:hi"
    assert await gated.on_invoke_tool(tctx("d2", {"text": "raw:x"}), json.dumps({"text": "raw:x"})) == "echo:x"
    err = json.loads(await gated.on_invoke_tool(tctx("d3", {"text": "risky"}), json.dumps({"text": "risky"})))
    assert err["error"] == "approval_required" and err["retryable"] is True and err["approval_id"]
    err = json.loads(await gated.on_invoke_tool(tctx("d4", {"text": "deny"}), json.dumps({"text": "deny"})))
    assert err["error"] == "authorization_denied"
    assert not ad._inflight
