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
