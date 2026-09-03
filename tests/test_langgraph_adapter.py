"""LangGraph adapter: end-to-end through a StateGraph + ToolNode with a scripted agent node."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace
from typing import Annotated, Any, TypedDict

import pytest

langgraph = pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402
from langgraph.types import Command  # noqa: E402

from agent_tool_gateway import (  # noqa: E402
    AgentIdentity,
    Gateway,
    InMemoryAuditSink,
    Principal,
    SessionState,
    SideEffect,
    ToolRegistry,
)
from agent_tool_gateway.adapters.langgraph import (  # noqa: E402
    DENIED_MESSAGE,
    LangGraphAdapter,
    default_identity,
    manifest_from_tool,
)
from agent_tool_gateway.stages import RulePolicy, TokenBucketLimiter, default_stages  # noqa: E402

# ------------------------------------------------------------- harness


@tool
def echo(text: str) -> str:
    """Echo the text back."""
    return f"echo:{text}"


@tool
def pay(amount: int) -> str:
    """Charge an amount."""
    return f"paid:{amount}"


@tool
def leak() -> str:
    """Return something that needs redaction."""
    return "ssn 123-45-6789"


TOOLS = [echo, pay, leak]


@dataclasses.dataclass
class Identity:
    principal: Principal
    agent: AgentIdentity
    session: SessionState


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def make(**session_kw):
    audit = InMemoryAuditSink()
    reg = ToolRegistry(
        [
            manifest_from_tool(echo, required_scopes=["echo"]),
            manifest_from_tool(pay, side_effect="write", cost_usd=0.01),
            manifest_from_tool(leak),
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


def build(tools_node, calls, *, context_schema=None):
    """StateGraph: scripted agent -> tools -> agent ... -> END, with a checkpointer for interrupts."""
    queue = list(calls)

    def agent(state):
        if queue:
            name, args, cid = queue.pop(0)
            return {"messages": [AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": cid}])]}
        return {"messages": [AIMessage(content="done")]}

    def route(state):
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    g = StateGraph(State, context_schema=context_schema)
    g.add_node("agent", agent)
    g.add_node("tools", tools_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, ["tools", END])
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=InMemorySaver())


def cfg(ident, thread: str) -> dict:
    return {"configurable": {"thread_id": thread, "gateway_identity": ident}}


def start() -> dict:
    return {"messages": [HumanMessage("go")]}


def tool_msgs(result) -> list[tuple[Any, str]]:
    return [(m.content, m.status) for m in result["messages"] if isinstance(m, ToolMessage)]


def final(result) -> str:
    return result["messages"][-1].content


# --------------------------------------------------------------- tests


def test_harness_runs_without_gateway():
    _, _, ident = make()
    g = build(ToolNode(TOOLS), [("echo", {"text": "hi"}, "c0")])
    r = g.invoke(start(), cfg(ident, "t0"))
    assert tool_msgs(r) == [("echo:hi", "success")] and final(r) == "done"
