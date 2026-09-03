# LangGraph Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run every tool call executed by a LangGraph `ToolNode` (and LangChain `create_agent`) through the gateway via `wrap_tool_call` / `awrap_tool_call`, mapping ALLOW / TRANSFORM / REQUIRE_APPROVAL / DENY onto `ToolMessage` results, `ToolCallRequest.override`, and `interrupt()` / `Command(resume=...)`.

**Architecture:** `LangGraphAdapter` exposes a sync and an async wrapper with the `(request, execute)` signature both `ToolNode` and `AgentMiddleware` use (same `ToolCallRequest` class). Shared helpers build the gateway context, turn errors into error `ToolMessage`s, apply TRANSFORM via `request.override`, and run `gateway.after` on the returned message. Approval calls `interrupt()`; on replay the resume value approves (`before_approved`) or denies. The sync wrapper drives gateway coroutines through `adapters.wrap._run_sync`.

**Tech Stack:** Python 3.11+, `langgraph` 1.2.x + `langchain-core` 1.6.x (spiked on 1.2.11 / 1.6.1), `langchain` 1.4.x for the middleware test only, pytest, a hand-built `StateGraph` with a scripted agent node and `InMemorySaver`.

Spec: `docs/superpowers/specs/2026-09-03-langgraph-adapter-design.md`.

---

## File structure

- Create `src/agent_tool_gateway/adapters/langgraph.py` — `default_identity`, `manifest_from_tool`, `LangGraphAdapter` (`wrap_tool_call`, `awrap_tool_call`, `tool_node`, `middleware`). All SDK imports lazy.
- Create `tests/test_langgraph_adapter.py` — graph harness plus all spec cases.
- Modify `pyproject.toml`, `README.md`, `docs/ARCHITECTURE.md`.
- Do not touch `adapters/__init__.py`.

Environment: the project venv (`uv venv --python 3.12`, `uv pip install -e ".[dev]"`).

---

### Task 1: Packaging and test harness

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/test_langgraph_adapter.py`

- [x] **Step 1: Add the extra**

Replace the optional-dependencies block in `pyproject.toml` with:

```toml
[project.optional-dependencies]
jsonschema = ["jsonschema>=4.0"]
claude = ["claude-agent-sdk"]
openai = ["openai-agents>=0.22,<0.23"]
pydantic = ["pydantic-ai-slim>=2.38,<2.39"]
langgraph = ["langgraph>=1.2,<2", "langchain-core>=1.6,<2"]
dev = [
  "pytest>=8", "pytest-asyncio>=0.23", "ruff>=0.5", "mypy>=1.10", "jsonschema>=4.0",
  "openai-agents>=0.22,<0.23", "pydantic-ai-slim>=2.38,<2.39",
  "langgraph>=1.2,<2", "langchain-core>=1.6,<2", "langchain>=1.4,<2",
]
```

- [x] **Step 2: Install**

Run: `uv pip install -e ".[dev]"`
Expected: `python -c "import langgraph, langchain; print('ok')"` prints `ok`.

- [x] **Step 3: Write the harness with one smoke test**

Create `tests/test_langgraph_adapter.py`:

```python
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
```

- [x] **Step 4: Run it**

Run: `pytest tests/test_langgraph_adapter.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'agent_tool_gateway.adapters.langgraph'`.

- [x] **Step 5: Commit**

```bash
git add pyproject.toml tests/test_langgraph_adapter.py
git commit -m "Add langgraph extra and StateGraph test harness for the LangGraph adapter

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Adapter module with the sync wrapper

**Files:**
- Create: `src/agent_tool_gateway/adapters/langgraph.py`
- Test: `tests/test_langgraph_adapter.py`

- [x] **Step 1: Write the failing tests**

Append to `tests/test_langgraph_adapter.py`:

```python
def test_default_identity_sources_and_error():
    _, _, ident = make()
    rt = SimpleNamespace(context=None, config={"configurable": {"gateway_identity": ident}})
    assert default_identity(rt) == (ident.principal, ident.agent, ident.session)
    rt = SimpleNamespace(context=ident, config={"configurable": {}})
    assert default_identity(rt) == (ident.principal, ident.agent, ident.session)
    with pytest.raises(RuntimeError, match="gateway_identity"):
        default_identity(SimpleNamespace(context={"x": 1}, config={"configurable": {"thread_id": "t"}}))


def test_manifest_from_tool_copies_schema_and_applies_overrides():
    m = manifest_from_tool(echo)
    assert m.name == "echo" and m.description == "Echo the text back."
    assert m.input_schema["required"] == ["text"] and m.side_effect is SideEffect.READ
    m2 = manifest_from_tool(echo, side_effect="write", required_scopes=["x"], cost_usd=0.5)
    assert m2.side_effect is SideEffect.WRITE and m2.required_scopes == frozenset({"x"}) and m2.cost_usd == 0.5


def test_allow_runs_tool_and_audits():
    gw, audit, ident = make()
    ad = LangGraphAdapter(gw)
    r = build(ad.tool_node(TOOLS), [("echo", {"text": "hi"}, "c1")]).invoke(start(), cfg(ident, "t1"))
    assert tool_msgs(r) == [("echo:hi", "success")] and final(r) == "done"
    assert [e.phase for e in audit.events] == ["decision", "execution"]
    assert audit.events[0].decision == "allow" and audit.events[0].tool == "echo"
    assert len(ident.session.recent_calls) == 1 and ident.session.turn == 1


def test_deny_returns_error_tool_message_and_run_continues():
    gw, _, ident = make()
    ad = LangGraphAdapter(gw)
    r = build(ad.tool_node(TOOLS), [("echo", {"text": "deny"}, "c2")]).invoke(start(), cfg(ident, "t2"))
    (content, status), = tool_msgs(r)
    err = json.loads(content)
    assert status == "error" and err["error"] == "authorization_denied" and "denied text" in err["message"]
    assert final(r) == "done"


def test_schema_deny_is_retryable_invalid_arguments():
    gw, _, ident = make()
    ad = LangGraphAdapter(gw)
    r = build(ad.tool_node(TOOLS), [("echo", {}, "c3")]).invoke(start(), cfg(ident, "t3"))
    err = json.loads(tool_msgs(r)[0][0])
    assert err["error"] == "invalid_arguments" and err["retryable"] is True


def test_unregistered_tool_is_denied_not_crashed():
    gw, _, ident = make()
    ad = LangGraphAdapter(gw)

    @tool
    def mystery() -> str:
        """Not in the registry."""
        return "ran"

    r = build(ad.tool_node([mystery]), [("mystery", {}, "c4")]).invoke(start(), cfg(ident, "t4"))
    assert json.loads(tool_msgs(r)[0][0])["error"] == "tool_not_registered" and final(r) == "done"


def test_transform_rewrites_arguments():
    gw, _, ident = make()
    ad = LangGraphAdapter(gw)
    r = build(ad.tool_node(TOOLS), [("echo", {"text": "raw:hi"}, "c5")]).invoke(start(), cfg(ident, "t5"))
    assert tool_msgs(r) == [("echo:hi", "success")]
```

- [x] **Step 2: Run to verify they fail**

Run: `pytest tests/test_langgraph_adapter.py -q`
Expected: collection error, module missing.

- [x] **Step 3: Create the module**

Create `src/agent_tool_gateway/adapters/langgraph.py`:

```python
"""Tier-2 adapter: LangGraph ``ToolNode`` and LangChain ``create_agent`` middleware.

One wrapper with the ``(request, execute)`` signature serves ``ToolNode(wrap_tool_call=...)``,
``create_react_agent(tools=ToolNode(...))`` and ``create_agent(middleware=[...])`` — they share
the same ``ToolCallRequest`` class.

    Decision.ALLOW             -> execute(request)
    Decision.TRANSFORM         -> execute(request.override(tool_call={..., "args": rewritten}))
    Decision.REQUIRE_APPROVAL  -> interrupt(payload); on resume True / {"approved": True} the call
                                  is re-evaluated with a one-shot approval and executed, anything
                                  else returns an error ToolMessage
    Decision.DENY              -> error ToolMessage with GatewayError.to_model_result() as JSON

Output guards run on the returned ``ToolMessage.content``. A ``Command`` returned by a tool is
passed through untouched.

Replay: LangGraph re-executes the node on ``Command(resume=...)``, so the wrapper runs from the
top again — ``before`` asks again, ``interrupt()`` returns the resume value, and
``before_approved`` re-checks budget and rate limits. ``session.turn`` and the
"require_approval" audit event therefore occur twice for an approved call. Interrupts need a
checkpointer and ``configurable.thread_id``.

Identity: ``runtime.context`` with ``principal`` / ``agent`` / ``session`` attributes, else
``config["configurable"]["gateway_identity"]`` with the same attributes. Pass ``identity=`` for
other shapes. Cross-process resume still needs the same ``SessionState`` object on both sides.

Usage:

    adapter = LangGraphAdapter(gateway)
    graph.add_node("tools", adapter.tool_node([read_file, send_email]))
    # or: create_agent(model, tools, middleware=[adapter.middleware()], checkpointer=...)
    graph.invoke(state, {"configurable": {"thread_id": "t", "gateway_identity": Identity(...)}})
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from ..context import AgentIdentity, Principal, SessionState, ToolCallContext, ToolResult
from ..decision import Decision, DecisionResult
from ..errors import GatewayError
from ..manifest import ToolManifest
from ..pipeline import Gateway
from .wrap import _run_sync

IdentityProvider = Callable[[Any], tuple[Principal, AgentIdentity, SessionState]]
"""Receives the ToolRuntime and returns the gateway identity."""

CONFIG_KEY = "gateway_identity"
DENIED_MESSAGE = "The tool call was denied."
_DECISION_KEY = "hook_decision"


def default_identity(runtime: Any) -> tuple[Principal, AgentIdentity, SessionState]:
    """``runtime.context`` first, then ``config["configurable"]["gateway_identity"]``."""
    config: Any = getattr(runtime, "config", None) or {}
    sources = (getattr(runtime, "context", None), (config.get("configurable") or {}).get(CONFIG_KEY))
    for source in sources:
        if source is None:
            continue
        try:
            return source.principal, source.agent, source.session
        except AttributeError:
            continue
    raise RuntimeError(
        "no gateway identity: put an object with .principal/.agent/.session in the graph context or in "
        f"config['configurable']['{CONFIG_KEY}'], or pass identity=..."
    )


def manifest_from_tool(tool: Any, **overrides: Any) -> ToolManifest:
    """Build a manifest from a LangChain ``BaseTool``'s name, description and call schema."""
    schema_model = getattr(tool, "tool_call_schema", None)
    if schema_model is not None and hasattr(schema_model, "model_json_schema"):
        schema = dict(schema_model.model_json_schema())
    else:
        schema = {"type": "object", "properties": dict(getattr(tool, "args", None) or {})}
    fields: dict[str, Any] = {"name": tool.name, "description": tool.description or "", "input_schema": schema}
    fields.update(overrides)
    return ToolManifest.from_dict(fields)


def _approved(answer: Any) -> bool:
    return answer is True or (isinstance(answer, Mapping) and answer.get("approved") is True)


def _denial_message(answer: Any) -> str:
    if isinstance(answer, Mapping) and answer.get("message"):
        return str(answer["message"])
    return DENIED_MESSAGE


class LangGraphAdapter:
    def __init__(self, gateway: Gateway, identity: IdentityProvider = default_identity) -> None:
        self.gateway = gateway
        self.identity = identity

    # ------------------------------------------------------------ builders
    def tool_node(self, tools: Any, **kwargs: Any) -> Any:
        """``ToolNode(tools, wrap_tool_call=..., awrap_tool_call=..., **kwargs)``."""
        from langgraph.prebuilt import ToolNode

        return ToolNode(tools, wrap_tool_call=self.wrap_tool_call, awrap_tool_call=self.awrap_tool_call, **kwargs)

    def middleware(self) -> Any:
        """An ``AgentMiddleware`` for LangChain ``create_agent`` that delegates to this adapter."""
        from langchain.agents.middleware import AgentMiddleware

        adapter = self

        class GatewayMiddleware(AgentMiddleware):  # type: ignore[type-arg]
            name = "agent_tool_gateway"

            def wrap_tool_call(self, request: Any, handler: Any) -> Any:
                return adapter.wrap_tool_call(request, handler)

            async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
                return await adapter.awrap_tool_call(request, handler)

        return GatewayMiddleware()

    # ------------------------------------------------------------ wrappers
    def wrap_tool_call(self, request: Any, execute: Callable[[Any], Any]) -> Any:
        """Sync wrapper for ``ToolNode.invoke`` / ``AgentMiddleware.wrap_tool_call``."""
        try:
            ctx = self._context(request)
        except GatewayError as e:
            return self._error_message(request, e)
        decision = _run_sync(self.gateway.before(ctx))
        if decision.decision is Decision.REQUIRE_APPROVAL:
            answer = self._ask(ctx, decision)  # interrupt() on this thread, not inside the coroutine
            if not _approved(answer):
                return self._text_error(request, _denial_message(answer))
            decision = _run_sync(self.gateway.before_approved(ctx))
        blocked = self._check(request, ctx, decision)
        if blocked is not None:
            return blocked
        try:
            msg = execute(self._apply(request, ctx, decision))
        except BaseException:
            self.gateway.release(ctx)
            raise
        try:
            return _run_sync(self._finish(ctx, msg))
        except GatewayError as e:
            return self._error_message(request, e)

    async def awrap_tool_call(self, request: Any, execute: Callable[[Any], Any]) -> Any:
        """Async wrapper for ``ToolNode.ainvoke`` / ``AgentMiddleware.awrap_tool_call``."""
        try:
            ctx = self._context(request)
        except GatewayError as e:
            return self._error_message(request, e)
        decision = await self.gateway.before(ctx)
        if decision.decision is Decision.REQUIRE_APPROVAL:
            answer = self._ask(ctx, decision)
            if not _approved(answer):
                return self._text_error(request, _denial_message(answer))
            decision = await self.gateway.before_approved(ctx)
        blocked = self._check(request, ctx, decision)
        if blocked is not None:
            return blocked
        try:
            msg = await execute(self._apply(request, ctx, decision))
        except BaseException:
            self.gateway.release(ctx)
            raise
        try:
            return await self._finish(ctx, msg)
        except GatewayError as e:
            return self._error_message(request, e)

    # ------------------------------------------------------------- helpers
    def _context(self, request: Any) -> ToolCallContext:
        tc = request.tool_call
        runtime = request.runtime
        principal, agent, session = self.identity(runtime)
        session.turn += 1
        return self.gateway.build_context(
            tc["name"],
            dict(tc.get("args") or {}),
            principal=principal,
            agent=agent,
            session=session,
            tool_call_id=tc.get("id") or getattr(runtime, "tool_call_id", None),
        )

    def _ask(self, ctx: ToolCallContext, decision: DecisionResult) -> Any:
        from langgraph.types import interrupt

        return interrupt(
            {
                "approval_id": decision.approval_id,
                "reason": decision.reason,
                "tool": ctx.tool.name,
                "args": dict(ctx.args),
                "tool_call_id": ctx.tool_call_id,
            }
        )

    def _check(self, request: Any, ctx: ToolCallContext, decision: DecisionResult) -> Any:
        """Record the decision; return an error ToolMessage if it blocks, else None."""
        ctx.metadata[_DECISION_KEY] = decision.decision.value
        if not decision.blocked:
            return None
        try:
            Gateway.raise_for_decision(decision)
        except GatewayError as e:
            return self._error_message(request, e)
        return None  # pragma: no cover

    @staticmethod
    def _apply(request: Any, ctx: ToolCallContext, decision: DecisionResult) -> Any:
        if decision.decision is Decision.TRANSFORM:
            return request.override(tool_call={**request.tool_call, "args": dict(ctx.args)})
        return request

    async def _finish(self, ctx: ToolCallContext, msg: Any) -> Any:
        from langchain_core.messages import ToolMessage

        if not isinstance(msg, ToolMessage):  # a Command from a state-updating tool
            self.gateway.settle(ctx)
            return msg
        result = await self.gateway.after(ctx, ToolResult(content=msg.content))
        msg.content = result.content
        return msg

    @staticmethod
    def _error_message(request: Any, err: GatewayError) -> Any:
        return LangGraphAdapter._text_error(request, json.dumps(err.to_model_result(), default=str))

    @staticmethod
    def _text_error(request: Any, text: str) -> Any:
        from langchain_core.messages import ToolMessage

        tc = request.tool_call
        return ToolMessage(content=text, tool_call_id=tc["id"], name=tc["name"], status="error")
```

- [x] **Step 4: Run the tests**

Run: `pytest tests/test_langgraph_adapter.py -q`
Expected: 8 pass.

- [x] **Step 5: Lint and type-check**

Run: `ruff check src tests examples && mypy src`
Expected: clean. If mypy objects to the `type: ignore[type-arg]` as unused, delete that comment.

- [x] **Step 6: Commit**

```bash
git add src/agent_tool_gateway/adapters/langgraph.py tests/test_langgraph_adapter.py
git commit -m "LangGraph adapter: ToolNode wrap_tool_call with allow/deny/transform through the gateway

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Approvals, async path, context identity

**Files:**
- Test: `tests/test_langgraph_adapter.py`
- Modify (only if a test fails): `src/agent_tool_gateway/adapters/langgraph.py`

- [x] **Step 1: Write the tests**

Append to `tests/test_langgraph_adapter.py`:

```python
def test_require_approval_interrupts_without_running_or_reserving():
    gw, audit, ident = make(budget_limit_usd=0.05)
    ad = LangGraphAdapter(gw)
    r = build(ad.tool_node(TOOLS), [("pay", {"amount": 3}, "c6")]).invoke(start(), cfg(ident, "t6"))
    (intr,) = r["__interrupt__"]
    assert intr.value["tool"] == "pay" and intr.value["args"] == {"amount": 3} and intr.value["approval_id"]
    assert intr.value["reason"] == "money" and intr.value["tool_call_id"] == "c6"
    assert tool_msgs(r) == []
    assert ident.session.budget_reserved_usd == 0.0 and len(ident.session.recent_calls) == 0
    assert audit.events[-1].decision == "require_approval"


def test_resume_true_runs_tool_and_settles_budget():
    gw, audit, ident = make(budget_limit_usd=0.05)
    ad = LangGraphAdapter(gw)
    g = build(ad.tool_node(TOOLS), [("pay", {"amount": 3}, "c7")])
    c = cfg(ident, "t7")
    g.invoke(start(), c)
    r = g.invoke(Command(resume=True), c)
    assert tool_msgs(r) == [("paid:3", "success")] and final(r) == "done"
    assert ident.session.budget_used_usd == pytest.approx(0.01) and ident.session.budget_reserved_usd == 0.0
    assert len(ident.session.recent_calls) == 1
    assert audit.events[-1].phase == "execution" and audit.events[-1].error_code is None


def test_resume_after_budget_exhausted_denies():
    gw, _, ident = make(budget_limit_usd=0.05)
    ad = LangGraphAdapter(gw)
    g = build(ad.tool_node(TOOLS), [("pay", {"amount": 3}, "c8")])
    c = cfg(ident, "t8")
    g.invoke(start(), c)
    ident.session.budget_used_usd = 0.05
    r = g.invoke(Command(resume=True), c)
    (content, status), = tool_msgs(r)
    assert status == "error" and json.loads(content)["error"] == "budget_exceeded"
    assert len(ident.session.recent_calls) == 0


def test_resume_false_and_resume_with_message_deny():
    gw, _, ident = make()
    ad = LangGraphAdapter(gw)
    g = build(ad.tool_node(TOOLS), [("echo", {"text": "risky"}, "c9")])
    c = cfg(ident, "t9")
    g.invoke(start(), c)
    r = g.invoke(Command(resume=False), c)
    assert tool_msgs(r) == [(DENIED_MESSAGE, "error")] and final(r) == "done"

    g = build(ad.tool_node(TOOLS), [("echo", {"text": "risky"}, "c9b")])
    c = cfg(ident, "t9b")
    g.invoke(start(), c)
    r = g.invoke(Command(resume={"approved": False, "message": "nope"}), c)
    assert tool_msgs(r) == [("nope", "error")]
    assert len(ident.session.recent_calls) == 0


async def test_async_path_allow_and_approval_round_trip():
    gw, _, ident = make(budget_limit_usd=0.05)
    ad = LangGraphAdapter(gw)
    r = await build(ad.tool_node(TOOLS), [("echo", {"text": "raw:hi"}, "c10")]).ainvoke(start(), cfg(ident, "t10"))
    assert tool_msgs(r) == [("echo:hi", "success")]

    g = build(ad.tool_node(TOOLS), [("pay", {"amount": 3}, "c11")])
    c = cfg(ident, "t11")
    r = await g.ainvoke(start(), c)
    assert r["__interrupt__"][0].value["tool"] == "pay"
    r = await g.ainvoke(Command(resume=True), c)
    assert tool_msgs(r) == [("paid:3", "success")]
    assert ident.session.budget_used_usd == pytest.approx(0.01)


def test_identity_from_graph_context():
    gw, audit, ident = make()
    ad = LangGraphAdapter(gw)
    g = build(ad.tool_node(TOOLS), [("echo", {"text": "hi"}, "c12")], context_schema=Identity)
    r = g.invoke(start(), {"configurable": {"thread_id": "t12"}}, context=ident)
    assert tool_msgs(r) == [("echo:hi", "success")] and audit.events[0].principal == "u"
```

- [x] **Step 2: Run them**

Run: `pytest tests/test_langgraph_adapter.py -q`
Expected: all pass. The spike verified `interrupt` inside `wrap_tool_call`, `__interrupt__` in the invoke result, and `Command(resume=...)` re-running the wrapper on langgraph 1.2.11. If `test_identity_from_graph_context` fails because `runtime.context` is a dict rather than the dataclass, check how `context_schema` materialises the context in this version and adapt `default_identity` to also accept a mapping with the three keys.

- [x] **Step 3: Commit**

```bash
git add tests/test_langgraph_adapter.py src/agent_tool_gateway/adapters/langgraph.py
git commit -m "LangGraph adapter: interrupt/resume approvals, async path, context identity tests

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Output guards and `create_agent` middleware

**Files:**
- Test: `tests/test_langgraph_adapter.py`
- Modify (only if a test fails): `src/agent_tool_gateway/adapters/langgraph.py`

- [x] **Step 1: Write the tests**

Append to `tests/test_langgraph_adapter.py`:

```python
def test_output_guards_rewrite_what_the_model_sees():
    gw, _, ident = make()
    ad = LangGraphAdapter(gw)
    r = build(ad.tool_node(TOOLS), [("leak", {}, "c13")]).invoke(start(), cfg(ident, "t13"))
    assert tool_msgs(r) == [("ssn [REDACTED]", "success")]


def test_tool_exception_releases_reservation_and_propagates():
    gw, _, ident = make(budget_limit_usd=0.05)
    ad = LangGraphAdapter(gw)

    @tool
    def boom() -> str:
        """Always fails."""
        raise RuntimeError("kaboom")

    gw.registry.register(manifest_from_tool(boom, cost_usd=0.01))
    gw.stages[2].policy.allow("boom")
    g = build(ad.tool_node([boom], handle_tool_errors=False), [("boom", {}, "c14")])
    with pytest.raises(RuntimeError, match="kaboom"):
        g.invoke(start(), cfg(ident, "t14"))
    assert ident.session.budget_reserved_usd == 0.0 and ident.session.budget_used_usd == 0.0


def test_create_agent_middleware_end_to_end():
    langchain = pytest.importorskip("langchain")
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    class ScriptedChat(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    gw, _, ident = make()
    ad = LangGraphAdapter(gw)
    model = ScriptedChat(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "echo", "args": {"text": "deny"}, "id": "m1"}]),
            AIMessage(content="", tool_calls=[{"name": "echo", "args": {"text": "raw:hi"}, "id": "m2"}]),
            AIMessage(content="done"),
        ]
    )
    agent = create_agent(model, TOOLS, middleware=[ad.middleware()], checkpointer=InMemorySaver())
    r = agent.invoke(start(), cfg(ident, "t15"))
    msgs = tool_msgs(r)
    assert json.loads(msgs[0][0])["error"] == "authorization_denied" and msgs[0][1] == "error"
    assert msgs[1] == ("echo:hi", "success") and final(r) == "done"
    assert langchain is not None
```

- [x] **Step 2: Run everything**

Run: `pytest -q && ruff check src tests examples && mypy src`
Expected: all pass (111 existing + 1 skipped + 17 new), ruff and mypy clean.

If `create_agent` rejects the fake model (e.g. requires a `profile` or a provider string), replace the body of `test_create_agent_middleware_end_to_end` after the `make()` line with a delegation check:

```python
    mw = ad.middleware()
    seen = {}

    def handler(request):
        seen["request"] = request
        return ToolMessage(content="ok", tool_call_id="x", name="echo")

    req = SimpleNamespace(
        tool_call={"name": "echo", "args": {"text": "hi"}, "id": "x"},
        runtime=SimpleNamespace(context=None, config=cfg(ident, "t15"), tool_call_id="x"),
        override=lambda **kw: req,
    )
    assert mw.wrap_tool_call(req, handler).content == "ok" and seen["request"] is req
```

and note the reason in the test's docstring.

- [x] **Step 3: Commit**

```bash
git add tests/test_langgraph_adapter.py src/agent_tool_gateway/adapters/langgraph.py
git commit -m "LangGraph adapter: output guards, failure release, create_agent middleware tests

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/ARCHITECTURE.md`

- [x] **Step 1: README adapter section**

Directly before `## Loading tools on demand` in `README.md`, insert:

````markdown
### LangGraph / LangChain adapter

```python
from langgraph.types import Command
from agent_tool_gateway.adapters.langgraph import LangGraphAdapter, manifest_from_tool

adapter = LangGraphAdapter(gw)
graph.add_node("tools", adapter.tool_node([read_file, send_email]))       # or create_react_agent(model, tools=adapter.tool_node(...))
# LangChain create_agent: create_agent(model, tools, middleware=[adapter.middleware()], checkpointer=...)

config = {"configurable": {"thread_id": "t1", "gateway_identity": Identity(principal, agent_id, session)}}
result = graph.invoke(state, config)
if result.get("__interrupt__"):                                           # REQUIRE_APPROVAL
    result = graph.invoke(Command(resume=True), config)                   # or resume={"approved": False, "message": "..."}
```

`DENY` becomes an error `ToolMessage` carrying the structured error; `TRANSFORM` rewrites the arguments via `ToolCallRequest.override`; `REQUIRE_APPROVAL` calls `interrupt()`, so a checkpointer and `thread_id` are required, and the approved resume is re-evaluated (a budget that ran out while waiting still denies). Output guards rewrite `ToolMessage.content`. Identity can also come from the graph context (`context_schema` + `context=`).
````

- [x] **Step 2: README tier table and roadmap**

Replace the tier-2 row with:

```markdown
| 2 | Toolset wrappers | Pydantic AI, LangGraph / LangChain, Strands, Agno, Google ADK | Pydantic AI ✅ · LangGraph ✅ · others planned |
```

Replace `- [ ] LangGraph toolset adapter` with `- [x] LangGraph / LangChain adapter (`ToolNode.wrap_tool_call` + `interrupt` approvals)`.

- [x] **Step 3: ARCHITECTURE adapter rule 3**

Replace the rule-3 text so it reads:

```markdown
3. `REQUIRE_APPROVAL` maps to the framework's native approval mechanism where one exists
   (Claude Agent SDK `ask` → `can_use_tool`; OpenAI Agents SDK `needs_approval` → `RunState`
   interruption; Pydantic AI `ApprovalRequired` → `DeferredToolRequests`; LangGraph `interrupt()`
   → `Command(resume=...)`); otherwise to a structured `approval_required` error the host
   application handles.
```

- [x] **Step 4: Verify**

Run: `pytest -q && ruff check src tests examples && python examples/claude_sdk_coding_agent.py > /dev/null && echo ok`
Expected: tests pass, ruff clean, `ok`.

- [x] **Step 5: Commit**

```bash
git add README.md docs/ARCHITECTURE.md
git commit -m "Document the LangGraph adapter

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review

- **Spec coverage:** decision table, override-based TRANSFORM, interrupt payload and resume contract, `before_approved` on approval, `Command` pass-through, release on exception → Task 2 code; approvals/replay/async/context identity → Task 3; output guards, failure propagation, middleware → Task 4; `default_identity`, `manifest_from_tool` → Task 2; packaging → Task 1; docs → Task 5. Spec cases 1–15 all mapped (case 15 with its documented fallback).
- **Placeholders:** none.
- **Type consistency:** `LangGraphAdapter(gateway, identity=default_identity)`, `wrap_tool_call(request, execute)`, `awrap_tool_call(request, execute)`, `tool_node(tools, **kwargs)`, `middleware()`, `manifest_from_tool(tool, **overrides)`, `default_identity(runtime)`, `DENIED_MESSAGE`, `CONFIG_KEY` used consistently. `gw.stages[2]` is `PolicyStage` in `default_stages`.
