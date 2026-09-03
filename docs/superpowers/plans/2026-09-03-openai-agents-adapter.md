# OpenAI Agents SDK Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run every `FunctionTool` call of an OpenAI Agents SDK agent through the gateway, mapping ALLOW / TRANSFORM / REQUIRE_APPROVAL / DENY onto the SDK's native `needs_approval` interruption and a wrapped `on_invoke_tool`.

**Architecture:** `OpenAIAgentsAdapter.gate_tool` returns a `dataclasses.replace` copy of a `FunctionTool` with `needs_approval` pointing at the adapter (where `gateway.before` runs once per call, at planning time, cached by `call_id`) and `on_invoke_tool` pointing at a wrapper that applies the cached decision, calls the original invoker, then runs `gateway.after` on the output. The SDK is imported lazily inside functions. No policy lives in the adapter.

**Tech Stack:** Python 3.11+, `openai-agents` 0.22.x (spiked against 0.22.0), pytest with `pytest.importorskip`, a scripted `Model` so `Runner.run` executes end to end without network.

Spec: `docs/superpowers/specs/2026-09-03-openai-agents-adapter-design.md`.

---

## File structure

- Create `src/agent_tool_gateway/adapters/openai_agents.py` — the adapter: `default_identity`, `OpenAIAgentsAdapter`, `gate_tools`, `manifest_from_function_tool`. One file, mirrors `adapters/claude_agent_sdk.py`.
- Create `tests/test_openai_agents_adapter.py` — scripted-model harness plus all cases from the spec.
- Modify `pyproject.toml` — `openai` extra, added to `dev`.
- Modify `README.md` — adapter section, integration tier table, roadmap.
- Do not touch `adapters/__init__.py` (no eager import; keeps `agents` optional).

Environment note: system `python3` is 3.9. Use the project venv created for this repo (`uv venv --python 3.12`, then `uv pip install -e ".[dev]"`). All commands below assume `python`, `pytest`, `ruff`, `mypy` resolve to that venv.

---

### Task 1: Packaging and test harness

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/test_openai_agents_adapter.py`

- [ ] **Step 1: Add the extra**

In `pyproject.toml`, replace the optional-dependencies block with:

```toml
[project.optional-dependencies]
jsonschema = ["jsonschema>=4.0"]
claude = ["claude-agent-sdk"]
openai = ["openai-agents>=0.22,<0.23"]
dev = ["pytest>=8", "pytest-asyncio>=0.23", "ruff>=0.5", "mypy>=1.10", "jsonschema>=4.0", "openai-agents>=0.22,<0.23"]
```

- [ ] **Step 2: Install it**

Run: `uv pip install -e ".[dev]"` (or `pip install -e ".[dev]"`)
Expected: `openai-agents` 0.22.x installed; `python -c "import agents; print(agents.__name__)"` prints `agents`.

- [ ] **Step 3: Write the harness with one smoke test**

Create `tests/test_openai_agents_adapter.py`:

```python
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
```

- [ ] **Step 4: Run it**

Run: `pytest tests/test_openai_agents_adapter.py -q`
Expected: 1 error at collection: `ModuleNotFoundError: No module named 'agent_tool_gateway.adapters.openai_agents'`. That is the failing state for Task 2. (If the SDK were missing the whole file would skip instead.)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/test_openai_agents_adapter.py
git commit -m "Add openai extra and scripted-model test harness for the OpenAI Agents adapter

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `default_identity` and `manifest_from_function_tool`

**Files:**
- Create: `src/agent_tool_gateway/adapters/openai_agents.py`
- Test: `tests/test_openai_agents_adapter.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_openai_agents_adapter.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_openai_agents_adapter.py -q`
Expected: collection error, module missing.

- [ ] **Step 3: Create the module with only these two functions**

Create `src/agent_tool_gateway/adapters/openai_agents.py`:

```python
"""Tier-1 adapter: OpenAI Agents SDK.

Wraps each ``FunctionTool`` so the gateway decides before the tool runs:

    Decision.ALLOW             -> tool runs with the model's arguments
    Decision.TRANSFORM         -> tool runs with the rewritten arguments
    Decision.REQUIRE_APPROVAL  -> ``needs_approval`` returns True; the run interrupts with a
                                  ``ToolApprovalItem`` for ``RunState.approve`` / ``reject``
    Decision.DENY              -> the tool does not run; the model receives the structured
                                  error from ``GatewayError.to_model_result()`` and continues

``gateway.before`` runs exactly once per call, inside the SDK's planning-time
``needs_approval`` hook, and the resulting context is cached by ``call_id`` for the
wrapped ``on_invoke_tool``. Output guards run in the wrapper via ``gateway.after``.

Identity comes from ``RunContextWrapper.context``: by default the object you pass as
``Runner.run(..., context=...)`` must expose ``principal``, ``agent`` and ``session``.
Pass ``identity=`` for any other shape.

Out of scope: hosted tools (never enter the process), cross-process resume
(``SessionState`` is in-memory), and gateway ``timeout_s`` (the SDK owns tool timeouts).
The ``agents`` package is imported lazily so the core stays dependency-free.

Usage:

    adapter = OpenAIAgentsAdapter(gateway)
    agent = Agent(name="a", tools=adapter.gate_tools([read_file, send_email]))
    result = await Runner.run(agent, prompt, context=Identity(principal, agent_id, session))
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..context import AgentIdentity, Principal, SessionState
from ..manifest import ToolManifest

IdentityProvider = Callable[[Any], tuple[Principal, AgentIdentity, SessionState]]
"""Receives the SDK's RunContextWrapper (or ToolContext) and returns the gateway identity."""


def default_identity(run_ctx: Any) -> tuple[Principal, AgentIdentity, SessionState]:
    """Read ``principal`` / ``agent`` / ``session`` attributes from ``run_ctx.context``."""
    obj = getattr(run_ctx, "context", None)
    try:
        return obj.principal, obj.agent, obj.session
    except AttributeError:
        raise RuntimeError(
            "run context must expose .principal, .agent and .session; pass identity=... for other shapes"
        ) from None


def manifest_from_function_tool(tool: Any, **overrides: Any) -> ToolManifest:
    """Build a manifest from a ``FunctionTool``'s name, description and JSON schema.

    ``overrides`` take the same values as ``ToolManifest.from_dict`` (enum names or
    values, lists for scopes/tags).
    """
    fields: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": dict(tool.params_json_schema or {}),
    }
    fields.update(overrides)
    return ToolManifest.from_dict(fields)
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_openai_agents_adapter.py -q`
Expected: the two new tests pass; `test_harness_runs_without_gateway` passes; collection fails on `OpenAIAgentsAdapter` / `gate_tools` import. To confirm just these two, temporarily run with `-k "identity or manifest"` after commenting nothing: the import error blocks the file, so instead verify by `python -c "from agent_tool_gateway.adapters.openai_agents import default_identity, manifest_from_function_tool; print('ok')"` → `ok`. The full file goes green in Task 3.

- [ ] **Step 5: Commit**

```bash
git add src/agent_tool_gateway/adapters/openai_agents.py tests/test_openai_agents_adapter.py
git commit -m "OpenAI adapter: default_identity and manifest_from_function_tool

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `OpenAIAgentsAdapter` with allow / deny / schema / transform

**Files:**
- Modify: `src/agent_tool_gateway/adapters/openai_agents.py`
- Test: `tests/test_openai_agents_adapter.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_openai_agents_adapter.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_openai_agents_adapter.py -q`
Expected: collection error `ImportError: cannot import name 'OpenAIAgentsAdapter'`.

- [ ] **Step 3: Implement the adapter**

Replace the `from __future__` line through the end of the imports block in `src/agent_tool_gateway/adapters/openai_agents.py` with:

```python
from __future__ import annotations

import dataclasses
import inspect
import json
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any

from ..context import AgentIdentity, Principal, SessionState, ToolCallContext, ToolResult
from ..decision import Decision, DecisionResult
from ..errors import GatewayError
from ..manifest import ToolManifest
from ..pipeline import Gateway

IdentityProvider = Callable[[Any], tuple[Principal, AgentIdentity, SessionState]]
"""Receives the SDK's RunContextWrapper (or ToolContext) and returns the gateway identity."""

_DECISION_KEY = "hook_decision"  # same key the Claude adapter uses
_RESULT_KEY = "_decision_result"  # DecisionResult cached between needs_approval and invoke
```

Then append after `manifest_from_function_tool`:

```python
def _error_json(err: GatewayError) -> str:
    return json.dumps(err.to_model_result(), default=str)


def _parse_args(args_json: str) -> dict[str, Any]:
    try:
        parsed = json.loads(args_json or "{}")
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class OpenAIAgentsAdapter:
    def __init__(self, gateway: Gateway, identity: IdentityProvider = default_identity, *, max_inflight: int = 256) -> None:
        self.gateway = gateway
        self.identity = identity
        self.max_inflight = max_inflight
        # call_id -> context built at planning time, or the GatewayError that prevented building one.
        # Bounded: a rejected approval never reaches invoke, so entries can be orphaned.
        self._inflight: OrderedDict[str, ToolCallContext | GatewayError] = OrderedDict()

    def _remember(self, call_id: str, entry: ToolCallContext | GatewayError) -> None:
        self._inflight[call_id] = entry
        while len(self._inflight) > self.max_inflight:
            self._inflight.popitem(last=False)

    # -------------------------------------------------------------- wrapping
    def gate_tool(self, tool: Any) -> Any:
        """Return a copy of ``tool`` (a ``FunctionTool``) whose calls go through the gateway."""
        from agents import FunctionTool

        if not isinstance(tool, FunctionTool):
            raise TypeError(f"gate_tool expects a FunctionTool, got {type(tool).__name__}")
        original_invoke = tool.on_invoke_tool
        user_needs = tool.needs_approval
        tool_name = tool.name

        async def needs_approval(run_ctx: Any, params: dict[str, Any], call_id: str) -> bool:
            gateway_needs = await self.needs_approval(tool_name, run_ctx, params, call_id)
            if user_needs is True:
                return True
            if callable(user_needs):
                maybe = user_needs(run_ctx, params, call_id)
                if inspect.isawaitable(maybe):
                    maybe = await maybe
                return bool(maybe) or gateway_needs
            return gateway_needs

        async def on_invoke(tool_ctx: Any, args_json: str) -> Any:
            return await self.invoke(tool_name, original_invoke, tool_ctx, args_json)

        return dataclasses.replace(tool, needs_approval=needs_approval, on_invoke_tool=on_invoke)

    def gate_tools(self, tools: Sequence[Any]) -> list[Any]:
        """Wrap every ``FunctionTool``; hosted and other tool types pass through untouched."""
        from agents import FunctionTool

        return [self.gate_tool(t) if isinstance(t, FunctionTool) else t for t in tools]

    # ---------------------------------------------------------------- hooks
    async def needs_approval(self, tool_name: str, run_ctx: Any, params: dict[str, Any], call_id: str) -> bool:
        """Planning-time hook: run ``before`` once, cache the outcome, interrupt on REQUIRE_APPROVAL."""
        principal, agent, session = self.identity(run_ctx)
        session.turn += 1
        try:
            ctx = self.gateway.build_context(
                tool_name, params, principal=principal, agent=agent, session=session, tool_call_id=call_id
            )
        except GatewayError as e:
            self._remember(call_id, e)
            return False
        decision = await self.gateway.before(ctx)
        ctx.metadata[_DECISION_KEY] = decision.decision.value
        ctx.metadata[_RESULT_KEY] = decision
        self._remember(call_id, ctx)
        return decision.decision is Decision.REQUIRE_APPROVAL

    async def invoke(self, tool_name: str, original_invoke: Any, tool_ctx: Any, args_json: str) -> Any:
        """Execution-time wrapper: apply the cached decision, run the tool, run output guards."""
        call_id = getattr(tool_ctx, "tool_call_id", None) or ""
        entry = self._inflight.pop(call_id, None)
        if isinstance(entry, GatewayError):
            return _error_json(entry)

        ctx = entry
        if ctx is None:
            # No planning step ran (tool invoked directly). Decide inline; an approval can no
            # longer interrupt the run, so it surfaces as an approval_required error instead.
            principal, agent, session = self.identity(tool_ctx)
            try:
                ctx = self.gateway.build_context(
                    tool_name, _parse_args(args_json), principal=principal, agent=agent, session=session,
                    tool_call_id=call_id or None,
                )
                decision = await self.gateway.before(ctx)
                Gateway.raise_for_decision(decision)
            except GatewayError as e:
                return _error_json(e)
        else:
            decision = ctx.metadata.pop(_RESULT_KEY)
            if decision.decision is Decision.DENY:
                try:
                    Gateway.raise_for_decision(decision)
                except GatewayError as e:
                    return _error_json(e)
            elif decision.decision is Decision.REQUIRE_APPROVAL:
                # The host approved via RunState.approve; the gateway never saw the call start.
                self.gateway.reserve(ctx)

        payload = json.dumps(ctx.args, default=str) if decision.decision is Decision.TRANSFORM else args_json
        raw = await original_invoke(tool_ctx, payload)
        try:
            result = await self.gateway.after(ctx, ToolResult(content=raw))
        except GatewayError as e:
            return _error_json(e)
        return result.content


def gate_tools(gateway: Gateway, tools: Sequence[Any], identity: IdentityProvider = default_identity) -> list[Any]:
    """One-shot convenience: ``OpenAIAgentsAdapter(gateway, identity).gate_tools(tools)``."""
    return OpenAIAgentsAdapter(gateway, identity).gate_tools(tools)
```

Note for the implementer: `DecisionResult` is imported for the type of `decision`; if ruff flags it unused, annotate `decision: DecisionResult` at first assignment in `invoke` rather than removing the import.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_openai_agents_adapter.py -q`
Expected: all 9 tests pass. If `test_schema_deny_is_retryable_invalid_arguments` shows the SDK rejecting `{}` before the hook runs (output is not gateway JSON), change the call to `tool_call("echo", {"text": 5}, "c3")` — the type mismatch also fails the schema stage.

- [ ] **Step 5: Lint and type-check**

Run: `ruff check src tests examples && mypy src`
Expected: both clean. If mypy complains about `decision` possibly unbound in `invoke`, add `decision: DecisionResult` before the `if ctx is None:` branch and assign in both branches (the code above already assigns in both).

- [ ] **Step 6: Commit**

```bash
git add src/agent_tool_gateway/adapters/openai_agents.py tests/test_openai_agents_adapter.py
git commit -m "OpenAI adapter: gate FunctionTools with allow/deny/transform through the gateway

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Approvals — interrupt, approve, reject, eviction, user `needs_approval`

**Files:**
- Test: `tests/test_openai_agents_adapter.py`
- Modify (only if a test fails): `src/agent_tool_gateway/adapters/openai_agents.py`

- [ ] **Step 1: Write the tests**

Append to `tests/test_openai_agents_adapter.py`:

```python
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
```

- [ ] **Step 2: Run them**

Run: `pytest tests/test_openai_agents_adapter.py -q`
Expected: all pass with the Task 3 implementation. The spike verified interrupt → `to_state` → `approve` / `reject` → `Runner.run(agent, state)` with a replaced `on_invoke_tool` on SDK 0.22.0.

If `test_approve_then_resume_runs_tool_and_settles_budget` fails on `budget_used_usd`, the `REQUIRE_APPROVAL` branch in `invoke` did not call `gateway.reserve(ctx)` before the tool ran; `after` settles only what was reserved.

- [ ] **Step 3: Commit**

```bash
git add tests/test_openai_agents_adapter.py src/agent_tool_gateway/adapters/openai_agents.py
git commit -m "OpenAI adapter: approval interrupt, approve/reject resume, bounded cache tests

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Output guards and the direct-invoke fallback

**Files:**
- Test: `tests/test_openai_agents_adapter.py`
- Modify (only if a test fails): `src/agent_tool_gateway/adapters/openai_agents.py`

- [ ] **Step 1: Write the tests**

Append to `tests/test_openai_agents_adapter.py`:

```python
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
```

- [ ] **Step 2: Run them**

Run: `pytest tests/test_openai_agents_adapter.py -q`
Expected: all pass. If `ToolContext(...)` rejects the keyword set, check `inspect.signature(ToolContext.__init__)`; on 0.22.0 the parameters are `context, usage, tool_name, tool_call_id, tool_arguments, ...` with the rest optional.

- [ ] **Step 3: Run the whole suite plus lint and types**

Run: `pytest -q && ruff check src tests examples && mypy src`
Expected: all tests pass (77 existing + 16 new = 93), ruff clean, mypy clean.

- [ ] **Step 4: Commit**

```bash
git add tests/test_openai_agents_adapter.py src/agent_tool_gateway/adapters/openai_agents.py
git commit -m "OpenAI adapter: output guard rewrite and direct-invoke fallback tests

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/ARCHITECTURE.md`

- [ ] **Step 1: README adapter section**

In `README.md`, directly after the "Claude Agent SDK adapter" block (the paragraph ending with the link to `examples/claude_sdk_coding_agent.py`), insert:

````markdown
### OpenAI Agents SDK adapter

```python
from agents import Agent, Runner
from agent_tool_gateway.adapters.openai_agents import OpenAIAgentsAdapter, manifest_from_function_tool

registry = ToolRegistry([manifest_from_function_tool(read_file, required_scopes=["fs:read"]),
                         manifest_from_function_tool(send_email, side_effect="irreversible", risk_tier="high")])
adapter = OpenAIAgentsAdapter(Gateway(registry, default_stages(policy)))

agent = Agent(name="assistant", tools=adapter.gate_tools([read_file, send_email]))
result = await Runner.run(agent, "...", context=Identity(principal, agent_id, session))  # any object with .principal/.agent/.session
if result.interruptions:                       # REQUIRE_APPROVAL -> native ToolApprovalItem
    state = result.to_state(); state.approve(result.interruptions[0])
    result = await Runner.run(agent, state)
```

`DENY` returns the structured error to the model and the run continues; `TRANSFORM` rewrites the arguments the tool receives; `REQUIRE_APPROVAL` uses the SDK's `needs_approval` interruption so `RunState.approve` / `reject` work unchanged; output guards rewrite what the model sees. Hosted tools pass through untouched. Approve-and-resume works in-process today (`SessionState` is not yet serialisable).
````

- [ ] **Step 2: README integration tier table and roadmap**

Replace the tier-1 row:

```markdown
| 1 | Native decision hooks | Claude Agent SDK; OpenAI Agents SDK (`needs_approval` + wrapped invoke) | Claude ✅ · OpenAI ✅ |
```

Replace the roadmap line `- [ ] OpenAI Agents SDK adapter (tool guardrails + `RunState` approvals)` with `- [x] OpenAI Agents SDK adapter (`needs_approval` + `RunState` approvals)`.

- [ ] **Step 3: ARCHITECTURE adapter rules**

In `docs/ARCHITECTURE.md`, under "Adapter rules", replace item 3 with:

```markdown
3. `REQUIRE_APPROVAL` maps to the framework's native approval mechanism where one exists
   (Claude Agent SDK `ask` → `can_use_tool`; OpenAI Agents SDK `needs_approval` → `RunState`
   interruption); otherwise to a structured `approval_required` error the host application handles.
4. When the host grants an approval the gateway did not see start, the adapter calls
   `Gateway.reserve(ctx)` before execution so loop detection and budget accounting stay correct.
```

- [ ] **Step 4: Verify docs render and nothing else broke**

Run: `pytest -q && ruff check src tests examples && python examples/claude_sdk_coding_agent.py > /dev/null && echo ok`
Expected: tests pass, ruff clean, `ok`.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/ARCHITECTURE.md
git commit -m "Document the OpenAI Agents SDK adapter

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review

- **Spec coverage:** decision table → Task 3; user `needs_approval` OR → Task 3 code, Task 4 test; rejection untouched + eviction → Task 4; no-cache fallback → Task 3 code, Task 5 test; `ToolNotRegistered` cached as error → Task 3; output guards → Task 5; `default_identity`, `manifest_from_function_tool` → Task 2; packaging → Task 1; README/ARCHITECTURE → Task 6; 13 spec cases map to the 16 tests (spec case 1 split into allow + unregistered; case 9 covers both bool and callable).
- **Placeholders:** none.
- **Type consistency:** `OpenAIAgentsAdapter(gateway, identity=default_identity, *, max_inflight)`, `gate_tool`, `gate_tools`, `needs_approval(tool_name, run_ctx, params, call_id)`, `invoke(tool_name, original_invoke, tool_ctx, args_json)`, module-level `gate_tools(gateway, tools, identity)`, `manifest_from_function_tool(tool, **overrides)` are used with the same names and arities in every task.
