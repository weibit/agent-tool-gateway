# Pydantic AI Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run every toolset tool call of a Pydantic AI agent through the gateway via a `WrapperToolset` subclass, mapping ALLOW / TRANSFORM / REQUIRE_APPROVAL / DENY onto the tool return value and `ApprovalRequired` / `DeferredToolRequests`.

**Architecture:** `GatedToolset(WrapperToolset)` overrides `call_tool`, the single interception point that receives validated args, the `RunContext` (with `tool_call_id`, `tool_call_approved`, `deps`) and the `ToolsetTool`. Before, execute and after all happen inside it, so there is no per-call cache. DENY returns the `to_model_result()` dict (never `ModelRetry`, which exhausts `max_retries`). Approval raises `ApprovalRequired`; the approved re-entry calls `gateway.reserve` before running.

**Tech Stack:** Python 3.11+, `pydantic-ai-slim` 2.38.x (spiked against 2.38.0), pytest, `FunctionModel` with scripted `ToolCallPart`s so `Agent.run` executes without network.

Spec: `docs/superpowers/specs/2026-09-03-pydantic-ai-adapter-design.md`.

---

## File structure

- Create `src/agent_tool_gateway/adapters/pydantic_ai.py` — `default_identity`, `GatedToolset`, `gate_toolset`, `manifest_from_tool_def`. Imports the SDK at module level (dataclass base class), guarded with a clear `ImportError`.
- Create `tests/test_pydantic_ai_adapter.py` — scripted `FunctionModel` harness plus all spec cases.
- Modify `pyproject.toml` — `pydantic` extra, added to `dev`.
- Modify `README.md`, `docs/ARCHITECTURE.md`.
- Do not touch `adapters/__init__.py`.

Environment: use the project venv (`uv venv --python 3.12`, `uv pip install -e ".[dev]"`); `python`, `pytest`, `ruff`, `mypy` below mean that venv's.

---

### Task 1: Packaging and test harness

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/test_pydantic_ai_adapter.py`

- [ ] **Step 1: Add the extra**

In `pyproject.toml` replace the optional-dependencies block with:

```toml
[project.optional-dependencies]
jsonschema = ["jsonschema>=4.0"]
claude = ["claude-agent-sdk"]
openai = ["openai-agents>=0.22,<0.23"]
pydantic = ["pydantic-ai-slim>=2.38,<2.39"]
dev = [
  "pytest>=8", "pytest-asyncio>=0.23", "ruff>=0.5", "mypy>=1.10", "jsonschema>=4.0",
  "openai-agents>=0.22,<0.23", "pydantic-ai-slim>=2.38,<2.39",
]
```

- [ ] **Step 2: Install**

Run: `uv pip install -e ".[dev]"`
Expected: `python -c "import importlib.metadata as m; print(m.version('pydantic-ai-slim'))"` prints `2.38.x`.

- [ ] **Step 3: Write the harness with one smoke test**

Create `tests/test_pydantic_ai_adapter.py`:

```python
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
```

- [ ] **Step 4: Run it**

Run: `pytest tests/test_pydantic_ai_adapter.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'agent_tool_gateway.adapters.pydantic_ai'`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/test_pydantic_ai_adapter.py
git commit -m "Add pydantic extra and scripted FunctionModel harness for the Pydantic AI adapter

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: The adapter module

**Files:**
- Create: `src/agent_tool_gateway/adapters/pydantic_ai.py`
- Test: `tests/test_pydantic_ai_adapter.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pydantic_ai_adapter.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_pydantic_ai_adapter.py -q`
Expected: collection error, module missing.

- [ ] **Step 3: Create the module**

Create `src/agent_tool_gateway/adapters/pydantic_ai.py`:

```python
"""Tier-2 adapter: Pydantic AI toolset wrapper.

``GatedToolset`` wraps any toolset (``FunctionToolset``, MCP, ...) so every call goes
through the gateway:

    Decision.ALLOW             -> wrapped tool runs with the model's arguments
    Decision.TRANSFORM         -> wrapped tool runs with the rewritten arguments
    Decision.REQUIRE_APPROVAL  -> raises ``ApprovalRequired``; the run ends with
                                  ``DeferredToolRequests`` and resumes via ``DeferredToolResults``
    Decision.DENY              -> returns ``GatewayError.to_model_result()`` as the tool result

Deny is a returned dict on purpose: ``ModelRetry`` counts against the tool's ``max_retries``
(default 1) and a second denied call would end the run with ``UnexpectedModelBehavior``.

``before`` runs on every entry, including the approved re-entry (``ctx.tool_call_approved``),
so state that changed while the approval was pending is still enforced. Output guards run via
``gateway.after``.

Identity comes from ``ctx.deps``: by default the ``deps`` object must expose ``principal``,
``agent`` and ``session``. Pass ``identity=`` for other shapes.

Gate outermost. ``PrefixedToolset`` strips its prefix before delegating inward, so wrap the
prefixed toolset (``GatedToolset(ts.prefixed("x"), gw)``) to see the names the model uses.

The agent's ``output_type`` must include ``DeferredToolRequests`` for approvals to surface;
Pydantic AI raises ``UserError`` otherwise. Cross-process resume needs a serialisable
``SessionState`` (not yet available). Gateway ``timeout_s`` is not applied; use
``ToolDefinition.timeout``.

Usage:

    agent = Agent(model, toolsets=[GatedToolset(my_toolset, gateway)], deps_type=Identity,
                  output_type=[str, DeferredToolRequests])
    result = await agent.run(prompt, deps=Identity(principal, agent_id, session))
    if isinstance(result.output, DeferredToolRequests):
        approvals = {c.tool_call_id: True for c in result.output.approvals}
        result = await agent.run(message_history=result.all_messages(),
                                 deferred_tool_results=DeferredToolResults(approvals=approvals), deps=...)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

try:
    from pydantic_ai import ApprovalRequired
    from pydantic_ai.tools import AgentDepsT, RunContext
    from pydantic_ai.toolsets import WrapperToolset
    from pydantic_ai.toolsets.abstract import ToolsetTool
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError("agent_tool_gateway.adapters.pydantic_ai needs the 'pydantic' extra: pip install "
                      "'agent-tool-gateway[pydantic]'") from e

from ..context import AgentIdentity, Principal, SessionState, ToolResult
from ..decision import Decision
from ..errors import GatewayError
from ..manifest import ToolManifest
from ..pipeline import Gateway

IdentityProvider = Callable[[Any], tuple[Principal, AgentIdentity, SessionState]]
"""Receives the RunContext and returns the gateway identity."""

_DECISION_KEY = "hook_decision"  # same key the other adapters use


def default_identity(ctx: Any) -> tuple[Principal, AgentIdentity, SessionState]:
    """Read ``principal`` / ``agent`` / ``session`` attributes from ``ctx.deps``."""
    deps: Any = getattr(ctx, "deps", None)
    try:
        return deps.principal, deps.agent, deps.session
    except AttributeError:
        raise RuntimeError(
            "deps must expose .principal, .agent and .session; pass identity=... for other shapes"
        ) from None


def manifest_from_tool_def(tool_def: Any, **overrides: Any) -> ToolManifest:
    """Build a manifest from a ``ToolDefinition``'s name, description and JSON schema."""
    fields: dict[str, Any] = {
        "name": tool_def.name,
        "description": tool_def.description or "",
        "input_schema": dict(tool_def.parameters_json_schema or {}),
    }
    fields.update(overrides)
    return ToolManifest.from_dict(fields)


@dataclass
class GatedToolset(WrapperToolset[AgentDepsT]):
    """A toolset whose every call is decided by the gateway before it reaches ``wrapped``."""

    gateway: Gateway
    identity: IdentityProvider = default_identity

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        principal, agent, session = self.identity(ctx)
        session.turn += 1
        try:
            gw_ctx = self.gateway.build_context(
                name, tool_args, principal=principal, agent=agent, session=session, tool_call_id=ctx.tool_call_id
            )
        except GatewayError as e:
            return e.to_model_result()

        decision = await self.gateway.before(gw_ctx)
        gw_ctx.metadata[_DECISION_KEY] = decision.decision.value
        if decision.decision is Decision.DENY:
            try:
                Gateway.raise_for_decision(decision)
            except GatewayError as e:
                return e.to_model_result()
        elif decision.decision is Decision.REQUIRE_APPROVAL:
            if not ctx.tool_call_approved:
                raise ApprovalRequired(metadata={"approval_id": decision.approval_id, "reason": decision.reason})
            self.gateway.reserve(gw_ctx)  # the host approved via DeferredToolResults

        args = gw_ctx.args if decision.decision is Decision.TRANSFORM else tool_args
        try:
            raw = await super().call_tool(name, args, ctx, tool)
        except BaseException:
            self.gateway.release(gw_ctx)  # the SDK owns the failure; just drop the reservation
            raise
        try:
            result = await self.gateway.after(gw_ctx, ToolResult(content=raw))
        except GatewayError as e:
            return e.to_model_result()
        return result.content


def gate_toolset(
    gateway: Gateway, toolset: Any, identity: IdentityProvider = default_identity
) -> GatedToolset[Any]:
    """Convenience: ``GatedToolset(toolset, gateway, identity)``."""
    return GatedToolset(toolset, gateway, identity)
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_pydantic_ai_adapter.py -q`
Expected: 9 pass (or 8 pass and 1 skip if the SDK validates `{"text": 5}` before the toolset; the skip is acceptable and documents SDK behaviour).

- [ ] **Step 5: Lint and type-check**

Run: `ruff check src tests examples && mypy src`
Expected: clean. Likely mypy notes: if `WrapperToolset[AgentDepsT]` generic subscripting complains, annotate `class GatedToolset(WrapperToolset[AgentDepsT])` exactly as above (it is how `ApprovalRequiredToolset` is declared in the SDK). If mypy cannot find stubs, `ignore_missing_imports` in `pyproject.toml` already covers it.

- [ ] **Step 6: Commit**

```bash
git add src/agent_tool_gateway/adapters/pydantic_ai.py tests/test_pydantic_ai_adapter.py
git commit -m "Pydantic AI adapter: GatedToolset with allow/deny/transform through the gateway

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Approvals — DeferredToolRequests, approve, deny, override args

**Files:**
- Test: `tests/test_pydantic_ai_adapter.py`
- Modify (only if a test fails): `src/agent_tool_gateway/adapters/pydantic_ai.py`

- [ ] **Step 1: Write the tests**

Append to `tests/test_pydantic_ai_adapter.py`:

```python
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
```

- [ ] **Step 2: Run them**

Run: `pytest tests/test_pydantic_ai_adapter.py -q`
Expected: all pass. The spike verified `ApprovalRequired` → `DeferredToolRequests` → `DeferredToolResults(approvals=...)` re-entry with `tool_call_approved=True`, and `ToolDenied` not re-entering the toolset, on 2.38.0.

If `test_approval_with_override_args_is_evaluated_on_overridden_args` fails because `tool_args` still carries the original args, check whether the SDK applies `override_args` before `call_tool` in this version; if it does not, drop the test and note it in the README section.

- [ ] **Step 3: Commit**

```bash
git add tests/test_pydantic_ai_adapter.py src/agent_tool_gateway/adapters/pydantic_ai.py
git commit -m "Pydantic AI adapter: approval round-trip tests (approve, deny, override, stale budget)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Output guards, sync run, composition

**Files:**
- Test: `tests/test_pydantic_ai_adapter.py`
- Modify (only if a test fails): `src/agent_tool_gateway/adapters/pydantic_ai.py`

- [ ] **Step 1: Write the tests**

Append to `tests/test_pydantic_ai_adapter.py`:

```python
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
```

- [ ] **Step 2: Run everything**

Run: `pytest -q && ruff check src tests examples && mypy src`
Expected: all tests pass (92 existing + 18 new, one of which may skip), ruff and mypy clean.

- [ ] **Step 3: Commit**

```bash
git add tests/test_pydantic_ai_adapter.py src/agent_tool_gateway/adapters/pydantic_ai.py
git commit -m "Pydantic AI adapter: output guards, run_sync, prefixed composition, failure release tests

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/ARCHITECTURE.md`

- [ ] **Step 1: README adapter section**

In `README.md`, directly before the line `## Loading tools on demand`, insert:

````markdown
### Pydantic AI adapter

```python
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults
from agent_tool_gateway.adapters.pydantic_ai import GatedToolset, manifest_from_tool_def

agent = Agent("openai:gpt-5", toolsets=[GatedToolset(my_toolset, gw)], deps_type=Identity,
              output_type=[str, DeferredToolRequests])          # DeferredToolRequests is required for approvals
result = await agent.run("...", deps=Identity(principal, agent_id, session))
if isinstance(result.output, DeferredToolRequests):              # REQUIRE_APPROVAL
    ok = {c.tool_call_id: True for c in result.output.approvals}
    result = await agent.run(message_history=result.all_messages(),
                             deferred_tool_results=DeferredToolResults(approvals=ok), deps=...)
```

`DENY` is returned as the tool result (not `ModelRetry`, which would count against `max_retries`); `TRANSFORM` rewrites the arguments the tool receives; `REQUIRE_APPROVAL` raises `ApprovalRequired` and the approved re-entry is re-evaluated, so a budget that ran out while waiting still denies. Gate outermost: `PrefixedToolset` strips its prefix before delegating inward, so wrap `ts.prefixed("x")` rather than prefixing a gated toolset.
````

- [ ] **Step 2: README tier table and roadmap**

Replace the tier-2 row:

```markdown
| 2 | Toolset wrappers | Pydantic AI, LangGraph, Strands, Agno, Google ADK | Pydantic AI ✅ · others planned |
```

Replace `- [ ] Pydantic AI / LangGraph toolset adapters` with:

```markdown
- [x] Pydantic AI toolset adapter (`GatedToolset` + `DeferredToolRequests` approvals)
- [ ] LangGraph toolset adapter
```

- [ ] **Step 3: ARCHITECTURE adapter rule 3**

Replace the rule-3 text so it reads:

```markdown
3. `REQUIRE_APPROVAL` maps to the framework's native approval mechanism where one exists
   (Claude Agent SDK `ask` → `can_use_tool`; OpenAI Agents SDK `needs_approval` → `RunState`
   interruption; Pydantic AI `ApprovalRequired` → `DeferredToolRequests`); otherwise to a
   structured `approval_required` error the host application handles.
```

- [ ] **Step 4: Verify**

Run: `pytest -q && ruff check src tests examples && python examples/claude_sdk_coding_agent.py > /dev/null && echo ok`
Expected: tests pass, ruff clean, `ok`.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/ARCHITECTURE.md
git commit -m "Document the Pydantic AI adapter

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review

- **Spec coverage:** decision table, returned-dict deny, approved re-entry with `reserve`, re-evaluation on resume → Task 2 code, Task 3 tests; `ToolDenied` and `ToolApproved(override_args)` → Task 3; output guards, `run_sync`, prefixed composition, exception releases reservation → Task 4; `default_identity`, `manifest_from_tool_def` → Task 2; packaging → Task 1; docs → Task 5. Spec cases 1–15 all have a test (case 4 may skip if the SDK validates first; the skip states why).
- **Placeholders:** none.
- **Type consistency:** `GatedToolset(wrapped, gateway, identity=default_identity)`, `gate_toolset(gateway, toolset, identity)`, `manifest_from_tool_def(tool_def, **overrides)`, `default_identity(ctx)` used identically throughout. `gw.stages[2]` is `PolicyStage` in `default_stages` (schema, scope, policy, ...).
