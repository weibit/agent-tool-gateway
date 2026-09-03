# Pydantic AI adapter — design

Date: 2026-09-03. Target: `pydantic-ai-slim` 2.38.x (pin the minor at implementation time).

## Goal

A tier-2 toolset wrapper that runs every tool call of a Pydantic AI agent through the gateway
pipeline, mapping the four decisions onto Pydantic AI's native mechanisms (tool return value,
`ApprovalRequired` / `DeferredToolRequests`), with the same identity callback shape as the other
adapters and no policy inside the adapter.

## Non-goals

- Cross-process approve-and-resume. `DeferredToolRequests` round-trips serialise; `SessionState`
  does not yet.
- Gateway `timeout_s`. `ToolDefinition.timeout` exists on the Pydantic AI side.
- Output tools and provider built-in tools. Only toolset tools pass through `call_tool`.
- Enforcing `output_type`. The agent must list `DeferredToolRequests` in `output_type` for
  approvals to surface; the adapter documents this and cannot enforce it.

## Approach

`GatedToolset(WrapperToolset)` overrides `call_tool`. That single method receives the validated
argument dict, the `RunContext` (with `tool_call_id`, `tool_call_approved`, `deps`), and the
`ToolsetTool` (with `tool_def`). Before, execution and after all happen inside it, so the adapter
holds no per-call state.

Rejected alternatives: a decorator on `@agent.tool` functions (misses MCP and third-party toolsets,
duplicates `gw_wrap`); `prepare_tools` (can hide tools, cannot decide per call).

Deny is a returned dict, not `ModelRetry`. Verified on 2.38.0: `ModelRetry` counts against the
tool's `max_retries` (default 1), so two consecutive denied calls raise `UnexpectedModelBehavior`
and end the run. A returned `to_model_result()` dict is shown to the model as the tool result and
does not count as a retry.

## Decision mapping

| Gateway decision | `call_tool` |
|---|---|
| ALLOW | `await super().call_tool(name, tool_args, ctx, tool)` |
| TRANSFORM | same, with `gw_ctx.args` |
| REQUIRE_APPROVAL and `ctx.tool_call_approved` is false | `raise ApprovalRequired(metadata={"approval_id": ..., "reason": ...})` |
| REQUIRE_APPROVAL and `ctx.tool_call_approved` is true | `gateway.reserve(gw_ctx)`, then run (host approved via `DeferredToolResults`) |
| DENY | `return err.to_model_result()` (dict) |

After execution: `result = await gateway.after(gw_ctx, ToolResult(content=raw))`, return
`result.content`. A `GatewayError` raised by an output guard returns its `to_model_result()` dict.

`before` runs on every entry, including the approved re-entry. If the state changed while the
approval was pending (budget exhausted, rate limit), the re-entry is denied normally.

`ToolDenied` on resume never reaches the toolset; nothing to clean up. `ToolApproved(override_args=...)`
arrives as `tool_args`, so the gateway evaluates the overridden arguments.

## Components

`src/agent_tool_gateway/adapters/pydantic_ai.py`. A dataclass subclass cannot defer its base
class, so this module imports `WrapperToolset` and `ApprovalRequired` at module level, wrapped in
`try/except ImportError` that re-raises with a message naming the `pydantic` extra. Nothing in the
core or in `adapters/__init__.py` imports this module, so the SDK stays optional.

```python
IdentityProvider = Callable[[Any], tuple[Principal, AgentIdentity, SessionState]]  # arg: RunContext

def default_identity(ctx) -> tuple[...]:
    """Read .principal / .agent / .session from ctx.deps; RuntimeError if absent."""

@dataclass
class GatedToolset(WrapperToolset[AgentDepsT]):
    gateway: Gateway
    identity: IdentityProvider = default_identity
    async def call_tool(self, name, tool_args, ctx, tool) -> Any: ...

def gate_toolset(gateway, toolset, identity=default_identity) -> GatedToolset
def manifest_from_tool_def(tool_def: ToolDefinition, **overrides) -> ToolManifest
```

`GatedToolset.id` returns `None` (wrapper convention). `label` inherits
`GatedToolset(<wrapped label>)`. Composition with `prefixed`, `filtered`, etc. works in either
order, but the name the gateway sees depends on it: `PrefixedToolset.call_tool` strips its prefix
before delegating inward (verified in 2.38.0). Gate outermost — `GatedToolset(toolset.prefixed("x"))`
— so manifests and policy use the name the model uses (`x_echo`). The README says so.

`session.turn` is incremented once per `call_tool` entry (per-call semantics, as in the other
adapters; `ctx.run_step` is the model turn if a host wants it).

`manifest_from_tool_def` maps `name`, `description or ""`, `parameters_json_schema` →
`input_schema`, then applies `overrides` via `ToolManifest.from_dict`.

## Error handling

- `ToolNotRegistered` from `build_context`: return its `to_model_result()` dict (run continues).
- Exceptions raised by the wrapped tool propagate unchanged; Pydantic AI already turns
  `ModelRetry` into a retry prompt and lets others fail the run. The gateway audits nothing for
  those (the SDK owns the failure) except that the reservation is released: wrap the
  `super().call_tool` call in `try/except BaseException: gateway.release(gw_ctx); raise`.
- Exceptions in the adapter itself propagate (fail closed).

## Packaging

- `pyproject.toml`: extra `pydantic = ["pydantic-ai-slim>=2.38,<2.39"]`; add to `dev`.
- README: adapter example, tier table row 2 gains "Pydantic AI ✅", roadmap item split so
  Pydantic AI is checked and LangGraph stays open.
- ARCHITECTURE adapter rule 3 mentions `ApprovalRequired` → `DeferredToolRequests`.

## Testing

`tests/test_pydantic_ai_adapter.py`, `pytest.importorskip("pydantic_ai")`.

`FunctionModel` with a scripted function returning one `ToolCallPart` per queued call, then a
`TextPart("done")`. `Agent(model, toolsets=[GatedToolset(FunctionToolset(...), gateway)],
deps_type=Identity, output_type=[str, DeferredToolRequests])`.

Cases:

1. allow: tool ran; `ToolReturnPart.content == "echo:hi"`; audit decision + execution; `recent_calls` recorded; `turn == 1`
2. deny: `ToolReturnPart.content` is the dict with `error == "authorization_denied"`; `output == "done"`
3. two consecutive denies do not raise (no retry exhaustion)
4. schema deny: `error == "invalid_arguments"`, `retryable is True`
5. unregistered tool: `error == "tool_not_registered"`
6. transform: tool received rewritten args
7. approval: `output` is `DeferredToolRequests` with one approval; tool not run; no reservation; `approval_id` in metadata
8. approve and resume: `DeferredToolResults(approvals={id: True})` runs the tool; budget settled; `recent_calls` once
9. `ToolDenied` resume: tool not run; return content is the denial message
10. `ToolApproved(override_args=...)`: gateway evaluates and tool receives the overridden args
11. output redaction: SSN replaced in `ToolReturnPart.content`
12. `run_sync` path works
13. `default_identity` raises `RuntimeError` when deps lack the attributes
14. `manifest_from_tool_def` copies schema and applies overrides
15. `GatedToolset(toolset.prefixed("x"), gw)`: the gateway resolves `x_echo` (manifest registered
    under that name) and the tool still runs; audit records `x_echo`
