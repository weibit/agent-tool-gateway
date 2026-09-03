# OpenAI Agents SDK adapter — design

Date: 2026-09-03. Target SDK: `openai-agents` 0.22.x (pin the minor at implementation time).

## Goal

A tier-1 adapter that runs every `FunctionTool` call of an OpenAI Agents SDK agent through the
gateway pipeline, with the four decisions mapped onto the SDK's native mechanisms, the same
identity callback shape as the Claude adapter, and no policy inside the adapter.

## Non-goals

- Hosted tools (web search, file search, computer use, hosted MCP). They never enter the process.
- Cross-process approve-and-resume. `RunState` serialises; `SessionState` does not yet.
- Gateway `timeout_s`. The SDK owns tool timeouts (`FunctionTool.timeout_seconds`).
- Handoffs and agent-as-tool. Not gated in this iteration.

## Approach

Wrap each `FunctionTool` (dataclass `replace`) with two fields swapped:

- `needs_approval` → adapter callback `(run_ctx, params: dict, call_id: str) -> bool`.
  The SDK evaluates it at planning time, before execution. This is where `gateway.before` runs,
  exactly once per call. The resulting `ToolCallContext` is cached by `call_id`.
- `on_invoke_tool` → adapter wrapper `(tool_ctx, args_json: str) -> Any` around the original.

Everything else on the tool (schema, strictness, the user's guardrails, timeout, failure handler)
is untouched and keeps running underneath.

Rejected alternatives: guardrails-only (input guardrails cannot rewrite arguments, so `TRANSFORM`
would degrade to deny); `RunHooks` (observe only).

## Decision mapping

| Gateway decision | `needs_approval` returns | `on_invoke_tool` wrapper |
|---|---|---|
| ALLOW | `False` | run original with original JSON |
| TRANSFORM | `False` | run original with `json.dumps(ctx.args)` |
| REQUIRE_APPROVAL | `True` → run interrupts with a `ToolApprovalItem` | if invoked later the host approved via `RunState.approve`; call `gateway.reserve(ctx)`, then run |
| DENY | `False` | return `json.dumps(err.to_model_result())`; the model sees a structured, recoverable error and the run continues |

After the original returns, the wrapper calls `gateway.after(ctx, ToolResult(content=raw))` and
returns `result.content`. A `GatewayError` raised by an output guard is returned as
`json.dumps(err.to_model_result())`.

If the user's tool already had `needs_approval`, the wrapper evaluates it too and returns
`user_needs or gateway_needs`.

Rejections by the host use `RunState.reject(item, rejection_message=...)` unchanged; the adapter
does not intercept them. If the host rejects, the wrapper is never invoked and the cached context
is evicted by the bounded cache.

Fallback when the wrapper finds no cached context for `call_id` (tool invoked without the planning
step, e.g. by a test calling `on_invoke_tool` directly): run `before` inline; ALLOW/TRANSFORM
proceed; DENY returns the error; REQUIRE_APPROVAL returns an `approval_required` error because
interruption is no longer possible.

## Components

All in `src/agent_tool_gateway/adapters/openai_agents.py`. `agents` is imported lazily inside
functions so the core stays dependency-free.

```python
IdentityProvider = Callable[[Any], tuple[Principal, AgentIdentity, SessionState]]  # arg: RunContextWrapper

def default_identity(run_ctx) -> tuple[...]:
    """Read .principal / .agent / .session from run_ctx.context; raise RuntimeError if absent."""

class OpenAIAgentsAdapter:
    def __init__(self, gateway: Gateway, identity: IdentityProvider = default_identity, *, max_inflight: int = 256)
    def gate_tool(self, tool: FunctionTool) -> FunctionTool
    def gate_tools(self, tools: Sequence[Tool]) -> list[Tool]   # FunctionTools wrapped, others passed through
    async def needs_approval(self, run_ctx, params: dict, call_id: str) -> bool
    async def invoke(self, tool: FunctionTool, original_invoke, tool_ctx, args_json: str) -> Any

def gate_tools(gateway, tools, identity=default_identity) -> list[Tool]   # convenience
def manifest_from_function_tool(tool: FunctionTool, **overrides) -> ToolManifest
```

`manifest_from_function_tool` maps `name`, `description`, `params_json_schema` → `input_schema`;
overrides set `side_effect`, `risk_tier`, `required_scopes`, etc. It lets the registry and the
agent's tool list be built from the same objects.

Identity: the SDK passes the user's `context` object through `RunContextWrapper.context`. The
default provider expects that object to expose `principal`, `agent`, `session`. Hosts with other
shapes pass their own callback, as with the Claude adapter.

In-flight cache: `OrderedDict[call_id, ToolCallContext]`, evict oldest above `max_inflight`.
`ctx.metadata["hook_decision"]` records the decision as in the Claude adapter.

`session.turn` is incremented once per `needs_approval` evaluation, matching the Claude adapter's
per-call semantics (documented limitation).

## Error handling

- `ToolNotRegistered` from `build_context` in `needs_approval`: cache a DENY decision; the wrapper
  returns the model-facing error. The run continues.
- Exceptions inside the original tool are already converted to strings by the SDK's failure
  handler; the wrapper sees a string and runs `after` on it. The gateway does not audit a
  tool_execution_error in that case (the SDK owns it); it audits the execution as normal.
- Exceptions in the adapter itself propagate (SDK fails the run), matching fail-closed.
- Invalid JSON in `args_json` (only possible in the no-cache fallback): let the original tool's own
  handling deal with it; the gateway runs `before` with `{}`.

## Packaging

- `pyproject.toml`: new extra `openai = ["openai-agents>=0.22,<0.23"]`; add to `dev`.
- `adapters/__init__.py`: no eager import of the new module (keeps `agents` optional).
- README: Claude adapter paragraph gains an OpenAI sibling; integration tier table flips
  "OpenAI planned" to ✅; roadmap item checked.

## Testing

`tests/test_openai_agents_adapter.py`, `pytest.importorskip("agents")`.

A `ScriptedModel(Model)` returns a queued list of `ModelResponse`s: first a function call
(`ResponseFunctionToolCall`), then a final text message. `Agent(model=ScriptedModel(...),
tools=adapter.gate_tools([...]))`, run with `Runner.run(agent, "go", context=ctx_obj)`.

Cases:

1. allow: tool ran, output item present, audit has decision + execution, `recent_calls` recorded
2. deny: tool did not run; tool output item content is the JSON error with `"error": "authorization_denied"`; run finished
3. schema deny: `"error": "invalid_arguments"`, `retryable: true`
4. transform: tool received rewritten args
5. require approval: `result.interruptions` has one item; tool not run; `session.budget_reserved_usd == 0`
6. approve then resume: `state.approve(item)`; `Runner.run(agent, state)` runs the tool; budget settled; `recent_calls` recorded once
7. reject then resume: tool not run; cache does not grow unboundedly (assert eviction with `max_inflight=1`)
8. output redaction: tool returns a string with an SSN; the output item shows `[REDACTED]`
9. user `needs_approval=True` on the tool is honoured even when gateway allows
10. hosted tool instance in `gate_tools` input is returned unchanged
11. `default_identity` raises `RuntimeError` when context lacks the attributes
12. `manifest_from_function_tool` copies schema and applies overrides
13. direct `on_invoke_tool` call without planning: allow proceeds; require-approval returns `approval_required` error
