# LangGraph adapter — design

Date: 2026-09-03. Targets: `langgraph` 1.2.x, `langchain-core` 1.6.x; `langchain` 1.4.x optional
(for `create_agent` middleware). Pin minors at implementation time.

## Goal

A tier-2 adapter that runs every tool call executed by a LangGraph `ToolNode` (and therefore by
`create_react_agent` and LangChain's `create_agent`) through the gateway, mapping the four
decisions onto `ToolMessage` results, `ToolCallRequest.override`, and `interrupt()` /
`Command(resume=...)`, with the same identity callback shape as the other adapters and no policy
inside the adapter.

## Non-goals

- Tools invoked outside a `ToolNode` / `create_agent` (e.g. `tool.invoke(...)` by hand).
- Cross-process session state. The checkpointer persists graph state; `SessionState` is still an
  in-memory object the host must keep alive between interrupt and resume.
- Gateway `timeout_s`.
- Deduplicating the replay side effects described below.

## Approach

`ToolNode(tools, wrap_tool_call=..., awrap_tool_call=...)` calls a wrapper
`(request: ToolCallRequest, execute) -> ToolMessage | Command` around every tool call.
`request.tool_call` is the `{"name", "args", "id"}` dict, `request.runtime` carries
`tool_call_id`, `config`, `context`, `state`. LangChain's `AgentMiddleware.wrap_tool_call` /
`awrap_tool_call` receive the **same** `ToolCallRequest` class (verified: identical object on
langgraph 1.2.11 / langchain 1.4.0), so one implementation serves both entry points.

Rejected alternatives: wrapping each `BaseTool` (awkward `tool_call_id` plumbing, still needs
graph context for `interrupt`); `interrupt_before=["tools"]` (no per-argument decisions).

## Decision mapping

| Gateway decision | wrapper |
|---|---|
| ALLOW | `execute(request)` |
| TRANSFORM | `execute(request.override(tool_call={**tc, "args": gw_ctx.args}))` |
| REQUIRE_APPROVAL | `answer = interrupt(payload)`; approved → `gateway.before_approved(gw_ctx)` then execute (or deny if that blocks); not approved → error `ToolMessage` |
| DENY | `ToolMessage(content=json.dumps(err.to_model_result()), tool_call_id, name, status="error")` |

Interrupt payload: `{"approval_id", "reason", "tool", "args", "tool_call_id"}`.
Resume value: `True` or a mapping with `approved: True` approves; any other value denies. The denial
message is `value["message"]` when the value is a mapping with that key, else
`"The tool call was denied."`.

After execution: `result = await gateway.after(gw_ctx, ToolResult(content=msg.content))`;
`msg.content = result.content`; return `msg`. A `GatewayError` from an output guard returns an
error `ToolMessage` with its `to_model_result()`. If `execute` returns a `Command` (a tool that
updates state), it is returned untouched without running `after`.

An exception escaping `execute` (including `GraphInterrupt` raised by a tool that itself calls
`interrupt`) releases the reservation and propagates, preserving `ToolNode.handle_tool_errors`.

### Replay semantics

On `Command(resume=...)` LangGraph re-executes the node. The wrapper runs from the top:
`before` → REQUIRE_APPROVAL → `interrupt()` returns the resume value instead of raising →
`before_approved` re-checks budget/rate limit → execute. Consequences, documented in the module:
`session.turn` increments twice for an approved call and the "require_approval" decision is
audited twice. Interrupts need a checkpointer and `configurable.thread_id`.

### Sync and async

`ToolNode` calls `wrap_tool_call` from `invoke` and `awrap_tool_call` from `ainvoke` (falling
back to the sync one if the async is absent). The adapter provides both. The sync wrapper drives
the async gateway (`before`, `before_approved`, `after`) through `adapters.wrap._run_sync`, which
runs the coroutine on a private loop in a worker thread with the caller's contextvars. `interrupt`
is called on the wrapper's own thread, never inside that coroutine, because it relies on LangGraph's
runtime context.

## Components

`src/agent_tool_gateway/adapters/langgraph.py`. `langgraph` / `langchain_core` imported lazily
inside functions; `langchain` imported lazily only in `middleware()`.

```python
IdentityProvider = Callable[[Any], tuple[Principal, AgentIdentity, SessionState]]  # arg: ToolRuntime

def default_identity(runtime) -> tuple[...]:
    """runtime.context.{principal,agent,session} if present, else
    runtime.config["configurable"]["gateway_identity"].{...}; RuntimeError otherwise."""

class LangGraphAdapter:
    def __init__(self, gateway: Gateway, identity: IdentityProvider = default_identity)
    def wrap_tool_call(self, request, execute) -> ToolMessage | Command      # sync
    async def awrap_tool_call(self, request, execute) -> ToolMessage | Command  # async
    def tool_node(self, tools, **kwargs) -> ToolNode   # ToolNode(tools, wrap_tool_call=..., awrap_tool_call=..., **kwargs)
    def middleware(self) -> AgentMiddleware           # langchain.agents.middleware; lazy import

def manifest_from_tool(tool: BaseTool, **overrides) -> ToolManifest
```

Internal structure: one async core `_decide(request, runtime) -> tuple[ctx, decision]` and one
`_finish(ctx, msg) -> msg`, used by both wrappers; the only difference between sync and async is
whether `execute` is awaited and whether `_run_sync` wraps the gateway coroutines.

`manifest_from_tool`: `name`, `description or ""`, `tool.tool_call_schema.model_json_schema()`
→ `input_schema` (falls back to `{"type": "object", "properties": tool.args}` if
`tool_call_schema` is missing), then `overrides` via `ToolManifest.from_dict`.

`ctx.metadata["hook_decision"]` records the decision as in the other adapters.

## Error handling

- `ToolNotRegistered` from `build_context`: error `ToolMessage` with `to_model_result()`.
- Any `GatewayError` from `raise_for_decision` (deny, or a stage still asking after approval):
  error `ToolMessage`.
- Exceptions in the adapter itself propagate (fail closed).

## Packaging

- `pyproject.toml`: extra `langgraph = ["langgraph>=1.2,<2", "langchain-core>=1.6,<2"]`; dev adds
  those plus `"langchain>=1.4,<2"`.
- README: adapter section; tier-2 row gains "LangGraph ✅"; roadmap item checked.
- ARCHITECTURE adapter rule 3 mentions `interrupt` → `Command(resume=...)`.

## Testing

`tests/test_langgraph_adapter.py`, `pytest.importorskip("langgraph")`.

Harness: a `StateGraph` with a scripted `agent` node that emits one `AIMessage` with a tool
call per queued `(name, args, id)` and then `AIMessage("done")`, a `tools` node from
`adapter.tool_node([...])`, `tools_condition`-style routing, compiled with `InMemorySaver`.
Identity passed as `config["configurable"]["gateway_identity"]`.

Cases:

1. allow: tool ran; `ToolMessage.content == "echo:hi"`, `status == "success"`; audit decision + execution; `recent_calls` recorded
2. deny: `ToolMessage` with `status == "error"` whose JSON has `error == "authorization_denied"`; graph finishes with "done"
3. schema deny: `error == "invalid_arguments"`, `retryable is True`
4. unregistered tool: `error == "tool_not_registered"`
5. transform: tool received rewritten args
6. approval: `invoke` returns `__interrupt__` with payload containing `approval_id`, `tool`, `args`; no tool message; no reservation; no `recent_calls`
7. resume `True`: tool ran; budget settled; `recent_calls` once
8. resume `True` after budget exhausted: error `ToolMessage` with `budget_exceeded`
9. resume `False`: error `ToolMessage` "The tool call was denied."; resume `{"approved": False, "message": "nope"}` → "nope"
10. output redaction: SSN replaced in `ToolMessage.content`
11. async path via `graph.ainvoke` (allow and approval round trip)
12. identity from `runtime.context` (graph compiled with `context_schema`, run with `context=`)
13. `default_identity` raises `RuntimeError` with neither source present
14. `manifest_from_tool` copies schema and applies overrides
15. `middleware()`: if a fake chat model supporting `bind_tools` exists in `langchain_core`, run `create_agent(model, tools, middleware=[adapter.middleware()], checkpointer=...)` for allow and deny; otherwise assert the middleware's `wrap_tool_call` / `awrap_tool_call` delegate to the adapter's methods with the same request object
