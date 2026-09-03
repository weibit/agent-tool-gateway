# agent-tool-gateway

**In-process authorization, risk scoring, rate limiting and guardrails at the tool-call boundary of AI agents.**

Every tool call an agent makes passes through one pipeline that decides `ALLOW`, `TRANSFORM`, `REQUIRE_APPROVAL`, or `DENY` — using the full runtime context a network proxy never sees: who delegated the session, which agent in the spawn chain is acting, what has been spent, and whether the model has just consumed untrusted content.

```
resolve → schema → scope → policy → taint → risk → guardrails → loop_detect → budget → rate_limit → execute → output guardrails → audit
```

## Why not an MCP gateway?

An MCP gateway governs traffic between MCP clients and MCP servers. It is the right place for transport auth, routing, and fleet-wide observability. It cannot see plain function tools, sub-agent handoffs, session budgets, or the fact that the last tool result came from a web page.

`agent-tool-gateway` sits *inside* the agent runtime, at the invocation boundary, and governs every locally-executed tool regardless of how it is implemented. The two are complementary: the tool gateway decides whether and how a call may happen; an MCP gateway (or the provider) governs transport and hosted tools.

| | MCP gateway | agent-tool-gateway |
|---|---|---|
| Position | network proxy | in-process library |
| Covers | MCP tools only | any local tool: functions, MCP, OpenAPI, sub-agents |
| Context | headers, tool name, args | + principal, delegation chain, session budget, taint, turn |
| Granularity | tool-level | argument-level |
| Outcomes | allow / deny | allow / transform / require_approval / deny |
| Hosted / provider-executed tools | yes | no (by construction) |

## Core ideas

- **Authorize, then score risk.** Policy answers "may this principal/agent do this?" Risk answers "how dangerous is doing it *now*?" and maps a score onto approve/deny thresholds.
- **Manifests carry security metadata.** Side-effect class (`read` / `write` / `irreversible`), risk tier, required scopes, whether the tool reaches untrusted content, output classification, nominal cost. Policy keys off these, not the name.
- **Authority only attenuates.** `agent.spawn(...)` yields a child whose scopes are a strict subset of the parent's. Effective authority is `principal.scopes ∩ agent.effective_scopes`.
- **Tool output is untrusted input.** Results from `reaches_untrusted` tools taint the session; subsequent side-effecting calls require approval. This is prompt-injection defense enforced in the loop, not in the prompt.
- **Two error channels.** Every error has a model-safe message (structured, recoverable, non-leaking) and a full-fidelity audit detail. Adapters only ever surface the former.
- **Fail closed, with dry-run.** A stage bug denies. `dry_run=True` logs shadow decisions (including rewrites) without enforcing anything, for safe policy rollout.
- **Approvals are exact.** `REQUIRE_APPROVAL` carries an `approval_id`; redeeming it with `grant_approval_by_id` allows that tool with exactly those arguments and nothing else.
- **Adapters translate; they never contain policy.** The same manifests and policy produce identical behavior in every framework.

## Quickstart

```bash
pip install agent-tool-gateway            # core has zero dependencies
pip install "agent-tool-gateway[jsonschema]"   # full JSON Schema validation
```

```python
from agent_tool_gateway import *
from agent_tool_gateway.stages import RulePolicy, default_stages

registry = ToolRegistry([
    ToolManifest("read_file", side_effect=SideEffect.READ, required_scopes={"fs:read"},
                 input_schema={"type": "object", "required": ["path"]}),
    ToolManifest("send_email", side_effect=SideEffect.IRREVERSIBLE, risk_tier=RiskTier.HIGH,
                 required_scopes={"email:send"}, cost_usd=0.01),
    ToolManifest("fetch_url", side_effect=SideEffect.READ, reaches_untrusted=True, required_scopes={"net:read"}),
])

policy = (RulePolicy()
    .allow("read_file")
    .deny("read_file", when=lambda c: c.args["path"].startswith("/etc"), priority=10, reason="system files")
    .allow("fetch_url")
    .require_approval("send_email", reason="outbound email needs a human"))

gw = Gateway(registry, default_stages(policy), audit=JsonlAuditSink())

principal = Principal("alice", scopes=frozenset({"fs:read", "net:read", "email:send"}))
agent     = AgentIdentity("orchestrator", scopes=frozenset({"fs:read", "net:read", "email:send"}))
session   = SessionState(budget_limit_usd=1.00)

ctx = gw.build_context("read_file", {"path": "/etc/passwd"}, principal=principal, agent=agent, session=session)
decision = await gw.before(ctx)     # DENY, stage="policy", reason="system files"
```

### Generic adapter — wrap any callable

```python
from agent_tool_gateway.adapters import gw_wrap, bind

@gw_wrap(gw, "read_file")
def read_file(path: str) -> str: ...

with bind(principal=principal, agent=agent, session=session):
    read_file(path="README.md")          # runs the full pipeline
    read_file(path="/etc/passwd")        # -> {"error": "authorization_denied", "message": ..., "retryable": False}
    read_file()                          # -> {"error": "invalid_arguments", ..., "retryable": True}
```

Sync tools run in a worker thread so `timeout_s` applies to them; async tools are awaited directly.

### Claude Agent SDK adapter

```python
from claude_agent_sdk import ClaudeAgentOptions, query
from agent_tool_gateway.adapters.claude_agent_sdk import build_hooks

hooks = build_hooks(gw, identity=lambda hook_input: (principal, agent, session))
options = ClaudeAgentOptions(hooks=hooks)
```

`DENY` → `permissionDecision: deny`, `REQUIRE_APPROVAL` → `ask`, `TRANSFORM` → `allow` + `updatedInput`. `PostToolUse` runs output guardrails and taint tagging and returns redacted/truncated content as `updatedToolOutput`; `PostToolUseFailure` releases the budget reservation. The SDK routes `ask` to your `can_use_tool` callback, so configure one or `ask` has nothing to answer it. See [`examples/claude_sdk_coding_agent.py`](examples/claude_sdk_coding_agent.py).

## Loading tools on demand

Production agents load tools you did not write, often hundreds of them from MCP servers. The registry resolves a name through layers, first hit wins: explicit manifests, then resolvers in order, then a global default.

```python
from agent_tool_gateway import ToolRegistry, glob_overlay, lookup
from agent_tool_gateway.discovery import manifests_from_mcp, mcp_default

discovered = manifests_from_mcp("github", tools_list_result)        # untrusted by default: clamped to WRITE/HIGH
operator_overlay = glob_overlay({                                     # a human relaxes what they have reviewed
    "mcp__github__get_*":  {"side_effect": "read", "risk_tier": "low"},
    "mcp__github__list_*": {"side_effect": "read", "risk_tier": "low"},
})
registry = ToolRegistry(
    resolvers=[operator_overlay, lookup(discovered), glob_overlay({"mcp__github__*": mcp_default("github")})],
    default=ToolManifest("*", side_effect=SideEffect.WRITE, risk_tier=RiskTier.HIGH),
)
```

Every manifest from a server carries `required_scopes={"mcp:<server>"}`, so a principal must be granted the server before any of its tools pass the scope stage. MCP annotations (`readOnlyHint`, `destructiveHint`, `openWorldHint`) map onto `side_effect` and `reaches_untrusted` for `trusted=True` servers only. On `tools/list_changed`, rebuild and swap: `gateway.registry = new_registry`. Contexts already in flight keep the manifest they were built with.

## Integration tiers

| Tier | Mechanism | Frameworks | Status |
|---|---|---|---|
| 1 | Native decision hooks | Claude Agent SDK; OpenAI Agents SDK (tool guardrails + approvals) | Claude ✅ · OpenAI planned |
| 2 | Toolset wrappers | Pydantic AI, LangGraph, Strands, Agno, Google ADK | planned |
| 3 | Function wrapping | anything that calls a Python callable | ✅ |

Hosted / provider-executed tools (server-side web search, hosted MCP connectors) never enter the process and are out of scope; govern those at the provider or MCP-gateway layer.

## Built-in stages

| Stage | Purpose |
|---|---|
| `SchemaStage` | validate args against the manifest's JSON Schema |
| `ScopeStage` | `required_scopes ⊆ principal.scopes ∩ agent.effective_scopes` |
| `PolicyStage` | first-match glob + predicate rules; allow / deny / require_approval / rewrite |
| `TaintStage` | taint on untrusted output; gate side effects while tainted |
| `RiskStage` | additive signals (tier, side effect, taint, depth, classification) → thresholds |
| `GuardrailStage` | input guards (e.g. credentials in args), output guards (PII redaction, size cap) |
| `LoopDetectStage` | deny the N-th identical call in a window |
| `BudgetStage` | per-session cost ceiling; cost is reserved before execution and settled after, so parallel calls cannot overshoot |
| `RateLimitStage` | token bucket keyed on (principal, agent, tool); swap in Redis for multi-process |

Write your own by subclassing `BaseStage`. Policy backends (Cedar, OPA, CEL) implement `evaluate(ctx) -> DecisionResult`.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```

## Roadmap

- [ ] OpenAI Agents SDK adapter (tool guardrails + `RunState` approvals)
- [ ] Pydantic AI / LangGraph toolset adapters
- [ ] Redis-backed limiter and shared session store
- [ ] Cedar policy backend
- [ ] Capability-token (macaroon-style) delegation across processes
- [ ] OpenTelemetry span export from the audit sink
- [ ] Sidecar mode: HTTP policy decision point over the same core
- [ ] Audit-log replay against a new policy version

## License

Apache-2.0
