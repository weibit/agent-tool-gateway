# Architecture

## Thesis

Authorization for agents has to live at the *invocation* boundary, not the *network* boundary.
The information needed to decide well — delegated principal, agent spawn chain, session budget,
taint from untrusted tool output, turn count — exists only inside the agent loop.

## Authority model

```
effective authority = delegated authority (principal)
                    ∩ agent identity (attenuated along the spawn chain)
                    ∩ resource policy (argument-level rules)
                    ∩ runtime context (taint, budget, depth, turn)
```

Evaluation is two-stage:

1. **Authorization** — deterministic. Schema, scopes, policy rules. Answers *may*.
2. **Risk** — contextual. Additive signals mapped onto thresholds. Answers *should, now*.

A policy-level `REQUIRE_APPROVAL` short-circuits risk scoring; a granted approval is recorded on
the session keyed by `(tool, args_hash)` so the exact call passes on retry and nothing else does.
The decision's `approval_id` maps to that key in `session.pending_approvals`, so a UI can redeem it
with `grant_approval_by_id` without reconstructing the call.

Budget is two-phase: `before` reserves the tool's nominal cost, `after` settles it, and a failed
execution (or the Claude SDK's `PostToolUseFailure`) releases it. `BudgetStage` counts reservations,
so concurrent calls cannot overshoot the limit.

## Pipeline

```
Gateway.before(ctx)                              Gateway.after(ctx, result)
  ├─ SchemaStage        deny                        ├─ RateLimitStage      (no-op)
  ├─ ScopeStage         deny                        ├─ BudgetStage         (no-op)
  ├─ PolicyStage        allow/deny/ask/transform    ├─ LoopDetectStage     (no-op)
  ├─ TaintStage         ask/deny while tainted      ├─ GuardrailStage      redact, truncate
  ├─ RiskStage          allow/ask/deny by score     ├─ RiskStage           (no-op)
  ├─ GuardrailStage     deny on input guard         ├─ TaintStage          mark tainted
  ├─ LoopDetectStage    deny                        ├─ PolicyStage         (no-op)
  ├─ BudgetStage        deny                        ├─ ScopeStage          (no-op)
  └─ RateLimitStage     deny (retry_after)          └─ SchemaStage         (no-op)
        │                                                  │
        └── audit "decision" ─────────────────────── audit "execution"
```

`TRANSFORM` rewrites `ctx.args` in place and continues, so downstream stages evaluate the
rewritten arguments. `DENY` and `REQUIRE_APPROVAL` short-circuit. `after` runs stages in reverse.

## Public contract (stable)

- `ToolManifest` and its enums
- `Decision` / `DecisionResult`
- `ToolCallContext`, `Principal`, `AgentIdentity`, `SessionState`, `ToolResult`
- `Stage` protocol, `Gateway.before` / `Gateway.after` / `Gateway.call`
- `GatewayError.to_model_result()` shape

Everything else may change before 1.0.

## Adapter rules

1. Adapters translate; they never contain policy.
2. Adapters surface `model_message` / `to_model_result()` only. `audit_detail` goes to the sink.
3. `REQUIRE_APPROVAL` maps to the framework's native approval mechanism where one exists
   (Claude Agent SDK `ask`, OpenAI Agents SDK `RunState` interruption); otherwise to a
   structured `approval_required` error the host application handles.

## Out of scope

Hosted / provider-executed tools (server-side web search, hosted MCP connectors) never enter the
process. Govern those at the provider or with an MCP gateway. This project deliberately does not
compete on proxying MCP traffic.
