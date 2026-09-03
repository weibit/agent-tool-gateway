"""The gateway pipeline.

A Gateway is an ordered list of Stages. ``before`` runs stages in order until one
blocks (DENY / REQUIRE_APPROVAL). TRANSFORM rewrites ``ctx.args`` and continues,
so later stages evaluate the rewritten arguments. ``after`` runs stages in
reverse order over the tool result.

Budget accounting is two-phase: ``before`` reserves the tool's nominal cost when
it lets a call proceed, ``after`` settles it, and a failed execution releases it.
Concurrent calls therefore cannot overshoot the session budget.

The core has no framework or I/O imports. Adapters translate framework calls
into ``before``/``after``; ``call`` is a convenience that runs the whole
lifecycle around a Python callable. Sync callables run in a worker thread so
``timeout_s`` applies to them too (a timed-out thread keeps running to
completion; the gateway just stops waiting for it).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from .audit import AuditEvent, AuditSink, NullAuditSink
from .context import AgentIdentity, Principal, SessionState, ToolCallContext, ToolResult
from .decision import Decision, DecisionResult
from .errors import (
    ApprovalRequired,
    AuthorizationDenied,
    BudgetExceeded,
    GatewayError,
    GuardrailViolation,
    RateLimited,
    SchemaValidationError,
    ToolExecutionError,
    ToolNotRegistered,
    ToolTimeout,
)
from .registry import ToolRegistry

log = logging.getLogger("agent_tool_gateway")

# DecisionResult.details["error"] -> exception class raised by ``raise_for_decision``.
_DENY_ERRORS: dict[str, type[GatewayError]] = {
    "invalid_arguments": SchemaValidationError,
    "guardrail_violation": GuardrailViolation,
    "budget_exceeded": BudgetExceeded,
    "rate_limited": RateLimited,
    "tool_not_registered": ToolNotRegistered,
}
_RESERVED_KEY = "_budget_reserved"
_MAX_PENDING_APPROVALS = 256


@runtime_checkable
class Stage(Protocol):
    name: str

    async def before(self, ctx: ToolCallContext) -> DecisionResult | None:
        """Return None to pass through, or a DecisionResult."""
        ...

    async def after(self, ctx: ToolCallContext, result: ToolResult) -> ToolResult:
        """Inspect / rewrite the tool result. Default implementations return it unchanged."""
        ...


class BaseStage:
    """Convenience base: pass-through on both sides."""

    name = "base"

    async def before(self, ctx: ToolCallContext) -> DecisionResult | None:
        return None

    async def after(self, ctx: ToolCallContext, result: ToolResult) -> ToolResult:
        return result


ToolFn = Callable[..., Any] | Callable[..., Awaitable[Any]]


class Gateway:
    def __init__(
        self,
        registry: ToolRegistry,
        stages: Sequence[Stage],
        *,
        audit: AuditSink | None = None,
        dry_run: bool = False,
        fail_closed: bool = True,
    ) -> None:
        self.registry = registry
        self.stages = list(stages)
        self.audit = audit or NullAuditSink()
        self.dry_run = dry_run
        self.fail_closed = fail_closed

    # ------------------------------------------------------------------ build
    def build_context(
        self,
        tool_name: str,
        args: dict[str, Any],
        *,
        principal: Principal,
        agent: AgentIdentity,
        session: SessionState,
        tool_call_id: str | None = None,
        trace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolCallContext:
        manifest = self.registry.resolve(tool_name)
        ctx = ToolCallContext(
            tool=manifest,
            args=dict(args),
            principal=principal,
            agent=agent,
            session=session,
            tool_call_id=tool_call_id,
            metadata=metadata or {},
        )
        if trace_id:
            ctx.trace_id = trace_id
        return ctx

    # ----------------------------------------------------------------- before
    async def before(self, ctx: ToolCallContext) -> DecisionResult:
        """Run pre-execution stages. Never raises for policy outcomes; returns a DecisionResult."""
        original_args = dict(ctx.args)
        final = DecisionResult.allow(stage="pipeline")
        risk_score: float | None = None
        for stage in self.stages:
            try:
                res = await stage.before(ctx)
            except GatewayError as e:
                res = DecisionResult.deny(e.model_message, stage=stage.name, error=e.code, _detail=e.audit_detail)
            except Exception as e:  # stage bug
                log.exception("stage %s raised", stage.name)
                if not self.fail_closed:
                    continue
                res = DecisionResult.deny(
                    "Internal policy error.",
                    stage=stage.name,
                    error="internal_error",
                    _detail=f"{type(e).__name__}: {e}",
                )
            if res is None or res.decision is Decision.ALLOW:
                if res is not None and res.risk_score is not None:
                    risk_score = res.risk_score
                continue
            if res.decision is Decision.TRANSFORM:
                assert res.updated_args is not None
                ctx.args = dict(res.updated_args)
                final = DecisionResult.transform(ctx.args, res.reason, stage=res.stage)
                continue
            # DENY / REQUIRE_APPROVAL short-circuit
            final = res
            break

        if final.risk_score is None:
            final.risk_score = risk_score

        if self.dry_run and final.decision is not Decision.ALLOW:
            # Shadow mode: log what would have happened, enforce nothing (not even rewrites).
            await self.audit.emit(AuditEvent.from_decision(ctx, final, phase="dry_run"))
            ctx.args = original_args
            shadow = DecisionResult.allow(stage="dry_run", reason=f"shadow:{final.decision.value}")
            shadow.risk_score = final.risk_score
            shadow.details["shadow_decision"] = final.decision.value
            shadow.details["shadow_reason"] = final.reason
            return shadow

        await self.audit.emit(AuditEvent.from_decision(ctx, final))
        if final.decision is Decision.REQUIRE_APPROVAL and final.approval_id:
            pending = ctx.session.pending_approvals
            pending[final.approval_id] = ctx.approval_key
            while len(pending) > _MAX_PENDING_APPROVALS:
                del pending[next(iter(pending))]
        if not final.blocked:
            self.reserve(ctx)
        return final

    # ------------------------------------------------------------ accounting
    def reserve(self, ctx: ToolCallContext) -> None:
        """Mark the call as going ahead: loop-detection history plus a budget reservation.

        ``before`` calls this for every non-blocked decision. Adapters call it themselves
        when a REQUIRE_APPROVAL was granted outside the gateway (e.g. by the host's UI).
        """
        ctx.session.record_call(ctx.tool.name, ctx.args_hash)
        if ctx.tool.cost_usd and _RESERVED_KEY not in ctx.metadata:
            ctx.session.budget_reserved_usd += ctx.tool.cost_usd
            ctx.metadata[_RESERVED_KEY] = ctx.tool.cost_usd

    def release(self, ctx: ToolCallContext) -> None:
        """Drop the reservation without charging (the tool did not run to completion)."""
        reserved = ctx.metadata.pop(_RESERVED_KEY, 0.0)
        if reserved:
            ctx.session.budget_reserved_usd = max(0.0, ctx.session.budget_reserved_usd - reserved)

    def settle(self, ctx: ToolCallContext) -> None:
        """Convert the reservation into spend (the tool ran)."""
        self.release(ctx)
        ctx.session.budget_used_usd += ctx.tool.cost_usd

    # ------------------------------------------------------------------ after
    async def after(self, ctx: ToolCallContext, result: ToolResult, *, duration_ms: float | None = None) -> ToolResult:
        try:
            for stage in reversed(self.stages):
                result = await stage.after(ctx, result)
        except GatewayError as e:
            self.settle(ctx)
            await self.audit.emit(
                AuditEvent.from_context(
                    ctx, "execution", error_code=e.code, error_detail=e.audit_detail, duration_ms=duration_ms
                )
            )
            raise
        self.settle(ctx)
        await self.audit.emit(
            AuditEvent.from_context(
                ctx, "execution", duration_ms=duration_ms, tainted=result.tainted, extra=dict(result.metadata)
            )
        )
        return result

    # ------------------------------------------------------------------- call
    async def call(self, ctx: ToolCallContext, fn: ToolFn) -> ToolResult:
        """Full lifecycle around a callable. Raises GatewayError subclasses on block/failure."""
        decision = await self.before(ctx)
        self.raise_for_decision(decision)

        start = time.perf_counter()
        err: GatewayError
        try:
            raw = await asyncio.wait_for(_invoke(fn, ctx.args), timeout=ctx.tool.timeout_s or None)
        except TimeoutError as e:
            err = ToolTimeout(f"Tool '{ctx.tool.name}' timed out after {ctx.tool.timeout_s}s.", str(e) or "timeout")
            await self._fail(ctx, err)
            raise err from None
        except GatewayError as e:
            await self._fail(ctx, e)
            raise
        except Exception as e:
            err = ToolExecutionError.from_exception(e, ctx.tool.name)
            await self._fail(ctx, err)
            raise err from e
        duration_ms = (time.perf_counter() - start) * 1000
        result = raw if isinstance(raw, ToolResult) else ToolResult(content=raw)
        return await self.after(ctx, result, duration_ms=duration_ms)

    async def _fail(self, ctx: ToolCallContext, err: GatewayError) -> None:
        self.release(ctx)
        await self.audit.emit(
            AuditEvent.from_context(ctx, "execution", error_code=err.code, error_detail=err.audit_detail)
        )

    @staticmethod
    def raise_for_decision(decision: DecisionResult) -> None:
        """Turn a blocking decision into the matching typed error (schema, guardrail, budget, ...)."""
        if decision.decision is Decision.DENY:
            cls = _DENY_ERRORS.get(decision.details.get("error", ""), AuthorizationDenied)
            extra: dict[str, Any] = {"stage": decision.stage}
            if decision.retry_after_s is not None:
                extra["retry_after_s"] = decision.retry_after_s
            raise cls(decision.reason, decision.details.get("_detail"), **extra)
        if decision.decision is Decision.REQUIRE_APPROVAL:
            raise ApprovalRequired(decision.reason, approval_id=decision.approval_id)


async def _invoke(fn: ToolFn, args: dict[str, Any]) -> Any:
    if inspect.iscoroutinefunction(fn):
        return await fn(**args)
    out = await asyncio.to_thread(fn, **args)
    if inspect.isawaitable(out):
        return await out
    return out
