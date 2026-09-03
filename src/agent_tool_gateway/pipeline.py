"""The gateway pipeline.

A Gateway is an ordered list of Stages. ``before`` runs stages in order until one
blocks (DENY / REQUIRE_APPROVAL). TRANSFORM rewrites ``ctx.args`` and continues,
so later stages evaluate the rewritten arguments. ``after`` runs stages in
reverse order over the tool result.

The core has no framework or I/O imports. Adapters translate framework calls
into ``before``/``after``; ``call`` is a convenience that runs the whole
lifecycle around a Python callable.
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
    GatewayError,
    RateLimited,
    ToolExecutionError,
    ToolTimeout,
)
from .registry import ToolRegistry

log = logging.getLogger("agent_tool_gateway")


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
        final = DecisionResult.allow(stage="pipeline")
        transformed = False
        for stage in self.stages:
            try:
                res = await stage.before(ctx)
            except GatewayError as e:
                res = DecisionResult.deny(e.model_message, stage=stage.name, error=e.code)
            except Exception as e:  # stage bug
                log.exception("stage %s raised", stage.name)
                if self.fail_closed:
                    res = DecisionResult.deny("Internal policy error.", stage=stage.name, _detail=str(e))
                else:
                    res = None
            if res is None or res.decision is Decision.ALLOW:
                if res is not None and res.risk_score is not None:
                    final.risk_score = res.risk_score
                continue
            if res.decision is Decision.TRANSFORM:
                assert res.updated_args is not None
                ctx.args = dict(res.updated_args)
                transformed = True
                final = DecisionResult.transform(ctx.args, res.reason, stage=res.stage)
                continue
            # DENY / REQUIRE_APPROVAL short-circuit
            final = res
            break

        if transformed and final.decision is Decision.ALLOW:
            final = DecisionResult.transform(ctx.args, "arguments rewritten", stage="pipeline")

        if self.dry_run and final.blocked:
            await self.audit.emit(AuditEvent.from_decision(ctx, final, phase="dry_run"))
            shadow = DecisionResult.allow(stage="dry_run", reason=f"shadow:{final.decision.value}")
            shadow.details["shadow_decision"] = final.decision.value
            shadow.details["shadow_reason"] = final.reason
            return shadow

        await self.audit.emit(AuditEvent.from_decision(ctx, final))
        if not final.blocked:
            ctx.session.record_call(ctx.tool.name, ctx.args_hash)
        return final

    # ------------------------------------------------------------------ after
    async def after(self, ctx: ToolCallContext, result: ToolResult, *, duration_ms: float | None = None) -> ToolResult:
        for stage in reversed(self.stages):
            try:
                result = await stage.after(ctx, result)
            except GatewayError as e:
                await self.audit.emit(
                    AuditEvent.from_context(
                        ctx, "execution", error_code=e.code, error_detail=e.audit_detail, duration_ms=duration_ms
                    )
                )
                raise
        ctx.session.budget_used_usd += ctx.tool.cost_usd
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
        try:
            if ctx.tool.timeout_s:
                raw = await asyncio.wait_for(_invoke(fn, ctx.args), timeout=ctx.tool.timeout_s)
            else:
                raw = await _invoke(fn, ctx.args)
        except TimeoutError as e:
            err = ToolTimeout(f"Tool '{ctx.tool.name}' timed out after {ctx.tool.timeout_s}s.", str(e))
            await self.audit.emit(
                AuditEvent.from_context(ctx, "execution", error_code=err.code, error_detail=err.audit_detail)
            )
            raise err from None
        except GatewayError:
            raise
        except Exception as e:
            err = ToolExecutionError.from_exception(e, ctx.tool.name)
            await self.audit.emit(
                AuditEvent.from_context(ctx, "execution", error_code=err.code, error_detail=err.audit_detail)
            )
            raise err from e
        duration_ms = (time.perf_counter() - start) * 1000
        result = raw if isinstance(raw, ToolResult) else ToolResult(content=raw)
        return await self.after(ctx, result, duration_ms=duration_ms)

    @staticmethod
    def raise_for_decision(decision: DecisionResult) -> None:
        if decision.decision is Decision.DENY:
            if decision.details.get("error") == "rate_limited":
                raise RateLimited(decision.reason, retry_after_s=decision.retry_after_s)
            raise AuthorizationDenied(decision.reason, stage=decision.stage)
        if decision.decision is Decision.REQUIRE_APPROVAL:
            raise ApprovalRequired(decision.reason, approval_id=decision.approval_id)


async def _invoke(fn: ToolFn, args: dict[str, Any]) -> Any:
    out = fn(**args)
    if inspect.isawaitable(out):
        return await out
    return out
