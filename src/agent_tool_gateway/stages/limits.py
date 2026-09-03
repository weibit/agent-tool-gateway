"""Rate limits, cost budgets and loop detection.

Agents burn budgets in loops, not bursts, so limits are keyed on
(principal, agent, tool) and paired with a per-session cost budget and a
repeated-identical-call detector.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from ..context import ToolCallContext
from ..decision import DecisionResult
from ..pipeline import BaseStage

KeyFn = Callable[[ToolCallContext], str]


def default_key(ctx: ToolCallContext) -> str:
    return f"{ctx.principal.id}|{ctx.agent.id}|{ctx.tool.name}"


class RateLimiter(Protocol):
    async def acquire(self, key: str, cost: float = 1.0) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds)."""
        ...


@dataclass
class _Bucket:
    tokens: float
    updated: float


@dataclass
class TokenBucketLimiter:
    """In-memory token bucket. Swap for a Redis-backed implementation in multi-process deployments."""

    rate_per_s: float
    burst: float
    _buckets: dict[str, _Bucket] = field(default_factory=dict)

    async def acquire(self, key: str, cost: float = 1.0) -> tuple[bool, float]:
        now = time.monotonic()
        b = self._buckets.get(key)
        if b is None:
            b = _Bucket(tokens=self.burst, updated=now)
            self._buckets[key] = b
        b.tokens = min(self.burst, b.tokens + (now - b.updated) * self.rate_per_s)
        b.updated = now
        if b.tokens >= cost:
            b.tokens -= cost
            return True, 0.0
        return False, (cost - b.tokens) / self.rate_per_s if self.rate_per_s > 0 else float("inf")


class RateLimitStage(BaseStage):
    name = "rate_limit"

    def __init__(self, limiter: RateLimiter, key: KeyFn = default_key) -> None:
        self.limiter = limiter
        self.key = key

    async def before(self, ctx: ToolCallContext) -> DecisionResult | None:
        ok, retry_after = await self.limiter.acquire(self.key(ctx))
        if ok:
            return None
        res = DecisionResult.deny(
            f"Rate limit reached for '{ctx.tool.name}'. Retry in {retry_after:.1f}s.",
            stage=self.name,
            error="rate_limited",
        )
        res.retry_after_s = retry_after
        return res


class BudgetStage(BaseStage):
    """Deny when the nominal cost of this call would exceed the session budget."""

    name = "budget"

    async def before(self, ctx: ToolCallContext) -> DecisionResult | None:
        limit = ctx.session.budget_limit_usd
        if limit is None:
            return None
        projected = ctx.session.budget_used_usd + ctx.tool.cost_usd
        if projected > limit:
            return DecisionResult.deny(
                "Session budget exhausted; no further paid tool calls are allowed.",
                stage=self.name,
                error="budget_exceeded",
                used=ctx.session.budget_used_usd,
                limit=limit,
            )
        return None


class LoopDetectStage(BaseStage):
    """Deny the (max_repeats+1)-th identical call within ``window_s`` seconds."""

    name = "loop_detect"

    def __init__(self, max_repeats: int = 3, window_s: float = 60.0) -> None:
        self.max_repeats = max_repeats
        self.window_s = window_s

    async def before(self, ctx: ToolCallContext) -> DecisionResult | None:
        now = time.monotonic()
        h = ctx.args_hash
        repeats = sum(
            1
            for tool, ah, ts in ctx.session.recent_calls
            if tool == ctx.tool.name and ah == h and now - ts <= self.window_s
        )
        if repeats >= self.max_repeats:
            return DecisionResult.deny(
                f"'{ctx.tool.name}' has been called with identical arguments {repeats} times recently. "
                "Change your approach instead of retrying.",
                stage=self.name,
                error="loop_detected",
            )
        return None
