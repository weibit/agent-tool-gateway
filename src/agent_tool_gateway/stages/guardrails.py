"""Input/output guardrails and taint propagation."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from ..context import ToolCallContext, ToolResult
from ..decision import DecisionResult
from ..manifest import SideEffect
from ..pipeline import BaseStage
from .authz import _approved

InputGuard = Callable[[ToolCallContext], str | None]  # return violation message or None
OutputGuard = Callable[[ToolCallContext, ToolResult], ToolResult]


class GuardrailStage(BaseStage):
    """Runs input guards before execution and output guards after."""

    name = "guardrails"

    def __init__(self, inputs: list[InputGuard] | None = None, outputs: list[OutputGuard] | None = None) -> None:
        self.inputs = inputs or []
        self.outputs = outputs or []

    async def before(self, ctx: ToolCallContext) -> DecisionResult | None:
        for guard in self.inputs:
            msg = guard(ctx)
            if msg:
                return DecisionResult.deny(msg, stage=self.name, error="guardrail_violation", guard=guard.__name__)
        return None

    async def after(self, ctx: ToolCallContext, result: ToolResult) -> ToolResult:
        for guard in self.outputs:
            result = guard(ctx, result)
        return result


# ---- reference guards -------------------------------------------------------

_SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*\S+"),
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key id
    re.compile(r"sk-[A-Za-z0-9]{20,}"),  # generic sk- style key
]
_PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # US SSN
    re.compile(r"\b(?:\d[ -]?){13,16}\b"),  # card-like number
]


def no_secrets_in_args(ctx: ToolCallContext) -> str | None:
    blob = json.dumps(ctx.args, default=str)
    if any(p.search(blob) for p in _SECRET_PATTERNS):
        return "Arguments appear to contain a credential; remove it and retry."
    return None


def redact_pii(ctx: ToolCallContext, result: ToolResult) -> ToolResult:
    if not isinstance(result.content, str):
        return result
    text = result.content
    for p in _PII_PATTERNS:
        text = p.sub("[REDACTED]", text)
    if text != result.content:
        result.metadata["pii_redacted"] = True
    result.content = text
    return result


@dataclass
class MaxOutputChars:
    """Bound tool output so a single result cannot flood the context window."""

    limit: int = 20_000

    def __call__(self, ctx: ToolCallContext, result: ToolResult) -> ToolResult:
        if isinstance(result.content, str) and len(result.content) > self.limit:
            result.content = result.content[: self.limit] + f"\n...[truncated {len(result.content) - self.limit} chars]"
            result.truncated = True
        return result


# ---- taint ------------------------------------------------------------------


class TaintStage(BaseStage):
    """Tool output is untrusted input.

    after():  results from tools marked ``reaches_untrusted`` taint the session.
    before(): while tainted, side-effecting calls require approval (or are denied
              for IRREVERSIBLE tools when ``deny_irreversible`` is set).
    """

    name = "taint"

    def __init__(self, *, require_approval_when_tainted: bool = True, deny_irreversible: bool = False) -> None:
        self.require_approval_when_tainted = require_approval_when_tainted
        self.deny_irreversible = deny_irreversible

    async def before(self, ctx: ToolCallContext) -> DecisionResult | None:
        if not ctx.session.tainted or ctx.tool.side_effect is SideEffect.READ:
            return None
        sources = ", ".join(ctx.session.taint_sources[-3:])
        if self.deny_irreversible and ctx.tool.side_effect is SideEffect.IRREVERSIBLE:
            return DecisionResult.deny(
                f"'{ctx.tool.name}' is irreversible and this session has consumed untrusted content ({sources}).",
                stage=self.name,
                error="tainted_context",
            )
        if self.require_approval_when_tainted and not _approved(ctx):
            return DecisionResult.require_approval(
                f"'{ctx.tool.name}' modifies state after untrusted content was read ({sources}); a human must confirm.",
                stage=self.name,
                taint_sources=list(ctx.session.taint_sources),
            )
        return None

    async def after(self, ctx: ToolCallContext, result: ToolResult) -> ToolResult:
        if ctx.tool.reaches_untrusted:
            result.tainted = True
            ctx.session.mark_tainted(ctx.tool.name)
        return result
