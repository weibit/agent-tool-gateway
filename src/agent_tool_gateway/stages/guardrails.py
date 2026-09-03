"""Input/output guardrails and taint propagation.

Guards walk nested dict/list content, so structured tool results (the common
shape for framework-native tools, e.g. ``{"stdout": ..., "stderr": ...}``) are
covered as well as plain strings.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

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


# ---- helpers ------------------------------------------------------------------


def _map_strings(obj: Any, fn: Callable[[str], str]) -> Any:
    """Return a copy of ``obj`` with ``fn`` applied to every string leaf. Never mutates the input."""
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: _map_strings(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_map_strings(v, fn) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_map_strings(v, fn) for v in obj)
    return obj


def _iter_strings(obj: Any, key: str | None = None) -> Iterator[tuple[str | None, str]]:
    """Yield ``(nearest_dict_key, string)`` for every string leaf."""
    if isinstance(obj, str):
        yield key, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _iter_strings(v, str(k))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_strings(v, key)


# ---- reference guards -------------------------------------------------------

# Argument names that hold credentials by convention. Deliberately excludes bare
# "token" (pagination tokens, cancellation tokens) — inline values catch real ones.
_SECRET_KEYS = re.compile(
    r"(?i)^(?:api[_-]?key|secret|client[_-]?secret|password|passwd|access[_-]?token|auth[_-]?token|"
    r"private[_-]?key|authorization)$"
)
_SECRET_VALUES = [
    re.compile(r"(?i)\b(?:api[_-]?key|secret|token|password|passwd)\s*[:=]\s*\S+"),  # inline KEY=value
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}"),  # OpenAI / Anthropic style keys
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36}\b"),  # GitHub tokens
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{20,}=*"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]
_SECRET_MSG = "Arguments appear to contain a credential; remove it and retry."

_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")  # US SSN
_CARD = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")  # 13-19 digits, optional separators; Luhn-checked


def no_secrets_in_args(ctx: ToolCallContext) -> str | None:
    for key, value in _iter_strings(ctx.args):
        if not value:
            continue
        if key is not None and _SECRET_KEYS.match(key):
            return _SECRET_MSG
        if any(p.search(value) for p in _SECRET_VALUES):
            return _SECRET_MSG
    return None


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _card_repl(m: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", m.group(0))
    return "[REDACTED]" if 13 <= len(digits) <= 19 and _luhn_ok(digits) else m.group(0)


def redact_pii(ctx: ToolCallContext, result: ToolResult) -> ToolResult:
    changed = False

    def scrub(text: str) -> str:
        nonlocal changed
        out = _CARD.sub(_card_repl, _SSN.sub("[REDACTED]", text))
        if out != text:
            changed = True
        return out

    result.content = _map_strings(result.content, scrub)
    if changed:
        result.metadata["pii_redacted"] = True
    return result


@dataclass
class MaxOutputChars:
    """Bound each string in a tool result so a single result cannot flood the context window."""

    limit: int = 20_000

    def __call__(self, ctx: ToolCallContext, result: ToolResult) -> ToolResult:
        truncated = False

        def cap(text: str) -> str:
            nonlocal truncated
            if len(text) <= self.limit:
                return text
            truncated = True
            return text[: self.limit] + f"\n...[truncated {len(text) - self.limit} chars]"

        result.content = _map_strings(result.content, cap)
        if truncated:
            result.truncated = True
        return result


# ---- taint ------------------------------------------------------------------


class TaintStage(BaseStage):
    """Tool output is untrusted input.

    after():  results from tools marked ``reaches_untrusted`` taint the session.
    before(): while tainted, side-effecting calls require approval (or are denied
              for IRREVERSIBLE tools when ``deny_irreversible`` is set).

    Taint is sticky for the session; call ``SessionState.clear_taint()`` from the
    host when a human has reviewed the untrusted content.
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
