"""Schema validation, scope authorization, and rule-based resource policy."""

from __future__ import annotations

import fnmatch
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..context import SessionState, ToolCallContext
from ..decision import Decision, DecisionResult
from ..pipeline import BaseStage

# --------------------------------------------------------------------- schema


class SchemaStage(BaseStage):
    """Validate ``ctx.args`` against ``manifest.input_schema``.

    Uses ``jsonschema`` when installed; otherwise falls back to a minimal
    required/type check so the core has no hard dependency.
    """

    name = "schema"

    async def before(self, ctx: ToolCallContext) -> DecisionResult | None:
        schema = ctx.tool.input_schema
        if not schema:
            return None
        err = _validate(schema, ctx.args)
        if err:
            return DecisionResult.deny(
                f"Invalid arguments for '{ctx.tool.name}': {err}", stage=self.name, error="invalid_arguments"
            )
        return None


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list, tuple),
    "null": (type(None),),
}


def _validate(schema: dict[str, Any], args: dict[str, Any]) -> str | None:
    """Return a model-facing validation message, or None when the args are valid.

    A malformed *schema* (a manifest bug) raises, which the pipeline turns into a
    fail-closed deny with the detail in the audit log rather than blaming the model.
    """
    try:
        import jsonschema
    except ImportError:
        return _validate_fallback(schema, args)
    try:
        jsonschema.validate(args, schema)
    except jsonschema.ValidationError as e:
        return str(e.message)
    return None


def _validate_fallback(schema: dict[str, Any], args: dict[str, Any]) -> str | None:
    """Minimal required/type/additionalProperties check for the top level only."""
    for key in schema.get("required", []):
        if key not in args:
            return f"missing required field '{key}'"
    props = schema.get("properties", {})
    for key, val in args.items():
        spec = props.get(key)
        if spec is None:
            if schema.get("additionalProperties") is False:
                return f"unexpected field '{key}'"
            continue
        t = spec.get("type")
        if t and t in _JSON_TYPES and not isinstance(val, _JSON_TYPES[t]):
            return f"field '{key}' must be {t}"
        if t == "integer" and isinstance(val, bool):
            return f"field '{key}' must be integer"
    return None


# --------------------------------------------------------------------- scopes


class ScopeStage(BaseStage):
    """Tool.required_scopes must be a subset of principal.scopes ∩ agent.effective_scopes."""

    name = "scope"

    async def before(self, ctx: ToolCallContext) -> DecisionResult | None:
        missing = ctx.tool.required_scopes - ctx.effective_scopes
        if missing:
            return DecisionResult.deny(
                f"Not authorized to use '{ctx.tool.name}'.",
                stage=self.name,
                missing_scopes=sorted(missing),
            )
        return None


# --------------------------------------------------------------------- policy

Condition = Callable[[ToolCallContext], bool]
Transformer = Callable[[ToolCallContext], dict[str, Any]]


@dataclass
class Rule:
    """A single policy rule. ``tool`` is a glob; ``when`` is an argument/context predicate."""

    tool: str
    effect: Decision
    reason: str = ""
    when: Condition | None = None
    transform: Transformer | None = None  # used when effect == TRANSFORM
    priority: int = 0

    def matches(self, ctx: ToolCallContext) -> bool:
        # fnmatchcase: plain fnmatch is case-insensitive on Windows, which would make policy OS-dependent.
        if not fnmatch.fnmatchcase(ctx.tool.name, self.tool):
            return False
        return True if self.when is None else bool(self.when(ctx))


@dataclass
class RulePolicy:
    """First-match rule evaluation, highest priority first. Fail-closed by default.

    This is intentionally simple. A Cedar / OPA / CEL backend implements the same
    ``PolicyStage``-facing interface (``evaluate(ctx) -> DecisionResult``).
    """

    rules: list[Rule] = field(default_factory=list)
    default: Decision = Decision.DENY
    default_reason: str = "No policy rule permits this call."

    def add(self, rule: Rule) -> RulePolicy:
        self.rules.append(rule)
        self.rules.sort(key=lambda r: -r.priority)
        return self

    def allow(self, tool: str, when: Condition | None = None, reason: str = "", priority: int = 0) -> RulePolicy:
        return self.add(Rule(tool, Decision.ALLOW, reason, when, priority=priority))

    def deny(self, tool: str, when: Condition | None = None, reason: str = "", priority: int = 0) -> RulePolicy:
        return self.add(Rule(tool, Decision.DENY, reason, when, priority=priority))

    def require_approval(
        self, tool: str, when: Condition | None = None, reason: str = "", priority: int = 0
    ) -> RulePolicy:
        return self.add(Rule(tool, Decision.REQUIRE_APPROVAL, reason, when, priority=priority))

    def rewrite(
        self, tool: str, transform: Transformer, when: Condition | None = None, reason: str = "", priority: int = 0
    ) -> RulePolicy:
        return self.add(Rule(tool, Decision.TRANSFORM, reason, when, transform=transform, priority=priority))

    def evaluate(self, ctx: ToolCallContext) -> DecisionResult:
        for rule in self.rules:
            if not rule.matches(ctx):
                continue
            if rule.effect is Decision.ALLOW:
                return DecisionResult.allow(stage="policy", reason=rule.reason)
            if rule.effect is Decision.DENY:
                return DecisionResult.deny(rule.reason or "Denied by policy.", stage="policy")
            if rule.effect is Decision.REQUIRE_APPROVAL:
                return DecisionResult.require_approval(rule.reason or "Approval required by policy.", stage="policy")
            if rule.effect is Decision.TRANSFORM and rule.transform:
                return DecisionResult.transform(
                    rule.transform(ctx), rule.reason or "Arguments rewritten by policy.", stage="policy"
                )
        if self.default is Decision.ALLOW:
            return DecisionResult.allow(stage="policy", reason="default allow")
        return DecisionResult.deny(self.default_reason, stage="policy")


class PolicyStage(BaseStage):
    """Argument-level authorization. Honors approvals already granted in the session."""

    name = "policy"

    def __init__(self, policy: RulePolicy) -> None:
        self.policy = policy

    async def before(self, ctx: ToolCallContext) -> DecisionResult | None:
        res = self.policy.evaluate(ctx)
        if res.decision is Decision.REQUIRE_APPROVAL and _approved(ctx):
            return DecisionResult.allow(stage=self.name, reason="pre-approved")
        return res


def _approved(ctx: ToolCallContext) -> bool:
    """An approval is granted for (tool, args_hash); a wildcard approves the tool."""
    return ctx.approval_key in ctx.session.approvals or f"{ctx.tool.name}:*" in ctx.session.approvals


def grant_approval(ctx: ToolCallContext, *, any_args: bool = False) -> str:
    """Helper for adapters/UIs: record a human approval on the session."""
    key = f"{ctx.tool.name}:*" if any_args else ctx.approval_key
    ctx.session.approvals.add(key)
    return key


def grant_approval_by_id(session: SessionState, approval_id: str) -> str | None:
    """Redeem the ``approval_id`` from a REQUIRE_APPROVAL decision.

    Returns the approval key now granted, or None if the id is unknown (already
    redeemed, expired from the bounded pending list, or from another session).
    """
    key = session.pending_approvals.pop(approval_id, None)
    if key is not None:
        session.approvals.add(key)
    return key
