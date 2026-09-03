"""Built-in stages and a sensible default ordering.

Order matters. The default pipeline is:

    schema → scope → policy → taint → risk → guardrails → loop_detect → budget → rate_limit

Cheap, deterministic checks first; approval-producing checks before risk so a
policy-level approval short-circuits scoring; limits last so denied calls do
not consume quota.
"""

from __future__ import annotations

from ..pipeline import Stage
from .authz import PolicyStage, Rule, RulePolicy, SchemaStage, ScopeStage, grant_approval
from .guardrails import (
    GuardrailStage,
    InputGuard,
    MaxOutputChars,
    OutputGuard,
    TaintStage,
    no_secrets_in_args,
    redact_pii,
)
from .limits import BudgetStage, LoopDetectStage, RateLimiter, RateLimitStage, TokenBucketLimiter
from .risk import RiskScorer, RiskStage

__all__ = [
    "BudgetStage",
    "GuardrailStage",
    "InputGuard",
    "LoopDetectStage",
    "MaxOutputChars",
    "OutputGuard",
    "PolicyStage",
    "RateLimitStage",
    "RateLimiter",
    "RiskScorer",
    "RiskStage",
    "Rule",
    "RulePolicy",
    "SchemaStage",
    "ScopeStage",
    "TaintStage",
    "TokenBucketLimiter",
    "default_stages",
    "grant_approval",
    "no_secrets_in_args",
    "redact_pii",
]


def default_stages(
    policy: RulePolicy,
    *,
    limiter: RateLimiter | None = None,
    scorer: RiskScorer | None = None,
    input_guards: list[InputGuard] | None = None,
    output_guards: list[OutputGuard] | None = None,
) -> list[Stage]:
    return [
        SchemaStage(),
        ScopeStage(),
        PolicyStage(policy),
        TaintStage(),
        RiskStage(scorer),
        GuardrailStage(
            inputs=input_guards if input_guards is not None else [no_secrets_in_args],
            outputs=output_guards if output_guards is not None else [redact_pii, MaxOutputChars()],
        ),
        LoopDetectStage(),
        BudgetStage(),
        RateLimitStage(limiter or TokenBucketLimiter(rate_per_s=5.0, burst=20.0)),
    ]
