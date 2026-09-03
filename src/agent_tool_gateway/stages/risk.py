"""Risk scoring: the second stage of authorize-then-risk.

Policy answers "may this principal/agent do this?". Risk answers "given the
runtime context, how dangerous is doing it *now*?" and maps the score onto
ALLOW / REQUIRE_APPROVAL / DENY.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..context import ToolCallContext
from ..decision import DecisionResult
from ..manifest import SideEffect
from ..pipeline import BaseStage
from .authz import _approved

RiskSignal = Callable[[ToolCallContext], float]


@dataclass
class RiskScorer:
    """Additive risk model. Replace or extend the signals list to tune."""

    approval_threshold: float = 6.0
    deny_threshold: float = 10.0
    signals: list[RiskSignal] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.signals:
            self.signals = [
                tier_signal,
                side_effect_signal,
                taint_signal,
                depth_signal,
                classification_signal,
            ]

    def score(self, ctx: ToolCallContext) -> tuple[float, dict[str, float]]:
        breakdown = {s.__name__: float(s(ctx)) for s in self.signals}
        return sum(breakdown.values()), breakdown


# ---- default signals --------------------------------------------------------


def tier_signal(ctx: ToolCallContext) -> float:
    return {1: 0.0, 2: 1.5, 3: 3.0, 4: 6.0}[int(ctx.tool.risk_tier)]


def side_effect_signal(ctx: ToolCallContext) -> float:
    return {SideEffect.READ: 0.0, SideEffect.WRITE: 2.0, SideEffect.IRREVERSIBLE: 4.0}[ctx.tool.side_effect]


def taint_signal(ctx: ToolCallContext) -> float:
    """Side-effecting calls issued after consuming untrusted content are XPIA-shaped."""
    if not ctx.session.tainted:
        return 0.0
    return 0.0 if ctx.tool.side_effect is SideEffect.READ else 3.0


def depth_signal(ctx: ToolCallContext) -> float:
    return 0.5 * ctx.agent.depth


def classification_signal(ctx: ToolCallContext) -> float:
    return {"public": 0.0, "internal": 0.0, "confidential": 1.0, "pii": 2.0}.get(ctx.tool.output_classification, 0.5)


class RiskStage(BaseStage):
    name = "risk"

    def __init__(self, scorer: RiskScorer | None = None) -> None:
        self.scorer = scorer or RiskScorer()

    async def before(self, ctx: ToolCallContext) -> DecisionResult | None:
        score, breakdown = self.scorer.score(ctx)
        ctx.metadata["risk_breakdown"] = breakdown
        if score >= self.scorer.deny_threshold:
            res = DecisionResult.deny(
                f"Call to '{ctx.tool.name}' exceeds the risk limit in the current context.",
                stage=self.name,
                breakdown=breakdown,
            )
        elif score >= self.scorer.approval_threshold and not _approved(ctx):
            res = DecisionResult.require_approval(
                f"'{ctx.tool.name}' requires approval in the current context.", stage=self.name, breakdown=breakdown
            )
        else:
            res = DecisionResult.allow(stage=self.name)
        res.risk_score = score
        return res
