"""The decision model. Four outcomes, all first-class.

ALLOW            proceed with the arguments as given
TRANSFORM        proceed, but with rewritten arguments
REQUIRE_APPROVAL pause; a human (or higher authority) must grant an approval id
DENY             do not execute; the model receives a structured, recoverable error
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Decision(str, Enum):
    ALLOW = "allow"
    TRANSFORM = "transform"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


@dataclass
class DecisionResult:
    decision: Decision
    reason: str = ""
    stage: str = ""
    updated_args: dict[str, Any] | None = None
    approval_id: str | None = None
    risk_score: float | None = None
    retry_after_s: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return self.decision in (Decision.DENY, Decision.REQUIRE_APPROVAL)

    # ---- constructors -----------------------------------------------------
    @classmethod
    def allow(cls, stage: str = "", reason: str = "") -> DecisionResult:
        return cls(Decision.ALLOW, reason=reason, stage=stage)

    @classmethod
    def deny(cls, reason: str, stage: str = "", **details: Any) -> DecisionResult:
        return cls(Decision.DENY, reason=reason, stage=stage, details=details)

    @classmethod
    def transform(cls, updated_args: dict[str, Any], reason: str, stage: str = "") -> DecisionResult:
        return cls(Decision.TRANSFORM, reason=reason, stage=stage, updated_args=updated_args)

    @classmethod
    def require_approval(
        cls, reason: str, stage: str = "", approval_id: str | None = None, **details: Any
    ) -> DecisionResult:
        return cls(
            Decision.REQUIRE_APPROVAL,
            reason=reason,
            stage=stage,
            approval_id=approval_id or uuid.uuid4().hex[:12],
            details=details,
        )
