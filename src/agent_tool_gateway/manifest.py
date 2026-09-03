"""Tool manifests: the security metadata every tool carries.

The manifest is the stable public contract of this project. Policy, risk scoring,
rate limiting and taint handling all key off these fields rather than off the
tool name alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any


class SideEffect(str, Enum):
    """What the tool does to the world."""

    READ = "read"  # observes only
    WRITE = "write"  # mutates, but recoverable
    IRREVERSIBLE = "irreversible"  # cannot be undone (send, delete, pay, deploy)


class RiskTier(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass(frozen=True)
class ToolManifest:
    name: str
    description: str = ""
    version: str = "1.0"
    input_schema: dict[str, Any] = field(default_factory=dict)
    side_effect: SideEffect = SideEffect.READ
    risk_tier: RiskTier = RiskTier.LOW
    required_scopes: frozenset[str] = frozenset()
    reaches_untrusted: bool = False  # output may contain attacker-controlled content
    output_classification: str = "internal"  # e.g. public / internal / confidential / pii
    cost_usd: float = 0.0  # nominal cost per call, charged against session budget
    timeout_s: float | None = 30.0
    tags: frozenset[str] = frozenset()

    def with_overrides(self, **kwargs: Any) -> ToolManifest:
        from dataclasses import replace

        return replace(self, **kwargs)
