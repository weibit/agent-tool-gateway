"""agent-tool-gateway: in-process policy enforcement at the tool-call boundary."""

from .audit import AuditEvent, AuditSink, InMemoryAuditSink, JsonlAuditSink, NullAuditSink
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
from .manifest import RiskTier, SideEffect, ToolManifest
from .pipeline import BaseStage, Gateway, Stage
from .registry import Resolver, ToolRegistry, glob_overlay, lookup

__version__ = "0.1.0"

__all__ = [
    "AgentIdentity",
    "ApprovalRequired",
    "AuditEvent",
    "AuditSink",
    "AuthorizationDenied",
    "BaseStage",
    "BudgetExceeded",
    "Decision",
    "DecisionResult",
    "Gateway",
    "GatewayError",
    "GuardrailViolation",
    "InMemoryAuditSink",
    "JsonlAuditSink",
    "NullAuditSink",
    "Principal",
    "RateLimited",
    "Resolver",
    "RiskTier",
    "SchemaValidationError",
    "SessionState",
    "SideEffect",
    "Stage",
    "ToolCallContext",
    "ToolExecutionError",
    "ToolManifest",
    "ToolNotRegistered",
    "ToolRegistry",
    "ToolResult",
    "ToolTimeout",
    "glob_overlay",
    "lookup",
]
