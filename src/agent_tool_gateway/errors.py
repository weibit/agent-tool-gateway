"""Exception taxonomy with two output channels.

Every error carries a ``model_message`` (safe, non-leaking, actionable for the
LLM) and an ``audit_detail`` (full fidelity, for logs only). Adapters must only
ever surface ``to_model_result()`` to the model.
"""

from __future__ import annotations

from typing import Any


class GatewayError(Exception):
    code = "gateway_error"
    retryable = False

    def __init__(self, model_message: str, audit_detail: str | None = None, **extra: Any) -> None:
        super().__init__(model_message)
        self.model_message = model_message
        self.audit_detail = audit_detail or model_message
        self.extra = extra

    def to_model_result(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "error": self.code,
            "message": self.model_message,
            "retryable": self.retryable,
        }
        out.update({k: v for k, v in self.extra.items() if not k.startswith("_")})
        return out


class ToolNotRegistered(GatewayError):
    code = "tool_not_registered"


class SchemaValidationError(GatewayError):
    code = "invalid_arguments"
    retryable = True  # the model can fix its arguments


class AuthorizationDenied(GatewayError):
    code = "authorization_denied"


class ApprovalRequired(GatewayError):
    code = "approval_required"
    retryable = True  # retry once approval is granted


class RateLimited(GatewayError):
    code = "rate_limited"
    retryable = True


class BudgetExceeded(GatewayError):
    code = "budget_exceeded"


class GuardrailViolation(GatewayError):
    code = "guardrail_violation"
    retryable = True


class ToolTimeout(GatewayError):
    code = "tool_timeout"
    retryable = True


class ToolExecutionError(GatewayError):
    """Wraps an arbitrary exception raised by the tool itself.

    The original exception text is kept in ``audit_detail`` only.
    """

    code = "tool_execution_error"
    retryable = True

    @classmethod
    def from_exception(cls, exc: BaseException, tool_name: str) -> ToolExecutionError:
        return cls(
            model_message=f"Tool '{tool_name}' failed to execute. You may retry or try another approach.",
            audit_detail=f"{type(exc).__name__}: {exc}",
        )
