"""Tier-1 adapter: OpenAI Agents SDK.

Wraps each ``FunctionTool`` so the gateway decides before the tool runs:

    Decision.ALLOW             -> tool runs with the model's arguments
    Decision.TRANSFORM         -> tool runs with the rewritten arguments
    Decision.REQUIRE_APPROVAL  -> ``needs_approval`` returns True; the run interrupts with a
                                  ``ToolApprovalItem`` for ``RunState.approve`` / ``reject``
    Decision.DENY              -> the tool does not run; the model receives the structured
                                  error from ``GatewayError.to_model_result()`` and continues

``gateway.before`` runs exactly once per call, inside the SDK's planning-time
``needs_approval`` hook, and the resulting context is cached by ``call_id`` for the
wrapped ``on_invoke_tool``. Output guards run in the wrapper via ``gateway.after``.

Identity comes from ``RunContextWrapper.context``: by default the object you pass as
``Runner.run(..., context=...)`` must expose ``principal``, ``agent`` and ``session``.
Pass ``identity=`` for any other shape.

Out of scope: hosted tools (never enter the process), cross-process resume
(``SessionState`` is in-memory), and gateway ``timeout_s`` (the SDK owns tool timeouts).
The ``agents`` package is imported lazily so the core stays dependency-free.

Usage:

    adapter = OpenAIAgentsAdapter(gateway)
    agent = Agent(name="a", tools=adapter.gate_tools([read_file, send_email]))
    result = await Runner.run(agent, prompt, context=Identity(principal, agent_id, session))
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..context import AgentIdentity, Principal, SessionState
from ..manifest import ToolManifest

IdentityProvider = Callable[[Any], tuple[Principal, AgentIdentity, SessionState]]
"""Receives the SDK's RunContextWrapper (or ToolContext) and returns the gateway identity."""


def default_identity(run_ctx: Any) -> tuple[Principal, AgentIdentity, SessionState]:
    """Read ``principal`` / ``agent`` / ``session`` attributes from ``run_ctx.context``."""
    obj = getattr(run_ctx, "context", None)
    try:
        return obj.principal, obj.agent, obj.session
    except AttributeError:
        raise RuntimeError(
            "run context must expose .principal, .agent and .session; pass identity=... for other shapes"
        ) from None


def manifest_from_function_tool(tool: Any, **overrides: Any) -> ToolManifest:
    """Build a manifest from a ``FunctionTool``'s name, description and JSON schema.

    ``overrides`` take the same values as ``ToolManifest.from_dict`` (enum names or
    values, lists for scopes/tags).
    """
    fields: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": dict(tool.params_json_schema or {}),
    }
    fields.update(overrides)
    return ToolManifest.from_dict(fields)
