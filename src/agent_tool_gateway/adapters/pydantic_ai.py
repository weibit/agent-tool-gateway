"""Tier-2 adapter: Pydantic AI toolset wrapper.

``GatedToolset`` wraps any toolset (``FunctionToolset``, MCP, ...) so every call goes
through the gateway:

    Decision.ALLOW             -> wrapped tool runs with the model's arguments
    Decision.TRANSFORM         -> wrapped tool runs with the rewritten arguments
    Decision.REQUIRE_APPROVAL  -> raises ``ApprovalRequired``; the run ends with
                                  ``DeferredToolRequests`` and resumes via ``DeferredToolResults``
    Decision.DENY              -> returns ``GatewayError.to_model_result()`` as the tool result

Deny is a returned dict on purpose: ``ModelRetry`` counts against the tool's ``max_retries``
(default 1) and a second denied call would end the run with ``UnexpectedModelBehavior``.

``before`` runs on every entry, including the approved re-entry (``ctx.tool_call_approved``),
so state that changed while the approval was pending is still enforced. Output guards run via
``gateway.after``.

Identity comes from ``ctx.deps``: by default the ``deps`` object must expose ``principal``,
``agent`` and ``session``. Pass ``identity=`` for other shapes.

Gate outermost. ``PrefixedToolset`` strips its prefix before delegating inward, so wrap the
prefixed toolset (``GatedToolset(ts.prefixed("x"), gw)``) to see the names the model uses.

The agent's ``output_type`` must include ``DeferredToolRequests`` for approvals to surface;
Pydantic AI raises ``UserError`` otherwise. Cross-process resume needs a serialisable
``SessionState`` (not yet available). Gateway ``timeout_s`` is not applied; use
``ToolDefinition.timeout``.

Usage:

    agent = Agent(model, toolsets=[GatedToolset(my_toolset, gateway)], deps_type=Identity,
                  output_type=[str, DeferredToolRequests])
    result = await agent.run(prompt, deps=Identity(principal, agent_id, session))
    if isinstance(result.output, DeferredToolRequests):
        approvals = {c.tool_call_id: True for c in result.output.approvals}
        result = await agent.run(message_history=result.all_messages(),
                                 deferred_tool_results=DeferredToolResults(approvals=approvals), deps=...)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

try:
    from pydantic_ai import ApprovalRequired
    from pydantic_ai.tools import AgentDepsT, RunContext
    from pydantic_ai.toolsets import WrapperToolset
    from pydantic_ai.toolsets.abstract import ToolsetTool
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "agent_tool_gateway.adapters.pydantic_ai needs the 'pydantic' extra: pip install 'agent-tool-gateway[pydantic]'"
    ) from e

from ..context import AgentIdentity, Principal, SessionState, ToolResult
from ..decision import Decision
from ..errors import GatewayError
from ..manifest import ToolManifest
from ..pipeline import Gateway

IdentityProvider = Callable[[Any], tuple[Principal, AgentIdentity, SessionState]]
"""Receives the RunContext and returns the gateway identity."""

_DECISION_KEY = "hook_decision"  # same key the other adapters use


def default_identity(ctx: Any) -> tuple[Principal, AgentIdentity, SessionState]:
    """Read ``principal`` / ``agent`` / ``session`` attributes from ``ctx.deps``."""
    deps: Any = getattr(ctx, "deps", None)
    try:
        return deps.principal, deps.agent, deps.session
    except AttributeError:
        raise RuntimeError(
            "deps must expose .principal, .agent and .session; pass identity=... for other shapes"
        ) from None


def manifest_from_tool_def(tool_def: Any, **overrides: Any) -> ToolManifest:
    """Build a manifest from a ``ToolDefinition``'s name, description and JSON schema."""
    fields: dict[str, Any] = {
        "name": tool_def.name,
        "description": tool_def.description or "",
        "input_schema": dict(tool_def.parameters_json_schema or {}),
    }
    fields.update(overrides)
    return ToolManifest.from_dict(fields)


@dataclass
class GatedToolset(WrapperToolset[AgentDepsT]):
    """A toolset whose every call is decided by the gateway before it reaches ``wrapped``."""

    gateway: Gateway
    identity: IdentityProvider = default_identity

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        principal, agent, session = self.identity(ctx)
        session.turn += 1
        try:
            gw_ctx = self.gateway.build_context(
                name, tool_args, principal=principal, agent=agent, session=session, tool_call_id=ctx.tool_call_id
            )
        except GatewayError as e:
            return e.to_model_result()

        decision = await self.gateway.before(gw_ctx)
        gw_ctx.metadata[_DECISION_KEY] = decision.decision.value
        if decision.decision is Decision.DENY:
            try:
                Gateway.raise_for_decision(decision)
            except GatewayError as e:
                return e.to_model_result()
        elif decision.decision is Decision.REQUIRE_APPROVAL:
            if not ctx.tool_call_approved:
                raise ApprovalRequired(metadata={"approval_id": decision.approval_id, "reason": decision.reason})
            self.gateway.reserve(gw_ctx)  # the host approved via DeferredToolResults

        args = gw_ctx.args if decision.decision is Decision.TRANSFORM else tool_args
        try:
            raw = await super().call_tool(name, args, ctx, tool)
        except BaseException:
            self.gateway.release(gw_ctx)  # the SDK owns the failure; just drop the reservation
            raise
        try:
            result = await self.gateway.after(gw_ctx, ToolResult(content=raw))
        except GatewayError as e:
            return e.to_model_result()
        return result.content


def gate_toolset(gateway: Gateway, toolset: Any, identity: IdentityProvider = default_identity) -> GatedToolset[Any]:
    """Convenience: ``GatedToolset(toolset, gateway, identity)``."""
    return GatedToolset(toolset, gateway, identity)
