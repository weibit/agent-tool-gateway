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

import dataclasses
import inspect
import json
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any

from ..context import AgentIdentity, Principal, SessionState, ToolCallContext, ToolResult
from ..decision import Decision, DecisionResult
from ..errors import GatewayError
from ..manifest import ToolManifest
from ..pipeline import Gateway

IdentityProvider = Callable[[Any], tuple[Principal, AgentIdentity, SessionState]]
"""Receives the SDK's RunContextWrapper (or ToolContext) and returns the gateway identity."""

_DECISION_KEY = "hook_decision"  # same key the Claude adapter uses
_RESULT_KEY = "_decision_result"  # DecisionResult cached between needs_approval and invoke


def default_identity(run_ctx: Any) -> tuple[Principal, AgentIdentity, SessionState]:
    """Read ``principal`` / ``agent`` / ``session`` attributes from ``run_ctx.context``."""
    obj: Any = getattr(run_ctx, "context", None)
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


def _error_json(err: GatewayError) -> str:
    return json.dumps(err.to_model_result(), default=str)


def _parse_args(args_json: str) -> dict[str, Any]:
    try:
        parsed = json.loads(args_json or "{}")
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class OpenAIAgentsAdapter:
    def __init__(
        self, gateway: Gateway, identity: IdentityProvider = default_identity, *, max_inflight: int = 256
    ) -> None:
        self.gateway = gateway
        self.identity = identity
        self.max_inflight = max_inflight
        # call_id -> context built at planning time, or the GatewayError that prevented building one.
        # Bounded: a rejected approval never reaches invoke, so entries can be orphaned.
        self._inflight: OrderedDict[str, ToolCallContext | GatewayError] = OrderedDict()

    def _remember(self, call_id: str, entry: ToolCallContext | GatewayError) -> None:
        self._inflight[call_id] = entry
        while len(self._inflight) > self.max_inflight:
            self._inflight.popitem(last=False)

    # -------------------------------------------------------------- wrapping
    def gate_tool(self, tool: Any) -> Any:
        """Return a copy of ``tool`` (a ``FunctionTool``) whose calls go through the gateway."""
        from agents import FunctionTool

        if not isinstance(tool, FunctionTool):
            raise TypeError(f"gate_tool expects a FunctionTool, got {type(tool).__name__}")
        original_invoke = tool.on_invoke_tool
        user_needs = tool.needs_approval
        tool_name = tool.name

        async def needs_approval(run_ctx: Any, params: dict[str, Any], call_id: str) -> bool:
            gateway_needs = await self.needs_approval(tool_name, run_ctx, params, call_id)
            if user_needs is True:
                return True
            if callable(user_needs):
                maybe: Any = user_needs(run_ctx, params, call_id)
                if inspect.isawaitable(maybe):
                    maybe = await maybe
                return bool(maybe) or gateway_needs
            return gateway_needs

        async def on_invoke(tool_ctx: Any, args_json: str) -> Any:
            return await self.invoke(tool_name, original_invoke, tool_ctx, args_json)

        return dataclasses.replace(tool, needs_approval=needs_approval, on_invoke_tool=on_invoke)

    def gate_tools(self, tools: Sequence[Any]) -> list[Any]:
        """Wrap every ``FunctionTool``; hosted and other tool types pass through untouched."""
        from agents import FunctionTool

        return [self.gate_tool(t) if isinstance(t, FunctionTool) else t for t in tools]

    # ---------------------------------------------------------------- hooks
    async def needs_approval(self, tool_name: str, run_ctx: Any, params: dict[str, Any], call_id: str) -> bool:
        """Planning-time hook: run ``before`` once, cache the outcome, interrupt on REQUIRE_APPROVAL."""
        principal, agent, session = self.identity(run_ctx)
        session.turn += 1
        try:
            ctx = self.gateway.build_context(
                tool_name, params, principal=principal, agent=agent, session=session, tool_call_id=call_id
            )
        except GatewayError as e:
            self._remember(call_id, e)
            return False
        decision = await self.gateway.before(ctx)
        ctx.metadata[_DECISION_KEY] = decision.decision.value
        ctx.metadata[_RESULT_KEY] = decision
        self._remember(call_id, ctx)
        return decision.decision is Decision.REQUIRE_APPROVAL

    async def invoke(self, tool_name: str, original_invoke: Any, tool_ctx: Any, args_json: str) -> Any:
        """Execution-time wrapper: apply the cached decision, run the tool, run output guards."""
        call_id = getattr(tool_ctx, "tool_call_id", None) or ""
        entry = self._inflight.pop(call_id, None)
        if isinstance(entry, GatewayError):
            return _error_json(entry)

        ctx = entry
        decision: DecisionResult
        if ctx is None:
            # No planning step ran (tool invoked directly). Decide inline; an approval can no
            # longer interrupt the run, so it surfaces as an approval_required error instead.
            principal, agent, session = self.identity(tool_ctx)
            try:
                ctx = self.gateway.build_context(
                    tool_name,
                    _parse_args(args_json),
                    principal=principal,
                    agent=agent,
                    session=session,
                    tool_call_id=call_id or None,
                )
                decision = await self.gateway.before(ctx)
                Gateway.raise_for_decision(decision)
            except GatewayError as e:
                return _error_json(e)
        else:
            decision = ctx.metadata.pop(_RESULT_KEY)
            if decision.decision is Decision.DENY:
                try:
                    Gateway.raise_for_decision(decision)
                except GatewayError as e:
                    return _error_json(e)
            elif decision.decision is Decision.REQUIRE_APPROVAL:
                # The host approved via RunState.approve; the gateway never saw the call start.
                self.gateway.reserve(ctx)

        payload = json.dumps(ctx.args, default=str) if decision.decision is Decision.TRANSFORM else args_json
        raw = await original_invoke(tool_ctx, payload)
        try:
            result = await self.gateway.after(ctx, ToolResult(content=raw))
        except GatewayError as e:
            return _error_json(e)
        return result.content


def gate_tools(gateway: Gateway, tools: Sequence[Any], identity: IdentityProvider = default_identity) -> list[Any]:
    """One-shot convenience: ``OpenAIAgentsAdapter(gateway, identity).gate_tools(tools)``."""
    return OpenAIAgentsAdapter(gateway, identity).gate_tools(tools)
