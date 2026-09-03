"""Tier-2 adapter: LangGraph ``ToolNode`` and LangChain ``create_agent`` middleware.

One wrapper with the ``(request, execute)`` signature serves ``ToolNode(wrap_tool_call=...)``,
``create_react_agent(tools=ToolNode(...))`` and ``create_agent(middleware=[...])`` — they share
the same ``ToolCallRequest`` class.

    Decision.ALLOW             -> execute(request)
    Decision.TRANSFORM         -> execute(request.override(tool_call={..., "args": rewritten}))
    Decision.REQUIRE_APPROVAL  -> interrupt(payload); on resume True / {"approved": True} the call
                                  is re-evaluated with a one-shot approval and executed, anything
                                  else returns an error ToolMessage
    Decision.DENY              -> error ToolMessage with GatewayError.to_model_result() as JSON

Output guards run on the returned ``ToolMessage.content``. A ``Command`` returned by a tool is
passed through untouched.

Replay: LangGraph re-executes the node on ``Command(resume=...)``, so the wrapper runs from the
top again — ``before`` asks again, ``interrupt()`` returns the resume value, and
``before_approved`` re-checks budget and rate limits. ``session.turn`` and the
"require_approval" audit event therefore occur twice for an approved call. Interrupts need a
checkpointer and ``configurable.thread_id``.

Identity: ``runtime.context`` with ``principal`` / ``agent`` / ``session`` attributes, else
``config["configurable"]["gateway_identity"]`` with the same attributes. Pass ``identity=`` for
other shapes. Cross-process resume still needs the same ``SessionState`` object on both sides.

Usage:

    adapter = LangGraphAdapter(gateway)
    graph.add_node("tools", adapter.tool_node([read_file, send_email]))
    # or: create_agent(model, tools, middleware=[adapter.middleware()], checkpointer=...)
    graph.invoke(state, {"configurable": {"thread_id": "t", "gateway_identity": Identity(...)}})
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from ..context import AgentIdentity, Principal, SessionState, ToolCallContext, ToolResult
from ..decision import Decision, DecisionResult
from ..errors import GatewayError
from ..manifest import ToolManifest
from ..pipeline import Gateway
from .wrap import _run_sync

IdentityProvider = Callable[[Any], tuple[Principal, AgentIdentity, SessionState]]
"""Receives the ToolRuntime and returns the gateway identity."""

CONFIG_KEY = "gateway_identity"
DENIED_MESSAGE = "The tool call was denied."
_DECISION_KEY = "hook_decision"


def default_identity(runtime: Any) -> tuple[Principal, AgentIdentity, SessionState]:
    """``runtime.context`` first, then ``config["configurable"]["gateway_identity"]``."""
    config: Any = getattr(runtime, "config", None) or {}
    sources = (getattr(runtime, "context", None), (config.get("configurable") or {}).get(CONFIG_KEY))
    for source in sources:
        if source is None:
            continue
        try:
            return source.principal, source.agent, source.session
        except AttributeError:
            continue
    raise RuntimeError(
        "no gateway identity: put an object with .principal/.agent/.session in the graph context or in "
        f"config['configurable']['{CONFIG_KEY}'], or pass identity=..."
    )


def manifest_from_tool(tool: Any, **overrides: Any) -> ToolManifest:
    """Build a manifest from a LangChain ``BaseTool``'s name, description and call schema."""
    schema_model = getattr(tool, "tool_call_schema", None)
    if schema_model is not None and hasattr(schema_model, "model_json_schema"):
        schema = dict(schema_model.model_json_schema())
    else:
        schema = {"type": "object", "properties": dict(getattr(tool, "args", None) or {})}
    fields: dict[str, Any] = {"name": tool.name, "description": tool.description or "", "input_schema": schema}
    fields.update(overrides)
    return ToolManifest.from_dict(fields)


def _approved(answer: Any) -> bool:
    return answer is True or (isinstance(answer, Mapping) and answer.get("approved") is True)


def _denial_message(answer: Any) -> str:
    if isinstance(answer, Mapping) and answer.get("message"):
        return str(answer["message"])
    return DENIED_MESSAGE


class LangGraphAdapter:
    def __init__(self, gateway: Gateway, identity: IdentityProvider = default_identity) -> None:
        self.gateway = gateway
        self.identity = identity

    # ------------------------------------------------------------ builders
    def tool_node(self, tools: Any, **kwargs: Any) -> Any:
        """``ToolNode(tools, wrap_tool_call=..., awrap_tool_call=..., **kwargs)``."""
        from langgraph.prebuilt import ToolNode

        return ToolNode(tools, wrap_tool_call=self.wrap_tool_call, awrap_tool_call=self.awrap_tool_call, **kwargs)

    def middleware(self) -> Any:
        """An ``AgentMiddleware`` for LangChain ``create_agent`` that delegates to this adapter."""
        from langchain.agents.middleware import AgentMiddleware

        adapter = self

        class GatewayMiddleware(AgentMiddleware):
            name = "agent_tool_gateway"

            def wrap_tool_call(self, request: Any, handler: Any) -> Any:
                return adapter.wrap_tool_call(request, handler)

            async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
                return await adapter.awrap_tool_call(request, handler)

        return GatewayMiddleware()

    # ------------------------------------------------------------ wrappers
    def wrap_tool_call(self, request: Any, execute: Callable[[Any], Any]) -> Any:
        """Sync wrapper for ``ToolNode.invoke`` / ``AgentMiddleware.wrap_tool_call``."""
        try:
            ctx = self._context(request)
        except GatewayError as e:
            return self._error_message(request, e)
        decision = _run_sync(self.gateway.before(ctx))
        if decision.decision is Decision.REQUIRE_APPROVAL:
            answer = self._ask(ctx, decision)  # interrupt() on this thread, not inside the coroutine
            if not _approved(answer):
                return self._text_error(request, _denial_message(answer))
            decision = _run_sync(self.gateway.before_approved(ctx))
        blocked = self._check(request, ctx, decision)
        if blocked is not None:
            return blocked
        try:
            msg = execute(self._apply(request, ctx, decision))
        except BaseException:
            self.gateway.release(ctx)
            raise
        try:
            return _run_sync(self._finish(ctx, msg))
        except GatewayError as e:
            return self._error_message(request, e)

    async def awrap_tool_call(self, request: Any, execute: Callable[[Any], Any]) -> Any:
        """Async wrapper for ``ToolNode.ainvoke`` / ``AgentMiddleware.awrap_tool_call``."""
        try:
            ctx = self._context(request)
        except GatewayError as e:
            return self._error_message(request, e)
        decision = await self.gateway.before(ctx)
        if decision.decision is Decision.REQUIRE_APPROVAL:
            answer = self._ask(ctx, decision)
            if not _approved(answer):
                return self._text_error(request, _denial_message(answer))
            decision = await self.gateway.before_approved(ctx)
        blocked = self._check(request, ctx, decision)
        if blocked is not None:
            return blocked
        try:
            msg = await execute(self._apply(request, ctx, decision))
        except BaseException:
            self.gateway.release(ctx)
            raise
        try:
            return await self._finish(ctx, msg)
        except GatewayError as e:
            return self._error_message(request, e)

    # ------------------------------------------------------------- helpers
    def _context(self, request: Any) -> ToolCallContext:
        tc = request.tool_call
        runtime = request.runtime
        principal, agent, session = self.identity(runtime)
        session.turn += 1
        return self.gateway.build_context(
            tc["name"],
            dict(tc.get("args") or {}),
            principal=principal,
            agent=agent,
            session=session,
            tool_call_id=tc.get("id") or getattr(runtime, "tool_call_id", None),
        )

    def _ask(self, ctx: ToolCallContext, decision: DecisionResult) -> Any:
        from langgraph.types import interrupt

        return interrupt(
            {
                "approval_id": decision.approval_id,
                "reason": decision.reason,
                "tool": ctx.tool.name,
                "args": dict(ctx.args),
                "tool_call_id": ctx.tool_call_id,
            }
        )

    def _check(self, request: Any, ctx: ToolCallContext, decision: DecisionResult) -> Any:
        """Record the decision; return an error ToolMessage if it blocks, else None."""
        ctx.metadata[_DECISION_KEY] = decision.decision.value
        if not decision.blocked:
            return None
        try:
            Gateway.raise_for_decision(decision)
        except GatewayError as e:
            return self._error_message(request, e)
        return None  # pragma: no cover

    @staticmethod
    def _apply(request: Any, ctx: ToolCallContext, decision: DecisionResult) -> Any:
        if decision.decision is Decision.TRANSFORM:
            return request.override(tool_call={**request.tool_call, "args": dict(ctx.args)})
        return request

    async def _finish(self, ctx: ToolCallContext, msg: Any) -> Any:
        from langchain_core.messages import ToolMessage

        if not isinstance(msg, ToolMessage):  # a Command from a state-updating tool
            self.gateway.settle(ctx)
            return msg
        result = await self.gateway.after(ctx, ToolResult(content=msg.content))
        msg.content = result.content
        return msg

    @staticmethod
    def _error_message(request: Any, err: GatewayError) -> Any:
        return LangGraphAdapter._text_error(request, json.dumps(err.to_model_result(), default=str))

    @staticmethod
    def _text_error(request: Any, text: str) -> Any:
        from langchain_core.messages import ToolMessage

        tc = request.tool_call
        return ToolMessage(content=text, tool_call_id=tc["id"], name=tc["name"], status="error")
