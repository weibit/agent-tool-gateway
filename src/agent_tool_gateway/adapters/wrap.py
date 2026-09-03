"""Tier-3 adapter: wrap any Python callable.

Works with hand-rolled agent loops and any framework that ultimately calls a
Python function. Identity/session are bound with ``bind()`` (contextvars) so
wrapped tools need no signature changes.

    gw = Gateway(registry, default_stages(policy))

    @gw_wrap(gw, "read_file")
    def read_file(path: str) -> str: ...

    with bind(principal=p, agent=a, session=s):
        read_file(path="README.md")
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from ..context import AgentIdentity, Principal, SessionState
from ..errors import GatewayError
from ..pipeline import Gateway

_principal: contextvars.ContextVar[Principal | None] = contextvars.ContextVar("atg_principal", default=None)
_agent: contextvars.ContextVar[AgentIdentity | None] = contextvars.ContextVar("atg_agent", default=None)
_session: contextvars.ContextVar[SessionState | None] = contextvars.ContextVar("atg_session", default=None)


@contextmanager
def bind(*, principal: Principal, agent: AgentIdentity, session: SessionState) -> Iterator[None]:
    tokens = (_principal.set(principal), _agent.set(agent), _session.set(session))
    try:
        yield
    finally:
        _principal.reset(tokens[0])
        _agent.reset(tokens[1])
        _session.reset(tokens[2])


def current() -> tuple[Principal, AgentIdentity, SessionState]:
    p, a, s = _principal.get(), _agent.get(), _session.get()
    if p is None or a is None or s is None:
        raise RuntimeError("no gateway identity bound; wrap the call in adapters.wrap.bind(...)")
    return p, a, s


def gw_wrap(
    gateway: Gateway, tool_name: str | None = None, *, return_errors: bool = True
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator. If ``return_errors`` the wrapper returns ``err.to_model_result()`` instead of raising."""

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        name = tool_name or fn.__name__
        is_async = inspect.iscoroutinefunction(fn)

        async def run(kwargs: dict[str, Any]) -> Any:
            p, a, s = current()
            ctx = gateway.build_context(name, kwargs, principal=p, agent=a, session=s)
            try:
                res = await gateway.call(ctx, fn)
                return res.content
            except GatewayError as e:
                if return_errors:
                    return e.to_model_result()
                raise

        if is_async:

            @functools.wraps(fn)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                return await run(_kwargs_of(fn, args, kwargs))

            return awrapper

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return _run_sync(run(_kwargs_of(fn, args, kwargs)))

        return wrapper

    return deco


def _kwargs_of(fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    bound = inspect.signature(fn).bind_partial(*args, **kwargs)
    return dict(bound.arguments)


def _run_sync(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Called from inside a running loop (e.g. a sync tool invoked by an async framework).
    # Run on a private loop in a worker thread, carrying the caller's contextvars (bind()).
    import concurrent.futures

    context = contextvars.copy_context()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(context.run, asyncio.run, coro).result()
