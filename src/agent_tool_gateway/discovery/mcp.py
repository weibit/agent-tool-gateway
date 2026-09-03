"""Map MCP ``tools/list`` results onto manifests.

MCP tool annotations (readOnlyHint, destructiveHint, idempotentHint,
openWorldHint) are hints declared by the server. For a ``trusted`` server they
map directly onto the manifest. For an untrusted server (the default) they are
clamped: nothing is treated as read-only, risk is at least HIGH, and output is
always considered untrusted. Relax individual tools with an operator overlay
once a human has reviewed them.

Absent annotations take the MCP spec defaults: ``destructiveHint=true`` and
``openWorldHint=true``, so an unannotated tool is IRREVERSIBLE and untrusted.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..manifest import RiskTier, SideEffect, ToolManifest

DEFAULT_PREFIX = "mcp__{server}__{tool}"  # the Claude Agent SDK / Claude Code naming convention

_TIER_FOR = {SideEffect.READ: RiskTier.LOW, SideEffect.WRITE: RiskTier.MEDIUM, SideEffect.IRREVERSIBLE: RiskTier.HIGH}


def _as_mapping(tool: Any) -> Mapping[str, Any]:
    if isinstance(tool, Mapping):
        return tool
    dump = getattr(tool, "model_dump", None)  # the `mcp` SDK's pydantic Tool model
    if callable(dump):
        dumped = dump()
        if isinstance(dumped, Mapping):
            return dumped
    raise TypeError(f"unsupported MCP tool description: {type(tool).__name__}")


def manifests_from_mcp(
    server: str,
    tools: Iterable[Any],
    *,
    trusted: bool = False,
    required_scopes: Iterable[str] | None = None,
    prefix: str = DEFAULT_PREFIX,
) -> list[ToolManifest]:
    """Build one manifest per MCP tool.

    ``required_scopes`` defaults to ``{"mcp:<server>"}`` so a principal must be
    granted the server before any of its tools resolve past the scope stage.
    """
    scopes = frozenset({f"mcp:{server}"}) if required_scopes is None else frozenset(required_scopes)
    base_tags = {"mcp", f"mcp:{server}", "trusted" if trusted else "untrusted"}
    out: list[ToolManifest] = []
    for raw in tools:
        tool = _as_mapping(raw)
        tool_name = tool.get("name")
        if not tool_name:
            raise ValueError(f"MCP tool from server {server!r} has no name: {dict(tool)!r}")
        ann = tool.get("annotations") or {}
        read_only = bool(ann.get("readOnlyHint", False))
        destructive = bool(ann.get("destructiveHint", True))
        idempotent = bool(ann.get("idempotentHint", False))
        open_world = bool(ann.get("openWorldHint", True))

        if read_only:
            side_effect = SideEffect.READ
        elif destructive:
            side_effect = SideEffect.IRREVERSIBLE
        else:
            side_effect = SideEffect.WRITE
        tier = _TIER_FOR[side_effect]
        reaches_untrusted = open_world

        if not trusted:
            if side_effect is SideEffect.READ:
                side_effect = SideEffect.WRITE
            tier = max(tier, RiskTier.HIGH)
            reaches_untrusted = True

        tags = set(base_tags)
        if idempotent:
            tags.add("idempotent")
        out.append(
            ToolManifest(
                name=prefix.format(server=server, tool=tool_name),
                description=str(tool.get("description") or ""),
                input_schema=dict(tool.get("inputSchema") or {}),
                side_effect=side_effect,
                risk_tier=tier,
                required_scopes=scopes,
                reaches_untrusted=reaches_untrusted,
                tags=frozenset(tags),
            )
        )
    return out


def mcp_default(
    server: str, *, required_scopes: Iterable[str] | None = None, prefix: str = DEFAULT_PREFIX
) -> ToolManifest:
    """Conservative template for tools of ``server`` that discovery has not seen.

    Use as ``glob_overlay({m.name: m})`` below the discovered manifests so a tool
    that appears after ``tools/list`` ran is still gated by the server's scope.
    """
    return ToolManifest(
        name=prefix.format(server=server, tool="*"),
        side_effect=SideEffect.WRITE,
        risk_tier=RiskTier.HIGH,
        required_scopes=frozenset({f"mcp:{server}"}) if required_scopes is None else frozenset(required_scopes),
        reaches_untrusted=True,
        tags=frozenset({"mcp", f"mcp:{server}", "untrusted"}),
    )
