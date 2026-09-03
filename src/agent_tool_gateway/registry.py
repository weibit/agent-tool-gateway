"""Tool registry: name -> manifest, resolved through a chain of layers.

Resolution order, first hit wins:

    1. explicit manifests registered by hand          (highest trust)
    2. resolvers, in order — typically an operator overlay keyed by glob,
       then manifests produced by discovery, then per-namespace defaults
    3. the global ``default`` manifest
    4. ``ToolNotRegistered``

Discovered manifests are self-declarations by the tool's source, so they sit
below the operator overlay: a human can tighten or relax them without editing
the discovered data. Swap a whole registry (``gateway.registry = new``) rather
than mutating one in place when reloading; contexts already built keep the
manifest they were built with.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

from .errors import ToolNotRegistered
from .manifest import ToolManifest

Resolver = Callable[[str], ToolManifest | None]
"""Return the manifest for a tool name, or None to fall through to the next layer."""


class ToolRegistry:
    def __init__(
        self,
        manifests: Iterable[ToolManifest] = (),
        *,
        resolvers: Iterable[Resolver] = (),
        default: ToolManifest | None = None,
    ) -> None:
        self._tools: dict[str, ToolManifest] = {}
        self._resolvers: list[Resolver] = list(resolvers)
        self._default = default
        for m in manifests:
            self.register(m)

    def register(self, manifest: ToolManifest, *, replace: bool = False) -> None:
        if manifest.name in self._tools and not replace:
            raise ValueError(f"tool already registered: {manifest.name}")
        self._tools[manifest.name] = manifest

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def add_resolver(self, resolver: Resolver, *, first: bool = False) -> None:
        if first:
            self._resolvers.insert(0, resolver)
        else:
            self._resolvers.append(resolver)

    def resolve(self, name: str) -> ToolManifest:
        if name in self._tools:
            return self._tools[name]
        for resolver in self._resolvers:
            m = resolver(name)
            if m is not None:
                return m if m.name == name else m.with_overrides(name=name)
        if self._default is not None:
            return self._default.with_overrides(name=name)
        raise ToolNotRegistered(
            model_message=f"Tool '{name}' is not available.",
            audit_detail=f"unregistered tool requested: {name}",
        )

    def match(self, pattern: str) -> list[ToolManifest]:
        """Explicitly registered manifests whose name matches ``pattern`` (case-sensitive glob)."""
        return [m for n, m in self._tools.items() if fnmatch.fnmatchcase(n, pattern)]

    def __contains__(self, name: object) -> bool:
        """True only for explicitly registered names; use ``resolve`` to consult the chain."""
        return name in self._tools

    def __iter__(self) -> Iterator[ToolManifest]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)


# ---------------------------------------------------------------- resolvers


def lookup(manifests: Iterable[ToolManifest]) -> Resolver:
    """Exact-name resolver over a fixed set, e.g. the output of a discovery module."""
    table = {m.name: m for m in manifests}

    def resolve(name: str) -> ToolManifest | None:
        return table.get(name)

    return resolve


def glob_overlay(rules: Mapping[str, ToolManifest | Mapping[str, Any]]) -> Resolver:
    """Glob-keyed templates, first match wins, case-sensitive.

    Values are manifests or plain dicts (see ``ToolManifest.from_dict``), so the
    overlay can be loaded from YAML::

        glob_overlay({
            "mcp__github__get_*": {"side_effect": "read", "risk_tier": "low"},
            "mcp__github__*":     {"side_effect": "write", "required_scopes": ["mcp:github"]},
        })
    """
    compiled: list[tuple[str, ToolManifest]] = [
        (pattern, m if isinstance(m, ToolManifest) else ToolManifest.from_dict({"name": pattern, **m}))
        for pattern, m in rules.items()
    ]

    def resolve(name: str) -> ToolManifest | None:
        for pattern, template in compiled:
            if fnmatch.fnmatchcase(name, pattern):
                return template.with_overrides(name=name)
        return None

    return resolve
