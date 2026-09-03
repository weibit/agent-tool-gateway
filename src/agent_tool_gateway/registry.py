"""Tool registry: name -> manifest, with optional default manifest for unknown tools."""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable, Iterator

from .errors import ToolNotRegistered
from .manifest import ToolManifest


class ToolRegistry:
    def __init__(self, manifests: Iterable[ToolManifest] = (), *, default: ToolManifest | None = None) -> None:
        self._tools: dict[str, ToolManifest] = {}
        self._default = default
        for m in manifests:
            self.register(m)

    def register(self, manifest: ToolManifest, *, replace: bool = False) -> None:
        if manifest.name in self._tools and not replace:
            raise ValueError(f"tool already registered: {manifest.name}")
        self._tools[manifest.name] = manifest

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def resolve(self, name: str) -> ToolManifest:
        if name in self._tools:
            return self._tools[name]
        if self._default is not None:
            return self._default.with_overrides(name=name)
        raise ToolNotRegistered(
            model_message=f"Tool '{name}' is not available.",
            audit_detail=f"unregistered tool requested: {name}",
        )

    def match(self, pattern: str) -> list[ToolManifest]:
        return [m for n, m in self._tools.items() if fnmatch.fnmatch(n, pattern)]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[ToolManifest]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)
