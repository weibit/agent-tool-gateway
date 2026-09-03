"""Discovery: turn a source's own description of its tools into manifests.

Discovery modules do not connect to anything. You hand them the data a source
already gave you (an MCP ``tools/list`` result, an OpenAPI document, ...) and get
``ToolManifest`` objects back to feed a ``ToolRegistry`` — normally through
``lookup(...)`` placed *below* an operator ``glob_overlay`` so humans keep the
last word over what a source claims about itself.
"""

from .mcp import DEFAULT_PREFIX, manifests_from_mcp, mcp_default

__all__ = ["DEFAULT_PREFIX", "manifests_from_mcp", "mcp_default"]
