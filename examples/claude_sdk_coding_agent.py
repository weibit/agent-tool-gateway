"""Gate a coding agent's Bash / Read / Write / Edit / WebFetch tools.

Run directly to see the hook decisions without the SDK:

    python examples/claude_sdk_coding_agent.py

With ``claude-agent-sdk`` installed, ``hooks`` plugs straight into
``ClaudeAgentOptions(hooks=...)``.
"""

from __future__ import annotations

import asyncio
import os
import shlex

from agent_tool_gateway import (
    AgentIdentity,
    Gateway,
    JsonlAuditSink,
    Principal,
    RiskTier,
    SessionState,
    SideEffect,
    ToolManifest,
    ToolRegistry,
)
from agent_tool_gateway.adapters.claude_agent_sdk import ClaudeAgentSDKAdapter
from agent_tool_gateway.stages import RulePolicy, default_stages

WORKSPACE = os.path.abspath(os.getcwd())

# ---- 1. manifests for the coding agent's built-in tools ---------------------
registry = ToolRegistry(
    [
        ToolManifest("Read", side_effect=SideEffect.READ, required_scopes=frozenset({"fs:read"})),
        ToolManifest("Glob", side_effect=SideEffect.READ, required_scopes=frozenset({"fs:read"})),
        ToolManifest("Grep", side_effect=SideEffect.READ, required_scopes=frozenset({"fs:read"})),
        ToolManifest(
            "Write", side_effect=SideEffect.WRITE, risk_tier=RiskTier.MEDIUM, required_scopes=frozenset({"fs:write"})
        ),
        ToolManifest(
            "Edit", side_effect=SideEffect.WRITE, risk_tier=RiskTier.MEDIUM, required_scopes=frozenset({"fs:write"})
        ),
        ToolManifest(
            "Bash", side_effect=SideEffect.WRITE, risk_tier=RiskTier.HIGH, required_scopes=frozenset({"shell"})
        ),
        ToolManifest(
            "WebFetch", side_effect=SideEffect.READ, reaches_untrusted=True, required_scopes=frozenset({"net:read"})
        ),
        ToolManifest(
            "WebSearch", side_effect=SideEffect.READ, reaches_untrusted=True, required_scopes=frozenset({"net:read"})
        ),
    ],
    # Unknown tools (e.g. from MCP servers) get a conservative default rather than a hard error.
    default=ToolManifest("*", side_effect=SideEffect.WRITE, risk_tier=RiskTier.HIGH),
)


# ---- 2. argument-level policy -------------------------------------------------
def inside_workspace(ctx) -> bool:
    path = ctx.args.get("file_path") or ctx.args.get("path") or ""
    return os.path.abspath(path).startswith(WORKSPACE)


DANGEROUS = ("rm -rf", "git push --force", "curl ", "wget ", "sudo ", "chmod 777", "> /dev/")


def bash_is_dangerous(ctx) -> bool:
    cmd = ctx.args.get("command", "")
    return any(tok in cmd for tok in DANGEROUS)


def bash_is_readonly(ctx) -> bool:
    cmd = ctx.args.get("command", "")
    first = shlex.split(cmd)[0] if cmd.strip() else ""
    return first in {"ls", "cat", "grep", "find", "pytest", "python", "git", "npm", "make"} and not bash_is_dangerous(
        ctx
    )


policy = (
    RulePolicy()
    .allow("Read")
    .allow("Glob")
    .allow("Grep")
    .allow("Write", when=inside_workspace)
    .allow("Edit", when=inside_workspace)
    .deny(
        "Write",
        when=lambda c: not inside_workspace(c),
        reason="writes outside the workspace are not allowed",
        priority=10,
    )
    .deny(
        "Edit",
        when=lambda c: not inside_workspace(c),
        reason="edits outside the workspace are not allowed",
        priority=10,
    )
    .deny("Bash", when=bash_is_dangerous, reason="destructive or exfiltrating shell command", priority=10)
    .allow("Bash", when=bash_is_readonly)
    .require_approval("Bash", reason="non-allowlisted shell command")
    .allow("WebFetch")
    .allow("WebSearch")
    .require_approval("*", reason="unknown tool")  # everything else (MCP tools) asks
)

# ---- 3. gateway + identity ----------------------------------------------------
gateway = Gateway(registry, default_stages(policy), audit=JsonlAuditSink())

principal = Principal("bit", scopes=frozenset({"fs:read", "fs:write", "shell", "net:read"}))
agent = AgentIdentity(
    "claude-code", scopes=frozenset({"fs:read", "fs:write", "shell", "net:read"}), kind="coding-agent"
)
session = SessionState(budget_limit_usd=5.0)

adapter = ClaudeAgentSDKAdapter(gateway, identity=lambda _hook_input: (principal, agent, session))
hooks = adapter.hooks()  # -> ClaudeAgentOptions(hooks=hooks)


# ---- 4. demo without the SDK --------------------------------------------------
async def demo() -> None:
    calls = [
        ("Read", {"file_path": f"{WORKSPACE}/README.md"}),
        ("Write", {"file_path": "/etc/hosts", "content": "x"}),
        ("Bash", {"command": "pytest -q"}),
        ("Bash", {"command": "rm -rf /"}),
        ("Bash", {"command": "docker compose up"}),
        ("WebFetch", {"url": "https://example.com"}),
        ("Write", {"file_path": f"{WORKSPACE}/notes.md", "content": "hi"}),  # tainted -> ask
    ]
    for i, (tool, args) in enumerate(calls):
        out = await adapter.pre_tool_use({"tool_name": tool, "tool_input": args}, f"call-{i}")
        spec = out["hookSpecificOutput"]
        print(f"{tool:9} {spec['permissionDecision']:5}  {spec['permissionDecisionReason']}")
        if tool == "WebFetch":
            await adapter.post_tool_use(
                {"tool_name": tool, "tool_input": args, "tool_response": "<html>..."}, f"call-{i}"
            )


if __name__ == "__main__":
    asyncio.run(demo())
