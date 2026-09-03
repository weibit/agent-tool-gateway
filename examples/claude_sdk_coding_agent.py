"""Gate a coding agent's Bash / Read / Write / Edit / WebFetch tools.

Run directly to see the hook decisions without the SDK:

    python examples/claude_sdk_coding_agent.py

With ``claude-agent-sdk`` installed, ``hooks`` plugs straight into
``ClaudeAgentOptions(hooks=...)``. Pair it with a ``can_use_tool`` callback so
that ``ask`` decisions have something to answer them.
"""

from __future__ import annotations

import asyncio
import os
import re
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

WORKSPACE = os.path.realpath(os.getcwd())

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
    """True only for paths that resolve (symlinks included) to inside WORKSPACE."""
    path = ctx.args.get("file_path") or ctx.args.get("path")
    if not path:
        return False
    real = os.path.realpath(path)
    return os.path.commonpath([WORKSPACE, real]) == WORKSPACE


# Coarse denylist for obviously destructive text. It is easy to evade on its own;
# the allowlist in ``bash_is_readonly`` is the real control, and everything that is
# neither denied nor allowlisted falls through to ``require_approval``.
DANGEROUS = ("rm -rf", "git push --force", "curl ", "wget ", "sudo ", "chmod 777", "> /dev/")

# Commands that only read. No interpreters, build tools or VCS: those can do anything.
READONLY_COMMANDS = {"ls", "cat", "head", "tail", "wc", "pwd", "grep", "rg", "which", "stat", "du", "tree"}
SHELL_META = re.compile(r"[;&|<>`$\n]")  # chaining, pipes, redirection, substitution


def bash_is_dangerous(ctx) -> bool:
    cmd = ctx.args.get("command", "")
    return any(tok in cmd for tok in DANGEROUS)


def bash_is_readonly(ctx) -> bool:
    cmd = ctx.args.get("command", "")
    if not cmd.strip() or SHELL_META.search(cmd) or bash_is_dangerous(ctx):
        return False
    try:
        argv = shlex.split(cmd)
    except ValueError:  # unbalanced quotes
        return False
    return bool(argv) and os.path.basename(argv[0]) in READONLY_COMMANDS


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
hooks = adapter.hooks()  # -> ClaudeAgentOptions(hooks=hooks, can_use_tool=...)


# ---- 4. demo without the SDK --------------------------------------------------
async def demo() -> None:
    calls = [
        ("Read", {"file_path": f"{WORKSPACE}/README.md"}),  # allow
        ("Write", {"file_path": "/etc/hosts", "content": "x"}),  # deny: outside workspace
        ("Write", {"file_path": f"{WORKSPACE}-evil/x", "content": "x"}),  # deny: sibling dir
        ("Bash", {"command": "ls -la"}),  # allow: read-only
        ("Bash", {"command": "pytest -q"}),  # ask: not allowlisted
        ("Bash", {"command": "ls && rm -r -f /"}),  # ask: chained, not allowlisted
        ("Bash", {"command": "rm -rf /"}),  # deny
        ("WebFetch", {"url": "https://example.com"}),  # allow, taints the session
        ("Write", {"file_path": f"{WORKSPACE}/notes.md", "content": "hi"}),  # ask: tainted
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
