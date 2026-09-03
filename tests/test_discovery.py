"""Resolver chain, overlays, and MCP discovery."""

from __future__ import annotations

import pytest

from agent_tool_gateway import (
    AgentIdentity,
    Decision,
    Gateway,
    Principal,
    RiskTier,
    SessionState,
    SideEffect,
    ToolManifest,
    ToolNotRegistered,
    ToolRegistry,
    glob_overlay,
    lookup,
)
from agent_tool_gateway.discovery import manifests_from_mcp, mcp_default
from agent_tool_gateway.stages import RulePolicy, default_stages

# ------------------------------------------------------------ manifest


def test_manifest_from_dict_coerces_types():
    m = ToolManifest.from_dict(
        {
            "name": "x",
            "side_effect": "irreversible",
            "risk_tier": "high",
            "required_scopes": ["a", "b"],
            "tags": ["t"],
            "reaches_untrusted": True,
        }
    )
    assert m.side_effect is SideEffect.IRREVERSIBLE
    assert m.risk_tier is RiskTier.HIGH
    assert m.required_scopes == frozenset({"a", "b"}) and m.tags == frozenset({"t"})
    assert ToolManifest.from_dict({"name": "y", "risk_tier": 2}).risk_tier is RiskTier.MEDIUM
    assert ToolManifest.from_dict({"name": "z", "side_effect": SideEffect.READ}).side_effect is SideEffect.READ
    with pytest.raises(ValueError):
        ToolManifest.from_dict({"name": "bad", "side_effect": "explode"})
    with pytest.raises(TypeError):
        ToolManifest.from_dict({"name": "bad", "nope": 1})


# ------------------------------------------------------------ registry


EXPLICIT = ToolManifest("mcp__gh__get_issue", side_effect=SideEffect.READ, tags=frozenset({"explicit"}))
DISCOVERED = ToolManifest("mcp__gh__create_issue", side_effect=SideEffect.WRITE, tags=frozenset({"discovered"}))
DISCOVERED2 = ToolManifest("mcp__gh__get_issue", side_effect=SideEffect.WRITE, tags=frozenset({"discovered"}))
OVERLAY = glob_overlay({"mcp__gh__create_*": {"side_effect": "irreversible", "tags": ["overlay"]}})


def test_resolution_order_explicit_overlay_discovered_default():
    reg = ToolRegistry(
        [EXPLICIT],
        resolvers=[OVERLAY, lookup([DISCOVERED, DISCOVERED2])],
        default=ToolManifest("*", tags=frozenset({"default"})),
    )
    assert reg.resolve("mcp__gh__get_issue").tags == {"explicit"}  # explicit beats discovered
    m = reg.resolve("mcp__gh__create_issue")
    assert m.tags == {"overlay"} and m.name == "mcp__gh__create_issue"  # overlay beats discovered
    assert m.side_effect is SideEffect.IRREVERSIBLE
    assert reg.resolve("mcp__gh__list_repos").tags == {"default"}  # nothing matched -> default
    assert "mcp__gh__get_issue" in reg and "mcp__gh__create_issue" not in reg  # `in` is explicit only
    assert len(reg) == 1


def test_resolvers_fall_through_and_can_be_prepended():
    reg = ToolRegistry(resolvers=[lookup([DISCOVERED])])
    with pytest.raises(ToolNotRegistered):
        reg.resolve("unknown")
    reg.add_resolver(glob_overlay({"unknown": ToolManifest("unknown", tags=frozenset({"late"}))}))
    assert reg.resolve("unknown").tags == {"late"}
    reg.add_resolver(glob_overlay({"*": ToolManifest("*", tags=frozenset({"first"}))}), first=True)
    assert reg.resolve("unknown").tags == {"first"}
    assert reg.resolve("mcp__gh__create_issue").tags == {"first"}


def test_glob_overlay_is_case_sensitive_and_first_match_wins():
    ov = glob_overlay(
        {
            "Read": ToolManifest("Read", side_effect=SideEffect.READ),
            "Re*": ToolManifest("Re*", side_effect=SideEffect.WRITE),
        }
    )
    assert ov("Read").side_effect is SideEffect.READ
    assert ov("Rename").side_effect is SideEffect.WRITE
    assert ov("read") is None


# ------------------------------------------------------------- mcp


READ_TOOL = {
    "name": "get_issue",
    "description": "Fetch an issue",
    "inputSchema": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]},
    "annotations": {"readOnlyHint": True, "openWorldHint": False, "idempotentHint": True},
}
WRITE_TOOL = {
    "name": "add_label",
    "inputSchema": {"type": "object"},
    "annotations": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False},
}
DESTRUCTIVE_TOOL = {"name": "delete_repo", "annotations": {"destructiveHint": True, "openWorldHint": True}}
BARE_TOOL = {"name": "mystery"}  # no annotations at all -> MCP spec defaults apply


def by_name(ms):
    return {m.name: m for m in ms}


def test_mcp_trusted_mapping_follows_annotations_and_spec_defaults():
    ms = by_name(manifests_from_mcp("gh", [READ_TOOL, WRITE_TOOL, DESTRUCTIVE_TOOL, BARE_TOOL], trusted=True))
    assert set(ms) == {"mcp__gh__get_issue", "mcp__gh__add_label", "mcp__gh__delete_repo", "mcp__gh__mystery"}

    r = ms["mcp__gh__get_issue"]
    assert (r.side_effect, r.risk_tier, r.reaches_untrusted) == (SideEffect.READ, RiskTier.LOW, False)
    assert r.description == "Fetch an issue" and r.input_schema["required"] == ["id"]
    assert r.required_scopes == frozenset({"mcp:gh"})
    assert {"mcp", "mcp:gh", "trusted", "idempotent"} <= r.tags

    w = ms["mcp__gh__add_label"]
    assert (w.side_effect, w.risk_tier, w.reaches_untrusted) == (SideEffect.WRITE, RiskTier.MEDIUM, False)

    d = ms["mcp__gh__delete_repo"]
    assert (d.side_effect, d.risk_tier, d.reaches_untrusted) == (SideEffect.IRREVERSIBLE, RiskTier.HIGH, True)

    # Spec defaults: destructiveHint=true, openWorldHint=true when absent.
    b = ms["mcp__gh__mystery"]
    assert (b.side_effect, b.risk_tier, b.reaches_untrusted) == (SideEffect.IRREVERSIBLE, RiskTier.HIGH, True)
    assert b.input_schema == {}


def test_mcp_untrusted_clamps_self_declared_safety():
    ms = by_name(manifests_from_mcp("gh", [READ_TOOL, WRITE_TOOL, DESTRUCTIVE_TOOL]))  # trusted=False
    r = ms["mcp__gh__get_issue"]
    assert (r.side_effect, r.risk_tier, r.reaches_untrusted) == (SideEffect.WRITE, RiskTier.HIGH, True)
    assert "untrusted" in r.tags and "trusted" not in r.tags
    w = ms["mcp__gh__add_label"]
    assert (w.side_effect, w.risk_tier, w.reaches_untrusted) == (SideEffect.WRITE, RiskTier.HIGH, True)
    d = ms["mcp__gh__delete_repo"]
    assert (d.side_effect, d.risk_tier) == (SideEffect.IRREVERSIBLE, RiskTier.HIGH)


def test_mcp_options_scopes_prefix_and_objects():
    class Tool:  # duck-types the `mcp` SDK's pydantic model
        def model_dump(self):
            return dict(READ_TOOL)

    ms = manifests_from_mcp("gh", [Tool()], trusted=True, required_scopes=frozenset({"gh:read"}), prefix="{server}/{tool}")
    assert ms[0].name == "gh/get_issue" and ms[0].required_scopes == frozenset({"gh:read"})

    with pytest.raises(ValueError):
        manifests_from_mcp("gh", [{"description": "no name"}])


def test_mcp_default_is_a_conservative_namespace_template():
    d = mcp_default("gh")
    assert d.name == "mcp__gh__*"
    assert (d.side_effect, d.risk_tier, d.reaches_untrusted) == (SideEffect.WRITE, RiskTier.HIGH, True)
    assert d.required_scopes == frozenset({"mcp:gh"})
    reg = ToolRegistry(resolvers=[glob_overlay({d.name: d})])
    assert reg.resolve("mcp__gh__anything").required_scopes == frozenset({"mcp:gh"})


# ------------------------------------------------------ end to end


async def test_discovered_tools_are_governed_by_scope_and_overlay():
    discovered = manifests_from_mcp("gh", [READ_TOOL, DESTRUCTIVE_TOOL])  # untrusted server
    operator_overlay = glob_overlay(
        {"mcp__gh__get_issue": {"side_effect": "read", "risk_tier": "low", "required_scopes": ["mcp:gh"]}}
    )
    reg = ToolRegistry(
        resolvers=[operator_overlay, lookup(discovered), glob_overlay({"mcp__gh__*": mcp_default("gh")})],
        default=ToolManifest("*", side_effect=SideEffect.WRITE, risk_tier=RiskTier.HIGH),
    )
    gw = Gateway(reg, default_stages(RulePolicy().allow("mcp__gh__*")))
    agent = AgentIdentity("a", scopes=frozenset({"mcp:gh"}))

    # Principal without the server scope: every tool from that server is denied at the scope stage.
    p = Principal("u", scopes=frozenset())
    d = await gw.before(gw.build_context("mcp__gh__get_issue", {"id": 1}, principal=p, agent=agent, session=SessionState()))
    assert d.decision is Decision.DENY and d.stage == "scope"

    # With the scope, the overlay-relaxed read tool is allowed even in a tainted session...
    p = Principal("u", scopes=frozenset({"mcp:gh"}))
    s = SessionState()
    s.mark_tainted("web")
    d = await gw.before(gw.build_context("mcp__gh__get_issue", {"id": 1}, principal=p, agent=agent, session=s))
    assert d.decision is Decision.ALLOW

    # ...while the clamped destructive tool needs approval from risk scoring alone (IRREVERSIBLE + HIGH = 7).
    d = await gw.before(gw.build_context("mcp__gh__delete_repo", {}, principal=p, agent=agent, session=SessionState()))
    assert d.decision is Decision.REQUIRE_APPROVAL and d.stage == "risk"

    # Unknown tool from that server falls to the namespace default and its scope requirement.
    d = await gw.before(gw.build_context("mcp__gh__new_tool", {}, principal=Principal("u"), agent=agent, session=s))
    assert d.decision is Decision.DENY and d.stage == "scope"


def test_registry_hot_swap_leaves_inflight_context_untouched():
    reg = ToolRegistry([ToolManifest("t", version="1")])
    gw = Gateway(reg, [])
    ctx = gw.build_context("t", {}, principal=Principal("u"), agent=AgentIdentity("a"), session=SessionState())
    gw.registry = ToolRegistry([ToolManifest("t", version="2")])
    assert ctx.tool.version == "1" and gw.registry.resolve("t").version == "2"
