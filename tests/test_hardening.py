"""Regression tests for the audit findings: error mapping, timeouts, contextvars,
dry-run, guards, budget reservation, approvals by id, and the Claude SDK adapter."""

from __future__ import annotations

import asyncio
import time

import pytest

from agent_tool_gateway import (
    AgentIdentity,
    AuthorizationDenied,
    BudgetExceeded,
    Decision,
    Gateway,
    GuardrailViolation,
    InMemoryAuditSink,
    Principal,
    SchemaValidationError,
    SessionState,
    SideEffect,
    ToolExecutionError,
    ToolManifest,
    ToolRegistry,
    ToolResult,
    ToolTimeout,
)
from agent_tool_gateway.adapters.claude_agent_sdk import ClaudeAgentSDKAdapter
from agent_tool_gateway.adapters.wrap import bind, gw_wrap
from agent_tool_gateway.pipeline import BaseStage
from agent_tool_gateway.stages import (
    MaxOutputChars,
    RulePolicy,
    TokenBucketLimiter,
    default_stages,
    grant_approval_by_id,
    no_secrets_in_args,
    redact_pii,
)
from agent_tool_gateway.stages.authz import _validate_fallback

READ_FILE = ToolManifest(
    "read_file",
    input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    required_scopes=frozenset({"fs:read"}),
)
SLOW = ToolManifest("slow", timeout_s=0.1)
PAY = ToolManifest("pay", cost_usd=0.01)
SCOPES = frozenset({"fs:read"})


def policy() -> RulePolicy:
    return (
        RulePolicy()
        .allow("read_file")
        .allow("slow")
        .allow("pay")
        .rewrite(
            "read_file",
            transform=lambda c: {**c.args, "path": "ws/" + c.args["path"]},
            when=lambda c: not c.args.get("path", "").startswith("ws/"),
            priority=5,
        )
    )


def make(stages=None, **kw):
    audit = InMemoryAuditSink()
    reg = ToolRegistry([READ_FILE, SLOW, PAY])
    st = stages or default_stages(policy(), limiter=TokenBucketLimiter(rate_per_s=100, burst=100))
    return Gateway(reg, st, audit=audit, **kw), audit


def ids():
    return Principal("u", scopes=SCOPES), AgentIdentity("ag", scopes=SCOPES), SessionState()


class Ctx:
    """Minimal stand-in for ToolCallContext in guard unit tests."""

    def __init__(self, args):
        self.args = args


# ------------------------------------------------------------ error mapping


async def test_deny_error_codes_map_to_typed_errors():
    gw, _ = make()
    p, a, s = ids()
    with pytest.raises(SchemaValidationError) as ei:
        await gw.call(gw.build_context("read_file", {}, principal=p, agent=a, session=s), lambda path: "")
    assert ei.value.retryable is True
    assert ei.value.to_model_result()["error"] == "invalid_arguments"

    with pytest.raises(GuardrailViolation):
        await gw.call(
            gw.build_context("read_file", {"path": "ws/x", "extra": "api_key=ABCDEF"}, principal=p, agent=a, session=s),
            lambda **kw: "",
        )

    s.budget_limit_usd = 0.0
    with pytest.raises(BudgetExceeded):
        await gw.call(gw.build_context("pay", {}, principal=p, agent=a, session=s), lambda: "")

    with pytest.raises(AuthorizationDenied):
        p2 = Principal("u", scopes=frozenset())
        await gw.call(gw.build_context("read_file", {"path": "ws/x"}, principal=p2, agent=a, session=s), lambda path: "")


# ------------------------------------------------------------- timeouts


async def test_sync_tool_is_timed_out():
    gw, audit = make()
    p, a, s = ids()
    ctx = gw.build_context("slow", {}, principal=p, agent=a, session=s)

    def blocking():
        time.sleep(0.5)
        return "done"

    start = time.perf_counter()
    with pytest.raises(ToolTimeout):
        await gw.call(ctx, blocking)
    assert time.perf_counter() - start < 0.4
    assert audit.events[-1].error_code == "tool_timeout"


# ---------------------------------------------------------- wrap adapter


async def test_sync_wrapper_inside_running_loop_keeps_identity():
    gw, _ = make()
    p, a, s = ids()

    @gw_wrap(gw, "read_file")
    def read_file(path: str) -> str:
        return f"read {path}"

    with bind(principal=p, agent=a, session=s):
        assert read_file("ws/a") == "read ws/a"


# -------------------------------------------------------------- dry run


async def test_dry_run_does_not_apply_transform():
    gw, audit = make(dry_run=True)
    p, a, s = ids()
    ctx = gw.build_context("read_file", {"path": "notes.txt"}, principal=p, agent=a, session=s)
    d = await gw.before(ctx)
    assert d.decision is Decision.ALLOW
    assert d.details["shadow_decision"] == "transform"
    assert ctx.args["path"] == "notes.txt"
    assert audit.events[-1].phase == "dry_run"


# ----------------------------------------------------------- fail closed


class Boom(BaseStage):
    name = "boom"

    async def before(self, ctx):
        raise RuntimeError("internal host db-7 down")


async def test_stage_bug_fails_closed_and_audits_detail():
    gw, audit = make(stages=[Boom()])
    p, a, s = ids()
    d = await gw.before(gw.build_context("read_file", {"path": "x"}, principal=p, agent=a, session=s))
    assert d.decision is Decision.DENY and d.stage == "boom"
    assert "db-7" not in d.reason
    assert "db-7" in (audit.events[-1].error_detail or "")
    assert "_detail" not in audit.events[-1].extra


async def test_stage_bug_fails_open_when_configured():
    gw, _ = make(stages=[Boom()], fail_closed=False)
    p, a, s = ids()
    d = await gw.before(gw.build_context("read_file", {"path": "x"}, principal=p, agent=a, session=s))
    assert d.decision is Decision.ALLOW


class RaisesGatewayError(BaseStage):
    name = "raiser"

    async def before(self, ctx):
        raise GuardrailViolation("Blocked by guard.", audit_detail="matched rule R-42 on arg 'q'")


async def test_stage_gateway_error_keeps_both_channels():
    gw, audit = make(stages=[RaisesGatewayError()])
    p, a, s = ids()
    ctx = gw.build_context("read_file", {"path": "x"}, principal=p, agent=a, session=s)
    d = await gw.before(ctx)
    assert d.decision is Decision.DENY and d.details["error"] == "guardrail_violation"
    assert audit.events[-1].error_code == "guardrail_violation"
    assert "R-42" in audit.events[-1].error_detail
    with pytest.raises(GuardrailViolation):
        gw.raise_for_decision(d)


# ---------------------------------------------------------------- guards


@pytest.mark.parametrize(
    "args",
    [
        {"api_key": "ABCDEF1234567890XYZ"},
        {"password": "hunter2"},
        {"headers": {"Authorization": "Bearer abcdefghijklmnopqrstuvwxyz"}},
        {"token": "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789"},
        {"cmd": "export API_KEY=abc"},
        {"key": "AKIAABCDEFGHIJKLMNOP"},
        {"nested": [{"client_secret": "s3cr3t"}]},
    ],
)
def test_secret_guard_catches_common_shapes(args):
    assert no_secrets_in_args(Ctx(args)) is not None


@pytest.mark.parametrize(
    "args",
    [
        {"path": "README.md"},
        {"secret": ""},
        {"query": "how do I rotate an api key"},
        {"page_token": "abc123"},
    ],
)
def test_secret_guard_allows_benign(args):
    assert no_secrets_in_args(Ctx(args)) is None


def test_pii_redaction_uses_luhn_and_walks_structures():
    r = redact_pii(None, ToolResult(content="ts=1725312000000 card=4111 1111 1111 1111 ssn=123-45-6789"))
    assert r.content == "ts=1725312000000 card=[REDACTED] ssn=[REDACTED]"
    assert r.metadata.get("pii_redacted") is True

    r = redact_pii(None, ToolResult(content={"stdout": "ssn 123-45-6789", "items": ["4111111111111111", 7]}))
    assert r.content == {"stdout": "ssn [REDACTED]", "items": ["[REDACTED]", 7]}

    r = redact_pii(None, ToolResult(content="order 12345678901234"))
    assert r.content == "order 12345678901234" and "pii_redacted" not in r.metadata


def test_max_output_chars_walks_structures():
    cap = MaxOutputChars(limit=5)
    r = cap(None, ToolResult(content={"stdout": "x" * 20, "stderr": ""}))
    assert r.truncated and r.content["stdout"].startswith("xxxxx\n...[truncated 15 chars]")
    assert r.content["stderr"] == ""


def test_fallback_validator():
    schema = {
        "type": "object",
        "properties": {"n": {"type": "integer"}, "s": {"type": "string"}},
        "required": ["n"],
        "additionalProperties": False,
    }
    assert _validate_fallback(schema, {"n": 1}) is None
    assert "missing" in _validate_fallback(schema, {})
    assert "integer" in _validate_fallback(schema, {"n": True})
    assert "unexpected" in _validate_fallback(schema, {"n": 1, "z": 2})


def test_policy_glob_is_case_sensitive():
    gw, _ = make(stages=default_stages(RulePolicy().allow("READ_FILE")))
    p, a, s = ids()
    ctx = gw.build_context("read_file", {"path": "x"}, principal=p, agent=a, session=s)
    assert asyncio.run(gw.before(ctx)).decision is Decision.DENY


# ------------------------------------------------------------- budget


async def test_budget_is_reserved_across_concurrent_calls():
    gw, _ = make()
    p, a, _ = ids()
    s = SessionState(budget_limit_usd=0.01)

    async def tool():
        await asyncio.sleep(0.01)
        return "ok"

    async def one(i):
        try:
            await gw.call(gw.build_context("pay", {}, principal=p, agent=a, session=s), tool)
            return "ran"
        except BudgetExceeded:
            return "denied"

    out = await asyncio.gather(*(one(i) for i in range(3)))
    assert sorted(out) == ["denied", "denied", "ran"]
    assert s.budget_used_usd == pytest.approx(0.01) and s.budget_reserved_usd == 0.0


async def test_budget_reservation_released_on_tool_failure():
    gw, _ = make()
    p, a, _ = ids()
    s = SessionState(budget_limit_usd=0.01)

    def boom():
        raise RuntimeError("x")

    with pytest.raises(ToolExecutionError):
        await gw.call(gw.build_context("pay", {}, principal=p, agent=a, session=s), boom)
    assert s.budget_reserved_usd == 0.0 and s.budget_used_usd == 0.0
    await gw.call(gw.build_context("pay", {}, principal=p, agent=a, session=s), lambda: "ok")


# ---------------------------------------------------------- approvals


async def test_grant_approval_by_id():
    pol = RulePolicy().require_approval("read_file", reason="ask first")
    gw, _ = make(stages=default_stages(pol))
    p, a, s = ids()
    ctx = gw.build_context("read_file", {"path": "x"}, principal=p, agent=a, session=s)
    d = await gw.before(ctx)
    assert d.decision is Decision.REQUIRE_APPROVAL
    assert grant_approval_by_id(s, "nope") is None
    assert grant_approval_by_id(s, d.approval_id) == ctx.approval_key
    assert (await gw.before(ctx)).decision is Decision.ALLOW
    other = gw.build_context("read_file", {"path": "y"}, principal=p, agent=a, session=s)
    assert (await gw.before(other)).decision is Decision.REQUIRE_APPROVAL


async def test_before_approved_runs_later_stages_and_leaves_no_approval_behind():
    pol = RulePolicy().require_approval("pay", reason="money")
    gw, _ = make(stages=default_stages(pol))
    p, a, _ = ids()
    s = SessionState(budget_limit_usd=0.05)
    ctx = gw.build_context("pay", {}, principal=p, agent=a, session=s)
    assert (await gw.before(ctx)).decision is Decision.REQUIRE_APPROVAL

    d = await gw.before_approved(ctx)  # host approved: policy passes, budget stage now runs and reserves
    assert d.decision is Decision.ALLOW and s.budget_reserved_usd == pytest.approx(0.01)
    assert s.approvals == set()  # one-shot: nothing granted beyond this call
    gw.release(ctx)

    s.budget_used_usd = 0.05
    d = await gw.before_approved(ctx)
    assert d.decision is Decision.DENY and d.stage == "budget"
    assert (await gw.before(ctx)).decision is Decision.REQUIRE_APPROVAL  # repeat still asks


# ---------------------------------------------------- claude sdk adapter


def adapter(gw, s):
    p, a, _ = ids()
    return ClaudeAgentSDKAdapter(gw, identity=lambda _: (p, a, s))


async def test_adapter_does_not_retain_denied_calls_and_is_bounded():
    gw, _ = make()
    ad = adapter(gw, SessionState())
    ad.max_inflight = 3
    for i in range(5):
        await ad.pre_tool_use({"tool_name": "read_file", "tool_input": {}}, f"deny-{i}")
    assert len(ad._inflight) == 0
    for i in range(5):
        await ad.pre_tool_use({"tool_name": "read_file", "tool_input": {"path": f"ws/{i}"}}, f"ok-{i}")
    assert list(ad._inflight) == ["ok-2", "ok-3", "ok-4"]


async def test_adapter_rewrites_structured_tool_output():
    gw, _ = make()
    ad = adapter(gw, SessionState())
    await ad.pre_tool_use({"tool_name": "read_file", "tool_input": {"path": "ws/a"}}, "t1")
    raw = {"stdout": "ssn 123-45-6789", "stderr": "", "interrupted": False}
    out = await ad.post_tool_use({"tool_name": "read_file", "tool_input": {"path": "ws/a"}, "tool_response": raw}, "t1")
    assert out["hookSpecificOutput"]["updatedToolOutput"] == {"stdout": "ssn [REDACTED]", "stderr": "", "interrupted": False}
    assert raw["stdout"] == "ssn 123-45-6789"  # input not mutated
    assert "t1" not in ad._inflight

    await ad.pre_tool_use({"tool_name": "read_file", "tool_input": {"path": "ws/b"}}, "t2")
    out = await ad.post_tool_use({"tool_name": "read_file", "tool_input": {"path": "ws/b"}, "tool_response": "clean"}, "t2")
    assert "updatedToolOutput" not in out.get("hookSpecificOutput", {})


async def test_adapter_failure_hook_releases_budget_and_audits():
    gw, audit = make()
    s = SessionState(budget_limit_usd=0.01)
    ad = adapter(gw, s)
    hooks = ad.hooks()
    assert set(hooks) == {"PreToolUse", "PostToolUse", "PostToolUseFailure"}
    await ad.pre_tool_use({"tool_name": "pay", "tool_input": {}}, "t1")
    assert s.budget_reserved_usd == pytest.approx(0.01)
    await ad.post_tool_use_failure({"tool_name": "pay", "tool_input": {}, "error": "boom"}, "t1")
    assert s.budget_reserved_usd == 0.0 and s.budget_used_usd == 0.0
    assert audit.events[-1].error_code == "tool_execution_error" and "boom" in audit.events[-1].error_detail
    assert "t1" not in ad._inflight


async def test_adapter_records_call_when_ask_is_approved_by_host():
    pol = RulePolicy().require_approval("read_file")
    gw, _ = make(stages=default_stages(pol))
    s = SessionState()
    ad = adapter(gw, s)
    out = await ad.pre_tool_use({"tool_name": "read_file", "tool_input": {"path": "x"}}, "t1")
    assert out["hookSpecificOutput"]["permissionDecision"] == "ask"
    assert len(s.recent_calls) == 0
    await ad.post_tool_use({"tool_name": "read_file", "tool_input": {"path": "x"}, "tool_response": "ok"}, "t1")
    assert len(s.recent_calls) == 1


def test_manifest_side_effect_enum_roundtrip():
    assert SideEffect("write") is SideEffect.WRITE
