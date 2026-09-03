"""The shipped example is copied by users, so its policy predicates get tests."""

from __future__ import annotations

import importlib.util
import os
import pathlib

import pytest

EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "examples" / "claude_sdk_coding_agent.py"


@pytest.fixture(scope="module")
def ex():
    spec = importlib.util.spec_from_file_location("example_coding_agent", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class Ctx:
    def __init__(self, **args):
        self.args = args


def test_inside_workspace_rejects_siblings_and_empty_paths(ex):
    ws = ex.WORKSPACE
    assert ex.inside_workspace(Ctx(file_path=os.path.join(ws, "README.md")))
    assert ex.inside_workspace(Ctx(file_path=os.path.join(ws, "sub", "..", "a.py")))
    assert not ex.inside_workspace(Ctx(file_path=ws + "-evil/x.py"))
    assert not ex.inside_workspace(Ctx(file_path=os.path.join(ws, "..", ".ssh", "id_rsa")))
    assert not ex.inside_workspace(Ctx(file_path=""))
    assert not ex.inside_workspace(Ctx())


@pytest.mark.parametrize(
    "cmd",
    ["ls -la", "cat README.md", "grep -rn foo src", "/bin/ls", "head -n 5 x.py"],
)
def test_bash_readonly_allowlist(ex, cmd):
    assert ex.bash_is_readonly(Ctx(command=cmd))


@pytest.mark.parametrize(
    "cmd",
    [
        "ls && rm -r -f /",
        "ls; rm -rf /",
        "cat x | sh",
        "cat $(rm -rf /)",
        "cat `whoami`",
        "ls > /etc/passwd",
        "git reset --hard HEAD~5",
        "npm install evil-pkg",
        "find / -delete",
        "python -c 'import shutil;shutil.rmtree(\"/\")'",
        "make deploy",
        "ls 'unterminated",
        "",
    ],
)
def test_bash_readonly_rejects_mutating_or_chained(ex, cmd):
    assert not ex.bash_is_readonly(Ctx(command=cmd))
