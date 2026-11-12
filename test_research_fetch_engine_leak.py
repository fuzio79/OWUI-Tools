"""Regression test for the fetch_engine override leaking between calls.

The bug, measured against the live tool:

    call-a (no override)  -> crawl4ai
    call-b (direct)       -> direct
    call-c (no override)  -> direct    <- wrong

Cause: OWUI reuses ONE `Tools` instance for the life of the process, so
`self.valves` is shared by every call, and `_Pipeline` receives that very object
(`pipeline.v is tools.valves`). `run()` was writing the override into it, so a
single call permanently changed the operator's configuration.

The PR that introduced the feature shipped a test for "does not leak" that built
a fresh `Tools()` per call. That test could never fail: it created a new valves
object every time, so there was nothing to share. **Reusing one instance is the
whole point**, and it is what these tests do.
"""

from __future__ import annotations

import asyncio
import importlib.util
import pathlib

import pytest

_PATH = pathlib.Path(__file__).parent / "research-openwebui-tool.py"
_spec = importlib.util.spec_from_file_location("research_tool", _PATH)
_research = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_research)


@pytest.fixture
def one_tool(monkeypatch):
    """One `Tools` instance reused across calls, exactly as OWUI does.

    `_run_inner` is stubbed so nothing hits the network; everything else --
    `research()`, `_Pipeline.__init__`, the override resolution -- is real.
    """
    seen: list[tuple[str, str]] = []

    def fake_inner(self, query, depth, deadline):
        seen.append((query, self._fetch_engine()))
        return {"status": "ok", "source": "stub", "context": ""}

    monkeypatch.setattr(_research._Pipeline, "_run_inner", fake_inner)
    return _research.Tools(), seen


def _run(tool, query, engine=None):
    kwargs = {} if engine is None else {"fetch_engine": engine}
    return asyncio.run(tool.research(query, "deep", **kwargs))


# ---------------------------------------------------------------------------
# the regression
# ---------------------------------------------------------------------------


def test_override_does_not_stick_to_the_next_call(one_tool):
    """The exact live sequence that exposed this."""
    tool, seen = one_tool

    _run(tool, "call-a")
    _run(tool, "call-b", "direct")
    _run(tool, "call-c")

    assert [e for _, e in seen] == ["crawl4ai", "direct", "crawl4ai"]


def test_the_valves_object_is_never_mutated(one_tool):
    """`self.v` is the caller's own object; writing to it is the bug itself."""
    tool, _ = one_tool
    before = tool.valves.fetch_engine

    _run(tool, "q", "direct")
    _run(tool, "q", "direct")

    assert tool.valves.fetch_engine == before
    assert tool.valves.fetch_engine == "crawl4ai"


def test_the_pipeline_shares_the_tools_valves(one_tool):
    """States the precondition: without this, the leak is impossible.

    If a future refactor copies the valves instead of sharing them, the leak
    test above stops being meaningful and this one says so.
    """
    tool, _ = one_tool
    pipeline = _research._Pipeline(tool.valves)
    assert pipeline.v is tool.valves


def test_override_reverts_on_the_very_next_call(one_tool):
    tool, seen = one_tool

    _run(tool, "a", "direct")
    _run(tool, "b")
    _run(tool, "c", "direct")
    _run(tool, "d")

    assert [e for _, e in seen] == ["direct", "crawl4ai", "direct", "crawl4ai"]


# ---------------------------------------------------------------------------
# the configured default still wins when no override is given
# ---------------------------------------------------------------------------


def test_operator_default_is_honoured_without_an_override(one_tool):
    """An operator who sets the valve to `direct` gets `direct`."""
    tool, seen = one_tool
    tool.valves.fetch_engine = "direct"

    _run(tool, "q")

    assert seen[0][1] == "direct"


def test_override_wins_over_the_operator_default(one_tool):
    """But a per-call choice still beats it, for that call only."""
    tool, seen = one_tool
    tool.valves.fetch_engine = "direct"

    _run(tool, "q", "crawl4ai")
    _run(tool, "q")

    assert [e for _, e in seen] == ["crawl4ai", "direct"]


def test_unknown_engine_leaves_the_default_alone(one_tool):
    tool, seen = one_tool
    result = _run(tool, "q", "headless")

    assert result["status"] == "error"
    assert seen == []
    assert tool.valves.fetch_engine == "crawl4ai"
