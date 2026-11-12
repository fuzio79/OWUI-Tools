"""Tests for `fetch_engine` as a per-call tool parameter.

The engine used to be a valve, and valves are operator-only, so the model could
not choose. That is backwards for this particular setting: the model is the one
that knows whether the URL it wants is a static document or a rendered app.

Two things are pinned:

* the parameter reaches the pipeline and actually changes behaviour;
* it does not leak between calls. `run()` applies the override by mutating
  `self.v`, so a bug there would make one call's choice sticky for the next --
  a cross-request state leak of exactly the kind that is invisible in a
  single-call test.
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


def _declared_params() -> list[str]:
    """Parameter names OWUI will show the model, read from the signature."""
    import inspect

    sig = inspect.signature(_research.Tools.research)
    return [p.name for p in sig.parameters.values() if not p.name.startswith("__")]


class _Recorder:
    """Stands in for _Pipeline, recording the engine it was asked to run."""

    calls: list = []

    def __init__(self, valves):
        self.v = valves

    def run(self, query, depth, fetch_engine=""):
        _Recorder.calls.append((query, depth, fetch_engine, self.v.fetch_engine))
        return {"status": "ok", "source": "web", "context": "", "results": []}


@pytest.fixture
def recorder(monkeypatch):
    _Recorder.calls = []
    monkeypatch.setattr(_research, "_Pipeline", _Recorder)
    return _Recorder


# ---------------------------------------------------------------------------
# the model can actually see and use it
# ---------------------------------------------------------------------------


def test_fetch_engine_is_a_declared_tool_parameter():
    """The bug: a valve is invisible to the model, so `direct` was unreachable."""
    assert "fetch_engine" in _declared_params()


def test_parameter_defaults_to_empty_meaning_use_the_valve():
    """Empty keeps the operator's configured default authoritative."""
    import inspect

    sig = inspect.signature(_research.Tools.research)
    assert sig.parameters["fetch_engine"].default == ""


def test_docstring_explains_when_to_choose_each_engine():
    """The docstring is the schema; a bare param name teaches the model nothing."""
    doc = _research.Tools.research.__doc__ or ""
    assert "direct" in doc
    assert "crawl4ai" in doc
    # And it must tell the model what to do when direct refuses a page.
    assert "thin_pages" in doc


# ---------------------------------------------------------------------------
# the parameter reaches the pipeline
# ---------------------------------------------------------------------------


def test_direct_is_passed_through_to_the_pipeline(recorder):
    asyncio.run(_research.Tools().research("q", "deep", "direct"))

    _, depth, engine, _ = recorder.calls[0]
    assert depth == "deep"
    assert engine == "direct"


def test_empty_engine_is_passed_as_empty_not_as_a_default(recorder):
    """The pipeline distinguishes "not asked" from "asked for crawl4ai"."""
    asyncio.run(_research.Tools().research("q", "deep"))

    _, _, engine, _ = recorder.calls[0]
    assert engine == ""


def test_engine_is_normalised(recorder):
    asyncio.run(_research.Tools().research("q", "deep", "  DIRECT  "))

    _, _, engine, _ = recorder.calls[0]
    assert engine == "direct"


# ---------------------------------------------------------------------------
# bad input is rejected, not silently ignored
# ---------------------------------------------------------------------------


def test_unknown_engine_is_rejected(recorder):
    """Silently substituting crawl4ai would hide the mistake forever."""
    result = asyncio.run(_research.Tools().research("q", "deep", "headless"))

    assert result["status"] == "error"
    assert "headless" in result["detail"]
    # And it never ran the pipeline.
    assert recorder.calls == []


def test_rejection_message_lists_the_valid_engines(recorder):
    result = asyncio.run(_research.Tools().research("q", "deep", "nonsense"))
    for engine in _research._FETCH_ENGINES:
        assert engine in result["detail"]


# ---------------------------------------------------------------------------
# no cross-call leak
# ---------------------------------------------------------------------------


def test_override_applies_to_the_pipeline_valves(monkeypatch):
    """`run()` must actually change the engine the fetch will use.

    Asserts on `_fetch_engine()`, the single place the decision is made, NOT on
    `p.v.fetch_engine`: writing to the valves was the original bug, so a test
    that checked the valves would have demanded the defect stay.
    """
    p = _research._Pipeline(_research.Tools.Valves())
    assert p._fetch_engine() == "crawl4ai"

    seen = {}

    def fake_inner(query, depth, deadline):
        seen["engine"] = p._fetch_engine()
        return {"status": "ok"}

    monkeypatch.setattr(p, "_run_inner", fake_inner)
    p.run("q", "deep", "direct")

    assert seen["engine"] == "direct"
    # And the shared config object is untouched.
    assert p.v.fetch_engine == "crawl4ai"


def test_empty_override_leaves_the_valve_alone(monkeypatch):
    """Not asking must not silently reset an operator's configured engine."""
    p = _research._Pipeline(_research.Tools.Valves())
    p.v.fetch_engine = "direct"

    seen = {}
    monkeypatch.setattr(
        p, "_run_inner", lambda q, d, dl: seen.setdefault("e", p._fetch_engine())
    )
    p.run("q", "deep", "")

    assert seen["e"] == "direct", "an empty override must mean 'use the valve'"


def test_override_does_not_leak_into_a_fresh_pipeline():
    """One `_Pipeline` per call is what makes mutating `self.v` safe.

    If that ever changed to a shared instance, this is the test that would say
    so -- a sticky engine is invisible when you only ever make one call.
    """
    first = _research._Pipeline(_research.Tools.Valves())
    first.v.fetch_engine = "direct"

    second = _research._Pipeline(_research.Tools.Valves())
    assert second.v.fetch_engine == "crawl4ai", (
        "a new pipeline must start from the configured default, not the "
        "previous call's choice"
    )


def test_two_tools_instances_do_not_share_valves():
    """`Tools()` builds its own Valves; sharing them would leak the override."""
    a = _research.Tools()
    b = _research.Tools()
    a.valves.fetch_engine = "direct"
    assert b.valves.fetch_engine == "crawl4ai"


def test_unknown_engine_does_not_mutate_the_valves(recorder):
    """A rejected call must leave the pipeline config untouched.

    Drives the real `research()` so a future code path that applies the
    override before validating would be caught.
    """
    before = _research.Tools.Valves().fetch_engine
    asyncio.run(_research.Tools().research("q", "deep", "garbage"))

    assert recorder.calls == []
    assert _research.Tools.Valves().fetch_engine == before
