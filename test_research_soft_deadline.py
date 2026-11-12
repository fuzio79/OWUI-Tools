"""Tests for the soft-deadline rename and its reporting.

The valve was called `total_timeout` and described as an "Overall deadline…
individual requests are capped to the remaining time". Neither was true:

* with the shipped defaults a run could take 3x longer than the valve says;
* a request is still started when the deadline has already passed, because
  `max(1.0, remaining)` floors at one second.

It was renamed rather than enforced, because enforcing it would silently
truncate runs that currently finish. The rename is only half the fix: a
truncated answer used to be indistinguishable from a complete one, so the
result now says when the deadline stopped it.
"""

from __future__ import annotations

import importlib.util
import pathlib
import time

import pytest

_PATH = pathlib.Path(__file__).parent / "research-openwebui-tool.py"
_spec = importlib.util.spec_from_file_location("research_tool", _PATH)
_research = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_research)


def _pipeline(**valves):
    p = _research._Pipeline(_research.Tools.Valves())
    p.v.use_cache = False
    p.v.persist = False
    for k, v in valves.items():
        setattr(p.v, k, v)
    return p


# ---------------------------------------------------------------------------
# the rename
# ---------------------------------------------------------------------------


def test_the_valve_is_named_soft_deadline():
    assert hasattr(_research.Tools.Valves(), "soft_deadline")


def test_the_old_name_is_gone():
    """Not kept as an alias: two fields meaning one thing is how they drift."""
    assert not hasattr(_research.Tools.Valves(), "total_timeout")


def test_the_description_no_longer_claims_a_hard_bound():
    """The old text was the defect as much as the behaviour."""
    desc = _research.Tools.Valves.model_fields["soft_deadline"].description.lower()
    assert "soft" in desc
    # It must admit the two ways it can be exceeded.
    assert "overrun" in desc or "not a hard bound" in desc
    assert "soft_deadline_reached" in desc


def test_the_pipeline_uses_the_new_valve(monkeypatch):
    """A rename that left the loop reading the old attribute would crash."""
    p = _pipeline(soft_deadline=42)
    seen = {}

    def fake_inner(query, depth, deadline):
        seen["budget"] = deadline - time.monotonic()
        return {"status": "ok"}

    monkeypatch.setattr(p, "_run_inner", fake_inner)
    p.run("q", "deep")

    assert 40 < seen["budget"] <= 42


# ---------------------------------------------------------------------------
# the reporting
# ---------------------------------------------------------------------------


def _drive(p, pages, budget=3):
    p.v.fetch_budget = budget
    p.search = lambda q, t: {
        "status": "ok",
        "count": len(pages),
        "used_engine": "searxng",
        "results": [{"title": u, "url": u, "content": "x"} for u in pages],
    }
    p.fetch_page = lambda url, t: {"status": "ok", "markdown": "body", "engine": "direct"}


def test_a_completed_run_does_not_claim_a_deadline():
    """The flag must be absent, not False, so a clean result stays clean."""
    p = _pipeline()
    _drive(p, ["https://example.com/a"])
    out = p._run_inner("q", "deep", time.monotonic() + 30)

    assert "soft_deadline_reached" not in out
    assert "note_deadline" not in out


def test_hitting_the_deadline_is_reported():
    """Previously a log line only, so truncation looked like completeness."""
    p = _pipeline()
    _drive(p, ["https://example.com/a", "https://example.com/b", "https://example.org/c"])
    # Deadline already passed -> the loop must stop before fetching anything.
    out = p._run_inner("q", "deep", time.monotonic() - 1)

    assert out.get("soft_deadline_reached") is True
    assert "soft_deadline" in out.get("note_deadline", "")


def test_the_note_explains_what_to_change():
    """A flag without a remedy leaves the model unable to act."""
    p = _pipeline()
    _drive(p, ["https://example.com/a"])
    out = p._run_inner("q", "deep", time.monotonic() - 1)

    note = out.get("note_deadline", "").lower()
    assert "incomplete" in note
    assert "soft_deadline" in note


def test_partial_fetch_still_reports_what_it_got():
    """Stopping early must not discard the pages already fetched.

    The deadline has to expire *during* the loop, so the stub fetch burns real
    time. A `now + 0.001` deadline does not work here: two instant stub calls
    both finish before it, so nothing is truncated and the test would assert on
    a situation that never arose.
    """
    p = _pipeline()
    _drive(p, ["https://example.com/a", "https://example.com/b"])

    def slow_fetch(url, t):
        time.sleep(0.02)
        return {"status": "ok", "markdown": "body", "engine": "direct"}

    p.fetch_page = slow_fetch

    # Expires after roughly the first fetch, before the second is started.
    out = p._run_inner("q", "deep", time.monotonic() + 0.01)

    assert out["status"] == "ok"
    assert out.get("soft_deadline_reached") is True
    # The page fetched before the deadline is still in the answer.
    assert out.get("source") == "web_fetched"
    assert len(out.get("fetched") or []) >= 1


def test_persist_loop_deadline_is_also_reported():
    """Both break sites must set the flag, not just the fetch loop."""
    p = _pipeline()
    p.v.persist = True
    _drive(p, ["https://example.com/a"])
    p.ingest = lambda url, t: {"status": "ok"}

    out = p._run_inner("q", "deep", time.monotonic() - 1)

    assert out.get("soft_deadline_reached") is True
