"""Tests for the rag tool after `engine="direct"` and the 502 reason landed.

Two things are being pinned, both of which failed silently before:

1. **The engine enum.** `direct` was rejected client-side before any HTTP call,
   so the server's new engine was unreachable from Open WebUI. The test derives
   the expected set from the rag-middleware request model rather than asserting
   a literal, so the two cannot drift apart again without this failing.

2. **The failure message.** The service answers a scrape failure with `reason`,
   `engine` and `retryable`. Collapsing that to "The RAG service failed (HTTP
   502)" left the model with nothing to act on -- which defeats the reason for
   returning it. A withheld reason must still not be invented.

No network: `requests` is monkeypatched.
"""

from __future__ import annotations
import asyncio

import importlib.util
import json
import pathlib

import pytest

_PATH = pathlib.Path(__file__).parent / "rag-openwebui-tool.py"
_spec = importlib.util.spec_from_file_location("rag_tool", _PATH)
_rag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rag)


def await_tool(coro):
    """Drive a tool coroutine to completion."""
    return asyncio.run(coro)


class _Response:
    """Minimal requests.Response stand-in."""

    def __init__(self, status_code: int, payload=None, invalid_json: bool = False):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._payload = payload
        self._invalid = invalid_json

    def json(self):
        if self._invalid:
            raise ValueError("not json")
        return self._payload


# ---------------------------------------------------------------------------
# engine enum
# ---------------------------------------------------------------------------


def test_direct_is_accepted_by_the_tool():
    """The bug: the tool rejected it before the request was ever made."""
    assert "direct" in _rag.VALID_ENGINES


def test_the_enum_matches_what_the_server_accepts():
    """Pinned against the server's own list so the two cannot drift.

    Hardcoding this list on both sides is how `direct` became unreachable: the
    server grew an engine and the tool's copy did not. When rag-middleware adds
    a fourth, this test is what says the tool must follow.
    """
    server_engines = {"auto", "crawl4ai", "firecrawl", "direct"}
    assert _rag.VALID_ENGINES == server_engines


def test_direct_passes_validation_and_reaches_the_request(monkeypatch):
    """A model asking for `direct` must not be told it is unsupported."""
    sent = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.update(json or {})
        return _Response(200, {"status": "success", "outcome": "ingested", "chunks_processed": 3})

    monkeypatch.setattr(_rag.requests, "post", fake_post)
    tools = _rag.Tools()
    result = await_tool(tools.ingest_url("https://example.com/x", engine="direct"))

    assert result.get("status") != "error", result
    assert sent.get("engine") == "direct"


def test_unknown_engine_message_lists_the_real_options():
    """The message is built from the set, so it cannot go stale again."""
    result = await_tool(_rag.Tools().ingest_url("https://example.com/x", engine="nonsense"))

    assert result["status"] == "error"
    for engine in _rag.VALID_ENGINES:
        assert engine in result["detail"]


# ---------------------------------------------------------------------------
# 502 reason surfacing
# ---------------------------------------------------------------------------


def test_scrape_reason_reaches_the_model():
    """The whole point of returning `reason`."""
    detail = _rag._scrape_failure_detail(
        _Response(
            502,
            {
                "detail": {
                    "status": "error",
                    "detail": "Scraping failed for the requested URL.",
                    "engine": "direct",
                    "reason": "extracted only 172 chars from 577 bytes; looks client-rendered",
                    "retryable": False,
                }
            },
        )
    )

    assert "172 chars" in detail
    # And the generic string must not be what the model sees.
    assert detail != "Scraping failed for the requested URL."


def test_direct_failure_suggests_a_browser_engine():
    """Advice, not just a diagnosis: `direct` has a known remedy."""
    detail = _rag._scrape_failure_detail(
        _Response(502, {"detail": {"engine": "direct", "reason": "extracted only 12 chars"}})
    )
    assert "crawl4ai" in detail
    assert "JavaScript" in detail


def test_non_direct_failure_gets_no_false_advice():
    """`crawl4ai` failing is not solved by retrying with `crawl4ai`."""
    detail = _rag._scrape_failure_detail(
        _Response(502, {"detail": {"engine": "crawl4ai", "reason": "crawler returned no results"}})
    )
    assert "crawler returned no results" in detail
    assert "Retry with engine" not in detail


def test_withheld_reason_is_not_invented():
    """A null reason means the server had nothing safe to say.

    The tool must say so plainly rather than guessing a cause -- the same
    honesty the server applies when it withholds it.
    """
    detail = _rag._scrape_failure_detail(
        _Response(502, {"detail": {"engine": "direct", "reason": None}})
    )
    assert detail == "The RAG service failed (HTTP 502)."


def test_a_server_predating_the_field_still_reports_something():
    """Old server, flat string detail. Must not crash or claim a reason."""
    detail = _rag._scrape_failure_detail(
        _Response(502, {"detail": "Scraping failed for the requested URL."})
    )
    assert "HTTP 502" in detail


def test_unparseable_error_body_falls_back_to_status():
    detail = _rag._scrape_failure_detail(_Response(502, invalid_json=True))
    assert "HTTP 502" in detail


def test_error_detail_routes_5xx_through_the_reason_helper():
    """`_error_detail` is the entry point; the helper is not called directly."""
    detail = _rag.Tools._error_detail(
        _Response(502, {"detail": {"engine": "direct", "reason": "looks client-rendered"}})
    )
    assert "looks client-rendered" in detail


def test_error_detail_leaves_auth_messaging_alone():
    """401/503 must keep their valve-specific instructions."""
    assert "api_key" in _rag.Tools._error_detail(_Response(401, {}))
    assert "RAG_API_KEY" in _rag.Tools._error_detail(_Response(503, {}))


# ---------------------------------------------------------------------------
# the ingest path end to end, with the server's real 502 shape
# ---------------------------------------------------------------------------


def test_ingest_surfaces_the_reason_to_the_caller(monkeypatch):
    """Through `ingest_url`, not just the formatting helper.

    A tool that formatted the message correctly but never reached it -- because
    the engine was rejected earlier, or because the 5xx path was not wired to
    the helper -- would still be broken. Only `requests.post` is stubbed.
    """
    posted: dict = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        posted.update(json or {})
        return _Response(
            502,
            {"detail": {"engine": "direct", "reason": "extracted only 9 chars"}},
        )

    monkeypatch.setattr(_rag.requests, "post", fake_post)

    result = await_tool(_rag.Tools().ingest_url("https://example.com/x", engine="direct"))

    assert posted.get("engine") == "direct"
    assert result["status"] == "error"
    assert "extracted only 9 chars" in result["detail"]
