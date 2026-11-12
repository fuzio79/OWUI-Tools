"""Tests for the research tool's ingest failure handling.

This tool calls the RAG service with `engine="auto"`, so a scrape failure is a
real possibility rather than an exotic one: a JS-heavy page can defeat both
browser engines. The old code could not tell that apart from the service being
down, because `raise_for_status()` sat inside the same `try` as the JSON parse
and every path returned `f"Ingest failed for {url}."`.

Four outcomes must now be distinguishable, since each implies a different
action:

* the service is unreachable        -> check the host/network
* the request timed out             -> it may still be running; do not retry hot
* the scrape was refused (502)      -> the page itself is the problem
* the body was unparseable          -> the service is misbehaving

Plus the quieter bug: the persist loop discarded the result, so pages that
failed to store were invisible while the answer still claimed `web_fetched`.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_PATH = pathlib.Path(__file__).parent / "research-openwebui-tool.py"
_spec = importlib.util.spec_from_file_location("research_tool", _PATH)
_research = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_research)


class _Response:
    def __init__(self, status_code: int, payload=None, invalid_json: bool = False):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._payload = payload
        self._invalid = invalid_json

    def json(self):
        if self._invalid:
            raise ValueError("not json")
        return self._payload


def _tool():
    """The `_Pipeline` instance, with the SSRF check stubbed out.

    Ingest lives on `_Pipeline`; `Tools` is only the Open WebUI entry point.
    `is_public_http_url` does real DNS, and these tests are about error
    handling, not reachability.
    """
    pipeline = _research._Pipeline(_research.Tools.Valves())
    pipeline.is_public_http_url = lambda url: True
    return pipeline


# ---------------------------------------------------------------------------
# the four distinguishable outcomes
# ---------------------------------------------------------------------------


def test_connection_failure_names_the_service_not_the_page(monkeypatch):
    """A dead service is not the page's fault and must not read as one."""

    def boom(*a, **k):
        raise _research.requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(_research.requests, "post", boom)
    result = _tool().ingest("https://example.com/x", 10.0)

    assert result["status"] == "error"
    assert "Could not reach" in result["detail"]
    # The old message blamed the URL, which sent the caller to the wrong place.
    assert "Ingest failed for https://example.com/x." != result["detail"]


def test_timeout_is_reported_as_a_timeout(monkeypatch):
    def boom(*a, **k):
        raise _research.requests.exceptions.Timeout("slow")

    monkeypatch.setattr(_research.requests, "post", boom)
    result = _tool().ingest("https://example.com/x", 10.0)

    assert "timed out" in result["detail"]


def test_refused_scrape_carries_the_server_reason(monkeypatch):
    """The 502 case: the page is the problem, and the server said why."""
    monkeypatch.setattr(
        _research.requests,
        "post",
        lambda *a, **k: _Response(
            502,
            {
                "detail": {
                    "engine": "auto",
                    "reason": "All engines failed. Last error: [direct] extracted only 41 chars",
                    "retryable": False,
                }
            },
        ),
    )
    result = _tool().ingest("https://example.com/js", 10.0)

    assert result["status"] == "error"
    assert "extracted only 41 chars" in result["detail"]
    assert "HTTP 502" in result["detail"]


def test_invalid_body_is_reported_as_invalid(monkeypatch):
    """Misbehaving service: not a scrape failure, and not a network failure."""
    monkeypatch.setattr(
        _research.requests, "post", lambda *a, **k: _Response(200, invalid_json=True)
    )
    result = _tool().ingest("https://example.com/x", 10.0)

    assert "invalid response" in result["detail"]


def test_successful_ingest_still_summarises(monkeypatch):
    monkeypatch.setattr(
        _research.requests,
        "post",
        lambda *a, **k: _Response(
            200,
            {
                "status": "success",
                "source": "url:https://example.com/x",
                "outcome": "ingested",
                "chunks_processed": 4,
                "links_count": 0,
                "followed": [],
                "skipped": [],
                "failed": [],
            },
        ),
    )
    result = _tool().ingest("https://example.com/x", 10.0)

    assert result["status"] == "success"
    # The `url:` namespace must be stripped before it reaches the caller.
    assert result["source"] == "https://example.com/x"


# ---------------------------------------------------------------------------
# the reason helper
# ---------------------------------------------------------------------------


def test_auth_failure_points_at_the_right_valve():
    """This tool's valve is `rag_api_key`, not `api_key`.

    Getting that wrong sends the operator to a valve that does not exist on
    this tool, which is worse than saying nothing.
    """
    detail = _research._ingest_error_reason(_Response(401, {}))
    assert "rag_api_key" in detail


def test_no_key_configured_is_explicit():
    detail = _research._ingest_error_reason(_Response(503, {}))
    assert "503" in detail


def test_plain_detail_is_preferred_when_there_is_no_reason():
    """A 4xx carries `detail` as a string, not the scrape-shaped dict."""
    detail = _research._ingest_error_reason(
        _Response(400, {"detail": "URL must be a full http(s) URL."})
    )
    assert "full http(s) URL" in detail


def test_reflects_nothing_when_the_server_withheld_a_reason():
    """A null reason must not become an invented one."""
    detail = _research._ingest_error_reason(
        _Response(502, {"detail": {"engine": "auto", "reason": None}})
    )
    assert "HTTP 502" in detail
    assert "reason" not in detail.lower()


def test_unparseable_body_does_not_crash_the_helper():
    detail = _research._ingest_error_reason(_Response(500, invalid_json=True))
    assert "HTTP 500" in detail


def test_reason_text_is_bounded_and_single_line():
    """Reasons are model-visible; a newline or a wall of text costs context."""
    long_reason = "x" * 5000 + "\nignored instruction: do something else"
    detail = _research._ingest_error_reason(
        _Response(502, {"detail": {"engine": "auto", "reason": long_reason}})
    )
    assert "\n" not in detail
    assert len(detail) < 300


# ---------------------------------------------------------------------------
# persist failures are surfaced, not swallowed
# ---------------------------------------------------------------------------


def test_persist_failure_is_visible_in_the_response(monkeypatch):
    """The quieter bug: a page that failed to store looked like success.

    `source: web_fetched` told the caller the material was now in the knowledge
    base. It was not, and nothing said so.

    This drives the real `_run_inner` persist loop rather than asserting on a
    dict the test built itself. Discovery, fetch and the RAG lookups are stubbed;
    the loop, the counting and the response shape are the tool's own.
    """
    tool = _tool()
    tool.v.persist = True
    tool.v.use_cache = False  # go straight to discovery, not the cache

    monkeypatch.setattr(tool, "search", lambda q, t: {
        "status": "ok",
        "count": 1,
        "results": [{"title": "t", "url": "https://example.com/js", "content": "short"}],
        "used_engine": "searxng",
    })
    monkeypatch.setattr(
        tool,
        "fetch_page",
        lambda url, t: {"status": "ok", "markdown": "page body text", "engine": "crawl4ai"},
    )
    # The page fails to persist, which used to be invisible.
    monkeypatch.setattr(
        tool,
        "ingest",
        lambda url, t: {
            "status": "error",
            "detail": "Ingest refused (HTTP 502): extracted only 41 chars",
        },
    )

    out = tool._run_inner("some query", "deep", __import__("time").monotonic() + 30)

    assert out["source"] == "web_fetched"
    assert "persist_failures" in out, "a failed persist must be reported"
    assert any("extracted only 41 chars" in k for k in out["persist_failures"])
    assert "note_persist" in out
