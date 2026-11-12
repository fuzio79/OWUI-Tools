"""Tests for the two model-facing tools added on top of the new endpoints.

Both hand the model an action that was previously operator-only, so the tests
concentrate on the guardrails rather than the happy path:

* `delete_source` must **default to dry_run**, so a casual call reports rather
  than removes. That default is the entire safety design and a refactor that
  flipped it would silently make deletion the easy path.
* `fetch_url` must be unmistakably **non-persisting** in its result, so a model
  does not later claim the page is in the knowledge base.

No network: `requests.post` is stubbed.
"""

from __future__ import annotations
import asyncio

import importlib.util
import inspect
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
    def __init__(self, status_code=200, payload=None, invalid_json=False):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._payload = payload
        self._invalid = invalid_json

    def json(self):
        if self._invalid:
            raise ValueError("not json")
        return self._payload


@pytest.fixture
def posted(monkeypatch):
    """Capture the request payload and reply with a scripted response."""
    calls: list[tuple[str, dict]] = []

    def install(payload, status=200, invalid_json=False):
        def fake_post(url, json=None, headers=None, timeout=None):
            calls.append((url, json or {}))
            return _Response(status, payload, invalid_json)

        monkeypatch.setattr(_rag.requests, "post", fake_post)
        return calls

    return install


# ---------------------------------------------------------------------------
# the tools exist and are reachable by the model
# ---------------------------------------------------------------------------


def _tool_methods() -> set[str]:
    return {
        name
        for name, _ in inspect.getmembers(_rag.Tools, predicate=inspect.isfunction)
        if not name.startswith("_")
    }


def test_both_tools_are_exposed():
    methods = _tool_methods()
    assert "fetch_url" in methods
    assert "delete_source" in methods


def test_each_tool_has_a_docstring():
    """The docstring is the schema; a bare method teaches the model nothing."""
    assert _rag.Tools.fetch_url.__doc__
    assert _rag.Tools.delete_source.__doc__


# ---------------------------------------------------------------------------
# delete_source: the dry-run default is the guardrail
# ---------------------------------------------------------------------------


def test_delete_defaults_to_dry_run(posted):
    """A casual call must report, not delete.

    This is the safety design in one assertion: if the default flipped, taking
    content out of the knowledge base would become the path of least resistance
    for a model that was only asked a question.
    """
    sig = inspect.signature(_rag.Tools.delete_source)
    assert sig.parameters["dry_run"].default is True


def test_the_dry_run_default_reaches_the_server(posted):
    calls = posted({"status": "success", "dry_run": True, "chunks_to_delete": 7, "keys": {}})

    result = await_tool(_rag.Tools().delete_source("https://example.com/x"))

    assert calls[0][0].endswith("/api/v1/rag/delete")
    assert calls[0][1]["dry_run"] is True
    assert result["dry_run"] is True
    assert result["chunks_to_delete"] == 7


def test_the_dry_run_note_tells_the_model_to_confirm(posted):
    """A count without an instruction invites the model to guess what next."""
    posted({"status": "success", "dry_run": True, "chunks_to_delete": 3, "keys": {}})
    note = await_tool(_rag.Tools().delete_source("https://example.com/x"))["note"].lower()

    assert "nothing was deleted" in note
    assert "confirm" in note
    assert "dry_run=false" in note


def test_an_explicit_delete_is_carried_through(posted):
    calls = posted({"status": "success", "chunks_deleted": 5})

    result = await_tool(_rag.Tools().delete_source("https://example.com/x", dry_run=False))

    assert calls[0][1]["dry_run"] is False
    assert result["dry_run"] is False
    assert result["chunks_deleted"] == 5
    assert "cannot be undone" in result["note"].lower()


def test_deleting_nothing_says_so_instead_of_claiming_success(posted):
    posted({"status": "not_found", "chunks_deleted": 0})

    result = await_tool(_rag.Tools().delete_source("https://example.com/never", dry_run=False))

    assert result["chunks_deleted"] == 0
    assert "nothing was deleted" in result["note"].lower()


def test_an_empty_source_is_rejected_without_a_request(posted):
    calls = posted({})

    result = await_tool(_rag.Tools().delete_source("   "))

    assert result["status"] == "error"
    assert calls == []


def test_a_server_error_is_reported(posted):
    posted({"detail": "boom"}, status=500)

    result = await_tool(_rag.Tools().delete_source("https://example.com/x", dry_run=False))

    assert result["status"] == "error"
    assert "detail" in result


# ---------------------------------------------------------------------------
# fetch_url: non-persisting, and says so
# ---------------------------------------------------------------------------


def test_fetch_uses_the_fetch_endpoint(posted):
    calls = posted(
        {
            "status": "success",
            "source": "url:https://example.com/x",
            "engine": "direct",
            "passages_returned": 2,
            "chunks_scanned": 9,
            "context": "passages here",
        }
    )

    await_tool(_rag.Tools().fetch_url("https://example.com/x", "what is it"))

    assert calls[0][0].endswith("/api/v1/rag/fetch")
    assert calls[0][1]["query"] == "what is it"


def test_the_result_says_it_was_not_stored(posted):
    """A model that thinks it indexed a page will make false claims later."""
    posted(
        {
            "status": "success",
            "source": "url:https://example.com/x",
            "engine": "direct",
            "passages_returned": 1,
            "chunks_scanned": 4,
            "context": "ctx",
        }
    )

    result = await_tool(_rag.Tools().fetch_url("https://example.com/x", "q"))

    assert result["stored"] is False
    assert "not stored" in result["note"].lower()


def test_the_source_prefix_is_stripped_for_display(posted):
    posted(
        {
            "status": "success",
            "source": "url:https://example.com/x",
            "engine": "direct",
            "passages_returned": 1,
            "chunks_scanned": 1,
            "context": "ctx",
        }
    )

    assert await_tool(_rag.Tools().fetch_url("https://example.com/x", "q"))["source"] == "https://example.com/x"


def test_top_k_is_clamped_not_rejected(posted):
    """A model asking for 50 should get a usable answer, not an error."""
    calls = posted(
        {
            "status": "success",
            "source": "url:https://example.com/x",
            "engine": "direct",
            "passages_returned": 20,
            "chunks_scanned": 30,
            "context": "ctx",
        }
    )

    result = await_tool(_rag.Tools().fetch_url("https://example.com/x", "q", top_k=50))

    assert calls[0][1]["top_k"] == 20
    assert result["status"] == "success"


def test_an_empty_query_is_rejected_without_a_request(posted):
    calls = posted({})

    result = await_tool(_rag.Tools().fetch_url("https://example.com/x", "  "))

    assert result["status"] == "error"
    assert calls == []


def test_an_unknown_engine_is_rejected_without_a_request(posted):
    calls = posted({})

    result = await_tool(_rag.Tools().fetch_url("https://example.com/x", "q", engine="headless"))

    assert result["status"] == "error"
    assert calls == []


def test_a_non_http_url_is_rejected_without_a_request(posted):
    calls = posted({})

    result = await_tool(_rag.Tools().fetch_url("file:///etc/passwd", "q"))

    assert result["status"] == "error"
    assert calls == []


def test_a_thin_page_reason_reaches_the_model(posted):
    """The whole point of the scrape reason: the model can act on it."""
    posted(
        {
            "detail": {
                "status": "error",
                "detail": "Scraping failed for the requested URL.",
                "engine": "direct",
                "reason": "extracted only 12 chars from 40 bytes",
                "retryable": False,
            }
        },
        status=502,
    )

    result = await_tool(_rag.Tools().fetch_url("https://example.com/x", "q", engine="direct"))

    assert result["status"] == "error"
    assert "12 chars" in result["detail"]
    assert "crawl4ai" in result["detail"]


def test_a_timeout_explains_the_likely_cause(posted, monkeypatch):
    """A timeout on a long page is not the model's fault and not retryable as-is."""

    def fake_post(*a, **k):
        raise _rag.requests.exceptions.Timeout("slow")

    monkeypatch.setattr(_rag.requests, "post", fake_post)

    result = await_tool(_rag.Tools().fetch_url("https://example.com/x", "q", engine="direct"))

    assert result["status"] == "error"
    detail = result["detail"].lower()
    assert "too long" in detail or "timeout" in detail
    # The remedy has to be nameable, otherwise the model can only retry blindly.
    assert "ingest_timeout_s" in detail or "smaller page" in detail


def test_a_connection_failure_is_reported_not_raised(posted, monkeypatch):
    def fake_post(*a, **k):
        raise _rag.requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(_rag.requests, "post", fake_post)

    result = await_tool(_rag.Tools().fetch_url("https://example.com/x", "q"))

    assert result["status"] == "error"
    assert "connect" in result["detail"].lower()


def test_a_delete_connection_failure_is_reported_not_raised(posted, monkeypatch):
    """An exception escaping here would fail the whole chat turn."""

    def fake_post(*a, **k):
        raise _rag.requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(_rag.requests, "post", fake_post)

    result = await_tool(_rag.Tools().delete_source("https://example.com/x"))

    assert result["status"] == "error"
    assert "connect" in result["detail"].lower()


def test_invalid_json_is_reported_not_raised(posted):
    posted(None, invalid_json=True)

    result = await_tool(_rag.Tools().fetch_url("https://example.com/x", "q"))

    assert result["status"] == "error"
    assert "invalid" in result["detail"].lower()
