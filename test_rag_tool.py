"""Regression tests for rag-openwebui-tool.py.

Covers the API-key handling (fail-closed service, 401/503 diagnosis) and the
rag-middleware v2 response shape: namespaced `source` keys and the per-link
skip/fail reasons that were being counted but not shown.

HTTP is stubbed; no live service is required.

Run:
    python -m unittest test_rag_tool -v
"""

import importlib.util
import pathlib
import sys
import asyncio
import unittest
from unittest import mock

import requests

# The tool filename contains hyphens, so load it by path (same approach as
# test_research_tool.py and test_docker_exec_tool.py in this repo).
_TOOL = pathlib.Path(__file__).with_name("rag-openwebui-tool.py")
_spec = importlib.util.spec_from_file_location("rag_openwebui_tool", _TOOL)
_rag = importlib.util.module_from_spec(_spec)
sys.modules["rag_openwebui_tool"] = _rag
_spec.loader.exec_module(_rag)

_bare_source = _rag._bare_source
_reason_summary = _rag._reason_summary


def _response(status_code, payload=None, ok=None):
    r = mock.Mock(spec=requests.Response)
    r.status_code = status_code
    r.ok = (200 <= status_code < 400) if ok is None else ok
    r.json.return_value = payload if payload is not None else {}
    r.text = str(payload)
    return r


def await_tool(coro):
    """Drive a tool coroutine to completion."""
    return asyncio.run(coro)


class ApiKeyHeaderTests(unittest.TestCase):
    """The X-API-Key header must be present exactly when a key is set."""

    def setUp(self):
        self.tools = _rag.Tools()

    def test_empty_key_sends_no_header(self):
        self.tools.valves.api_key = ""
        self.assertEqual(self.tools._headers(), {})

    def test_whitespace_only_key_sends_no_header(self):
        self.tools.valves.api_key = "   "
        self.assertEqual(self.tools._headers(), {})

    def test_key_is_sent(self):
        self.tools.valves.api_key = "s3cret"
        self.assertEqual(self.tools._headers(), {"X-API-Key": "s3cret"})

    def test_key_is_stripped(self):
        self.tools.valves.api_key = "  s3cret\n"
        self.assertEqual(self.tools._headers(), {"X-API-Key": "s3cret"})

    def test_header_read_per_call(self):
        """A valve change must take effect without re-importing the tool."""
        self.tools.valves.api_key = ""
        self.assertEqual(self.tools._headers(), {})
        self.tools.valves.api_key = "later"
        self.assertEqual(self.tools._headers(), {"X-API-Key": "later"})


class HeaderForwardingTests(unittest.TestCase):
    """Both HTTP calls must actually pass the header."""

    def setUp(self):
        self.tools = _rag.Tools()
        self.tools.valves.api_key = "k-123"

    def test_search_sends_header(self):
        with mock.patch.object(
            _rag.requests, "post", return_value=_response(200, {"status": "success"})
        ) as post:
            await_tool(self.tools.rag_search("what is a vector database"))
        self.assertEqual(post.call_args.kwargs["headers"], {"X-API-Key": "k-123"})

    def test_ingest_sends_header(self):
        with mock.patch.object(
            _rag.requests, "post", return_value=_response(200, {"status": "success"})
        ) as post:
            await_tool(self.tools.ingest_url("https://example.com/page"))
        self.assertEqual(post.call_args.kwargs["headers"], {"X-API-Key": "k-123"})

    def test_ingest_does_not_leak_key_into_payload(self):
        with mock.patch.object(
            _rag.requests, "post", return_value=_response(200, {"status": "success"})
        ) as post:
            await_tool(self.tools.ingest_url("https://example.com/page"))
        body = post.call_args.kwargs["json"]
        self.assertNotIn("k-123", str(body))


class AuthErrorReportingTests(unittest.TestCase):
    """401/503 must be explained, not collapsed into a generic failure."""

    def setUp(self):
        self.tools = _rag.Tools()

    def test_401_names_the_valve(self):
        with mock.patch.object(
            _rag.requests, "post", return_value=_response(401, {"detail": "nope"})
        ):
            result = await_tool(self.tools.rag_search("q"))
        self.assertEqual(result["status"], "error")
        self.assertIn("401", result["detail"])
        self.assertIn("api_key", result["detail"])

    def test_503_explains_server_misconfiguration(self):
        with mock.patch.object(
            _rag.requests, "post", return_value=_response(503, {"detail": "no key"})
        ):
            result = await_tool(self.tools.rag_search("q"))
        self.assertEqual(result["status"], "error")
        self.assertIn("503", result["detail"])
        self.assertIn("RAG_REQUIRE_AUTH", result["detail"])

    def test_401_on_ingest_also_explained(self):
        with mock.patch.object(
            _rag.requests, "post", return_value=_response(401, {"detail": "nope"})
        ):
            result = await_tool(self.tools.ingest_url("https://example.com/x"))
        self.assertEqual(result["status"], "error")
        self.assertIn("api_key", result["detail"])

    def test_500_is_a_server_error_not_a_config_error(self):
        with mock.patch.object(
            _rag.requests, "post", return_value=_response(500, {"detail": "boom"})
        ):
            result = await_tool(self.tools.rag_search("q"))
        self.assertIn("500", result["detail"])
        self.assertNotIn("api_key", result["detail"])

    def test_connection_error_still_generic(self):
        with mock.patch.object(
            _rag.requests,
            "post",
            side_effect=requests.exceptions.ConnectionError("refused"),
        ):
            result = await_tool(self.tools.rag_search("q"))
        self.assertEqual(result["status"], "error")
        self.assertIn("Could not connect", result["detail"])

    def test_success_is_returned_verbatim(self):
        payload = {"status": "success", "context": "<retrieved_passages>x</retrieved_passages>"}
        with mock.patch.object(_rag.requests, "post", return_value=_response(200, payload)):
            result = await_tool(self.tools.rag_search("q"))
        self.assertEqual(result, payload)


class ErrorDetailTests(unittest.TestCase):
    """422 and 404 must produce guidance, not a stringified Python repr (#14).

    FastAPI answers 422 with a LIST of validation objects. The previous
    fallback was str(detail)[:300], which gave the model
    "[{'loc': ['body', 'query'], 'msg': 'too long'}]" -- neither prose nor
    JSON, and it never named the rejected field.
    """

    def test_422_validation_error_is_readable(self):
        r = _response(422, {"detail": [
            {"loc": ["body", "query"], "msg": "String should have at most 4000 characters"}
        ]})
        detail = _rag.Tools._error_detail(r)
        self.assertIn("query", detail)
        self.assertIn("at most 4000", detail)
        # Neither a Python repr nor raw JSON.
        self.assertNotIn("'loc'", detail)
        self.assertNotIn("[{", detail)

    def test_422_with_empty_detail_falls_through(self):
        self.assertIn("422", _rag.Tools._error_detail(_response(422, {"detail": []})))

    def test_422_with_string_detail_still_returned(self):
        r = _response(422, {"detail": "plain reason"})
        self.assertEqual(_rag.Tools._error_detail(r), "plain reason")

    def test_404_names_the_base_url_valve(self):
        detail = _rag.Tools._error_detail(_response(404, {"detail": "Not Found"}))
        self.assertIn("404", detail)
        self.assertIn("rag_server_host", detail)

    def test_401_still_wins_over_404_and_422(self):
        """Order guard: the auth branches must be checked first."""
        self.assertIn("API key", _rag.Tools._error_detail(_response(401, {})))
        self.assertIn("RAG_REQUIRE_AUTH", _rag.Tools._error_detail(_response(503, {})))


class NamespacedSourceTests(unittest.TestCase):
    """rag-middleware stores `url:`/`upload:` keys; the prefix must not reach
    the model as part of a citation (#12)."""

    def test_url_prefix_is_stripped(self):
        self.assertEqual(_bare_source("url:https://example.com/", "fb"),
                         "https://example.com/")

    def test_upload_prefix_is_stripped(self):
        self.assertEqual(_bare_source("upload:notes.txt", "fb"), "notes.txt")

    def test_unprefixed_value_is_unchanged(self):
        self.assertEqual(_bare_source("https://x/", "fb"), "https://x/")

    def test_empty_value_falls_back(self):
        self.assertEqual(_bare_source("", "fb"), "fb")
        self.assertEqual(_bare_source(None, "fb"), "fb")

    def test_prefix_only_falls_back(self):
        # "url:" with nothing after it must not yield an empty citation.
        self.assertEqual(_bare_source("url:", "fb"), "fb")

    def test_ingest_summary_reports_a_bare_source(self):
        payload = {"status": "success", "source": "url:https://example.com/",
                   "outcome": "ingested", "chunks_processed": 2, "links_count": 0,
                   "followed": [], "skipped": [], "failed": []}
        with mock.patch.object(_rag.requests, "post", return_value=_response(200, payload)):
            result = await_tool(_rag.Tools().ingest_url("https://example.com/"))
        self.assertEqual(result["source"], "https://example.com/")
        self.assertNotIn("url:", result["source"])


class ReasonSummaryTests(unittest.TestCase):
    """Skip/fail reasons are surfaced, deduplicated and bounded (#13)."""

    def test_counts_by_reason(self):
        entries = [
            {"url": "a", "reason": "chunk budget exhausted"},
            {"url": "b", "reason": "chunk budget exhausted"},
            {"url": "c", "reason": "time budget exceeded"},
        ]
        self.assertEqual(_reason_summary(entries, "reason"),
                         {"chunk budget exhausted": 2, "time budget exceeded": 1})

    def test_bounded_to_five_reasons(self):
        entries = [{"reason": f"reason-{i}"} for i in range(12)]
        self.assertEqual(len(_reason_summary(entries, "reason")), 5)

    def test_non_list_and_bad_entries_do_not_raise(self):
        self.assertEqual(_reason_summary(None, "reason"), {})
        self.assertEqual(_reason_summary("nope", "reason"), {})
        self.assertEqual(_reason_summary(["str", 42, None], "reason"), {})

    def test_missing_reason_becomes_unknown(self):
        self.assertEqual(_reason_summary([{"url": "a"}], "reason"), {"unknown": 1})

    def test_ingest_summary_surfaces_reasons(self):
        payload = {
            "status": "success", "source": "url:https://x/", "outcome": "ingested",
            "chunks_processed": 1, "links_count": 4,
            "followed": [],
            "skipped": [{"url": "a", "reason": "chunk budget exhausted"},
                        {"url": "b", "reason": "chunk budget exhausted"},
                        {"url": "c", "reason": "time budget exceeded"}],
            "failed": [{"url": "d", "error": "scrape failed"}],
        }
        with mock.patch.object(_rag.requests, "post", return_value=_response(200, payload)):
            result = await_tool(_rag.Tools().ingest_url("https://x/"))
        self.assertEqual(result["skipped_reasons"],
                         {"chunk budget exhausted": 2, "time budget exceeded": 1})
        self.assertEqual(result["failed_reasons"], {"scrape failed": 1})
        # Counts are additive, not replaced.
        self.assertEqual(result["skipped_count"], 3)
        self.assertEqual(result["failed_count"], 1)

    def test_clean_ingest_omits_empty_reason_dicts(self):
        payload = {"status": "success", "source": "url:https://x/", "outcome": "ingested",
                   "chunks_processed": 2, "links_count": 0,
                   "followed": [], "skipped": [], "failed": []}
        with mock.patch.object(_rag.requests, "post", return_value=_response(200, payload)):
            result = await_tool(_rag.Tools().ingest_url("https://x/"))
        self.assertNotIn("skipped_reasons", result)
        self.assertNotIn("failed_reasons", result)


class AuthRegressionGuardTests(unittest.TestCase):
    """Guards against silently dropping the header in a future refactor."""

    def test_every_post_call_passes_headers(self):
        tools = _rag.Tools()
        tools.valves.api_key = "k"
        calls = [
            (tools.rag_search, ("q",), {}),
            (tools.ingest_url, ("https://example.com/x",), {}),
        ]
        with mock.patch.object(
            _rag.requests, "post", return_value=_response(200, {"status": "success"})
        ) as post:
            for fn, args, kwargs in calls:
                await_tool(fn(*args, **kwargs))
        self.assertEqual(post.call_count, len(calls))
        for call in post.call_args_list:
            self.assertIn("headers", call.kwargs, "a POST is missing the auth header")
            self.assertEqual(call.kwargs["headers"], {"X-API-Key": "k"})


if __name__ == "__main__":
    unittest.main()
