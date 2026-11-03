"""Regression tests for rag-openwebui-tool.py: API-key handling.

The RAG service is fail-closed: without a matching key every data endpoint
returns 401, and a server with no key configured returns 503. Both are
configuration problems, so the tool must report them as such instead of the
generic "could not connect" that would send a user hunting for a network fault.

HTTP is stubbed; no live service is required.

Run:
    python -m unittest test_rag_tool -v
"""

import importlib.util
import pathlib
import sys
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


def _response(status_code, payload=None, ok=None):
    r = mock.Mock(spec=requests.Response)
    r.status_code = status_code
    r.ok = (200 <= status_code < 400) if ok is None else ok
    r.json.return_value = payload if payload is not None else {}
    r.text = str(payload)
    return r


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
            self.tools.rag_search("what is a vector database")
        self.assertEqual(post.call_args.kwargs["headers"], {"X-API-Key": "k-123"})

    def test_ingest_sends_header(self):
        with mock.patch.object(
            _rag.requests, "post", return_value=_response(200, {"status": "success"})
        ) as post:
            self.tools.ingest_url("https://example.com/page")
        self.assertEqual(post.call_args.kwargs["headers"], {"X-API-Key": "k-123"})

    def test_ingest_does_not_leak_key_into_payload(self):
        with mock.patch.object(
            _rag.requests, "post", return_value=_response(200, {"status": "success"})
        ) as post:
            self.tools.ingest_url("https://example.com/page")
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
            result = self.tools.rag_search("q")
        self.assertEqual(result["status"], "error")
        self.assertIn("401", result["detail"])
        self.assertIn("api_key", result["detail"])

    def test_503_explains_server_misconfiguration(self):
        with mock.patch.object(
            _rag.requests, "post", return_value=_response(503, {"detail": "no key"})
        ):
            result = self.tools.rag_search("q")
        self.assertEqual(result["status"], "error")
        self.assertIn("503", result["detail"])
        self.assertIn("RAG_REQUIRE_AUTH", result["detail"])

    def test_401_on_ingest_also_explained(self):
        with mock.patch.object(
            _rag.requests, "post", return_value=_response(401, {"detail": "nope"})
        ):
            result = self.tools.ingest_url("https://example.com/x")
        self.assertEqual(result["status"], "error")
        self.assertIn("api_key", result["detail"])

    def test_500_is_a_server_error_not_a_config_error(self):
        with mock.patch.object(
            _rag.requests, "post", return_value=_response(500, {"detail": "boom"})
        ):
            result = self.tools.rag_search("q")
        self.assertIn("500", result["detail"])
        self.assertNotIn("api_key", result["detail"])

    def test_connection_error_still_generic(self):
        with mock.patch.object(
            _rag.requests,
            "post",
            side_effect=requests.exceptions.ConnectionError("refused"),
        ):
            result = self.tools.rag_search("q")
        self.assertEqual(result["status"], "error")
        self.assertIn("Could not connect", result["detail"])

    def test_success_is_returned_verbatim(self):
        payload = {"status": "success", "context": "<retrieved_passages>x</retrieved_passages>"}
        with mock.patch.object(_rag.requests, "post", return_value=_response(200, payload)):
            result = self.tools.rag_search("q")
        self.assertEqual(result, payload)


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
                fn(*args, **kwargs)
        self.assertEqual(post.call_count, len(calls))
        for call in post.call_args_list:
            self.assertIn("headers", call.kwargs, "a POST is missing the auth header")
            self.assertEqual(call.kwargs["headers"], {"X-API-Key": "k"})


if __name__ == "__main__":
    unittest.main()
