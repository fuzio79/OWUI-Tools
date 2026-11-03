"""Regression tests for the research tool's RAG API-key handling.

`research-openwebui-tool.py` calls the RAG service's /search and /ingest
endpoints internally. Once that service is fail-closed, a missing key turns
every cache lookup and every persist into a 401, and the tool silently degrades
to web-only — the failure is invisible because it is caught and returned as an
error dict.

These tests pin the header behaviour so that degradation cannot come back
unnoticed.

Run:
    python -m unittest test_research_rag_key -v
"""

import importlib.util
import pathlib
import sys
import unittest
from unittest import mock

import requests

_TOOL = pathlib.Path(__file__).with_name("research-openwebui-tool.py")
_spec = importlib.util.spec_from_file_location("research_openwebui_tool_key", _TOOL)
_research = importlib.util.module_from_spec(_spec)
sys.modules["research_openwebui_tool_key"] = _research
_spec.loader.exec_module(_research)


def _response(status_code, payload=None):
    r = mock.Mock(spec=requests.Response)
    r.status_code = status_code
    r.ok = 200 <= status_code < 400
    r.json.return_value = payload if payload is not None else {}
    r.text = str(payload)
    if not r.ok:
        r.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"{status_code} error"
        )
    else:
        r.raise_for_status.return_value = None
    return r


class _Base(unittest.TestCase):
    def setUp(self):
        self.tool = _research.Tools()
        self.pipeline = self.tool._pipeline() if hasattr(self.tool, "_pipeline") else None

    def _pipeline_obj(self):
        """The pipeline object that owns rag_search/ingest."""
        if self.pipeline is not None:
            return self.pipeline
        # Fall back: construct the internal pipeline class directly.
        for name in dir(_research):
            if "Pipeline" in name:
                cls = getattr(_research, name)
                try:
                    return cls(self.tool.valves)
                except TypeError:
                    return cls(self.tool)
        raise unittest.SkipTest("could not locate the internal pipeline class")


class RagHeaderTests(_Base):
    def test_no_key_sends_no_header(self):
        p = self._pipeline_obj()
        p.v.rag_api_key = ""
        self.assertEqual(p._rag_headers(), {})

    def test_whitespace_key_sends_no_header(self):
        p = self._pipeline_obj()
        p.v.rag_api_key = "   "
        self.assertEqual(p._rag_headers(), {})

    def test_key_is_sent_and_stripped(self):
        p = self._pipeline_obj()
        p.v.rag_api_key = "  k-9\n"
        self.assertEqual(p._rag_headers(), {"X-API-Key": "k-9"})


class RagCallForwardingTests(_Base):
    def test_search_sends_header(self):
        p = self._pipeline_obj()
        p.v.rag_api_key = "k-9"
        with mock.patch.object(
            _research.requests, "post", return_value=_response(200, {"context": "x"})
        ) as post:
            p.rag_search("q", 10)
        self.assertEqual(post.call_args.kwargs["headers"], {"X-API-Key": "k-9"})

    def test_ingest_sends_header(self):
        p = self._pipeline_obj()
        p.v.rag_api_key = "k-9"
        with mock.patch.object(
            _research.requests, "post", return_value=_response(200, {"status": "success"})
        ) as post:
            try:
                p.ingest("https://example.com/x", 10)
            except TypeError:
                p.ingest("https://example.com/x", 10, 10)
        self.assertEqual(post.call_args.kwargs["headers"], {"X-API-Key": "k-9"})


class RagAuthFailureTests(_Base):
    def test_401_on_search_degrades_without_raising(self):
        """A 401 must not crash the pipeline; it returns an error dict."""
        p = self._pipeline_obj()
        p.v.rag_api_key = "wrong"
        with mock.patch.object(
            _research.requests, "post", return_value=_response(401, {"detail": "no"})
        ):
            result = p.rag_search("q", 10)
        self.assertEqual(result.get("status"), "error")

    def test_401_is_not_reported_as_a_cache_hit(self):
        p = self._pipeline_obj()
        p.v.rag_api_key = "wrong"
        with mock.patch.object(
            _research.requests, "post", return_value=_response(401, {"detail": "no"})
        ):
            result = p.rag_search("q", 10)
        self.assertNotIn("context", result)


if __name__ == "__main__":
    unittest.main()
