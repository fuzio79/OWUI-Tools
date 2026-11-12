"""Regression tests for the rag-middleware v2 response shape in the research tool.

Kept separate from test_research_tool.py so it can be reviewed on its own: the
two defects here are about the *ingest* response, while that file is largely
concerned with the search/fetch pipeline.

Covers:
  * #12 — the server now returns `url:<url>` as `source`. `_clean_url` rejects
    the `url:` scheme, so the previous `_clean_url(source) or url` silently
    fell back to the caller's URL and hid the fact that the crawler landed
    elsewhere after a redirect.
  * #13 — `_summarize_ingest` counted skipped/failed links but discarded the
    server's per-link reasons, which are the actionable part.

Run:
    python -m unittest test_research_ingest_v2 -v
"""

import importlib.util
import pathlib
import sys
import unittest

_TOOL = pathlib.Path(__file__).with_name("research-openwebui-tool.py")
_spec = importlib.util.spec_from_file_location("research_tool_v2", _TOOL)
_m = importlib.util.module_from_spec(_spec)
sys.modules["research_tool_v2"] = _m
_spec.loader.exec_module(_m)

_Pipeline = _m._Pipeline


def _ingest_response(**overrides):
    """A minimal ingest reply in the shape rag-middleware actually sends."""
    payload = {
        "status": "success",
        "source": "url:https://example.com/",
        "engine": "crawl4ai",
        "outcome": "ingested",
        "chunks_processed": 2,
        "links_count": 0,
        "followed": [],
        "skipped": [],
        "failed": [],
    }
    payload.update(overrides)
    return payload


class BareSourceTests(unittest.TestCase):
    """The `url:`/`upload:` namespace is a storage key, not a citation (#12)."""

    def test_url_prefix_is_stripped(self):
        self.assertEqual(
            _m._bare_source("url:https://example.com/", "fb"),
            "https://example.com/",
        )

    def test_upload_prefix_is_stripped(self):
        self.assertEqual(_m._bare_source("upload:notes.txt", "fb"), "notes.txt")

    def test_unprefixed_value_is_unchanged(self):
        self.assertEqual(_m._bare_source("https://x/", "fb"), "https://x/")

    def test_empty_and_none_fall_back(self):
        self.assertEqual(_m._bare_source("", "fb"), "fb")
        self.assertEqual(_m._bare_source(None, "fb"), "fb")

    def test_prefix_only_falls_back(self):
        # "url:" with nothing after it must not become an empty citation.
        self.assertEqual(_m._bare_source("url:", "fb"), "fb")


class SummarizeIngestSourceTests(unittest.TestCase):
    """The redirect target must survive, and the prefix must not leak."""

    def test_namespaced_source_is_unwrapped(self):
        out = _Pipeline._summarize_ingest(
            _ingest_response(source="url:https://example.com/"),
            "https://example.com/",
        )
        self.assertEqual(out["source"], "https://example.com/")
        self.assertNotIn("url:", out["source"])

    def test_redirect_target_wins_over_the_requested_url(self):
        """This is why the fix matters: _clean_url used to blank the source.

        With `url:` in front, `_clean_url` returned "" and the summary fell
        back to the caller's URL, so a page that redirected reported the
        address that had been requested rather than the one that was stored.
        """
        out = _Pipeline._summarize_ingest(
            _ingest_response(source="url:https://landed.example/final"),
            "https://requested.example/",
        )
        self.assertEqual(out["source"], "https://landed.example/final")
        self.assertNotEqual(out["source"], "https://requested.example/")

    def test_unusable_source_falls_back_to_the_caller_url(self):
        # An upload: key is not a fetchable URL, so the caller's value wins.
        out = _Pipeline._summarize_ingest(
            _ingest_response(source="upload:notes.txt"),
            "https://requested.example/",
        )
        self.assertEqual(out["source"], "https://requested.example/")

    def test_bad_shape_is_still_handled(self):
        out = _Pipeline._summarize_ingest("not a dict", "https://a.example")
        self.assertEqual(out["source"], "https://a.example")


class ReasonSummaryTests(unittest.TestCase):
    """Skip/fail reasons are surfaced, deduplicated and bounded (#13)."""

    def test_counts_by_reason(self):
        entries = [
            {"url": "a", "reason": "chunk budget exhausted"},
            {"url": "b", "reason": "chunk budget exhausted"},
            {"url": "c", "reason": "time budget exceeded"},
        ]
        self.assertEqual(
            _m._reason_summary(entries, "reason"),
            {"chunk budget exhausted": 2, "time budget exceeded": 1},
        )

    def test_bounded_to_five_reasons(self):
        entries = [{"reason": f"reason-{i}"} for i in range(20)]
        self.assertEqual(len(_m._reason_summary(entries, "reason")), 5)

    def test_non_list_and_bad_entries_do_not_raise(self):
        self.assertEqual(_m._reason_summary(None, "reason"), {})
        self.assertEqual(_m._reason_summary("nope", "reason"), {})
        self.assertEqual(_m._reason_summary(["s", 42, None], "reason"), {})

    def test_missing_reason_becomes_unknown(self):
        self.assertEqual(_m._reason_summary([{"url": "a"}], "reason"), {"unknown": 1})


class SummarizeIngestReasonTests(unittest.TestCase):
    """The summary must show *why* links were skipped, not just how many."""

    def test_reasons_are_surfaced_alongside_counts(self):
        out = _Pipeline._summarize_ingest(
            _ingest_response(
                links_count=4,
                skipped=[
                    {"url": "a", "reason": "chunk budget exhausted"},
                    {"url": "b", "reason": "chunk budget exhausted"},
                    {"url": "c", "reason": "time budget exceeded"},
                ],
                failed=[{"url": "d", "error": "scrape failed"}],
            ),
            "https://example.com/",
        )
        self.assertEqual(
            out["skipped_reasons"],
            {"chunk budget exhausted": 2, "time budget exceeded": 1},
        )
        self.assertEqual(out["failed_reasons"], {"scrape failed": 1})
        # Counts are additive, not replaced.
        self.assertEqual(out["skipped_count"], 3)
        self.assertEqual(out["failed_count"], 1)

    def test_clean_ingest_omits_empty_reason_dicts(self):
        # An empty dict on every clean ingest reads as a problem.
        out = _Pipeline._summarize_ingest(_ingest_response(), "https://example.com/")
        self.assertNotIn("skipped_reasons", out)
        self.assertNotIn("failed_reasons", out)

    def test_summary_stays_bounded(self):
        import json

        out = _Pipeline._summarize_ingest(
            _ingest_response(
                skipped=[{"url": f"u{i}", "reason": "time budget exceeded"}
                         for i in range(500)],
            ),
            "https://example.com/",
        )
        self.assertEqual(out["skipped_count"], 500)
        # 500 identical reasons collapse to one entry.
        self.assertEqual(out["skipped_reasons"], {"time budget exceeded": 500})
        self.assertLess(len(json.dumps(out)), 600)


if __name__ == "__main__":
    unittest.main()
