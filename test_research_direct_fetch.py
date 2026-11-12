"""Tests for the research tool's direct fetch engine.

Three properties are being pinned, in order of how much they matter:

1. **Validate-and-fetch in one loop.** Each redirect hop is re-checked by the
   SSRF guard immediately before it is requested. The test asserts the private
   host was never *requested*, which is the only assertion that distinguishes
   this from "validate, then hope".
2. **The quality gate.** A client-rendered shell must be refused with the
   measurements, not returned as a two-chunk success. This is the failure mode
   that makes a plain fetcher dangerous: it looks like it worked.
3. **Dispatch.** `crawl4ai` stays the default; `direct` is opt-in.

`requests.get` is stubbed throughout -- no network.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_PATH = pathlib.Path(__file__).parent / "research-openwebui-tool.py"
_spec = importlib.util.spec_from_file_location("research_tool", _PATH)
_research = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_research)


class _Resp:
    def __init__(self, status=200, body=b"", content_type="text/html; charset=utf-8", location=None):
        self.status_code = status
        self.content = body
        self.headers = {"content-type": content_type}
        if location is not None:
            self.headers["location"] = location


class _Site:
    """URL -> _Resp, recording every URL actually requested."""

    def __init__(self, pages=None):
        self.pages = pages or {}
        self.seen = []

    def __call__(self, url, **kwargs):
        self.seen.append(url)
        reply = self.pages.get(url)
        if reply is None:
            return _Resp(status=404, body=b"not found")
        return reply


def _pipeline(site, **valves):
    """A _Pipeline with DNS stubbed and fetch_engine=direct."""
    p = _research._Pipeline(_research.Tools.Valves())
    p.v.fetch_engine = "direct"
    p.v.direct_min_chars = 200
    for k, v in valves.items():
        setattr(p.v, k, v)
    p.is_public_http_url = lambda url: True
    p._site = site
    return p


ARTICLE = (
    "<html><body>"
    "<nav><a href='/'>Home</a><a href='/docs'>Docs</a></nav>"
    "<article><h1>Retrieval</h1>"
    "<p>Chunking decides retrieval quality more than the embedding model does. "
    "A window that keeps a paragraph intact retrieves better than a tighter one "
    "that splits sentences, because the vector then describes a coherent idea "
    "rather than a fragment of one, and fragments match everything weakly.</p>"
    "<p>Cosine distance is the usual metric for normalised text embeddings.</p>"
    "</article></body></html>"
)

SHELL = "<html><body><div id='root'></div><nav><a href='/'>Home</a></nav></body></html>"


@pytest.fixture
def patch_get(monkeypatch):
    """Patch requests.get to dispatch to the current _Site."""

    def install(site):
        monkeypatch.setattr(_research.requests, "get", site)
        return site

    return install


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


def test_crawl4ai_remains_the_default():
    """The opt-in has to be opt-in: #30 measured direct extracting far less."""
    assert _research.Tools.Valves().fetch_engine == "crawl4ai"


def test_fetch_page_dispatches_to_direct(patch_get):
    site = patch_get(_Site({"https://example.com/a": _Resp(body=ARTICLE.encode())}))
    p = _pipeline(site)
    result = p.fetch_page("https://example.com/a", 10)

    assert result["status"] == "ok"
    assert result["engine"] == "direct"


def test_crawl4ai_path_is_unchanged(patch_get):
    """Selecting crawl4ai must not route through the direct code."""
    p = _pipeline(_Site())
    p.v.fetch_engine = "crawl4ai"
    p.v.crawl4ai_url = ""  # forces the documented early return
    result = p.fetch_page("https://example.com/a", 10)

    assert result["status"] == "error"
    assert "crawl4ai_url" in result["detail"]


# ---------------------------------------------------------------------------
# SSRF: validate and fetch in one pass
# ---------------------------------------------------------------------------


def test_redirect_to_a_private_host_is_never_requested(patch_get, monkeypatch):
    """The property that makes this engine defensible.

    A hop that resolves to a private address must be refused *before* any
    request is made for it. Asserting only that the result is an error would
    pass even if the request had already gone out.
    """
    site = _Site(
        {
            "https://public.example/go": _Resp(status=302, location="http://127.0.0.1:8080/"),
            "http://127.0.0.1:8080/": _Resp(body=ARTICLE.encode()),
        }
    )
    patch_get(site)

    p = _research._Pipeline(_research.Tools.Valves())
    p.v.fetch_engine = "direct"
    # Real guard semantics: private hosts are refused.
    p.is_public_http_url = lambda url: "127.0.0.1" not in url

    result = p.fetch_page("https://public.example/go", 10)

    assert result["status"] == "error"
    assert "Blocked" in result["detail"]
    # The decisive assertion.
    assert "http://127.0.0.1:8080/" not in site.seen


def test_public_redirect_is_followed_and_reported(patch_get):
    site = _Site(
        {
            "https://a.example/x": _Resp(status=301, location="https://b.example/y"),
            "https://b.example/y": _Resp(body=ARTICLE.encode()),
        }
    )
    result = _pipeline(patch_get(site)).fetch_page("https://a.example/x", 10)

    assert result["status"] == "ok"
    assert result["url"] == "https://b.example/y"


def test_redirect_chain_is_bounded(patch_get):
    site = _Site(
        {"https://loop.example/": _Resp(status=302, location="https://loop.example/")}
    )
    result = _pipeline(patch_get(site)).fetch_page("https://loop.example/", 10)

    assert result["status"] == "error"
    assert "Too many redirects" in result["detail"]


def test_redirect_without_location_is_an_error(patch_get):
    site = _Site({"https://a.example/x": _Resp(status=302)})
    result = _pipeline(patch_get(site)).fetch_page("https://a.example/x", 10)
    assert "without a Location" in result["detail"]


# ---------------------------------------------------------------------------
# the quality gate
# ---------------------------------------------------------------------------


def test_client_rendered_shell_is_refused_with_measurements(patch_get):
    """The failure this engine must never have: success over an empty page."""
    site = patch_get(_Site({"https://example.com/app": _Resp(body=SHELL.encode())}))
    result = _pipeline(site).fetch_page("https://example.com/app", 10)

    assert result["status"] == "error"
    assert result.get("thin") is True
    assert "chars from" in result["detail"]
    assert "client-rendered" in result["detail"]


def test_navigation_page_is_refused_as_boilerplate(patch_get):
    p = _pipeline(_Site(), direct_min_chars=1)
    nav = "<html><body><ul>" + "".join(
        f"<li><a href='/p{i}'>Section {i} of the manual</a></li>" for i in range(40)
    ) + "</ul></body></html>"
    site = patch_get(_Site({"https://example.com/docs": _Resp(body=nav.encode())}))

    result = p.fetch_page("https://example.com/docs", 10)

    assert result["status"] == "error"
    assert "link text" in result["detail"]


def test_unsupported_content_type_is_refused_by_name(patch_get):
    """A PDF returning zero characters would look like a thin page."""
    site = patch_get(
        _Site({"https://example.com/f.pdf": _Resp(body=b"%PDF-1.4", content_type="application/pdf")})
    )
    result = _pipeline(site).fetch_page("https://example.com/f.pdf", 10)

    assert result["status"] == "error"
    assert "application/pdf" in result["detail"]


def test_markdown_passes_through_without_the_html_gate(patch_get):
    """A short markdown file is legitimate; the gate is an HTML concern."""
    body = "# Title\n\nShort note.\n"
    site = patch_get(
        _Site({"https://example.com/n.md": _Resp(body=body.encode(), content_type="text/markdown")})
    )
    result = _pipeline(site).fetch_page("https://example.com/n.md", 10)

    assert result["status"] == "ok"
    assert result["markdown"].startswith("# Title")


def test_script_and_style_never_reach_the_text(patch_get):
    page = (
        "<html><head><style>body{color:red}</style></head><body>"
        "<script>var x='MUST_NOT_APPEAR';</script>"
        "<p>" + ("Real body text. " * 40) + "</p></body></html>"
    )
    site = patch_get(_Site({"https://example.com/a": _Resp(body=page.encode())}))
    result = _pipeline(site).fetch_page("https://example.com/a", 10)

    assert "MUST_NOT_APPEAR" not in result["markdown"]
    assert "color:red" not in result["markdown"]
    assert "Real body text" in result["markdown"]


def test_download_cap_truncates_the_body(patch_get):
    p = _pipeline(_Site(), direct_max_bytes=100_000)
    p.v.direct_max_bytes = 300
    site = patch_get(_Site({"https://example.com/big": _Resp(body=ARTICLE.encode() * 20)}))
    result = p.fetch_page("https://example.com/big", 10)

    # Either the truncated body still yields usable text, or the gate refuses
    # it. Both are correct; an unbounded read is not.
    assert result["status"] == "error" or len(result["markdown"]) <= p.v.max_page_chars


# ---------------------------------------------------------------------------
# failures are surfaced, not logged and dropped
# ---------------------------------------------------------------------------


def test_thin_pages_are_reported_separately_from_fetch_failures(monkeypatch):
    """`thin` means "needs a browser"; a fetch failure means "this didn't load".

    Different remedies, so they must not be merged into one counter.
    """
    p = _research._Pipeline(_research.Tools.Valves())
    p.v.fetch_engine = "direct"
    p.v.use_cache = False
    p.v.persist = False

    monkeypatch.setattr(p, "search", lambda q, t: {
        "status": "ok", "count": 2, "used_engine": "searxng",
        "results": [
            {"title": "a", "url": "https://example.com/js", "content": "x"},
            {"title": "b", "url": "https://example.com/ok", "content": "y"},
        ],
    })
    monkeypatch.setattr(p, "fetch_page", lambda url, t: (
        {"status": "error", "detail": "Page too thin to use: extracted only 12 chars", "thin": True}
        if "js" in url
        else {"status": "ok", "markdown": "body", "engine": "direct"}
    ))

    import time
    out = p._run_inner("query about a thing", "deep", time.monotonic() + 30)

    assert out["source"] == "web_fetched"
    assert "thin_pages" in out
    assert out["thin_pages"] == ["https://example.com/js"]
    assert "note_thin" in out
    assert "fetch_failures" in out
