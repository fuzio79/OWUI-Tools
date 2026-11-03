# OWUI Tools

A small collection of [Open WebUI](https://github.com/open-webui/open-webui) tool plugins
(Python modules that give the model callable functions), plus regression suites that pin the
behaviour that was expensive to get right.

Each tool is a single self-contained file. There is no package to install and no build step —
you paste the file into Open WebUI and configure it through the Valves panel.

> Layout note: every `test_*.py` loads its tool by path, so a tool and its test must stay
> siblings in the repository root. That is why the tree is flat.

---

## Contents

| File | What it does |
|---|---|
| [`rag-openwebui-tool.py`](rag-openwebui-tool.py) | Search a self-hosted RAG service, and ingest URLs into it |
| [`research-openwebui-tool.py`](research-openwebui-tool.py) | Deep research: SearXNG/Tavily discovery, then crawl4ai page fetching, with optional use of the same RAG service as a cache |

| Test suite | Covers |
|---|---|
| [`test_rag_tool.py`](test_rag_tool.py) | API-key header handling, auth error reporting |
| [`test_research_rag_key.py`](test_research_rag_key.py) | The research tool's RAG auth, and graceful degradation on 401 |
| [`test_research_tool.py`](test_research_tool.py) | Discovery/fallback, `depth` semantics, SSRF guard, untrusted-content wrapping, cache-hit scoring |

---

## Install

1. In Open WebUI: **Workspace → Tools → +** (New Tool).
2. Paste the contents of one tool file, give it a name and description, save.
3. Open the tool's **Valves** panel and set the values described below.
4. Enable the tool for a model (or for all of them) in the model's configuration.

Two things that catch people out:

- **Open WebUI caches the tool source and its valves.** Editing the file on disk changes
  nothing — re-import it. After re-importing, you may need to toggle the tool off and on
  before the Valves panel re-renders.
- **The valves are read per call, not at import.** Changing a valve takes effect on the next
  tool call, without a restart.

---

## `rag-openwebui-tool.py`

Two functions exposed to the model:

- `rag_search(query)` — semantic search over the knowledge base.
- `ingest_url(url, engine="auto")` — scrape, chunk, embed and store a URL.

It expects a RAG HTTP service exposing `POST /api/v1/rag/search` and `POST /api/v1/rag/ingest`.
The request bodies it sends are:

```jsonc
// /api/v1/rag/search
{ "query": "...", "top_k": 3, "max_context_chars": 4000, "max_chunk_chars": 1200 }

// /api/v1/rag/ingest
{ "url": "https://...", "engine": "auto", "follow_links": false, "max_follow": 20 }
```

Any service that speaks that shape will work; this repository does not ship one.

### Valves

| Valve | Default | Notes |
|---|---|---|
| `rag_server_host` | `http://localhost:8000` | Base URL of the RAG service |
| `api_key` | `""` | Sent as `X-API-Key`. Empty sends **no header at all** |
| `follow_links` | `false` | Also ingest same-site links found on the page |
| `max_follow` | `20` | 0–50 |
| `top_k` | `3` | 1–20 |
| `max_context_chars` | `4000` | 500–20000 |
| `max_chunk_chars` | `1200` | 200–5000 |
| `search_timeout_s` | `30` | 5–120 |
| `ingest_timeout_s` | `300` | 30–900; should exceed the server's scrape + link-follow budget |

### Auth

If the service runs with authentication enabled, set `api_key` to the service's shared key.
The tool distinguishes the failure modes rather than reporting a generic connection error:

| Server response | What the tool tells you |
|---|---|
| `401` | Your `api_key` valve does not match the server's key |
| `503` | The server has no key configured (it is fail-closed) |
| `5xx` / connection error | Reported as a server fault, not as a configuration problem |

An empty or whitespace-only `api_key` sends no `X-API-Key` header, so the tool keeps working
against a service with authentication disabled.

---

## `research-openwebui-tool.py`

One function:

```python
async def research(query: str, depth: str = "auto")
```

`depth` controls how much work is allowed:

| `depth` | Behaviour |
|---|---|
| `"quick"` | Discovery only — returns search snippets, never fetches a page |
| `"deep"` | **Always** fetches pages, even when a snippet would look sufficient |
| `"auto"` *(default)* | Answers from snippets when they are long enough, otherwise fetches. Also fetches when the query contains a deep-signal keyword |

A cache lookup against the RAG service runs first (unless `depth="deep"` or `use_cache` is
false). A hit must clear `min_score`; otherwise it is treated as a miss and the web is used.

**Requires at least one discovery engine:** a SearXNG instance with the JSON API enabled, or a
Tavily API key. Configuring both gives automatic fallback — the secondary engine is used when
the primary fails or returns nothing, and the result reports which engine answered.

### Valves (grouped as in the Valves panel)

**Engine selection**

| Valve | Default | Notes |
|---|---|---|
| `default_engine` | `searxng` | `searxng` or `tavily`; the other becomes the fallback |

**SearXNG** — `searxng_url` (`http://searxng:8080`), `num_results` (10), `time_range` (`""`),
`categories` (`""`), `searxng_timeout` (15), `max_text_chars` (1200)

**Tavily** — `tavily_api_key` (`""`), `tavily_depth` (`basic`), `tavily_timeout` (15)

**Page fetching (crawl4ai)** — `crawl4ai_url` (`http://crawl4ai:11235/crawl`),
`crawl4ai_token` (`""`), `crawl4ai_timeout` (60), `max_page_chars` (8000)

**SSRF guard** — `dns_timeout` (5)

**RAG service** — `rag_server_host` (`http://localhost:8000`), `rag_api_key` (`""`),
`top_k` (3), `max_context_chars` (4000), `max_chunk_chars` (1200), `min_score` (0.35),
`distance_metric` (`cosine`), `follow_links` (`false`), `max_follow` (10), `ingest_timeout` (120)

**Decision policy** — `use_cache` (`true`), `persist` (`false`), `fetch_budget` (3),
`snippet_min_chars` (300)

**Global limits** — `total_context_chars` (100000), `total_timeout` (180)

Note that **`rag_api_key` and `api_key` are different valves in different tools.** If both tools
point at the same RAG service, both need the same key.

The default hostnames (`searxng:8080`, `crawl4ai:11235`) assume those services are reachable
under those names — typically a Docker network. Set them to whatever URL you actually use.

---

## Tests

Standard library `unittest`; no test framework to install. HTTP is stubbed, so no live service
is needed and nothing touches the network.

```bash
pip install requests pydantic
python -m unittest test_rag_tool test_research_rag_key test_research_tool -v
```

63 tests. Use `python -m unittest`, not a bare `unittest`, so the repository root stays on
`sys.path`.

---

## Security notes

These are LLM-facing plugins, so the model chooses the arguments. Read this before exposing
them.

- **Fetched content is untrusted.** Text pulled from the web is wrapped in a delimiter block
  with delimiter tags inside it neutralised, so a page cannot close the wrapper and inject
  instructions as if they came from the system. It is a mitigation, not a guarantee — a model
  can still be persuaded by what it reads inside the block.
- **The SSRF guard is a best-effort pre-check.** The research tool validates `http`/`https`
  only, ports 80/443 only, and requires every resolved IP to be public, with a timeout on the
  DNS lookup. It cannot see redirects followed by crawl4ai or by the RAG service, nor a DNS
  answer that changes after the check (rebinding). **For real protection, restrict egress at
  the network level** on the containers doing the fetching.
- **`follow_links` delegates validation.** When enabled, links found on a page are fetched by
  the RAG service and are *not* re-validated by the tool. Leave it off unless you trust the
  targets and the network policy around the service.
- **API keys travel as a plain header** over whatever transport you configure. Use HTTPS if the
  service is not on a trusted network.
- **No key is ever sent when the valve is empty**, so a misconfiguration fails toward "no
  credential sent" rather than "credential sent somewhere unexpected".

---

## License

MIT — see [LICENSE](LICENSE).
