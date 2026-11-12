# OWUI Tools

A collection of [Open WebUI](https://github.com/open-webui/open-webui) tool plugins
(Python modules that give the model callable functions), plus regression suites that pin the
behaviour that was expensive to get right.

Each tool is a single self-contained file. There is no package to install and no build step —
you paste the file into Open WebUI and configure it through the Valves panel.

> Layout note: every `test_*.py` loads its tool **by path**, resolving it relative to its own
> file. A tool and its test must therefore stay siblings in the repository root. That is why the
> tree is flat, and CI asserts it.

---

## Contents

### Tools

| File | What it does | Functions |
|---|---|---|
| [`docker-exe-openwebui-tool.py`](docker-exe-openwebui-tool.py) | Manage explicitly-trusted Docker containers through the [`docker-exe`](https://github.com/fuzio79/docker-exe) service | `docker_list`, `docker_stats`, `docker_logs`, `docker_exec`, `docker_start`, `docker_restart`, `docker_stop`, `docker_remove` |
| [`dockhand-openwebui-tool.py`](dockhand-openwebui-tool.py) | **Read-only** container/stack inspection via the Dockhand API, with secret redaction | `list_environments`, `list_containers`, `inspect_container`, `container_logs`, `list_stacks`, `get_stack_compose`, `container_stats` |
| [`portainer-openwebui-tool.py`](portainer-openwebui-tool.py) | **Read-only** container/stack inspection via the Portainer API | `list_endpoints`, `list_containers`, `inspect_container`, `container_logs`, `list_stacks`, `get_stack_compose` |
| [`pi-agent-openwebui-tool.py`](pi-agent-openwebui-tool.py) | Run the Pi coding agent on coding tasks, one-shot or in a **session that remembers the conversation** | `pi_run`, `pi_sessions`, `pi_session_close`, `pi_health` |
| [`rag-openwebui-tool.py`](rag-openwebui-tool.py) | Search a self-hosted RAG service, ingest URLs into it, read a page **without** storing it, and remove stored content | `rag_search`, `ingest_url`, `fetch_url`, `delete_source` |
| [`research-openwebui-tool.py`](research-openwebui-tool.py) | Deep research: SearXNG/Tavily discovery, then crawl4ai page fetching, with optional use of the RAG service as a cache | `research` |
| [`searx-search.py`](searx-search.py) | Minimal single-shot SearXNG search | `searxng_search` |
| [`nextcloud-calendar-calling.py`](nextcloud-calendar-calling.py) | List and create Nextcloud calendar events over CalDAV | `list_upcoming_events`, `create_calendar_event` |

### Test suites

| Suite | Tool | Covers |
|---|---|---|
| [`test_docker_exec_tool.py`](test_docker_exec_tool.py) | docker-exe | Trust model, allowlist, error mapping, `cmd` argv parsing and shape rejection |
| [`test_dockhand_tool.py`](test_dockhand_tool.py) | dockhand | Secret redaction, response shapes |
| [`test_pi_agent_tool.py`](test_pi_agent_tool.py) | pi-agent | Allowlist enforcement, session ids, status-code mapping |
| [`test_rag_tool.py`](test_rag_tool.py) | rag | API-key header handling, auth error reporting |
| [`test_rag_tool_direct.py`](test_rag_tool_direct.py) | rag | `engine="direct"` error surfacing and retry hints |
| [`test_rag_tool_fetch_delete.py`](test_rag_tool_fetch_delete.py) | rag | `fetch_url` non-persistence, `delete_source` dry-run default, timeout and invalid-JSON paths |
| [`test_research_tool.py`](test_research_tool.py) | research | Discovery/fallback, `depth` semantics, SSRF guard, untrusted-content wrapping, cache-hit scoring |
| [`test_research_rag_key.py`](test_research_rag_key.py) | research | RAG auth, graceful degradation on 401 |
| [`test_research_ingest_v2.py`](test_research_ingest_v2.py) | research | Ingest request shape |
| [`test_research_ingest_errors.py`](test_research_ingest_errors.py) | research | Per-link skip/fail reasons, persist failures |
| [`test_research_direct_fetch.py`](test_research_direct_fetch.py) | research | The plain-HTTP fetch engine and its thin-page quality gate |
| [`test_research_fetch_engine_param.py`](test_research_fetch_engine_param.py) | research | The per-call `fetch_engine` parameter |
| [`test_research_fetch_engine_leak.py`](test_research_fetch_engine_leak.py) | research | That a per-call override does **not** persist to the shared valves object |
| [`test_research_soft_deadline.py`](test_research_soft_deadline.py) | research | `soft_deadline` rename and the `soft_deadline_reached` report |
| [`test_tools_async_guard.py`](test_tools_async_guard.py) | all | No tool blocks the Open WebUI event loop: every public method is `async`, and every blocking call is behind `asyncio.to_thread` |

Not every tool has a dedicated suite. `portainer-openwebui-tool.py`, `searx-search.py` and
`nextcloud-calendar-calling.py` currently do not, but `test_tools_async_guard.py` covers
all of them for the event-loop property, which is the one defect in this repo that takes
the whole UI down rather than failing a single tool call.

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

## `docker-exe-openwebui-tool.py`

Manages containers that the `docker-exe` service has explicitly trusted. The service is not a
general Docker API proxy: a container is only visible to it when it carries a `managed=true`
label **and** a `managed.signature` that matches an HMAC over its full container ID, so a
container cannot promote itself by setting a label.

Tools that mutate state (`docker_start`, `docker_restart`, `docker_stop`, `docker_remove`,
`docker_exec`) act only on containers that pass that check. Everything else is read-only.

| Valve | Default | Notes |
|---|---|---|
| `server_url` | `http://localhost:8091` | Base URL of the docker-exe service |
| `api_key` | `""` | Sent as `X-API-Key`. Empty sends no header |
| `default_log_lines` | `200` | 10–5000 |
| `default_exec_timeout` | `30` | 1–600 |
| `confirm_exec` | `true` | Ask for confirmation in the UI before every `docker_exec` |
| `loop_guard_window` | `30` | Sliding window (seconds) for the loop guard |
| `loop_guard_max_repeats` | `3` | Identical calls within the window before short-circuiting |

### `cmd` must be one element per word

`cmd` is an **argv list**, not a command line. `docker-exe` runs no shell, so nothing
re-splits it:

```jsonc
["echo", "probe-ok"]        // correct: runs echo with one argument
["echo probe-ok"]            // WRONG: looks for a binary literally named "echo probe-ok"
```

The second form fails with `exit_code 127` and
`timeout: failed to run command 'echo probe-ok': No such file or directory`. The `timeout:`
prefix comes from the service's kill-wrapper, and is not evidence that the arguments were
joined — there was only ever one argument, and the message reproduces it unchanged.

The tool now refuses that shape locally, with a message naming the corrected form, rather than
forwarding it. For a pipe or redirect, pass three elements:
`["sh", "-c", "ps aux | grep node"]`.

A JSON-string array containing **literal** newlines or tabs is accepted. Models paste heredoc
bodies with real linebreaks, and `json.loads` rejects raw control characters unless told not to,
so those calls used to fail with `could not be parsed` — a message that told the caller to do
what it had already done. Structural errors (trailing commas, single quotes, unbalanced quotes)
are still refused, as is the single-element-with-spaces shape above.

`cmd` also defaults to `None`, so a model that emits only `name` gets an actionable error.
Previously that raised
`Tools.docker_exec() missing 1 required positional argument: 'cmd'` inside Open WebUI, before
the function body ran — the model saw the tool vanish, and no request ever reached the service.

`server_url` must be reachable **from the Open WebUI container**. If the service runs on the
host, `localhost` resolves to the Open WebUI container itself — use the Docker host's LAN
address instead.

---

## `dockhand-openwebui-tool.py` and `portainer-openwebui-tool.py`

Both are **read-only** inspection tools for a container-management UI. Neither can start, stop
or remove anything. They exist separately because the two APIs differ in auth
(`Bearer` token for Dockhand, `X-API-Key` for Portainer) and in response shape.

They share most valves but not all - the names are UPPER CASE, so guessing from one tool
to the other gets you the wrong name (and `API_TOKEN` is the credential one).

| Valve | Dockhand | Portainer | Notes |
|---|---|---|---|
| `DOCKHAND_URL` | `http://dockhand:3000` | - | Base URL, no trailing slash |
| `PORTAINER_URL` | - | `http://portainer:9000` | Base URL, no trailing slash |
| `API_TOKEN` | empty | empty | Bearer token (Dockhand) / access token (Portainer) |
| `ENV_ID` | `1` | - | Dockhand environment id; discover via `list_environments` |
| `ENDPOINT_ID` | - | `1` | Portainer endpoint id containing the Docker host |
| `VERIFY_TLS` | `true` | `true` | Verify the TLS certificate; keep on for HTTPS |
| `TIMEOUT_SECONDS` | `20` | `20` | Normal request timeout |
| `STATS_TIMEOUT_SECONDS` | `45` | - | `container_stats` is slow on busy hosts |
| `MAX_LOG_LINES` | `300` | `300` | Log lines returned |
| `MAX_LOG_LINE_CHARS` | `2000` | - | Characters kept per log line |
| `MAX_OUTPUT_CHARS` | `20000` | - | Size cap on JSON returned to the model |
| `REDACT_SECRETS` | `true` | - | Redact env vars and compose values whose key looks secret |

Both redact secret-looking values before returning them. Portainer has no redaction valve,
so limit what it can see with network policy instead.

---

## `pi-agent-openwebui-tool.py`

Runs the Pi coding agent through the `pi-agent-runner` service. Two modes, and the difference
matters:

- **No `session_id`** — a fresh container per call. Nothing carries over.
- **Same `session_id` reused** — same container, same `/workspace`, **and the same conversation
  history**, so follow-ups like "now add a test for that" work.

Sessions are the feature most often missed: an LLM that never passes `session_id` gets a
stateless agent and will re-explain itself every turn.

| Valve | Default | Notes |
|---|---|---|
| `runner_host` | `http://pi-agent-runner:8092` | Use the container name, not a host IP |
| `default_timeout` | `300` | 10–1800 |

Tool arguments include `task`, `session_id`, `timeout`, `force`, and optionally
`provider`/`model` — but only the provider/model pairs the runner's allowlist permits, and both
must be given together. The tool rejects anything else locally with the permitted set named,
rather than letting the runner bounce it.

Up to 5 sessions may be open at once; sessions expire after 30 minutes idle or 24 hours.

---

## `rag-openwebui-tool.py`

Four functions:

- `rag_search(query)` — semantic search over the knowledge base.
- `ingest_url(url, engine="auto")` — scrape, chunk, embed and store a URL.
- `fetch_url(url, query, engine="auto", top_k=3)` — read **one** page and answer
  a question from it, **storing nothing**. Only the passages relevant to
  `query` are returned, so one fact from a long document costs a few paragraphs
  instead of the whole page.
- `delete_source(source, dry_run=True)` — remove everything stored for a URL or
  uploaded file. **`dry_run` defaults to `True`**, so a call reports what would
  be removed and changes nothing; the model must be told to confirm before
  anything is deleted.

It expects a RAG HTTP service exposing `POST /api/v1/rag/search`,
`POST /api/v1/rag/ingest`, `POST /api/v1/rag/fetch` and `POST /api/v1/rag/delete`:

```jsonc
// /api/v1/rag/search
{ "query": "...", "top_k": 3, "max_context_chars": 4000, "max_chunk_chars": 1200 }

// /api/v1/rag/ingest
{ "url": "https://...", "engine": "auto", "follow_links": false, "max_follow": 20 }

// /api/v1/rag/fetch   -> returns { stored: false, passages_returned, context }
{ "url": "https://...", "query": "what is the refund window", "engine": "auto", "top_k": 3 }

// /api/v1/rag/delete   -> returns { chunks_deleted } or, with dry_run, { chunks_to_delete }
{ "source": "https://...", "dry_run": true }
```

Any service speaking that shape works; [`fuzio79/rag-middleware`](https://github.com/fuzio79/rag-middleware)
is one. The two newer functions degrade cleanly against a service that lacks
those routes: they return an error naming the endpoint rather than raising.

| Valve | Default | Notes |
|---|---|---|
| `rag_server_host` | `http://localhost:8000` | Base URL of the RAG service. **This default only works if Open WebUI and the RAG service share a network namespace** — when they are separate containers, set it to the service name (e.g. `http://rag-service:8000`) |
| `api_key` | `""` | Sent as `X-API-Key`. Empty sends **no header at all** |
| `follow_links` | `false` | Also ingest same-site links found on the page |
| `max_follow` | `20` | 0–50 |
| `top_k` | `3` | 1–20 |
| `max_context_chars` | `4000` | 500–20000 |
| `max_chunk_chars` | `1200` | 200–5000 |
| `search_timeout_s` | `30` | 5–120 |
| `ingest_timeout_s` | `300` | 30–900; should exceed the server's scrape + link-follow budget. Also used as the timeout for `fetch_url` and `delete_source` |

### Auth

The tool distinguishes failure modes rather than reporting one generic connection error:

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

**Engine selection** — `default_engine` (`searxng`; `searxng` or `tavily`, the other becomes
the fallback)

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

Default hostnames (`searxng:8080`, `crawl4ai:11235`, `pi-agent-runner:8092`, `dockhand:3000`,
`portainer:9000`) assume those services resolve under those names — typically a Docker network.
Set them to whatever URL you actually use.

---

## `searx-search.py` and `nextcloud-calendar-calling.py`

**`searx-search.py`** is a minimal SearXNG search: one function, `searxng_search`, and six
valves. Use it when you want raw search results without the research tool's decision policy.

| Valve | Default | Notes |
|---|---|---|
| `searxng_url` | `http://searxng:8080` | Base URL of the SearXNG instance |
| `num_results` | `10` | Maximum results returned |
| `time_range` | empty | Freshness filter: `day`, `week`, `month`, `year` |
| `categories` | empty | Comma-separated SearXNG categories |
| `timeout` | `15` | Request timeout in seconds |
| `max_text_chars` | `1200` | Characters per result title and content |

**`nextcloud-calendar-calling.py`** talks CalDAV to Nextcloud to list and create events. It is
the only tool with extra dependencies:

```bash
pip install caldav icalendar
```

| Valve | Default | Notes |
|---|---|---|
| `nextcloud_url` | `""` | Full CalDAV calendar URL, e.g. `https://cloud.example.com/remote.php/dav/calendars/username/` |
| `nextcloud_user` | `""` | Nextcloud username |
| `nextcloud_pass` | `""` | App password |
| `calendar_name` | empty | Optional calendar name; blank uses the first returned |
| `default_timezone` | `Europe/Warsaw` | IANA zone name; change it |
| `reminder_minutes` | `30` | Minutes before the event to remind; `0` disables |

It has no test suite, unlike most of the others.

---

## Tests

HTTP is stubbed in every suite, so no live service is needed and nothing touches the network.

There are **15 suites in two runner families**, and the distinction matters because it decides
the command you have to type:

| Runner | Suites | Requires |
|---|---|---|
| `unittest` (stdlib) | 7 | `requirements.txt` only |
| `pytest` | 8 | `pip install pytest` |

### The `unittest` suites — 215 tests

```bash
pip install -r requirements.txt
python -m unittest test_docker_exec_tool test_dockhand_tool test_pi_agent_tool \
                   test_rag_tool test_research_rag_key test_research_ingest_v2 \
                   test_research_tool -v
```

Use `python -m unittest`, not a bare `unittest`, so the repository root stays on `sys.path`.

### The `pytest` suites — 100 tests

The eight newest suites use `pytest` and **`unittest` cannot collect them**: their
tests are module-level `def test_*` functions, which `unittest` ignores entirely (it only looks
at `TestCase` subclasses). Four of them also use `pytest.fixture`.

```bash
pip install -r requirements.txt pytest
python -m pytest test_rag_tool_direct.py test_rag_tool_fetch_delete.py \
                 test_research_direct_fetch.py test_research_fetch_engine_leak.py \
                 test_research_fetch_engine_param.py test_research_ingest_errors.py \
                 test_research_soft_deadline.py test_tools_async_guard.py -q
```

Or run both together with `python -m pytest -q`, which collects all 15 and reports **315**
(`pytest` runs `TestCase`-style tests too):

```
315 passed in 1.9s
```

> **`python -m unittest discover` silently under-reports.** It does *not* error on the pytest
> modules — it imports them, finds no `TestCase`, and reports `Ran 215 tests ... OK`. A green run
> that skipped a third of the tree. Prefer `python -m pytest -q`, which cannot fail this way.

> **CI runs both halves**, as two explicit steps: the seven `unittest` suites and the eight
> `pytest` suites. A third step asserts that every `test_*.py` on disk is named in the workflow,
> so a new suite cannot be added without being run. That guard exists because seven suites once
> sat unexecuted while every push reported green.

### Tools must not block the event loop

Open WebUI calls a tool from an async route, but it does **not** offload a sync tool to a
thread. `get_async_tool_function_and_apply_extra_params` branches on
`inspect.iscoroutinefunction` and, for a plain `def`, calls it inline:

```python
if inspect.iscoroutinefunction(function):
    async def new_function(*args, **kwargs):
        return await partial_func(*args, **coerce_kwargs(kwargs))
else:
    async def new_function(*args, **kwargs):
        return partial_func(*args, **coerce_kwargs(kwargs))   # <- ON the loop
```

So a `def` tool that calls `requests.post(...)` blocks that event loop for the whole HTTP
call. uvicorn runs a single worker here, and that one loop also serves every other request
and every chat token stream, so a single long call freezes the entire UI for every user
until it returns, and the Docker healthcheck cannot answer either.

This was a real defect, not a theory. Every tool in this repo used plain `def` methods, and
`pi_run` waited `timeout + 10` seconds, **600 s by default and 1810 s at the maximum**. A
real Pi run froze Open WebUI for minutes. The suite was green the whole time: every test
stubbed `requests` and asserted on the returned value, and nothing measured the loop.

Every public tool method is now `async def`, and every blocking call goes through
`asyncio.to_thread`. Two shapes are acceptable:

```python
# 1. the blocking library call is the to_thread argument
resp = await asyncio.to_thread(requests.post, url, json=payload, timeout=t)

# 2. a private SYNC helper holds the blocking half; the async method offloads it
def _list_sync(self, days_ahead):   # may block freely
    ...

async def list_events(self, days_ahead=7):
    return await asyncio.to_thread(self._list_sync, days_ahead)
```

What is **not** acceptable: an `async def` that calls a sync helper inline, or a blocking
call made directly in an async body. That only moves the block one frame deeper while still
running on the loop. [`test_tools_async_guard.py`](test_tools_async_guard.py) fails on both,
and its last assertion is dynamic: a 20 ms heartbeat runs alongside a tool call whose HTTP
request sleeps 300 ms and must keep ticking. Verified by mutation: reverting `pi_run` to the
old shape makes that test report a **0.32 s stall**, and it is the only check that does.

> **`async def` alone is not a fix.** A coroutine that calls `requests.post` without
> `to_thread` blocks exactly as much as a `def` did, which is why the guard checks both the
> method kind *and* the call shape.

### CI also checks three things the unit tests cannot

- **Compile check** — `python -m py_compile *.py`. These files are pasted into Open WebUI by
  hand, so a syntax error would reach every user.
- **Layout guard** — asserts every `test_*.py` references a tool file that sits beside it.
  Moving a tool into a subdirectory breaks its suite silently otherwise.
- **[`scripts/check_valve_defaults.py`](scripts/check_valve_defaults.py)** — parses each tool
  with `ast` and fails if any valve default is a credential, a private address, or an
  internal-looking hostname. A valve default ships to strangers, so this is the guard that
  stops a working config from leaking into a public repo.

  It uses `ast` rather than grep on purpose: credentials are declared as multi-line
  `Field(...)` calls, which a line-oriented regex misses while reporting "clean".

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
- **`docker_exec` is remote command execution, by design.** It is confined to containers the
  service has explicitly trusted, but that is a trust boundary, not a sandbox. Treat the
  runner and its API key accordingly.
- **The Pi sandbox is not an isolation boundary.** Sandboxes run as root, unconstrained, and
  on the same network as your other services; the provider key is readable inside them. See
  [`fuzio79/pi-agent-runner`](https://github.com/fuzio79/pi-agent-runner) for the full posture.
- **API keys travel as a plain header** over whatever transport you configure. Use HTTPS if
  the service is not on a trusted network.
- **No key is ever sent when the valve is empty**, so a misconfiguration fails toward "no
  credential sent" rather than "credential sent somewhere unexpected".

---

## License

MIT — see [LICENSE](LICENSE).
