"""
title: Pi Coding Agent
author: fuzio
version: 1.6.0
description: Run the Pi coding agent on coding tasks — one-shot, or in a persistent session that remembers the conversation. The agent can delegate to subagents (reviewer, verifier, programmer, oracle).
requirements: requests, pydantic
"""

import asyncio
import logging
import re
import requests
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pi_agent_tools")

# ── Wire contract with pi-agent-runner (v1.1.8+) ──────────────────────────
#
# These regexes mirror the runner's `_SESSION_ID_RE`, `_PROVIDER_RE` and
# `_MODEL_RE` EXACTLY. That is deliberate: a local rejection costs nothing and
# returns a precise message, while a mismatch here silently blocks a request the
# runner would have accepted.
#
# `provider` and `model` are NOT interchangeable. Provider ids are flat
# (`opencode-go`), but model ids may carry a vendor prefix with a slash. A
# shared pattern without `/` previously rejected every slash-bearing model
# before it was ever sent.
#
# All three require an alphanumeric first character: a leading `-` would make
# the value flag-shaped, and `--provider -x` is malformed even though argv
# cannot inject a shell command.
#
# The session-id rule is pi's own, quoted from the error it emits: non-empty,
# alphanumeric plus '-', '_', '.', and must start AND end alphanumeric. The
# runner turns this into a path component under ~/.pi/agent/sessions/, so `..`
# must not match — and it cannot, because it does not start alphanumeric.
_SESSION_ID_RE = r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,62}[A-Za-z0-9])?$"
_PROVIDER_RE = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$"
_MODEL_RE = r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$"

# Repository slug, owner/name. Mirrors the runner's `_REPO_RE`. Each segment
# starts alphanumeric, so `..` cannot match: the value becomes both a path
# component (/workspace/<name>) and part of a clone URL on the runner side, so
# a traversal-shaped value is rejected in both places.
_REPO_RE = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}/[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$"

# The runner's hard task cap (MAX_TASK_CHARS). Checked here so an oversized
# task fails with a clear message instead of a 422 the tool would otherwise
# have to interpret.
_MAX_TASK_CHARS = 128 * 1024

# Runner-side caps, mirrored for local validation.
_MAX_CONCURRENT_SESSIONS = 5
_MAX_TIMEOUT = 1800

# The runner's provider/model ALLOWLIST. This mirrors
# `_ALLOWED_PROVIDERS` / `_ALLOWED_MODELS` in the runner's main.py.
#
# Kept here so a stale model name fails locally with the permitted set named,
# rather than being sent and bounced with a 422. The runner is still the
# authority — this is a fast, clearer-message copy, not a replacement.
#
# If these two drift, the runner wins and you get its message. Update both IN
# THE SAME COMMIT: adding a provider to the runner without adding it here makes
# this tool reject a request the runner would have accepted, and the error will
# name a provider that is, in fact, allowed.
_ALLOWED_PROVIDERS = ("opencode-go", "commandcode")
_ALLOWED_MODELS: Dict[str, tuple] = {
    "opencode-go": ("deepseek-v4.1-flash", "glm-5.3-flash"),
    # Command Code (GOAT). Model ids carry a vendor slash here, which is why
    # _MODEL_RE admits '/'. Two providers, ONE model: the choice is about
    # quota and rate-limit pools, never output quality.
    "commandcode": ("deepseek/deepseek-v4.1-flash",),
}
_DEFAULT_PROVIDER = "opencode-go"
_DEFAULT_MODEL = "deepseek-v4.1-flash"

# Where the subagent runtime writes full child transcripts. NOT under
# /workspace, so it is NOT covered by the session's working files:
#
#   one-shot run  -> the container is removed after logs are read, so the
#                    artifact is destroyed unless it was copied out first
#   session run   -> survives while the container lives (30 min idle / 24 h cap)
#
# The child's verdict is still in `stdout`; this path only matters when the
# full reasoning is wanted. Documented rather than surfaced as a method because
# there is no runner endpoint to read it — asking the agent to copy the file
# into /workspace is the supported route.
_ARTIFACT_DIR = "/root/.pi/agent/sessions/--workspace--/subagent-artifacts/"


class Tools:
    """
    Pi coding agent tools.

    Pi is a coding agent: it reads, writes and runs code inside a sandbox with
    its own /workspace. It is not a general assistant — for explanations or
    questions, answer directly.

    Two features are most often missed:

    1. Sessions. Passing a session_id keeps BOTH the files in /workspace AND
       the conversation history, so follow-up tasks build on what the agent
       already did.
    2. Delegation. The agent can spawn focused subagents with their own context
       windows. It decides when to do this; you do not need to request it, but
       a task that says "review it after implementing" gets more out of it than
       one that does not.
    """

    class Valves(BaseModel):
        runner_host: str = Field(
            default="http://pi-agent-runner:8092",
            description="Base URL of the pi-agent-runner service. "
                        "Use the container name (same Docker network) rather than "
                        "the host IP, which is not routable from inside the container network.",
        )
        default_timeout: int = Field(
            default=600,
            ge=10,
            le=1800,
            description="Default timeout in seconds for a Pi run (10-1800).",
        )

    def __init__(self):
        self.valves = self.Valves()
        self._run_endpoint = f"{self.valves.runner_host}/pi/run"
        self._session_close_endpoint = f"{self.valves.runner_host}/pi/session/close"
        self._sessions_endpoint = f"{self.valves.runner_host}/pi/sessions"

    # ── RUN ────────────────────────────────────────────────────────────

    async def pi_run(
        self,
        task: str,
        session_id: Optional[str] = None,
        timeout: int = 0,
        force: bool = False,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        repo: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the Pi coding agent on a coding task and return its output.

        Use this for CODING work: write, modify, debug or run code, scripts,
        programs, or any multi-step file work. Do NOT use it for general
        questions, explanations or non-coding requests — answer those directly.

        SESSION MODE (strongly preferred for related tasks): pass the same
        session_id on every call that belongs to the same piece of work. The
        agent then keeps BOTH /workspace files AND its conversation history, so
        follow-ups like "now add a test for that" or "why did that fail?" work.
        Without a session_id each call is a fresh container that remembers
        nothing. Use pi_sessions to see open sessions and pi_session_close when
        the work is done.

        DELEGATION: the agent can spawn subagents with their own fresh context
        windows. You do not command this — the agent decides — but you can ask
        for it in plain language ("implement it, then have a reviewer check it"),
        and you should expect that phrasing to take longer and produce more text.
        Known roles:

            scout      fast read-only recon: relevant files, entry points, risks
            reviewer   read-only judgement; CANNOT run commands by design
            verifier   runs the project's checks with bash; cannot edit source
            programmer executes an approved plan, then verifies its own work
            oracle     an expensive second opinion on a different model

        Two distinctions worth knowing when reading a result:
          - `reviewer` reasons about whether a test *should* change; `verifier`
            actually runs it. A reviewer will report a command "a supervisor
            must run" rather than running it.
          - A delegated run is several agent loops, not one. See `timeout`.

        Args:
            task: The coding task for the agent. Up to 128 KiB. Describe the
                  goal and any files involved; the agent can read and write in
                  its own /workspace across the session.
            session_id: Recommended. 1-64 chars of letters, digits, '-', '_',
                        '.', starting and ending with a letter or digit.
                        Reusing an id continues that session's files AND
                        conversation; a new id starts fresh work. Pick a short
                        stable name per piece of work, e.g. "refactor-auth".
                        Omit only for a genuinely one-off task.
                        NOTE: one-shot runs (no session_id) destroy the
                        container after reading logs, so subagent artifact
                        files are lost. Use a session when you want them.
            timeout: Max seconds before the run is abandoned. 0 uses the valve
                     default (600). Range 10-1800. A run that delegates is
                     several agent loops deep — a review-fix-review cycle can
                     take well over five minutes, so do not set this tightly.
            force: Bypass the heuristic that blocks obviously non-coding tasks.
                   Set True when you are confident the task is coding.
            provider: Optional. Must be given together with model. Two are
                      permitted: "opencode-go" and "commandcode". Both serve
                      the same DeepSeek model, so the choice only affects
                      which subscription is billed -- never the answer.
                      Prefer leaving both unset and letting the runner default.
            model: Optional. Must be given together with provider.
                   "opencode-go": "deepseek-v4.1-flash" (default) or
                   "glm-5.3-flash". "commandcode":
                   "deepseek/deepseek-v4.1-flash". GLM costs a separate
                   allowance -- request it deliberately, not routinely.
            repo: Optional. Repository slug, "owner/name", e.g.
                  "LLM/pi-agent-runner". REQUIRES session_id -- cloning is
                  session-mode only. The runner clones it into
                  /workspace/<name> BEFORE the agent starts, so the code is
                  already there on the first tool call; ask the agent about
                  files in that directory, do not ask it to clone.
                  Only allowlisted slugs are accepted (a server-side
                  deployment setting). An unlisted slug is rejected; the
                  error does not reveal which slugs are permitted.
                  Read-only token: the agent can clone and read, not push.

        Returns:
            On success: {"exit_code": 0, "stdout": "...", "stderr": "",
                         "duration_s": ..., "timed_out": ...}
                exit_code is the agent's exit status; timed_out is true only
                when the run was killed at its deadline.
                `stdout` contains the agent's final answer. When it delegated,
                that includes the child's returned report, but the child's full
                transcript is a separate artifact file (see below) and is NOT
                included.
            {"status": "timeout"}      the run exceeded its deadline
            {"status": "capacity"}     all 5 session slots are in use — close
                                       an idle session and retry; the backend
                                       is fine
            {"status": "busy"}         that session_id is already running a
                                       task — retry when it finishes
            {"status": "invalid"}      bad provider/model/task/session_id; the
                                       detail names what is allowed
            {"status": "error"}        the run failed, or the backend is
                                       unreachable (detail says which)
            {"status": "not_dispatched"} non-coding task, blocked before
                                       dispatch; answer directly or force=True

        Subagent artifacts: a child's full output is written to
        /root/.pi/agent/sessions/--workspace--/subagent-artifacts/<uuid>.md,
        which is OUTSIDE /workspace. To keep one, ask the agent in a later turn
        of the same session to copy it into /workspace, or read it before the
        session is closed.
        """
        # ── Intent gate ────────────────────────────────────────────────
        if not force:
            intent = self._classify(task)
            if intent != "coding":
                logger.info(
                    "pi_run short-circuited: intent=%s, task=%r", intent, task[:80]
                )
                return {
                    "status": "not_dispatched",
                    "intent": intent,
                    "detail": "Task classified as non-coding. Answer directly.",
                }

        # ── Validate task ──────────────────────────────────────────────
        # The runner caps this at 128 KiB because the task becomes ONE argv
        # element. Above the kernel's ARG_MAX the container create raises
        # OSError/E2BIG, which the runner reports as a 500.
        if not isinstance(task, str) or not task.strip():
            return {"status": "invalid", "detail": "task must be a non-empty string."}
        if len(task) > _MAX_TASK_CHARS:
            return {
                "status": "invalid",
                "detail": (
                    f"task is {len(task)} chars; the runner accepts at most "
                    f"{_MAX_TASK_CHARS}. Inline less and write files from the sandbox instead."
                ),
            }

        # ── Validate session_id ────────────────────────────────────────
        if session_id is not None and not re.fullmatch(_SESSION_ID_RE, session_id):
            return {
                "status": "invalid",
                "detail": "Invalid session_id format. Must be 1-64 chars of "
                          "[A-Za-z0-9_.-], starting and ending with a letter or digit.",
            }

        # ── Validate repo slug ─────────────────────────────────────────
        # Mirrors the runner's shape check so a malformed value fails locally
        # with a precise message instead of a 422 round trip.
        if repo is not None and not re.fullmatch(_REPO_RE, repo):
            return {
                "status": "invalid",
                "detail": "Invalid repo format. Expected owner/name, each segment "
                          "alphanumeric plus - _ ., starting alphanumeric.",
            }

        # repo is SESSION MODE ONLY on the runner: a clone needs a container that
        # outlives it. Rejecting it here gives a clear message rather than a 422
        # whose wording blames the runner.
        if repo and not session_id:
            return {
                "status": "invalid",
                "detail": "repo requires session_id; repository cloning is "
                          "session-mode only.",
            }

        # ── Validate provider / model (separate patterns) ──────────────
        for label, value, pattern in (
            ("provider", provider, _PROVIDER_RE),
            ("model", model, _MODEL_RE),
        ):
            if value is not None and not re.fullmatch(pattern, value):
                return {
                    "status": "invalid",
                    "detail": f"Invalid {label} format: {value!r}.",
                }

        # ── Cross-field: provider and model are all-or-nothing ─────────
        #
        # Both directions matter, and for different reasons:
        #   - a bare `model` is provider-specific, so it cannot be resolved;
        #   - a bare `provider` would run the default model against a
        #     provider that may not serve it.
        if model and not provider:
            return {
                "status": "invalid",
                "detail": "model was provided without provider; specify provider or omit model.",
            }
        if provider and not model:
            return {
                "status": "invalid",
                "detail": "provider was provided without model; specify model or omit provider.",
            }

        # ── Allowlist (mirror of the runner's) ─────────────────────────
        if provider is not None:
            if provider not in _ALLOWED_PROVIDERS:
                return {
                    "status": "invalid",
                    "detail": (
                        f"Provider {provider!r} is not allowed. "
                        f"Allowed: {', '.join(_ALLOWED_PROVIDERS)}."
                    ),
                }
            permitted = _ALLOWED_MODELS.get(provider, ())
            if model not in permitted:
                return {
                    "status": "invalid",
                    "detail": (
                        f"Model {model!r} is not allowed on {provider!r}. "
                        f"Allowed: {', '.join(permitted)}."
                    ),
                }

        # ── Validate & clamp timeout ───────────────────────────────────
        if timeout <= 0:
            timeout = self.valves.default_timeout
        timeout = max(10, min(int(timeout), _MAX_TIMEOUT))

        # ── Dispatch ───────────────────────────────────────────────────
        payload: Dict[str, Any] = {"task": task, "timeout": timeout}
        if session_id:
            payload["session_id"] = session_id
        if provider:
            payload["provider"] = provider
        if model:
            payload["model"] = model
        if repo:
            payload["repo"] = repo

        try:
            logger.info(
                "Dispatching pi_run: task=%r, session_id=%s, timeout=%d, force=%s, provider=%s, model=%s",
                task[:80], session_id, timeout, force, provider, model,
            )
            resp = await asyncio.to_thread(
                requests.post,
                self._run_endpoint,
                json=payload,
                timeout=(5, timeout + 10),
            )

            # ── Map status codes BEFORE raise_for_status() ─────────────
            #
            # raise_for_status() turns every non-2xx into an HTTPError, which
            # is a RequestException, so a single blanket handler collapsed
            # 409/422/429/500 into "Could not reach pi-agent-runner." The
            # service WAS reached and answered deliberately. A 429 in
            # particular must not read as an outage: the correct action is
            # "close an idle session and retry", not "the backend is down".
            #
            # The runner writes every `detail` string to be shown to a caller,
            # so prefer it over anything generic.
            if resp.status_code == 504:
                return {
                    "status": "timeout",
                    "detail": self._detail_or(
                        resp, f"Pi run timed out after {timeout}s."
                    ),
                }
            if resp.status_code == 429:
                return {
                    "status": "capacity",
                    "detail": self._detail_or(
                        resp,
                        f"Runner is at capacity (max {_MAX_CONCURRENT_SESSIONS} "
                        "sessions). Close an idle session and retry.",
                    ),
                }
            if resp.status_code == 409:
                return {
                    "status": "busy",
                    "detail": self._detail_or(
                        resp, "That session is already running a task; retry when it finishes."
                    ),
                }
            if resp.status_code == 422:
                return {
                    "status": "invalid",
                    "detail": self._validation_detail(resp),
                }
            if resp.status_code >= 500:
                return {
                    "status": "error",
                    "detail": self._detail_or(
                        resp, f"Runner failed with HTTP {resp.status_code}."
                    ),
                }

            resp.raise_for_status()
            result = self._safe_json(resp, "Runner returned non-JSON response.")
            logger.info(
                "pi_run complete: exit_code=%s, stdout_len=%d, timed_out=%s",
                result.get("exit_code"),
                len(result.get("stdout", "") or ""),
                result.get("timed_out"),
            )
            return result
        except requests.exceptions.Timeout:
            logger.error("pi_run timed out (client-side) for task %r", task[:80])
            return {
                "status": "timeout",
                "detail": f"Request to pi-agent-runner timed out after {timeout}s "
                          "on the client side. The run may still be going; check "
                          "pi_sessions() before retrying. A delegated run is "
                          "several agent loops, so consider a larger timeout.",
            }
        except requests.exceptions.RequestException as e:
            logger.error("pi_run dispatch failed: %s", e)
            return {
                "status": "error",
                "detail": f"Could not reach pi-agent-runner at {self.valves.runner_host}: {e}",
            }

    # ── SESSION MANAGEMENT ─────────────────────────────────────────────

    async def pi_session_close(self, session_id: str) -> Dict[str, Any]:
        """
        Close a persistent Pi session and free its container.

        Call this when the user's work is finished. Closing DISCARDS the
        session's /workspace files, its conversation history, and any subagent
        artifacts held under /root/.pi/agent/ — it cannot be resumed
        afterwards. If a subagent produced a report worth keeping, have the
        agent copy it into /workspace BEFORE closing.

        Sessions also expire on their own after 30 minutes idle or 24 hours,
        so this is about releasing resources sooner and about not losing
        artifacts by surprise.

        Args:
            session_id: The session_id used in prior pi_run calls.

        Returns:
            {"status": "closed", "session_id": ...}
            {"status": "not_found", ...} if it was already gone (closed or
                expired) — this is not an error
            On error: {"status": "error", "detail": "..."}
        """
        if not re.fullmatch(_SESSION_ID_RE, session_id):
            return {"status": "invalid", "detail": "Invalid session_id format."}

        try:
            resp = await asyncio.to_thread(
                requests.post,
                f"{self._session_close_endpoint}/{session_id}",
                timeout=15,
            )
            # A 404 here is the single most likely outcome: the idle reaper
            # (30 min) or the 24 h age cap may already have removed it. That
            # is not a transport failure and must not be reported as one.
            if resp.status_code == 404:
                return {
                    "status": "not_found",
                    "detail": self._detail_or(
                        resp,
                        f"Session {session_id!r} does not exist — it was already "
                        "closed, or reaped after 30 min idle / 24 h.",
                    ),
                }
            if resp.status_code == 409:
                return {
                    "status": "busy",
                    "detail": self._detail_or(
                        resp, "Session is running a task; retry when it finishes."
                    ),
                }
            resp.raise_for_status()
            return self._safe_json(resp, "Runner returned non-JSON response.")
        except requests.exceptions.RequestException as e:
            logger.error("pi_session_close failed: %s", e)
            return {
                "status": "error",
                "detail": f"Could not reach pi-agent-runner at {self.valves.runner_host}: {e}",
            }

    async def pi_sessions(self) -> Dict[str, Any]:
        """
        List open Pi coding sessions, their idle time, and whether they are busy.

        Call this before starting work that might continue an earlier session,
        so you reuse the existing session_id instead of starting fresh and
        losing its files and conversation. Also useful to check whether a
        session is free before running a task on it.

        At most 5 sessions may be open at once; when full, a new pi_run gets
        {"status": "capacity"} until one is closed.

        Returns:
            {"sessions": [{"session_id", "idle_seconds", "age_seconds",
                           "busy", "ready"}, ...], "limit": 5}
            "ready" is false while a session's container is still being
            created; "busy" is true while a task is running on it.
        """
        try:
            resp = await asyncio.to_thread(
                requests.get, self._sessions_endpoint, timeout=10
            )
            resp.raise_for_status()
            return self._safe_json(resp, "Runner returned non-JSON response.")
        except requests.exceptions.RequestException as e:
            logger.error("pi_sessions failed: %s", e)
            return {
                "status": "error",
                "detail": f"Could not reach pi-agent-runner at {self.valves.runner_host}: {e}",
            }

    # ── HELPERS ────────────────────────────────────────────────────────

    @staticmethod
    def _detail_or(resp, fallback: str) -> str:
        """
        Return the runner's own `detail` string when it has one.

        The runner writes these for a caller: "Session limit reached (5);
        close an idle session and retry." Discarding them and substituting a
        generic message is what made a capacity problem look like an outage.
        """
        try:
            body = resp.json()
        except ValueError:
            return fallback
        if isinstance(body, dict):
            detail = body.get("detail")
            if isinstance(detail, str) and detail.strip():
                return detail
        return fallback

    @staticmethod
    def _validation_detail(resp) -> str:
        """
        Format a FastAPI 422 into readable prose.

        For validation errors `detail` is a LIST of objects, not a string, so
        the naive path stringifies a Python repr with single quotes and the
        caller cannot tell which field was rejected.
        """
        try:
            body = resp.json()
        except ValueError:
            return "The runner rejected the request (HTTP 422)."

        detail = body.get("detail") if isinstance(body, dict) else None

        if isinstance(detail, str) and detail.strip():
            return detail

        if isinstance(detail, list) and detail:
            parts = []
            for err in detail[:3]:
                if not isinstance(err, dict):
                    continue
                # Drop the leading 'body' element of loc; it is noise.
                loc = ".".join(str(x) for x in (err.get("loc") or [])[1:]) or "request"
                parts.append(f"{loc}: {err.get('msg', 'invalid')}")
            if parts:
                return "The runner rejected the request: " + "; ".join(parts)

        return "The runner rejected the request (HTTP 422)."

    @staticmethod
    def _safe_json(resp, fallback_detail: str) -> Any:
        """
        Safely parse a response as JSON. If the body is not valid JSON,
        return a dict with the fallback detail and a snippet of the body.
        """
        try:
            return resp.json()
        except (ValueError, requests.exceptions.JSONDecodeError):
            logger.warning("Non-JSON response from runner: %s", resp.text[:200])
            return {
                "status": "error",
                "detail": fallback_detail,
                "body": resp.text[:500],
            }

    @staticmethod
    def _classify(task: str) -> str:
        """
        Classify a task as "coding" or "not_coding".

        Heuristic fast-path: score coding vs. non-coding signal hits.
        The classifier acts as a veto — it blocks obvious non-coding, but
        when ambiguous it defers to the LLM's judgment (dispatch).

        Returns:
            "coding" or "not_coding"
        """
        task_lower = task.lower()

        coding_signals = [
            "write", "create", "build", "implement", "code", "script",
            "function", "class", "module", "package", "program",
            "debug", "fix", "refactor", "compile", "run", "test",
            "file", "directory", "folder", "path", "python", "javascript",
            "typescript", "rust", "go", "java", "c++", "bash", "shell",
            "docker", "api", "endpoint", "server", "database", "sql",
            "html", "css", "json", "yaml", "xml", "regex",
            "algorithm", "data structure", "parse", "serialize",
            "import", "export", "install", "deploy", "configure",
            "add", "remove", "update", "modify", "change",
            "error", "exception", "traceback", "stack trace",
            "unit test", "integration test", "pytest", "unittest",
            "makefile", "dockerfile", "requirements", "package.json",
            "git", "commit", "branch", "merge", "pull request",
            # Delegation vocabulary: a task phrased around "review/verify the
            # change" is coding work even when it names no language.
            "review", "verify", "lint", "check", "repo", "repository",
        ]

        non_coding_signals = [
            "what is", "who is", "explain", "define", "describe",
            "tell me", "list", "compare", "summarize", "translate",
            "recipe", "story", "poem", "essay", "opinion",
            "weather", "news", "stock", "price", "recommend",
            "meaning", "purpose", "history", "why", "how does",
            "what does", "can you", "help me understand",
        ]

        coding_score = sum(1 for s in coding_signals if s in task_lower)
        non_coding_score = sum(1 for s in non_coding_signals if s in task_lower)

        # Strong non-coding signal → block
        if non_coding_score >= 2 and coding_score == 0:
            return "not_coding"

        # Strong coding signal → dispatch
        if coding_score >= 2:
            return "coding"

        # Ambiguous → dispatch (let the agent figure it out)
        return "coding"

    # ── HEALTH ─────────────────────────────────────────────────────────

    async def pi_health(self) -> Dict[str, Any]:
        """
        Check that the pi-agent-runner service is up and its Docker socket is
        reachable. Use this to verify the agent backend is available.

        Also reports the runner's version and the provider/model allowlist, so
        a redeploy — or a config drift — can be confirmed from one call.

        This says nothing about what the SANDBOX image can do. If a run fails
        because a subagent role is missing, the image is the suspect, not this
        endpoint.
        """
        try:
            resp = await asyncio.to_thread(
                requests.get, f"{self.valves.runner_host}/health", timeout=10
            )
            if resp.status_code == 503:
                return {
                    "status": "error",
                    "detail": self._detail_or(
                        resp, "Runner is up but its Docker socket is unreachable."
                    ),
                    "version": self._version_of(resp),
                }
            resp.raise_for_status()
            return self._safe_json(resp, "Runner returned non-JSON response.")
        except requests.exceptions.RequestException as e:
            logger.error("pi_health check failed: %s", e)
            return {
                "status": "error",
                "detail": f"Could not reach pi-agent-runner at {self.valves.runner_host}: {e}",
            }

    @staticmethod
    def _version_of(resp) -> Optional[str]:
        """Extract `version` from a runner response, if present."""
        try:
            body = resp.json()
        except ValueError:
            return None
        if isinstance(body, dict):
            version = body.get("version")
            if isinstance(version, str):
                return version
        return None
