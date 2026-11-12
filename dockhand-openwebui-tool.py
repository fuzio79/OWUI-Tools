"""
title: Dockhand Read-Only
author: your-name
version: 0.2.2
description: Read-only container and stack inspection through the Dockhand API, with secret redaction.
requirements: requests
"""

import json
import re
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote

import asyncio
import requests
from pydantic import BaseModel, Field


# Sensitive key words. Matched as whole segments (bounded by non-alphanumerics
# or string boundaries) so e.g. "key" does not match "monkey" and "auth" does
# not match "author". Includes DSN / connection-string style keys (url, uri,
# dsn, conn, jwt, client_id, ...) so DATABASE_URL / REDIS_URL / MONGO_URI are
# caught by key name.
# CHANGED: dropped the bare `url|uri|conn|connection` and `key` alternatives.
# They matched so much legitimate config (NEXT_PUBLIC_URL, S3_KEY_ID,
# SSH_PUBLIC_KEY, CONNECTION_IDLE_TIMEOUT, HEALTHCHECK_TEST_URL) that real
# values were being replaced with ***REDACTED***, losing data the model needed.
# Credential-bearing URLs are still caught by _CRED_URL / _CRED_URL_ANY
# regardless of the key name, and the specific *_URL / *_URI names that do
# carry credentials are listed explicitly below.
_SENSITIVE_WORDS = (
    "password|passwd|pwd|secret|token|credential|auth|api_?key"
    "|private|jwt|signing|client_?id|client_?secret|dsn"
    "|database_url|db_url|redis_url|mongo_uri|connection_string"
)
_SENSITIVE = re.compile(
    r"(?<![A-Za-z0-9])(" + _SENSITIVE_WORDS + r")(?![A-Za-z0-9])",
    re.I,
)

# Values that look like a URL with embedded credentials (scheme://user:pass@...).
# Caught even when the key name is not itself flagged as sensitive, so a DSN
# such as postgres://user:supersecret@host/db is redacted regardless of the key.
_CRED_URL = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://[^@\s/:]+:[^@\s/]+@", re.I)

# NEW: log-line variants. Logs are free text, so these match anywhere in the
# line rather than being anchored like the compose/env patterns.
_LOG_SECRET_LINE = re.compile(
    # group 1 = the whole "KEY=" prefix, so the key and separator survive;
    # group 2 = the value, which is replaced.
    r"(?<![A-Za-z0-9])((?:" + _SENSITIVE_WORDS + r")[A-Za-z0-9_.\-]*\s*[:=]\s*)(\S+)",
    re.I,
)
_CRED_URL_ANY = re.compile(
    r"([A-Za-z][A-Za-z0-9+.\-]*)://[^@\s/:]+:[^@\s/]+@", re.I
)

# Matches "KEY: value", "KEY=value" and "- KEY=value" lines in compose text
# where KEY contains a sensitive word (as a whole segment). Keeps the key,
# replaces the value.
_COMPOSE_SECRET_LINE = re.compile(
    r"^(\s*(?:-\s*)?[\"']?[A-Za-z0-9_.\-]*"
    r"(?<![A-Za-z0-9])"
    + _SENSITIVE_WORDS
    + r"(?![A-Za-z0-9])"
    r"[A-Za-z0-9_.\-]*[\"']?\s*[:=]\s*)(\S.*)$",
    re.I | re.M,
)

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")
_REDACTED = "***REDACTED***"


class Tools:
    """
    Dockhand read-only inspection tools.

    Talks to the Dockhand HTTP API (Bearer token auth) to list/inspect
    containers, read logs, and view stacks. Only GET requests are ever sent.
    Secrets in environment variables and compose files are redacted by default.

    Dockhand uses an `env` query parameter to select the target environment
    (Docker host). Interactive API docs: /api/docs/ui (needs FEAT_API_DOCS=true
    on the Dockhand container).
    """

    # ── VALVES ──────────────────────────────────────────────────────
    class Valves(BaseModel):
        DOCKHAND_URL: str = Field(
            default="http://dockhand:3000",
            description="Dockhand base URL, without a trailing slash.",
        )
        API_TOKEN: str = Field(
            default="",
            description="Dockhand API bearer token (dh_...). Create it in Dockhand Settings > API Tokens, ideally with minimal scope.",
        )
        ENV_ID: int = Field(
            default=1,
            description="Default Dockhand environment ID. Discover it via list_environments.",
        )
        VERIFY_TLS: bool = Field(
            default=True,
            description="Verify Dockhand's TLS certificate. Keep enabled when using HTTPS.",
        )
        TIMEOUT_SECONDS: int = Field(
            default=20,
            description="HTTP request timeout for normal calls.",
        )
        STATS_TIMEOUT_SECONDS: int = Field(
            default=45,
            description="HTTP request timeout for container_stats (can be slow on busy hosts).",
        )
        MAX_LOG_LINES: int = Field(
            default=300,
            description="Maximum number of log lines returned.",
        )
        MAX_LOG_LINE_CHARS: int = Field(
            default=2000,
            description="Maximum characters kept per log line.",
        )
        MAX_OUTPUT_CHARS: int = Field(
            default=20000,
            description="Maximum size of JSON results returned to the model.",
        )
        REDACT_SECRETS: bool = Field(
            default=True,
            description="Redact env vars / compose values whose key looks like a secret (pass, token, key, ...) or whose value looks like a credentialed URL.",
        )

    # ── INIT ────────────────────────────────────────────────────────
    def __init__(self):
        self.valves = self.Valves()

    # ── HELPERS ─────────────────────────────────────────────────────
    @staticmethod
    def _http_error(exc: Exception) -> Dict[str, Any]:
        """Convert a requests exception into a safe, readable error (no headers/token)."""
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            code = exc.response.status_code
            if code == 401:
                return {"error": "Dockhand returned 401 Unauthorized. Check the API_TOKEN valve."}
            if code == 403:
                return {"error": "Dockhand returned 403 Forbidden. The token lacks permission for this resource."}
            if code == 404:
                return {"error": "Dockhand returned 404 Not Found (wrong name, environment, or endpoint)."}
            return {"error": f"Dockhand returned HTTP {code}."}
        if isinstance(exc, requests.Timeout):
            return {"error": "Dockhand request timed out."}
        if isinstance(exc, requests.ConnectionError):
            return {"error": "Could not connect to Dockhand. Check DOCKHAND_URL."}
        return {"error": f"Dockhand request failed: {type(exc).__name__}"}

    async def _get(
        self,
        path: str,
        params: Optional[dict] = None,
        timeout: Optional[int] = None,
    ) -> Union[requests.Response, Dict[str, Any]]:
        """
        Authenticated GET. Returns the Response on success or an {"error": ...} dict.
        This is the only function that talks to the network, and it only does GET,
        which enforces the read-only guarantee in code.
        """
        if not self.valves.API_TOKEN:
            return {"error": "Set the API_TOKEN valve in the tool settings."}

        url = self.valves.DOCKHAND_URL.rstrip("/") + path
        try:
            response = await asyncio.to_thread(
                requests.get,
                url,
                headers={"Authorization": f"Bearer {self.valves.API_TOKEN}"},
                params=params,
                # CHANGED: `timeout or default` silently replaced an
                # explicit 0 with the default. Only None should fall back.
                timeout=timeout if timeout is not None else self.valves.TIMEOUT_SECONDS,
                verify=self.valves.VERIFY_TLS,
            )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            return self._http_error(exc)

    async def _get_data(
        self,
        path: str,
        params: Optional[dict] = None,
        timeout: Optional[int] = None,
    ) -> Any:
        """GET and return parsed JSON (or text). Errors come back as {"error": ...}."""
        result = await self._get(path, params=params, timeout=timeout)
        if isinstance(result, dict):
            return result
        if "application/json" in result.headers.get("content-type", ""):
            try:
                return result.json()
            except ValueError:
                return {"error": "Dockhand returned invalid JSON."}
        return result.text

    @staticmethod
    def _is_error(data: Any) -> bool:
        """CHANGED: no longer requires the dict to have exactly one key.

        The old `len(data) == 1` test missed a real error body such as
        {"error": "...", "detail": "..."} (surfaced as a misleading
        "Unexpected response") and falsely flagged {"error": null}.
        """
        if not isinstance(data, dict):
            return False
        err = data.get("error")
        if err is None or err is False:
            return False
        if isinstance(err, str) and not err.strip():
            return False
        return True

    def _env(self, env_id: Optional[int]) -> str:
        return str(env_id if env_id is not None else self.valves.ENV_ID)

    @staticmethod
    def _valid_name(name: str) -> Optional[str]:
        """Return a cleaned name/ID, or None if it is not a safe path segment."""
        cleaned = (name or "").strip().lstrip("/")
        return cleaned if _NAME_RE.match(cleaned) else None

    @staticmethod
    def _extract_list(data: Any, keys: tuple) -> Optional[list]:
        """Dockhand may return a bare list or an object wrapper; normalise to a list."""
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in keys:
                if isinstance(data.get(key), list):
                    return data[key]
        return None

    def _limit(self, obj: Any) -> Any:
        """Cap the serialized size of a result so it fits in the model context."""
        try:
            text = json.dumps(obj, default=str)
        except (TypeError, ValueError):
            text = str(obj)
        cap = self.valves.MAX_OUTPUT_CHARS
        if len(text) <= cap:
            return obj
        return {
            "truncated": True,
            "note": f"Output was {len(text)} chars; showing the first {cap}.",
            "partial": text[:cap],
        }

    # ── REDACTION ───────────────────────────────────────────────────
    @staticmethod
    def _redact_env_list(env: list) -> list:
        out = []
        for item in env:
            if isinstance(item, str):
                key, sep, value = item.partition("=")
                if sep and (_SENSITIVE.search(key) or _CRED_URL_ANY.search(value)):
                    out.append(f"{key}={_REDACTED}")
                else:
                    out.append(item)
            else:
                out.append(item)
        return out

    def _redact_obj(self, obj: Any) -> Any:
        """Recursively redact env lists, secret-looking keys, and credentialed URLs."""
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                if isinstance(k, str) and k.lower() == "env" and isinstance(v, list):
                    result[k] = self._redact_env_list(v)
                elif isinstance(k, str) and _SENSITIVE.search(k) and v is not None:
                    # CHANGED: redact regardless of value type. A secret stored
                    # as a list, or any non-None non-str value, used to pass
                    # through untouched. None stays None: an absent value is
                    # not a secret, and masking it loses information.
                    result[k] = _REDACTED
                elif isinstance(v, str) and _CRED_URL_ANY.search(v):
                    result[k] = _REDACTED
                else:
                    result[k] = self._redact_obj(v)
            return result
        if isinstance(obj, list):
            return [self._redact_obj(i) for i in obj]
        return obj

    @staticmethod
    def _redact_compose_text(text: str) -> str:
        return _COMPOSE_SECRET_LINE.sub(lambda m: m.group(1) + _REDACTED, text)

    @staticmethod
    def _redact_log_line(line: str) -> str:
        """NEW: redact secrets from a single log line.

        Logs are the most likely place a service prints a credential at
        startup (DATABASE_URL=postgres://user:pass@host/db), and this tool
        previously returned them verbatim despite advertising redaction.
        """
        # KEY=value / KEY: value anywhere in the line (not just at the start).
        line = _LOG_SECRET_LINE.sub(lambda m: m.group(1) + _REDACTED, line)
        # scheme://user:pass@host anywhere in the line.
        line = _CRED_URL_ANY.sub(lambda m: m.group(1) + "://" + _REDACTED + "@", line)
        return line

    # ── LOG PARSING ─────────────────────────────────────────────────
    def _parse_logs(self, raw: bytes) -> List[str]:
        """
        Parse Docker log output into plain lines.

        Multiplexed format, per frame: 1 byte stream type (0=stdin, 1=stdout,
        2=stderr), 3 zero bytes, 4 bytes big-endian payload length, payload.
        TTY containers (or proxies that demux) return plain text, so the header
        is validated before parsing and we fall back to plain text otherwise.
        """
        max_chars = self.valves.MAX_LOG_LINE_CHARS
        lines: List[str] = []

        def add(label: Optional[str], part: str) -> None:
            if not part:
                return
            if len(part) > max_chars:
                part = part[:max_chars] + "...[truncated]"
            lines.append(f"[{label}] {part}" if label else part)

        n = len(raw)
        if n >= 8 and raw[0] in (0, 1, 2) and raw[1:4] == b"\x00\x00\x00":
            i = 0
            while i + 8 <= n:
                stream = raw[i]
                length = int.from_bytes(raw[i + 4 : i + 8], "big")
                i += 8
                # CHANGED: bail out if the frame claims more bytes than remain.
                # Otherwise a coincidental header match swallows the rest of
                # the stream as one frame.
                if length > n - i:
                    msg = raw[i:]  # tolerate a truncated final frame
                    i = n
                else:
                    msg = raw[i : i + length]
                    i += length
                label = "stderr" if stream == 2 else "stdout"
                for part in msg.decode("utf-8", errors="replace").split("\n"):
                    add(label, part.rstrip("\r"))
            return lines

        for part in raw.decode("utf-8", errors="replace").split("\n"):
            add(None, part.rstrip("\r"))
        return lines

    def _log_lines_from_text(self, text: str) -> List[str]:
        """Split a plain-text log body into capped lines (no stream labels)."""
        max_chars = self.valves.MAX_LOG_LINE_CHARS
        lines: List[str] = []
        for part in text.split("\n"):
            part = part.rstrip("\r")
            if not part:
                continue
            if len(part) > max_chars:
                part = part[:max_chars] + "...[truncated]"
            lines.append(part)
        return lines

    def _extract_log_text(self, result: requests.Response) -> Optional[str]:
        """
        Return the log text from a Dockhand logs response, or None if the body
        is not the JSON {"logs": "..."} shape and should be treated as raw bytes.

        Dockhand returns logs as JSON ({"logs": "line1\\nline2..."}), not as raw
        multiplexed Docker bytes. The newlines are escaped inside the JSON string,
        so the body must be JSON-decoded first.
        """
        ctype = result.headers.get("content-type", "")
        if "application/json" in ctype:
            try:
                payload = result.json()
            except ValueError:
                return None
            if isinstance(payload, dict) and isinstance(payload.get("logs"), str):
                return payload["logs"]
        return None

    # ── READ-ONLY TOOLS ─────────────────────────────────────────────

    async def list_environments(self) -> Dict[str, Any]:
        """
        List Dockhand environments (Docker hosts).

        Use this to discover the env_id to pass to other tools.

        Returns:
            {"count": N, "environments": [{"id", "name", "type", "status", "url"}, ...]}
            or {"error": ...} on failure.
        """
        data = await self._get_data("/api/environments")
        if self._is_error(data):
            return data

        items = self._extract_list(data, ("environments", "data", "items", "results"))
        if items is None:
            return {"error": "Unexpected response from Dockhand."}

        summary: List[Dict[str, Any]] = []
        for e in items:
            if not isinstance(e, dict):
                continue
            summary.append(
                {
                    "id": e.get("id", e.get("Id")),
                    "name": e.get("name", e.get("Name", "")),
                    "type": e.get("type", e.get("Type", "")),
                    "status": e.get("status", e.get("Status", "")),
                    "url": e.get("url", e.get("URL", "")),
                }
            )
        return {"count": len(summary), "environments": summary}

    async def list_containers(
        self, all_containers: bool = True, env_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        List containers in a Dockhand environment.

        Returns a compact summary (name, id, state, status, image) per container.

        Args:
            all_containers: If True, include stopped containers (default). If False, only running ones.
            env_id: Environment ID. Defaults to the ENV_ID valve.

        Returns:
            {"count": N, "containers": [...]} or {"error": ...} on failure.
        """
        params = {"env": self._env(env_id), "all": "true" if all_containers else "false"}
        data = await self._get_data("/api/containers", params=params)
        if self._is_error(data):
            return data

        items = self._extract_list(data, ("containers", "data", "items", "results"))
        if items is None:
            return {"error": "Unexpected response from Dockhand."}

        summary: List[Dict[str, Any]] = []
        for c in items:
            if not isinstance(c, dict):
                continue

            cid = str(c.get("id") or c.get("Id") or "")
            names = c.get("Names") or c.get("names")
            name = c.get("name") or c.get("Name") or ""
            if not name and isinstance(names, list) and names:
                name = str(names[0])
            name = name.lstrip("/") or cid[:12]

            state = c.get("state", c.get("State", ""))
            # Filter client-side too, in case Dockhand ignores the "all" param.
            if not all_containers and str(state).lower() != "running":
                continue

            summary.append(
                {
                    "name": name,
                    "id": cid,
                    "state": state,
                    "status": c.get("status", c.get("Status", "")),
                    "image": c.get("image", c.get("Image", "")),
                }
            )
        return {"count": len(summary), "containers": summary}

    async def inspect_container(
        self, container_id_or_name: str, env_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Inspect one container by ID or name. Secret-looking env vars are redacted.

        Args:
            container_id_or_name: Container ID or name.
            env_id: Environment ID. Defaults to the ENV_ID valve.

        Returns:
            {"container": <inspect JSON>} or {"error": ...} on failure.
        """
        name = self._valid_name(container_id_or_name)
        if not name:
            return {"error": "Invalid container name or ID."}

        data = await self._get_data(
            f"/api/containers/{quote(name, safe='')}/inspect",
            params={"env": self._env(env_id)},
        )
        if self._is_error(data):
            return data
        if self.valves.REDACT_SECRETS:
            data = self._redact_obj(data)
        return {"container": self._limit(data)}

    async def container_logs(
        self,
        container_id_or_name: str,
        tail: int = 100,
        timestamps: bool = False,
        env_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get recent stdout/stderr logs for a container.

        Args:
            container_id_or_name: Container ID or name.
            tail: Number of lines to tail. Clamped to MAX_LOG_LINES.
            timestamps: If True, keep Docker's timestamps in the output.
            env_id: Environment ID. Defaults to the ENV_ID valve.

        Returns:
            {"container": ..., "count": N, "lines": [...]} or {"error": ...} on failure.
        """
        name = self._valid_name(container_id_or_name)
        if not name:
            return {"error": "Invalid container name or ID."}

        try:
            tail_n = int(tail)
        except (TypeError, ValueError):
            tail_n = 100
        tail_n = max(1, min(tail_n, self.valves.MAX_LOG_LINES))

        params = {
            "env": self._env(env_id),
            "tail": str(tail_n),
            "timestamps": str(bool(timestamps)).lower(),
        }
        result = await self._get(f"/api/containers/{quote(name, safe='')}/logs", params=params)
        if isinstance(result, dict):
            return result

        # Dockhand returns logs as JSON ({"logs": "..."}); extract the text and
        # split it. Otherwise fall back to raw/multiplexed byte parsing.
        log_text = self._extract_log_text(result)
        if log_text is not None:
            lines = self._log_lines_from_text(log_text)
        else:
            lines = self._parse_logs(result.content)

        # Enforce the tail client-side in case Dockhand ignores the parameter.
        lines = lines[-tail_n:]

        # CHANGED: redact secrets before returning. Logs are the most likely
        # place a service echoes a credential at startup, and this path used
        # to return them verbatim while the tool advertised secret redaction.
        if self.valves.REDACT_SECRETS:
            lines = [self._redact_log_line(line) for line in lines]

        # CHANGED: apply MAX_OUTPUT_CHARS. Per-line and line-count caps alone
        # allowed MAX_LOG_LINES x MAX_LOG_LINE_CHARS (300 x 2000 = 600k chars)
        # to reach the model and flood its context.
        cap = self.valves.MAX_OUTPUT_CHARS
        out: List[str] = []
        used = 0
        for line in lines:
            if used + len(line) + 1 > cap:
                out.append(
                    f"...[truncated: {len(lines)} lines total, "
                    f"{cap} char cap reached]"
                )
                break
            out.append(line)
            used += len(line) + 1

        return {"container": name, "count": len(out), "lines": out}

    async def list_stacks(self, env_id: Optional[int] = None) -> Dict[str, Any]:
        """
        List Dockhand compose stacks.

        Args:
            env_id: Environment ID. Defaults to the ENV_ID valve.

        Returns:
            {"count": N, "stacks": [...]} or {"error": ...} on failure.
        """
        data = await self._get_data("/api/stacks", params={"env": self._env(env_id)})
        if self._is_error(data):
            return data

        items = self._extract_list(data, ("stacks", "data", "items", "results"))
        if items is None:
            return {"error": "Unexpected response from Dockhand."}
        if self.valves.REDACT_SECRETS:
            items = self._redact_obj(items)
        return {"count": len(items), "stacks": self._limit(items)}

    async def get_stack_compose(
        self, stack_name: str, env_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Retrieve the compose file for a stack by name. Secret-looking values are redacted.

        Dockhand does not document a dedicated "get compose file" endpoint, so this
        tries the most likely read endpoints. Verify the real one in /api/docs/ui
        and trim the candidate list once known.

        Args:
            stack_name: Stack name.
            env_id: Environment ID. Defaults to the ENV_ID valve.

        Returns:
            {"stack": ..., "compose": ...} or {"error": ...} on failure.
        """
        name = self._valid_name(stack_name)
        if not name:
            return {"error": "Invalid stack name."}

        stack = quote(name, safe="")
        params = {"env": self._env(env_id)}
        candidates = [
            f"/api/stacks/{stack}/file",
            f"/api/stacks/{stack}/compose",
            f"/api/stacks/{stack}",
        ]

        last_error: Optional[Dict[str, Any]] = None
        fallback_shape: Optional[Dict[str, Any]] = None
        for path in candidates:
            data = await self._get_data(path, params=params)

            if self._is_error(data):
                msg = data["error"]
                # Auth problems will not be fixed by trying another path.
                if "401" in msg or "403" in msg or "API_TOKEN" in msg or "connect" in msg.lower():
                    return data
                last_error = data
                continue

            if isinstance(data, str):
                return {"stack": name, "compose": self._finalize_compose(data)}

            if isinstance(data, dict):
                for key in ("compose", "stackFileContent", "StackFileContent", "file", "content", "yaml"):
                    if isinstance(data.get(key), str):
                        return {"stack": name, "compose": self._finalize_compose(data[key])}
                # CHANGED: an unrecognised dict shape used to be returned
                # immediately, so /compose and / were never tried. Remember it
                # as a last resort and keep looking for a real compose body.
                if fallback_shape is None:
                    fallback_shape = data
                continue

        if fallback_shape is not None:
            if self.valves.REDACT_SECRETS:
                fallback_shape = self._redact_obj(fallback_shape)
            return {"stack": name, "data": self._limit(fallback_shape)}

        return {
            "error": "Could not retrieve the compose file; the endpoint may differ in this Dockhand version (check /api/docs/ui).",
            "last_error": (last_error or {}).get("error"),
        }

    def _finalize_compose(self, text: str) -> Any:
        if self.valves.REDACT_SECRETS:
            text = self._redact_compose_text(text)
        cap = self.valves.MAX_OUTPUT_CHARS
        if len(text) > cap:
            return text[:cap] + f"\n...[truncated, {len(text)} chars total]"
        return text

    async def container_stats(self, env_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get CPU, memory, network and disk I/O for every running container.

        Args:
            env_id: Environment ID. Defaults to the ENV_ID valve.

        Returns:
            {"count": N, "stats": [...]} or {"error": ...} on failure.
        """
        data = await self._get_data(
            "/api/containers/stats",
            params={"env": self._env(env_id)},
            timeout=self.valves.STATS_TIMEOUT_SECONDS,
        )
        if self._is_error(data):
            return data

        items = self._extract_list(data, ("stats", "containers", "data", "items", "results"))
        if items is None:
            return {"error": "Unexpected response from Dockhand."}
        return {"count": len(items), "stats": self._limit(items)}
