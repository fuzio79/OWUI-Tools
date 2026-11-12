"""
title: Docker Exec
description: Run commands and inspect containers in a trusted, explicitly-allowlisted set on the Docker host.
requirements: requests
version: 1.6.0
"""

import asyncio
import hashlib
import json
import logging
import shlex
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── LOOP GUARD ───────────────────────────────────────────────────────

class _LoopGuard:
    """
    Sliding-window loop guard for tool calls.

    Maintains a per-key deque of timestamps.  A key is
    ``(chat_id, tool_name, args_hash)``.  If the same key has been
    recorded ``max_repeats`` times within the last ``window`` seconds,
    a short-circuit response is returned instead of executing the tool.

    Two-phase API:
      * ``check()``  – does not count the current call; returns ``None``
        (allow) or a block dict.
      * ``record()`` – called only after a successful (non-error) result.

    This is a module-level singleton so state survives across OWUI
    request/turn instantiations.  Note: editing/saving the tool in OWUI
    re-executes the module and resets the guard.  In multi-worker
    deployments each worker keeps its own guard instance.
    """

    def __init__(self):
        self._recent: Dict[str, deque] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _make_key(chat_id: str, tool_name: str, kwargs: Dict[str, Any]) -> str:
        """Build a stable, compact key from chat_id, tool name, and args."""
        payload = json.dumps(kwargs, sort_keys=True, default=str)
        return f"{chat_id}:{tool_name}:{hashlib.sha256(payload.encode()).hexdigest()}"

    def check(
        self,
        chat_id: str,
        tool_name: str,
        window: int,
        max_repeats: int,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """
        Return ``None`` if the call is allowed, or a short-circuit dict
        if a loop is detected.  Does not count the current call.
        """
        if not chat_id:
            # OWUI didn't inject __chat_id__; guard is disabled to avoid
            # cross-chat collisions.
            return None
        try:
            key = self._make_key(chat_id, tool_name, kwargs)
            now = time.time()
            with self._lock:
                dq = self._recent.get(key)
                if dq is None:
                    return None
                # Evict timestamps older than the window.
                while dq and dq[0] < now - window:
                    dq.popleft()
                if not dq:
                    self._recent.pop(key, None)
                    return None
                if len(dq) >= max_repeats:
                    logger.warning(
                        "loop_guard: %s called %d times in last %ds (chat %s)",
                        tool_name, len(dq), window, chat_id,
                    )
                    return {
                        "status": "loop_guard",
                        "detail": (
                            f"You have called {tool_name} with identical arguments "
                            f"{len(dq)} times in the last {window} seconds. "
                            "Do not repeat this exact call. "
                            "If you need more information, use a different approach "
                            "or tool. If the task is complete, stop."
                        ),
                    }
            return None
        except Exception as e:
            # Fail open: guard errors must never block the tool call.
            logger.warning("loop_guard check failed (fail open): %s", e)
            return None

    def record(
        self,
        chat_id: str,
        tool_name: str,
        window: int,
        **kwargs,
    ) -> None:
        """Record a successful call.  Only called after a non-error result."""
        if not chat_id:
            return
        try:
            key = self._make_key(chat_id, tool_name, kwargs)
            now = time.time()
            with self._lock:
                dq = self._recent.setdefault(key, deque())
                dq.append(now)
                # Trim to keep memory bounded.
                while dq and dq[0] < now - window:
                    dq.popleft()
                if not dq:
                    self._recent.pop(key, None)
        except Exception as e:
            logger.warning("loop_guard record failed (ignored): %s", e)


# Module-level singleton: survives OWUI per-turn instantiation.
_guard = _LoopGuard()


# ── TOOLS ────────────────────────────────────────────────────────────

class Tools:
    """
    Docker container management tools.
    Manages containers labelled managed=true on the local Docker host.
    Read-only operations run directly; destructive operations and command
    execution require a real confirmation dialog in the UI.

    A sliding-window loop guard prevents the model from calling the same
    tool with identical arguments more than ``loop_guard_max_repeats``
    times within a window of ``loop_guard_window`` seconds.  The guard
    covers successful executions; declined confirmations are not counted.
    """

    # ── VALVES ──────────────────────────────────────────────────────
    class Valves(BaseModel):
        server_url: str = Field(
            default="http://localhost:8091",
            description=(
                "Base URL of the docker-exec service. This must be an address "
                "REACHABLE FROM THE open-webui CONTAINER -- normally the "
                "Docker host's LAN IP (e.g. http://192.0.2.10:8091), NOT "
                "localhost. Inside a container, localhost is the open-webui "
                "container itself, so the default only works if this tool runs "
                "on the same host/namespace as docker-exec. The API is plain "
                "HTTP and the key travels in a header, so keep it on a trusted "
                "network."
            ),
        )
        api_key: str = Field(
            default="",
            description="API key for the docker-exec service (sent as X-API-Key).",
        )
        default_log_lines: int = Field(
            default=200, ge=10, le=5000,
            description="Default number of log lines to tail (10–5000).",
        )
        default_exec_timeout: int = Field(
            default=30, ge=1, le=600,
            description="Default timeout in seconds for exec commands (1–600).",
        )
        confirm_exec: bool = Field(
            default=True,
            description="Ask for confirmation in the UI before every docker_exec.",
        )
        loop_guard_window: int = Field(
            default=30, ge=5, le=300,
            description="Sliding window in seconds for the loop guard.",
        )
        loop_guard_max_repeats: int = Field(
            default=3, ge=1, le=20,
            description="Max successful calls with identical args within the window before short-circuiting.",
        )

    # ── INIT ────────────────────────────────────────────────────────
    def __init__(self):
        self.valves = self.Valves()

    # ── INTERNALS ───────────────────────────────────────────────────

    def _request(self, method: str, path: str, timeout: float = 15, **kwargs) -> Any:
        """Blocking HTTP call. Always returns JSON-able data; never raises."""
        url = f"{self.valves.server_url.rstrip('/')}{path}"
        headers = {"X-API-Key": self.valves.api_key} if self.valves.api_key else {}
        try:
            r = requests.request(method, url, headers=headers, timeout=timeout, **kwargs)
        except requests.exceptions.RequestException as e:
            logger.error("docker tool %s %s failed: %s", method, path, e)
            return {"status": "error", "detail": str(e)}

        if not r.ok:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            return {"status": "error", "http_status": r.status_code, "detail": detail}
        try:
            return r.json()
        except ValueError:
            return {"status": "error", "detail": "Server returned non-JSON response."}

    async def _call(self, *args, **kwargs) -> Any:
        """Run the blocking request off the event loop."""
        return await asyncio.to_thread(self._request, *args, **kwargs)

    @staticmethod
    async def _confirm(event_call, title: str, message: str) -> bool:
        """Real UI confirmation. Denies if the UI can't ask."""
        if event_call is None:
            return False
        try:
            result = await event_call(
                {"type": "confirmation", "data": {"title": title, "message": message}}
            )
        except Exception as e:
            logger.error("confirmation failed: %s", e)
            return False
        return bool(result)

    def _is_error(self, result: Any) -> bool:
        """
        True if the result is an error dict.
        Must list ALL non-success statuses returned by _request and tool
        bodies. Add new statuses here as they are introduced.

        A Docker exec result is NOT an error dict even when the command
        failed: it carries ``exit_code`` (127, 1, ...) and no ``status`` key.
        Do not add non-zero ``exit_code`` here -- that would stop failing
        calls being recorded by the loop guard, which is precisely the
        behaviour that lets three identical failures trip it.
        """
        return isinstance(result, dict) and result.get("status") in (
            "error", "loop_guard", "cancelled", "timeout",
        )

    def _record_success(self, chat_id: str, tool_name: str, **kwargs) -> None:
        """Record an executed call in the loop guard.

        "Success" here means the call REACHED the service and returned a
        result -- not that the command exited 0. A non-zero exit is a real
        answer, so it is counted: three identical failing execs trip the guard
        and stop the model burning turns on the same broken call.

        See GuardCountsFailuresTests, which pins this so a future "a failure
        is an error" refactor cannot silently blind the guard.
        """
        _guard.record(
            chat_id, tool_name,
            window=self.valves.loop_guard_window,
            **kwargs,
        )

    def _check_guard(self, chat_id: str, tool_name: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Check the loop guard; returns block dict or None."""
        return _guard.check(
            chat_id, tool_name,
            window=self.valves.loop_guard_window,
            max_repeats=self.valves.loop_guard_max_repeats,
            **kwargs,
        )

    @staticmethod
    def _coerce_int(value: Any, default: int = 0) -> int:
        """Coerce a value to int, falling back to default on failure."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        """Coerce a value to bool, handling string 'true'/'false' correctly."""
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return bool(value)

    @staticmethod
    def _truncation_note(result: Dict[str, Any]) -> str:
        """Build a note for the model when exec/log output was cut off.

        Without this the model cannot tell a complete answer from a prefix, and
        silently reasons on half the output.
        """
        parts = []
        if result.get("stdout_truncated"):
            parts.append(
                f"stdout was cut off after {result.get('bytes_total_out', '?')} bytes"
            )
        if result.get("stderr_truncated"):
            parts.append(
                f"stderr was cut off after {result.get('bytes_total_err', '?')} bytes"
            )
        if result.get("truncated"):
            parts.append(f"logs were cut off ({result.get('bytes_total', '?')} bytes total)")
        return "; ".join(parts)

    @staticmethod
    def _validate_argv(argv: List[str]) -> List[str]:
        """Reject argv shapes that cannot mean what the caller intended.

        The one case worth refusing is a single element containing
        whitespace, because it is ambiguous in a way that fails confusingly:

            ["echo probe-ok"]     -> execve("echo probe-ok") -> exit 127
            ["echo", "probe-ok"]  -> the intended thing

        ``cmd`` goes straight to the Docker exec API, which does no shell
        parsing, so the space is part of the *filename*. Measured on the live
        service before this check existed:

            exit_code 127
            stderr: timeout: failed to run command 'echo probe-ok':
                    No such file or directory

        That ``timeout:`` prefix is what made this look like the service had
        joined the arguments. It had not -- there was only ever one argument.

        The alternative shapes are unaffected and still work:
        ``["sh", "-c", "echo a b"]`` is three elements, as is
        ``["python3", "-c", "print(1)"]``.

        Raises:
            ValueError: if argv is a single whitespace-bearing element.
        """
        if len(argv) == 1:
            only = argv[0]
            if not only.strip():
                raise ValueError("`cmd` has one element that is empty or blank.")
            if any(ch.isspace() for ch in only.strip()):
                raise ValueError(
                    f"`cmd` has one element containing spaces: {only!r}. "
                    "docker-exec does NOT run a shell, so this is a request to "
                    f"execute a binary literally named {only!r}, which returns "
                    "exit 127 'No such file or directory'. Split it into "
                    "separate elements, e.g. [\"echo\", \"probe-ok\"], or use "
                    "[\"sh\", \"-c\", \"...\"] for pipes and redirects. If the "
                    "space is part of one path, run it via "
                    "[\"sh\", \"-c\", \"'<path>'\"]."
                )
        return argv

    @staticmethod
    def _parse_cmd_argv(cmd: Any) -> List[str]:
        """
        Normalise the `cmd` argument into a plain argv list of strings.

        Open WebUI's chat model sometimes emits the argv list as a JSON
        *string* instead of a real list. A bare ``shlex.split()`` cannot handle
        that: it treats ``"`` as a quoting character and removes it, so

            '["python3", "-c", "print(1)"]'  ->  ['[python3,', '-c,', 'print(1)]']

        and the container fails with
        ``exec: "[python3,": executable file not found in $PATH`` (exit 127).
        For a single-element list it becomes ``['[python3]']`` instead.

        Strategy — accept JSON, refuse everything else, never guess:
          1. list/tuple  -> validated list of str.
          2. JSON array  -> decoded (tolerates one extra level of encoding).
          3. anything else -> ValueError naming the argv shape to send.

        Step 2 decodes with ``strict=False``. Models routinely emit a JSON array
        containing a body with *literal* control characters -- a heredoc body
        pasted with real newlines, or a tab inside an argument -- and strict
        ``json.loads`` rejects those with ``Invalid control character``. The
        array is otherwise perfectly well formed, so refusing it produces a
        message ("could not be parsed") that tells the caller to do what it
        already did. Measured on the live tool:

            '["sh","-c","cat > /tmp/x <<EOF\nline1\nEOF"]'   (literal \n)
              strict=True   -> JSONDecodeError -> refused
              strict=False  -> parses to 3 correct elements

        ``strict=False`` permits control characters *inside string literals*
        only. It does not accept trailing commas, single quotes, or unquoted
        keys, and every path still ends in ``_validate_argv``, so it cannot
        admit a shape that ``strict=True`` correctly refused for a structural
        reason.

        There is deliberately NO shell-string fallback (removed in v1.6.0). The
        previous version had a fourth step that ran ``shlex.split`` on a bare
        command string. It was the only parse path that could produce *wrong
        output without reporting a failure*: shlex removes every double quote
        and then word-splits on the spaces that were inside them, so a ``sh -c``
        body with nested quotes was cut short and the remainder became extra
        elements the shell ignores as positional parameters. Measured on the
        removed code:

            'sh -c "printf "line1\\nline2\\n" > /tmp/f"'
              -> ['sh', '-c', 'printf line1nline2n > /tmp/f']
              -> exit_code 0, empty stderr, WRONG bytes written

        Every other refusal here names the shape to send. That path instead
        obeyed the caller's intent silently and wrongly, which is the defect
        class D-45 exists to prevent. It was originally kept because a model
        might emit a raw shell string rather than an array; the tool description
        has told models to send an argv list since #28, so the convenience no
        longer outweighs a silent wrong answer.

        Raises:
            ValueError: if the input cannot be turned into a reliable argv.
        """
        if isinstance(cmd, (list, tuple)):
            argv = list(cmd)
            if not all(isinstance(a, str) for a in argv):
                raise ValueError("`cmd` list must contain only strings.")
            return Tools._validate_argv(argv)

        if not isinstance(cmd, str):
            raise ValueError(
                f"`cmd` must be a list of strings (got {type(cmd).__name__}): "
                'e.g. ["curl", "-s", "http://example.com"]'
            )

        text = cmd.strip()
        # Tolerate a model that JSON-encodes its own JSON string.
        for _ in range(2):
            try:
                # strict=False: accept literal newlines/tabs inside string
                # literals. Models paste heredoc bodies with real linebreaks,
                # and strict=True rejects those as "Invalid control character"
                # even though the array is otherwise well formed. Structural
                # errors are still rejected -- see the docstring.
                decoded = json.loads(text, strict=False)
            except ValueError:
                break
            if isinstance(decoded, list):
                if not all(isinstance(a, str) for a in decoded):
                    raise ValueError("`cmd` JSON array must contain only strings.")
                return Tools._validate_argv(decoded)
            if isinstance(decoded, str) and decoded.strip() != text:
                text = decoded.strip()
                continue
            break

        # Looked like a JSON array but did not decode. Refusing is much safer
        # than guessing at what the caller meant.
        if text.startswith('["') or text.startswith("['"):
            raise ValueError(
                "`cmd` looks like a JSON array but could not be parsed. "
                "Pass a real JSON array of strings instead. Inside a JSON "
                "string every double quote must be escaped as \\u0022."
            )

        # A bare command string. This used to be split with shlex, which
        # silently mis-split any nested quotes -- see the docstring. Refusing
        # is the point: a string reaching this line cannot be split faithfully,
        # and splitting it anyway yields a wrong answer with exit_code 0.
        raise ValueError(
            "`cmd` was a bare string, not an argv list. It cannot be split "
            "faithfully: a shell-style split drops the double quotes and then "
            "splits on the spaces that were inside them, so a nested-quote "
            "payload is cut short and writes wrong output while still exiting "
            "0. Send an argv list, one element per word, as a real JSON "
            'array -- e.g. ["sh", "-c", "<your script>"] or ["hostname"].'
        )

    # ── READ-ONLY ───────────────────────────────────────────────────

    async def docker_list(self, all: bool = False, __chat_id__: str = "") -> Any:
        """
        List all managed Docker containers.
        Returns name, image, status, and ports for each container.
        Use this first to see what is running before any other Docker operation.

        Args:
            all: If true, also list STOPPED containers. Without this a container
                 stopped with docker_stop is invisible, and its name can never be
                 discovered again (which used to make it impossible to start).
        """
        all = self._coerce_bool(all)
        blocked = self._check_guard(__chat_id__, "docker_list", all=all)
        if blocked:
            return blocked
        params = {"all": "true"} if all else None
        result = await self._call("GET", "/docker/ps", timeout=10, params=params)
        if not self._is_error(result):
            self._record_success(__chat_id__, "docker_list", all=all)
        return result

    async def docker_stats(self, name: str = "", __chat_id__: str = "") -> Any:
        """
        CPU, memory, block I/O, and network usage per managed container.
        Pass a container name to check one, or leave empty for all running managed containers.

        Args:
            name: Container name (optional). Empty = all managed containers.
        """
        # Normalise: strip and treat "" the same.
        name = name.strip() if name else ""
        blocked = self._check_guard(__chat_id__, "docker_stats", name=name)
        if blocked:
            return blocked
        params = {"name": name} if name else None
        result = await self._call("GET", "/docker/stats", timeout=60, params=params)
        if not self._is_error(result):
            self._record_success(__chat_id__, "docker_stats", name=name)
        return result

    async def docker_logs(self, name: str, lines: int = 0, __chat_id__: str = "") -> Any:
        """
        Tail recent log output for a managed container.
        Requires the container name. Returns up to `lines` lines of stdout+stderr.

        Args:
            name: Container name (required).
            lines: Number of lines to tail. 0 = use default from valves.
        """
        # Normalise before guard.
        name = name.strip() if name else ""
        if not name:
            return {"status": "error", "detail": "`name` is required."}
        lines = self._coerce_int(lines, 0)
        if lines <= 0:
            lines = self.valves.default_log_lines
        lines = max(1, min(lines, 5000))
        blocked = self._check_guard(__chat_id__, "docker_logs", name=name, lines=lines)
        if blocked:
            return blocked
        result = await self._call(
            "GET", f"/docker/logs/{quote(name, safe='')}", timeout=20, params={"lines": lines}
        )
        if not self._is_error(result):
            self._record_success(__chat_id__, "docker_logs", name=name, lines=lines)
            note = self._truncation_note(result)
            if note:
                result["note"] = note
        return result

    # ── COMMAND EXECUTION ───────────────────────────────────────────

    async def docker_exec(
        self,
        name: str,
        cmd: List[str],
        workdir: Optional[str] = None,
        user: Optional[str] = None,
        timeout: int = 0,
        __chat_id__: str = "",
        __event_call__=None,
    ) -> Any:
        """
        Run a single command inside a managed container and return its output.
        The command is passed as an argv list (not through a shell), so shell
        metacharacters are not interpreted. To use pipes or redirects, call
        ["sh", "-c", "..."] explicitly.

        Use this for connectivity tests (curl, ping), health checks, file
        inspection, and general debugging inside containers.

        It is always safe to retry: the server never re-executes a command, so a
        timeout or truncation cannot cause a second run with side effects.

        Args:
            name: Container name (required).
            cmd: (required) Non-empty argv list with ONE ELEMENT PER WORD, e.g.
                 ["curl", "-s", "http://example.com"] or
                 ["ping", "-c", "1", "docker-exec"].
                 ["echo", "hello"] is correct. ["echo hello"] is NOT: that is a
                 single element, so it searches for a binary whose name really
                 is "echo hello" and returns exit 127.
                 For pipes or redirects pass ["sh", "-c", "cmd1 | cmd2"].
                 A bare string is REFUSED, and so is a stringified array like
                 "[\"curl\", \"-s\"]". Neither can be split faithfully: a
                 shell-style split drops the double quotes and re-splits on the
                 spaces inside them, which silently writes the wrong output
                 while still exiting 0. Send a real JSON array.
            workdir: Optional working directory inside the container.
            user: Optional user to run as inside the container.
            timeout: Max seconds before the command is killed. 0 = use default from valves.

        Returns:
            {"name", "exit_code", "stdout", "stderr", "timed_out",
             "stdout_truncated", "stderr_truncated", "bytes_total_out",
             "bytes_total_err", "duration_s", "timeout_enforced", "hint"}
            `timeout_enforced: false` means the container has no `timeout`
            binary, so the command is not killed at the deadline and may still be
            running after this returns.
            `hint` is null unless the service can explain a baffling result; when
            set, it says what to change. A non-null `hint` means the command shape
            was wrong, not that the container is broken.
            A 127 from `stderr` may be preceded by `timeout: failed to run command
            ...`. That prefix is the service's kill wrapper, NOT evidence that it
            joined your arguments.
        """
        # `cmd` is REQUIRED in the signature, deliberately, and that is a
        # reversal of the PR #28 fix.
        #
        # #28 gave `cmd` a default of None to turn the
        # `missing 1 required positional argument` TypeError into a friendly
        # message. That reasoning was wrong in a way that made the tool WORSE:
        # Open WebUI derives the JSON schema it shows the model directly from
        # the Python defaults (`convert_function_to_pydantic_model`:
        # `default_value = param.default if param.default is not param.empty
        # else ...`), so a default dropped `cmd` from `required`. The model was
        # then TOLD `cmd` was optional:
        #
        #     required = ['name']
        #     cmd = {"type": "array", "items": {"type": "string"}}
        #
        # Models duly omitted it, and the tool returned "`cmd` was not provided"
        # instead of running the command. That is the intermittency reported as
        # "docker_exec drops cmd on valid payloads": the payload the model meant
        # to send was never sent, because the schema invited it not to.
        #
        # It was also unnecessary. The OWUI call site is
        # `result = await function(**params)` inside `try/except Exception`, so
        # an omitted argument is caught and returned as `{"error": ...}`. It was
        # never the silently-unreported crash #28 assumed.
        #
        # With no default pydantic marks it required, so the model is obliged to
        # send it; and `None` sent explicitly is still answered below with an
        # actionable message.
        if cmd is None:
            logger.warning("docker_exec: called without `cmd` (name=%r)", name)
            return {
                "status": "error",
                "detail": (
                    "`cmd` was not provided, and this tool needs both `name` and "
                    "`cmd`. Retry with an argv list, e.g. "
                    'docker_exec(name="<container>", cmd=["echo", "ok"])'
                ),
            }

        # Models sometimes pass the argv list as a JSON string. Decode that as
        # JSON — shlex.split() would strip the quotes and turn
        # '["python3", "-c", "print(1)"]' into ['[python3,', '-c,', 'print(1)]'].
        try:
            cmd = self._parse_cmd_argv(cmd)
        except ValueError as e:
            logger.warning("docker_exec: rejected unparseable `cmd` (%r): %s", cmd, e)
            return {
                "status": "error",
                "detail": (
                    f"Invalid `cmd`: {e} "
                    'Retry with a JSON array of strings, e.g. ["python3", "-c", "print(1)"].'
                ),
            }
        if not cmd:
            return {"status": "error", "detail": "`cmd` must be a non-empty list."}

        # Normalise before guard key.
        name = name.strip() if name else ""
        if not name:
            return {"status": "error", "detail": "`name` is required."}
        workdir = workdir.strip() if workdir else None
        user = user.strip() if user else None
        timeout = self._coerce_int(timeout, 0)
        if timeout <= 0:
            timeout = self.valves.default_exec_timeout
        timeout = max(1, min(timeout, 600))

        blocked = self._check_guard(
            __chat_id__, "docker_exec",
            name=name, cmd=cmd, workdir=workdir, user=user, timeout=timeout,
        )
        if blocked:
            return blocked

        if self.valves.confirm_exec:
            shown = " ".join(shlex.quote(c) for c in cmd)
            ok = await self._confirm(
                __event_call__,
                "Run command in container?",
                f"Container: {name}\nUser: {user or 'default'}\nWorkdir: {workdir or 'default'}\n"
                f"Timeout: {timeout}s\n\n$ {shown}",
            )
            if not ok:
                return {"status": "cancelled", "detail": "User declined or confirmation unavailable."}

        payload = {"cmd": cmd, "workdir": workdir, "user": user, "timeout": timeout}
        result = await self._call(
            "POST", f"/docker/exec/{quote(name, safe='')}", timeout=timeout + 15, json=payload
        )
        if not self._is_error(result) and isinstance(result, dict):
            self._record_success(
                __chat_id__, "docker_exec",
                name=name, cmd=cmd, workdir=workdir, user=user, timeout=timeout,
            )
            # Surface truncation / unenforced timeouts so the model does not
            # treat a partial result as a complete one.
            notes = []
            trunc = self._truncation_note(result)
            if trunc:
                notes.append(trunc)
            if result.get("timed_out"):
                notes.append(f"the command hit the {timeout}s timeout")
            if result.get("timeout_enforced") is False:
                notes.append(
                    "this container has no `timeout` binary, so the command was "
                    "NOT killed and may still be running"
                )
            if notes:
                result["note"] = "; ".join(notes)
        return result

    # ── DESTRUCTIVE (UI confirmation required) ──────────────────────

    async def docker_start(self, name: str, __chat_id__: str = "", __event_call__=None) -> Any:
        """
        Start a stopped managed container.
        The user is asked to confirm in the UI before anything happens.

        Args:
            name: Container name to start.
        """
        name = name.strip() if name else ""
        if not name:
            return {"status": "error", "detail": "`name` is required."}
        blocked = self._check_guard(__chat_id__, "docker_start", name=name)
        if blocked:
            return blocked
        ok = await self._confirm(
            __event_call__,
            "Start container?",
            f"Start '{name}'?",
        )
        if not ok:
            return {"status": "cancelled", "detail": "User declined or confirmation unavailable."}
        result = await self._call("POST", f"/docker/start/{quote(name, safe='')}", timeout=45)
        if not self._is_error(result):
            self._record_success(__chat_id__, "docker_start", name=name)
        return result

    async def docker_restart(self, name: str, __chat_id__: str = "", __event_call__=None) -> Any:
        """
        Restart a managed container. Causes brief downtime.
        The user is asked to confirm in the UI before anything happens.

        Args:
            name: Container name to restart.
        """
        name = name.strip() if name else ""
        if not name:
            return {"status": "error", "detail": "`name` is required."}
        blocked = self._check_guard(__chat_id__, "docker_restart", name=name)
        if blocked:
            return blocked
        ok = await self._confirm(
            __event_call__,
            "Restart container?",
            f"Restart '{name}'? This causes brief downtime.",
        )
        if not ok:
            return {"status": "cancelled", "detail": "User declined or confirmation unavailable."}
        result = await self._call("POST", f"/docker/restart/{quote(name, safe='')}", timeout=45)
        if not self._is_error(result):
            self._record_success(__chat_id__, "docker_restart", name=name)
        return result

    async def docker_stop(self, name: str, __chat_id__: str = "", __event_call__=None) -> Any:
        """
        Stop a managed container. Use docker_start to bring it back.
        The user is asked to confirm in the UI before anything happens.

        Args:
            name: Container name to stop.
        """
        name = name.strip() if name else ""
        if not name:
            return {"status": "error", "detail": "`name` is required."}
        blocked = self._check_guard(__chat_id__, "docker_stop", name=name)
        if blocked:
            return blocked
        ok = await self._confirm(
            __event_call__,
            "Stop container?",
            f"Stop '{name}'? It stays down until started again with docker_start.",
        )
        if not ok:
            return {"status": "cancelled", "detail": "User declined or confirmation unavailable."}
        result = await self._call("POST", f"/docker/stop/{quote(name, safe='')}", timeout=45)
        if not self._is_error(result):
            self._record_success(__chat_id__, "docker_stop", name=name)
        return result

    async def docker_remove(self, name: str, force: bool = False, __chat_id__: str = "", __event_call__=None) -> Any:
        """
        Permanently remove a managed container.
        The user is asked to confirm in the UI before anything happens.

        Args:
            name: Container name to remove.
            force: If true, force removal even if running.
        """
        name = name.strip() if name else ""
        if not name:
            return {"status": "error", "detail": "`name` is required."}
        force = self._coerce_bool(force)
        blocked = self._check_guard(__chat_id__, "docker_remove", name=name, force=force)
        if blocked:
            return blocked
        ok = await self._confirm(
            __event_call__,
            "Remove container?",
            f"Permanently remove '{name}'"
            + (" (FORCE: kills it if running)" if force else "")
            + "? This cannot be undone.",
        )
        if not ok:
            return {"status": "cancelled", "detail": "User declined or confirmation unavailable."}
        result = await self._call(
            "DELETE", f"/docker/rm/{quote(name, safe='')}", timeout=45,
            params={"force": str(force).lower()},
        )
        if not self._is_error(result):
            self._record_success(__chat_id__, "docker_remove", name=name, force=force)
        return result
