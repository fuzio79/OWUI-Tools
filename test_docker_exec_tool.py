"""Regression tests for docker-exe-openwebui-tool.py — `cmd` parsing.

The original bug: Open WebUI's chat model intermittently serialises the
`cmd` argument as a JSON *string* instead of a list. The tool's fallback was
a bare ``shlex.split()``, which strips the JSON double quotes and leaves the
brackets glued to the first token:

    '["python3", "-c", "print(1)"]'  ->  ['[python3,', '-c,', 'print(1)]']

so argv[0] became the literal string ``[python3,`` and the container returned

    OCI runtime exec failed: exec: "[python3,": executable file not found in $PATH

These tests pin the corrected behaviour: JSON arrays are decoded as JSON, and
anything else fails *closed* with ``{"status": "error"}`` instead of being
guessed at. There is deliberately no shell-string fallback -- see
``BareStringIsRefusedTests`` and issue #32.

Run:
    python -m unittest test_docker_exec_tool -v
    pytest test_docker_exec_tool.py -v

Requires ``pydantic``. ``requests`` is imported by the tool but never called
here — every HTTP path is stubbed.
"""

import asyncio
import importlib.util
import json
import shlex
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TOOL_PATH = _HERE / "docker-exe-openwebui-tool.py"

# Import by path: the module filename contains hyphens, so a normal import
# statement cannot be used. Mirrors the pattern used by test_research_tool.py.
_spec = importlib.util.spec_from_file_location("docker_exe_tool", _TOOL_PATH)
tool_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool_mod)


def _stub_tool():
    """A Tools instance with HTTP and confirmation stubbed out."""
    tools = tool_mod.Tools()
    tools.valves.api_key = "test-key"
    tools.valves.confirm_exec = False  # no UI in tests
    tools.calls = []

    def fake_request(method, path, timeout=15, **kwargs):
        tools.calls.append({"method": method, "path": path, "kwargs": kwargs})
        return {
            "name": "x",
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
            "stdout_truncated": False,
            "stderr_truncated": False,
            "duration_s": 0.0,
        }

    tools._request = fake_request
    return tools


def _run(coro):
    return asyncio.run(coro)


def _sent_cmd(tools):
    """The `cmd` actually placed in the POST body."""
    assert len(tools.calls) == 1, tools.calls
    return tools.calls[0]["kwargs"]["json"]["cmd"]


class ParseCmdArgvTests(unittest.TestCase):
    """_parse_cmd_argv — the pure parsing layer."""

    def setUp(self):
        self.tools = _stub_tool()

    def parse(self, cmd):
        return self.tools._parse_cmd_argv(cmd)

    # ── already a list: returned as an equal list, never touched ──────

    def test_list_passthrough(self):
        cmd = ["python3", "-c", "print(1)"]
        self.assertEqual(self.parse(cmd), cmd)

    def test_tuple_accepted(self):
        self.assertEqual(self.parse(("echo", "ok")), ["echo", "ok"])

    def test_list_is_copied_not_aliased(self):
        cmd = ["echo", "ok"]
        out = self.parse(cmd)
        self.assertEqual(out, cmd)
        self.assertIsNot(out, cmd)

    def test_single_element_list(self):
        self.assertEqual(self.parse(["hostname"]), ["hostname"])

    def test_empty_list_is_allowed(self):
        # Emptiness is rejected later by docker_exec with a clear message.
        self.assertEqual(self.parse([]), [])

    # ── the regression: JSON strings ─────────────────────────────────

    def test_json_string_spaced(self):
        """The exact shape that produced '[python3,' as argv[0]."""
        self.assertEqual(
            self.parse('["python3", "-c", "print(1)"]'),
            ["python3", "-c", "print(1)"],
        )

    def test_json_string_no_spaces(self):
        self.assertEqual(self.parse('["ls","-la"]'), ["ls", "-la"])

    def test_json_string_single_element(self):
        """Used to become ['[python3]'] — the second reported symptom."""
        self.assertEqual(self.parse('["python3"]'), ["python3"])

    def test_json_string_with_spaces_inside_args(self):
        self.assertEqual(
            self.parse(json.dumps(["sh", "-c", "echo hi && echo bye"])),
            ["sh", "-c", "echo hi && echo bye"],
        )

    def test_json_string_with_pipe(self):
        cmd = json.dumps(["sh", "-c", "curl -s localhost:8000 | head -c 300"])
        self.assertEqual(self.parse(cmd), ["sh", "-c", "curl -s localhost:8000 | head -c 300"])

    def test_json_string_surrounded_by_whitespace(self):
        self.assertEqual(self.parse('  ["ls","-la"]  '), ["ls", "-la"])

    def test_double_encoded_json(self):
        """A model that JSON-encodes its own JSON string still decodes once."""
        self.assertEqual(self.parse(json.dumps(json.dumps(["ls", "-la"]))), ["ls", "-la"])

    def test_json_empty_array(self):
        self.assertEqual(self.parse("[]"), [])

    def test_json_unicode(self):
        self.assertEqual(self.parse('["echo", "héllo"]'), ["echo", "héllo"])

    # ── a bare command string is REFUSED, not split ──────────────────
    #
    # These five used to assert that ``shlex.split`` handled a shell-style
    # string. That fallback was removed in v1.6.0 (#32) because it was the only
    # parse path that could produce wrong output while reporting success. The
    # tests now assert the refusal, and the message is checked for the argv
    # shape to send instead.

    def test_plain_command_string_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self.parse("hostname")
        self.assertIn("argv list", str(ctx.exception))

    def test_shell_string_with_quotes_is_refused(self):
        with self.assertRaises(ValueError):
            self.parse('sh -c "echo hi"')

    def test_shell_string_with_nested_quotes_is_refused_and_says_why(self):
        """The reported shape: quoting inside a quoted script."""
        with self.assertRaises(ValueError) as ctx:
            self.parse('curl -s -X POST "http://x/y" -d {"a": 1}')
        msg = str(ctx.exception)
        self.assertIn("double quotes", msg)
        self.assertIn("exit", msg, "must say the old path still exited 0")

    def test_bracket_test_builtin_is_refused_not_shell_split(self):
        """`[ -f /etc/hostname ]` is not JSON, and is no longer shlex-split."""
        with self.assertRaises(ValueError):
            self.parse("[ -f /etc/hostname ]")

    def test_string_containing_brackets_is_refused(self):
        with self.assertRaises(ValueError):
            self.parse("echo [ok]")

    def test_refusal_names_an_argv_example(self):
        """A refusal the model cannot act on is barely better than none."""
        with self.assertRaises(ValueError) as ctx:
            self.parse("echo hi")
        self.assertIn('["sh", "-c"', str(ctx.exception))

    # ── fail closed, never guess ─────────────────────────────────────

    def test_malformed_json_array_raises(self):
        with self.assertRaises(ValueError):
            self.parse('["python3", "-c", "print(1)",]')

    def test_single_quoted_array_raises(self):
        """Single quotes are not JSON. Refused rather than guessed at."""
        with self.assertRaises(ValueError):
            self.parse("['python3', '-c']")

    def test_json_array_of_non_strings_raises(self):
        with self.assertRaises(ValueError):
            self.parse('["python3", 42]')

    def test_list_of_non_strings_raises(self):
        with self.assertRaises(ValueError):
            self.parse(["python3", 42])

    def test_unbalanced_quote_raises(self):
        with self.assertRaises(ValueError):
            self.parse('sh -c "oops')

    def test_non_string_non_list_raises(self):
        for bad in (None, 42, {"cmd": ["ls"]}, b"ls"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.parse(bad)


class DockerExecIntegrationTests(unittest.TestCase):
    """docker_exec end-to-end: what actually lands in the POST body."""

    def test_json_string_reaches_service_as_list(self):
        tools = _stub_tool()
        _run(tools.docker_exec("c1", '["python3", "-c", "print(1)"]'))
        self.assertEqual(_sent_cmd(tools), ["python3", "-c", "print(1)"])

    def test_json_single_element_reaches_service_as_list(self):
        tools = _stub_tool()
        _run(tools.docker_exec("c1", '["hostname"]'))
        self.assertEqual(_sent_cmd(tools), ["hostname"])

    def test_shlex_never_produces_bracket_token(self):
        """The precise failure mode: no argv element may start with '['."""
        tools = _stub_tool()
        _run(tools.docker_exec("c1", '["python3", "-c", "print(1)"]'))
        for arg in _sent_cmd(tools):
            self.assertFalse(
                arg.startswith("["),
                f"argv element {arg!r} still carries a JSON bracket",
            )

    def test_real_list_still_passes_through(self):
        tools = _stub_tool()
        _run(tools.docker_exec("c1", ["echo", "ok"]))
        self.assertEqual(_sent_cmd(tools), ["echo", "ok"])

    def test_shell_string_is_refused_before_any_http_call(self):
        """The removed fallback must not quietly come back as a code path."""
        tools = _stub_tool()
        result = _run(tools.docker_exec("c1", 'sh -c "echo hi"'))
        self.assertEqual(result["status"], "error")
        self.assertEqual(tools.calls, [], "nothing should reach the service")

    def test_workdir_and_user_forwarded(self):
        tools = _stub_tool()
        _run(tools.docker_exec("c1", '["ls"]', workdir="/app", user="root", timeout=7))
        body = tools.calls[0]["kwargs"]["json"]
        self.assertEqual(body["workdir"], "/app")
        self.assertEqual(body["user"], "root")
        self.assertEqual(body["timeout"], 7)

    def test_empty_cmd_rejected(self):
        tools = _stub_tool()
        result = _run(tools.docker_exec("c1", "[]"))
        self.assertEqual(result["status"], "error")
        self.assertEqual(tools.calls, [], "nothing should reach the service")

    def test_unparseable_cmd_rejected_without_http(self):
        tools = _stub_tool()
        result = _run(tools.docker_exec("c1", '["python3", 42]'))
        self.assertEqual(result["status"], "error")
        self.assertEqual(tools.calls, [])

    def test_missing_name_rejected(self):
        tools = _stub_tool()
        result = _run(tools.docker_exec("   ", ["ls"]))
        self.assertEqual(result["status"], "error")
        self.assertEqual(tools.calls, [])

    def test_parse_error_does_not_trip_loop_guard(self):
        """A rejected call must not poison the guard for the corrected retry."""
        tools = _stub_tool()
        _run(tools.docker_exec("c1", '["python3", 42]'))
        result = _run(tools.docker_exec("c1", '["python3", "-c", "print(1)"]'))
        self.assertNotEqual(result.get("status"), "loop_guard")
        self.assertEqual(_sent_cmd(tools), ["python3", "-c", "print(1)"])




class SingleElementWithSpacesTests(unittest.TestCase):
    """The 127 class: one argv element containing spaces.

    Observed in production as:
        cmd: ["echo probe-ok"]
        -> exit_code 127
        -> stderr "timeout: failed to run command 'echo probe-ok':
                   No such file or directory"

    docker-exec runs no shell, so that space is part of the executable name.
    The tool must refuse it rather than forward it.
    """

    def setUp(self):
        self.tools = _stub_tool()

    def parse(self, cmd):
        return self.tools._parse_cmd_argv(cmd)

    def test_the_exact_production_payload_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self.parse(["echo probe-ok"])
        msg = str(ctx.exception)
        self.assertIn("spaces", msg)
        self.assertIn("127", msg, "the error must name the symptom the user sees")

    def test_refused_before_any_http_call(self):
        """A refused argv must never reach the service."""
        result = _run(self.tools.docker_exec("c1", ["echo probe-ok"]))
        self.assertEqual(result["status"], "error")
        self.assertEqual(self.tools.calls, [], "nothing should reach the service")

    def test_error_tells_the_model_the_corrected_shape(self):
        result = _run(self.tools.docker_exec("c1", ["echo probe-ok"]))
        detail = result["detail"]
        self.assertIn("sh", detail, "must offer the sh -c escape hatch")
        self.assertIn("echo", detail)

    def test_two_element_form_still_works(self):
        self.assertEqual(self.parse(["echo", "hello"]), ["echo", "hello"])

    def test_sh_dash_c_is_unaffected(self):
        """Three elements: never triggers the single-element check."""
        cmd = ["sh", "-c", "echo a b | wc -c"]
        self.assertEqual(self.parse(cmd), cmd)

    def test_python_dash_c_is_unaffected(self):
        cmd = ["python3", "-c", "print(1)"]
        self.assertEqual(self.parse(cmd), cmd)

    def test_single_element_without_spaces_is_fine(self):
        for cmd in (["hostname"], ["true"], ["ls"]):
            self.assertEqual(self.parse(cmd), cmd)

    def test_single_empty_or_blank_element_is_refused(self):
        for bad in ([""], ["   "]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.parse(bad)

    def test_single_element_with_tab_is_refused(self):
        """isspace() catches tabs, not just spaces."""
        with self.assertRaises(ValueError):
            self.parse(["echo\thi"])

    def test_validation_covers_the_json_array_path(self):
        """A stringified array decodes to one spaced element -> still refused.

        Without this the check could be applied only to the list branch and the
        JSON branch would smuggle the bad shape through.
        """
        with self.assertRaises(ValueError):
            self.parse('["echo probe-ok"]')

    def test_validation_covers_the_bare_string_path(self):
        """A spaced command string must never become a single spaced element.

        Before v1.6.0 this payload reached ``_validate_argv`` via ``shlex``.
        Now it is refused earlier, as a bare string. Either way it must not be
        forwarded -- only the refusal point moved.
        """
        with self.assertRaises(ValueError):
            self.parse("'echo probe-ok'")

    def test_positive_control_double_quoted_string_is_now_refused(self):
        """This used to JSON-unwrap to a correct argv. That path is gone.

        Recorded rather than deleted: it was the one shell-string shape that
        DID split correctly, so its removal is the real cost of dropping the
        fallback and a future reader should see that explicitly.
        """
        with self.assertRaises(ValueError):
            self.parse('"echo probe-ok"')

    def test_positive_control_shell_string_is_now_refused(self):
        """An unquoted command string used to split. Now refused."""
        with self.assertRaises(ValueError):
            self.parse("echo probe-ok")

    def test_the_list_form_of_the_same_command_still_works(self):
        """The positive control that replaces them: same intent, list shape."""
        self.assertEqual(self.parse(["echo", "probe-ok"]), ["echo", "probe-ok"])


class MissingCmdTests(unittest.TestCase):
    """`cmd` is REQUIRED in the signature; an explicit None is handled.

    PR #28 gave `cmd` a default of None so that a call carrying only `name`
    produced a friendly message instead of
    `Tools.docker_exec() missing 1 required positional argument: 'cmd'`.

    That was wrong twice over, and this class now pins the corrected shape:

    1. It made the tool worse. Open WebUI derives the schema the MODEL sees
       from the Python defaults, so a default removed `cmd` from `required`
       and the model was actively told it could omit it. Models obliged.
    2. The crash it was avoiding was already handled: OWUI calls
       `await function(**params)` inside `try/except Exception`, so a missing
       argument became `{"error": ...}` rather than an unreported failure.

    The lesson pinned here: a tool signature is not just a Python API, it is
    the schema, and a defensive default is a behavioural change to what every
    model is told.
    """

    def setUp(self):
        self.tools = _stub_tool()

    def test_cmd_is_required_in_the_signature(self):
        """The property that keeps `cmd` out of "optional" in the schema.

        Asserted on the signature because that is what OWUI converts. A test
        that only called the method could not see this regression.
        """
        import inspect

        sig = inspect.signature(self.tools.docker_exec)
        self.assertIs(
            sig.parameters["cmd"].default,
            inspect.Parameter.empty,
            "`cmd` must have NO default: any default removes it from the "
            "required list in the schema the model is shown",
        )

    def test_schema_lists_cmd_as_required(self):
        """The end-to-end property, built the way OWUI builds it.

        This is the assertion whose absence let #28 ship: the signature was
        inspected for "has a default", never for "appears in required".
        """
        from typing import Any

        from pydantic import create_model

        import inspect as _inspect

        sig = _inspect.signature(self.tools.docker_exec)
        fields = {}
        for name, param in sig.parameters.items():
            if name == "self" or name.startswith("__"):
                continue
            ann = param.annotation
            if ann is _inspect.Parameter.empty:
                ann = Any
            default = ... if param.default is _inspect.Parameter.empty else param.default
            fields[name] = (ann, default)

        model = create_model("docker_exec", **fields)
        required = model.model_json_schema().get("required", [])
        self.assertIn(
            "cmd",
            required,
            f"`cmd` is not in the schema's required list ({required}); "
            "the model is being told it may omit the command",
        )

    def test_explicit_none_is_rejected_with_a_message(self):
        """`None` passed explicitly must not reach the service."""
        result = _run(self.tools.docker_exec("c1", None))
        self.assertEqual(result["status"], "error")
        self.assertIn("cmd", result["detail"])
        self.assertEqual(self.tools.calls, [])

    def test_explicit_none_does_not_trip_loop_guard(self):
        """A rejected call must not poison the guard for the corrected retry."""
        _run(self.tools.docker_exec("c1", None))
        result = _run(self.tools.docker_exec("c1", ["echo", "ok"]))
        self.assertNotEqual(result.get("status"), "loop_guard")


class GuardCountsFailuresTests(unittest.TestCase):
    """Repetition of a FAILING call must trip the guard.

    A Docker exec result carries `exit_code`, not `status`, so it is not an
    error dict. That is deliberate: three identical non-zero exits are exactly
    the runaway pattern the guard exists to stop. This pins the behaviour so a
    future "a failure is an error" change cannot silently blind it.
    """

    def setUp(self):
        # The guard is a module-level singleton, so state leaks between tests.
        # Without this the class is order-dependent: whichever test runs first
        # consumes the repeat budget and the next one sees 0 HTTP calls.
        tool_mod._guard._recent.clear()

    def _failing_tools(self):
        tools = _stub_tool()
        tools.valves.loop_guard_max_repeats = 3
        tools.valves.loop_guard_window = 300

        def failing_request(method, path, timeout=15, **kwargs):
            tools.calls.append({"method": method, "path": path, "kwargs": kwargs})
            return {
                "name": "x",
                "exit_code": 127,
                "stdout": "",
                "stderr": "No such file or directory",
                "timed_out": False,
                "duration_s": 0.0,
            }

        tools._request = failing_request
        return tools

    def test_nonzero_exit_is_not_an_error_dict(self):
        tools = self._failing_tools()
        self.assertFalse(tools._is_error({"name": "x", "exit_code": 127}))

    def test_three_identical_failures_trip_the_guard(self):
        tools = self._failing_tools()
        for _ in range(3):
            _run(tools.docker_exec("c1", ["nope"], __chat_id__="chat-1"))
        blocked = _run(tools.docker_exec("c1", ["nope"], __chat_id__="chat-1"))
        self.assertEqual(blocked.get("status"), "loop_guard")
        self.assertEqual(
            len(tools.calls), 3, "the 4th call must be short-circuited, not sent"
        )

    def test_guard_is_scoped_per_chat(self):
        tools = self._failing_tools()
        for _ in range(3):
            _run(tools.docker_exec("c1", ["nope"], __chat_id__="chat-1"))
        other = _run(tools.docker_exec("c1", ["nope"], __chat_id__="chat-2"))
        self.assertNotEqual(other.get("status"), "loop_guard")

    def test_real_error_dict_is_not_counted(self):
        """A transport error must NOT count, or the guard blocks recovery."""
        tools = _stub_tool()
        tools._request = lambda *a, **k: {"status": "error", "detail": "boom"}
        for _ in range(4):
            result = _run(tools.docker_exec("c1", ["echo"], __chat_id__="chat-3"))
            self.assertEqual(result.get("status"), "error")


class LiteralControlCharacterTests(unittest.TestCase):
    """A JSON-string array containing LITERAL newlines/tabs must be accepted.

    The reported symptom was:

        cmd: '["sh","-c","cat > /tmp/x <<EOF\nline1\nEOF"]'   (REAL newlines)
        -> Invalid `cmd`: `cmd` looks like a JSON array but could not be parsed

    ``json.loads`` defaults to ``strict=True``, which rejects raw control
    characters inside string literals with "Invalid control character". The
    array is otherwise well formed, so the refusal message told the caller to
    do exactly what it had already done.

    Models hit this constantly: a heredoc body pasted into an argument carries
    real linebreaks, and ``json.dumps`` is not involved in producing it.

    ``strict=False`` relaxes ONLY the control-character rule. Everything that
    is structurally invalid must still be refused, which the negative
    controls below pin -- otherwise this fix would be a hole rather than a
    narrowing.
    """

    def setUp(self):
        self.tools = _stub_tool()

    def parse(self, cmd):
        return self.tools._parse_cmd_argv(cmd)

    # ── the regression ───────────────────────────────────────────────

    def test_literal_newlines_in_json_string_are_accepted(self):
        nl = chr(10)
        cmd = '["sh", "-c", "cat > /tmp/x <<EOF' + nl + 'line1' + nl + 'EOF"]'
        self.assertEqual(
            self.parse(cmd),
            ["sh", "-c", "cat > /tmp/x <<EOF" + nl + "line1" + nl + "EOF"],
        )

    def test_literal_tab_in_json_string_is_accepted(self):
        cmd = '["sh", "-c", "printf a' + chr(9) + 'b"]'
        self.assertEqual(self.parse(cmd), ["sh", "-c", "printf a" + chr(9) + "b"])

    def test_escaped_newlines_still_accepted(self):
        """The json.dumps form must keep working -- this is the positive control.

        Without it, a "fix" that replaced the JSON path with something looser
        could pass the literal-newline test while breaking the correct form.
        """
        cmd = json.dumps(["sh", "-c", "echo a" + chr(10) + "b"])
        self.assertEqual(self.parse(cmd), ["sh", "-c", "echo a" + chr(10) + "b"])

    def test_real_list_still_accepted(self):
        cmd = ["sh", "-c", "cat > /tmp/x <<EOF" + chr(10) + "l" + chr(10) + "EOF"]
        self.assertEqual(self.parse(cmd), cmd)

    def test_reaches_the_service_with_literal_newlines(self):
        nl = chr(10)
        body = "cat > /tmp/x <<EOF" + nl + "line1" + nl + "EOF"
        _run(self.tools.docker_exec("c1", '["sh", "-c", "' + body + '"]'))
        self.assertEqual(_sent_cmd(self.tools)[2], body)

    # ── negative controls: strict=False is NOT a general loosening ────

    def test_trailing_comma_still_refused(self):
        """strict=False does not permit trailing commas."""
        with self.assertRaises(ValueError):
            self.parse('["a", "b",]')

    def test_single_quoted_array_still_refused(self):
        with self.assertRaises(ValueError):
            self.parse("['a', 'b']")

    def test_unquoted_key_shape_still_refused(self):
        with self.assertRaises(ValueError):
            self.parse('["a", 42]')

    def test_unbalanced_quote_still_refused(self):
        with self.assertRaises(ValueError):
            self.parse('sh -c "oops')

    def test_single_element_with_space_still_refused(self):
        """The 127 guard must survive this change.

        strict=False could in principle let a spaced single element through as
        a decoded array; _validate_argv must still catch it.
        """
        with self.assertRaises(ValueError):
            self.parse('["echo probe-ok"]')

    def test_single_element_with_literal_newline_still_refused(self):
        with self.assertRaises(ValueError):
            self.parse('["echo' + chr(10) + 'probe"]')


class BareStringIsRefusedTests(unittest.TestCase):
    """Issue #32: a bare command string must be refused, not silently split.

    The removed fallback ran ``shlex.split`` on a bare string. ``shlex`` strips
    every double quote and then word-splits on the spaces that were *inside*
    them, so a nested-quote ``sh -c`` body was cut short and the remainder
    became extra argv elements. ``sh`` ignores those as positional parameters,
    so the command reported success while doing the wrong thing:

        'sh -c "printf "line1\nline2\n" > /tmp/c1.txt"'
          -> ['sh', '-c', 'printf line1nline2n > /tmp/c1.txt']
          -> exit_code 0, empty stderr, file contains 'line1nline2n'

    This class pins the *property* that makes the old code unacceptable: the
    parse must never return an argv that differs from what the caller sent
    without raising. Refusal is verified, not the absence of a string.
    """

    def setUp(self):
        self.tools = _stub_tool()

    def parse(self, cmd):
        return self.tools._parse_cmd_argv(cmd)

    # ── the exact reported payloads ──────────────────────────────────

    def test_nested_quote_sh_c_payload_is_refused(self):
        q, bs, nl = chr(34), chr(92), chr(10)
        payload = 'sh -c ' + q + 'printf ' + q + 'line1' + bs + nl + 'line2' + bs + nl + q + ' > /tmp/x' + q
        with self.assertRaises(ValueError):
            self.parse(payload)

    def test_no_wrong_argv_is_ever_returned(self):
        """The core property: no silent truncation of the script.

        Before the fix this returned a 3-element argv with the script cut at
        the first escaped quote. Asserting the raise is what makes the old
        behaviour impossible, rather than asserting a particular message.
        """
        q = chr(34)
        payload = 'sh -c ' + q + 'echo ' + q + 'a b' + q + q
        with self.assertRaises(ValueError):
            self.parse(payload)

    def test_empty_string_is_refused(self):
        for bad in ("", "   "):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.parse(bad)

    # ── the shapes that must keep working ───────────────────────────

    def test_real_list_with_nested_quotes_is_byte_exact(self):
        """A list is the supported form, and it must not be altered at all.

        This is the actual remedy for the reported bug: the same nested-quote
        payload, passed as a list, survives byte-for-byte.
        """
        q, bs, nl = chr(34), chr(92), chr(10)
        script = 'printf ' + q + 'line1' + bs + nl + 'line2' + bs + nl + q + ' > /tmp/x'
        argv = ["sh", "-c", script]
        self.assertEqual(self.parse(argv), argv)

    def test_list_with_every_awkward_character_is_byte_exact(self):
        script = 'echo "a b" ' + chr(39) + 'c d' + chr(39) + ' $HOME `x` back' + chr(92) + 'slash'
        argv = ["sh", "-c", script]
        out = self.parse(argv)
        self.assertEqual(out[-1], script, "the script must not be re-quoted")

    def test_json_array_form_still_accepted(self):
        """A stringified ARRAY is still decoded -- that path was never the bug."""
        self.assertEqual(
            self.parse(json.dumps(["sh", "-c", 'printf "a b"'])),
            ["sh", "-c", 'printf "a b"'],
        )

    # ── end to end: nothing reaches the service ──────────────────────

    def test_bare_string_never_reaches_the_service(self):
        result = _run(self.tools.docker_exec("c1", "hostname"))
        self.assertEqual(result["status"], "error")
        self.assertEqual(self.tools.calls, [], "nothing should reach the service")

    def test_bare_string_does_not_trip_the_loop_guard(self):
        """A refusal must not poison the guard for the corrected retry."""
        _run(self.tools.docker_exec("c1", "hostname", __chat_id__="chat-9"))
        result = _run(self.tools.docker_exec("c1", ["hostname"], __chat_id__="chat-9"))
        self.assertNotEqual(result.get("status"), "loop_guard")


class EmptyCmdListTests(unittest.TestCase):
    """An empty argv is refused after parsing, with a distinct message."""

    def setUp(self):
        self.tools = _stub_tool()

    def test_empty_json_array_rejected_without_http(self):
        result = _run(self.tools.docker_exec("c1", "[]"))
        self.assertEqual(result["status"], "error")
        self.assertIn("non-empty", result["detail"])
        self.assertEqual(self.tools.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
