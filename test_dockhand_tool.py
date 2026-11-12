"""Regression tests for `dockhand-openwebui-tool.py` (issue #1).

No network: the low-level `_get`/`_get_data` helpers are left alone and only
pure functions plus `container_logs` (with a stubbed `_get`) are exercised.

Run with:
    python -m unittest test_dockhand_tool -v
"""

import importlib.util
import os
import asyncio
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULE_PATH = os.path.join(_HERE, "dockhand-openwebui-tool.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("dockhand_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def await_tool(coro):
    """Drive a tool coroutine to completion."""
    return asyncio.run(coro)


class RedactionTests(unittest.TestCase):
    def setUp(self):
        self.module = _load_module()
        self.Tools = self.module.Tools

    def _tools(self, **overrides):
        t = self.Tools()
        t.valves.REDACT_SECRETS = True
        for k, v in overrides.items():
            setattr(t.valves, k, v)
        return t

    def test_secrets_are_redacted(self):
        t = self._tools()
        out = t._redact_env_list(["API_KEY=abc123", "DB_PASSWORD=hunter2"])
        self.assertEqual(out, ["API_KEY=***REDACTED***", "DB_PASSWORD=***REDACTED***"])

    def test_legitimate_keys_are_not_redacted(self):
        """Over-broad url/uri/conn/key words used to destroy these."""
        t = self._tools()
        keep = [
            "NEXT_PUBLIC_URL=https://x.example",
            "SSH_PUBLIC_KEY=ssh-rsa AAA",
            "CONNECTION_IDLE_TIMEOUT=30",
            "HEALTHCHECK_TEST_URL=http://h/",
        ]
        self.assertEqual(t._redact_env_list(list(keep)), keep)

    def test_embedded_credential_url_is_redacted(self):
        """Anchored matching missed DSNs embedded mid-value."""
        t = self._tools()
        out = t._redact_env_list(
            ["JAVA_TOOL_OPTIONS=-Durl=jdbc:mysql://user:pass@host/db"]
        )
        self.assertNotIn("pass", out[0])

    def test_log_line_credentials_are_redacted(self):
        """container_logs previously returned these verbatim."""
        t = self._tools()
        self.assertNotIn(
            "secret", t._redact_log_line("DATABASE_URL=postgres://u:secret@h/db")
        )
        self.assertNotIn(
            "hunter2", t._redact_log_line("starting with DB_PASSWORD=hunter2 now")
        )

    def test_log_line_keeps_benign_content(self):
        t = self._tools()
        line = "listening on :8080"
        self.assertEqual(t._redact_log_line(line), line)

    def test_non_string_secret_is_redacted(self):
        """A secret stored as a list used to pass through untouched."""
        t = self._tools()
        out = t._redact_obj({"api_key": ["a", "b"]})
        self.assertEqual(out["api_key"], "***REDACTED***")

    def test_none_secret_stays_none(self):
        t = self._tools()
        self.assertIsNone(t._redact_obj({"password": None})["password"])


class IsErrorTests(unittest.TestCase):
    def setUp(self):
        self.Tools = _load_module().Tools

    def test_error_with_extra_keys_is_recognised(self):
        """len(data) == 1 missed {\"error\", \"detail\"}."""
        self.assertTrue(self.Tools._is_error({"error": "boom", "detail": "x"}))

    def test_null_error_is_not_an_error(self):
        self.assertFalse(self.Tools._is_error({"error": None}))

    def test_plain_payload_is_not_an_error(self):
        self.assertFalse(self.Tools._is_error({"ok": 1}))
        self.assertFalse(self.Tools._is_error([1, 2]))


class ContainerLogsBoundsTests(unittest.TestCase):
    """container_logs must redact and must respect MAX_OUTPUT_CHARS."""

    def setUp(self):
        self.module = _load_module()
        self.Tools = self.module.Tools

    def _tools_with_body(self, body, **overrides):
        t = self.Tools()
        t.valves.REDACT_SECRETS = True
        for k, v in overrides.items():
            setattr(t.valves, k, v)

        class _Resp:
            headers = {"content-type": "application/json"}
            content = b""

            @staticmethod
            def json():
                return {"logs": body}

        async def _fake_get(*a, **k):
            return _Resp()

        t._get = _fake_get
        return t

    def test_logs_are_redacted(self):
        t = self._tools_with_body("boot DATABASE_URL=postgres://u:secret@h/db\nready")
        out = await_tool(t.container_logs("svc"))
        joined = "\n".join(out["lines"])
        self.assertNotIn("secret@", joined)

    def test_output_is_capped_by_max_output_chars(self):
        body = "\n".join("line %03d " % i + "x" * 500 for i in range(300))
        t = self._tools_with_body(body, MAX_OUTPUT_CHARS=500)
        out = await_tool(t.container_logs("svc"))
        total = sum(len(x) + 1 for x in out["lines"])
        self.assertLessEqual(total, 700)
        self.assertLess(len(out["lines"]), 300)
        self.assertTrue(any("truncated" in x for x in out["lines"]))
