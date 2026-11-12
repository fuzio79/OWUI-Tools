"""Regression tests for `pi-agent-openwebui-tool.py`.

No network: `requests.post` / `requests.get` are stubbed on the loaded module,
so these cover the validation gates and the status-code mapping — the two
places where a wrong answer silently misinforms the model.

Run with:
    python -m unittest test_pi_agent_tool -v
"""

import importlib.util
import os
import asyncio
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULE_PATH = os.path.join(_HERE, "pi-agent-openwebui-tool.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("pi_agent_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def await_tool(coro):
    """Drive a tool coroutine to completion."""
    return asyncio.run(coro)


class _Resp:
    """Minimal stand-in for a requests.Response."""

    def __init__(self, status_code, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else str(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _tools(**valve_overrides):
    Tools = _load_module().Tools
    t = Tools()
    for k, v in valve_overrides.items():
        setattr(t.valves, k, v)
    return t


def _stub_post(module, resp):
    module.requests.post = lambda *a, **k: resp


def _stub_get(module, resp):
    module.requests.get = lambda *a, **k: resp


class ModelRegexTests(unittest.TestCase):
    """The runner's _MODEL_RE allows '/', so the tool must too."""

    def setUp(self):
        self.module = _load_module()

    def test_vendor_prefixed_models_are_accepted(self):
        """models.json declares both of these for openrouter."""
        for model in ("openai/gpt-4o", "anthropic/claude-sonnet-4"):
            self.assertRegex(model, self.module._MODEL_RE)

    def test_flat_models_still_accepted(self):
        for model in ("glm-5.3-flash", "Qwen3.8-RCO", "grok-3",
                      "deepseek-v4.1-flash"):
            self.assertRegex(model, self.module._MODEL_RE)

    def test_flag_shaped_values_still_refused(self):
        for bad in ("-x", "--model", "/leading", ""):
            self.assertNotRegex(bad, self.module._MODEL_RE)

    def test_provider_pattern_has_no_slash(self):
        """Provider ids are flat; only the model field gained '/'.

        Keeping these separate is the fix — a shared pattern without '/'
        rejected every OpenRouter model at the tool gate.
        """
        self.assertNotRegex("openai/gpt-4o", self.module._PROVIDER_RE)
        for provider in ("opencode-go", "llama-cpp", "openrouter", "grok"):
            self.assertRegex(provider, self.module._PROVIDER_RE)


class ValidationGateTests(unittest.TestCase):
    def test_bare_provider_is_rejected(self):
        """A bare provider would run the default model against it."""
        out = await_tool(_tools().pi_run("write a script", provider="openrouter"))
        self.assertEqual(out["status"], "invalid")
        self.assertIn("model", out["detail"])

    def test_bare_model_is_rejected(self):
        out = await_tool(_tools().pi_run("write a script", model="glm-5.3-flash"))
        self.assertEqual(out["status"], "invalid")
        self.assertIn("provider", out["detail"])

    def test_allowed_model_passes_the_gate(self):
        """A permitted provider/model pair must reach dispatch."""
        module = _load_module()
        resp = _Resp(200, {"exit_code": 0, "stdout": "ok", "stderr": "",
                           "duration_s": 1.0, "timed_out": False})
        _stub_post(module, resp)
        t = module.Tools()
        for provider, model in (("opencode-go", "deepseek-v4.1-flash"),
                                ("opencode-go", "glm-5.3-flash")):
            out = await_tool(t.pi_run("write a script", provider=provider, model=model))
            self.assertEqual(out.get("exit_code"), 0)

    def test_disallowed_provider_is_rejected_locally(self):
        """openrouter was removed in runner v1.1.8."""
        out = await_tool(_tools().pi_run("write a script", provider="openrouter",
                              model="openai/gpt-4o"))
        self.assertEqual(out["status"], "invalid")
        self.assertIn("not allowed", out["detail"])
        self.assertIn("opencode-go", out["detail"])

    def test_disallowed_model_on_allowed_provider(self):
        out = await_tool(_tools().pi_run("write a script", provider="opencode-go", model="gpt-5"))
        self.assertEqual(out["status"], "invalid")
        self.assertIn("deepseek-v4.1-flash", out["detail"])

    def test_oversized_task_is_rejected(self):
        module = _load_module()
        out = await_tool(module.Tools().pi_run("x" * (module._MAX_TASK_CHARS + 1)))
        self.assertEqual(out["status"], "invalid")
        self.assertIn("at most", out["detail"])

    def test_empty_task_is_rejected(self):
        out = await_tool(_tools().pi_run("   "))
        self.assertEqual(out["status"], "invalid")

    def test_bad_session_id_is_rejected(self):
        out = await_tool(_tools().pi_run("write a script", session_id="bad id!"))
        self.assertEqual(out["status"], "invalid")


class StatusMappingTests(unittest.TestCase):
    """Every non-2xx used to become 'Could not reach pi-agent-runner.'"""

    def setUp(self):
        self.module = _load_module()

    def _run(self, status, payload=None):
        _stub_post(self.module, _Resp(status, payload))
        return await_tool(self.module.Tools().pi_run("write a script"))

    def test_429_is_capacity_not_an_outage(self):
        out = self._run(429, {"detail": "Session limit reached (5); close an idle session and retry."})
        self.assertEqual(out["status"], "capacity")
        self.assertIn("Session limit reached", out["detail"])
        self.assertNotIn("Could not reach", out["detail"])

    def test_429_falls_back_when_body_is_not_json(self):
        out = self._run(429, None)
        self.assertEqual(out["status"], "capacity")
        self.assertIn("capacity", out["detail"].lower())

    def test_409_is_busy(self):
        out = self._run(409, {"detail": "Session 'x' is already running a task."})
        self.assertEqual(out["status"], "busy")
        self.assertIn("already running", out["detail"])

    def test_504_is_timeout(self):
        out = self._run(504, {"detail": "Pi run timed out after 5s."})
        self.assertEqual(out["status"], "timeout")

    def test_500_is_error_with_runner_detail(self):
        out = self._run(500, {"detail": "Session create failed: boom"})
        self.assertEqual(out["status"], "error")
        self.assertIn("boom", out["detail"])

    def test_422_string_detail_is_passed_through(self):
        out = self._run(422, {"detail": "`provider` requires `model`."})
        self.assertEqual(out["status"], "invalid")
        self.assertIn("requires", out["detail"])

    def test_422_validation_list_is_readable_prose(self):
        """FastAPI returns a list of objects; a raw repr is unusable."""
        out = self._run(422, {"detail": [
            {"loc": ["body", "task"], "msg": "String should have at most 131072 characters"},
        ]})
        self.assertEqual(out["status"], "invalid")
        self.assertIn("task:", out["detail"])
        self.assertIn("131072", out["detail"])
        self.assertNotIn("[{", out["detail"])

    def test_500_without_detail_mentions_status(self):
        out = self._run(502, None)
        self.assertEqual(out["status"], "error")
        self.assertIn("502", out["detail"])

    def test_connection_error_names_the_host(self):
        def boom(*a, **k):
            raise self.module.requests.exceptions.ConnectionError("refused")
        self.module.requests.post = boom
        out = await_tool(self.module.Tools().pi_run("write a script"))
        self.assertEqual(out["status"], "error")
        self.assertIn("pi-agent-runner", out["detail"])


class SessionCloseTests(unittest.TestCase):
    def setUp(self):
        self.module = _load_module()

    def _close(self, status, payload=None):
        _stub_post(self.module, _Resp(status, payload))
        return await_tool(self.module.Tools().pi_session_close("reviewcheck"))

    def test_404_is_not_found_not_a_transport_failure(self):
        """The idle reaper makes this the most likely outcome."""
        out = self._close(404, {"detail": "Session not found."})
        self.assertEqual(out["status"], "not_found")
        self.assertNotIn("Could not reach", out["detail"])

    def test_409_is_busy(self):
        out = self._close(409, {"detail": "Session is running a task."})
        self.assertEqual(out["status"], "busy")

    def test_200_is_passed_through(self):
        out = self._close(200, {"status": "closed", "session_id": "reviewcheck"})
        self.assertEqual(out["status"], "closed")


class HealthTests(unittest.TestCase):
    def setUp(self):
        self.module = _load_module()

    def test_reports_version(self):
        _stub_get(self.module, _Resp(200, {"status": "ok", "docker": "reachable",
                                           "version": "1.1.7"}))
        out = await_tool(self.module.Tools().pi_health())
        self.assertEqual(out["version"], "1.1.7")

    def test_503_is_error_with_detail(self):
        _stub_get(self.module, _Resp(503, {"status": "error",
                                           "detail": "socket unreachable",
                                           "version": "1.1.7"}))
        out = await_tool(self.module.Tools().pi_health())
        self.assertEqual(out["status"], "error")
        self.assertIn("unreachable", out["detail"])
        self.assertEqual(out["version"], "1.1.7")


class HelperTests(unittest.TestCase):
    def setUp(self):
        self.Tools = _load_module().Tools

    def test_detail_or_prefers_runner_text(self):
        self.assertEqual(
            self.Tools._detail_or(_Resp(429, {"detail": "runner says"}), "fb"),
            "runner says",
        )

    def test_detail_or_uses_fallback_for_non_json(self):
        self.assertEqual(self.Tools._detail_or(_Resp(429, None), "fb"), "fb")

    def test_detail_or_ignores_blank_and_non_string(self):
        self.assertEqual(self.Tools._detail_or(_Resp(429, {"detail": "  "}), "fb"), "fb")
        self.assertEqual(self.Tools._detail_or(_Resp(429, {"detail": ["x"]}), "fb"), "fb")


if __name__ == "__main__":
    unittest.main()
