"""Guards that a tool never blocks the Open WebUI event loop.

WHY THIS FILE EXISTS
--------------------
Every OWUI tool is invoked from an async route. OWUI wraps each tool
function with get_async_tool_function_and_apply_extra_params, which branches
on inspect.iscoroutinefunction and, for a plain def, calls the function
INLINE on the event loop rather than offloading it to a thread pool:

    else:
        async def new_function(*args, **kwargs):
            return partial_func(*args, **coerce_kwargs(kwargs))

A def tool that calls requests.post therefore blocks that loop for the whole
HTTP call. uvicorn runs a single worker here, so that loop also serves every
other request and every chat token stream: one long call freezes the UI for
every user until it returns.

This is not hypothetical. pi_run waited up to timeout + 10 seconds (600 s by
default) on a plain def, and a real run froze Open WebUI for minutes. The
suite stayed green throughout, because every existing test stubbed requests
and asserted on the returned value -- nothing measured the loop.

WHAT IS PINNED HERE
-------------------
Every static check reads the source with ast, so it needs no third-party
import (the Nextcloud tool imports caldav, which CI deliberately does not
install).

1. A public tool method must be async def -- otherwise OWUI calls it inline.
2. No blocking call may sit directly in an async def body; it must be the
   argument of asyncio.to_thread, which is what moves it off the loop.
3. An async def must not call a private sync helper inline: that moves the
   block one frame deeper while still running on the loop.
4. Dynamic, on the tools CI can import: a 20 ms heartbeat running alongside
   the call must keep ticking while the tool HTTP call sleeps. This is the
   only assertion that measures the loop rather than the shape of the code.

Run with:
    python -m pytest test_tools_async_guard.py -q
"""

import ast
import asyncio
import importlib.util
import inspect
import pathlib
import time

import pytest

HERE = pathlib.Path(__file__).resolve().parent

# Every tool that becomes a callable in Open WebUI. Listed explicitly rather
# than globbed, so a tool dropping out of this guard is a visible diff.
TOOL_FILES = [
    "docker-exe-openwebui-tool.py",
    "dockhand-openwebui-tool.py",
    "nextcloud-calendar-calling.py",
    "pi-agent-openwebui-tool.py",
    "portainer-openwebui-tool.py",
    "rag-openwebui-tool.py",
    "research-openwebui-tool.py",
]

# Calls that occupy the calling thread for a whole network round trip.
BLOCKING_MODULES = {"requests", "urllib", "http", "socket", "subprocess"}


def _parsed(name):
    path = HERE / name
    assert path.exists(), name + " is listed in this guard but does not exist"
    return ast.parse(path.read_text())


def _classes(tree):
    return [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]


def _methods(cls):
    return [n for n in cls.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _calls_in_body(fn):
    """Calls in this function own body, not inside a nested callable."""
    found = []
    stack = list(fn.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue          # a nested def is a separate callable
        found.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return [n for n in found if isinstance(n, ast.Call)]


def _offloaded_calls(fn):
    """ids of Call nodes passed as an argument of asyncio.to_thread(...)."""
    ids = set()
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("to_thread", "run_in_executor")):
            for arg in node.args:
                for sub in ast.walk(arg):
                    if isinstance(sub, ast.Call):
                        ids.add(id(sub))
    return ids


def _self_calls(fn):
    """Names called on `self` in this function own body."""
    out = set()
    for call in _calls_in_body(fn):
        f = call.func
        if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                and f.value.id == "self"):
            out.add(f.attr)
    return out


def _module_calls(fn):
    """Blocking-module calls made directly in this function own body."""
    return [c for c in _calls_in_body(fn)
            if isinstance(c.func, ast.Attribute)
            and isinstance(c.func.value, ast.Name)
            and c.func.value.id in BLOCKING_MODULES]


def _blocking_helpers(cls):
    """Private sync helpers that TRANSITIVELY reach a blocking call.

    Pure helpers (_is_error, _headers, _url) touch no network and are
    deliberately excluded: running them on the loop costs microseconds.
    """
    methods = {f.name: f for f in _methods(cls)}
    blocking = set()
    for name, fn in methods.items():
        if _module_calls(fn):
            blocking.add(name)
    changed = True
    while changed:
        changed = False
        for name, fn in methods.items():
            if name in blocking:
                continue
            if _self_calls(fn) & blocking:
                blocking.add(name)
                changed = True
    return {n for n in blocking
            if n.startswith("_") and not isinstance(methods[n], ast.AsyncFunctionDef)}




def _describe(name, cls, fn, call, kind):
    return (name + ":" + str(call.lineno) + " " + cls.name + "." + fn.name
            + " -> " + kind)


def _tool_class(tree):
    """The class OWUI turns into tools. Everything else is private plumbing."""
    for cls in _classes(tree):
        if cls.name == "Tools":
            return cls
    return None


def _loop_blockers(name):
    out = []
    cls = _tool_class(_parsed(name))
    if cls is None:
        return out
    risky = _blocking_helpers(cls)
    for fn in _methods(cls):
        if not isinstance(fn, ast.AsyncFunctionDef):
            continue
        offloaded = _offloaded_calls(fn)
        for call in _module_calls(fn):
            if id(call) in offloaded:
                continue
            out.append(_describe(name, cls, fn, call,
                                call.func.value.id + "." + call.func.attr))
        for call in _calls_in_body(fn):
            f = call.func
            if not isinstance(f, ast.Attribute):
                continue
            if not isinstance(f.value, ast.Name) or f.value.id != "self":
                continue
            if f.attr in risky:
                out.append(_describe(name, cls, fn, call,
                                    "sync helper self." + f.attr))
    return out


class TestPublicMethodsAreCoroutines:
    """A plain def tool is called INLINE on the event loop by OWUI.

    OWUI enumerates the members of the module Tools class; private plumbing
    (_LoopGuard, _Pipeline, _BodyExtractor, Valves validators) is never
    invoked as a tool, so only Tools is in scope here.
    """

    def test_every_public_tool_method_is_async(self):
        offenders = []
        for name in TOOL_FILES:
            cls = _tool_class(_parsed(name))
            assert cls is not None, name + " declares no Tools class"
            for fn in _methods(cls):
                if fn.name.startswith("_"):
                    continue
                if not isinstance(fn, ast.AsyncFunctionDef):
                    offenders.append(name + "::Tools." + fn.name)
        assert not offenders, (
            "these public tool methods are plain def, so Open WebUI calls them "
            "inline on the event loop and any HTTP call freezes the whole UI: "
            + ", ".join(offenders)
        )

    def test_a_known_good_shape_reports_nothing(self):
        tree = ast.parse("class Tools:" + chr(10)
                         + "    async def ok(self):" + chr(10)
                         + "        return 1" + chr(10))
        cls = _tool_class(tree)
        found = [f.name for f in _methods(cls)
                 if not isinstance(f, ast.AsyncFunctionDef)]
        assert found == []

    def test_the_detector_catches_a_sync_tool(self):
        """Prove it fires: a guard that cannot fail is worse than none."""
        tree = ast.parse("class Tools:" + chr(10)
                         + "    def bad(self):" + chr(10)
                         + "        return 1" + chr(10))
        cls = _tool_class(tree)
        found = [f.name for f in _methods(cls)
                 if not isinstance(f, ast.AsyncFunctionDef)]
        assert found == ["bad"]
def _load(name):
    path = HERE / name
    spec = importlib.util.spec_from_file_location(
        "guard_" + path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



class TestNoBlockingCallRunsOnTheLoop:
    """Static: nothing reachable from an async Tools method may block it."""

    def test_no_blocking_call_in_an_async_body(self):
        offenders = []
        for name in TOOL_FILES:
            offenders.extend(_loop_blockers(name))
        assert not offenders, (
            "these run on the event loop; hand them to asyncio.to_thread: "
            + "; ".join(offenders)
        )

    @staticmethod
    def _scan(src):
        """Run the detector over a synthetic module instead of a real file."""
        original = globals()["_parsed"]
        globals()["_parsed"] = lambda name: ast.parse(src)
        try:
            return _loop_blockers("synthetic-module")
        finally:
            globals()["_parsed"] = original

    def test_detector_flags_a_bare_blocking_call(self):
        offenders = self._scan(
            "import requests" + chr(10)
            + "class Tools:" + chr(10)
            + "    async def bad(self):" + chr(10)
            + "        return requests.post(URL)" + chr(10))
        assert len(offenders) == 1, offenders
        assert "Tools.bad" in offenders[0]
        assert "requests.post" in offenders[0]

    def test_detector_accepts_both_offloaded_shapes(self):
        offenders = self._scan(
            "import asyncio" + chr(10)
            + "import requests" + chr(10)
            + "class Tools:" + chr(10)
            + "    async def good(self):" + chr(10)
            + "        return await asyncio.to_thread(requests.post, URL)" + chr(10)
            + "    def _sync(self):" + chr(10)
            + "        return requests.get(URL)" + chr(10)
            + "    async def also_good(self):" + chr(10)
            + "        return await asyncio.to_thread(self._sync)" + chr(10))
        assert offenders == [], offenders

    def test_detector_flags_a_transitively_blocking_helper_called_inline(self):
        """Moving the block one frame deeper does not move it off the loop."""
        offenders = self._scan(
            "import requests" + chr(10)
            + "class Tools:" + chr(10)
            + "    def _fetch(self):" + chr(10)
            + "        return requests.get(URL)" + chr(10)
            + "    async def bad(self):" + chr(10)
            + "        return self._fetch()" + chr(10))
        assert len(offenders) == 1, offenders
        assert "sync helper self._fetch" in offenders[0]

    def test_a_pure_helper_on_the_loop_is_allowed(self):
        """_is_error and friends cost microseconds and must not be flagged."""
        offenders = self._scan(
            "class Tools:" + chr(10)
            + "    def _is_error(self, r):" + chr(10)
            + "        return False" + chr(10)
            + "    async def fine(self):" + chr(10)
            + "        return self._is_error(1)" + chr(10))
        assert offenders == [], offenders

class TestLoopStaysResponsive:
    SLEEP = 0.30
    TICK = 0.02

    @staticmethod
    def _measure(coro):
        async def main():
            gaps = []
            stop = {"v": False}

            async def heartbeat():
                last = time.perf_counter()
                while not stop["v"]:
                    await asyncio.sleep(TestLoopStaysResponsive.TICK)
                    now = time.perf_counter()
                    gaps.append(now - last)
                    last = now

            hb = asyncio.create_task(heartbeat())
            await asyncio.sleep(2 * TestLoopStaysResponsive.TICK)
            try:
                result = await coro
            finally:
                stop["v"] = True
                await hb
            return result, (max(gaps) if gaps else 0.0)

        return asyncio.run(main())

    def test_pi_run_does_not_block_the_loop(self, monkeypatch):
        """pi_run is the worst case: timeout + 10 seconds, 600 s by default."""
        module = _load("pi-agent-openwebui-tool.py")

        def slow_post(*a, **k):
            time.sleep(TestLoopStaysResponsive.SLEEP)
            raise module.requests.exceptions.ConnectionError("simulated")

        monkeypatch.setattr(module.requests, "post", slow_post)
        result, stall = self._measure(module.Tools().pi_run("write a script"))

        assert result["status"] == "error"      # the stubbed failure is mapped
        assert stall < self.SLEEP / 3, (
            "pi_run blocked the event loop for " + str(round(stall, 2))
            + "s; Open WebUI would be frozen for the whole run"
        )

    def test_every_importable_tool_leaves_the_loop_responsive(self, monkeypatch):
        cases = []
        cases.append((_load("rag-openwebui-tool.py"), "post",
                      lambda t: t.rag_search("q")))
        cases.append((_load("searx-search.py"), "get",
                      lambda t: t.searxng_search("q")))
        cases.append((_load("portainer-openwebui-tool.py"), "request",
                      lambda t: t.list_endpoints()))
        cases.append((_load("dockhand-openwebui-tool.py"), "get",
                      lambda t: t.list_environments()))

        failures = []
        for module, attr, factory in cases:
            original = getattr(module.requests, attr)

            def slow(*a, _mod=module, **k):
                time.sleep(TestLoopStaysResponsive.SLEEP)
                raise _mod.requests.exceptions.ConnectionError("simulated")

            monkeypatch.setattr(module.requests, attr, slow)
            try:
                _, stall = self._measure(factory(module.Tools()))
            finally:
                monkeypatch.setattr(module.requests, attr, original)
            if stall >= self.SLEEP / 3:
                failures.append(module.__name__ + "." + attr + ": " + str(round(stall, 2)) + "s")

        assert not failures, (
            "these tools blocked the event loop (a frozen Open WebUI): "
            + ", ".join(failures)
        )

    def test_the_measurement_can_detect_blocking(self):
        """Control case: a deliberately blocking coroutine MUST be caught.

        Without this, a broken heartbeat would make every case above pass.
        """

        async def blocks():
            time.sleep(self.SLEEP)
            return "done"

        result, stall = self._measure(blocks())
        assert result == "done"
        assert stall >= self.SLEEP / 2, (
            "the heartbeat missed a " + str(self.SLEEP) + "s block (measured "
            + str(round(stall, 2)) + "s), so it cannot prove anything"
        )
