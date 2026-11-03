#!/usr/bin/env python3
"""Guard the *defaults* of Open WebUI tool valves.

A valve default ships in a public repository, so it must be safe for a
stranger: no credential, no private address, no internal hostname. The unittest
suites cannot catch this -- they exercise behaviour, not defaults -- so it is
checked here, structurally.

Parsed with ``ast`` rather than grep on purpose. Both tools declare credentials
as a multi-line call::

    api_key: str = Field(
        default="",
        description=...,
    )

A line-oriented regex that expects ``Field(`` and ``default=`` on one line
misses every one of those and reports "clean" while a live key sits in the
default. That false negative is why this file exists in this form.

No secret value is ever echoed -- only the file, line and valve name.
"""

from __future__ import annotations

import ast
import ipaddress
import re
import sys
from pathlib import Path

# A valve whose default must be empty: it carries a credential.
CREDENTIALISH = re.compile(r"(?i)(api_key|api_token|token|secret|password|passwd|auth)")
# Hostnames that look like an internal network rather than a public service.
INTERNAL_HOST = re.compile(r"(?i)\.(nuc|local|lan|internal|corp|home|test)$")


def field_defaults(node: ast.Call):
    """Yield (lineno, value) for each literal ``default=`` of a ``Field(...)``."""
    for kw in node.keywords:
        if kw.arg != "default":
            continue
        try:
            value = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            continue
        yield kw.value.lineno, value


def host_of(value: str) -> str | None:
    """Extract a hostname from a URL-ish valve default, if it has one."""
    if "://" not in value:
        return None
    authority = value.split("://", 1)[1].split("/", 1)[0].rsplit("@", 1)[-1]
    return authority.split(":", 1)[0] or None


def check_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    problems: list[str] = []

    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        if cls.name != "Valves":
            continue
        for stmt in cls.body:
            if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
                continue
            if not isinstance(stmt.value, ast.Call):
                continue
            name = stmt.target.id
            for lineno, value in field_defaults(stmt.value):
                where = f"{path.name}:{lineno} valve {name!r}"

                if not isinstance(value, str) or not value.strip():
                    continue  # an empty default is always fine

                if CREDENTIALISH.search(name):
                    problems.append(f"{where} has a non-empty default; a credential must not be committed")
                    continue

                host = host_of(value)
                if not host:
                    continue
                try:
                    ip = ipaddress.ip_address(host)
                except ValueError:
                    if INTERNAL_HOST.search(host):
                        problems.append(f"{where} defaults to an internal-looking hostname ({host})")
                    continue
                if not ip.is_global:
                    problems.append(f"{where} defaults to a non-public address ({host})")

    return problems


def main(argv: list[str]) -> int:
    targets = [Path(a) for a in argv[1:]] or sorted(Path.cwd().glob("*.py"))
    if not targets:
        print("::error::no Python files to check", file=sys.stderr)
        return 1

    problems: list[str] = []
    for path in targets:
        try:
            problems.extend(check_file(path))
        except SyntaxError as exc:
            problems.append(f"{path.name}: not parseable ({exc})")

    for problem in problems:
        print(f"::error::{problem}")
    if problems:
        print(f"\n{len(problems)} unsafe valve default(s) found", file=sys.stderr)
        return 1

    print(f"valve defaults clean: {len(targets)} file(s) checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
