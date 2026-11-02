#!/usr/bin/env python3
"""Manual live check for the web-search example: real HTTP, real keys.

Deliberately outside the test suite. ``tests/test_web_search_extension.py`` is
deterministic and key-free; this script spends real API quota and fails when a
key expires, which is exactly what makes it useful as a diagnostic rather than a
regression test::

    uv run scripts/live_web_search.py
    uv run scripts/live_web_search.py exa perplexity

Each provider section prints the tool's own output, so a rejected key reads as
``serper failed: google.serper.dev: HTTP 403`` instead of a stack trace, and the
``/websearch status`` block at the end names the key variables the extension
looked for. Nothing here prints a key value.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

EXTENSION_PATH = Path(__file__).resolve().parents[1] / "examples" / "extensions" / "web_search.py"


def _load_extension() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tau_web_search_live", EXTENSION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ProbeApi:
    """Stands in for the extension API: enough to reach the tools and command."""

    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}
        self.commands: dict[str, Any] = {}
        self.hooks: dict[str, list[Any]] = {}

    def register_tool(self, tool: Any) -> None:
        self.tools[tool.name] = tool

    def register_command(self, name: str, handler: Any, **kwargs: Any) -> None:
        del kwargs
        self.commands[name] = handler

    def on(self, event: str, handler: Any = None) -> Any:
        def decorate(fn: Any) -> Any:
            self.hooks.setdefault(event, []).append(fn)
            return fn

        return decorate if handler is None else decorate(handler)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("providers", nargs="*", help="provider names to probe (default: all)")
    parser.add_argument("--query", default="uv python package manager astral docs")
    parser.add_argument("--fetch", default="https://docs.astral.sh/uv/")
    args = parser.parse_args()

    ws = _load_extension()
    api = ProbeApi()
    ws.setup(api)
    settings = ws.settings_from_env(os.environ)
    print(f"detected key envs: {sorted(settings.keys)}")
    print(f"usable order:     {' -> '.join(settings.available_providers()) or 'none'}\n")

    wanted = args.providers or list(ws.PROVIDERS)
    failures: list[str] = []
    for provider in wanted:
        print("=" * 26, provider, "=" * 26)
        result = await api.tools["web_search"].execute_fn(
            "live", {"query": args.query, "provider": provider, "count": 3}
        )
        print(result.text[:1200])
        if result.details["status"] != "ok":
            failures.append(provider)
        print()

    print("=" * 26, "auto", "=" * 26)
    auto = await api.tools["web_search"].execute_fn("live", {"query": args.query, "count": 5})
    print(auto.text[:1400])
    if auto.details["status"] != "ok":
        failures.append("auto")

    print("\n" + "=" * 26, "web_fetch", "=" * 26)
    fetched = await api.tools["web_fetch"].execute_fn("live", {"url": args.fetch, "maxChars": 1200})
    print(fetched.text[:1600])
    if fetched.details["status"] == "error":
        failures.append("web_fetch")

    print("\n" + "=" * 26, "/websearch status", "=" * 26)
    print(api.commands["websearch"]("status", None))
    print()

    for handler in api.hooks.get("session_shutdown", []):
        await handler(None, None)  # close the pooled client before exiting
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("all probes returned results")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
