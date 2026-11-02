"""Behavioral tests for the `web_search` / `web_fetch` example extension.

The load-bearing behavior is graceful degradation: one backend's rejected key,
timeout, or thin answer must fall through to the next engine instead of failing
the turn, and a credential that was already rejected must not cost another
round-trip on every later search. Those are driven through the extension's real
tool handlers with a stubbed transport, so no test needs a key or the network.
Payload parsing, reader-mode extraction, and the settings ladder are checked
directly against the module; registration is covered by the shipped-examples
loader in ``test_example_extensions.py``.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.anyio

EXTENSION_PATH = Path(__file__).resolve().parents[1] / "examples" / "extensions" / "web_search.py"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _load_extension() -> ModuleType:
    """Import the shipped example module without registering it as an extension."""
    spec = importlib.util.spec_from_file_location("tau_web_search_example", EXTENSION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ws = _load_extension()

# -- recorded provider payloads ------------------------------------------------


SERPER_OK: dict[str, Any] = {
    "organic": [
        {
            "title": "httpx 0.28 release notes",
            "link": "https://www.python-httpx.org/releases/",
            "snippet": "0.28 changes the   proxy API.\nSupport for SOCKS stays in httpx[socks].",
            "date": "3 days ago",
            "source": "python-httpx.org",
        },
        {"title": "no snippet here", "link": "https://example.com/b"},
    ],
    "news": [
        {"title": "Serper status", "link": "https://status.serper.dev", "snippet": "all green"}
    ],
}

EXA_OK: dict[str, Any] = {
    "results": [
        {
            "title": "Exa docs",
            "url": "https://docs.exa.ai/reference/search",
            "highlights": ["semantic search", "returns page text"],
            "publishedDate": "2026-04-01T00:00:00.000Z",
        },
        {"title": "", "url": "https://exa.ai"},
        {"title": "dropped, no url"},
    ]
}

EXA_TITLES_ONLY: dict[str, Any] = {
    "results": [{"title": "Title without a snippet", "url": "https://a.test/x"}]
}

FIRECRAWL_V2: dict[str, Any] = {
    "success": True,
    "data": {"web": [{"title": "FC hit", "url": "https://a.test/1", "description": "first"}]},
}
FIRECRAWL_V1: dict[str, Any] = {
    "success": True,
    "data": [{"title": "FC v1", "url": "https://a.test/2", "snippet": "flat list"}],
}

PERPLEXITY_FLAT: dict[str, Any] = {
    "choices": [{"message": {"content": "Serper is a SERP API. [1]"}}],
    "citations": ["https://serper.dev", "not-a-url"],
}
PERPLEXITY_NESTED: dict[str, Any] = {
    "choices": [
        {
            "message": {
                "content": "Exa returns text.",
                "meta": {"citations": [{"url": "https://exa.ai/x"}, {"nope": 1}]},
            }
        }
    ]
}
PERPLEXITY_NO_ANSWER: dict[str, Any] = {
    "choices": [{"message": {"content": ""}}],
    "citations": ["https://serper.dev"],
}

DDG_HTML = """
<div class="result">
  <a rel="nofollow" class="result__a"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.astral.sh%2Fuv%2F&rut=abc">uv docs</a>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.astral.sh%2Fuv%2F">
     an <b>extremely</b> fast Python package manager</a>
</div>
<div class="result">
  <a class="result__a" href="https://direct.example.org/page">direct link</a>
  <div class="result__snippet">plain div snippet</div>
</div>
"""

PAGE_HTML = """
<html><head><title>Reader mode test</title><style>p{color:red}</style>
<script>var tracking = "leak";</script></head>
<body><header>site banner</header><nav>menu menu menu</nav>
<h2>Heading</h2><p>First   paragraph.</p>
<ul><li>alpha</li><li>beta</li></ul>
<table><tr><td>one</td><td>two</td></tr></table>
<div>tail <!-- keep --> text</div>
<aside>related links related links</aside><footer>copyright footer</footer></body></html>
"""

NEWS_HTML = """
<html><head><title>Headlines for today</title></head><body>
<div class="menu-main"><a href="/">Home</a><a href="/world">World</a></div>
<div id="site-nav">skip to content</div>
<header class="masthead">site banner</header>
<main><article class="story">
  <header><div class="article-header"><h1>Real headline goes here</h1></div></header>
  <div class="story-body"><p>Body copy that actually matters.</p></div>
  <div class="share-bar">share share</div>
  <div class="related-articles">related related</div>
</article></main>
<div class="newsletter-signup">subscribe</div>
<footer class="footer-container">copyright footer</footer>
</body></html>
"""

# -- environment ---------------------------------------------------------------

#: Every environment name the extension reads. Tests start from a clean slate so
#: a key configured on the developer's machine cannot change which backend runs.
_MANAGED_ENV: tuple[str, ...] = (
    "SERPER_API_KEY",
    "SERPER_DEV_API_KEY",
    "EXA_API_KEY",
    "PERPLEXITY_API_KEY",
    "PPLX_API_KEY",
    "FIRECRAWL_API_KEY",
    "TAU_WEB_SEARCH_PROVIDER",
    "TAU_WEB_SEARCH_MAX_RESULTS",
    "TAU_WEB_SEARCH_SNIPPET_CHARS",
    "TAU_WEB_SEARCH_TIMEOUT_SECONDS",
    "TAU_WEB_SEARCH_ALLOW_KEYLESS",
    "TAU_WEB_SEARCH_SNIPPET_FLOOR",
    "TAU_WEB_SEARCH_PERPLEXITY_MODEL",
    "TAU_WEB_SEARCH_CACHE_SECONDS",
    "TAU_WEB_SEARCH_DEAD_TTL_SECONDS",
    "TAU_WEB_FETCH_MAX_CHARS",
)

_ENV_NAME: re.Pattern[str] = re.compile(
    r"\b(?:TAU_WEB_[A-Z0-9_]+|[A-Z][A-Z0-9_]*_(?:API_KEY|KEY))\b"
)


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install an exact provider/key environment for one test."""

    def apply(values: Mapping[str, str]) -> None:
        for name in _MANAGED_ENV:
            monkeypatch.delenv(name, raising=False)
        for name, value in values.items():
            monkeypatch.setenv(name, value)

    return apply


def test_the_harness_tracks_every_environment_name_the_extension_reads() -> None:
    """A new knob or key variable must join the managed list, not leak through."""
    names = set(_ENV_NAME.findall(Path(EXTENSION_PATH).read_text(encoding="utf-8")))

    assert names, "the extension stopped naming its environment variables"
    assert names <= set(_MANAGED_ENV), (
        f"unmanaged environment names: {sorted(names - set(_MANAGED_ENV))}"
    )


# -- transport and registration doubles ---------------------------------------


@dataclass
class FakeResponse:
    """Stand-in for the ``httpx.Response`` the transport hands to ``fetch_page``."""

    status_code: int = 200
    payload: Any = None
    text: str = ""
    headers: Mapping[str, str] = field(default_factory=lambda: {"content-type": "text/html"})
    url: str = "https://example.test/page"

    def json(self) -> Any:
        if self.payload is None:
            raise ValueError("no json")
        return self.payload


class FakeHttp:
    """Records calls and replays scripted responses or errors, keyed by URL."""

    def __init__(self, responses: Mapping[str, FakeResponse | Exception] | None = None) -> None:
        self.responses = dict(responses or {})
        self.calls: list[tuple[str, str, Any]] = []
        self.timeout = 15.0
        self.closed = False

    def _next(self, url: str) -> FakeResponse:
        for marker, outcome in self.responses.items():
            if marker in url:
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome
        raise ws.ProviderError(f"unscripted call to {url}")

    async def post_json(
        self, url: str, *, headers: Mapping[str, str], body: Mapping[str, Any]
    ) -> Any:
        self.calls.append(("POST", url, body))
        return self._next(url).json()

    async def get_text(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, str] | None = None,
    ) -> FakeResponse:
        self.calls.append(("GET", url, params or {}))
        return self._next(url)

    async def close(self) -> None:
        self.closed = True


class RecordingApi:
    """Captures what ``setup()`` registers so tools and commands run directly."""

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


@pytest.fixture
def install(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install the extension against a stubbed transport.

    Registration goes through the extension's own ``setup()`` because the stub
    has to replace the transport before the pooled client is built. The shipped
    loader path — discovery, import, tool and command registration — is covered
    in ``test_example_extensions.py``.
    """

    def apply(fake_http: FakeHttp) -> tuple[RecordingApi, dict[str, Any]]:
        monkeypatch.setattr(ws, "Http", lambda timeout: fake_http)
        api = RecordingApi()
        ws.setup(api)
        return api, api.tools

    return apply


# -- configuration -------------------------------------------------------------


def test_defaults_and_key_discovery() -> None:
    settings = ws.settings_from_env({"SERPER_DEV_API_KEY": "serper-dev", "EXA_API_KEY": "exa"})

    assert settings.provider == "auto"
    assert settings.max_results == ws.DEFAULT_MAX_RESULTS
    assert settings.keys == {"serper": "serper-dev", "exa": "exa"}
    assert settings.available_providers() == ("serper", "exa", "duckduckgo")
    assert settings.missing_providers() == ("perplexity", "firecrawl")


def test_serper_prefers_the_plain_key_name() -> None:
    settings = ws.settings_from_env({"SERPER_API_KEY": "plain", "SERPER_DEV_API_KEY": "dev"})

    assert settings.keys["serper"] == "plain"


def test_env_overrides_are_clamped() -> None:
    settings = ws.settings_from_env(
        {
            "TAU_WEB_SEARCH_PROVIDER": "EXA",
            "TAU_WEB_SEARCH_MAX_RESULTS": "999",
            "TAU_WEB_SEARCH_TIMEOUT_SECONDS": "0",
            "TAU_WEB_SEARCH_ALLOW_KEYLESS": "0",
            "TAU_WEB_SEARCH_PERPLEXITY_MODEL": "  ",
        }
    )

    assert settings.provider == "exa"
    assert settings.max_results == ws.MAX_RESULTS_LIMIT
    assert settings.timeout_seconds >= 2
    assert settings.available_providers() == ()
    assert settings.perplexity_model == "sonar"


def test_an_unknown_provider_pin_falls_back_to_auto() -> None:
    assert ws.settings_from_env({"TAU_WEB_SEARCH_PROVIDER": "bogus"}).provider == "auto"
    # "google" is not a backend name any more; it is Serper's alias.
    assert ws.settings_from_env({"TAU_WEB_SEARCH_PROVIDER": "google"}).provider == "serper"


async def test_a_pinned_backend_without_a_key_says_which_variable_to_set(
    install: Any, env: Any
) -> None:
    _, tools = install(FakeHttp())
    env({"TAU_WEB_SEARCH_PROVIDER": "serper"})

    result = await tools["web_search"].execute_fn("call-1", {"query": "anything"})

    assert result.details["status"] == "empty"
    assert "SERPER_API_KEY" in result.text


# -- payload parsing -----------------------------------------------------------


def test_serper_maps_organic_and_news_and_dedupes() -> None:
    results = ws.parse_serper(SERPER_OK, 320)

    assert [result.url for result in results] == [
        "https://www.python-httpx.org/releases/",
        "https://example.com/b",
        "https://status.serper.dev",
    ]
    first = results[0]
    assert first.site == "python-httpx.org"
    assert first.published == "3 days ago"
    assert " ".join(first.snippet.split()) == (
        "0.28 changes the proxy API. Support for SOCKS stays in httpx[socks]."
    )
    assert ws.parse_serper([], 320) == []
    assert ws.parse_serper({"unexpected": True}, 320) == []


def test_exa_joins_highlights_and_skips_a_result_without_a_url() -> None:
    results = ws.parse_exa(EXA_OK, 320)

    assert len(results) == 2
    assert results[0].snippet == "semantic search returns page text"
    assert results[0].published == "2026-04-01"
    assert results[1].title == "exa.ai"


def test_firecrawl_answers_both_api_shapes() -> None:
    assert ws.parse_firecrawl(FIRECRAWL_V2, 320)[0].url == "https://a.test/1"

    v1 = ws.parse_firecrawl(FIRECRAWL_V1, 320)
    assert v1[0].url == "https://a.test/2"
    assert v1[0].snippet == "flat list"


def test_perplexity_reads_both_citation_shapes() -> None:
    answer, urls = ws.parse_perplexity(PERPLEXITY_FLAT)

    assert answer.startswith("Serper is a SERP API")
    assert urls == ["https://serper.dev"]
    assert ws.parse_perplexity(PERPLEXITY_NESTED)[1] == ["https://exa.ai/x"]
    assert ws.parse_perplexity("garbage") == ("", [])


def test_duckduckgo_unwraps_its_redirect_links() -> None:
    assert (
        ws.unwrap_ddg("//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.test%2Fx&rut=deadbeef")
        == "https://a.test/x"
    )
    assert ws.unwrap_ddg("https://plain.test/y") == "https://plain.test/y"
    assert ws.unwrap_ddg("") == ""


def test_duckduckgo_parses_title_url_and_snippet() -> None:
    results = ws.parse_duckduckgo(DDG_HTML, 320)

    assert [result.url for result in results] == [
        "https://docs.astral.sh/uv/",
        "https://direct.example.org/page",
    ]
    assert results[0].title == "uv docs"
    assert results[0].snippet == "an extremely fast Python package manager"
    assert results[1].snippet == "plain div snippet"
    assert results[1].site == "direct.example.org"


def test_site_of_strips_www() -> None:
    assert ws.site_of("https://www.example.com/a") == "example.com"


# -- reader mode ---------------------------------------------------------------


def test_reader_mode_drops_page_chrome_but_keeps_the_title() -> None:
    text, title = ws.html_to_text(PAGE_HTML)

    assert title == "Reader mode test"
    assert "tracking" not in text
    assert "color:red" not in text
    # Page chrome carries no answer and dominates the first fetch window.
    for chrome in ("menu menu menu", "site banner", "related links", "copyright footer"):
        assert chrome not in text
    assert "Heading" in text
    assert "alpha" in text
    assert "one | two" in text
    assert "tail" in text
    assert "\n\n\n" not in text


def test_class_named_chrome_is_dropped_but_the_headline_survives() -> None:
    """Sites mark chrome with class names, and news headlines live in <header>."""
    text, title = ws.html_to_text(NEWS_HTML)

    assert title == "Headlines for today"
    assert "Real headline goes here" in text
    assert "Body copy that actually matters." in text
    for chrome in (
        "Home",
        "skip to content",
        "site banner",
        "share share",
        "related related",
        "subscribe",
        "copyright footer",
    ):
        assert chrome not in text


# -- transcript formatting -----------------------------------------------------


def test_search_output_lists_ranked_results() -> None:
    outcome = ws.SearchOutcome(
        provider="serper",
        query="httpx socks",
        results=[
            ws.SearchResult(title="A", url="https://a.test/1", snippet="first", site="a.test")
        ],
        notes=["exa failed: HTTP 401"],
        elapsed_ms=210,
    )

    text = ws.format_search(outcome)

    assert "provider: serper" in text
    assert "1. A" in text
    assert "https://a.test/1 (a.test)" in text
    assert "exa failed: HTTP 401" in text
    assert "web_fetch(url)" in text


def test_an_empty_search_says_so() -> None:
    outcome = ws.SearchOutcome(query="nothing", notes=["serper: 0 results"])

    text = ws.format_search(outcome)

    assert "No results" in text
    assert "serper: 0 results" in text


# -- provider fallthrough ------------------------------------------------------


async def test_a_rejected_key_falls_through_to_the_next_backend(install: Any, env: Any) -> None:
    http = FakeHttp(
        {
            "google.serper.dev": ws.ProviderError("google.serper.dev: HTTP 401 invalid key"),
            "api.exa.ai": FakeResponse(payload=EXA_OK),
        }
    )
    _, tools = install(http)
    env({"SERPER_API_KEY": "bad", "EXA_API_KEY": "good", "TAU_WEB_SEARCH_CACHE_SECONDS": "0"})

    result = await tools["web_search"].execute_fn("call-1", {"query": "exa docs", "count": 2})

    assert result.details["status"] == "ok"
    assert result.details["provider"] == "exa"
    assert result.details["count"] == 2
    assert "google.serper.dev: HTTP 401" in result.text


async def test_every_backend_being_down_is_empty_not_an_exception(install: Any, env: Any) -> None:
    http = FakeHttp({"serper.dev": RuntimeError("boom"), "api.exa.ai": FakeResponse(payload={})})
    _, tools = install(http)
    env({"SERPER_API_KEY": "k", "EXA_API_KEY": "k", "TAU_WEB_SEARCH_ALLOW_KEYLESS": "0"})

    result = await tools["web_search"].execute_fn("call-1", {"query": "q"})

    assert result.details["status"] == "empty"
    assert result.details["provider"] is None
    assert "exa: 0 results" in result.text
    assert "RuntimeError" in result.text


async def test_the_cache_answers_the_second_identical_query(install: Any, env: Any) -> None:
    http = FakeHttp({"api.exa.ai": FakeResponse(payload=EXA_OK)})
    _, tools = install(http)
    env({"TAU_WEB_SEARCH_PROVIDER": "exa", "EXA_API_KEY": "k"})

    first = await tools["web_search"].execute_fn("call-1", {"query": "cached?"})
    second = await tools["web_search"].execute_fn("call-2", {"query": "cached?"})

    assert not first.details["cached"]
    assert second.details["cached"]
    assert len([call for call in http.calls if "api.exa.ai" in call[1]]) == 1


async def test_freshness_and_count_reach_the_provider(install: Any, env: Any) -> None:
    http = FakeHttp({"google.serper.dev": FakeResponse(payload=SERPER_OK)})
    _, tools = install(http)
    env({"SERPER_API_KEY": "k", "TAU_WEB_SEARCH_CACHE_SECONDS": "0"})

    await tools["web_search"].execute_fn(
        "call-1", {"query": "release", "count": 3, "freshness": "week"}
    )

    body = http.calls[0][2]
    assert body["num"] == 3
    assert body["tbs"] == "qdr:w"


async def test_a_blank_query_is_rejected_before_the_network(install: Any) -> None:
    http = FakeHttp()
    _, tools = install(http)

    result = await tools["web_search"].execute_fn("call-1", {"query": "   "})

    assert result.details["status"] == "error"
    assert http.calls == []


async def test_the_keyless_engine_is_the_last_resort_when_it_is_the_only_one(
    install: Any, env: Any
) -> None:
    http = FakeHttp({"html.duckduckgo.com": FakeResponse(payload=None, text=DDG_HTML)})
    _, tools = install(http)
    env({})

    result = await tools["web_search"].execute_fn("call-1", {"query": "uv package manager"})

    assert result.details["provider"] == "duckduckgo"
    assert result.details["count"] == 2


async def test_a_duckduckgo_challenge_page_blames_the_address_not_the_query(
    install: Any, env: Any
) -> None:
    challenge = "<html><body>anomaly: please verify you are not a robot</body></html>"
    http = FakeHttp({"html.duckduckgo.com": FakeResponse(payload=None, text=challenge)})
    _, tools = install(http)
    env({})

    result = await tools["web_search"].execute_fn("call-1", {"query": "q"})

    assert result.details["status"] == "empty"
    assert "challenge page" in result.text
    assert "rate-limited" in result.text


async def test_a_keyless_page_with_no_anchors_reports_a_layout_change(
    install: Any, env: Any
) -> None:
    blank = "<html><body><div class='other'>no anchors here</div></body></html>"
    http = FakeHttp({"html.duckduckgo.com": FakeResponse(payload=None, text=blank)})
    _, tools = install(http)
    env({})

    result = await tools["web_search"].execute_fn("call-1", {"query": "q"})

    assert "no results parsed" in result.text
    assert "layout change" in result.text


# -- provider names ------------------------------------------------------------


def test_aliases_resolve_to_canonical_names() -> None:
    for alias, canonical in (("ddgs", "duckduckgo"), ("pplx", "perplexity"), ("fc", "firecrawl")):
        assert ws.resolve_provider(alias) == canonical
    assert ws.resolve_provider("  EXA ") == "exa"
    assert ws.resolve_provider("nonsense") == "nonsense"


def test_the_environment_pin_accepts_an_alias() -> None:
    settings = ws.settings_from_env({"TAU_WEB_SEARCH_PROVIDER": "pplx", "PERPLEXITY_API_KEY": "k"})

    assert settings.provider == "perplexity"
    assert settings.available_providers() == ("perplexity",)


async def test_a_tool_call_accepts_an_alias(install: Any, env: Any) -> None:
    http = FakeHttp({"html.duckduckgo.com": FakeResponse(payload=None, text=DDG_HTML)})
    _, tools = install(http)
    env({})

    result = await tools["web_search"].execute_fn("call-1", {"query": "uv", "provider": "ddgs"})

    assert result.details["provider"] == "duckduckgo"


async def test_an_unknown_provider_lists_the_canonical_names(install: Any) -> None:
    http = FakeHttp()
    _, tools = install(http)

    result = await tools["web_search"].execute_fn("call-1", {"query": "q", "provider": "googlebot"})

    assert result.details["status"] == "error"
    assert "Choose one of: serper, exa" in result.text
    assert http.calls == []


# -- snippet quality -----------------------------------------------------------

_SNIPPET_FLOOR_ENV = {
    "EXA_API_KEY": "k",
    "FIRECRAWL_API_KEY": "k",
    "TAU_WEB_SEARCH_CACHE_SECONDS": "0",
    "TAU_WEB_SEARCH_ALLOW_KEYLESS": "0",
}


async def test_titles_only_backend_advances_to_one_with_snippets(install: Any, env: Any) -> None:
    http = FakeHttp(
        {
            "api.exa.ai": FakeResponse(payload=EXA_TITLES_ONLY),
            "api.firecrawl.dev": FakeResponse(payload=FIRECRAWL_V2),
        }
    )
    _, tools = install(http)
    env(_SNIPPET_FLOOR_ENV)

    result = await tools["web_search"].execute_fn("call-1", {"query": "q"})

    assert result.details["provider"] == "firecrawl"
    assert "exa: titles only, no snippets" in result.text


async def test_turning_the_floor_off_accepts_the_first_titles(install: Any, env: Any) -> None:
    http = FakeHttp(
        {
            "api.exa.ai": FakeResponse(payload=EXA_TITLES_ONLY),
            "api.firecrawl.dev": FakeResponse(payload=FIRECRAWL_V2),
        }
    )
    _, tools = install(http)
    env({**_SNIPPET_FLOOR_ENV, "TAU_WEB_SEARCH_SNIPPET_FLOOR": "0"})

    result = await tools["web_search"].execute_fn("call-1", {"query": "q"})

    assert result.details["provider"] == "exa"
    assert len([call for call in http.calls if "firecrawl" in call[1]]) == 0


async def test_titles_only_results_survive_when_nothing_better_exists(
    install: Any, env: Any
) -> None:
    http = FakeHttp(
        {
            "api.exa.ai": FakeResponse(payload=EXA_TITLES_ONLY),
            "api.firecrawl.dev": ws.ProviderError("api.firecrawl.dev: HTTP 500"),
        }
    )
    _, tools = install(http)
    env(_SNIPPET_FLOOR_ENV)

    result = await tools["web_search"].execute_fn("call-1", {"query": "q"})

    assert result.details["status"] == "ok"
    assert result.details["provider"] == "exa"
    assert "no backend gave snippets" in result.text


async def test_citations_without_an_answer_fall_through(install: Any, env: Any) -> None:
    http = FakeHttp(
        {
            "api.perplexity.ai": FakeResponse(payload=PERPLEXITY_NO_ANSWER),
            "api.firecrawl.dev": FakeResponse(payload=FIRECRAWL_V2),
        }
    )
    _, tools = install(http)
    env(
        {
            "PERPLEXITY_API_KEY": "k",
            "FIRECRAWL_API_KEY": "k",
            "TAU_WEB_SEARCH_CACHE_SECONDS": "0",
            "TAU_WEB_SEARCH_ALLOW_KEYLESS": "0",
        }
    )

    result = await tools["web_search"].execute_fn("call-1", {"query": "q"})

    assert result.details["provider"] == "firecrawl"
    assert "citations without an answer" in result.text


# -- quarantining a dead key ---------------------------------------------------

_QUARANTINE_ENV = {
    "SERPER_API_KEY": "bad",
    "EXA_API_KEY": "good",
    "TAU_WEB_SEARCH_CACHE_SECONDS": "0",
    "TAU_WEB_SEARCH_DEAD_TTL_SECONDS": "900",
}


async def test_an_auth_rejection_quarantines_the_backend(install: Any, env: Any) -> None:
    http = FakeHttp(
        {
            "google.serper.dev": ws.ProviderAuthError("google.serper.dev: HTTP 403 Unauthorized"),
            "api.exa.ai": FakeResponse(payload=EXA_OK),
        }
    )
    _, tools = install(http)
    env(_QUARANTINE_ENV)

    first = await tools["web_search"].execute_fn("call-1", {"query": "q1"})
    second = await tools["web_search"].execute_fn("call-2", {"query": "q2"})

    assert len([call for call in http.calls if "serper.dev" in call[1]]) == 1, (
        "the dead backend was probed twice"
    )
    assert "rejected the key" in first.text
    assert "skipped, credentials rejected" in second.text
    assert second.details["provider"] == "exa"


async def test_a_zero_quarantine_window_keeps_retrying(install: Any, env: Any) -> None:
    http = FakeHttp(
        {
            "google.serper.dev": ws.ProviderAuthError("google.serper.dev: HTTP 401"),
            "api.exa.ai": FakeResponse(payload=EXA_OK),
        }
    )
    _, tools = install(http)
    env({**_QUARANTINE_ENV, "TAU_WEB_SEARCH_DEAD_TTL_SECONDS": "0"})

    await tools["web_search"].execute_fn("call-1", {"query": "q1"})
    await tools["web_search"].execute_fn("call-2", {"query": "q2"})

    assert len([call for call in http.calls if "serper.dev" in call[1]]) == 2


async def test_clearing_lifts_the_quarantine(install: Any, env: Any) -> None:
    http = FakeHttp(
        {
            "google.serper.dev": ws.ProviderAuthError("google.serper.dev: HTTP 403"),
            "api.exa.ai": FakeResponse(payload=EXA_OK),
        }
    )
    api, tools = install(http)
    env(_QUARANTINE_ENV)

    await tools["web_search"].execute_fn("call-1", {"query": "q1"})
    report = api.commands["websearch"]("clear", None)
    await tools["web_search"].execute_fn("call-2", {"query": "q2"})

    assert "1 quarantined backend" in report
    assert len([call for call in http.calls if "serper.dev" in call[1]]) == 2


async def test_a_transient_fault_is_never_quarantined(install: Any, env: Any) -> None:
    http = FakeHttp(
        {
            "google.serper.dev": ws.ProviderError("google.serper.dev: HTTP 503"),
            "api.exa.ai": FakeResponse(payload=EXA_OK),
        }
    )
    api, tools = install(http)
    env(_QUARANTINE_ENV)

    await tools["web_search"].execute_fn("call-1", {"query": "q1"})
    status = api.commands["websearch"]("status", None)
    await tools["web_search"].execute_fn("call-2", {"query": "q2"})

    assert "skipped for now" not in status
    assert len([call for call in http.calls if "serper.dev" in call[1]]) == 2


async def test_status_names_the_backend_whose_key_was_rejected(install: Any, env: Any) -> None:
    http = FakeHttp(
        {
            "google.serper.dev": ws.ProviderAuthError("google.serper.dev: HTTP 403"),
            "api.exa.ai": FakeResponse(payload=EXA_OK),
        }
    )
    api, tools = install(http)
    env(_QUARANTINE_ENV)

    await tools["web_search"].execute_fn("call-1", {"query": "q"})
    status = api.commands["websearch"]("status", None)

    assert "serper" in status
    assert "rejected" in status
    assert "key-rejected backends skipped" in status


# -- fetching ------------------------------------------------------------------


async def test_reader_mode_and_the_paging_window(install: Any, env: Any) -> None:
    long_page = "<html><body><p>" + ("lorem ipsum dolor sit amet " * 60) + "</p></body></html>"
    http = FakeHttp({"example.test": FakeResponse(text=long_page, url="https://example.test/doc")})
    _, tools = install(http)
    env({"TAU_WEB_FETCH_MAX_CHARS": "600"})

    first = await tools["web_fetch"].execute_fn("call-1", {"url": "https://example.test/doc"})

    assert "chars 0-600" in first.text
    assert "offset=600" in first.text
    assert first.details["chars"] == 600
    assert first.details["totalChars"] > 1200

    second = await tools["web_fetch"].execute_fn(
        "call-2", {"url": "https://example.test/doc", "offset": 600, "maxChars": 600}
    )

    assert "chars 600-1200" in second.text


async def test_a_schemeless_url_is_fetched_over_https(install: Any) -> None:
    http = FakeHttp({"example.test": FakeResponse(text="<html><body>ok</body></html>")})
    _, tools = install(http)

    await tools["web_fetch"].execute_fn("call-1", {"url": "example.test"})

    assert http.calls[0][1] == "https://example.test"


async def test_non_html_content_is_returned_as_is(install: Any) -> None:
    http = FakeHttp(
        {
            "api.github.com": FakeResponse(
                payload=None,
                text='{"tag": "v1.2.3"}',
                headers={"content-type": "application/json"},
            )
        }
    )
    _, tools = install(http)

    result = await tools["web_fetch"].execute_fn(
        "call-1", {"url": "https://api.github.com/repos/x/y/releases/latest"}
    )

    assert "application/json" in result.text
    assert '"tag": "v1.2.3"' in result.text


async def test_a_transport_failure_stays_a_clean_message(install: Any) -> None:
    http = FakeHttp({"down.test": ws.ProviderError("down.test: timed out after 15s")})
    _, tools = install(http)

    result = await tools["web_fetch"].execute_fn("call-1", {"url": "https://down.test/"})

    assert result.details["status"] == "error"
    assert "timed out" in result.text


async def test_only_http_urls_are_fetched(install: Any) -> None:
    http = FakeHttp()
    _, tools = install(http)

    result = await tools["web_fetch"].execute_fn("call-1", {"url": "file:///etc/passwd"})

    assert result.details["status"] == "error"
    assert http.calls == []


# -- the /websearch command ----------------------------------------------------


def test_status_names_keys_but_never_prints_a_value(install: Any, env: Any) -> None:
    api, _ = install(FakeHttp())
    env({"EXA_API_KEY": "super-secret-value", "TAU_WEB_SEARCH_ALLOW_KEYLESS": "0"})

    text = api.commands["websearch"]("status", None)

    assert "exa" in text
    assert "found in EXA_API_KEY" in text
    assert "super-secret-value" not in text
    assert "missing PERPLEXITY_API_KEY" in text
    assert "keyless, disabled" in text


async def test_clearing_reports_the_cache_it_emptied(install: Any, env: Any) -> None:
    http = FakeHttp({"api.exa.ai": FakeResponse(payload=EXA_OK)})
    api, tools = install(http)
    env({"TAU_WEB_SEARCH_PROVIDER": "exa", "EXA_API_KEY": "k"})

    await tools["web_search"].execute_fn("call-1", {"query": "warm"})
    report = api.commands["websearch"]("clear", None)

    assert "cleared 1 cached query" in report


async def test_session_hooks_close_the_pooled_client(install: Any) -> None:
    http = FakeHttp()
    api, _ = install(http)

    for handler in api.hooks["session_shutdown"]:
        outcome = handler(object(), None)
        if hasattr(outcome, "__await__"):
            await outcome

    assert http.closed
