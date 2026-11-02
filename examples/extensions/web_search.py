"""Tau extension that adds web search and page reading as two tools.

The failure it fixes: Tau has no search tool at all (grep the installed package
for ``web_search`` and find nothing), so a model either answers from a frozen
cutoff or shells out to ``curl`` and pastes raw HTML into the transcript. This
extension registers ``web_search`` (ranked results, optional synthesized answer)
and ``web_fetch`` (readable page text with paging), and teaches the prompt when
to use them.

No new wheels. ``tau_ai`` already depends on ``httpx[socks]>=0.27``, so the
provider calls run on the interpreter that hosts Tau, and every payload is
parsed with the standard library. Third-party search SDKs (``exa-py``,
``firecrawl-py``, ``ddgs``) are thin wrappers over the same four HTTP endpoints
and would add a dependency the extension does not need; the kernel extension's
stdlib-only rule holds here too.

Backends, tried in order until one returns usable results
(``TAU_WEB_SEARCH_PROVIDER`` pins one, and aliases like ``ddgs``/``pplx``/``fc``/
``google`` are accepted):

* ``serper``     -- POST https://google.serper.dev/search (Google organic+news)
* ``exa``        -- POST https://api.exa.ai/search (semantic, returns page text)
* ``perplexity`` -- POST https://api.perplexity.ai/chat/completions (answer + citations)
* ``firecrawl``  -- POST https://api.firecrawl.dev/v2/search (and /v2/scrape for fetch)
* ``duckduckgo`` -- GET https://html.duckduckgo.com/html/ (keyless last resort)

Keys are read from the environment only and never echoed: diagnostics carry
provider names, HTTP status codes, and key *variable names*. Serper is checked
as ``SERPER_API_KEY`` first and ``SERPER_DEV_API_KEY`` second, matching the
name the Serper console actually issues.

Failure policy: a provider error degrades to the next provider and is reported
inline, so a dead key still yields results rather than a failed tool call. A
backend that answers with titles only (Exa plans without page text) also steps
aside for one that returns snippets, and is used only as a last resort
(``TAU_WEB_SEARCH_SNIPPET_FLOOR=0`` disables that). Nothing is retried more than
once, only on 429/5xx/transport errors, and every request has a wall-clock
timeout -- a hung search engine must not stall a turn.
HTML parsing is defensive: unknown response shapes produce zero results and a
diagnostic rather than an exception.

Install by copying into ``~/.tau/extensions/``, or run:

    tau -e examples/extensions/web_search.py
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from tau_agent.messages import TextContent
from tau_agent.tools import (
    AgentTool,
    AgentToolResult,
    ToolCancellationToken,
    ToolUpdateCallback,
)
from tau_agent.types import JSONValue
from tau_coding.extensions import (
    ExtensionAPI,
    ExtensionCommandContext,
    ExtensionContext,
    SessionShutdownEvent,
)

COMMAND_NAME = "websearch"

#: Backend order used when ``TAU_WEB_SEARCH_PROVIDER=auto``. Deterministic
#: engines come before the synthesizing one: they are cheaper, faster, and give
#: citable URLs the model can re-read with web_fetch.
PROVIDERS: tuple[str, ...] = ("serper", "exa", "perplexity", "firecrawl", "duckduckgo")

#: Environment variables probed per provider, in priority order.
KEY_ENVS: dict[str, tuple[str, ...]] = {
    "serper": ("SERPER_API_KEY", "SERPER_DEV_API_KEY"),
    "exa": ("EXA_API_KEY",),
    "perplexity": ("PERPLEXITY_API_KEY", "PPLX_API_KEY"),
    "firecrawl": ("FIRECRAWL_API_KEY",),
    "duckduckgo": (),
}

FRESHNESS: tuple[str, ...] = ("day", "week", "month", "year")
FETCH_FORMATS: tuple[str, ...] = ("text", "markdown", "html")

#: Shorthand names accepted wherever a provider name is taken. The short forms
#: are what people (and models) type; the canonical names above stay the ones
#: shown in status output and error messages.
PROVIDER_ALIASES: dict[str, str] = {
    "ddg": "duckduckgo",
    "ddgs": "duckduckgo",
    "duck": "duckduckgo",
    "html": "duckduckgo",
    "duckduckgo_html": "duckduckgo",
    "pplx": "perplexity",
    "ppl": "perplexity",
    "perplexity_search": "perplexity",
    "sonar": "perplexity",
    "google": "serper",
    "serperdev": "serper",
    "serper_dev": "serper",
    "fc": "firecrawl",
    "firecrawl_search": "firecrawl",
    "exa_ai": "exa",
    "exasearch": "exa",
    "semantic": "exa",
    "any": "auto",
    "all": "auto",
    "default": "auto",
    "auto": "auto",
}


def resolve_provider(raw: object) -> str:
    """Canonicalize a provider name or alias; ``""`` and unknown stay unusable."""
    name = str(raw or "").strip().lower()
    return PROVIDER_ALIASES.get(name, name)


DEFAULT_TIMEOUT_SECONDS = 15.0
DEFAULT_MAX_RESULTS = 8
MAX_RESULTS_LIMIT = 20
DEFAULT_FETCH_MAX_CHARS = 20_000
FETCH_MAX_CHARS_LIMIT = 120_000
DEFAULT_SNIPPET_CHARS = 320
CACHE_TTL_SECONDS = 600
CACHE_MAX_ENTRIES = 64
#: A backend whose credentials are rejected is skipped for this long instead of
#: costing a 403 round-trip on every search. Transient faults (timeouts, 5xx) are
#: never quarantined: they are retried every call.
DEAD_TTL_SECONDS = 900
AUTH_STATUSES: frozenset[int] = frozenset({401, 403})
RETRY_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


class ProviderError(RuntimeError):
    """One backend failed; the caller may fall through to the next one."""


class ProviderAuthError(ProviderError):
    """Credentials were rejected, so retrying this backend is pointless for a while."""


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Settings:
    """Resolved per-call configuration: env only, so a shell export takes effect."""

    provider: str
    max_results: int
    snippet_chars: int
    fetch_max_chars: int
    timeout_seconds: float
    allow_keyless: bool
    #: A backend that hands back titles with no snippet (Exa on accounts without
    #: page text enabled) is a weak answer for a search turn, so the chain keeps
    #: going and only falls back to it when nothing better exists.
    snippet_floor: bool
    perplexity_model: str
    cache_ttl_seconds: int
    dead_ttl_seconds: int
    keys: Mapping[str, str]

    def available_providers(self) -> tuple[str, ...]:
        """Backends usable right now, in preference order."""
        usable = [name for name in PROVIDERS if name == "duckduckgo" or name in self.keys]
        if not self.allow_keyless:
            usable = [name for name in usable if name != "duckduckgo"]
        if self.provider != "auto":
            return (self.provider,) if self.provider in usable else ()
        return tuple(usable)

    def missing_providers(self) -> tuple[str, ...]:
        """Backends that are unusable, paired with the variable to set."""
        return tuple(
            name
            for name in PROVIDERS
            if name != "duckduckgo"
            and name not in self.keys
            and (self.provider == "auto" or self.provider == name)
        )


def _env_int(env: Mapping[str, str], name: str, default: int, *, low: int, high: int) -> int:
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(low, min(value, high))


def _env_flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = (env.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def settings_from_env(env: Mapping[str, str]) -> Settings:
    """Read configuration and API keys. Called per call, not once at setup."""
    wanted = resolve_provider(env.get("TAU_WEB_SEARCH_PROVIDER"))
    if wanted not in PROVIDERS and wanted != "auto":
        wanted = "auto"
    keys: dict[str, str] = {}
    for provider, names in KEY_ENVS.items():
        for name in names:
            value = (env.get(name) or "").strip()
            if value:
                keys[provider] = value
                break
    return Settings(
        provider=wanted,
        max_results=_env_int(
            env, "TAU_WEB_SEARCH_MAX_RESULTS", DEFAULT_MAX_RESULTS, low=1, high=MAX_RESULTS_LIMIT
        ),
        snippet_chars=_env_int(
            env, "TAU_WEB_SEARCH_SNIPPET_CHARS", DEFAULT_SNIPPET_CHARS, low=0, high=2000
        ),
        fetch_max_chars=_env_int(
            env,
            "TAU_WEB_FETCH_MAX_CHARS",
            DEFAULT_FETCH_MAX_CHARS,
            low=500,
            high=FETCH_MAX_CHARS_LIMIT,
        ),
        timeout_seconds=float(
            _env_int(
                env, "TAU_WEB_SEARCH_TIMEOUT_SECONDS", int(DEFAULT_TIMEOUT_SECONDS), low=2, high=120
            )
        ),
        allow_keyless=_env_flag(env, "TAU_WEB_SEARCH_ALLOW_KEYLESS", True),
        snippet_floor=_env_flag(env, "TAU_WEB_SEARCH_SNIPPET_FLOOR", True),
        perplexity_model=(env.get("TAU_WEB_SEARCH_PERPLEXITY_MODEL") or "sonar").strip() or "sonar",
        cache_ttl_seconds=_env_int(
            env, "TAU_WEB_SEARCH_CACHE_SECONDS", CACHE_TTL_SECONDS, low=0, high=86_400
        ),
        dead_ttl_seconds=_env_int(
            env, "TAU_WEB_SEARCH_DEAD_TTL_SECONDS", DEAD_TTL_SECONDS, low=0, high=86_400
        ),
        keys=keys,
    )


# --------------------------------------------------------------------------- #
# Result shapes                                                               #
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class SearchResult:
    """One ranked hit."""

    title: str
    url: str
    snippet: str = ""
    published: str = ""
    site: str = ""

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "published": self.published,
            "site": self.site,
        }


@dataclass(slots=True)
class SearchOutcome:
    """What one ``web_search`` call produced, including the fail trail."""

    provider: str = ""
    query: str = ""
    answer: str = ""
    results: list[SearchResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    elapsed_ms: int = 0
    cached: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.results or self.answer)


def site_of(url: str) -> str:
    host = urlparse(url).netloc
    return host[4:] if host.startswith("www.") else host


def _clean_text(value: object, limit: int) -> str:
    """Squeeze any provider string into one whitespace-normalized line."""
    text = " ".join(str(value or "").split())
    if limit > 0 and len(text) > limit:
        text = text[: limit - 1].rstrip() + "\u2026"
    return text


# --------------------------------------------------------------------------- #
# HTTP plumbing                                                               #
# --------------------------------------------------------------------------- #


class Http:
    """One shared pooled async client; connections survive across tool calls."""

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=min(self.timeout, 10.0)),
                follow_redirects=True,
                limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
            )
        return self._client

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json_body: Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        content: bytes | None = None,
        accept: str = "application/json",
    ) -> httpx.Response:
        """Issue one request with a single bounded retry on retryable failures.

        Secrets never reach the exception text: only the status, the URL host,
        and a truncated response body are kept.
        """
        client = await self.client()
        merged = {"User-Agent": USER_AGENT, "Accept": accept, **headers}
        last: str = ""
        for attempt in (0, 1):
            try:
                response = await client.request(
                    method,
                    url,
                    headers=merged,
                    json=json_body,
                    params=params,
                    content=content,
                    timeout=httpx.Timeout(self.timeout, connect=min(self.timeout, 10.0)),
                )
            except httpx.TimeoutException:
                last = f"{urlparse(url).netloc}: timed out after {self.timeout:g}s"
            except httpx.HTTPError as exc:
                last = f"{urlparse(url).netloc}: {type(exc).__name__}"
            else:
                if response.status_code in RETRY_STATUSES and attempt == 0:
                    last = f"{urlparse(url).netloc}: HTTP {response.status_code}"
                    await asyncio.sleep(0.6)
                    continue
                return response
            if attempt == 0:
                await asyncio.sleep(0.6)
        raise ProviderError(last or "request failed")

    async def post_json(
        self, url: str, *, headers: Mapping[str, str], body: Mapping[str, Any]
    ) -> Any:
        """POST and decode JSON, mapping non-2xx to a ``ProviderError``."""
        response = await self.request("POST", url, headers=headers, json_body=body)
        return _decode(response, url)

    async def get_text(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        """GET a document (no JSON assumption)."""
        return await self.request("GET", url, headers=headers or {}, params=params, accept="*/*")

    async def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None and not client.is_closed:
            await client.aclose()


def _decode(response: httpx.Response, url: str) -> Any:
    host = urlparse(url).netloc
    if response.status_code in AUTH_STATUSES:
        raise ProviderAuthError(f"{host}: HTTP {response.status_code} {_body_hint(response)}")
    if response.status_code >= 400:
        raise ProviderError(f"{host}: HTTP {response.status_code} {_body_hint(response)}")
    try:
        return response.json()
    except ValueError:
        raise ProviderError(f"{host}: response was not JSON ({response.text[:120]!r})") from None


def _body_hint(response: httpx.Response) -> str:
    """One line of an error body, with echoed credentials scrubbed."""
    text = " ".join(response.text.split())
    for marker in ("Bearer ", "api_key=", "x-api-key"):
        index = text.lower().find(marker.lower())
        if index >= 0:
            text = f"{text[:index]}{marker}[redacted]"
    return text[:160]


# --------------------------------------------------------------------------- #
# Providers                                                                   #
# --------------------------------------------------------------------------- #


def _days_back(freshness: str) -> int | None:
    return {"day": 1, "week": 7, "month": 31, "year": 365}.get(freshness)


async def _search_serper(
    http: Http, settings: Settings, query: str, count: int, freshness: str
) -> list[SearchResult]:
    body: dict[str, Any] = {"q": query, "num": max(1, min(count, MAX_RESULTS_LIMIT))}
    tbs = {"day": "qdr:d", "week": "qdr:w", "month": "qdr:m", "year": "qdr:y"}.get(freshness)
    if tbs:
        body["tbs"] = tbs
    payload = await http.post_json(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": settings.keys["serper"], "Content-Type": "application/json"},
        body=body,
    )
    return parse_serper(payload, settings.snippet_chars)


def parse_serper(payload: Any, snippet_chars: int) -> list[SearchResult]:
    """Map Serper's organic/news arrays; unknown shapes yield nothing."""
    if not isinstance(payload, Mapping):
        return []
    results: list[SearchResult] = []
    seen: set[str] = set()
    for key in ("organic", "news"):
        for item in payload.get(key) or []:
            if not isinstance(item, Mapping):
                continue
            url = str(item.get("link") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            results.append(
                SearchResult(
                    title=_clean_text(item.get("title"), 200) or url,
                    url=url,
                    snippet=_clean_text(item.get("snippet"), snippet_chars),
                    published=_clean_text(item.get("date"), 40),
                    site=_clean_text(item.get("source"), 80) or site_of(url),
                )
            )
    return results[:MAX_RESULTS_LIMIT]


async def _search_exa(
    http: Http, settings: Settings, query: str, count: int, freshness: str
) -> list[SearchResult]:
    body: dict[str, Any] = {
        "query": query,
        "numResults": max(1, min(count, MAX_RESULTS_LIMIT)),
        "type": "auto",
        "text": True,
        # Accounts that do not return page text still answer this, and the
        # parser joins the highlights into the snippet either way.
        "highlights": {"num_sentences": 2, "highlightsPerUrl": 1},
    }
    days = _days_back(freshness)
    if days:
        body["startPublishedDate"] = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%d")
    payload = await http.post_json(
        "https://api.exa.ai/search",
        headers={"x-api-key": settings.keys["exa"], "Content-Type": "application/json"},
        body=body,
    )
    return parse_exa(payload, settings.snippet_chars)


def parse_exa(payload: Any, snippet_chars: int) -> list[SearchResult]:
    """Map Exa results. Text may arrive as ``text``, ``snippet``, or highlights."""
    if not isinstance(payload, Mapping):
        return []
    items = payload.get("results") or payload.get("sections") or []
    results: list[SearchResult] = []
    if not isinstance(items, list):
        return results
    for item in items:
        if not isinstance(item, Mapping):
            continue
        url = str(item.get("url") or item.get("uri") or "").strip()
        if not url:
            continue
        text = item.get("snippet") or item.get("text") or item.get("highlights") or ""
        if isinstance(text, list):
            text = " ".join(str(part) for part in text)
        results.append(
            SearchResult(
                title=_clean_text(item.get("title"), 200) or site_of(url),
                url=url,
                snippet=_clean_text(text, snippet_chars),
                published=_clean_text(item.get("publishedDate"), 40)[:10],
                site=site_of(url),
            )
        )
    return results


async def _search_perplexity(
    http: Http, settings: Settings, query: str, count: int, freshness: str
) -> tuple[str, list[SearchResult]]:
    hint = {
        "day": "within the last day",
        "week": "within the last week",
        "month": "within the last month",
        "year": "within the last year",
    }.get(freshness, "")
    scope = f" Prefer results {hint}." if hint else ""
    prompt = (
        "Answer the following research request in at most 200 words, with numbered inline "
        f"citations that map to the sources you used.{scope}"
        f"\n\nRequest: {query}"
    )
    payload = await http.post_json(
        "https://api.perplexity.ai/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.keys['perplexity']}",
            "Content-Type": "application/json",
        },
        body={
            "model": settings.perplexity_model,
            "max_tokens": 700,
            "temperature": 0.2,
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    answer, citations = parse_perplexity(payload)
    results = [
        SearchResult(title=site_of(url), url=url, snippet="cited source")
        for url in citations[:count]
    ]
    return answer, results


def parse_perplexity(payload: Any) -> tuple[str, list[str]]:
    """Return (answer text, citation URLs). Handles both citation shapes."""
    if not isinstance(payload, Mapping):
        return "", []
    choices = payload.get("choices") or []
    answer = ""
    citations_raw: Any = payload.get("citations") or []
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        message = choices[0].get("message")
        if isinstance(message, Mapping):
            answer = _clean_text(message.get("content"), 4000)
        if isinstance(message, Mapping):
            meta = message.get("meta") if isinstance(message.get("meta"), Mapping) else {}
            citations_raw = citations_raw or meta.get("citations") or []
    urls: list[str] = []
    if isinstance(citations_raw, list):
        for item in citations_raw:
            if isinstance(item, str):
                urls.append(item)
            elif isinstance(item, Mapping):
                url = str(item.get("url") or "").strip()
                if url:
                    urls.append(url)
    return answer, [url for url in urls if url.startswith("http")]


async def _search_firecrawl(
    http: Http, settings: Settings, query: str, count: int, freshness: str
) -> list[SearchResult]:
    body: dict[str, Any] = {
        "query": query,
        "limit": max(1, min(count, MAX_RESULTS_LIMIT)),
        "sources": [{"type": "web"}],
    }
    tbs = {"day": "d", "week": "w", "month": "m", "year": "y"}.get(freshness)
    if tbs:
        body["tbs"] = f"qdr:{tbs}"
    payload = await _firecrawl_post(http, settings, "/search", body)
    return parse_firecrawl(payload, settings.snippet_chars)


async def _firecrawl_post(
    http: Http, settings: Settings, path: str, body: Mapping[str, Any]
) -> Any:
    """POST against v2, falling back to v1 when the install predates it."""
    headers = {
        "Authorization": f"Bearer {settings.keys['firecrawl']}",
        "Content-Type": "application/json",
    }
    try:
        return await http.post_json(
            f"https://api.firecrawl.dev/v2{path}", headers=headers, body=body
        )
    except ProviderError as exc:
        if "HTTP 404" not in str(exc):
            raise
        return await http.post_json(
            f"https://api.firecrawl.dev/v1{path}", headers=headers, body=body
        )


def parse_firecrawl(payload: Any, snippet_chars: int) -> list[SearchResult]:
    """Accept both ``data.web`` (v2) and a flat ``data`` list (v1)."""
    if not isinstance(payload, Mapping):
        return []
    data = payload.get("data")
    if isinstance(data, Mapping):
        items: Any = data.get("web") or data.get("results") or []
    elif isinstance(data, list):
        items = data
    else:
        items = payload.get("results") or []
    results: list[SearchResult] = []
    if not isinstance(items, list):
        return results
    for item in items:
        if not isinstance(item, Mapping):
            continue
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        results.append(
            SearchResult(
                title=_clean_text(item.get("title"), 200) or site_of(url),
                url=url,
                snippet=_clean_text(item.get("description") or item.get("snippet"), snippet_chars),
                published=_clean_text(item.get("publishedDate") or item.get("date"), 40)[:10],
                site=site_of(url),
            )
        )
    return results


async def _search_duckduckgo(
    http: Http, settings: Settings, query: str, count: int, freshness: str
) -> list[SearchResult]:
    params = {"q": query}
    if freshness:
        params["df"] = {"day": "d", "week": "w", "month": "m", "year": "y"}[freshness]
    response = await http.get_text(
        "https://html.duckduckgo.com/html/",
        params=params,
        headers={"Accept": "text/html,application/xhtml+xml", "Accept-Language": "en-US,en;q=0.8"},
    )
    if response.status_code >= 400:
        raise ProviderError(f"html.duckduckgo.com: HTTP {response.status_code}")
    results = parse_duckduckgo(response.text, settings.snippet_chars)
    if not results:
        # DDG answers a suspicious address with an "anomaly" challenge page that
        # carries no result anchors at all, which reads differently from a
        # layout change: waiting or configuring a keyed backend helps here, and
        # nothing helps in the other case.
        if any(marker in response.text.lower() for marker in ("anomaly", "challenge")):
            raise ProviderError(
                "html.duckduckgo.com: served a challenge page (this address is rate-limited)"
            )
        raise ProviderError("html.duckduckgo.com: no results parsed (layout change?)")
    return results[:count]


class _DdgResults(HTMLParser):
    """Pull ``result__a`` anchors and ``result__snippet`` blocks out of DDG HTML."""

    def __init__(self, snippet_chars: int) -> None:
        super().__init__(convert_charrefs=True)
        self.snippet_chars = snippet_chars
        self.links: list[tuple[str, str]] = []
        self.snippets: list[str] = []
        self._capture: str | None = None
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = " ".join(str(value or "") for name, value in attrs if name == "class")
        if tag == "a" and "result__a" in classes:
            href = next((str(value or "") for name, value in attrs if name == "href"), "")
            self._capture = "link"
            self._buffer = [href]
        elif tag in {"a", "div"} and "result__snippet" in classes:
            self._capture = "snippet"
            self._buffer = []

    def handle_endtag(self, tag: str) -> None:
        if self._capture == "link" and tag == "a":
            href = self._buffer[0] if self._buffer else ""
            title = _clean_text("".join(self._buffer[1:]), 200)
            self.links.append((unwrap_ddg(href), title))
            self._capture = None
        elif self._capture == "snippet" and tag in {"a", "div"}:
            self.snippets.append(_clean_text("".join(self._buffer), self.snippet_chars))
            self._capture = None

    def handle_data(self, data: str) -> None:
        if self._capture is not None:
            self._buffer.append(data)


def unwrap_ddg(href: str) -> str:
    """DDG wraps hits as //duckduckgo.com/l/?uddg=<encoded>&rut=<hash>."""
    href = href.strip()
    if not href:
        return ""
    if href.startswith("//"):
        href = f"https:{href}"
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg") or []
        if target:
            return unquote(target[0])
    return href


def parse_duckduckgo(markup: str, snippet_chars: int) -> list[SearchResult]:
    parser = _DdgResults(snippet_chars)
    parser.feed(markup)
    parser.close()
    results: list[SearchResult] = []
    seen: set[str] = set()
    for index, (url, title) in enumerate(parser.links):
        if not url.startswith("http") or url in seen:
            continue
        seen.add(url)
        snippet = parser.snippets[index] if index < len(parser.snippets) else ""
        results.append(
            SearchResult(title=title or site_of(url), url=url, snippet=snippet, site=site_of(url))
        )
    return results


# --------------------------------------------------------------------------- #
# Page reading                                                                #
# --------------------------------------------------------------------------- #


#: Dropped entirely. Chrome rarely carries the answer, and reader-mode noise is
#: what makes a fetched page useless in a transcript: MkDocs Material puts the
#: whole sidebar and search UI in these elements, so "skip to content", the
#: section index, and repeated titles otherwise dominate the first window.
_SKIP_TAGS: frozenset[str] = frozenset(
    {
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "head",
        "iframe",
        "canvas",
        "nav",
        "aside",
        "header",
        "footer",
        "form",
        "picture",
        "button",
        "select",
        "option",
    }
)
_BLOCK_TAGS: frozenset[str] = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "main",
        "li",
        "ul",
        "ol",
        "br",
        "hr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "blockquote",
        "pre",
        "tr",
        "table",
        "figure",
        "figcaption",
        "details",
        "summary",
    }
)
_CELL_TAGS: frozenset[str] = frozenset({"td", "th"})

#: ``header``/``footer`` mean page chrome at the top level but the article's own
#: header (the headline!) inside ``<article>``/``<main>``. Dropping them
#: unconditionally loses the headline, which is the one thing a news fetch needs.
_CONTEXTUAL_TAGS: frozenset[str] = frozenset({"header", "footer"})
_ARTICLE_SCOPE: frozenset[str] = frozenset({"article", "main"})

#: Elements that cannot contain content, so they need no skip bookkeeping.
_VOID_TAGS: frozenset[str] = frozenset(
    {
        "img",
        "input",
        "link",
        "meta",
        "source",
        "track",
        "wbr",
        "col",
        "base",
        "embed",
        "param",
        "area",
    }
)

#: Chrome that survives the tag-based skips above because sites mark it with
#: class names instead of semantic tags -- news sites especially ("menu-main",
#: "footer-container", "share-bar"). The first hyphen/underscore-separated token
#: must be one of these words, so "nav-header" is dropped while "article-header",
#: "entry-footer" and "post-share" (the actual headline and body wrappers) are
#: kept. Deliberately absent: "header", "content", "text", "search", "ad" --
#: too easy to hit a real article wrapper.
_BOILERPLATE_TOKEN: re.Pattern[str] = re.compile(
    r"^(?:site|main|global|top|mobile|sub|primary|secondary)-"
    r"?(?:nav|navbar|navigation|menu|menubar|sidebar|side-nav|sidenav|masthead|breadcrumb|breadcrumbs)"
    r"|(?:^|-)(?:nav|navbar|navigation|menu|menubar|sidebar|sidenav|footer|masthead|"
    r"breadcrumb|breadcrumbs|social|share|sharing|comment|comments|related|promo|"
    r"newsletter|subscribe|signup|sign-up|cookie|consent|advert|ads|sponsor|paywall|"
    r"modal|popup|toc|pagination)(?:$|-|_)"
    r"|^(?:nav|menu|footer|sidebar)$",
    re.IGNORECASE,
)


class _HtmlToText(HTMLParser):
    """Crude but adequate reader-mode extraction: no external deps, no JS.

    Skipping works on a depth stack rather than a counter because the two skip
    rules (tag name and class/id match) both start at an arbitrary nesting level
    and end when that element closes.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._depth = 0
        self._skip_from: list[int] = []
        self._stack: list[str] = []
        self._in_title = False

    @staticmethod
    def _is_boilerplate(attrs: list[tuple[str, str | None]]) -> bool:
        for name, value in attrs:
            if name not in {"class", "id"} or not value:
                continue
            for token in str(value).replace(".", " ").split():
                if _BOILERPLATE_TOKEN.match(token.strip("-_.")):
                    return True
        return False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
            return
        if tag in _VOID_TAGS:
            return
        self._depth += 1
        self._stack.append(tag)
        chrome = tag in _SKIP_TAGS and not (
            tag in _CONTEXTUAL_TAGS and any(name in _ARTICLE_SCOPE for name in self._stack[:-1])
        )
        if chrome or self._is_boilerplate(attrs):
            self._skip_from.append(self._depth)
            return
        if self._skip_from:
            return
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")
        elif tag in _CELL_TAGS:
            self.parts.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
            return
        if tag in _VOID_TAGS:
            return
        if self._depth:
            self._depth -= 1
        for index in range(len(self._stack) - 1, -1, -1):
            if self._stack[index] == tag:
                del self._stack[index:]
                break
        while self._skip_from and self._skip_from[-1] > self._depth:
            self._skip_from.pop()
        if self._skip_from:
            return
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
            return
        if self._skip_from:
            return
        self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        lines = [" ".join(line.split()) for line in raw.splitlines()]
        out: list[str] = []
        for line in lines:
            if line:
                out.append(line)
            elif out and out[-1] != "":
                out.append("")
        return "\n".join(out).strip()


def html_to_text(markup: str) -> tuple[str, str]:
    """Return (body text, document title)."""
    parser = _HtmlToText()
    parser.feed(markup)
    parser.close()
    return parser.text(), _clean_text(parser.title, 200)


@dataclass(slots=True)
class FetchOutcome:
    """A page, or why it could not be read."""

    url: str
    status: int
    content_type: str
    text: str
    title: str = ""
    total_chars: int = 0
    offset: int = 0
    notes: list[str] = field(default_factory=list)


async def fetch_page(
    http: Http,
    settings: Settings,
    url: str,
    *,
    max_chars: int,
    offset: int,
    output_format: str,
    signal: ToolCancellationToken | None,
) -> FetchOutcome:
    """Read one URL as text/markdown/html, escalating to Firecrawl when useful."""
    notes: list[str] = []
    if output_format == "markdown" and "firecrawl" in settings.keys:
        markdown = await _firecrawl_scrape(http, settings, url)
        if markdown is not None:
            return FetchOutcome(
                url=url,
                status=200,
                content_type="text/markdown",
                text=markdown,
                title="",
                total_chars=len(markdown),
                offset=offset,
                notes=[*notes, "rendered by firecrawl scrape"],
            )
    response = await http.get_text(
        url,
        headers={
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    if signal is not None and signal.is_cancelled():
        raise ProviderError("cancelled")
    content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
    body = response.text
    if response.status_code >= 400:
        notes.append(f"HTTP {response.status_code}")
        if "firecrawl" in settings.keys:
            markdown = await _firecrawl_scrape(http, settings, url)
            if markdown is not None:
                notes.append("recovered with firecrawl scrape")
                return FetchOutcome(
                    url=url,
                    status=response.status_code,
                    content_type="text/markdown",
                    text=markdown,
                    total_chars=len(markdown),
                    offset=offset,
                    notes=notes,
                )
        return FetchOutcome(
            url=url,
            status=response.status_code,
            content_type=content_type,
            text=_clean_text(body, 400),
            total_chars=len(body),
            offset=offset,
            notes=notes,
        )
    title = ""
    if output_format == "html":
        text = body
        notes.append("raw html: expect markup, not reader text")
    elif "html" in content_type or "<html" in body[:2000].lower():
        text, title = html_to_text(body)
        if len(text) < 240 and "firecrawl" in settings.keys:
            markdown = await _firecrawl_scrape(http, settings, url)
            if markdown is not None and len(markdown) > len(text):
                notes.append("thin static html; rendered by firecrawl scrape")
                text = markdown
                content_type = "text/markdown"
    else:
        text = body
        notes.append(f"non-html content-type {content_type or 'unknown'}: returned as-is")
    total = len(text)
    start = max(0, offset)
    window = text[start : start + max_chars]
    if start or total > start + max_chars:
        end = start + len(window)
        notes.append(f"chars {start}-{end} of {total}; pass offset={end} to continue")
    return FetchOutcome(
        url=str(response.url or url),
        status=response.status_code,
        content_type=content_type,
        text=window,
        title=title,
        total_chars=total,
        offset=start,
        notes=notes,
    )


async def _firecrawl_scrape(http: Http, settings: Settings, url: str) -> str | None:
    """Reader-mode markdown via Firecrawl; ``None`` when it cannot help."""
    try:
        payload = await _firecrawl_post(
            http,
            settings,
            "/scrape",
            {"url": url, "formats": ["markdown"], "onlyMainContent": True},
        )
    except ProviderError:
        return None
    if not isinstance(payload, Mapping):
        return None
    data = payload.get("data")
    if not isinstance(data, Mapping):
        return None
    for key in ("markdown", "html", "text"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        if key == "html":
            body, _ = html_to_text(value)
            return body
        return value.strip()
    return None


# --------------------------------------------------------------------------- #
# Presentation                                                                #
# --------------------------------------------------------------------------- #


def format_search(outcome: SearchOutcome) -> str:
    """Model-facing transcript for one search."""
    lines: list[str] = []
    if not outcome.ok:
        lines.append(f"No results for {outcome.query!r}.")
    else:
        header = (
            f"provider: {outcome.provider} \u00b7 {len(outcome.results)} results "
            f"\u00b7 {outcome.elapsed_ms} ms"
        )
        if outcome.cached:
            header += " \u00b7 cached"
        lines.append(header)
        if outcome.answer:
            lines += ["", "answer:", outcome.answer, ""]
        for index, result in enumerate(outcome.results, start=1):
            meta = " | ".join(part for part in (result.site, result.published) if part)
            lines.append(f"{index}. {result.title}")
            lines.append(f"   {result.url}" + (f" ({meta})" if meta else ""))
            if result.snippet and result.snippet != "cited source":
                lines.append(f"   {result.snippet}")
    notes = list(outcome.notes)
    if notes:
        lines += ["", "notes:"] + [f"- {note}" for note in notes]
    lines += ["", "Read a page in full with web_fetch(url); cite the URL you actually used."]
    return "\n".join(lines)


def format_fetch(outcome: FetchOutcome) -> str:
    lines = [
        f"url: {outcome.url}",
        f"status: {outcome.status} \u00b7 content-type: {outcome.content_type or 'unknown'}",
        f"chars: {outcome.total_chars}",
    ]
    if outcome.title:
        lines.append(f"title: {outcome.title}")
    if outcome.notes:
        lines.append("notes: " + "; ".join(outcome.notes))
    lines += ["", outcome.text if outcome.text.strip() else "(no extractable text)"]
    return "\n".join(lines)


def _result(text: str, *, details: dict[str, JSONValue]) -> AgentToolResult:
    return AgentToolResult(content=[TextContent(text=text)], details=details)


def _render_search_call(arguments: Mapping[str, JSONValue]) -> str | None:
    query = _clean_text(arguments.get("query"), 90)
    if not query:
        return None
    provider = str(arguments.get("provider") or "").strip()
    return f"web_search {query!r}" + (f" via {provider}" if provider else "")


def _render_search_result(result: AgentToolResult, *, expanded: bool) -> str | None:
    del expanded
    details = result.details if isinstance(result.details, Mapping) else {}
    count = details.get("count")
    provider = str(details.get("provider") or "none")
    marker = "\u2713" if isinstance(count, int) and count or details.get("hasAnswer") else "!"
    shown = count if count is not None else 0
    return f"[dim]{marker} web_search \u00b7 {provider} \u00b7 {shown} results[/dim]"


def _render_fetch_result(result: AgentToolResult, *, expanded: bool) -> str | None:
    del expanded
    details = result.details if isinstance(result.details, Mapping) else {}
    status = details.get("status")
    marker = (
        "[green]\u2713[/green]" if isinstance(status, int) and status < 400 else "[red]\u2717[/red]"
    )
    chars = details.get("chars")
    return f"{marker} [dim]web_fetch \u00b7 {status} \u00b7 {chars} chars[/dim]"


# --------------------------------------------------------------------------- #
# Setup                                                                       #
# --------------------------------------------------------------------------- #


def setup(tau: ExtensionAPI) -> None:  # noqa: C901 - one cohesive registration body
    """Register web_search, web_fetch, their renderers, and the status command."""
    http = Http(DEFAULT_TIMEOUT_SECONDS)
    cache: dict[tuple[str, str, int, str], tuple[float, SearchOutcome]] = {}
    quarantined: dict[str, float] = {}
    last_provider: list[str] = []

    def cached(
        provider: str, query: str, count: int, freshness: str, ttl: int
    ) -> SearchOutcome | None:
        if ttl <= 0:
            return None
        entry = cache.get((provider, query, count, freshness))
        if entry is None:
            return None
        stamp, outcome = entry
        if time.monotonic() - stamp > ttl:
            del cache[(provider, query, count, freshness)]
            return None
        outcome.cached = True
        return outcome

    def remember(
        provider: str, query: str, count: int, freshness: str, outcome: SearchOutcome, ttl: int
    ) -> None:
        if ttl <= 0 or not outcome.ok:
            return
        if len(cache) >= CACHE_MAX_ENTRIES:
            oldest = min(cache, key=lambda key: cache[key][0])
            del cache[oldest]
        cache[(provider, query, count, freshness)] = (time.monotonic(), outcome)

    async def run_search(
        settings: Settings,
        query: str,
        count: int,
        freshness: str,
        signal: ToolCancellationToken | None,
    ) -> SearchOutcome:
        """Try each usable backend in order; keep every failure as a note."""
        started = time.monotonic()
        notes: list[str] = []
        fallback: SearchOutcome | None = None
        if settings.provider != "auto" and settings.provider not in settings.available_providers():
            missing = ", ".join(KEY_ENVS.get(settings.provider, ())) or "unknown provider"
            notes.append(f"{settings.provider}: unavailable (set {missing})")
            return SearchOutcome(
                query=query, notes=notes, elapsed_ms=int((time.monotonic() - started) * 1000)
            )
        for provider in settings.available_providers():
            if signal is not None and signal.is_cancelled():
                notes.append("cancelled before trying " + provider)
                break
            stamp = quarantined.get(provider)
            if stamp is not None and settings.dead_ttl_seconds > 0:
                age = time.monotonic() - stamp
                if age < settings.dead_ttl_seconds:
                    left = int((settings.dead_ttl_seconds - age) / 60)
                    notes.append(
                        f"{provider}: skipped, credentials rejected {int(age / 60) or '<1'}m ago "
                        f"(retry in ~{left}m; /websearch clear retries now)"
                    )
                    continue
                del quarantined[provider]
            cached_hit = cached(provider, query, count, freshness, settings.cache_ttl_seconds)
            if cached_hit is not None:
                cached_hit.notes = [*notes, *cached_hit.notes]
                return cached_hit
            try:
                if provider == "serper":
                    results, answer = (
                        await _search_serper(http, settings, query, count, freshness),
                        "",
                    )
                elif provider == "exa":
                    results, answer = await _search_exa(http, settings, query, count, freshness), ""
                elif provider == "perplexity":
                    answer, results = await _search_perplexity(
                        http, settings, query, count, freshness
                    )
                elif provider == "firecrawl":
                    results, answer = (
                        await _search_firecrawl(http, settings, query, count, freshness),
                        "",
                    )
                else:
                    results, answer = (
                        await _search_duckduckgo(http, settings, query, count, freshness),
                        "",
                    )
            except ProviderAuthError as exc:
                quarantined[provider] = time.monotonic()
                notes.append(f"{provider} rejected the key: {exc}")
                continue
            except ProviderError as exc:
                notes.append(f"{provider} failed: {exc}")
                continue
            except Exception as exc:  # defensive: a provider must not kill the turn
                notes.append(f"{provider} raised {type(exc).__name__}: {exc}")
                continue
            outcome = SearchOutcome(
                provider=provider,
                query=query,
                answer=answer,
                results=results[:count],
                notes=notes,
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
            if outcome.ok and provider == "perplexity" and not outcome.answer:
                notes.append(f"{provider}: citations without an answer; trying the next backend")
                if fallback is None:
                    fallback = outcome
                continue
            if (
                outcome.ok
                and settings.snippet_floor
                and settings.snippet_chars > 0
                and not any(result.snippet for result in outcome.results)
            ):
                notes.append(f"{provider}: titles only, no snippets; trying the next backend")
                if fallback is None:
                    fallback = outcome
                continue
            if outcome.ok:
                last_provider.clear()
                last_provider.append(provider)
                remember(provider, query, count, freshness, outcome, settings.cache_ttl_seconds)
                return outcome
            notes.append(f"{provider}: 0 results")
        if fallback is not None:
            fallback.notes = [*notes, "no backend gave snippets; web_fetch(url) to read a result"]
            fallback.elapsed_ms = int((time.monotonic() - started) * 1000)
            if fallback.provider:
                last_provider.clear()
                last_provider.append(fallback.provider)
            return fallback
        return SearchOutcome(
            query=query, notes=notes, elapsed_ms=int((time.monotonic() - started) * 1000)
        )

    async def execute_web_search(
        tool_call_id: str,
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        """Search the web through the first healthy configured backend."""
        del tool_call_id, on_update
        query = _clean_text(arguments.get("query"), 400)
        if not query:
            return _result(
                "web_search needs a non-empty `query`.",
                details={"status": "error", "count": 0, "provider": None},
            )
        settings = http_rebind(settings_from_env(os.environ))
        count = settings.max_results
        raw_count = arguments.get("count")
        if isinstance(raw_count, (int, float)) and raw_count:
            count = max(1, min(int(raw_count), MAX_RESULTS_LIMIT))
        raw_provider = resolve_provider(arguments.get("provider"))
        if raw_provider and raw_provider != "auto":
            if raw_provider in PROVIDERS:
                settings = dataclass_replace(settings, provider=raw_provider)
            else:
                return _result(
                    f"Unknown provider {raw_provider!r}. Choose one of: {', '.join(PROVIDERS)} "
                    f"(aliases: {', '.join(sorted(set(PROVIDER_ALIASES)))}).",
                    details={"status": "error", "count": 0, "provider": None},
                )
        freshness = str(arguments.get("freshness") or "").strip().lower()
        if freshness not in FRESHNESS:
            freshness = ""
        outcome = await run_search(settings, query, count, freshness, signal)
        details: dict[str, JSONValue] = {
            "status": "ok" if outcome.ok else "empty",
            "provider": outcome.provider or None,
            "count": len(outcome.results),
            "hasAnswer": bool(outcome.answer),
            "cached": outcome.cached,
            "elapsedMs": outcome.elapsed_ms,
            "results": [result.as_dict() for result in outcome.results],
        }
        return _result(format_search(outcome), details=details)

    def http_rebind(settings: Settings) -> Settings:
        """Apply the configured timeout to the pooled client (per-request override)."""
        http.timeout = settings.timeout_seconds
        return settings

    async def execute_web_fetch(
        tool_call_id: str,
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        """Read one URL as reader-mode text, with paging."""
        del tool_call_id, on_update
        url = str(arguments.get("url") or "").strip()
        if not url:
            return _result(
                "web_fetch needs a non-empty `url`.", details={"status": "error", "chars": 0}
            )
        if urlparse(url).scheme not in {"http", "https"}:
            if urlparse(url).scheme:
                return _result(
                    f"web_fetch reads only http(s) URLs, got {url!r}.",
                    details={"status": "error", "chars": 0},
                )
            url = f"https://{url}"
        settings = http_rebind(settings_from_env(os.environ))
        max_chars = settings.fetch_max_chars
        raw_max = arguments.get("maxChars")
        if isinstance(raw_max, (int, float)) and raw_max:
            max_chars = max(500, min(int(raw_max), FETCH_MAX_CHARS_LIMIT))
        offset = 0
        raw_offset = arguments.get("offset")
        if isinstance(raw_offset, (int, float)) and raw_offset:
            offset = max(0, int(raw_offset))
        output_format = str(arguments.get("format") or "text").strip().lower()
        if output_format not in FETCH_FORMATS:
            output_format = "text"
        try:
            outcome = await fetch_page(
                http,
                settings,
                url,
                max_chars=max_chars,
                offset=offset,
                output_format=output_format,
                signal=signal,
            )
        except ProviderError as exc:
            return _result(
                f"web_fetch could not read {url}\nerror: {exc}\n"
                "Check the URL, raise TAU_WEB_SEARCH_TIMEOUT_SECONDS, or try web_search.",
                details={"status": "error", "chars": 0, "url": url},
            )
        except Exception as exc:  # defensive: a hostile page must not kill the turn
            return _result(
                f"web_fetch raised {type(exc).__name__} for {url}: {exc}",
                details={"status": "error", "chars": 0, "url": url},
            )
        details = {
            "status": outcome.status,
            "chars": len(outcome.text),
            "totalChars": outcome.total_chars,
            "url": outcome.url,
            "contentType": outcome.content_type,
        }
        return _result(format_fetch(outcome), details=details)

    def handle_status(args: str, context: ExtensionCommandContext) -> str:
        """Report detected keys (names only), backend order, and cache state."""
        del context
        request = args.strip().lower()
        settings = settings_from_env(os.environ)
        if request in {"clear", "reset"}:
            queries = len(cache)
            backends = len(quarantined)
            cache.clear()
            quarantined.clear()
            return f"websearch: cleared {queries} cached quer{'y' if queries == 1 else 'ies'}" + (
                f" and {backends} quarantined backend{'s' if backends != 1 else ''}"
                if backends
                else ""
            )
        usable = settings.available_providers()
        lines = [
            f"provider order: {' \u2192 '.join(usable) if usable else 'none usable'}"
            + (
                f" (pinned by TAU_WEB_SEARCH_PROVIDER={settings.provider})"
                if settings.provider != "auto"
                else ""
            ),
            "keys:",
        ]
        for provider in PROVIDERS:
            envs = KEY_ENVS.get(provider, ())
            if provider == "duckduckgo":
                state = (
                    "keyless"
                    if settings.allow_keyless
                    else "keyless, disabled by TAU_WEB_SEARCH_ALLOW_KEYLESS=0"
                )
            elif provider in settings.keys:
                state = f"found in {next(name for name in envs if os.environ.get(name))}"
                stamp = quarantined.get(provider)
                if stamp is not None:
                    state += f", rejected {int(time.monotonic() - stamp)}s ago (skipped for now)"
            else:
                state = "missing " + " or ".join(envs)
            lines.append(f"  {provider:<11} {state}")
        missing = settings.missing_providers()
        if missing:
            lines.append("unconfigured: " + ", ".join(missing))
        lines += [
            f"results: default {settings.max_results}, cap {MAX_RESULTS_LIMIT}; "
            f"snippet {settings.snippet_chars} chars"
            + (
                "; titles-only backends are skipped"
                if settings.snippet_floor and settings.snippet_chars
                else "; titles-only backends accepted (TAU_WEB_SEARCH_SNIPPET_FLOOR=1 to skip)"
            ),
            f"fetch: {settings.fetch_max_chars} chars per window, cap {FETCH_MAX_CHARS_LIMIT}",
            f"timeout: {settings.timeout_seconds:g}s per request, one retry on 429/5xx",
            f"cache: {len(cache)} entries, ttl {settings.cache_ttl_seconds}s; "
            f"key-rejected backends skipped for {settings.dead_ttl_seconds}s",
            f"last search backend: "
            f"{last_provider[0] if last_provider else 'not used yet in this session'}",
            "`/websearch clear` drops cached queries and retries key-rejected backends",
        ]
        return "\n".join(lines)

    tau.register_command(
        COMMAND_NAME,
        handle_status,
        description="Show web-search backends, detected keys, and limits; clear the cache",
        usage="[status|clear]",
    )

    tau.register_tool(
        AgentTool(
            name="web_search",
            label="web search",
            description=(
                "Search the web and return ranked results (title, URL, snippet). Use for anything "
                "you cannot know: recent events, releases, prices, current library APIs, error "
                "messages, or facts you must cite. Zero results is an answer, not a failure; a "
                "dead backend falls through to the next and the notes line says why."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1, "description": "The search query."},
                    "count": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_RESULTS_LIMIT,
                        "description": f"How many results (default {DEFAULT_MAX_RESULTS}).",
                    },
                    "freshness": {
                        "type": "string",
                        "enum": list(FRESHNESS),
                        "description": "Restrict to recent pages for time-sensitive answers.",
                    },
                    "provider": {
                        "type": "string",
                        "enum": list(PROVIDERS),
                        "description": (
                            "Force one backend; the default tries each in order. "
                            "Aliases like ddgs/pplx/fc are accepted too."
                        ),
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            execute_fn=execute_web_search,
            prompt_snippet=(
                "web_search: current facts, releases, docs, errors. web_fetch: read a page."
            ),
            prompt_guidelines=(
                "Search instead of guessing when a claim is time-sensitive, when you are unsure "
                "of a current API/version, or when the user asks what is happening now; report the "
                "outcome honestly, including zero results.",
                "After web_search, call web_fetch on the URL you actually rely on before stating "
                "details from it, and cite URLs rather than paraphrasing unseen pages.",
                "Never invent a URL or present training-cutoff knowledge as a live search result.",
            ),
            execution_mode="parallel",
            render_call=_render_search_call,
            render_result=_render_search_result,
        )
    )

    tau.register_tool(
        AgentTool(
            name="web_fetch",
            label="web fetch",
            description=(
                "Read one http(s) URL as reader-mode text (script/style stripped), with paging via "
                "offset. Use after web_search to read the source before citing it, or to pull a "
                "known page: changelogs, READMEs, API references, error pages."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "minLength": 1, "description": "The page to read."},
                    "maxChars": {
                        "type": "integer",
                        "minimum": 500,
                        "maximum": FETCH_MAX_CHARS_LIMIT,
                        "description": (
                            f"Characters per window (default {DEFAULT_FETCH_MAX_CHARS})."
                        ),
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Character offset for the next window of the same page.",
                    },
                    "format": {
                        "type": "string",
                        "enum": list(FETCH_FORMATS),
                        "description": (
                            "text (default, reader mode), markdown (Firecrawl when configured), "
                            "or html (raw markup)."
                        ),
                    },
                },
                "required": ["url"],
                "additionalProperties": False,
            },
            execute_fn=execute_web_fetch,
            prompt_snippet="web_fetch(url): read a page's text; use offset for long pages.",
            execution_mode="parallel",
            render_call=lambda arguments: _clean_text(arguments.get("url"), 100) or None,
            render_result=_render_fetch_result,
        )
    )

    @tau.on("session_start")
    def on_session_start(event: object, context: ExtensionContext) -> None:
        """Re-read config for the new session and drop caches inherited from another cwd."""
        del event, context
        cache.clear()

    @tau.on("session_shutdown")
    async def on_session_shutdown(event: SessionShutdownEvent, context: ExtensionContext) -> None:
        """Close the pooled client so no socket outlives the session."""
        del event, context
        await http.close()
