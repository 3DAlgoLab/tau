---
title: "Web search and page reading example extension"
---

Tau now carries `examples/extensions/web_search.py`, one file that adds two tools
and one command: `web_search` (ranked results, optional synthesized answer),
`web_fetch` (reader-mode page text, paged by `offset`), and `/websearch`
(`status` / `clear`). It gives a session the two things a model cannot supply
from weights — a current fact and the text of a specific URL — without adding a
dependency or a built-in tool.

## Problem

Without a search seam the model answers current questions from its cutoff and
cites nothing. The available workarounds were all worse: `bash curl` spends the
transcript on markup and leaks cookies into shell history, and a "just fetch the
site I guessed" loop produces invented URLs. Meanwhile the harness has no web
tool, and Pi's own `websearch`/`webfetch` extensions exist precisely because the
core deliberately does not grow one (see the posture section below).

## Shape

```text
web_search ──> provider ladder ──> first backend with a usable answer
                 │  serper → exa → perplexity → firecrawl → duckduckgo
                 └── each failure becomes a note line, never an exception

web_fetch  ──> direct GET + stdlib reader mode ──> optional Firecrawl scrape
```

No new wheels. `tau_ai` already depends on `httpx[socks]`, and every parser is
stdlib `html.parser` plus dict walks, so installing the extension stays one file
and one `httpx.AsyncClient` (pooled, closed on `session_shutdown`).

## The ladder, and why that order

Backends are tried in `PROVIDERS` order and the first one that returns a usable
answer wins. Deterministic engines come before the synthesizing one because they
are cheaper, faster, and return citable URLs the model can re-read with
`web_fetch`:

| Backend | Endpoint | Key | What it adds |
|---|---|---|---|
| `serper` | `google.serper.dev/search` | `SERPER_API_KEY` / `SERPER_DEV_API_KEY` | Google organic + news, `tbs` freshness |
| `exa` | `api.exa.ai/search` | `EXA_API_KEY` | semantic search, `highlights` per hit |
| `perplexity` | `api.perplexity.ai/chat/completions` | `PERPLEXITY_API_KEY` / `PPLX_API_KEY` | an answer paragraph plus citations |
| `firecrawl` | `api.firecrawl.dev/v2/search` | `FIRECRAWL_API_KEY` | SERP plus `/v2/scrape` for fetch |
| `duckduckgo` | `html.duckduckgo.com/html/` | none | last resort, gated by `TAU_WEB_SEARCH_ALLOW_KEYLESS` |

Provider names are canonical (`serper`, `exa`, `perplexity`, `firecrawl`,
`duckduckgo`); the short forms people actually type (`ddgs`, `pplx`, `fc`,
`google`, `semantic`, `any`) resolve through `PROVIDER_ALIASES` before the
lookup, in the tool call and in `TAU_WEB_SEARCH_PROVIDER` alike. The error for an
unknown name lists the canonical five plus the aliases, because the failure mode
it replaces is a model inventing a backend.

## Three failure modes that needed their own rule

**A dead backend must not end the turn.** Every provider call is wrapped, and a
failure appends a note and moves on; the tool returns `status: empty` with all
notes rather than raising. A note names the host and status, never a key.

**A rejected key must not cost a round-trip per search.** `401`/`403` puts the
backend in `quarantined` for `TAU_WEB_SEARCH_DEAD_TTL_SECONDS` (900s default);
later calls skip it and say so (`skipped, credentials rejected 4m ago (retry in
~14m; /websearch clear retries now)`). Transient faults (`429`/`5xx`, timeouts)
are never quarantined: they get the one bounded retry inside `Http.request` and
are retried every call. This came from a real session in which one rotated key
made every search pay a 403.

**Titles are not an answer.** Exa on an account without page text, and several
other engines, return titles with no snippet — technically results, useless for
reading. A result set with no snippet at all is held as `fallback` and the ladder
keeps going; only if nothing better exists does it come back, with the note
`no backend gave snippets; web_fetch(url) to read a result`.
`TAU_WEB_SEARCH_SNIPPET_FLOOR=0` restores "first backend that returns anything".
Perplexity gets the same rule when it answers with citations but an empty body.

## Reading a page

`web_fetch` is a GET plus a stdlib reader-mode pass, then a window
(`offset`/`maxChars`) so a long page costs one page of context at a time. Reader
mode is where most of the care went, because page chrome — not the paywall — is
what makes a fetched page useless in a transcript:

- Semantic chrome (`nav`, `aside`, `header`, `footer`, `form`, …) is skipped.
- Most real chrome is class-named, not semantic, so `_BOILERPLATE_TOKEN` matches
  the first token of a `class`/`id` against nav/menu/sidebar/footer/share/related/
  newsletter/consent/paywall words. Deliberately absent: `header`, `content`,
  `text`, `search`, `ad` — those hit article wrappers far too often.
- `header`/`footer` are *contextual*: inside `<article>` or `<main>` they are the
  headline and byline. A news index page proved this — the strict tag rule dropped
  every headline, which is the one thing the fetch was for.
- Skips are tracked on a depth stack, not a counter: an unclosed `</div>` from a
  real-world page unwinds the stack to the matching tag, so a stray end tag can
  never leak chrome into the answer.
- Three escalations to `POST /v2/scrape` (Firecrawl, only when configured):
  `format: "markdown"` asks for it directly, a `4xx` from the origin tries it as
  recovery, and a page that yields under 240 characters of static text takes it
  only if the rendered result is longer. Each path says which route produced the
  text, so the transcript never hides a fallback.

## Keys and prompt-injection posture

Keys are read from the environment per call, so a shell export takes effect
without a reload, and the precedence is explicit (`SERPER_API_KEY` before
`SERPER_DEV_API_KEY`, `PERPLEXITY_API_KEY` before `PPLX_API_KEY`). Values are
never printed: `/websearch status` names the variable it found, and `_body_hint`
truncates an error body and redacts anything after `Bearer `, `api_key=`, or
`x-api-key` before it can reach a transcript. Fetched pages are data for the
model, not instructions — the same stance the built-in `read` tool takes, and the
reason the tool description says to cite the URL it actually used.

## Configuration

| Variable | Meaning |
|---|---|
| `SERPER_API_KEY` / `SERPER_DEV_API_KEY` | Serper credentials |
| `EXA_API_KEY` | Exa credentials |
| `PERPLEXITY_API_KEY` / `PPLX_API_KEY` | Perplexity credentials |
| `FIRECRAWL_API_KEY` | Firecrawl search, and the fetch scrape fallback |
| `TAU_WEB_SEARCH_PROVIDER` | pin one backend (aliases accepted) |
| `TAU_WEB_SEARCH_MAX_RESULTS` | default result count (`8`, cap `20`) |
| `TAU_WEB_SEARCH_SNIPPET_CHARS` | per-result snippet budget (`320`) |
| `TAU_WEB_FETCH_MAX_CHARS` | default fetch window (`20000`, cap `120000`) |
| `TAU_WEB_SEARCH_TIMEOUT_SECONDS` | per-request timeout (`15`, floor `2`) |
| `TAU_WEB_SEARCH_ALLOW_KEYLESS` | allow scraping DuckDuckGo as last resort (`1`) |
| `TAU_WEB_SEARCH_SNIPPET_FLOOR` | skip titles-only backends (`1`) |
| `TAU_WEB_SEARCH_PERPLEXITY_MODEL` | model for the answer path (`sonar`) |
| `TAU_WEB_SEARCH_CACHE_SECONDS` | per-backend query cache (`600`, 64 entries) |
| `TAU_WEB_SEARCH_DEAD_TTL_SECONDS` | how long a rejected key is skipped (`900`) |

## Tau APIs used

`tau.register_tool` twice (`execution_mode="parallel"`, `render_call`,
`render_result`, and a `prompt_snippet` plus three `prompt_guidelines` so the
model reaches for the tools instead of guessing); `tau.register_command` for
`/websearch status|clear`; `tau.on("session_start")` to drop a cache inherited
from another cwd; `tau.on("session_shutdown")` to close the pooled client. The
JSON schemas are provider-facing: `count`/`maxChars` carry min/max, `freshness`
and `format` are enums, and `additionalProperties` is closed.

## Shipping posture

The example is **repo-only, not packaged and not a built-in**. It sits with
`python_kernel.py`, `stall_guard.py`, and `permission_gate.py` in
`examples/extensions/`, not in `src/tau_coding/data/examples/` (which
`tests/test_package_metadata.py` pins at four files), and it is not declared in
`BUILT_IN_EXTENSIONS` (which today holds exactly one entry, `llama.cpp`).

Why not a built-in tool, when the capability is this useful:

- A built-in loads for every session, including under `--no-extensions`, and is
  trusted package code. Two network tools plus five third-party endpoints is not
  a default everyone can pay for or audit.
- Its value is conditional on keys. With no key at all the ladder degenerates to
  scraping one search engine, which is the least defensible rung.
- Everything it returns is untrusted remote text. Keeping the capability in the
  untrusted-extension tier (opt-in, listed in `/extensions`, removable) keeps the
  trusted/untrusted line where the extension design put it.

Adopting it is therefore one copy:
`mkdir -p ~/.tau/extensions && cp examples/extensions/web_search.py ~/.tau/extensions/`
or `tau -e examples/extensions/web_search.py`.

## Testing

```bash
uv run pytest tests/test_web_search_extension.py -q   # 46 tests
uv run pytest tests/test_example_extensions.py -q     # shipped-example load
```

Nothing in the suite touches the network or a key. The fallthrough, snippet-floor,
quarantine, cache, alias, paging, and command tests drive the extension's own
`setup()` against a scripted transport, and the payload tests use recorded
provider bodies (both Firecrawl API shapes, both Perplexity citation shapes, a
DuckDuckGo redirect anchor, a challenge page). One test scans the extension's
source for environment names and fails if a new one is not in the harness's
managed list, so a key configured on the developer's machine cannot silently
change which backend a test expects. Registration through the real
`ExtensionRuntime` is covered by `test_example_extensions.py`.

Live behavior is checked by hand, because it spends quota and fails when a key
rotates:

```bash
uv run scripts/live_web_search.py            # every backend, then auto, then fetch
uv run scripts/live_web_search.py exa pplx   # one or two backends
```

It prints each backend's own output, the detected key **names**, the resolved
ladder, and the `/websearch status` block, and exits non-zero if any probe came
back empty.

## Limits and likely follow-ups

- DuckDuckGo answers a busy address with an "anomaly" challenge page; the note
  says the address is rate-limited rather than pretending the query was bad. A
  keyless second engine (or `lite.duckduckgo.com`) is the obvious follow-up.
- Reader mode has no JS. SPAs still need `format: "markdown"` with Firecrawl
  configured, which is a paid dependency; there is no local rendering path.
- `web_fetch` follows redirects with the shared client and has no per-host policy,
  proxy configuration, or size cap before decode — a very large non-HTML body is
  read into memory before the window is cut.
- The cache is per extension instance and process-local; two sessions in the same
  project pay separately. `session_start` deliberately clears it.
- Serper is first in the ladder because it is cheapest, but it is also the
  backend most often found dead in practice; the quarantine rule exists for that.
  Reordering the ladder, or making the order a setting, is a reasonable next step
  if more installs configure different keys.
- `prompt_guidelines` push the model toward search-then-fetch-then-cite. If that
  proves too eager (searching for facts it already has), the tuning knob is the
  snippet/guideline text, not the tool schema.
