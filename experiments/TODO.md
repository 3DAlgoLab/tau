# Porting to Tau

Porting Pi's some essential features to Tau

## TODO List

- [x] Stall Guard Porting — done 2026-11-02, see "Stall Guard" below
  - original code: `~/.pi/agent/extensions/stall-guard.ts`

- [x] Persistent Python REPL Tool — done 2026-11-02, see "Persistent Python kernel" below
  - original code: `~/.pi/agent/git/github.com/3DAlgoLab/pi-pytool`
    *(path is stale as of 2026-11-03 — that cache directory no longer exists; the checkout is
    `~/dev-pi/pi-pytool`)*
  - It's just persistent python REPL keeping memory for session. 
  - It means NOT code-mode like replaciing whole tool-calls with REPL. 
    -> The port matches that: one `python` tool registered beside `read`/`bash`/`edit`/`write`,
       no tool-surface takeover, no prompt collapse.

- [ ] Subagents? Check first, if it is already supported..
  - Checked 2026-11-02: **not built in**, but already supported as an extension —
    `website/content/guides/extensions.md` points at
    [rian-dolphin/tau-subagents](https://github.com/rian-dolphin/tau-subagents)
    (an `agent` tool spawning in-process subagents, `.tau/agents/*.md` types, `/agents`,
    `get_subagent_result`, `steer_subagent`). No API gap found. Decide between vendoring it
    as a shipped example, or porting a native version.

- [ ] Stall Guard follow-up (optional fidelity): add a `session_start` handler that clears the session  budget, matching Pi's reset on `session_start`/`session_tree`. Currently only `agent_start` (per-run)  and `/stall-guard reset` (session) clear counters.

- [ ] Stall Guard drift — **the installed copy is ahead of the repo** (found 2026-11-03):
  `~/.tau/extensions/stall_guard.py` has 30 lines that `examples/extensions/stall_guard.py` does not
  (+30/−2), including the up-front exclusion `if message.stop_reason in {"error", "aborted", "length"}` and the
  doc paragraph explaining why a `length` stop is never retried. Port that back into `examples/` and
  cover it with a test, or the behaviour lives only in one machine's install.

- [ ] Heartbeat / `schedule_prompt` port — **installable and verified loading as of
  2026-11-02, still NOT official** (out-of-tree at `../tau-schedule-prompt`); see
  "Heartbeat / `schedule_prompt`" below
  - original code: `../pi-schedule-prompt` (`tintinweb/pi-schedule-prompt`, v0.4.1)
  - the port lives in **`../tau-schedule-prompt`** — a separate sibling project, outside this repo.
    See "Heartbeat / `schedule_prompt` (unofficial, out-of-tree)" below.
  - Do **not** treat it as a Tau `examples/` extension: it is not referenced by the website docs and
    not in any release path. *(Corrected 2026-11-03: the earlier "not installed … its Tau entry point
    does not exist yet" is stale — `extension.py` exists and it is installed at
    `~/.tau/extensions/tau-schedule-prompt`; see "Gap closed" and "Install path" below.)*

Pi's features:
- [x] Set Last model used as a starting model for a new session — done 2026-11-02,
  see "Last model as the starting model" below
- [x] Implement slash command `/copy`: copy the last response of coding agent to the
  clipboard — done 2026-11-02, see "`/copy`" below
- [ ] Web search tool — **built and running, but not in this repo** (checked 2026-11-03): the only
  copy is `~/.tau/extensions/web_search.py` (1,448 lines, edited 2026-11-03 00:16), with out-of-tree
  tests at `~/.tau/tests/test_web_search.py` (677 lines) and a live probe `live_web_search.py`.
  Nothing in `examples/`, `dev-notes/`, `website/`, or `tests/` tracks it. See
  "Web search (live, untracked)" below.
  - No Pi original is cited for it: `~/.pi/agent/extensions/` holds only `stall-guard.ts`, and
    `~/.pi/agent/git/` does not exist, so until an upstream file is named this is a Tau-original
    design, not a port.

## Stall Guard (done)

- Extension: `examples/extensions/stall_guard.py`
- Tests: `tests/test_stall_guard_extension.py` (18: 10 end-to-end `CodingSession` + `FakeProvider`, 8 runtime-level)
- Docs: `website/content/guides/extensions.md` (example list + inline-dispatch note; also fixed the stale
  `context.api.notify(...)` example -> `tau.notify(...)`), `dev-notes/stall-guard-extension.md`
- Installed: `tau install ./examples/extensions/stall_guard.py` -> `~/.tau/extensions/stall_guard.py` (copied, not symlinked)

### Where Pi and Tau diverge

- Pi rescues at `agent_before_settle` (continue/resolve decision). Tau has no such hook;  `agent_settled` is observation-only and fires after teardown.
- Tau equivalent seam = the `turn_end` of a turn whose assistant message requested no tool call.
  `AgentHarness._run` awaits event listeners *before* `run_agent_loop` drains the follow-up queue,
  so `tau.send_custom_message(..., deliver_as="follow_up")` from a `turn_end` handler runs in the
  same model run (no TUI callback needed -> works in TUI, print mode, and tests).
- Budgets: `MAX_PER_RUN = 2`, `MAX_PER_SESSION = 8`, warn once when exhausted.
  Per-run budget resets on every `agent_start`; the session budget only resets via `/stall-guard reset`.
  (Pi also resets on `before_agent_start` / `session_start` / `session_tree`; the Tau port does **not**  subscribe to `session_start`, so `/new`, `/fork`, `/switch` keep the session counter — see TODO below.)
- Tau-specific: injection is a `custom_message` with custom_type="stall-guard:retry"` plus a  registered renderer (compact transcript line, full text when expanded); `/stall-guard` reports the budget.
- Pi reads `ctx.sessionManager.getPendingMessages()`; Tau exposes no queue read, so pending count is reconstructed from `queue_update` plus the `message_start` that drains it.

### Checks

- `uv run ruff check .` / `ruff format --check .` / `uv run mypy` — clean
- `uv run pytest -q` — 1685 passed, 1 failed, 1 skipped at the time of the landing.
- Re-run 2026-11-03 at `666257c` (with the fork's later commits and the uncommitted compaction work):
  **2218 passed, 1 failed, 3 skipped, 217s**. Same single failure, and it is **not** a flake:
  `tests/test_tui_app.py::test_tui_tree_labels_filter_timestamps_and_clear` asserts the UTC calendar
  date `2023-11-14` while this box runs `KST+0900`, so it renders `2023-11-15 07:13`. It passes under
  `TZ=UTC`, and the assertion already exists on `origin/main`. Two consequences: run the suite with
  `TZ=UTC` before opening any PR, and consider upstreaming a timezone-proof fixture.
- Installed-load probe against packaged `tau` 0.4.8: extension `stall_guard` (source `user`),
  4 handlers, command `stall-guard`, renderer `stall-guard:retry`, no error diagnostics.

### Drift (2026-11-03)

`diff` of `examples/extensions/stall_guard.py` against `~/.tau/extensions/stall_guard.py`: **+30 lines
installed-only** (the `stop_reason in {"error", "aborted", "length"}` exclusion and its rationale).
The repo is behind the live extension — see the TODO item above.

## Persistent Python kernel (done)

- Extension: `examples/extensions/python_kernel.py` (one file; the kernel bridge is an embedded
  stdlib source string started as `<interpreter> -u -c <src>`, so install stays one copy)
- Tests: `tests/test_python_kernel_extension.py` (16: config/caps/renderers without a kernel,
  real-interpreter persistence + deadline + reap + UTF-8, plus `CodingSession` wiring with `FakeProvider`)
- Docs: `website/content/guides/extensions.md` (example list), `dev-notes/python-kernel-extension.md`
- Registered: tool `python` (`execution_mode="sequential"`), command `/python [reset]`,
  `render_call`/`render_result`, four prompt guidelines
- Installed: yes — `~/.tau/extensions/python_kernel.py` (it adds a tool to every session, so
  the install was the user's call; it is now the live copy that `examples/` tracks).

### Project-venv follow-up (committed as `81ff167`)

The installed copy grew a project-venv rung; `examples/extensions/python_kernel.py` was
synced from `~/.tau/extensions/python_kernel.py` (+353/−28), and that sync is now committed as
`81ff167 Resolve the python kernel interpreter to the project venv`.

They are **no longer byte-identical** (checked 2026-11-03), but the entire delta is formatting: two
`choose_interpreter` return statements re-wrapped by `ruff format` in `examples/` (8 lines there, 4
installed). Same semantics, so the live copy is not running older logic; re-installing it would be
cosmetic.

- Behavior: interpreter ladder is now `pin → project .venv → PATH → sys.executable`.
  The `.venv` walk is bounded by the git root, else `$HOME`. `TAU_PY_VENV=prefer|require|off`,
  `TAU_PY_UV` pins uv, `/python [reset|venv [--force]]`, cell `details.venvState`,
  kernel restarts with a "kernel moved …" note when resolution changes mid-session.
- Docs: done — dev-note got an "Interpreter and environment ladder" section (bounded walk,
  modes, uv provisioning plan, per-cell re-resolution), the guide bullet names the
  project-`.venv` rule and `TAU_PY_VENV`, and `dev-notes/README.md`'s index line was updated.
- Tests: **done** — `tests/test_python_kernel_extension.py` now drives
  `choose_interpreter(env, cwd)` / `settings_from_env(env, cwd)` and holds the ported
  venv cases (ancestor walk + git/`$HOME` bounds, pin-beats-venv, `off`/`require`/broken
  reports, uv provision plan, real `uv` provision + reuse, `VenvReport.render`):
  `uv run pytest tests/test_python_kernel_extension.py -q` → **33 passed**.
  `~/.tau/tests/test_python_kernel_venv.py` (standalone `unittest`, hardcoded path) is
  now redundant and can be dropped.
- Commit boundary: `abdd490` (stall guard), `a6dc016` (python kernel) and `81ff167` (the venv rung:
  extension + tests + docs, one commit) are all on local `main`, none pushed. The earlier "not
  committed yet" and "blocked on the test port" notes are stale.

### Deliberate deviations from pi-pytool

- **No `ipykernel`/`jupyter_client`.** Pi drives a Jupyter kernel and interrupts over the ZMQ control
  channel. Here the bridge is pure stdlib over one persistent namespace dict and the host sends SIGINT:
  same guarantee (bytecode boundary), same failure mode (a cell parked inside C cannot be cut), and no
  managed-venv build at install time.
- **Interrupt ladder:** deadline -> SIGINT -> `TAU_PY_ABORT_GRACE_MS` -> terminate. A cell that yields
  returns `status: timeout` with partial output and its names intact; one that will not is `reaped`, and
  the next cell boots empty and leads with `<tau_python_kernel_reset>`.
- **Kept from Pi:** caps with visible cut markers (46080/cell, 4096/line), the UTF-8 transport pin
  (cp949 lesson), no snapshots, `/python` diagnostics naming interpreter + rule + pids + caps + names.
- **Left out:** warm boot at `session_start` (boots on the first cell, ~30 ms), streamed output
  (`on_update` is the seam), Windows tree kill (single child, so `terminate()` is complete), the
  import-hint, and the `bash`-pushback hook.

### Checks

- `ruff check` + `ruff format --check` on both new files, `uv run mypy examples/extensions/python_kernel.py`,
  `pyright` on both files — all clean, 0 diagnostics
- `uv run pytest tests/test_python_kernel_extension.py tests/test_example_extensions.py -q` — 32 passed

## Last model as the starting model (done)

- Code: `ProviderSettings.last_model` + `save_model_choice()` / `set_last_provider_model()`
  (`src/tau_coding/provider_config.py`), `CodingSession._carried_session_selection()`
  and `_persist_model_choice()` (`src/tau_coding/session.py`)
- Tests: `tests/test_provider_config.py` (memory precedence, provider-scoped narrowing,
  unusable memory, round-trip, junk tolerance, account-scoped pair) and
  `tests/test_coding_session.py` (`/new` carries the running pair, `/new` fallback when
  durable settings cannot name it, a switch is written to preferences)
- Docs: `website/content/guides/providers-and-models.md` ("Choosing and switching models"),
  `website/content/reference/slash-commands.md` (`/new` row), `dev-notes/last-model-starting-model.md`
- Key choice: `last_model` is a **separate `providers.json` preference**, not a
  `catalog.toml` `default_model`. The remembered pair may be account-scoped (Codex/Claude
  subscription models never enter the catalog), so writing it into the definition file would
  put a machine-specific id in a shared definition. Memory is skipped, never fatal, when this
  install cannot name it.

## `/copy` (done)

- Code: `/copy` + `CommandResult.copy_requested/copy_text` (`src/tau_coding/commands.py`),
  `CodingSession.last_assistant_text`, `src/tau_coding/clipboard.py` (tool backends, no OSC 52),
  TUI handler (`tui/app.py`, Textual chain incl. OSC 52), print-mode handler (`cli.py`)
- Tests: `tests/test_clipboard.py` (8: pyperclip preference/fallthrough, tool order, refusing
  tool, no-tool message, "text is data not a command line"), `tests/test_commands.py` (3),
  `tests/test_cli.py` (print mode end-to-end), `tests/test_tui_app.py` (app clipboard chain),
  `tests/test_coding_session.py` (`last_assistant_text` skips prose-less turns)
- Docs: `website/content/reference/slash-commands.md` (row + note),
  `website/content/guides/tui.md`, `dev-notes/copy-last-response-command.md`
- Key choice: the core handler decides **what** to copy and the frontend decides **how** —
  the same `export_requested`-style split. Textual owns OSC 52 in the TUI; the core helper only
  shells to `wl-copy`/`pbcopy`/`xclip`/`xsel`/`clip` with a fixed argv and stdin, never a shell.

## Web search (live, untracked)

**Status: works in this install, absent from the repository.** `~/.tau/extensions/web_search.py`
(1,448 lines) is the only source; `examples/extensions/` has no such file, `dev-notes/` has no
journal, `website/content/` never mentions it, and `uv run pytest` cannot see it. A fresh clone loses
it entirely.

- Registered: tools `web_search` (ranked results, optional synthesized answer) and `web_fetch`
  (readable page text, paged by `offset`), command `/websearch`, a `render_call` for each, and prompt
  guidance telling the model when to reach for them.
- Backend ladder, first one that returns results: `serper`, `exa`, `perplexity`, `firecrawl`,
  `duckduckgo` (`PROVIDERS`, line 77). Keys via `SERPER_API_KEY`/`SERPER_DEV_API_KEY`, `EXA_API_KEY`,
  `PERPLEXITY_API_KEY`/`PPLX_API_KEY`; `duckduckgo` is keyless and gated by
  `TAU_WEB_SEARCH_ALLOW_KEYLESS`; `TAU_WEB_SEARCH_PROVIDER` pins one backend.
- Knobs: `TAU_WEB_SEARCH_MAX_RESULTS`, `TAU_WEB_SEARCH_SNIPPET_CHARS`, `TAU_WEB_FETCH_MAX_CHARS`,
  `TAU_WEB_SEARCH_TIMEOUT_SECONDS`, `TAU_WEB_SEARCH_PERPLEXITY_MODEL`, `TAU_WEB_SEARCH_CACHE_SECONDS`,
  `TAU_WEB_SEARCH_DEAD_TTL_SECONDS`.
- No new wheels: it rides on the `httpx[socks]` dependency `tau_ai` already ships and parses payloads
  with the standard library — same rule the python-kernel port holds.

### What is missing before it counts as landed

1. Move the source to `examples/extensions/web_search.py` (single file, like `python_kernel.py`).
2. Port `~/.tau/tests/test_web_search.py` to `tests/test_web_search_extension.py` and add the
   `tests/test_example_extensions.py` discovery case, so `uv run pytest` guards it. Keep
   `live_web_search.py` as a manual probe, not a test.
3. Docs: `dev-notes/web-search-extension.md`, the `website/content/guides/extensions.md` example list,
   and the `dev-notes/README.md` index line.
4. Decide the shipping posture: it adds two tools to **every** session, as `python_kernel.py` does.
5. Reconcile with Tau's own stance on fetching: state in the dev-note why an extension rather than a
   built-in tool, so the roadmap issue does not have to be re-litigated.

## Heartbeat / `schedule_prompt` (unofficial, out-of-tree)

**Status note: this port is an experiment, not part of Tau.** It lives in the sibling directory
`../tau-schedule-prompt` (own `pyproject.toml`, own `.venv`, dependency `tau-ai = { path = "../tau",
editable = true }`). It is not under `examples/` and not documented under `website/content/`. As of
2026-11-02 it **is** installable and does load under the real extension runtime, but it stays
unofficial/out-of-tree: nothing in this repo depends on it, and Pi's interactive UI flows are still
unported. Only promote it into `examples/extensions/` (or publish it) after the finish order below.

### What exists (src 2894 lines at audit time, plus tests)

- `src/tau_schedule_prompt/`: `cron.py` (hand-ported cron parser, replaces Pi's `croner` dep),
  `scheduler.py` (validation + one asyncio wake-up loop instead of Pi's per-job timers), `tool.py`
  (`schedule_prompt` tool, actions `add|list|remove|enable|disable|update|cleanup`), `subagent.py`
  (background child-session runs via Tau's child-session seam), `widget.py` (below-prompt job table
  as Rich-markup lines, no Textual import), `types.py`, `storage.py`, `settings.py`, `timeutils.py`,
  `rendering.py`, and now `extension.py`
- `tests/`: `test_cron.py`, `test_scheduler.py`, `test_storage_settings.py`, `test_extension.py`
- Parity vs Pi: `scheduler`/`settings`/`storage`/`types`/`tool`/`subagent`/`cron-widget`/`index`
  written; Pi's `ui/add-flow.ts` (107) and `ui/jobs-view.ts` (249) have **no port yet**.

### Gap closed: the entry point now exists

`pyproject.toml` declares the Tau entry point:

```toml
[tool.tau]
extensions = ["src/tau_schedule_prompt/extension.py"]
```

That file now **exists** and is the port of Pi's `index.ts`: it loads layered settings, builds
per-scope `JobStorage` + `Scheduler`, registers the `schedule_prompt` tool through
`create_schedule_tool(ToolDeps(...))` (the `ToolDeps` callables resolve the host context lazily),
registers the `scheduled_prompt` message renderer, registers `/schedule-prompt`
(list / pause / resume / remove / cleanup / show / hide), mounts the below-prompt widget, and
starts/stops the scheduler on `session_start` / `session_shutdown`. `src/tau_schedule_prompt/py.typed`
was added, which unblocked `uv run mypy`.

The host seams it targets all exist in Tau, verified in this repo: `register_tool`,
`register_command`, `register_message_renderer`, `send_custom_message`
(`src/tau_coding/extensions/api.py`, `runtime.py`) and `set_slot_widget` (`api.py` +
`tui/app.py`, `placement="below_prompt"`).

### Install path that works

```bash
cd /home/j2y/dev-pi/tau
tau install ../tau-schedule-prompt        # -> ~/.tau/extensions/tau-schedule-prompt/
tau install --force ../tau-schedule-prompt  # re-install after edits
```

The installer honours `[tool.tau] extensions = [...]`, so the package directory is copied whole and
the declared entry file is what the loader imports; package-relative imports inside
`tau_schedule_prompt` still resolve because the entry stays inside its own package directory. Re-run
with `--force` after every edit — a second plain `tau install` refuses with "extension is already
installed … pass --force", and the installed copy is independent of the source tree.

Side note for this repo: `tau install` also copies `tests/`, `uv.lock` and `.ruff_cache/`
(`_local_copy_ignore` in `src/tau_coding/extension_installer.py` does not exclude them). Harmless,
but a candidate cleanup.

### Checks (rerun 2026-11-02, in `../tau-schedule-prompt`)

- `uv run pytest -q` — **66 passed** (51 before; 15 new in `tests/test_extension.py`)
- `uv run ruff check .` — **All checks passed**
- `uv run ruff format --check .` — clean (18 files)
- `uv run mypy` (strict) — **no issues in 12 source files**
- `tau install ../tau-schedule-prompt` — installs to `~/.tau/extensions/tau-schedule-prompt`
- Installed-load probe (`~/.local/share/uv/tools/tau-ai/bin/python /tmp/probe_schedule_prompt.py`,
  packaged Tau 0.4.9b2, discovering only the real `~/.tau/extensions`):
  - **0 error diagnostics**; tools `python`, `schedule_prompt`; commands `python`, `schedule-prompt`,
    `stall-guard`; `render_custom_message("scheduled_prompt", …)` returns a formatted marker; and
    `render_tool_call` / `render_tool_result` return markup
  - `add` / `list` wrote `<cwd>/.tau/schedule-prompts.json` with session scope
  - a `* * * * * *` cron job added after `emit_session_start` fired and reached the bound session as
    a `scheduled_prompt` custom message (`mode: inline`), with `run_count` advancing once per second
    — no double-fire
  - `emit_session_shutdown` stopped the loop cleanly (no pending-task warnings)
- TUI widget probe (a temporary file in this repo's `tests/`, since removed): the port's real
  `ScheduleWidget` mounted into `#below-prompt-slot` on `TauTuiApp` through the host component bridge,
  painted `⏰ Scheduled 1 job` plus the job row, and unmounted on `hide()`

### Deliberate deviations from Pi (so reviewers do not re-litigate them)

- **Inline delivery is one message.** Pi fires an empty-content visible marker plus a separate
  `sendUserMessage(job.prompt)`. The port delivers once via
  `send_custom_message(job.prompt, custom_type="scheduled_prompt", details=…)`, so the prompt enters
  model context exactly once and the renderer gets its structure from `details`. The route is `steer`
  while the session is running, `follow_up` otherwise.
- **Widget content is Rich-markup strings**, not a Textual widget, so the extension never imports
  Textual and degrades to widget-less (still functional) in print mode.
- **The scheduler repaints on every wake**, not only on state change, because the sleep is capped at
  `MAX_TICK_SECONDS`; that keeps the widget's "in 5m" labels honest.
- **Cron fields are matched in host-local naive time** (`to_local_naive`), as in Pi/Vixie — so
  `0 0 9 * * *` means 09:00 *local*, not 09:00 UTC. Tests asserting relative labels must not assume
  UTC.
- Model-pinned jobs run in a **child session** via Tau's child-session seam, forwarding parent trust
  (`trust_override`) and using core tools unless the job opts into extensions.

### Compatibility with released Tau (audited 2026-11-03)

The port's own suite plus an import probe, run against published `tau-ai` wheels (PyPI goes 0.1.0 …
0.4.8; `0.4.9b2` is local-only):

| `tau-ai` | result |
| --- | --- |
| 0.1.5 | `ModuleNotFoundError: No module named 'tau_coding.extensions'` — the extension system landed `ab5aa72` (#320, 2026-07-15) but shipped later |
| 0.1.6 | `ImportError: cannot import name 'ImageContent' from 'tau_agent.messages'` |
| 0.2.4, 0.3.13 | 66 passed |
| 0.4.0, 0.4.3, 0.4.5, 0.4.7, 0.4.8 | 66 passed |
| local 0.4.9b2 | 66 passed |

What the suite does **not** catch (its fixtures fake `ExtensionContext`):

- `extension.py:_snapshot()` reads `context.project_trusted` and `context.available_model_choices`
  unguarded; both first shipped in **0.4.8** (`637aaa2` #736 and `2454169`), and `context.paths` only
  since 0.4.3. `_snapshot()` is reached only from `_fire_subagent`, so inline jobs survive on older
  Tau but a `model=` job raises `AttributeError` below 0.4.8.
- `pyproject.toml` declares `tau-ai>=0.4.0`; the honest floor is **0.4.8** for background jobs.
- Strict mypy is version-sensitive: 2 errors against released 0.4.8 (`Argument 2 to "on" of
  "ExtensionAPI" …`) and 7 against 0.4.0. It is clean only against this checkout, because of the
  unpushed `9d57ba7` ("Type the `tau.on` decorator form …"). Nothing at *runtime* needs the unpushed
  commits.
- The job store is **not** Pi-compatible: `.tau/` vs `.pi/`, and camelCase vs snake_case (`intervalMs`,
  `createdAt`, `lastRun`, `lastStatus`, `nextRun`, `runCount` → `interval_ms`, `created_at`,
  `last_run`, `last_status`, `next_run`, `run_count`), plus an extra
  `last_status="pending"`. Pointed at a Pi-written store it loads happily and then drops the run
  history (`created_at: ""`, `run_count: 0`) on the first rewrite; the reverse direction feeds Pi
  `undefined` fields, since Pi casts `JSON.parse` with no validation.

### Order to finish it

1. ~~`py.typed`, lint/format/mypy, `extension.py`~~ — done, gates green.
2. ~~Port Pi's tests for `tool.ts`, `subagent.ts`, `cron-widget.ts`~~ — covered by
   `tests/test_extension.py` (registration, lifecycle, widget mount/clear, inline vs subagent fire,
   recursion guard, command paths). `tool.py` argument-validation edge cases are still thin.
3. Slash-command UI parity: Pi's `ui/add-flow.ts` (107) and `ui/jobs-view.ts` (249) have no port; the
   current `/schedule-prompt` is text-output only (no `open_main_view` picker).
4. In that repo: `git init`, `.gitignore`, CHANGELOG, CI, and its own `dev-notes/` journal — the
   README now documents install/use/limits (**not** a `WIP` placeholder any more) — **then** decide
   whether it becomes an official `examples/extensions` entry or a standalone `tau install` package.
5. If promoted: re-run the installed-load probe against the packaged Tau version being released, and
   add a repo-side test that loads it the way `tests/test_example_extensions.py` loads the examples.
6. Raise the pin to `tau-ai>=0.4.8` **or** guard `_snapshot()` (`getattr` with a documented fallback),
   so the manifest and the runtime agree about background jobs.
7. Add one test that builds `ParentSnapshot` from a real `ExtensionContext` instead of a fake — the
   fixture gap is what let the 0.4.8 dependency go unnoticed.
8. Decide the Pi-store question: accept camelCase keys on read (alias map in `job_from_json`), or
   state in the README that the store is Tau-only. No migration exists in either direction today.

## Memo

Tau upgrade tau
- Install: `uv tool install --force .` or `uv tool install --editable --force .`
  - The later one affects real code.

Extension install notes
- `tau install <path>` copies into `~/.tau/extensions/`; editing `examples/extensions/stall_guard.py`
  does **not** update the installed copy — re-run `tau install` (or `ln -sf`) after each edit.
- Extension identity is the module stem and collisions are first-loaded-wins; built-ins load before  user extensions, so a user extension sharing a built-in's stem is silently skipped  (`tau doctor` reports it as a duplicate extension). Keep the stem unique.
- Nothing is committed for the venv follow-up any more: `81ff167` landed it. `abdd490` (stall guard),
  `a6dc016`/`81ff167` (python kernel) are all local, unpushed commits.

Publishing state (verified 2026-11-03)
- `origin` is `github.com/3DAlgoLab/tau`, a fork of `huggingface/tau`. `main` is **10 commits ahead,
  nothing pushed** (`9d57ba7` … `666257c`), plus 7 uncommitted files (`+338/−52`).
- Guideline audit against `CONTRIBUTING.md`: every local commit carries tests + a `dev-notes/` note +
  a `website/content/` update, `tau_agent` gained no CLI/TUI/path imports, and
  `ruff check` / `ruff format --check` / `mypy` are clean. Roadmap alignment is unverifiable while the
  roadmap issue lives upstream at `huggingface/tau#1`.
- `666257c chore: release 0.4.9b2` bumps `pyproject.toml`, which the release workflow publishes on —
  CONTRIBUTING says that bump must reach `main` through a PR, so keep it out of a feature push.
- Work in progress, uncommitted: request-boundary compaction and repeated overflow recovery
  (`src/tau_agent/harness.py`, `src/tau_agent/loop.py`, `src/tau_coding/session.py`, two test files,
  `website/content/guides/context.md`), journaled in `dev-notes/context-overflow-recovery.md`.

Which copy is live (verified 2026-11-03)
- The **installed** copy is what runs. `~/.tau/extensions/tau-schedule-prompt` holds `cpython-313`
  bytecode compiled by the packaged Tau (0.4.9b2), while the sibling source tree's own `.venv` is 3.12
  and only used to run its tests. Nothing in this repo points at the source tree.
- Check drift before trusting either side:
  `diff -rq --exclude=__pycache__ --exclude=.venv --exclude=.ruff_cache ~/.tau/extensions/tau-schedule-prompt ../tau-schedule-prompt`
  — today they differ only in `README.md`, so the installed tree is one docs edit behind.
- Reload rule: `tau install --force` replaces the copy, but a running session keeps the old one until
  restart.
