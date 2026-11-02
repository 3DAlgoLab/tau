# `/copy`: copying the last response

## What was added

- `/copy` in the built-in slash command registry (`src/tau_coding/commands.py`).
- `CommandResult.copy_requested` / `copy_text`.
- `CodingSession.last_assistant_text` — the newest assistant prose.
- `src/tau_coding/clipboard.py` — a frontend-independent clipboard writer used by
  print mode.
- TUI handling in `src/tau_coding/tui/app.py`, which copies through Textual.

## Why it exists

Pi has `/copy`. The interesting part is not the keypress; it is that "the clipboard"
means different things per frontend. Textual knows whether this terminal answers
OSC 52 (the only way a copy can reach your local machine over SSH). Print mode has
no terminal protocol to lean on, only whatever clipboard binary the system offers.
A single implementation would get one of those wrong.

So the command answers *what* and the frontend answers *how*, which is the same
split `/export`, `/local`, and `/sidebar` already use: the core handler parses
intent and returns a flag; the host performs the action it is equipped to perform.

Decisions worth keeping:

- **What gets copied** is the most recent assistant *prose*. A turn that ended in
  tool calls has no prose, so the property skips it and returns the previous
  answer rather than an empty string, and `/copy` says "Nothing to copy yet" when
  the session has no answer at all. Copying a tool call's JSON, or silently
  copying nothing, are both worse.
- **No escape sequences from the core helper.** Writing OSC 52 from underneath a
  frontend that owns the screen corrupts that render. Textual emits its own.
- **Clipboard text is never a command line.** The helper passes a fixed argv with
  no shell and feeds the text on stdin, so hostile content cannot become an
  argument.
- **A missing clipboard is a message, not a traceback.** The helper tries each
  candidate (`wl-copy`, `pbcopy`, `clip.exe`, `xclip`, `xsel`, `clip`), tolerates
  a broken or timing-out one, and reports what to install when none exist.

## How it maps to Pi

Same command name and same user intent as Pi's `/copy`. Tau keeps the handler in
the provider-neutral command registry so the TUI, print mode, and any future
frontend share one definition of "the last response".

## How to test

```bash
uv run pytest tests/test_clipboard.py tests/test_commands.py tests/test_cli.py -q
uv run pytest tests/test_tui_app.py -q -k copy_command
uv run pytest tests/test_coding_session.py -q -k last_assistant_text
```

Look for:

- `test_copy_command_requests_a_frontend_copy`
- `test_copy_command_says_when_there_is_nothing_to_copy`
- `test_clipboard_text_is_data_and_never_a_command_line`
- `test_a_refusing_tool_moves_to_the_next_candidate`
- `test_run_print_mode_copy_command_uses_the_clipboard_helper`
- `test_tui_copy_command_writes_through_the_apps_clipboard_chain`
