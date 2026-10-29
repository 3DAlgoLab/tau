# The tau-rlm Launcher

The features on this track are still moving, and `tau` is the official install the
live sessions belong to. `tau-rlm` is a provisional second console script that runs
this checkout against its own home, so experimental work cannot corrupt durable
data it did not create.

## What it does

One module (`src/tau_coding/dev_cli.py`) and one `[project.scripts]` line:

```bash
uv run tau-rlm                  # TUI, from this checkout
uv run tau-rlm -p "..."         # print mode
```

Unless the caller set `TAU_HOME`, the launcher sets it to `TAU_RLM_HOME` or
`~/.tau-dev`. Sessions, settings, logs, and credentials for experimental work then
stay under that home, and `~/.tau/` -- including the sessions the official build
writes -- is never touched.

Priority order, and the reason for it:

| Variable | Effect |
| --- | --- |
| `TAU_HOME` | Wins outright. It is the knob Tau itself documents, so an explicit choice must survive. |
| `TAU_RLM_HOME` | Moves this track's state without repointing the whole environment for child processes that inherit `TAU_HOME`. |
| neither | `~/.tau-dev`. |

On an interactive start it prints which home it chose, and when that home has no
provider config it names the two ways in: `tau-rlm setup`, or copying
`providers.json` and `credentials.json` over from the live home. It does not copy
them by itself; duplicating secrets into another directory is not a launcher's
decision to make. The notice goes to stderr only when stderr is a terminal, so
pipes and CI logs stay clean.

## Why it is provisional

It is deletable on purpose: one module plus one script line. Deleting both removes
the entry point completely and leaves nothing that expected it.
`tests/test_dev_cli.py` pins the home rules, the refusal of a relative home, the
conditions under which the hint stays quiet, and the script registration itself --
which is the guard against this becoming load-bearing by accident.

The official command is unaffected: `tau` still resolves to the installed tool
environment rather than to this checkout.
