---
title: "Package installs inside python_repl (Phase 2d)"
date: "2026-10-30"
---





# Package installs inside `python_repl`


A kernel that can hold a 5 MB list is useless if the model cannot `import pandas`
to make it. Today `python_repl` takes `install: ["numpy", "scipy==1.15"]` beside
`code`, and the packages are present when the cell runs. The environment belongs to
the kernel, not to the model.


## Why a list of names and not a command


The tempting version is a `pip install` the model writes through `bash`. It is
rejected for the same reason the kernel has no shell: installing is a protocol
question, not a scripting one. Which resolver exists on this machine, which
interpreter it must resolve for, which directory the result has to land in, and how
long the whole thing may take are all answers Tau already knows and the model has
no way to check. Handing the command over means the model can also pick the wrong
one -- a `uv pip install --target` aimed at a directory the kernel never searches
fails as a mystery import error three steps later.

So the tool argument is a list of package specs. `validate_specs` keeps each entry
shaped like a name: a spec is passed as one argv element, never through a shell, and
anything that could read as an option of the resolver itself (`--target=…`, a leading
dash, an embedded NUL, an over-long string) is refused before a process starts.


## Durability is the part a namespace cannot offer


A kernel restart discards every bound name, and a session that had to reinstall
numpy on each cold start would not be worth the namespace it keeps. The environment
therefore lives on disk at `~/.tau/kernel-site/<project>/<implementation>-<version>/`
-- `tau-dev-a2653e9e/cpython-3.12` for this checkout. It survives a killed cell, a
killed kernel, compaction, and the session.

The project sits between the base and the ABI because a machine-wide environment
makes every session's packages everyone's problem: it grows without bound, and a
version one project pinned becomes a constraint another project inherited without
being asked. `project_root` takes the repository root when the session starts
inside one -- starting Tau in a subdirectory is how a person navigates, not a
decision to build a second numpy -- and the starting directory otherwise. The
directory name is a sanitised label plus eight hex digits of the root's path, so
two checkouts sharing a basename still land apart and `ls` still says which is
whose.

The interpreter is part of the directory name because a compiled wheel is not
portable across ABIs. `probe_interpreter` asks the child (`sys.implementation.name`
plus `sys.version_info[:2]`) rather than assuming the host's own version, and the
kernel refuses to start if the configured executable cannot answer that question.
Switching `codeModePython` in `~/.tau/settings.json` therefore does not hand a
different ABI an environment it cannot import; it gets its own directory, empty
until something is installed for it.


## Growth, and who prunes it

An environment that can only grow needs a ceiling, and the ceiling belongs to the
tool rather than to the model. `PackageInstaller` measures the target with
`site_usage_bytes` -- a walk that stops the moment the running total passes the
budget, because a budget check is not an inventory and an oversized environment is
exactly the case where counting every file costs the most -- and refuses before any
resolver spawns: 2 GiB per project per interpreter (`DEFAULT_SITE_LIMIT_BYTES`).
The refusal names the directory and says it is a cache the user owns.

There is no uninstall, and adding one is the wrong move rather than an unfinished
one. An install's blast radius is one call and its verification is the cell that
imports the package; an uninstall's blast radius is every session that has ever
shared the environment and its failure shows up in a *later* session as a
`ModuleNotFoundError` whose cause sits outside the transcript. The model is allowed
to add, never to remove, and the honest door is a user-facing one (`tau kernels
ls|prune`, still to build) plus the fact that the whole thing is a directory.

## Resolver chain, one budget


`uv pip install --python <executable> --target <dir>` first when `uv` is on the
machine (0.2 s for a small package here), the interpreter's own `python -m pip
install --target <dir>` if uv exits non-zero or is absent. `shutil.which` normalises
even an explicitly configured path, so a resolver that is not installed is skipped
before it is spawned rather than burned as a failed attempt.

One timeout covers the whole install, shared across resolvers, because the caller
named a wait and a second resolver after spending it would silently double how long
a tool call can block. Output is capped at 8 KiB by keeping the **tail** (`MAX_INSTALL_OUTPUT_BYTES`) --
`uv`'s progress bars are noise and the diagnosis is the last thing a resolver
prints, so the head is what gets dropped. When
every resolver fails, the install returns a report naming each resolver it tried and
quoting what it said: the model needs to know which one to blame and which spec to
fix, and `code_mode` turns that into an input refusal so the cell never runs against
an import that is not there.

Tau never installs without being asked, and nothing in the session installs by
default.


## Files and tests


`kernel/packages.py` is mechanism: identity probe, spec validation, resolver chain,
bounded execution. `code_mode.py` is policy: it starts the kernel first (the kernel
knows its own site directory), installs into it, refuses bad input before any
process spawns, and prepends one note line so the path the cell imports from is
visible in the result rather than guessed.

`tests/test_kernel_packages.py` (27) pins the identity mapping and its stability, an
executable that is not Python (`/bin/true` runs, exits 0, and says nothing about
itself), the spec shapes a model actually sends and the ones that must be refused,
uv preferred with pip never reached, pip catching a failing uv, an absent resolver
skipped before it runs, every-resolver-failing as a report instead of an exception,
and a resolver outstaying its budget being killed with no second attempt behind it.
The project keying has its own pins: repository root winning over the starting
directory, two same-named checkouts landing apart, a label with punctuation staying
filesystem-safe, the usage walk bailing once the budget is passed, and an
over-budget install refusing with no resolver spawned and the directory untouched.
`tests/test_kernel_packages.py::test_the_configured_base_is_scoped_to_the_session_project`
holds the wiring, since the base the session hands the kernel is no longer the
configured base.
Fake executables write a marker where they were told to install, so no test can
reach a package index. `tests/test_kernel_process.py` adds the child-side half: a
configured interpreter really being the one that runs, kernels on one interpreter
sharing one directory, and a package dropped into the site directory *after* the
kernel started still being importable -- the guarantee every resolver has to deliver.
`tests/test_code_mode.py` adds the tool half: install landing before the importing
cell, an installed package outliving the kernel that installed it, and a failed
install leaving no side effects behind. `tests/test_shell_config.py` covers
`codeModePython`.


## Left alone


No lockfile, no resolution pinning, no offline mirror, and no way to take packages
back out from inside a session. Resolution correctness is uv's and pip's business;
what Tau owns is that the answer arrives in a directory a given kernel searches,
inside a bounded wait, without the model choosing any of it.

Still unbuilt, now that environments are per-project and capped: a manifest naming
what each site directory holds and when it was last used, and the `tau kernels
ls|prune|rebuild` CLI that reads it. Without those, the budget stops a single
environment becoming absurd and nothing stops a hundred projects each keeping one.
Old flat `~/.tau/kernel-site/<implementation>-<version>/` directories from before
project scoping are simply not read any more -- delete them by hand.
