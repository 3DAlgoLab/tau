---
title: "Deterministic Compaction QA Probes"
---

Phase 1c of porting pi-fabric's deterministic compaction. It adds the check that
tells you a summary is *silently wrong*, which is the only way a compaction
engine can be trusted at all.

```text
src/tau_coding/compaction/qa.py   ground-truth probes over one compaction window
```

## Why probes, and why from ground truth

A compaction that produces garbage is obvious. A compaction that produces a
shorter summary missing one unresolved error is not obvious until the model
re-runs a command that already failed. So QA generates reconstruction questions
from the **normalized events the cut removed** and checks the rendered text for
the answers.

Reading the projections instead would grade the summary against itself: a fold
that forgets a file would also forget to probe for it. `generate_probes()` never
touches `Sections` or the rendered string, so a fold regression shows up as a
failed probe rather than as a shorter summary.

## Two classes, both mandatory

| class | what it proves | example answer |
| --- | --- | --- |
| `content` | a fact survived | `bash: uv run mypy: 9 errors found` |
| `address` | something that did *not* survive can still be found | `… omitted 6 file addresses; source entries w12 → w17` |

Dropping content is allowed; dropping content and its way back is not. A summary
that keeps every fact but loses the footer scores below 1.0 on `address_score`,
and a summary that keeps the pointers but loses an error scores below 1.0 on
`content_score`. `QaReport` reports the two separately so a regression is
attributable, and `qa_report(...).ok` is the gate.

Probe inventory: protected dialogue excerpts, the earliest historical request,
every sampled modified path, the omission markers for files/errors/turns, each
unresolved failure by identity, the last targeted path, earlier-turn one-liners
plus pointers for turns outside the transcript window, the newest `(#N)`
transcript pointer, and the expansion footer.

## Shared identity, not duplicated logic

QA has to agree with the folds about what counts as an unresolved error and how a
marker reads, otherwise a green report means nothing. Instead of reimplementing,
the folds expose the primitives and QA consumes them:

```python
identity_of(operation)        # file+tool+path, bash+tool+command, else canonical args
unresolved_operations(events) # failures no later success answered
error_subject(op) / error_signature(op)
iter_turns(events) -> Turn    # the same turn stream the earlier-turn fold samples
NOUN_FILES, NOUN_UNRESOLVED, ...  # omission marker phrasing, defined once
```

`[RESOLVED]` is decided by identity, never by error text, and QA asks the same
question the fold answers: did a later success share this identity? That is why a
retry of `uv run mypy` closes the earlier failure while an edit to a different
file does not, and why a resolved failure still renders with its address but
raises no content probe.

## What QA deliberately does not do

- No prose quality scoring and no LLM judge. Asking a model whether the summary
  is good would make the check non-deterministic and would cost a provider call,
  which is the thing this engine exists to avoid.
- No claim that a passing report means the summary is *useful*. It means nothing
  required was lost silently. Usefulness is judged by the next turn.

## How to test

```bash
uv run pytest tests/test_compaction_qa.py
```

The tests cover: a full render answering every probe, probes being stable and
independent of the text, a stripped file name failing a content probe, a removed
footer failing only the address score, an unresolved error requiring its
signature while a resolved one does not, addressed markers surviving a file
storm, CJK requests and CJK error text surviving the byte ceilings, and an empty
window producing no probes.

## Next

[Deterministic Compaction As An Extension](./compaction-engine-extension.md)
puts this gate in the product path: the engine writes a summary only when
`qa_report(...).ok`, and declines to Tau's summarizer when it does not.
