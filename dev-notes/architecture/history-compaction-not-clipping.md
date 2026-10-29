# Compacting child and root history instead of clipping it

Status: design, not implemented. Opened by 2c-iii's first live run: a child printed a
2,401-row file, got head-clipped at 2,048 bytes, and answered `{"error": "row not found"}`
because the row it needed was past the clip. The prompt fix landed with the run
(`rlm.py::_observe` now puts the loss notice *before* the text); this note is the part
that change did not finish.

## What happens today

| layer | code | behaviour |
|---|---|---|
| root cell output | `code_mode.py::_describe` -> `truncate_head` | head slice + a size note, appended after the text |
| child turn history | `rlm.py::_observe` + `_observations_block` | every observation head-clipped at `max_observation_bytes` (2 KB), all of them re-sent in the next prompt, no total budget |
| session history | Phase 1 engine (`compaction/*`) | deterministic, conservation-checked, byte-budgeted -- and not used by either of the two rows above |

Two distinct failures follow from that:

1. **Loss without address.** A head clip keeps the beginning of what the cell printed. For
   a file dump the interesting line is usually not the beginning, and nothing in the clip
   says what kind of content was dropped or how to get it back except by re-running code.
2. **No budget on the history itself.** Three turns of a chatty child re-send three
   clipped observations every turn. The clip is per observation, so the prompt still grows
   with turns; the child spends its window relaying its own old stdout.

## The rule to implement

Feed both layers through the Phase 1 primitives instead of `truncate_head` / per-item
clipping. The engine already has exactly the pieces this needs and they are deterministic
(no summarizer call, no prose classification):

* `compaction.bounds.bounded_excerpt` -- clip plus the exact lost byte count inside the text,
* `compaction.bounds.clip_utf8` -- never split a character,
* `compaction.bounds.omission_line` -- `... omitted N items; source entries a -> b`,
* `compaction.projections` graded decay -- oldest turns collapse to one line, recent ones
  stay verbatim.

New shape for the child's history (`rlm.py`):

```python
DEFAULT_MAX_HISTORY_BYTES = 6 * 1024   # whole prompt history, not per observation
DEFAULT_VERBATIM_TURNS = 1             # the newest observation stays whole
```

`_observations_block(observations, budget_bytes=..., verbatim=...)` renders oldest-first:

```text
--- cell 1 --- collapsed: 9,310 bytes
  first line of the cell's output
  last line of the cell's output
  [omitted 9,102 UTF-8 bytes between them]
--- cell 2 --- collapsed: 2,204 bytes  ...
--- cell 3 --- (verbatim)
```

Invariants, in the order I intend to test them:

1. **Conservation.** Every byte that is not shown is counted, and the count appears in the
   text the model reads. Silent shrinking is the bug class; this is the same rule
   `sample_addressed` enforces for session compaction.
2. **Both ends survive.** A collapsed observation keeps its first line *and* its last line.
   The failure that started this was a needle in the tail of a dump.
3. **A total budget.** The history block is bounded by `max_history_bytes` regardless of
   turn count, so a chatty child cannot re-flood its own window.
4. **Determinism.** Same observations, same render. No model call, no randomness.
5. **The verbatim tail is at least one turn.** The child must always see what its last cell
   printed, uncollapsed, or it cannot correct itself.

Root side is the same rule with a smaller change: `_describe` should lead with the
`bounded_excerpt` notice (as `_observe` now does) and keep head + tail rather than head
only, so a `print(f.read())` in the root session does not silently lose the end of the
file either.

## Deliberately not doing

* **No LLM summary of old turns.** The 2b lesson stands: the model's own prose is a bad
  place to store state an agent needs later. Collapsing to first/last line plus a counted
  omission is honest; a paraphrase is not.
* **No bigger clip.** Raising `max_observation_bytes` moves the cliff; it does not make the
  loss addressable, and it multiplies cost per turn.

## The cut had to move inside the kernel too

The first version of this fix lived only in `rlm.py`, and a test caught that it
wasn't enough: a child printing an 80 KB ledger was stopped by the *kernel's*
64 KiB per-cell output cap long before the recursion's budget mattered, and the
row it was looking for still sat past the cut. Whatever reaches the host can be
compacted; what the kernel throws away never is. So `kernel/child.py::_Budget`
stopped being a flat head allowance:

- three quarters of the allowance is spent as the cell streams, so a long-running
  cell still reports progress the way it used to;
- the rest is reserved and refilled from the end, so the last thing a cell prints
  survives a long print the way the first thing does;
- at cell end the reserved tail is sent as one frame, preceded by
  `[middle omitted: N UTF-8 bytes; this is the end of the output]`, and the
  marker's own bytes are charged to the tail — announcing the loss must not be
  what pushes a run back over the caller's budget;
- `stdout_omitted` counts every displaced byte, so the number in the text and the
  number in the frame are the same number.

`tests/test_kernel_process.py::test_output_is_capped_and_the_loss_is_counted`
pins the arithmetic: kept bytes plus announced bytes equal exactly what the cell
printed, and the print's last line is present.
