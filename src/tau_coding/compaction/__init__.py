"""Deterministic, provider-free compaction primitives.

These modules build a compaction summary from typed session structure alone: no
summarizer request, no prose classification, and byte-bounded output. The seam
that swaps them in is the ``before_compact`` extension hook.

```text
bounds.py   UTF-8 ceilings and conservation-checked sampling
events.py   typed normalization of one active-context window
cut.py      closure-safe cut selection and token-budget ceilings
```
"""

from tau_coding.compaction.bounds import (
    MAX_SUMMARY_BYTES,
    AddressedSample,
    BoundedExcerpt,
    CanonicalText,
    bounded_excerpt,
    canonicalize_text,
    clip_utf8,
    omission_line,
    sample_addressed,
    sample_addressed_from,
    utf8_bytes,
)
from tau_coding.compaction.cut import (
    DEFAULT_TARGET_OCCUPANCY,
    SUMMARY_RAW_TOKEN_BUDGET,
    CallResultSpan,
    ContextCut,
    CutBudget,
    call_result_spans,
    closure_safe,
    compute_cut,
)
from tau_coding.compaction.events import (
    BRANCH_SUMMARY_PREFIX,
    AssistantTextEvent,
    BashEvent,
    CompactionEvent,
    ContextRow,
    CustomMessageEvent,
    PriorSummaryEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserEvent,
    count_erased_thinking_blocks,
    first_line,
    normalize_rows,
)

__all__ = [
    "BRANCH_SUMMARY_PREFIX",
    "DEFAULT_TARGET_OCCUPANCY",
    "MAX_SUMMARY_BYTES",
    "SUMMARY_RAW_TOKEN_BUDGET",
    "AddressedSample",
    "AssistantTextEvent",
    "BashEvent",
    "BoundedExcerpt",
    "CallResultSpan",
    "CanonicalText",
    "CompactionEvent",
    "ContextCut",
    "ContextRow",
    "CustomMessageEvent",
    "CutBudget",
    "PriorSummaryEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "UserEvent",
    "bounded_excerpt",
    "call_result_spans",
    "canonicalize_text",
    "clip_utf8",
    "closure_safe",
    "compute_cut",
    "count_erased_thinking_blocks",
    "first_line",
    "normalize_rows",
    "omission_line",
    "sample_addressed",
    "sample_addressed_from",
    "utf8_bytes",
]
