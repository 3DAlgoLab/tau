"""Deterministic, provider-free compaction primitives.

These modules build a compaction summary from typed session structure alone: no
summarizer request, no prose classification, and byte-bounded output. The seam
that swaps them in is the ``before_compact`` extension hook.

```text
bounds.py       UTF-8 ceilings and conservation-checked sampling
events.py       typed normalization of one active-context window
cut.py          closure-safe cut selection and token-budget ceilings
dialogue.py     verbatim recent exchanges: the section that must not decay
projections.py  section folds that derive every summary section
render.py       byte-budgeted assembly of sections into one summary
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
from tau_coding.compaction.dialogue import (
    MAX_RECENT_EXCHANGES,
    DialogueExchange,
    DialogueMessage,
    project_dialogue,
    recent_dialogue,
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
from tau_coding.compaction.projections import (
    MAX_FILES_PER_KIND,
    MAX_UNRESOLVED,
    Sections,
    collect_operations,
    project,
    project_with_metadata,
    truncate,
)
from tau_coding.compaction.render import (
    RenderedSummary,
    RenderOptions,
    format_timestamp,
    render_summary,
    render_summary_with_metadata,
)

__all__ = [
    "BRANCH_SUMMARY_PREFIX",
    "DEFAULT_TARGET_OCCUPANCY",
    "MAX_FILES_PER_KIND",
    "MAX_RECENT_EXCHANGES",
    "MAX_SUMMARY_BYTES",
    "MAX_UNRESOLVED",
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
    "DialogueExchange",
    "DialogueMessage",
    "PriorSummaryEvent",
    "RenderOptions",
    "RenderedSummary",
    "Sections",
    "ToolCallEvent",
    "ToolResultEvent",
    "UserEvent",
    "bounded_excerpt",
    "call_result_spans",
    "canonicalize_text",
    "clip_utf8",
    "closure_safe",
    "collect_operations",
    "compute_cut",
    "count_erased_thinking_blocks",
    "first_line",
    "format_timestamp",
    "normalize_rows",
    "omission_line",
    "project",
    "project_dialogue",
    "project_with_metadata",
    "recent_dialogue",
    "render_summary",
    "render_summary_with_metadata",
    "sample_addressed",
    "sample_addressed_from",
    "truncate",
    "utf8_bytes",
]
