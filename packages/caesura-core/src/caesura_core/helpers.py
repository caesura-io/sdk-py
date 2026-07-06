"""Pure helper functions for the Caesura core engine.

Includes: message hashing, backend message building, TTL/keepLast selection,
template rendering, and block assembly.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from caesura_core.types import (
    AnalyzeMessage,
    CaesuraAnalysis,
    InjectConfig,
    ResolvedInjectConfig,
    TtlSeconds,
    TtlTurns,
)

if True:  # TYPE_CHECKING guard that works at runtime too
    from caesura_core.store import ConversationState, StoredRecommendation


# ---------------------------------------------------------------------------
# FNV-1a hashing (32-bit, matching the TS implementation exactly)
# ---------------------------------------------------------------------------

_FNV_OFFSET_BASIS = 0x811C9DC5
_FNV_PRIME = 0x01000193
_MASK_32 = 0xFFFFFFFF


def hash_message(speaker_name: str, text: str) -> str:
    """FNV-1a hash of a message's identity (speakerName + text).

    Matches the TS implementation exactly so hashes are interoperable.
    The result is the 32-bit unsigned hash encoded in base-36.
    """
    input_str = f"{speaker_name}\0{text}"
    h = _FNV_OFFSET_BASIS
    for ch in input_str:
        h ^= ord(ch)
        h = (h * _FNV_PRIME) & _MASK_32
    return _to_base36(h)


def _to_base36(n: int) -> str:
    """Convert unsigned 32-bit integer to base-36 string (matching JS ``(n >>> 0).toString(36)``)."""
    if n == 0:
        return "0"
    chars = "0123456789abcdefghijklmnopqrstuvwxyz"
    result: list[str] = []
    while n > 0:
        result.append(chars[n % 36])
        n //= 36
    return "".join(reversed(result))


# ---------------------------------------------------------------------------
# Build analyze messages (interleave buffered analyses)
# ---------------------------------------------------------------------------


def build_analyze_messages(
    collected: list[AnalyzeMessage],
    state: ConversationState,
) -> list[AnalyzeMessage]:
    """Build the backend ``messages`` array with buffered analyses interleaved.

    Uses content hashes (not indices) to find insertion points, so the result
    is correct even when the prompt has been trimmed, reordered, or mutated
    between turns.
    """
    if not state.recommendations:
        # Fast path: no analyses to interleave.
        return [
            AnalyzeMessage(
                speaker_role=c.speaker_role,
                speaker_name=c.speaker_name,
                text=c.text,
            )
            for c in collected
        ]

    # O(n): build hash → list of positions.
    hash_to_positions: dict[str, list[int]] = {}
    for i, c in enumerate(collected):
        h = hash_message(c.speaker_name or "", c.text)
        hash_to_positions.setdefault(h, []).append(i)

    # O(m): resolve each recommendation's insertion position backwards.
    placements: list[dict[str, Any]] = [{}] * len(state.recommendations)
    for i in range(len(state.recommendations) - 1, -1, -1):
        r = state.recommendations[i]
        positions = hash_to_positions.get(r.after_message_hash)
        position: int | None = positions.pop() if positions else None
        placements[i] = {
            "msg": AnalyzeMessage(
                speaker_role="assistant",
                text=json.dumps(r.analysis.to_dict()),
            ),
            "position": position,
        }

    # Build result array.
    result: list[AnalyzeMessage] = []

    # Prepend at most 1 analysis (the latest one) if its anchor was trimmed.
    unanchored = [p for p in placements if p["position"] is None]
    if unanchored:
        result.append(unanchored[-1]["msg"])

    # Walk collected messages and insert analyses after their anchor.
    for ci, c in enumerate(collected):
        result.append(
            AnalyzeMessage(
                speaker_role=c.speaker_role,
                speaker_name=c.speaker_name,
                text=c.text,
            )
        )
        for p in placements:
            if p["position"] == ci:
                result.append(p["msg"])

    return result


# ---------------------------------------------------------------------------
# Select active recommendations (TTL + keepLast)
# ---------------------------------------------------------------------------


def select_active(
    state: ConversationState,
    inject: ResolvedInjectConfig,
    now_ms: float,
) -> list[StoredRecommendation]:
    """Apply TTL + keepLast to pick recommendations currently eligible for context."""
    recs: Sequence[StoredRecommendation] = state.recommendations

    if isinstance(inject.ttl, TtlTurns):
        min_turn = state.turn - inject.ttl.turns
        recs = [r for r in recs if r.created_at_turn >= min_turn]
    elif isinstance(inject.ttl, TtlSeconds):
        cutoff = now_ms - inject.ttl.seconds * 1000
        recs = [r for r in recs if r.created_at_ms >= cutoff]

    if inject.keep_last != "all":
        recs = list(recs[-inject.keep_last :])
    else:
        recs = list(recs)

    return recs


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

_FIELD_TOKEN = re.compile(r"\{analysis(?:\.([a-zA-Z0-9_$]+))?\}")


def render_analysis(analysis: CaesuraAnalysis, template: str) -> str:
    """Render one analysis through the template, resolving ``{analysis}`` / ``{analysis.field}``."""
    lines = template.split("\n")
    rendered: list[str] = []

    for line in lines:
        saw_token = False
        all_empty = True

        def _replace(m: re.Match[str]) -> str:
            nonlocal saw_token, all_empty
            saw_token = True
            field_name = m.group(1)
            if field_name is None:
                # {analysis} → full JSON
                value: Any = analysis.to_dict()
            else:
                # {analysis.field} → specific field
                value = getattr(analysis, _camel_to_snake(field_name), None)
                if value is None:
                    value = analysis.extra.get(field_name)
            s = stringify_value(value)
            if s != "":
                all_empty = False
            return s

        out = _FIELD_TOKEN.sub(_replace, line)
        # Drop lines whose only content was empty token(s).
        if saw_token and all_empty:
            continue
        rendered.append(out)

    return "\n".join(rendered)


def _camel_to_snake(name: str) -> str:
    """Convert a camelCase field name to snake_case for attribute lookup."""
    # Handle known mappings first
    mappings = {
        "isSame": "is_same",
        "speakerRole": "speaker_role",
        "speakerName": "speaker_name",
    }
    if name in mappings:
        return mappings[name]
    return name


def stringify_value(value: Any) -> str:
    """Convert a value to its string representation for template rendering."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value).lower() if isinstance(value, bool) else str(value)
    return json.dumps(value)


# ---------------------------------------------------------------------------
# Render the injection block
# ---------------------------------------------------------------------------


def render_block(
    recs: list[StoredRecommendation],
    inject: ResolvedInjectConfig | InjectConfig,
) -> list[dict[str, Any]]:
    """Render the full injection block (rendered active analyses).

    Returns a list of dicts with keys: ``recommendation_id``, ``text``,
    ``after_message_hash``, ``created_at_turn``.
    """
    template = inject.template
    if template is None:
        from caesura_core.defaults import DEFAULT_TEMPLATE

        template = DEFAULT_TEMPLATE

    result: list[dict[str, Any]] = []
    for r in recs:
        text = render_analysis(r.analysis, template)
        if text.strip():
            result.append(
                {
                    "recommendation_id": r.id,
                    "text": text,
                    "after_message_hash": r.after_message_hash,
                    "created_at_turn": r.created_at_turn,
                }
            )
    return result
