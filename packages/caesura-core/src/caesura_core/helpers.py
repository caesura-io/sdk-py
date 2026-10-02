"""Pure helper functions for the CaesuraO core engine.

Includes: message hashing, backend message building, TTL/keepLast selection,
template rendering, and block assembly.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from caesura_core.types import (
    AnalyzeMessage,
    CaesuraAnalysis,
    InjectConfig,
    ResolvedInjectConfig,
    SendConfig,
    SpeakerNames,
    TtlSeconds,
    TtlTurns,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from caesura_core.store import ConversationState, StoredRecommendation

if True:  # TYPE_CHECKING guard that works at runtime too
    pass


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
    # JavaScript charCodeAt iterates UTF-16 units, including surrogate pairs.
    encoded = input_str.encode("utf-16-le", errors="surrogatepass")
    for i in range(0, len(encoded), 2):
        h ^= encoded[i] | (encoded[i + 1] << 8)
        h = (h * _FNV_PRIME) & _MASK_32
    return _to_base36(h)


def _to_base36(n: int) -> str:
    """Convert an unsigned integer to a base-36 string."""
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


def normalize_dialogue(message: AnalyzeMessage, names: SpeakerNames | None = None) -> AnalyzeMessage:
    """Snapshot participant identity without changing the provider's source role."""
    names = names if names is not None else SpeakerNames()
    return replace(
        message,
        speaker_name=message.speaker_name
        if message.speaker_name is not None
        else (names.agent if message.speaker_role == "assistant" else names.customer),
        speaker_index=message.speaker_index
        if message.speaker_index is not None
        else (0 if message.speaker_role == "assistant" else 1),
    )


_LONE_SURROGATE = re.compile(r"[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]")


def dialogue_anchors(collected: Sequence[AnalyzeMessage]) -> list[str]:
    """Identify occurrences in an append-only dialogue, not just matching text.

    Prefix fingerprints distinguish repeated turns even from the same speaker.
    Compute before SDK trimming. If the caller edits or removes earlier history,
    unmatched analyses use the existing latest-context fallback instead of being
    attached to a different occurrence of the same text.
    """
    # FNV-1a/64 over JSON.stringify identity arrays, matching JS charCodeAt.
    digest = 0xCBF29CE484222325
    anchors: list[str] = []
    for occurrence, message in enumerate(collected, 1):
        index = (
            message.speaker_index
            if message.speaker_index is not None
            else (0 if message.speaker_role == "assistant" else 1)
        )
        identity = json.dumps(
            [index, message.speaker_name or "", message.text], ensure_ascii=False, separators=(",", ":")
        )
        # Well-formed JSON.stringify escapes lone surrogates, but not valid pairs.
        identity = _LONE_SURROGATE.sub(lambda match: f"\\u{ord(match[0]):04x}", identity) + "\n"
        encoded = identity.encode("utf-16-le", errors="surrogatepass")
        for i in range(0, len(encoded), 2):
            code_unit = encoded[i] | (encoded[i + 1] << 8)
            digest = ((digest ^ code_unit) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        anchors.append(f"v1:{occurrence}:{_to_base36(digest)}")
    return anchors


def build_analyze_messages(
    collected: list[AnalyzeMessage],
    state: ConversationState,
    *,
    send: SendConfig | None = None,
    speaker_names: SpeakerNames | None = None,
) -> list[AnalyzeMessage]:
    """Build the backend ``messages`` array with buffered analyses interleaved.

    Anchors identify the analyzed occurrence in the supplied dialogue history.
    Older stored recommendations without occurrence anchors use content hashes.
    """
    recommendations = state.recommendations
    anchors = dialogue_anchors([normalize_dialogue(c, speaker_names) for c in collected]) if recommendations else []
    original_length = len(collected)
    if send is not None:
        collected = _limit_dialogue(collected, send)
        if not collected:
            return []

    def _copy_dialogue(c: AnalyzeMessage) -> AnalyzeMessage:
        # Both conversation participants are transcript input to the analyzer.
        # The backend's assistant role is reserved for prior CaesuraO analyses.
        return replace(normalize_dialogue(c, speaker_names), speaker_role="user")

    if not recommendations:
        # Fast path: no analyses to interleave.
        return [_copy_dialogue(c) for c in collected]

    # Resolve occurrence anchors before trimming changes text or local indices.
    offset = original_length - len(collected)
    anchor_to_position = {anchor: i - offset for i, anchor in enumerate(anchors) if i >= offset}

    # Legacy recommendations only have a content hash.
    hash_to_positions: dict[str, list[int]] = {}
    for i, c in enumerate(collected):
        h = hash_message(c.speaker_name or "", c.text)
        hash_to_positions.setdefault(h, []).append(i)

    # O(m): resolve each recommendation's insertion position backwards.
    placements: list[dict[str, Any]] = [{}] * len(recommendations)
    for i in range(len(recommendations) - 1, -1, -1):
        r = recommendations[i]
        if r.after_message_anchor is not None:
            position = anchor_to_position.get(r.after_message_anchor)
        else:
            positions = hash_to_positions.get(r.after_message_hash)
            position = positions.pop() if positions else None
        placements[i] = {
            "msg": AnalyzeMessage(
                speaker_role="assistant",
                text=stringify_value(r.analysis),
                speaker_index=-1,
            ),
            "position": position,
        }

    # Collapse trimmed/unmatched anchors before charging for emitted history.
    # Preserve chronological order so budgeting takes a contiguous newest suffix.
    latest_unanchored = next((p for p in reversed(placements) if p["position"] is None), None)
    placements = [p for p in placements if p["position"] is not None or p is latest_unanchored]
    if send is not None:
        slots = None if send.max_messages == "all" else send.max_messages - len(collected)
        chars = None if send.max_input_chars is None else send.max_input_chars - sum(len(c.text) for c in collected)
        retained = []
        for placement in reversed(placements):
            length = len(placement["msg"].text)
            if slots == 0 or (chars is not None and length > chars):
                break
            retained.append(placement)
            if slots is not None:
                slots -= 1
            if chars is not None:
                chars -= length
        placements = list(reversed(retained))

    # Build result array.
    result: list[AnalyzeMessage] = []

    # Prepend at most 1 analysis (the latest one) if its anchor was trimmed.
    unanchored = [p for p in placements if p["position"] is None]
    if unanchored:
        result.append(unanchored[-1]["msg"])

    # The backend persists the final message as the current utterance. When an
    # identical prompt matches a previous analysis's anchor, keep that history
    # before the current utterance rather than leaving analysis last.
    for ci, c in enumerate(collected):
        anchored = [p["msg"] for p in placements if p["position"] == ci]
        if ci == len(collected) - 1:
            result.extend(anchored)
            result.append(_copy_dialogue(c))
        else:
            result.append(_copy_dialogue(c))
            result.extend(anchored)

    return result


def _limit_dialogue(collected: list[AnalyzeMessage], send: SendConfig) -> list[AnalyzeMessage]:
    """Keep the newest dialogue, trimming the oldest retained text when needed."""
    if send.max_messages == 0 or send.max_input_chars == 0:
        return []
    window = collected if send.max_messages == "all" else collected[-send.max_messages :]
    if send.max_input_chars is None:
        return list(window)

    remaining = send.max_input_chars
    retained: list[AnalyzeMessage] = []
    for message in reversed(window):
        if remaining <= 0:
            break
        text = message.text[-remaining:]
        retained.append(
            AnalyzeMessage(
                speaker_role=message.speaker_role,
                speaker_name=message.speaker_name,
                speaker_index=message.speaker_index,
                text=text,
            )
        )
        remaining -= len(text)
    return list(reversed(retained))


# ---------------------------------------------------------------------------
# Select active recommendations (TTL + keepLast)
# ---------------------------------------------------------------------------


def select_active(
    state: ConversationState,
    inject: ResolvedInjectConfig,
    now_ms: float,
) -> list[StoredRecommendation]:
    """Apply TTL + keepLast to pick recommendations currently eligible for context."""
    if inject.keep_last == 0:
        return []
    recs: Sequence[StoredRecommendation] = state.recommendations

    if isinstance(inject.ttl, TtlTurns):
        min_turn = state.turn - inject.ttl.turns
        recs = [r for r in recs if r.created_at_turn >= min_turn]
    elif isinstance(inject.ttl, TtlSeconds):
        cutoff = now_ms - inject.ttl.seconds * 1000
        recs = [r for r in recs if r.created_at_ms >= cutoff]

    recs = list(recs[-inject.keep_last :]) if inject.keep_last != "all" else list(recs)

    return recs


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

_FIELD_TOKEN = re.compile(r"\{analysis(?:\.([^{}]+))?\}")


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
                # {analysis} → plain text or the complete JSON value
                value: CaesuraAnalysis = analysis
            else:
                # Object fields keep their exact backend names.
                value = analysis.get(field_name) if isinstance(analysis, dict) else None
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


def stringify_value(value: Any) -> str:
    """Convert a value to its string representation for template rendering."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value).lower() if isinstance(value, bool) else str(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Render the injection block
# ---------------------------------------------------------------------------


def render_block(
    recs: list[StoredRecommendation],
    inject: ResolvedInjectConfig | InjectConfig,
) -> list[dict[str, Any]]:
    """Render the full injection block (rendered active analyses).

    Returns a list of dicts with keys: ``recommendation_id``, ``text``,
    ``after_message_hash``, ``after_message_anchor``, ``created_at_turn``.
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
                    "after_message_anchor": r.after_message_anchor,
                    "created_at_turn": r.created_at_turn,
                }
            )
    return result
