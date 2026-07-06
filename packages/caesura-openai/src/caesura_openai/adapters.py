"""Adapters for translating between OpenAI's message format and Caesura's internal format."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from caesura_core.types import AnalyzeMessage, InjectedBlock, ResolvedInjectConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


def get_message_text(message: dict[str, Any]) -> str:
    """Extract text from an OpenAI message. Handles both string content and array of content parts."""
    content = message.get("content")
    if not content:
        # Check tool calls
        tool_calls = message.get("tool_calls")
        if tool_calls and isinstance(tool_calls, list):
            return json.dumps(tool_calls)
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
        return "\n".join(text_parts)

    return ""


def collect_openai_messages(messages: Iterable[dict[str, Any]]) -> list[AnalyzeMessage]:
    """Convert OpenAI messages into Caesura AnalyzeMessages.

    Only includes messages from 'user' and 'assistant' roles (system, tool,
    and developer messages are ignored for analysis).
    """
    collected: list[AnalyzeMessage] = []
    for m in messages:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue

        text = get_message_text(m)
        if not text:
            continue

        collected.append(
            AnalyzeMessage(
                speaker_role=role,
                speaker_name=m.get("name"),
                text=text,
            )
        )
    return collected


def apply_skill_prompt_openai(
    messages: list[dict[str, Any]],
    inject: ResolvedInjectConfig,
) -> tuple[list[dict[str, Any]], bool]:
    """Inject the skill prompt into the messages array.

    Returns the (possibly modified) messages array and a boolean indicating
    whether the skill prompt was successfully injected.
    """
    if not inject.skill_prompt:
        return messages, False

    # Find the last message matching the target role (developer/system)
    target_roles = {"developer", "system"} if inject.as_role in ("developer", "system") else {inject.as_role}

    last_match_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") in target_roles:
            last_match_idx = i
            break

    if last_match_idx >= 0:
        # Append to existing
        m = messages[last_match_idx]
        new_content = m.get("content", "")
        if isinstance(new_content, list):
            new_content = list(new_content)
            new_content.append({"type": "text", "text": f"\n\n{inject.skill_prompt}"})
        else:
            new_content = f"{new_content}\n\n{inject.skill_prompt}" if new_content else inject.skill_prompt

        new_msg = {**m, "content": new_content}
        result = list(messages)
        result[last_match_idx] = new_msg
        return result, True
    else:
        # Prepend new message
        new_msg = {
            "role": "developer" if inject.as_role == "developer" else "system",
            "content": inject.skill_prompt,
        }
        return [new_msg, *messages], True


def inject_blocks_openai(
    messages: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
    inject: ResolvedInjectConfig,
    hash_fn: Callable[[str, str], str],
) -> tuple[list[dict[str, Any]], list[InjectedBlock]]:
    """Inject rendered recommendation blocks into the OpenAI messages array."""
    if not blocks:
        return messages, []

    result = list(messages)
    injected: list[InjectedBlock] = []

    if inject.placement == "end":
        for b in blocks:
            result.append({"role": inject.as_role, "content": b["text"]})
            injected.append(
                InjectedBlock(recommendation_id=b["recommendation_id"], text=b["text"], index=len(result) - 1)
            )
        return result, injected

    # placement == 'after-last-analyzed' -> interleave them chronologically
    hash_to_positions: dict[str, list[int]] = {}
    for i, m in enumerate(result):
        role = m.get("role")
        if role in ("user", "assistant"):
            text = get_message_text(m)
            if text:
                h = hash_fn(m.get("name", "") or "", text)
                hash_to_positions.setdefault(h, []).append(i)

    # Group by turn
    turn_groups: dict[int, list[dict[str, Any]]] = {}
    for b in blocks:
        turn = b.get("created_at_turn", 0)
        turn_groups.setdefault(turn, []).append(b)

    sorted_turns = sorted(turn_groups.keys(), reverse=True)
    insertions: list[dict[str, Any]] = []
    latest_unanchored_turn: int | None = None

    for turn in sorted_turns:
        group_blocks = turn_groups[turn]
        after_hash = group_blocks[0]["after_message_hash"]
        positions = hash_to_positions.get(after_hash)
        pos = positions.pop() if positions else None

        if pos is not None:
            for b in group_blocks:
                insertions.append(
                    {
                        "index": pos + 1,
                        "text": b["text"],
                        "block_index": blocks.index(b),
                        "rec_id": b["recommendation_id"],
                    }
                )
        else:
            if latest_unanchored_turn is None:
                latest_unanchored_turn = turn

    if latest_unanchored_turn is not None:
        for b in turn_groups[latest_unanchored_turn]:
            insertions.append(
                {"index": 0, "text": b["text"], "block_index": blocks.index(b), "rec_id": b["recommendation_id"]}
            )

    grouped_insertions: dict[int, dict[str, list[Any]]] = {}
    for ins in insertions:
        idx = ins["index"]
        if idx not in grouped_insertions:
            grouped_insertions[idx] = {"texts": [], "block_indices": [], "rec_ids": []}
        grouped_insertions[idx]["texts"].append(ins["text"])
        grouped_insertions[idx]["block_indices"].append(ins["block_index"])
        grouped_insertions[idx]["rec_ids"].append(ins["rec_id"])

    sorted_indices = sorted(grouped_insertions.keys())

    for offset, idx in enumerate(sorted_indices):
        group = grouped_insertions[idx]
        sorted_group = sorted(
            zip(group["block_indices"], group["texts"], group["rec_ids"], strict=False), key=lambda x: x[0]
        )
        merged_text = "\n\n".join(g[1] for g in sorted_group)

        insert_pos = idx + offset
        msg = {"role": inject.as_role, "content": merged_text}
        result.insert(insert_pos, msg)

        for _bi, _, rec_id in sorted_group:
            injected.append(InjectedBlock(recommendation_id=rec_id, text=merged_text, index=insert_pos))

    # Ensure injected list matches order of original blocks for testing consistency
    injected_ordered = [None] * len(blocks)
    for inj in injected:
        for bi, b in enumerate(blocks):
            if b["recommendation_id"] == inj.recommendation_id:
                injected_ordered[bi] = inj  # type: ignore[call-overload]
                break

    return result, [i for i in injected_ordered if i is not None]  # type: ignore[misc]
