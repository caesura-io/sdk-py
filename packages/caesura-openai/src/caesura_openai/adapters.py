"""Adapters for translating between OpenAI's message format and CaesuraO's internal format."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from caesura_core.helpers import dialogue_anchors, normalize_dialogue
from caesura_core.types import AnalyzeMessage, InjectedBlock, ResolvedInjectConfig, SpeakerNames
from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


def get_message_text(message: dict[str, Any]) -> str:
    """Extract text from an OpenAI message. Handles both string content and array of content parts."""
    content = message.get("content")
    if not content:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = [
            part["text"]
            for part in content
            if isinstance(part, dict)
            and part.get("type") in ("text", "input_text", "output_text")
            and isinstance(part.get("text"), str)
        ]
        return "".join(text_parts)

    return ""


def _message_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, BaseModel):
        return item.model_dump()
    return item if isinstance(item, dict) else {}


def strip_injected_messages(messages: Iterable[Any], known: set[tuple[str, str]]) -> list[Any]:
    """Remove exact guidance messages emitted by this conversation's wrapper.

    Match role and text, not a prefix or the assistant role generally. Tool items,
    named participant messages, and assistant messages carrying tool calls remain.
    """
    result = []
    for item in messages:
        message = _message_dict(item)
        content = message.get("content")
        text_only = isinstance(content, str) or (
            isinstance(content, list)
            and all(
                isinstance(part, dict)
                and part.get("type") in ("text", "input_text", "output_text")
                and isinstance(part.get("text"), str)
                for part in content
            )
        )
        is_guidance = (
            message.get("type", "message") == "message"
            and not message.get("tool_calls")
            and message.get("function_call") is None
            and message.get("audio") is None
            and text_only
            and message.get("name") is None
            and message.get("speakerIndex") is None
            and (message.get("role"), get_message_text(message)) in known
        )
        if not is_guidance:
            result.append(item)
    return result


def _dialogue_items(
    messages: Iterable[Any], speaker_names: SpeakerNames | None = None
) -> list[tuple[int, AnalyzeMessage]]:
    """Shared collection and anchor rules, retaining original provider positions."""
    collected = []
    for position, item in enumerate(messages):
        message = _message_dict(item)
        role = message.get("role")
        if role not in ("user", "assistant") or message.get("type", "message") != "message":
            continue
        text = get_message_text(message)
        if not text.strip():
            continue
        index = message.get("speakerIndex")
        collected.append(
            (
                position,
                normalize_dialogue(
                    AnalyzeMessage(
                        speaker_role=role,
                        speaker_name=message.get("name"),
                        speaker_index=index
                        if isinstance(index, int) and not isinstance(index, bool)
                        else (0 if role == "assistant" else 1),
                        text=text,
                    ),
                    speaker_names,
                ),
            )
        )
    return collected


def collect_openai_messages(messages: Iterable[Any], speaker_names: SpeakerNames | None = None) -> list[AnalyzeMessage]:
    """Convert OpenAI messages into CaesuraO AnalyzeMessages.

    Only includes messages from 'user' and 'assistant' roles (system, tool,
    and developer messages are ignored for analysis). Tool-call metadata is
    excluded; any accompanying assistant text is preserved.
    """
    return [message for _, message in _dialogue_items(messages, speaker_names)]


def collect_openai_responses_messages(
    user_input: Any, speaker_names: SpeakerNames | None = None
) -> list[AnalyzeMessage]:
    """Collect user and assistant text, excluding tool and reasoning items."""
    if isinstance(user_input, str):
        return collect_openai_messages([{"role": "user", "content": user_input}], speaker_names)
    if not isinstance(user_input, list):
        return []

    return collect_openai_messages(user_input, speaker_names)


def apply_skill_prompt_responses(instructions: Any, inject: ResolvedInjectConfig) -> Any:
    """Keep the skill in instructions and avoid adding it again on reused requests."""
    skill = inject.skill_prompt
    if not skill or not skill.strip():
        return instructions
    existing = instructions if isinstance(instructions, str) else ""
    if skill in existing:
        return existing
    return f"{existing}\n\n{skill}" if existing else skill


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

    # The skill is an instruction, independent of the recommendation's role.
    target_roles = {"developer", "system"}

    last_match_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") in target_roles:
            last_match_idx = i
            break

    if last_match_idx >= 0:
        # Append to existing
        m = messages[last_match_idx]
        if inject.skill_prompt in get_message_text(m):
            return messages, False
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
    messages: list[Any],
    blocks: list[dict[str, Any]],
    inject: ResolvedInjectConfig,
    hash_fn: Callable[[str, str], str],
    speaker_names: SpeakerNames | None = None,
) -> tuple[list[Any], list[InjectedBlock]]:
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
    dialogue = _dialogue_items(result, speaker_names)
    for position, message in dialogue:
        h = hash_fn(message.speaker_name or "", message.text)
        hash_to_positions.setdefault(h, []).append(position)
        # Pre-normalization stores used the provider name (often absent).
        legacy_hash = hash_fn(_message_dict(result[position]).get("name") or "", message.text)
        if legacy_hash != h:
            hash_to_positions.setdefault(legacy_hash, []).append(position)
    anchor_to_position = dict(
        zip(dialogue_anchors([message for _, message in dialogue]), [pos for pos, _ in dialogue], strict=True)
    )

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
        anchor = group_blocks[0].get("after_message_anchor")
        if anchor is not None:
            pos = anchor_to_position.get(anchor)
        else:
            after_hash = group_blocks[0]["after_message_hash"]
            positions = hash_to_positions.get(after_hash)
            pos = positions.pop() if positions else None

        if pos is not None:
            # Keep an assistant tool call and all following tool results together.
            insertion_pos = pos + 1
            while insertion_pos < len(result):
                following = _message_dict(result[insertion_pos])
                if following.get("role") != "tool" and following.get("type") not in (
                    "function_call",
                    "function_call_output",
                    "reasoning",
                ):
                    break
                insertion_pos += 1
            for b in group_blocks:
                insertions.append(
                    {
                        "index": insertion_pos,
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
