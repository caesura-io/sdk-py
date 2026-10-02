"""Tests for caesura_openai.adapters — ports of adapters.test.ts."""

from __future__ import annotations

from typing import Any

from caesura_core.helpers import hash_message
from caesura_core.types import ResolvedInjectConfig, TtlNone
from caesura_openai.adapters import apply_skill_prompt_openai, collect_openai_messages, inject_blocks_openai


class TestCollectOpenAIMessages:
    def test_collects_only_user_and_assistant_roles(self) -> None:
        messages = [
            {"role": "system", "content": "You are a bot"},
            {"role": "user", "content": "Hi"},
            {"role": "tool", "content": "data"},
            {"role": "developer", "content": "dev msg"},
            {"role": "assistant", "content": "Hello!"},
        ]
        collected = collect_openai_messages(messages)
        assert len(collected) == 2
        assert collected[0].speaker_role == "user"
        assert collected[0].text == "Hi"
        assert collected[1].speaker_role == "assistant"
        assert collected[1].text == "Hello!"

    def test_guidance_filter_preserves_participant_identity_tools_and_multimodal_content(self) -> None:
        from caesura_openai.adapters import strip_injected_messages

        known = {("user", "guide")}
        retained = [
            {"role": "assistant", "content": "guide"},
            {"role": "user", "name": "", "content": "guide"},
            {"role": "user", "speakerIndex": 0, "content": "guide"},
            {"role": "user", "content": "guide", "tool_calls": [{"id": "call"}]},
            {"role": "user", "content": "guide", "function_call": {}},
            {"role": "user", "content": "guide", "audio": {"id": "audio"}},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "guide"},
                    {"type": "input_image", "image_url": "test-image"},
                ],
            },
            {"type": "reasoning", "role": "user", "content": "guide"},
        ]
        removed = [
            {"role": "user", "content": "guide"},
            {"role": "user", "content": [{"type": "text", "text": "guide"}]},
        ]
        original = [*removed, *retained]
        result = strip_injected_messages(original, known)
        assert result == retained
        assert all(a is b for a, b in zip(result, retained, strict=True))
        assert len(original) == len(retained) + 2

    def test_handles_content_arrays(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is "},
                    {"type": "image_url", "image_url": {"url": "foo"}},
                    {"type": "text", "text": "this?"},
                ],
            }
        ]
        collected = collect_openai_messages(messages)
        assert len(collected) == 1
        assert collected[0].text == "What is this?"

    def test_handles_tool_calls_with_no_content(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "function": {"name": "weather"}}],
            }
        ]
        collected = collect_openai_messages(messages)
        assert collected == []


class TestApplySkillPromptOpenAI:
    def _make_inject(self, role: str) -> ResolvedInjectConfig:
        return ResolvedInjectConfig(
            placement="end",
            as_role=role,  # type: ignore[arg-type]
            keep_last="all",
            ttl=TtlNone(),
            template="",
            skill_prompt="Be helpful.",
        )

    def test_appends_to_existing_system_message(self) -> None:
        messages = [
            {"role": "system", "content": "You are a bot."},
            {"role": "user", "content": "Hi"},
        ]
        inject = self._make_inject("system")
        out, injected = apply_skill_prompt_openai(messages, inject)

        assert injected is True
        assert len(out) == 2
        assert out[0]["content"] == "You are a bot.\n\nBe helpful."

    def test_appends_to_existing_developer_message(self) -> None:
        messages = [
            {"role": "developer", "content": "You are a bot."},
            {"role": "user", "content": "Hi"},
        ]
        inject = self._make_inject("developer")
        out, injected = apply_skill_prompt_openai(messages, inject)

        assert injected is True
        assert out[0]["content"] == "You are a bot.\n\nBe helpful."

    def test_prepends_new_developer_message_if_none_exists(self) -> None:
        messages = [{"role": "user", "content": "Hi"}]
        inject = self._make_inject("developer")
        out, injected = apply_skill_prompt_openai(messages, inject)

        assert injected is True
        assert len(out) == 2
        assert out[0]["role"] == "developer"
        assert out[0]["content"] == "Be helpful."

    def test_target_role_system_also_matches_developer(self) -> None:
        messages = [
            {"role": "developer", "content": "You are a bot."},
            {"role": "user", "content": "Hi"},
        ]
        inject = self._make_inject("system")
        out, injected = apply_skill_prompt_openai(messages, inject)

        assert injected is True
        assert out[0]["content"] == "You are a bot.\n\nBe helpful."

    def test_appends_to_array_content(self) -> None:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a bot."}],
            }
        ]
        inject = self._make_inject("system")
        out, injected = apply_skill_prompt_openai(messages, inject)

        assert injected is True
        content = out[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[1]["text"] == "\n\nBe helpful."

    def test_skips_if_no_skill_prompt(self) -> None:
        messages = [{"role": "system", "content": "sys"}]
        inject = self._make_inject("system")
        inject.skill_prompt = None
        out, injected = apply_skill_prompt_openai(messages, inject)

        assert injected is False
        assert out[0]["content"] == "sys"


class TestInjectBlocksOpenAI:
    def _make_inject(self, placement: str) -> ResolvedInjectConfig:
        return ResolvedInjectConfig(
            placement=placement,  # type: ignore[arg-type]
            as_role="system",
            keep_last="all",
            ttl=TtlNone(),
            template="",
            skill_prompt="",
        )

    def test_injects_after_last_analyzed(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "msg 1"},
            {"role": "user", "content": "msg 2"},
        ]
        h = hash_message("", "msg 1")
        blocks = [{"recommendation_id": "r1", "text": "Rec 1", "after_message_hash": h}]
        inject = self._make_inject("after-last-analyzed")

        out, injected = inject_blocks_openai(messages, blocks, inject, hash_message)
        assert len(injected) == 1
        assert len(out) == 4
        assert out[1]["content"] == "msg 1"
        assert out[2]["role"] == "system"
        assert out[2]["content"] == "Rec 1"
        assert out[3]["content"] == "msg 2"
        assert injected[0].index == 2  # Inserted between msg 1 and msg 2

    def test_injects_at_end_if_anchor_not_found(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "msg 1"},
        ]
        blocks = [{"recommendation_id": "r1", "text": "Rec 1", "after_message_hash": "missing"}]
        inject = self._make_inject("after-last-analyzed")

        out, injected = inject_blocks_openai(messages, blocks, inject, hash_message)
        assert len(out) == 3
        # When anchor isn't found, it defaults to the front (index 0)
        assert out[0]["content"] == "Rec 1"
        assert injected[0].index == 0

    def test_appends_to_existing_role_when_placement_is_end(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "msg 1"},
            {"role": "system", "content": "another sys"},
        ]
        blocks = [{"recommendation_id": "r1", "text": "Rec 1", "after_message_hash": ""}]
        inject = self._make_inject("end")

        out, injected = inject_blocks_openai(messages, blocks, inject, hash_message)
        # TS behavior for 'end' appends a new message at the end
        assert len(out) == 4
        assert out[3]["content"] == "Rec 1"
        assert injected[0].index == 3

    def test_preserves_parallel_tool_results_and_event_indices(self) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Check both cities"},
            {
                "role": "assistant",
                "content": "Checking the weather",
                "tool_calls": [{"id": "call_1"}, {"id": "call_2"}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
            {"role": "tool", "tool_call_id": "call_2", "content": "Rainy"},
            {"role": "assistant", "content": "Here is the forecast"},
        ]
        blocks = [
            {
                "recommendation_id": "r1",
                "text": "Ask about travel plans",
                "after_message_hash": hash_message("", "Check both cities"),
                "created_at_turn": 1,
            },
            {
                "recommendation_id": "r2",
                "text": "Summarize the weather",
                "after_message_hash": hash_message("", "Checking the weather"),
                "created_at_turn": 2,
            },
        ]
        out, injected = inject_blocks_openai(messages, blocks, self._make_inject("after-last-analyzed"), hash_message)
        assert out[2:5] == messages[1:4]
        assert out[5]["content"] == "Summarize the weather"
        assert [block.index for block in injected] == [1, 5]
        assert len(messages) == 5
        # Assistant speech stays in the transcript, but function arguments and results do not.
        assert [message.text for message in collect_openai_messages(messages)] == [
            "Check both cities",
            "Checking the weather",
            "Here is the forecast",
        ]

    def test_injects_after_single_tool_result(self) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "assistant", "content": "Checking", "tool_calls": [{"id": "call_1"}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "Done"},
        ]
        blocks = [{"recommendation_id": "r1", "text": "Ask", "after_message_hash": hash_message("", "Checking")}]
        out, injected = inject_blocks_openai(messages, blocks, self._make_inject("after-last-analyzed"), hash_message)
        assert out[:2] == messages
        assert injected[0].index == 2


class TestCollectOpenAIResponsesMessages:
    def test_string_input(self) -> None:
        from caesura_openai.adapters import collect_openai_responses_messages

        collected = collect_openai_responses_messages("שלום")
        assert [message.to_dict() for message in collected] == [
            {"speakerRole": "user", "speakerName": "Customer", "speakerIndex": 1, "text": "שלום"}
        ]
        assert collect_openai_responses_messages("") == []

    def test_preserves_sdk_output_message_text_only(self) -> None:
        from caesura_openai.adapters import collect_openai_responses_messages
        from openai.types.responses import ResponseOutputMessage, ResponseOutputText

        message = ResponseOutputMessage(
            id="message-1",
            type="message",
            role="assistant",
            status="completed",
            content=[ResponseOutputText(type="output_text", text="שלום 😊", annotations=[])],
        )
        collected = collect_openai_responses_messages([message])
        assert [item.to_dict() for item in collected] == [
            {"speakerRole": "assistant", "speakerName": "Agent", "speakerIndex": 0, "text": "שלום 😊"}
        ]
        assert message.content[0].type == "output_text"


def test_skill_prompt_stays_in_instructions_and_is_idempotent() -> None:
    from caesura_core import CaesuraConfig, resolve_config

    inject = resolve_config(CaesuraConfig(api_key="fake")).inject
    messages = [{"role": "system", "content": "System prompt"}, {"role": "user", "content": "Hello"}]
    updated, changed = apply_skill_prompt_openai(messages, inject)
    assert changed
    assert updated[-1] == messages[-1]
    assert inject.skill_prompt is not None and inject.skill_prompt in updated[0]["content"]
    repeated, changed_again = apply_skill_prompt_openai(updated, inject)
    assert not changed_again
    assert repeated == updated
    assert messages[0]["content"] == "System prompt"


def test_default_skill_creates_system_message_without_changing_user_text() -> None:
    from caesura_core import CaesuraConfig, resolve_config

    inject = resolve_config(CaesuraConfig(api_key="fake")).inject
    messages = [{"role": "user", "content": "Hello"}]
    updated, _ = apply_skill_prompt_openai(messages, inject)
    assert [m["role"] for m in updated] == ["system", "user"]
    assert updated[-1] == messages[0]
