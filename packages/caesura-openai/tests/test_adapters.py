"""Tests for caesura_openai.adapters — ports of adapters.test.ts."""

from __future__ import annotations

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
        assert collected[0].text == "What is \nthis?"

    def test_handles_tool_calls_with_no_content(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "function": {"name": "weather"}}],
            }
        ]
        collected = collect_openai_messages(messages)
        assert len(collected) == 1
        assert "call_1" in collected[0].text


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
