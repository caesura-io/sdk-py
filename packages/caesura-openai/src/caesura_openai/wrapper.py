"""Transparent wrappers around the OpenAI Python SDK clients.

These wrappers intercept calls to ``chat.completions.create`` and
``responses.create`` to inject Caesura recommendations, while delegating
all other attributes to the underlying OpenAI client via ``__getattr__``.
"""

from __future__ import annotations

import time
from typing import Any

from caesura_core.engine import AsyncCaesuraEngine, CaesuraEngine, create_async_caesura_engine, create_caesura_engine
from caesura_core.helpers import hash_message, render_block, select_active
from caesura_core.types import InjectedEvent

from caesura_openai.adapters import apply_skill_prompt_openai, collect_openai_messages, inject_blocks_openai
from caesura_openai.types import CaesuraOpenAIOptions


class _CaesuraCompletions:
    def __init__(self, original_completions: Any, engine: CaesuraEngine) -> None:
        self._original = original_completions
        self._engine = engine

    def create(self, *args: Any, **kwargs: Any) -> Any:
        conv_id = kwargs.pop("caesura_conversation_id", self._engine.config.conversation_id)
        if not conv_id:
            return self._original.create(*args, **kwargs)

        messages = list(kwargs.get("messages", []))
        collected = collect_openai_messages(messages)
        self._engine.observe(conv_id, collected)

        state = self._engine.store.get(conv_id)
        active = select_active(state, self._engine.config.inject, time.time() * 1000)
        blocks = render_block(active, self._engine.config.inject)

        messages, injected = inject_blocks_openai(messages, blocks, self._engine.config.inject, hash_message)
        messages, _ = apply_skill_prompt_openai(messages, self._engine.config.inject)
        kwargs["messages"] = messages

        if injected:
            self._engine.emit_event(
                InjectedEvent(
                    conversation_id=conv_id,
                    turn=state.turn,
                    blocks=injected,
                    placement=self._engine.config.inject.placement,
                )
            )

        state.turn += 1
        return self._original.create(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class _CaesuraChat:
    def __init__(self, original_chat: Any, engine: CaesuraEngine) -> None:
        self._original = original_chat
        self._engine = engine

    @property
    def completions(self) -> _CaesuraCompletions:
        return _CaesuraCompletions(self._original.completions, self._engine)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class _CaesuraResponses:
    def __init__(self, original_responses: Any, engine: CaesuraEngine) -> None:
        self._original = original_responses
        self._engine = engine

    def create(self, *args: Any, **kwargs: Any) -> Any:
        # Note: The Responses API does not use 'messages' in the same way.
        # This implementation aligns with the basic Responses API interception
        # which injects the system prompt (instruction) but cannot interleave
        # historical messages as easily.
        conv_id = kwargs.pop("caesura_conversation_id", self._engine.config.conversation_id)
        if not conv_id:
            return self._original.create(*args, **kwargs)

        # For Responses API, we treat 'input' as the user message if it's a string
        # or object, but the collect step requires more context to be accurate.
        # We perform a basic interception here.
        collected = []
        user_input = kwargs.get("input")
        if isinstance(user_input, str):
            from caesura_core.types import AnalyzeMessage
            collected.append(AnalyzeMessage(speaker_role="user", text=user_input))
        elif isinstance(user_input, list):
             from caesura_core.types import AnalyzeMessage

             from caesura_openai.adapters import get_message_text
             text = get_message_text({"content": user_input})
             if text:
                 collected.append(AnalyzeMessage(speaker_role="user", text=text))

        self._engine.observe(conv_id, collected)

        state = self._engine.store.get(conv_id)
        active = select_active(state, self._engine.config.inject, time.time() * 1000)
        blocks = render_block(active, self._engine.config.inject)

        if blocks:
            # Inject into the 'instructions' field for Responses API
            # This is simpler than the Chat API because Responses API
            # is instruction-driven per turn.
            instructions = kwargs.get("instructions", "")
            if not isinstance(instructions, str):
                instructions = str(instructions)

            skill = self._engine.config.inject.skill_prompt or ""
            combined_blocks = "\n\n".join(b["text"] for b in blocks)

            if instructions:
                kwargs["instructions"] = f"{instructions}\n\n{skill}\n\n{combined_blocks}"
            else:
                kwargs["instructions"] = f"{skill}\n\n{combined_blocks}"

            from caesura_core.types import InjectedBlock
            injected = [InjectedBlock(recommendation_id=b["recommendation_id"], text=b["text"], index=-1) for b in blocks]
            self._engine.emit_event(
                InjectedEvent(
                    conversation_id=conv_id,
                    turn=state.turn,
                    blocks=injected,
                    placement="end",
                )
            )

        state.turn += 1
        return self._original.create(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class CaesuraOpenAI:
    """Transparent wrapper for the synchronous OpenAI client."""

    def __init__(self, client: Any, options: CaesuraOpenAIOptions) -> None:
        self._client = client
        self._engine = create_caesura_engine(options)

    @property
    def chat(self) -> _CaesuraChat:
        return _CaesuraChat(self._client.chat, self._engine)

    @property
    def responses(self) -> _CaesuraResponses:
        return _CaesuraResponses(self._client.responses, self._engine)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


# ---------------------------------------------------------------------------
# Async wrappers
# ---------------------------------------------------------------------------


class _AsyncCaesuraCompletions:
    def __init__(self, original_completions: Any, engine: AsyncCaesuraEngine) -> None:
        self._original = original_completions
        self._engine = engine

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        conv_id = kwargs.pop("caesura_conversation_id", self._engine.config.conversation_id)
        if not conv_id:
            return await self._original.create(*args, **kwargs)

        messages = list(kwargs.get("messages", []))
        collected = collect_openai_messages(messages)
        await self._engine.observe(conv_id, collected)

        state = self._engine.store.get(conv_id)
        active = select_active(state, self._engine.config.inject, time.time() * 1000)
        blocks = render_block(active, self._engine.config.inject)

        messages, injected = inject_blocks_openai(messages, blocks, self._engine.config.inject, hash_message)
        messages, _ = apply_skill_prompt_openai(messages, self._engine.config.inject)
        kwargs["messages"] = messages

        if injected:
            self._engine.emit_event(
                InjectedEvent(
                    conversation_id=conv_id,
                    turn=state.turn,
                    blocks=injected,
                    placement=self._engine.config.inject.placement,
                )
            )

        state.turn += 1
        return await self._original.create(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class _AsyncCaesuraChat:
    def __init__(self, original_chat: Any, engine: AsyncCaesuraEngine) -> None:
        self._original = original_chat
        self._engine = engine

    @property
    def completions(self) -> _AsyncCaesuraCompletions:
        return _AsyncCaesuraCompletions(self._original.completions, self._engine)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class _AsyncCaesuraResponses:
    def __init__(self, original_responses: Any, engine: AsyncCaesuraEngine) -> None:
        self._original = original_responses
        self._engine = engine

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        conv_id = kwargs.pop("caesura_conversation_id", self._engine.config.conversation_id)
        if not conv_id:
            return await self._original.create(*args, **kwargs)

        collected = []
        user_input = kwargs.get("input")
        if isinstance(user_input, str):
            from caesura_core.types import AnalyzeMessage
            collected.append(AnalyzeMessage(speaker_role="user", text=user_input))
        elif isinstance(user_input, list):
             from caesura_core.types import AnalyzeMessage

             from caesura_openai.adapters import get_message_text
             text = get_message_text({"content": user_input})
             if text:
                 collected.append(AnalyzeMessage(speaker_role="user", text=text))

        await self._engine.observe(conv_id, collected)

        state = self._engine.store.get(conv_id)
        active = select_active(state, self._engine.config.inject, time.time() * 1000)
        blocks = render_block(active, self._engine.config.inject)

        if blocks:
            instructions = kwargs.get("instructions", "")
            if not isinstance(instructions, str):
                instructions = str(instructions)

            skill = self._engine.config.inject.skill_prompt or ""
            combined_blocks = "\n\n".join(b["text"] for b in blocks)

            if instructions:
                kwargs["instructions"] = f"{instructions}\n\n{skill}\n\n{combined_blocks}"
            else:
                kwargs["instructions"] = f"{skill}\n\n{combined_blocks}"

            from caesura_core.types import InjectedBlock
            injected = [InjectedBlock(recommendation_id=b["recommendation_id"], text=b["text"], index=-1) for b in blocks]
            self._engine.emit_event(
                InjectedEvent(
                    conversation_id=conv_id,
                    turn=state.turn,
                    blocks=injected,
                    placement="end",
                )
            )

        state.turn += 1
        return await self._original.create(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class AsyncCaesuraOpenAI:
    """Transparent wrapper for the asynchronous OpenAI client."""

    def __init__(self, client: Any, options: CaesuraOpenAIOptions) -> None:
        self._client = client
        self._engine = create_async_caesura_engine(options)

    @property
    def chat(self) -> _AsyncCaesuraChat:
        return _AsyncCaesuraChat(self._client.chat, self._engine)

    @property
    def responses(self) -> _AsyncCaesuraResponses:
        return _AsyncCaesuraResponses(self._client.responses, self._engine)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def create_caesura(client: Any, options: CaesuraOpenAIOptions) -> CaesuraOpenAI:
    """Wrap a synchronous OpenAI client."""
    return CaesuraOpenAI(client, options)


def create_async_caesura(client: Any, options: CaesuraOpenAIOptions) -> AsyncCaesuraOpenAI:
    """Wrap an asynchronous OpenAI client."""
    return AsyncCaesuraOpenAI(client, options)
