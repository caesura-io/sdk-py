"""Transparent wrappers around the OpenAI Python SDK clients.

These wrappers intercept calls to ``chat.completions.create`` and
``responses.create`` to inject CaesuraO recommendations, while delegating
all other attributes to the underlying OpenAI client via ``__getattr__``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from caesura_core.engine import AsyncCaesuraEngine, CaesuraEngine, create_async_caesura_engine, create_caesura_engine
from caesura_core.helpers import hash_message, render_block, select_active
from caesura_core.types import InjectedEvent

from caesura_openai.adapters import (
    apply_skill_prompt_openai,
    apply_skill_prompt_responses,
    collect_openai_messages,
    collect_openai_responses_messages,
    inject_blocks_openai,
    strip_injected_messages,
)

if TYPE_CHECKING:
    from types import TracebackType

    from caesura_openai.types import CaesuraOpenAIOptions


def _resolve_session(per_call: str | None, configured: str | None) -> str:
    if per_call is not None:
        return per_call
    return configured if configured is not None else "default"


def _clean_input(engine: CaesuraEngine | AsyncCaesuraEngine, conv_id: str, value: Any) -> Any:
    if isinstance(value, list):
        state = engine.store.get(conv_id)
        known = state.injected_messages | {
            (engine.config.inject.as_role, rec.injected_text)
            for rec in state.recommendations
            if rec.injected_text is not None
            and not any(text == rec.injected_text for _, text in state.injected_messages)
        }
        return strip_injected_messages(value, known)
    return value


def _inject_guidance(
    engine: CaesuraEngine | AsyncCaesuraEngine,
    conv_id: str,
    kwargs: dict[str, Any],
    value: Any,
    *,
    responses: bool,
) -> None:
    # Evaluate TTL after foreground analysis has completed.
    state = engine.store.get(conv_id)
    config = engine.config.inject
    active = select_active(state, config, time.time() * 1000)
    blocks = render_block(active, config)
    if responses:
        instructions = apply_skill_prompt_responses(kwargs.get("instructions"), config)
        if instructions is not None or "instructions" in kwargs:
            kwargs["instructions"] = instructions
        if isinstance(value, str):
            messages = [{"role": "user", "content": value}]
        elif isinstance(value, list):
            messages = value
        else:
            messages = []
    else:
        messages, _ = apply_skill_prompt_openai(value, config)
    messages, injected = inject_blocks_openai(messages, blocks, config, hash_message, engine.config.speaker_names)
    if responses:
        # Preserve the original input shape when no guidance needs inserting.
        if injected or "input" in kwargs:
            kwargs["input"] = messages if injected else value
    else:
        kwargs["messages"] = messages
    for block in injected:
        state.injected_messages.add((config.as_role, block.text))
        for recommendation in active:
            if recommendation.id == block.recommendation_id:
                recommendation.injected_text = block.text
                break
    if injected:
        engine.emit_event(
            InjectedEvent(
                conversation_id=conv_id,
                turn=state.turn,
                blocks=injected,
                placement=config.placement,
            )
        )


class _CaesuraCompletions:
    def __init__(self, original_completions: Any, engine: CaesuraEngine) -> None:
        self._original = original_completions
        self._engine = engine

    def create(self, *args: Any, **kwargs: Any) -> Any:
        conv_id = _resolve_session(kwargs.pop("caesura_conversation_id", None), self._engine.config.conversation_id)

        value = _clean_input(self._engine, conv_id, list(kwargs.get("messages", [])))
        collected = collect_openai_messages(value, self._engine.config.speaker_names)
        self._engine.observe(conv_id, collected)
        _inject_guidance(self._engine, conv_id, kwargs, value, responses=False)

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
        conv_id = _resolve_session(kwargs.pop("caesura_conversation_id", None), self._engine.config.conversation_id)

        value = _clean_input(self._engine, conv_id, kwargs.get("input"))
        collected = collect_openai_responses_messages(value, self._engine.config.speaker_names)
        self._engine.observe(conv_id, collected)
        _inject_guidance(self._engine, conv_id, kwargs, value, responses=True)

        return self._original.create(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class CaesuraOpenAI:
    """Transparent wrapper for the synchronous OpenAI client."""

    def __init__(self, client: Any, options: CaesuraOpenAIOptions) -> None:
        self._client = client
        self._engine = create_caesura_engine(options)

    def __enter__(self) -> CaesuraOpenAI:
        self._client.__enter__()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self._client.__exit__(exc_type, exc_value, traceback)

    def create_conversation(
        self, *, name: str | None = None, calendar_id: str | None = None, event_id: str | None = None
    ) -> str:
        """Create a CaesuraO conversation; reuse its ID as caesura_conversation_id on model calls."""
        return self._engine.create_conversation(name=name, calendar_id=calendar_id, event_id=event_id)

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
        conv_id = _resolve_session(kwargs.pop("caesura_conversation_id", None), self._engine.config.conversation_id)

        value = _clean_input(self._engine, conv_id, list(kwargs.get("messages", [])))
        collected = collect_openai_messages(value, self._engine.config.speaker_names)
        await self._engine.observe(conv_id, collected)
        _inject_guidance(self._engine, conv_id, kwargs, value, responses=False)

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
        conv_id = _resolve_session(kwargs.pop("caesura_conversation_id", None), self._engine.config.conversation_id)

        value = _clean_input(self._engine, conv_id, kwargs.get("input"))
        collected = collect_openai_responses_messages(value, self._engine.config.speaker_names)
        await self._engine.observe(conv_id, collected)
        _inject_guidance(self._engine, conv_id, kwargs, value, responses=True)

        return await self._original.create(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class AsyncCaesuraOpenAI:
    """Transparent wrapper for the asynchronous OpenAI client."""

    def __init__(self, client: Any, options: CaesuraOpenAIOptions) -> None:
        self._client = client
        self._engine = create_async_caesura_engine(options)

    async def __aenter__(self) -> AsyncCaesuraOpenAI:
        await self._client.__aenter__()
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        await self._client.__aexit__(exc_type, exc_value, traceback)

    async def create_conversation(
        self, *, name: str | None = None, calendar_id: str | None = None, event_id: str | None = None
    ) -> str:
        """Create a CaesuraO conversation; reuse its ID as caesura_conversation_id on model calls."""
        return await self._engine.create_conversation(name=name, calendar_id=calendar_id, event_id=event_id)

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
